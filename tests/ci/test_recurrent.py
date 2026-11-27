"""
Validates `recurrent.py`'s `build_lstm_cell_topologies` end-to-end: traces a real `torch.nn.LSTM`, builds
the per-timestep cell topologies from the traced op's own weights, EXECUTES those topologies (via a small
numpy interpreter understanding exactly the node ops the topologies use -- MUL_MAT/ADD/VIEW/SIGMOID/
TANH/MUL -- so this exercises the real VIEW byte-offset gate-slicing logic, not just the underlying gate
math in the abstract), threading h/c state across timesteps the same way the eventual C++
`LoomLuaBridge::l_run_recurrent` binding will, and compares the result against real PyTorch `nn.LSTM`
output. This is the correctness gate for LSTM export -- confirming the generated topology JSON is right is
cheaper and faster to iterate on than building the full C++ stepper first.
"""
import unittest

import numpy as np
import torch

import sys
from pathlib import Path

from loom_exporter.paths import CONVERTERS, driver_dir
import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
import coremltools as ct
from loom_exporter.recurrent import build_lstm_cell_topologies


def _run_topology(topo: dict, weights: dict, inputs: dict) -> np.ndarray:
    """Minimal interpreter for exactly the node ops a cell topology from recurrent.py uses, returning
    `{declared output name: value}`. Not a general-purpose GraphBuilder replacement -- just enough to
    execute what build_lstm_cell_topologies actually emits, as a numpy-level stand-in for the real ggml
    engine."""
    values = dict(weights)
    values.update(inputs)
    for node in topo["nodes"]:
        op, ins, outs = node["op"], node["inputs"], node["outputs"]
        if op == "MUL_MAT":
            # Loom convention: MUL_MAT(weight, x) == weight @ x (weight-first, matching op_mul_mat). A
            # SEQUENCE operand is `[T, I]`, one row per timestep as ggml lays `[I, T]` out, so the product
            # is a row per timestep too (the input projection, ADR-137).
            w, x = values[ins[0]], values[ins[1]]
            values[outs[0]] = x @ w.T if x.ndim == 2 else w @ x
        elif op == "ADD":
            values[outs[0]] = values[ins[0]] + values[ins[1]]
        elif op == "MUL":
            values[outs[0]] = values[ins[0]] * values[ins[1]]
        elif op == "VIEW":
            shape = node["attrs"]["shape"]
            offset_elems = node["attrs"]["offset"] // 4  # f32
            flat = values[ins[0]]
            values[outs[0]] = flat[offset_elems:offset_elems + shape[0]]
        elif op == "SIGMOID":
            values[outs[0]] = 1.0 / (1.0 + np.exp(-values[ins[0]]))
        elif op == "TANH":
            values[outs[0]] = np.tanh(values[ins[0]])
        else:
            raise NotImplementedError(f"test interpreter doesn't know op '{op}'")
    # Every declared output, by name: a cell topology now declares both halves of the step, and the
    # point of the change is that ONE evaluation of this node list yields both.
    return {name: values[name] for name in topo["outputs"]}


def _trace_lstm_op(hidden_dim, input_dim, seq_len, bidirectional):
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                          batch_first=False, bidirectional=bidirectional).eval()
    x = torch.randn(seq_len, 1, input_dim)

    class M(torch.nn.Module):
        def __init__(self, lstm):
            super().__init__()
            self.lstm = lstm

        def forward(self, x):
            out, _ = self.lstm(x)
            return out

    m = M(lstm).eval()
    with torch.no_grad():
        ref_out = m(x).squeeze(1).numpy()  # (T, DIRECTIONS*H)

    traced = torch.jit.trace(m, (x,))
    prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal")
    func = prog.functions["main"]
    op = next(o for o in func.operations if o.op_type == "lstm")
    return op, x.squeeze(1).numpy(), ref_out


class TestLstmRecurrent(unittest.TestCase):
    def _check(self, hidden_dim, input_dim, seq_len, bidirectional, project=False):
        op, x_np, ref_out = _trace_lstm_op(hidden_dim, input_dim, seq_len, bidirectional)
        result = build_lstm_cell_topologies(op, weight_namespace="test.", project_input=project)
        self.assertEqual(result["bidirectional"], bidirectional)
        self.assertEqual(result["hidden_dim"], hidden_dim)
        # A projected LSTM (ADR-137): `W_ih x + b` for the whole sequence first, one output per direction,
        # and each step's `layer_input` is that direction's row.
        projected = None
        if project:
            self.assertEqual(result["input"]["outputs"], ["gates_0", "gates_1"][:2 if bidirectional else 1])
            projected = _run_topology(result["input"], result["weights"], {"sequence": x_np})
        else:
            self.assertIsNone(result["input"])

        def run_direction(topo, reverse: bool, rows=None) -> np.ndarray:
            h = np.zeros(hidden_dim, dtype=np.float32)
            c = np.zeros(hidden_dim, dtype=np.float32)
            out = np.zeros((seq_len, hidden_dim), dtype=np.float32)
            order = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            for t in order:
                inputs = {"layer_input": (x_np if rows is None else rows)[t], "h_prev": h, "c_prev": c}
                # One topology, both outputs -- the interpreter reads each declared name out of the
                # same evaluated node list, which is exactly what the engine now does per call.
                values = _run_topology(topo, result["weights"], inputs)
                h_new, c_new = values["h_new"], values["c_new"]
                h, c = h_new, c_new
                out[t] = h
            return out

        fwd_out = run_direction(result["forward"], reverse=False, rows=projected and projected["gates_0"])
        if bidirectional:
            bwd_out = run_direction(result["backward"], reverse=True, rows=projected and projected["gates_1"])
            got = np.concatenate([fwd_out, bwd_out], axis=-1)
        else:
            got = fwd_out

        diff = np.abs(got - ref_out)
        self.assertLess(diff.max(), 1e-4, f"max diff {diff.max()} too large")

    def test_lstm_forward_unidirectional(self):
        self._check(hidden_dim=5, input_dim=8, seq_len=6, bidirectional=False)

    def test_lstm_bidirectional(self):
        self._check(hidden_dim=5, input_dim=8, seq_len=6, bidirectional=True)

    def test_lstm_different_dims(self):
        self._check(hidden_dim=13, input_dim=7, seq_len=4, bidirectional=False)

    def test_projected_lstm_unidirectional(self):
        self._check(hidden_dim=5, input_dim=8, seq_len=6, bidirectional=False, project=True)

    def test_projected_lstm_bidirectional(self):
        self._check(hidden_dim=13, input_dim=7, seq_len=9, bidirectional=True, project=True)

    def test_a_projected_stack_runs_a_layer_at_a_time(self):
        """ADR-144: every layer of a stack projected, driven the way `BiRecurrentCall` drives it -- layer k's
        `_input` reads layer k-1's `[h_fwd | h_bwd]` rows, then both directions sweep their own rows."""
        from loom_exporter.multi_phase_export import RecurrentPhase

        torch.manual_seed(0)
        hidden, width, seq_len = 6, 5, 7
        lstm = torch.nn.LSTM(width, hidden, num_layers=2, bidirectional=True).eval()
        x = torch.randn(seq_len, 1, width)
        with torch.no_grad():
            ref = lstm(x)[0].squeeze(1).numpy()
        topologies, weights = RecurrentPhase(name="s", module=lstm, number_layers=True,
                                             project_input=True).topologies()
        self.assertEqual(sorted(topologies), ["s_l0_bwd", "s_l0_fwd", "s_l0_input",
                                              "s_l1_bwd", "s_l1_fwd", "s_l1_input"])
        self.assertEqual(topologies["s_l1_input"]["inputs"][0]["shape"], [str(2 * hidden), "n_tokens"])

        def sweep(topo, rows, reverse):
            h = np.zeros(hidden, dtype=np.float32)
            c = np.zeros(hidden, dtype=np.float32)
            out = np.zeros((seq_len, hidden), dtype=np.float32)
            for t in (range(seq_len - 1, -1, -1) if reverse else range(seq_len)):
                v = _run_topology(topo, weights, {"layer_input": rows[t], "h_prev": h, "c_prev": c})
                h, c = v["h_new"], v["c_new"]
                out[t] = h
            return out

        def run(swap_layer=None):
            seq = x.squeeze(1).numpy()
            for k in range(2):
                gates = _run_topology(topologies[f"s_l{k}_input"], weights, {"sequence": seq})
                fwd_rows, bwd_rows = gates["gates_0"], gates["gates_1"]
                if k == swap_layer:
                    fwd_rows, bwd_rows = bwd_rows, fwd_rows
                seq = np.concatenate([sweep(topologies[f"s_l{k}_fwd"], fwd_rows, False),
                                      sweep(topologies[f"s_l{k}_bwd"], bwd_rows, True)], axis=-1)
            return seq

        self.assertLess(np.abs(run() - ref).max(), 1e-4)
        # The check can fail: layer 1's directions reading each other's projection is far off.
        self.assertGreater(np.abs(run(swap_layer=1) - ref).max(), 1e-2)

    def test_a_projected_cell_holds_no_input_weights(self):
        """The point of the split: the cell's per-step work is `W_hh h` alone."""
        op, _, _ = _trace_lstm_op(5, 8, 6, True)
        result = build_lstm_cell_topologies(op, weight_namespace="test.", project_input=True)
        for topo in (result["forward"], result["backward"]):
            used = {i for node in topo["nodes"] for i in node["inputs"]}
            self.assertFalse(any(u.endswith("weight_ih") or u.endswith("bias") for u in used), used)
            self.assertEqual(topo["inputs"][0]["shape"], [str(4 * 5)])


if __name__ == "__main__":
    unittest.main()
