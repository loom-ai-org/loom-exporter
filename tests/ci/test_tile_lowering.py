"""`tile` with LIVE reps: `expand_as` over a dynamic axis.

coremltools lowers `x.expand_as(t)` to `tile(x, reps=select(shape(t) == -1, x.shape, shape(t)) / x.shape)`,
so the reps are not a constant whenever `t` has a dynamic axis. `_op_tile` used to read them as ones and
emit an identity REPEAT. A broadcasting consumer hid that; a CONCAT did not -- Qwen3-TTS's ECAPA speaker
encoder concatenates a one-frame mean onto a T-frame hidden state, and `ggml_concat` aborted the process
on the first `waveform=` call of every published build.
"""
import unittest
from pathlib import Path

import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class PooledConcat(torch.nn.Module):
    """ECAPA's attentive-statistics input, reduced: `cat([h, mean(h).expand_as(h)], dim=1)`."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, 1)

    def forward(self, x):
        h = self.conv(x)
        return torch.cat([h, h.mean(dim=2).unsqueeze(2).expand_as(h)], dim=1)


def _repeat_nodes(module, shape):
    x = torch.randn(1, 4, 7)
    prog = ct.convert(torch.jit.trace(module.eval(), (x,)),
                      inputs=[ct.TensorType(name="x", shape=shape)], convert_to="milinternal")
    out = Path("test_tile_lowering.gguf")
    try:
        exporter = LoomGGUFExporter(prog, output_path=str(out), architecture="tile_test")
        exporter.export()
    finally:
        out.unlink(missing_ok=True)
    topo = next(iter(exporter.topologies.values()))
    return [n for n in topo["nodes"] if n["op"] == "REPEAT"]


class TestExpandAsOverADynamicAxis(unittest.TestCase):
    def test_the_repeat_reaches_the_dynamic_axis(self):
        (repeat,) = _repeat_nodes(PooledConcat(), (1, 4, ct.RangeDim(2, 100)))
        # ne-order: time first. The identity this replaced was ['1', '4', '1'].
        self.assertEqual([str(d) for d in repeat["attrs"]["shape"]], ["n_tokens", "4", "1"])

    def test_a_static_trace_still_folds_to_a_literal_target(self):
        repeats = _repeat_nodes(PooledConcat(), (1, 4, 7))
        for repeat in repeats:
            self.assertEqual([str(d) for d in repeat["attrs"]["shape"]], ["7", "4", "1"])


if __name__ == "__main__":
    unittest.main()
