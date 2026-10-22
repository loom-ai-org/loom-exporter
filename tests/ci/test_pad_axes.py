"""ADR-081's weight-axis pad (`loom_exporter/pad_axes.py`): exact, and only where a kernel wants it.

The pass rewrites weights and topology together, so the claim worth testing is numerical: the padded
graph computes what the unpadded one did. The engine is not a CI dependency, so these tests run both
graphs through `_Interpreter`, a numpy reading of the dozen ops the toys lower to that follows ggml's
layout rules where they matter here -- above all VIEW, which keeps its parent's row stride, so a VIEW
that shrinks ne[0] reads every row from the right place only if the stride is honoured. The interpreter
is checked against torch first: a reference that agrees with whatever it is compared to proves nothing.

Two sabotages show the comparison can fail: a padded graph missing one of its VIEWs, and a weight
padded on the wrong side.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

T = 40


def _toy_decoder():
    """Vocetta2's decoder in miniature, every width chosen to be unaligned: a conv (OC 30) feeding a
    LayerNorm, a LayerNorm-fed pointwise pair through GELU (30 -> 90 -> 30), a second pair through
    SIGMOID (not zero-preserving), and a head (30 -> 18) whose output is read by one more op."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(5, 30, 7, padding=3)
            self.norm = nn.LayerNorm(30)
            self.pw0, self.pw1 = nn.Linear(30, 90), nn.Linear(90, 30)
            self.sig, self.gate = nn.Linear(30, 42), nn.Linear(42, 30)
            self.norm2, self.head = nn.LayerNorm(30), nn.Linear(30, 18)

        def forward(self, x):
            h = self.conv(x).transpose(1, 2)
            h = h + self.pw1(F.gelu(self.pw0(self.norm(h))))
            h = h + self.gate(torch.sigmoid(self.sig(h)))
            return torch.tanh(self.head(self.norm2(h)))

    return Toy, (1, 5)


def _toy_convs():
    """A conv -> SILU -> conv pair whose second kernel (k = 5) is what a quantized export wants
    aligned: IC x 5 is a multiple of 32 only when IC is."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.c0 = nn.Conv1d(5, 30, 7, padding=3)
            self.c1 = nn.Conv1d(30, 16, 5, padding=2)

        def forward(self, x):
            return F.silu(self.c1(F.silu(self.c0(x))))

    return Toy, (1, 5)


def _aligned():
    import torch.nn as nn
    import torch.nn.functional as F

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(32, 64), nn.Linear(64, 32)

        def forward(self, x):
            return self.b(F.gelu(self.a(x)))

    return Toy, (1, T, 32)


def _trace(factory, seed=0):
    import coremltools as ct
    import torch

    from loom_exporter.exporter import LoomGGUFExporter
    from loom_exporter.passes import apply_loom_mil_passes

    cls, lead = factory()
    torch.manual_seed(seed)
    model = cls().eval()
    if len(lead) == 2:
        example = torch.randn(*lead, T)
        shape = (*lead, ct.RangeDim(8, 400))
    else:
        example = torch.randn(*lead)
        shape = (1, ct.RangeDim(8, 400), lead[2])
    traced = torch.jit.trace(model, (example,))
    program = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32,
                         minimum_deployment_target=ct.target.iOS17)
    apply_loom_mil_passes(program)
    exporter = LoomGGUFExporter(program, flat_namespace=True)
    topo = exporter.generate_graph_topology(program.functions["main"], "main_topology")
    exporter.topologies = {"main_topology": topo}
    with torch.no_grad():
        expected = model(example).numpy()
    return exporter, example.numpy(), expected


def _pad(exporter, quantize_names=(), block_size=1):
    """The pass on copies, as `pack_weights` calls it: returns (topology, weights, report)."""
    from loom_exporter.pad_axes import pad_weight_axes

    topologies = copy.deepcopy(exporter.topologies)
    weights = dict(exporter.weights)
    shapes = {name: exporter._var_shape_lookup(v) for name, v in exporter.topology_vars.items()}
    report = pad_weight_axes(topologies, weights, set(weights), shapes,
                             lambda name: name in quantize_names, block_size)
    return topologies["main_topology"], weights, report


class _Interpreter:
    """Every tensor is a 4-D numpy array in torch order, (ne3, ne2, ne1, ne0)."""

    def __init__(self, weights, symbols):
        self.weights, self.symbols = weights, symbols

    def _int(self, value):
        return int(eval(str(value), {}, dict(self.symbols)))

    @staticmethod
    def _4d(array):
        return array.reshape((1,) * (4 - array.ndim) + array.shape)

    def run(self, topo, inputs):
        env = {name: self._4d(np.asarray(v, dtype=np.float32)) for name, v in inputs.items()}
        for name, array in self.weights.items():
            if np.issubdtype(np.asarray(array).dtype, np.floating):
                env[name] = self._4d(np.asarray(array, dtype=np.float32))
        for node in topo["nodes"]:
            args = [env[name] for name in node["inputs"]]
            env[node["outputs"][0]] = getattr(self, "op_" + node["op"].lower())(node, *args)
        return env[topo["output"]]

    def op_conv_1d(self, node, kernel, x):
        import torch
        import torch.nn.functional as F

        a = node.get("attrs", {})
        w = np.asarray(self.weights[node["inputs"][0]], dtype=np.float32)
        out = F.conv1d(torch.from_numpy(np.ascontiguousarray(x[0])), torch.from_numpy(w),
                       stride=a.get("s0", 1), padding=a.get("p0", 0), dilation=a.get("d0", 1))
        return out.numpy()[None]

    def op_reshape(self, node, x):
        shape = [self._int(d) for d in node["attrs"]["shape"]]
        return self._4d(np.ascontiguousarray(x).reshape(tuple(reversed(shape))))

    def op_permute(self, node, x):
        axes = node["attrs"]["axes"]
        ne = x.transpose(3, 2, 1, 0)                      # ne order
        permuted = ne.transpose(np.argsort(axes))         # source axis i lands at axes[i]
        return permuted.transpose(3, 2, 1, 0)

    def op_cont(self, node, x):
        return np.ascontiguousarray(x)

    def op_view(self, node, x):
        parent = np.ascontiguousarray(x)                  # op_view conts a non-contiguous parent
        shape = [self._int(d) for d in node["attrs"]["shape"]]
        shape = shape + [1] * (4 - len(shape))
        offset = self._int(node["attrs"].get("offset", 0)) // 4
        nb = list(reversed(parent.strides))               # the PARENT's strides, as ggml_view_Nd
        return np.lib.stride_tricks.as_strided(parent.reshape(-1)[offset:], shape=tuple(reversed(shape)),
                                               strides=tuple(reversed(nb)))

    def op_pad_1d(self, node, x):
        a = node["attrs"]
        return np.pad(x, [(0, 0)] * 3 + [(a["lp0"], a["rp0"])])

    def op_mul_mat(self, node, w, x):
        return np.einsum("abtk,abnk->abtn", x, np.broadcast_to(w, x.shape[:2] + w.shape[2:]))

    def op_add(self, node, a, b):
        return a + b

    def op_mul(self, node, a, b):
        return a * b

    def op_layer_norm(self, node, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + node["attrs"]["eps"])

    def op_gelu(self, node, x):
        import torch
        return torch.nn.functional.gelu(torch.from_numpy(np.ascontiguousarray(x))).numpy()

    def op_silu(self, node, x):
        return x / (1.0 + np.exp(-x))

    def op_sigmoid(self, node, x):
        return 1.0 / (1.0 + np.exp(-x))

    def op_tanh(self, node, x):
        return np.tanh(x)


def _run(topo, weights, x):
    return _Interpreter(weights, {"n_tokens": T}).run(topo, {"x": x}).reshape(-1)


def _nodes(topo, op):
    return [n for n in topo["nodes"] if n["op"] == op]


class TestTheInterpreterIsAReference(unittest.TestCase):
    def test_it_agrees_with_torch_on_the_unpadded_graphs(self):
        for factory in (_toy_decoder, _toy_convs, _aligned):
            with self.subTest(factory.__name__):
                exporter, x, expected = _trace(factory)
                got = _run(exporter.topologies["main_topology"], exporter.weights, x)
                np.testing.assert_allclose(got, expected.reshape(-1), atol=2e-5, rtol=1e-4)


class TestF32PadsKAndNToFour(unittest.TestCase):
    def setUp(self):
        self.exporter, self.x, _ = _trace(_toy_decoder)
        self.topo, self.weights, self.report = _pad(self.exporter)

    def test_the_padded_graph_computes_the_same_values(self):
        before = _run(self.exporter.topologies["main_topology"], self.exporter.weights, self.x)
        after = _run(self.topo, self.weights, self.x)
        self.assertEqual(after.shape, before.shape)
        np.testing.assert_allclose(after, before, atol=2e-6, rtol=1e-5)

    def test_every_weight_lands_on_a_multiple_of_four(self):
        shapes = {name: self.weights[name].shape for name in
                  ("conv_weight", "pw0_weight", "pw1_weight", "sig_weight", "gate_weight", "head_weight")}
        self.assertEqual(shapes, {
            "conv_weight": (32, 5, 7),   # OC 30 -> 32; IC 5 x 7 has no producer to carry a pad
            "pw0_weight": (92, 32),      # K by PAD_1D (LayerNorm-fed), N carried into pw1
            "pw1_weight": (32, 92),
            "sig_weight": (44, 32),
            "gate_weight": (32, 44),     # SIGMOID is not zero-preserving: PAD_1D, not a carry
            "head_weight": (20, 32),
        })
        self.assertEqual(self.weights["pw0_bias"].shape, (92,))
        self.assertTrue(np.all(self.weights["pw0_bias"][90:] == 0))
        self.assertTrue(np.all(self.weights["pw1_weight"][:, 90:] == 0))

    def test_only_the_carried_pair_goes_without_a_view_or_a_pad(self):
        self.assertEqual(self.report.n_carried, 1)
        views = {n["outputs"][0] for n in _nodes(self.topo, "VIEW")}
        self.assertEqual(len(views), 5, "conv, pw1, sig, gate and head each slice back to their width")
        # PAD_1D before pw0 and head (LayerNorm-fed), sig (the residual sum) and gate (SIGMOID's output).
        pads = sorted((n["inputs"][0], n["attrs"]["rp0"]) for n in _nodes(self.topo, "PAD_1D"))
        self.assertEqual(len(pads), 4, pads)
        self.assertEqual(self.report.padded["pw0_bias"], (90, 0))
        self.assertEqual(self.report.padded["pw0_weight"], (90, 30))


class TestQuantizedWeightsPadKToTheBlock(unittest.TestCase):
    def test_k_goes_to_32_and_n_is_left_alone(self):
        exporter, x, _ = _trace(_toy_decoder)
        linear = {"pw0_weight", "pw1_weight", "sig_weight", "gate_weight", "head_weight"}
        topo, weights, _ = _pad(exporter, quantize_names=linear | {"conv_weight"}, block_size=32)
        self.assertEqual(weights["pw0_weight"].shape, (96, 32))   # N carried to what pw1's K needs
        self.assertEqual(weights["pw1_weight"].shape, (30, 96))   # N untouched
        self.assertEqual(weights["head_weight"].shape, (18, 32))
        # 42 -> 64 would be +52%, over the cap, so gate stays F32 and takes the F32 rule instead.
        self.assertEqual(weights["gate_weight"].shape, (32, 44))
        np.testing.assert_allclose(_run(topo, weights, x),
                                   _run(exporter.topologies["main_topology"], exporter.weights, x),
                                   atol=2e-6, rtol=1e-5)
        # The conv's IC 5 x 7 cannot reach 32 within the cap, so it is written F32 and gets that rule.
        self.assertEqual(weights["conv_weight"].shape, (32, 5, 7))

    def test_a_k5_conv_kernel_gets_its_ic_from_the_producer(self):
        exporter, x, _ = _trace(_toy_convs)
        topo, weights, report = _pad(exporter, quantize_names={"c0_weight", "c1_weight"}, block_size=32)
        self.assertEqual(weights["c0_weight"].shape, (32, 5, 7))
        self.assertEqual(weights["c1_weight"].shape, (16, 32, 5))  # 32 x 5 = 160, a multiple of 32
        self.assertEqual(report.n_carried, 1)
        self.assertEqual(_nodes(topo, "VIEW"), [])
        np.testing.assert_allclose(_run(topo, weights, x),
                                   _run(exporter.topologies["main_topology"], exporter.weights, x),
                                   atol=2e-6, rtol=1e-5)


class TestWhatItLeavesAlone(unittest.TestCase):
    def test_an_aligned_model_writes_the_same_bytes(self):
        files = []
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("auto", "off"):
                exporter, _, _ = _trace(_aligned)
                exporter.output_path = str(Path(tmp) / f"{mode}.gguf")
                exporter.kwargs["architecture"] = "test"
                exporter.pad_axes = mode
                exporter.write_gguf("-- driver")
                files.append(Path(exporter.output_path).read_bytes())
        self.assertEqual(files[0], files[1])

    def test_a_topology_with_no_recorded_shapes_is_not_touched(self):
        from loom_exporter.pad_axes import pad_weight_axes

        topo = {"output": "y", "nodes": [{"op": "MUL_MAT", "inputs": ["w", "x"], "outputs": ["y"]}]}
        weights = {"w": np.ones((6, 30), np.float32)}
        report = pad_weight_axes({"t": topo}, weights, {"w"}, {}, lambda n: False, 1)
        self.assertEqual(report.padded, {})
        self.assertEqual(weights["w"].shape, (6, 30))

    def test_a_shared_weight_and_an_oversized_pad_are_skipped(self):
        from loom_exporter.pad_axes import pad_weight_axes

        topo = {"output": "z", "nodes": [
            {"op": "MUL_MAT", "inputs": ["shared", "x"], "outputs": ["a"]},
            {"op": "MUL_MAT", "inputs": ["shared", "a"], "outputs": ["b"]},
            {"op": "MUL_MAT", "inputs": ["tiny", "b"], "outputs": ["c"]},  # K 6 -> 8 is +33%
            {"op": "TANH", "inputs": ["c"], "outputs": ["z"]},
        ]}
        weights = {"shared": np.ones((6, 6), np.float32), "tiny": np.ones((4, 6), np.float32)}
        report = pad_weight_axes({"t": topo}, weights, set(weights), {"t": lambda name: None},
                                 lambda n: False, 1)
        self.assertEqual(report.padded, {})


class TestSabotage(unittest.TestCase):
    """The comparison above must be able to fail."""

    def setUp(self):
        self.exporter, self.x, _ = _trace(_toy_decoder)
        self.before = _run(self.exporter.topologies["main_topology"], self.exporter.weights, self.x)
        self.topo, self.weights, _ = _pad(self.exporter)

    def test_a_missing_view_breaks_it(self):
        view = next(n for n in self.topo["nodes"] if n["op"] == "VIEW" and n["outputs"][0] == "_21")
        self.topo["nodes"].remove(view)
        for node in self.topo["nodes"]:
            node["outputs"] = ["_21" if o == view["inputs"][0] else o for o in node["outputs"]]
        with self.assertRaises(Exception):
            after = _run(self.topo, self.weights, self.x)
            np.testing.assert_allclose(after, self.before, atol=2e-6, rtol=1e-5)

    def test_padding_on_the_wrong_side_breaks_it(self):
        w = self.exporter.weights["pw0_weight"]
        self.weights["pw0_weight"] = np.pad(w, [(0, 2), (2, 0)])
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(_run(self.topo, self.weights, self.x), self.before,
                                       atol=2e-6, rtol=1e-5)


class TestTheOptionReachesTheExporter(unittest.TestCase):
    def test_config_carries_it_only_when_set(self):
        from loom_exporter.decomposition import Flattened
        from loom_exporter.export_config import LoomExportConfig

        config = LoomExportConfig(architecture="test", output_path="<unused>.gguf",
                                  decomposition=Flattened())
        self.assertNotIn("pad_axes", config.resolved_backend_kwargs())
        config.pad_axes = "off"
        self.assertEqual(config.resolved_backend_kwargs()["pad_axes"], "off")

    def test_an_unknown_mode_is_refused(self):
        from loom_exporter.exporter import LoomGGUFExporter

        with self.assertRaises(ValueError):
            LoomGGUFExporter(None, pad_axes="always")
        self.assertEqual(LoomGGUFExporter(None).pad_axes, "auto")

    def test_the_file_lists_what_was_padded(self):
        from gguf import GGUFReader

        from loom_exporter.exporter import PAD_IN_KEY, PAD_OUT_KEY, PAD_WEIGHTS_KEY

        exporter, _, _ = _trace(_toy_decoder)
        with tempfile.TemporaryDirectory() as tmp:
            exporter.output_path = str(Path(tmp) / "p.gguf")
            exporter.kwargs["architecture"] = "test"
            exporter.write_gguf("-- driver")
            reader = GGUFReader(exporter.output_path)
            field = reader.fields[PAD_WEIGHTS_KEY]
            names = [bytes(field.parts[i]).decode() for i in field.data]
            out = [int(reader.fields[PAD_OUT_KEY].parts[i][0]) for i in reader.fields[PAD_OUT_KEY].data]
            inn = [int(reader.fields[PAD_IN_KEY].parts[i][0]) for i in reader.fields[PAD_IN_KEY].data]
            shapes = {t.name: [int(d) for d in t.shape] for t in reader.tensors}
            topo_field = reader.fields["model.graph_topology.main_topology"]
            topo = json.loads(bytes(topo_field.parts[topo_field.data[0]]).decode())
        table = dict(zip(names, zip(out, inn)))
        self.assertEqual(table["pw1_weight"], (30, 90))
        self.assertEqual(table["conv_bias"], (30, 0))
        self.assertEqual(shapes["pw1_weight"], [92, 32])   # ggml order
        self.assertTrue(any(n["op"] == "PAD_1D" for n in topo["nodes"]))


if __name__ == "__main__":
    unittest.main()
