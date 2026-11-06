"""P4.53: `x.argmax(-1)` lowers to the engine's ARGMAX, so a greedy choice can stay on the device; any
other axis is refused rather than lowered to a reduction over the wrong one."""
import unittest
from pathlib import Path

import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class _Greedy(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(6, 11)
        self.dim = dim

    def forward(self, x):
        return self.proj(torch.tanh(x)).argmax(self.dim)


def _export(dim):
    m = _Greedy(dim).eval()
    x = torch.randn(4, 6)
    traced = torch.jit.trace(m, (x,))
    prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal",
                      compute_precision=ct.precision.FLOAT32)
    out = Path("test_argmax_output.gguf")
    exporter = LoomGGUFExporter(prog, output_path=str(out), architecture="argmax_test")
    try:
        exporter.export()
    finally:
        out.unlink(missing_ok=True)
    return next(iter(exporter.topologies.values()))


class TestArgmaxLowering(unittest.TestCase):
    def test_last_axis_is_one_argmax_node(self):
        topo = _export(-1)
        argmax = [n for n in topo["nodes"] if n["op"] == "ARGMAX"]
        self.assertEqual(len(argmax), 1)
        # It reduces the projection's output, not some reshaped or permuted copy of it.
        producer = {o: n for n in topo["nodes"] for o in n["outputs"]}
        self.assertIn(producer[argmax[0]["inputs"][0]]["op"], {"MUL_MAT", "ADD"})

    def test_another_axis_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "only the last axis"):
            _export(0)


if __name__ == "__main__":
    unittest.main()
