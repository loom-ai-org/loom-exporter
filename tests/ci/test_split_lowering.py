"""`split` with explicit, UNEQUAL sizes: a fused QKV projection under GQA.

`x.split((6, 2, 2))` reaches MIL with `split_sizes`, and `_op_split` used to ignore them and divide the
axis evenly -- three views of 3 rows here, of 384 for Audio8-TTS's `(896, 128, 128)`. The export
succeeded and the first call died in the q reshape (`input element count is not evenly divisible`).
An even split must render exactly as it always has, which is what keeps every gate snapshot still.
"""
import unittest
from pathlib import Path

import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class FusedQkv(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.proj = torch.nn.Linear(4, sum(sizes))

    def forward(self, x):
        q, k, v = self.proj(x).split(self.sizes, dim=-1)
        return q.sum(dim=-1) + k.sum(dim=-1) * 2.0 + v.sum(dim=-1) * 3.0


def _views(sizes):
    x = torch.randn(1, 5, 4)
    prog = ct.convert(torch.jit.trace(FusedQkv(sizes).eval(), (x,)),
                      inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 64), 4))],
                      convert_to="milinternal")
    out = Path("test_split_lowering.gguf")
    try:
        exporter = LoomGGUFExporter(prog, output_path=str(out), architecture="split_test")
        exporter.export()
    finally:
        out.unlink(missing_ok=True)
    topo = next(iter(exporter.topologies.values()))
    # The split's own views, not ADR-081's pad view (whose input is the padded `*_npad` projection).
    return [n for n in topo["nodes"] if n["op"] == "VIEW" and not n["inputs"][0].endswith("_npad")]


class TestSplitSizes(unittest.TestCase):
    def test_unequal_sizes_are_the_views(self):
        views = _views((6, 2, 2))
        self.assertEqual(len(views), 3)
        # ne-order: the split (feature) axis first. The bug wrote ['3', ...] three times.
        self.assertEqual([str(v["attrs"]["shape"][0]) for v in views], ["6", "2", "2"])
        self.assertEqual([str(v["attrs"]["offset"]) for v in views], ["0", "24", "32"])

    def test_equal_sizes_render_as_before(self):
        views = _views((4, 4, 4))
        self.assertEqual([str(v["attrs"]["shape"][0]) for v in views], ["4", "4", "4"])
        self.assertEqual([str(v["attrs"]["offset"]) for v in views], ["0", "16", "32"])


if __name__ == "__main__":
    unittest.main()
