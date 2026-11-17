"""A token axis must survive a fused RMS norm upstream of a shape-derived slice.

VoxCPM2's feature encoder is this module at full size: each token is a 5-row sequence, normalized, and
`h[:, 0, :]` keeps the first row. MIL mints a fresh symbol for that slice, and the exporter relates it
back to `n_tokens` by walking producers through the ops it knows preserve shape. `loom_rms_norm` was
not one of them, so once `fuse_rms_norm` fused VoxCPM2's norms the walk stopped there and the VIEW and
RESHAPE after the slice were written with a literal 1: `feat_encode` returned one row whatever it was
given, and every VoxCPM2 call failed at its first retained copy (2026-11-16).
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class _Norm(torch.nn.Module):
    def __init__(self, d, spelling):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(d) + 0.5)
        self.spelling = spelling

    def forward(self, x):
        sq = x * x if self.spelling == "mul" else x.pow(2)
        return x * torch.rsqrt(sq.mean(-1, keepdim=True) + 1e-5) * self.w


class _PatchEncoder(torch.nn.Module):
    """`(1, T, 4, 8) -> (1, T, 12)`: VoxCPM2's `FeatEncodePhase` with one norm for its stack."""

    def __init__(self, spelling):
        super().__init__()
        self.inp = torch.nn.Linear(8, 16)
        self.tok = torch.nn.Parameter(torch.randn(16))
        self.norm = _Norm(16, spelling)
        self.out = torch.nn.Linear(16, 12)

    def forward(self, patches):
        x = self.inp(patches)
        special = x[:, :, :1, :] * 0.0 + self.tok
        x = torch.cat([special, x], dim=2)[0]          # (T, 5, 16)
        return self.out(self.norm(x)[:, 0, :]).unsqueeze(0)


def _nodes(spelling):
    traced = torch.jit.trace(_PatchEncoder(spelling).eval(), (torch.randn(1, 3, 4, 8),))
    prog = ct.convert(traced, inputs=[ct.TensorType(name="patches", shape=(1, ct.RangeDim(1, 64, default=3), 4, 8))],
                      convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        exporter = LoomGGUFExporter(prog, output_path=str(Path(d) / "patch.gguf"), architecture="patch_test")
        exporter.export()
    return next(iter(exporter.topologies.values()))["nodes"]


class TestAxisWalkThroughRmsNorm(unittest.TestCase):
    def _check(self, spelling):
        nodes = _nodes(spelling)
        self.assertIn("RMS_NORM", [n["op"] for n in nodes], "the norm did not fuse; this test would prove nothing")
        out = nodes[-1]
        self.assertEqual(out["op"], "RESHAPE")
        # The output is one row per token: [12, n_tokens, 1], never [12, 1, 1].
        self.assertEqual(out["attrs"]["shape"], ["12", "n_tokens", "1"])

    def test_x_times_x(self):
        self._check("mul")

    def test_pow_2(self):
        self._check("pow")


if __name__ == "__main__":
    unittest.main()
