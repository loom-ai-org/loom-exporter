"""A `linear` lowers to MUL_MAT + ADD only when its bias is not all zero.

MIL hands a bias-free `nn.Linear` an all-zero bias const, and adding it was one kernel per projection
that adds 0.0: on CosyVoice3's Qwen2 LM, four a layer (o_proj, gate, up, down), 96 per token on a GPU
where each launch costs ~1.5 us (2026-11-12). `x + 0.0` is `x` but for the sign of a zero, so the ADD is
left out -- and kept for any bias with a single non-zero element.
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class _Proj(torch.nn.Module):
    def __init__(self, bias, zero_bias=False, poke=False):
        super().__init__()
        self.proj = torch.nn.Linear(6, 11, bias=bias)
        if bias:
            with torch.no_grad():
                if zero_bias:
                    self.proj.bias.zero_()
                if poke:
                    self.proj.bias[7] = 1e-30   # one tiny non-zero: still a bias
    def forward(self, x):
        return torch.tanh(self.proj(x))


def _nodes(module):
    x = torch.randn(4, 6)
    traced = torch.jit.trace(module.eval(), (x,))
    prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal",
                      compute_precision=ct.precision.FLOAT32)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        exporter = LoomGGUFExporter(prog, output_path=str(Path(d) / "linear.gguf"), architecture="linear_test")
        exporter.export()
    return next(iter(exporter.topologies.values()))["nodes"]


def _ops_feeding_tanh(nodes):
    producer = {o: n for n in nodes for o in n["outputs"]}
    tanh = next(n for n in nodes if n["op"] in ("TANH", "UNARY") or "TANH" in n["op"])
    return producer[tanh["inputs"][0]]["op"]


class TestLinearBiasLowering(unittest.TestCase):
    def test_a_bias_free_linear_is_one_mul_mat(self):
        nodes = _nodes(_Proj(bias=False))
        self.assertEqual([n["op"] for n in nodes if n["op"] in ("MUL_MAT", "ADD")], ["MUL_MAT"])
        self.assertEqual(_ops_feeding_tanh(nodes), "MUL_MAT")

    def test_an_all_zero_bias_is_left_out(self):
        nodes = _nodes(_Proj(bias=True, zero_bias=True))
        self.assertEqual([n["op"] for n in nodes if n["op"] in ("MUL_MAT", "ADD")], ["MUL_MAT"])

    def test_a_real_bias_is_added(self):
        nodes = _nodes(_Proj(bias=True))
        self.assertEqual([n["op"] for n in nodes if n["op"] in ("MUL_MAT", "ADD")], ["MUL_MAT", "ADD"])
        self.assertEqual(_ops_feeding_tanh(nodes), "ADD")

    def test_one_non_zero_element_keeps_the_add(self):
        nodes = _nodes(_Proj(bias=True, zero_bias=True, poke=True))
        self.assertEqual([n["op"] for n in nodes if n["op"] in ("MUL_MAT", "ADD")], ["MUL_MAT", "ADD"])


if __name__ == "__main__":
    unittest.main()
