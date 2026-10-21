"""`topology_cse.eliminate_common_subexpressions` (loom.cpp ADR-079 P4.32f).

The pass merges repeated pure nodes in a finished topology. What it must never do is merge two nodes
that are NOT the same tensor, or drop a name a driver reads -- so most of this is the refusals.
"""
import copy
import json
import unittest

import numpy as np

from loom_exporter.topology_cse import PURE_OPS, eliminate_common_subexpressions


def _mask(i, x, end="floor(n/2) + 1"):
    """One NeMo time-mask chain as Citrinet-256-ls's topology carries it, applied to `x`."""
    return [
        {"op": "RANGE_1D", "inputs": [], "outputs": [f"r{i}"], "attrs": {"start": 0.0, "end": end, "step": 1.0}},
        {"op": "RESHAPE", "inputs": [f"r{i}"], "outputs": [f"rs{i}"], "attrs": {"shape": [end, "1"]}},
        {"op": "CAST", "inputs": [f"rs{i}"], "outputs": [f"c{i}"], "attrs": {"dtype": "f32"}},
        {"op": "LESS", "inputs": [f"c{i}", "length"], "outputs": [f"v{i}"]},
        {"op": "MUL", "inputs": [x, f"v{i}"], "outputs": [f"y{i}"]},
    ]


class TestTopologyCse(unittest.TestCase):
    def test_repeated_mask_chains_collapse_to_one(self):
        nodes = [{"op": "RELU", "inputs": ["x"], "outputs": ["a0"]}]
        nodes += _mask(0, "a0")
        nodes += [{"op": "CONV_1D", "inputs": ["w", "y0"], "outputs": ["a1"]}]
        nodes += _mask(1, "a1")
        topo = {"inputs": [], "output": "y1", "nodes": nodes}

        out, removed = eliminate_common_subexpressions(topo)
        self.assertEqual(removed, 4)                       # RANGE_1D, RESHAPE, CAST, LESS of chain 1
        ops = [n["op"] for n in out["nodes"]]
        self.assertEqual(ops, ["RELU", "RANGE_1D", "RESHAPE", "CAST", "LESS", "MUL", "CONV_1D", "MUL"])
        # the second MUL now reads the FIRST mask, and directly follows the convolution it multiplies --
        # which is what lets ggml's CPU backend fold it into the convolution's epilogue
        self.assertEqual(out["nodes"][-1]["inputs"], ["a1", "v0"])
        self.assertEqual(topo["nodes"][-1]["inputs"], ["a1", "v1"], "the input topology was modified")

    def test_a_declared_output_is_never_removed(self):
        nodes = _mask(0, "x") + _mask(1, "x")
        out, _ = eliminate_common_subexpressions({"output": "y1", "nodes": nodes})
        self.assertIn("y1", [o for n in out["nodes"] for o in n["outputs"]])
        multi = {"outputs": ["y0", "y1"], "nodes": _mask(0, "x") + _mask(1, "x")}
        out, _ = eliminate_common_subexpressions(multi)
        names = [o for n in out["nodes"] for o in n["outputs"]]
        self.assertIn("y0", names)
        self.assertIn("y1", names)

    def test_impure_and_unlisted_ops_are_left_alone(self):
        for op in ("ATTENTION", "SHORT_CONV", "SSM_SCAN", "RWKV_WKV7", "CONV_1D", "MUL_MAT", "GET_ROWS"):
            self.assertNotIn(op, PURE_OPS)
            nodes = [{"op": op, "inputs": ["x", "w"], "outputs": ["a"]},
                     {"op": op, "inputs": ["x", "w"], "outputs": ["b"]},
                     {"op": "ADD", "inputs": ["a", "b"], "outputs": ["y"]}]
            out, removed = eliminate_common_subexpressions({"output": "y", "nodes": nodes})
            self.assertEqual(removed, 0, op)
            self.assertEqual(len(out["nodes"]), 3, op)

    def test_different_attributes_or_inputs_do_not_merge(self):
        nodes = (_mask(0, "x", end="floor(n/2) + 1") + _mask(1, "x", end="floor(n/4) + 1") +
                 [{"op": "LESS", "inputs": ["c0", "other_length"], "outputs": ["z"]},
                  {"op": "ADD", "inputs": ["y0", "y1"], "outputs": ["s"]},
                  {"op": "ADD", "inputs": ["s", "z"], "outputs": ["out"]}])
        out, removed = eliminate_common_subexpressions({"output": "out", "nodes": nodes})
        self.assertEqual(removed, 0)
        # operand order is part of the key: SUB(a, b) and SUB(b, a) are different tensors
        sub = [{"op": "SUB", "inputs": ["a", "b"], "outputs": ["p"]},
               {"op": "SUB", "inputs": ["b", "a"], "outputs": ["q"]},
               {"op": "ADD", "inputs": ["p", "q"], "outputs": ["y"]}]
        self.assertEqual(eliminate_common_subexpressions({"output": "y", "nodes": sub})[1], 0)

    def test_numpy_attributes_key_by_value_and_dtype(self):
        f = {"op": "FILL", "inputs": [], "attrs": {"value": np.float32(0.5), "shape": np.array([2, 3])}}
        nodes = [dict(f, outputs=["a"]), dict(f, outputs=["b"]),
                 {"op": "FILL", "inputs": [], "outputs": ["c"],
                  "attrs": {"value": np.float32(0.5), "shape": np.array([2, 3], dtype=np.int8)}},
                 {"op": "ADD", "inputs": ["a", "b"], "outputs": ["s"]},
                 {"op": "ADD", "inputs": ["s", "c"], "outputs": ["y"]}]
        out, removed = eliminate_common_subexpressions({"output": "y", "nodes": nodes})
        self.assertEqual(removed, 1)                       # b merges into a; c differs by dtype
        self.assertEqual(out["nodes"][2]["inputs"], ["a", "a"])
        json.dumps(out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))

    def test_multi_output_nodes_map_position_by_position(self):
        nodes = [{"op": "CONCAT", "inputs": ["x", "y"], "outputs": ["a", "b"], "attrs": {"axis": 0}},
                 {"op": "CONCAT", "inputs": ["x", "y"], "outputs": ["c", "d"], "attrs": {"axis": 0}},
                 {"op": "ADD", "inputs": ["b", "d"], "outputs": ["s"]},
                 {"op": "ADD", "inputs": ["a", "c"], "outputs": ["t"]}]
        out, removed = eliminate_common_subexpressions({"outputs": ["s", "t"], "nodes": nodes})
        self.assertEqual(removed, 1)
        self.assertEqual(out["nodes"][1]["inputs"], ["b", "b"])
        self.assertEqual(out["nodes"][2]["inputs"], ["a", "a"])

    def test_a_rebound_name_is_never_the_survivor(self):
        # Chatterbox's estimator (rc18 card gate): `transpose_0` is written, padded, and written again;
        # the residual conv's own PERMUTE(input_7) must not be redirected to the second `transpose_0`.
        nodes = [{"op": "PERMUTE", "inputs": ["input_7"], "outputs": ["transpose_0"], "attrs": {"axes": [1, 0, 2, 3]}},
                 {"op": "PAD_1D", "inputs": ["transpose_0"], "outputs": ["pad_0"], "attrs": {"lp0": 2, "rp0": 0}},
                 {"op": "PERMUTE", "inputs": ["pad_0"], "outputs": ["input_9"], "attrs": {"axes": [1, 0, 2, 3]}},
                 {"op": "PERMUTE", "inputs": ["input_9"], "outputs": ["transpose_0"], "attrs": {"axes": [1, 0, 2, 3]}},
                 {"op": "CONV_1D", "inputs": ["w0", "transpose_0"], "outputs": ["h"]},
                 {"op": "PERMUTE", "inputs": ["input_7"], "outputs": ["transpose_1"], "attrs": {"axes": [1, 0, 2, 3]}},
                 {"op": "CONV_1D", "inputs": ["w1", "transpose_1"], "outputs": ["r"]},
                 {"op": "ADD", "inputs": ["h", "r"], "outputs": ["y"]}]
        topo = {"inputs": [{"name": "input_7"}], "output": "y", "nodes": nodes}
        out, removed = eliminate_common_subexpressions(topo)
        self.assertEqual(removed, 0)
        self.assertEqual(out, topo)

    def test_a_rebound_name_is_never_merged_away(self):
        # the later `a` shadows the RELU; a use after it must still read the CONV_1D, not the first `a`
        nodes = [{"op": "RELU", "inputs": ["x"], "outputs": ["b"]},
                 {"op": "RELU", "inputs": ["x"], "outputs": ["a"]},
                 {"op": "ADD", "inputs": ["a", "b"], "outputs": ["s"]},
                 {"op": "CONV_1D", "inputs": ["w", "s"], "outputs": ["a"]},
                 {"op": "ADD", "inputs": ["a", "s"], "outputs": ["y"]}]
        topo = {"output": "y", "nodes": nodes}
        out, removed = eliminate_common_subexpressions(topo)
        self.assertEqual(removed, 0)
        self.assertEqual(out, topo)

    def test_the_same_name_either_side_of_a_rewrite_is_two_inputs(self):
        # `x` is a graph input written again by the CONV_1D: the two RELU(x) read different tensors
        nodes = [{"op": "RELU", "inputs": ["x"], "outputs": ["a"]},
                 {"op": "CONV_1D", "inputs": ["w", "a"], "outputs": ["x"]},
                 {"op": "RELU", "inputs": ["x"], "outputs": ["b"]},
                 {"op": "ADD", "inputs": ["a", "b"], "outputs": ["y"]}]
        topo = {"inputs": [{"name": "x"}], "output": "y", "nodes": nodes}
        out, removed = eliminate_common_subexpressions(topo)
        self.assertEqual(removed, 0)
        self.assertEqual(out, topo)
        # ... while two reads of the SAME definition still merge
        nodes2 = nodes + [{"op": "RELU", "inputs": ["x"], "outputs": ["c"]},
                          {"op": "ADD", "inputs": ["y", "c"], "outputs": ["z"]}]
        out, removed = eliminate_common_subexpressions({"inputs": [{"name": "x"}], "output": "z", "nodes": nodes2})
        self.assertEqual(removed, 1)
        self.assertEqual(out["nodes"][-1]["inputs"], ["y", "b"])

    def test_a_topology_without_duplicates_comes_back_equal(self):
        nodes = [{"op": "RELU", "inputs": ["x"], "outputs": ["a"]},
                 {"op": "MUL", "inputs": ["a", "x"], "outputs": ["y"]}]
        topo = {"inputs": [{"name": "x"}], "output": "y", "nodes": nodes}
        out, removed = eliminate_common_subexpressions(copy.deepcopy(topo))
        self.assertEqual(removed, 0)
        self.assertEqual(out, topo)


if __name__ == "__main__":
    unittest.main()
