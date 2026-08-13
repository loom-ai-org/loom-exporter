"""
Checks `passes.py`'s MIL->MIL canonicalizing passes (EXPORT-ROADMAP.md R2a/R2): `normalize_matmul`,
`insert_explicit_broadcasts`, `canonicalize_replicate_pad`, `canonicalize_conv_transpose_dw`,
`lower_stack`, `lower_reduce_mean`. Each runs as a real rewrite over a hand-built `mb.program`, checked
directly against the rewritten graph's structure -- `fuse_gqa_repeat_kv` (the pass these all join) has no
equivalent unit test of its own, relying entirely on numerical e2e reference tests instead, but not every
pass here has a model on the current roadmap that actually exercises its rewrite (matmul's
`transpose_x=True` has never been needed by any traced model; SupertonicTTS/Matcha/Kokoro/StyleTTS2 do
exercise the rest, via e2e reference tests covering the *pipeline*, not these passes in isolation) -- so
this is the only place every rewrite itself gets verified in isolation.
"""
import unittest

import sys
from pathlib import Path

import numpy as np
from coremltools.converters.mil.mil import Builder as mb, get_new_symbol, types

from loom_exporter.paths import CONVERTERS, driver_dir
from loom_exporter.passes import PASS_REGISTRY, apply_loom_mil_passes


def _ops(prog):
    return [op.op_type for op in prog.functions["main"].operations if op.op_type != "const"]


class TestNormalizeMatmul(unittest.TestCase):
    def test_rewrites_transpose_x_true(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(2, 4), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=True, transpose_y=False)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        self.assertEqual(_ops(prog), ["transpose", "matmul"])
        transpose_op, matmul_op = (op for op in prog.functions["main"].operations if op.op_type != "const")
        self.assertEqual(list(transpose_op.perm.val), [1, 0])
        self.assertFalse(bool(matmul_op.transpose_x.val))
        self.assertFalse(bool(matmul_op.transpose_y.val))
        self.assertEqual(matmul_op.x.op, transpose_op)
        # Output shape/name are preserved -- (3,4), matching the original transpose_x=True result.
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (3, 4))

    def test_preserves_transpose_y(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 2), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=True, transpose_y=True)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        matmul_op = prog.functions["main"].operations[-1]
        self.assertFalse(bool(matmul_op.transpose_x.val))
        self.assertTrue(bool(matmul_op.transpose_y.val))
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (3, 4))

    def test_leaves_transpose_x_false_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 3), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=False, transpose_y=True)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        self.assertEqual(_ops(prog), ["matmul"])


class TestInsertExplicitBroadcasts(unittest.TestCase):
    def test_mutual_broadcast_gets_two_loom_broadcast_to_ops(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(32, 1, 1), dtype=types.fp32),
                                  mb.TensorSpec(shape=(1, 5, 1), dtype=types.fp32)])
        def prog(x, y):
            return mb.mul(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["loom_broadcast_to", "loom_broadcast_to", "mul"])
        bx, by, mul_op = prog.functions["main"].operations
        self.assertEqual(mul_op.x.op, bx)
        self.assertEqual(mul_op.y.op, by)
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (32, 5, 1))

    def test_mutual_broadcast_with_a_dynamic_axis(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(32, 1, 1), dtype=types.fp32),
                                  mb.TensorSpec(shape=(1, length, 1), dtype=types.fp32)])
        def prog(x, y):
            return mb.add(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["loom_broadcast_to", "loom_broadcast_to", "add"])
        out_shape = prog.functions["main"].outputs[0].shape
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], length)
        self.assertEqual(out_shape[2], 1)

    def test_single_operand_broadcast_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(3, 5), dtype=types.fp32)])
        def prog(x, y):
            return mb.add(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["add"])

    def test_no_broadcast_needed_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(3, 5), dtype=types.fp32)])
        def prog(x, y):
            return mb.mul(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["mul"])


class TestCanonicalizeReplicatePad(unittest.TestCase):
    def test_rewrites_a_replicate_pad_into_loom_replicate_pad(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, length), dtype=types.fp32)])
        def prog(x):
            return mb.pad(x=x, pad=[2, 3], mode="replicate")

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["loom_replicate_pad"])
        op = prog.functions["main"].operations[-1]
        self.assertEqual(int(op.lp.val), 2)
        self.assertEqual(int(op.rp.val), 3)

    def test_a_zero_pad_is_aliased_away_entirely(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            padded = mb.pad(x=x, pad=[0, 0], mode="replicate")
            return mb.mul(x=padded, y=1.0)

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["mul"])
        mul_op = prog.functions["main"].operations[-1]
        self.assertEqual(mul_op.x.name, "x")

    def test_non_replicate_modes_are_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            return mb.pad(x=x, pad=[2, 3], mode="constant", constant_val=0.0)

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["pad"])

    def test_a_non_fastest_axis_replicate_pad_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            # pad=[1,1,0,0]: the FIRST (non-fastest, mil_axis=1) of the two padded axes is
            # non-zero; the last (fastest) axis is left alone.
            return mb.pad(x=x, pad=[1, 1, 0, 0], mode="replicate")

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)
        self.assertIn("fastest-varying", str(cm.exception))


class TestCanonicalizeConvTransposeDw(unittest.TestCase):
    def test_rewrites_a_depthwise_conv_transpose(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, length), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2], pad_type="valid", groups=4)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        self.assertEqual(_ops(prog), ["loom_conv_transpose_dw"])
        op = prog.functions["main"].operations[-1]
        self.assertEqual(int(op.stride.val), 2)
        self.assertIsNone(op.bias)

    def test_a_biased_depthwise_conv_transpose_keeps_its_bias(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3), dtype=np.float32), name="w")
            b = mb.const(val=np.ones((4,), dtype=np.float32), name="b")
            return mb.conv_transpose(x=x, weight=w, bias=b, strides=[2], pad_type="valid", groups=4)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        op = prog.functions["main"].operations[-1]
        self.assertEqual(op.op_type, "loom_conv_transpose_dw")
        self.assertIsNotNone(op.bias)
        self.assertTrue(np.all(op.bias.val == 1.0))

    def test_a_non_grouped_conv_transpose_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 4, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2], pad_type="valid", groups=1)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        self.assertEqual(_ops(prog), ["conv_transpose"])

    def test_a_2d_grouped_conv_transpose_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2, 2], pad_type="valid", groups=4)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)
        self.assertIn("groups=4", str(cm.exception))


class TestLowerStack(unittest.TestCase):
    def test_two_operand_stack_becomes_expand_dims_and_concat(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 5), dtype=types.fp32)])
        def prog(a, b):
            return mb.stack(values=(a, b), axis=-1)

        PASS_REGISTRY["loom::lower_stack"](prog)

        self.assertEqual(_ops(prog), ["expand_dims", "expand_dims", "concat"])
        out_var = prog.functions["main"].outputs[0]
        self.assertEqual(tuple(out_var.shape), (4, 5, 2))

    def test_single_operand_stack_becomes_a_bare_expand_dims(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5), dtype=types.fp32)])
        def prog(a):
            return mb.stack(values=(a,), axis=0)

        PASS_REGISTRY["loom::lower_stack"](prog)

        self.assertEqual(_ops(prog), ["expand_dims"])
        out_var = prog.functions["main"].outputs[0]
        self.assertEqual(tuple(out_var.shape), (1, 4, 5))


class TestFuseRmsNorm(unittest.TestCase):
    """`fuse_rms_norm` collapses PyTorch's RMSNorm chain to one `loom_rms_norm`.

    Every negative case below is a graph that LOOKS like the pattern and is not one, because that is
    where a fusion pass does its damage: emitting `RMS_NORM` for a chain that normalizes a different
    tensor, a different axis, or by something other than the mean square is silently wrong arithmetic
    that no shape check downstream would catch.
    """

    @staticmethod
    def _rms_program(eps=1e-6, exponent=2.0, axes=(-1,), keep_dims=True, feed_back_same_x=True):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            squared = mb.pow(x=x, y=np.float32(exponent))
            variance = mb.reduce_mean(x=squared, axes=list(axes), keep_dims=keep_dims)
            shifted = mb.add(x=variance, y=np.float32(eps))
            scale = mb.rsqrt(x=shifted)
            other = x if feed_back_same_x else mb.identity(x=x)
            return mb.mul(x=other, y=scale)
        return prog

    def test_the_chain_becomes_one_op(self):
        prog = self._rms_program()
        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)

        self.assertEqual(_ops(prog), ["loom_rms_norm"])

    def test_epsilon_includes_mils_own_rsqrt_epsilon(self):
        """MIL's `rsqrt` computes 1/sqrt(x + epsilon) with an epsilon of its own, so the value the
        engine must add to the mean square is the SUM of the two -- not the constant the model wrote.
        Getting this wrong is a wrong answer nothing else would flag."""
        prog = self._rms_program(eps=1e-6)
        PASS_REGISTRY["loom::fuse_rms_norm"](prog)

        op = next(o for o in prog.functions["main"].operations if o.op_type == "loom_rms_norm")
        rsqrt_default = 1e-12
        self.assertAlmostEqual(float(op.epsilon.val), 1e-6 + rsqrt_default, places=12)
        self.assertGreater(float(op.epsilon.val), 1e-6)

    def test_operands_may_be_written_in_either_order(self):
        """MIL does not normalize commutative operands, and both `x * rsqrt(v)` and `rsqrt(v) * x` are
        written in the wild."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            squared = mb.pow(x=x, y=np.float32(2.0))
            variance = mb.reduce_mean(x=squared, axes=[-1], keep_dims=True)
            scale = mb.rsqrt(x=mb.add(x=np.float32(1e-6), y=variance))
            return mb.mul(x=scale, y=x)

        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
        self.assertEqual(_ops(prog), ["loom_rms_norm"])

    def test_a_multiply_by_a_different_tensor_is_not_rms_norm(self):
        """The tensor being scaled must be the very one that was squared. Two structurally identical
        tensors are still two tensors, and normalizing by the wrong one is the bug this refuses."""
        prog = self._rms_program(feed_back_same_x=False)
        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        self.assertIn("rsqrt", _ops(prog))
        self.assertNotIn("loom_rms_norm", _ops(prog))

    def test_a_mean_over_another_axis_is_not_rms_norm(self):
        """`ggml_rms_norm` normalizes ne[0] and nothing else."""
        prog = self._rms_program(axes=(1,))
        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        self.assertNotIn("loom_rms_norm", _ops(prog))

    def test_a_power_other_than_two_is_not_rms_norm(self):
        prog = self._rms_program(exponent=3.0)
        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        self.assertNotIn("loom_rms_norm", _ops(prog))

    def test_a_shared_intermediate_is_left_alone(self):
        """A `variance` some other node also reads must survive the rewrite, so the rewrite does not
        happen. DCE would otherwise not collect it and the graph would keep both spellings."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            squared = mb.pow(x=x, y=np.float32(2.0))
            variance = mb.reduce_mean(x=squared, axes=[-1], keep_dims=True)
            scale = mb.rsqrt(x=mb.add(x=variance, y=np.float32(1e-6)))
            normed = mb.mul(x=x, y=scale)
            # `variance` escapes as a second output -- a real thing a traced model can do.
            return mb.add(x=normed, y=variance)

        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        self.assertNotIn("loom_rms_norm", _ops(prog))

    def test_the_learned_affine_stays_a_separate_multiply(self):
        """`ggml_rms_norm` has no affine of its own, matching LAYER_NORM/GROUP_NORM's convention."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            squared = mb.pow(x=x, y=np.float32(2.0))
            variance = mb.reduce_mean(x=squared, axes=[-1], keep_dims=True)
            scale = mb.rsqrt(x=mb.add(x=variance, y=np.float32(1e-6)))
            return mb.mul(x=mb.mul(x=x, y=scale), y=np.random.rand(8).astype(np.float32))

        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
        self.assertEqual(_ops(prog), ["loom_rms_norm", "mul"])


class TestLowerPow(unittest.TestCase):
    """`lower_pow` turns `pow(x, 2)` into `square`, which OP_MAP already carries to the engine's SQR.

    Every `pow` this exporter has ever emitted is a square (149 across the thirteen fixture models), so
    the interesting cases are the ones it must NOT claim.
    """

    @staticmethod
    def _pow_program(exponent):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32)])
        def prog(x):
            return mb.pow(x=x, y=np.float32(exponent))
        return prog

    def test_squaring_becomes_square(self):
        prog = self._pow_program(2.0)
        PASS_REGISTRY["loom::lower_pow"](prog)
        self.assertEqual(_ops(prog), ["square"])

    def test_any_other_exponent_is_left_alone(self):
        for exponent in (3.0, 0.5, 1.0, -1.0):
            prog = self._pow_program(exponent)
            PASS_REGISTRY["loom::lower_pow"](prog)
            self.assertEqual(_ops(prog), ["pow"], f"exponent {exponent} should not have been rewritten")

    def test_a_non_constant_exponent_is_left_alone(self):
        """A tensor exponent is a real `pow` -- there is no constant to check."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(2, 3), dtype=types.fp32)])
        def prog(x, y):
            return mb.pow(x=x, y=y)

        PASS_REGISTRY["loom::lower_pow"](prog)
        self.assertEqual(_ops(prog), ["pow"])


class TestFuseLayerNorm(unittest.TestCase):
    """`fuse_layer_norm` recognises a hand-rolled layer norm and emits MIL's `layer_norm`, transposed
    into place when the normalized axis is not the trailing one (`ggml_norm` only ever does ne[0])."""

    @staticmethod
    def _ln_program(axis, shape=(1, 8, 5), eps=1e-4, same_axis_for_variance=True, centre=True):
        @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp32)])
        def prog(x):
            mean = mb.reduce_mean(x=x, axes=[axis], keep_dims=True)
            centered = mb.sub(x=x, y=mean) if centre else x
            var_axis = axis if same_axis_for_variance else (axis + 1) % len(shape)
            variance = mb.reduce_mean(x=mb.pow(x=centered, y=np.float32(2.0)),
                                       axes=[var_axis], keep_dims=True)
            scale = mb.rsqrt(x=mb.add(x=variance, y=np.float32(eps)))
            return mb.mul(x=centered, y=scale)
        return prog

    def test_a_channel_axis_norm_is_transposed_into_ne0_and_back(self):
        """`ggml_norm` only does ne[0]. The two copies a permute costs are cheaper than the eight-op
        chain they replace -- measured, see the pass's docstring for the numbers and for the alternative
        rewrite that avoids the copies and is slower anyway."""
        prog = self._ln_program(axis=1)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)

        self.assertEqual(_ops(prog), ["transpose", "layer_norm", "transpose"])
        ops = [op for op in prog.functions["main"].operations if op.op_type != "const"]
        # The same permutation undoes itself, which is why one list serves both ends.
        self.assertEqual(list(ops[0].perm.val), [0, 2, 1])
        self.assertEqual(list(ops[2].perm.val), [0, 2, 1])
        self.assertEqual(list(ops[1].axes.val), [-1])

    def test_a_trailing_axis_norm_needs_no_transpose(self):
        prog = self._ln_program(axis=-1)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
        self.assertEqual(_ops(prog), ["layer_norm"])

    def test_epsilon_includes_mils_own_rsqrt_epsilon(self):
        """Asserted at eps=1e-6, not at the 1e-4 Matcha uses, and the difference is the point: fp32's
        ULP at 1e-4 is about 7e-12, so MIL's 1e-12 rsqrt epsilon rounds straight back out of the sum
        there and the fused value is indistinguishable from the written one. It survives at 1e-6, where
        the ULP is ~1e-13. Both are correct; only the smaller one can show the term was added at all."""
        prog = self._ln_program(axis=1, eps=1e-6)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        op = next(o for o in prog.functions["main"].operations if o.op_type == "layer_norm")
        # Against the fp32 ROUND-TRIP of the sum, not against the double: the op stores a fp32 const
        # like every other MIL float, so that is the value the engine will see.
        self.assertEqual(np.float32(op.epsilon.val), np.float32(1e-6 + 1e-12))
        self.assertGreater(float(op.epsilon.val), float(np.float32(1e-6)))

    def test_two_means_over_different_axes_are_not_a_layer_norm(self):
        """The mean and the variance must reduce the SAME axis; a graph where they differ is something
        else, whatever it looks like."""
        prog = self._ln_program(axis=1, same_axis_for_variance=False)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        self.assertNotIn("layer_norm", _ops(prog))

    def test_without_the_mean_centring_it_is_not_a_layer_norm(self):
        """That graph is an RMS norm, and belongs to the other pass. A matcher that treated `sub` as
        optional would emit LAYER_NORM for one the moment the sub failed to match."""
        prog = self._ln_program(axis=1, centre=False)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        self.assertNotIn("layer_norm", _ops(prog))

    def test_it_also_matches_after_lower_pow_has_run(self):
        """`_squares` accepts both `pow(x,2)` and `square`, so pipeline order is a preference rather
        than a dependency -- this is the assertion that keeps it that way."""
        prog = self._ln_program(axis=1)
        PASS_REGISTRY["loom::lower_pow"](prog)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
        self.assertEqual(_ops(prog), ["transpose", "layer_norm", "transpose"])

    def test_an_rms_norm_is_not_claimed_by_this_pass(self):
        """Both passes anchor on a `mul`; running the pair must not let either take the other's graph."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            variance = mb.reduce_mean(x=mb.pow(x=x, y=np.float32(2.0)), axes=[-1], keep_dims=True)
            return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=np.float32(1e-6))))

        PASS_REGISTRY["loom::fuse_rms_norm"](prog)
        PASS_REGISTRY["loom::fuse_layer_norm"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
        self.assertEqual(_ops(prog), ["loom_rms_norm"])


class TestLowerReduceMean(unittest.TestCase):
    def test_static_count_becomes_reduce_sum_and_loom_scale(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 192, 1), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[1], keep_dims=True)

        PASS_REGISTRY["loom::lower_reduce_mean"](prog)

        self.assertEqual(_ops(prog), ["reduce_sum", "loom_scale"])
        scale_op = prog.functions["main"].operations[-1]
        self.assertEqual(int(scale_op.n.val), 192)

    def test_dynamic_count_on_the_fastest_axis_becomes_loom_mean(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1024, length), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[-1], keep_dims=False)

        PASS_REGISTRY["loom::lower_reduce_mean"](prog)

        self.assertEqual(_ops(prog), ["loom_mean"])

    def test_dynamic_count_on_a_non_fastest_axis_raises(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(length, 512), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[0], keep_dims=False)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::lower_reduce_mean"](prog)
        self.assertIn("only known at run time", str(cm.exception))

    def test_multi_axis_reduce_mean_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8, 1), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[0, 1], keep_dims=False)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::lower_reduce_mean"](prog)
        self.assertIn("single reduction axis", str(cm.exception))


class TestFuseLoomAttention(unittest.TestCase):
    """`fuse_loom_attention` (KV-CACHE.md stage 2). Unlike the other passes here, a miss is not merely a
    missed optimization: this op is the only node type that can reach the engine's KV cache, so a
    silently-unmatched block is a model that cannot generate. The negative cases matter as much -- the
    pattern is generic SDPA, and it must not fire on graphs that only look like it."""

    N_HEAD, HEAD_DIM, SEQ = 4, 8, 6

    def _sdpa(self, scale=0.35355338, transpose_y=True, trailing=True, n_blocks=1):
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ
        specs = [mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32) for _ in range(3)]
        specs.append(mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32))

        @mb.program(input_specs=specs)
        def prog(q, k, v, mask):
            out = None
            for _ in range(n_blocks):
                qs = q if scale is None else mb.mul(x=q, y=np.float32(scale))
                scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=transpose_y)
                scores = mb.add(x=scores, y=mask)
                probs = mb.softmax(x=scores, axis=-1)
                ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
                if not trailing:
                    out = ctx
                    continue
                ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
                out = mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])
            return out

        return prog

    def _fused(self, prog):
        return [op for op in prog.functions["main"].operations if op.op_type == "loom_fused_attention"]

    def test_a_whole_sdpa_block_becomes_one_op(self):
        prog = self._sdpa()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = self._fused(prog)
        self.assertEqual(len(fused), 1)
        self.assertNotIn("softmax", _ops(prog))
        # The trailing transpose+reshape are absorbed too: op_attention returns the flattened context,
        # so the op's declared type has to be [b, seq, n_head*head_dim] or it disagrees with the engine.
        self.assertEqual(tuple(fused[0].outputs[0].shape), (1, self.SEQ, self.N_HEAD * self.HEAD_DIM))
        self.assertAlmostEqual(float(fused[0].scale.val), 0.35355338, places=6)

    def test_layer_indices_are_assigned_in_occurrence_order(self):
        # The index addresses a CACHE SLOT, and the cache has one slot per attention block -- so a dense
        # occurrence index is correct even for an architecture that interleaves non-attention layers,
        # where the torch module index would address past the end of the cache.
        prog = self._sdpa(n_blocks=3)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual([int(op.layer.val) for op in self._fused(prog)], [0, 1, 2])

    def test_a_late_value_projection_still_fuses_and_keeps_ssa_order(self):
        """V computed AFTER Q@K^T -- HF Whisper's decoder self-attention, and the shape that found the
        insertion-point bug (BACKEND.md P4.1). The old anchor was the QK matmul unconditionally, which
        put the fused op above the var it reads; nothing rejected it until `dead_code_elimination` walked
        the block in reverse and raised `Cannot delete op ... with active output`."""
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ
        specs = [mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32) for _ in range(3)]
        specs.append(mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32))

        @mb.program(input_specs=specs)
        def prog(q, k, v_src, mask):
            qs = mb.mul(x=q, y=np.float32(0.35355338))
            scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=True)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            # The value projection lands here, below the matmul that the fusion used to anchor on.
            v = mb.mul(x=v_src, y=np.float32(1.0))
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)
        fused = self._fused(prog)
        self.assertEqual(len(fused), 1)

        # Every operand is defined strictly above the op that reads it -- the property MIL's builder does
        # not check and DCE relies on.
        order = {id(op): i for i, op in enumerate(prog.functions["main"].operations)}
        for name in ("q", "k", "v", "mask"):
            producer = fused[0].inputs[name].op
            if producer is not None:
                self.assertLess(order[id(producer)], order[id(fused[0])], f"{name} is defined too late")
        PASS_REGISTRY["common::dead_code_elimination"](prog)

    def test_the_anchor_does_not_move_when_the_old_one_was_already_valid(self):
        """The earliest *valid* chain position is chosen, not simply the last -- which is what keeps
        every already-fusing model's node order (and therefore its exported topology) unchanged."""
        def real_ops(p):
            # `const`s are ignored: the fused op's own `scale`/`layer` constants are materialized at its
            # insertion point, so counting them would measure the builder rather than the anchor.
            return [op.op_type for op in p.functions["main"].operations if op.op_type != "const"]

        prog = self._sdpa()
        matmul_index = real_ops(prog).index("matmul")
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(real_ops(prog)[matmul_index], "loom_fused_attention")

    def test_an_unscaled_block_fuses_with_scale_1(self):
        # Recovered from the graph rather than recomputed as 1/sqrt(head_dim): a model with no scale (or
        # a non-default one) must not silently acquire one that was never traced.
        prog = self._sdpa(scale=None)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = self._fused(prog)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(float(fused[0].scale.val), 1.0, places=6)

    def test_a_matmul_that_is_not_q_at_k_transposed_is_untouched(self):
        # Built by hand rather than via _sdpa: with transpose_y=False the two operands genuinely do not
        # contract, so K has to be pre-transposed for the graph to type-check at all. That IS the shape
        # this guards -- a softmax fed by an ordinary matmul is not an attention block, and `transpose_y`
        # is how the traced pattern spells Q @ K^T.
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, head_dim, seq), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32),
        ])
        def prog(q, k_t, v, mask):
            scores = mb.matmul(x=q, y=k_t, transpose_x=False, transpose_y=False)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertIn("softmax", _ops(prog))

    def test_a_block_without_the_trailing_reshape_is_untouched(self):
        # A partial match must never half-rewrite: what does not match exports and runs as before, just
        # without a cache.
        prog = self._sdpa(trailing=False)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertIn("softmax", _ops(prog))

    def test_a_bare_softmax_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32)])
        def prog(x):
            return mb.softmax(x=x, axis=-1)

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertEqual(_ops(prog), ["softmax"])


class TestFuseLoomAttentionStripsGqaRepeat(unittest.TestCase):
    """KV-CACHE.md 2.3. `op_attention` reads n_head_kv off K's own shape and lets ggml_mul_mat's
    broadcast map query head i to KV head i // ratio -- the same interleaved correspondence repeat_kv()
    materializes -- so attending against the UN-repeated K/V is identical arithmetic on half the cache.
    Correctness never depends on the strip, which is why every guard bails to "leave it alone"."""

    N_HEAD, N_KV, HEAD_DIM, SEQ = 4, 2, 8, 6

    def _prog(self, expand_v=True):
        n_head, n_kv, head_dim, seq = self.N_HEAD, self.N_KV, self.HEAD_DIM, self.SEQ
        ratio = n_head // n_kv
        v_heads = n_kv if expand_v else n_head

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_kv, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, v_heads, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32),
        ])
        def prog(q, k_kv, v_in, mask):
            def expand(x):
                # Exactly what fuse_gqa_repeat_kv leaves behind: reshape -> tile -> reshape.
                r1 = mb.reshape(x=x, shape=[n_kv, 1, seq, head_dim])
                rep = mb.tile(x=r1, reps=[1, ratio, 1, 1])
                return mb.reshape(x=rep, shape=[1, n_head, seq, head_dim])
            k = expand(k_kv)
            v = expand(v_in) if expand_v else v_in
            qs = mb.mul(x=q, y=np.float32(0.35355338))
            scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=True)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        return prog

    def test_k_and_v_come_from_before_the_repeat(self):
        prog = self._prog()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = [op for op in prog.functions["main"].operations
                 if op.op_type == "loom_fused_attention"]
        self.assertEqual(len(fused), 1)
        # The stored heads are the checkpoint's own, not the expanded ones -- half the cache.
        self.assertEqual(fused[0].k.shape[1], self.N_KV)
        self.assertEqual(fused[0].v.shape[1], self.N_KV)
        self.assertEqual(fused[0].q.shape[1], self.N_HEAD)

    def test_k_is_not_stripped_alone_when_v_has_no_expansion(self):
        # The rule that matters most here: stripping one and not the other would leave the cache's K and
        # V widths disagreeing, and nothing downstream would catch it. Fusion still happens; the strip
        # declines, and both stay expanded.
        prog = self._prog(expand_v=False)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = [op for op in prog.functions["main"].operations
                 if op.op_type == "loom_fused_attention"]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].k.shape[1], self.N_HEAD)
        self.assertEqual(fused[0].v.shape[1], self.N_HEAD)


class TestFuseLoomAttentionBypassesTheMaskSlice(unittest.TestCase):
    """KV-CACHE.md 3.2. The traced mask does not reach the attention block directly: transformers slices
    it to the current KV length, which converts to `slice_by_index(mask, begin=[0,0,0,0],
    end_mask=[T,T,T,False])`. That slice is what pins the mask to the TRACE's kv width -- on a decode
    step it would cut the driver's real `[n_tokens, n_kv]` mask back to the prefill width. Removing it is
    also what lets the mask input be declared `["n_kv", "n_tokens"]` at all, since retyping is only sound
    while the input's own consumers are all fused-attention nodes.

    The mask inputs below are deliberately WIDER than the scores they end up added to, because that is
    the real shape of the thing: the slice is the only reason the traced graph type-checks, and the
    bypass is what hands the fused node the whole mask instead."""

    N_HEAD, HEAD_DIM, SEQ = 4, 8, 6

    def _prog(self, mask_shape=None, **slice_kwargs):
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ
        mask_shape = mask_shape or (1, 1, seq, seq + 2)

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=mask_shape, dtype=types.fp32),
        ])
        def prog(q, k, v, mask):
            kwargs = dict(begin=[0, 0, 0, 0], end=[0, 0, 0, seq],
                          begin_mask=[True, True, True, True],
                          end_mask=[True, True, True, False])
            kwargs.update(slice_kwargs)
            sliced = mb.slice_by_index(x=mask, **kwargs)
            qs = mb.mul(x=q, y=np.float32(0.35355338))
            scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=True)
            scores = mb.add(x=scores, y=sliced)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        return prog

    def _fused_mask(self, prog):
        fused = [op for op in prog.functions["main"].operations
                 if op.op_type == "loom_fused_attention"]
        self.assertEqual(len(fused), 1)
        return fused[0].inputs["mask"]

    def test_the_fused_op_reads_the_mask_input_itself(self):
        prog = self._prog()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        mask_var = self._fused_mask(prog)
        self.assertIs(mask_var, prog.functions["main"].inputs["mask"])
        # The whole mask, not the trace-width slice of it.
        self.assertEqual(tuple(mask_var.shape), (1, 1, self.SEQ, self.SEQ + 2))

    def test_the_orphaned_slice_leaves_the_graph(self):
        # The fusion removes only the ops it definitively subsumed and leaves the rest to DCE, which is
        # the pass that knows whether anything else still reads them -- the same division of labour the
        # mask's own slice chain already relied on. After the real pipeline's DCE, the mask input's only
        # consumer is the fused node, which is the property _retype_fused_mask_input then checks.
        prog = self._prog()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)
        PASS_REGISTRY["common::dead_code_elimination"](prog)

        mask_var = prog.functions["main"].inputs["mask"]
        self.assertEqual([c.op_type for c in mask_var.child_ops], ["loom_fused_attention"])

    def test_a_slice_that_also_cuts_an_earlier_axis_is_left_alone(self):
        # Only `mask[..., :kv_len]` is the slice a cached step makes redundant. One that narrows the
        # query axis too is doing something the driver's own mask would not reproduce.
        seq = self.SEQ
        prog = self._prog(mask_shape=(1, 1, seq + 1, seq + 2),
                          end=[0, 0, seq, seq], end_mask=[True, True, False, False])
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused_mask(prog).op.op_type, "slice_by_index")

    def test_a_slice_with_a_nonzero_begin_is_left_alone(self):
        seq = self.SEQ
        prog = self._prog(begin=[0, 0, 0, 2], end=[0, 0, 0, seq + 2],
                          begin_mask=[True, True, True, False],
                          end_mask=[True, True, True, False])
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused_mask(prog).op.op_type, "slice_by_index")

    def test_a_strided_slice_is_left_alone(self):
        seq = self.SEQ
        prog = self._prog(mask_shape=(1, 1, seq, 2 * seq),
                          end=[0, 0, 0, 2 * seq], stride=[1, 1, 1, 2])
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused_mask(prog).op.op_type, "slice_by_index")

    def test_a_mask_that_is_not_sliced_at_all_still_fuses(self):
        # The unsliced shape is what every other test program in this file uses, and it must keep
        # working: the bypass is "walk back through THIS slice if present", not a requirement.
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32),
        ])
        def prog(q, k, v, mask):
            qs = mb.mul(x=q, y=np.float32(0.35355338))
            scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=True)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)
        self.assertIs(self._fused_mask(prog), prog.functions["main"].inputs["mask"])


class TestAttentionFusionIsOptIn(unittest.TestCase):
    """Decision 4, and it is a correctness requirement rather than caution: the pattern is generic SDPA,
    so it matches the non-autoregressive TTS families' self-attention too -- and an ATTENTION node's
    `kv_cache` attr defaults to TRUE, so firing there would hand them persistent state they must never
    have (and would break their byte-identity gates)."""

    def _prog(self):
        return TestFuseLoomAttention()._sdpa()

    def test_the_pipeline_does_not_fuse_by_default(self):
        prog = self._prog()
        apply_loom_mil_passes(prog)
        self.assertEqual([op for op in prog.functions["main"].operations
                          if op.op_type == "loom_fused_attention"], [])

    def test_the_pipeline_fuses_when_asked(self):
        prog = self._prog()
        apply_loom_mil_passes(prog, fuse_attention=True)
        self.assertEqual(len([op for op in prog.functions["main"].operations
                              if op.op_type == "loom_fused_attention"]), 1)


if __name__ == "__main__":
    unittest.main()
