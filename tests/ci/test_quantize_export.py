"""`--quantize`: how it reaches the writer, and which weights it can actually reach.

Two halves, and they fail in opposite directions.

THE WIRING. Quantization is a property of the REQUESTED export, so no recognizer supplies it and no
checkpoint declares it -- it arrives from the caller and has to cross every family. The causal-LM
config carried a `quantize` field for a long time and passed it out of `backend_kwargs()` only under
`Flattened`, so asking the modular profile for Q8_0 was accepted and silently ignored: the export
succeeded and the file was the size it had always been. That is the same rot `test_export_hparams`
guards for `hparams`, so it is routed through `resolved_backend_kwargs()` instead, where forgetting it
is not expressible, and the first test below is what holds that.

THE ELIGIBILITY. A weight can only be packed where ggml can read a packed one: in a mul_mat's FIRST
operand. `ggml_compute_forward_mul_mat` asserts `src1->type == GGML_TYPE_F32` for the operand it
converts, so the second is F32 or nothing. Convolutions were excluded for exactly that reason -- they
lower to im2col + mul_mat and the kernel sat in the second slot -- which is why quantizing a
convolutional model changed nothing at all: conv kernels are 53-92% of the weight bytes in
Kokoro/Matcha/StyleTTS2/Supertonic and 73% of VITS, whose Q8_0 export came out byte-identical to its
F32 one. `primitives_conv.cpp` now puts a NON-F32 kernel in the first operand instead (F32 keeps the
old formulation exactly, so no existing export changes), and the eligibility list follows it.

The second test pins that the list tracks the ops the engine actually reformulated -- convolutions in,
CONV_TRANSPOSE_1D/2D out. Those last two are native ggml ops implementing F16/F32 only, with no
mul_mat to reorder, so declaring them eligible would produce a file ggml cannot compute. That is the
one exclusion that is a correctness claim rather than a size trade-off, so it gets its own test.

"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestQuantizeReachesEveryFamily(unittest.TestCase):
    """Routed through `resolved_backend_kwargs`, so the only way to lose it is to override that.

    Deliberately NOT a walk that calls `resolved_backend_kwargs()` on every family: it calls
    `contract()`, which several families answer only from a real checkpoint (Whisper reads TS_LO off
    the one it traced), so such a walk would raise on a fake path and prove nothing about routing.
    `test_export_hparams` avoids the same rock by checking `backend_kwargs()`. What is actually
    checkable without a checkpoint is the two halves below: that the merge happens, and that nobody
    has stepped around it.
    """

    def test_no_family_overrides_the_merge_point(self):
        """The whole argument for putting it there. `backend_kwargs()` is overridden by eight families
        and each would have to remember `quantize` -- the arrangement that already lost it for the
        modular causal-LM profile. `resolved_backend_kwargs` is inherited by all of them, and this is
        what notices the day one stops inheriting it."""
        from loom_exporter.export_config import LoomExportConfig
        from loom_exporter.registry import default_registry

        base = LoomExportConfig.resolved_backend_kwargs
        for task, entry in sorted(default_registry()._entries.items()):
            for recognizer in entry.recognizers:
                # Constructing is cheap and checkpoint-free -- it is `contract()` that reads one --
                # so the class comes off a real instance rather than off the recognizer, which does
                # not carry it.
                cls = type(recognizer.build_config(Path("<unused>"), "<unused>.gguf"))
                with self.subTest(task=task, model=recognizer.name):
                    self.assertIs(
                        cls.resolved_backend_kwargs, base,
                        f"{cls.__name__} overrides resolved_backend_kwargs, so --quantize (and the "
                        f"contract, and the phoneme table) reach it only if it remembered to merge them",
                    )

    def test_the_merge_carries_it_when_set_and_omits_it_when_not(self):
        """The base behaviour, on a config with no checkpoint to read. Omission is not tidiness: the
        writer falls back to $LOOM_QUANTIZE for an unset value, so writing "" over it would break the
        environment door that predates this flag."""
        from loom_exporter.decomposition import Flattened
        from loom_exporter.export_config import LoomExportConfig

        config = LoomExportConfig(architecture="test", output_path="<unused>.gguf",
                                  decomposition=Flattened())
        self.assertNotIn("quantize", config.resolved_backend_kwargs())
        config.quantize = "Q8_0"
        self.assertEqual(config.resolved_backend_kwargs()["quantize"], "Q8_0")


class TestQuantizeNamesAreValidated(unittest.TestCase):
    def test_a_name_is_case_insensitive(self):
        from loom_exporter.main_export import validate_quantize

        self.assertEqual(validate_quantize("q8_0"), "Q8_0")

    def test_an_unknown_name_lists_what_is_writable(self):
        """Rejected up front rather than as a KeyError inside the writer, which is minutes of tracing
        later -- and the message names the real set, taken from `gguf.quants` rather than hardcoded."""
        from loom_exporter.main_export import validate_quantize

        with self.assertRaises(ValueError) as ctx:
            validate_quantize("Q4_K_M")
        self.assertIn("Q8_0", str(ctx.exception))

    def test_the_choices_are_types_gguf_can_actually_write(self):
        """A name the enum has but `quants.quantize` raises on would turn a typo into a traceback."""
        from gguf import GGMLQuantizationType, quants

        from loom_exporter.main_export import quantize_choices

        for name in quantize_choices():
            with self.subTest(qtype=name):
                quants.quantize(np.zeros((1, 256), dtype=np.float32), GGMLQuantizationType[name])


class TestOnlyFirstOperandsAreEligible(unittest.TestCase):
    """The eligibility half, pinned against a model built to have one of each."""

    def _export(self, quantize, nodes, names):
        from gguf import GGUFReader

        from loom_exporter.exporter import LoomGGUFExporter

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "q.gguf")
            exporter = LoomGGUFExporter(None, output_path=out, architecture="test", quantize=quantize)
            exporter.topologies = {"t": {"nodes": nodes}}
            # DISTINCT payloads per name: the writer content-addresses weights and aliases identical
            # ones, so equal arrays would collapse into one tensor and the assertions below would be
            # about a tensor that is not there.
            exporter.weights = {name: np.full((4, 256), i + 1.0, dtype=np.float32)
                                for i, name in enumerate(names)}
            exporter.write_gguf("-- driver")
            return {t.name: t.tensor_type.name for t in GGUFReader(out).tensors}

    def test_a_conv_kernel_is_eligible_and_a_second_operand_is_not(self):
        """The change this file exists for. `conv_kernel` is a CONV_1D's first input, which the engine
        now places in mul_mat's first operand when it is not F32; `mm_activation` is a second operand
        in both nodes, which ggml requires as F32 whatever else changes."""
        types = self._export("Q8_0", [
            {"op": "MUL_MAT", "inputs": ["mm_weight", "mm_activation"]},
            {"op": "CONV_1D", "inputs": ["conv_kernel", "mm_activation"]},
        ], ("mm_weight", "conv_kernel", "mm_activation"))
        self.assertEqual(types["mm_weight"], "Q8_0")
        self.assertEqual(types["conv_kernel"], "Q8_0")
        self.assertEqual(types["mm_activation"], "F32", "second operands are never eligible")

    def test_a_transposed_convolution_kernel_stays_f32(self):
        """A correctness claim, not a size trade-off. ggml_conv_transpose_1d/2d are native ops that
        dispatch on the kernel's own dtype and implement F16/F32 only -- there is no mul_mat in them to
        reorder -- so a quantized kernel there would be a file the engine cannot compute."""
        types = self._export("Q8_0", [
            {"op": "CONV_TRANSPOSE_1D", "inputs": ["deconv_kernel", "act"]},
            {"op": "CONV_TRANSPOSE_2D", "inputs": ["deconv_kernel_2d", "act"]},
        ], ("deconv_kernel", "deconv_kernel_2d", "act"))
        self.assertEqual(types["deconv_kernel"], "F32")
        self.assertEqual(types["deconv_kernel_2d"], "F32")

    def test_the_eligible_ops_are_the_ones_the_engine_reformulated(self):
        """Keeps the list and the engine from drifting apart. Every name here is an op whose first
        input `primitives_conv.cpp` (or op_mul_mat) puts in ggml's first operand; adding one the engine
        has not reformulated writes a weight ggml will refuse at compute time, which is a crash rather
        than a bad number."""
        from loom_exporter.exporter import LoomGGUFExporter

        self.assertEqual(
            set(LoomGGUFExporter.PACKED_WEIGHT_FIRST_OPS),
            {"MUL_MAT", "CONV_1D", "CONV_2D", "CONV_1D_DW", "CONV_2D_DW", "SHORT_CONV"},
        )

    def test_requesting_none_leaves_everything_f32(self):
        types = self._export(None, [{"op": "MUL_MAT", "inputs": ["mm_weight", "mm_activation"]}],
                             ("mm_weight", "mm_activation"))
        self.assertEqual(set(types.values()), {"F32"})


class TestConvKernelsAreFoldedSoTheirBlocksAlign(unittest.TestCase):
    """The SHAPE half (P4.13), which is what makes the op half above actually pay.

    Clearing the op gate on its own changed nothing measurable: a conv kernel is stored `[K, IC, OC]`,
    ggml lays quantization blocks along ne[0], and ne[0] is the KERNEL WIDTH -- 1, 3, 5 -- which no
    block size divides. Every conv kernel in VITS cleared the op gate and every one of them was then
    declined for shape, which is why its Q8_0 export came out byte-identical to its F32 one.

    `_fold_conv_kernels_for_quantization` folds the spatial axes into ne[0], so ne[0] becomes `IC*K` --
    a channel count, and therefore a multiple of 32 essentially always. The bytes do not move (a C-order
    `(OC, IC, K)` reshaped to `(OC, IC*K)` is the same buffer, and it is the reshape `op_conv_1d`
    performed on every call anyway); what moves is the geometry, which the node now carries as attrs.
    """

    def _export(self, quantize, nodes, weights):
        from gguf import GGUFReader

        from loom_exporter.exporter import LoomGGUFExporter

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "q.gguf")
            exporter = LoomGGUFExporter(None, output_path=out, architecture="test", quantize=quantize)
            exporter.topologies = {"t": {"nodes": nodes}}
            exporter.weights = dict(weights)
            exporter.write_gguf("-- driver")
            reader = GGUFReader(out)
            # GGUFReader reports shapes in ggml ne order, which is numpy's reversed -- so a folded
            # `(OC, IC*K)` array reads back as `[IC*K, OC]`, exactly what the engine sees.
            tensors = {t.name: (t.tensor_type.name, [int(d) for d in t.shape]) for t in reader.tensors}
            field = reader.fields["model.graph_topology.t"]
            topo = json.loads(str(bytes(field.parts[field.data[0]]), "utf-8"))
            return tensors, {n["inputs"][0]: n.get("attrs", {}) for n in topo["nodes"]}

    @staticmethod
    def _distinct(shape, seed):
        """Distinct payloads per weight: the writer content-addresses and aliases identical ones, so
        equal arrays would collapse into one tensor and the assertions would be about a tensor that is
        not there."""
        return (np.arange(int(np.prod(shape)), dtype=np.float32) + seed).reshape(shape)

    def test_a_dense_conv_kernel_is_folded_and_carries_its_geometry(self):
        """The item itself. `(OC, IC, K)` -> `(OC, IC*K)`, `(OC, IC, KH, KW)` -> `(OC, IC*KH*KW)`, and
        the geometry the fold erased comes back on the node. OC is deliberately NOT among the attrs --
        it survives as ne[1], and a second spelling of it is a second thing to keep in step."""
        tensors, attrs = self._export("Q4_0", [
            {"op": "CONV_1D", "inputs": ["k1", "act"], "attrs": {"s0": 1, "p0": 1, "d0": 1}},
            {"op": "CONV_2D", "inputs": ["k2", "act"], "attrs": {"s0": 1, "s1": 1}},
        ], {"k1": self._distinct((8, 32, 3), 1.0),
            "k2": self._distinct((8, 32, 3, 3), 2.0),
            "act": self._distinct((4, 256), 3.0)})

        self.assertEqual(tensors["k1"], ("Q4_0", [96, 8]))
        self.assertEqual(attrs["k1"], {"s0": 1, "p0": 1, "d0": 1, "kernel_ic": 32, "kernel_k": 3})
        self.assertEqual(tensors["k2"], ("Q4_0", [288, 8]))
        self.assertEqual(attrs["k2"],
                         {"s0": 1, "s1": 1, "kernel_ic": 32, "kernel_kh": 3, "kernel_kw": 3})

    def test_the_fold_is_a_reinterpretation_and_not_a_permutation(self):
        """What makes the fold safe to do to a stored tensor rather than to a view of one. Exported at
        F16 so the bytes survive the round trip exactly (block size 1 means no fold, so this is the
        DECLARED layout) and compared against the flattened array element for element."""
        from gguf import GGUFReader

        from loom_exporter.exporter import LoomGGUFExporter

        kernel = self._distinct((8, 32, 3), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "f.gguf")
            exporter = LoomGGUFExporter(None, output_path=out, architecture="test", quantize=None)
            exporter.topologies = {"t": {"nodes": [{"op": "CONV_1D", "inputs": ["k1", "act"]}]}}
            exporter.weights = {"k1": kernel, "act": self._distinct((4, 8), 9.0)}
            exporter.write_gguf("-- driver")
            stored = {t.name: t for t in GGUFReader(out).tensors}["k1"]
        self.assertEqual([int(d) for d in stored.shape], [3, 32, 8], "no --quantize, so no fold")
        np.testing.assert_array_equal(np.array(stored.data).reshape(-1), kernel.reshape(-1))
        # And the folded shape is that same flat sequence read two axes at a time instead of three.
        np.testing.assert_array_equal(kernel.reshape(8, 96).reshape(-1), kernel.reshape(-1))

    def test_f16_does_not_fold_because_it_has_nothing_to_gain(self):
        """Block size 1 aligns everything already, and the declared 3-D shape is what
        `ggml_conv_2d_direct` -- the direct-conv lowering, worth 4.7x on long-activation convolutions --
        reads its geometry from. Folding here would cost that for nothing."""
        tensors, attrs = self._export("F16", [
            {"op": "CONV_1D", "inputs": ["k1", "act"], "attrs": {"s0": 1}},
        ], {"k1": self._distinct((8, 32, 3), 1.0), "act": self._distinct((4, 256), 3.0)})
        self.assertEqual(tensors["k1"], ("F16", [3, 32, 8]))
        self.assertNotIn("kernel_k", attrs["k1"])

    def test_a_depthwise_kernel_is_never_folded(self):
        """Not a size trade-off, a correctness one. CONV_1D_DW/CONV_2D_DW/SHORT_CONV lower to a mul_mat
        BATCHED over ne[2] -- each channel's patch matrix against that same channel's kernel slice --
        so folding ne[0..1] would turn a per-channel pairing into a cross product, which is a wrong
        answer rather than a slower one. It would not even pay: a depthwise kernel is `[K, 1, C]`, so
        the fold puts K alone on ne[0] and K is 3."""
        tensors, attrs = self._export("Q8_0", [
            {"op": "CONV_1D_DW", "inputs": ["dw", "act"], "attrs": {"s0": 1}},
        ], {"dw": self._distinct((64, 1, 3), 1.0), "act": self._distinct((4, 256), 3.0)})
        self.assertEqual(tensors["dw"], ("F32", [3, 1, 64]))
        self.assertNotIn("kernel_k", attrs["dw"])

    def test_a_kernel_another_node_also_reads_is_left_alone(self):
        """There is no shape that satisfies both consumers, so the fold declines rather than
        reconciles -- and the weight stays F32 rather than being written in a layout the other node
        would read wrongly."""
        tensors, attrs = self._export("Q8_0", [
            {"op": "CONV_1D", "inputs": ["shared", "act"], "attrs": {"s0": 1}},
            {"op": "MUL_MAT", "inputs": ["shared", "act"]},
        ], {"shared": self._distinct((8, 32, 3), 1.0), "act": self._distinct((4, 256), 3.0)})
        self.assertEqual(tensors["shared"], ("F32", [3, 32, 8]))
        self.assertNotIn("kernel_k", attrs["shared"])

    def test_a_kernel_still_unaligned_after_folding_keeps_its_declared_shape(self):
        """`IC*K` is a multiple of 32 essentially always, but not on VITS's three `[1, 1, 192]`
        duration-predictor pre-nets. Folding those would cost the geometry and gain nothing, so they
        stay 3-D and stay F32 -- and stay counted in the export's own declined-for-shape line."""
        tensors, attrs = self._export("Q8_0", [
            {"op": "CONV_1D", "inputs": ["tiny", "act"], "attrs": {"s0": 1}},
        ], {"tiny": self._distinct((192, 1, 1), 1.0), "act": self._distinct((4, 256), 3.0)})
        self.assertEqual(tensors["tiny"], ("F32", [1, 1, 192]))
        self.assertNotIn("kernel_k", attrs["tiny"])
