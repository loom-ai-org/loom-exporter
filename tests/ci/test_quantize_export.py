"""`--quantize`: how it reaches the writer, and which weights it can actually reach.

Two halves, and they fail in opposite directions.

THE WIRING. Quantization is a property of the REQUESTED export, so no recognizer supplies it and no
checkpoint declares it -- it arrives from the caller and has to cross every family. The causal-LM
config carried a `quantize` field for a long time and passed it out of `backend_kwargs()` only under
`Flattened`, so asking the modular profile for Q8_0 was accepted and silently ignored: the export
succeeded and the file was the size it had always been. That is the same rot `test_export_hparams`
guards for `hparams`, so it is routed through `resolved_backend_kwargs()` instead, where forgetting it
is not expressible, and the first test below is what holds that.

THE ELIGIBILITY. Only a MUL_MAT's FIRST operand can be quantized, and that is not an exporter policy
that could be relaxed by choosing better weights -- it is ggml's. `ggml_compute_forward_mul_mat`
asserts `src1->type == GGML_TYPE_F32` for the operand it converts, and loom's CONV_1D/CONV_2D
(src/ops/primitives_conv.cpp) build `ggml_mul_mat(im2col, kernel)` with the KERNEL in that second
slot. So a convolutional model's weights are unquantizable where they sit, and the second test pins
that the selection really is first-operands-only rather than asserting the number it happens to
produce today.

Measured consequence, from the shipped files: conv kernels are 56-92% of the weight bytes in Kokoro,
Matcha, StyleTTS2 and Supertonic, and 76% of VITS -- whose MUL_MATs are all activation-by-activation,
leaving it with ZERO eligible weights and a Q8_0 export byte-identical to its F32 one. That is why
the writer reports coverage and warns rather than printing SUCCESS over an unchanged file.
"""
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


class TestOnlyFirstMulMatOperandsAreEligible(unittest.TestCase):
    """The eligibility half, pinned against a model built to have one of each."""

    def _export(self, quantize):
        from gguf import GGUFReader

        from loom_exporter.exporter import LoomGGUFExporter

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "q.gguf")
            exporter = LoomGGUFExporter(None, output_path=out, architecture="test", quantize=quantize)
            # `mm_weight` is a MUL_MAT's first operand; `conv_kernel` is a CONV_1D's, which reaches
            # ggml as mul_mat's SECOND operand and must stay F32; `mm_activation` is a MUL_MAT's
            # second operand, i.e. a weight in the one slot ggml converts from F32.
            exporter.topologies = {"t": {"nodes": [
                {"op": "MUL_MAT", "inputs": ["mm_weight", "mm_activation"]},
                {"op": "CONV_1D", "inputs": ["conv_kernel", "mm_activation"]},
            ]}}
            # DISTINCT payloads per name: the writer content-addresses weights and aliases identical
            # ones, so three copies of `ones` would collapse into one tensor and the assertions below
            # would be about a tensor that is not there.
            exporter.weights = {name: np.full((4, 256), i + 1.0, dtype=np.float32)
                                for i, name in enumerate(("mm_weight", "conv_kernel", "mm_activation"))}
            exporter.write_gguf("-- driver")
            return {t.name: t.tensor_type.name for t in GGUFReader(out).tensors}

    def test_the_mul_mat_weight_is_quantized_and_the_conv_kernel_is_not(self):
        types = self._export("Q8_0")
        self.assertEqual(types["mm_weight"], "Q8_0")
        self.assertEqual(types["conv_kernel"], "F32",
                         "a conv kernel reaches ggml as mul_mat's src1, which it asserts is F32")
        self.assertEqual(types["mm_activation"], "F32", "second operands are never eligible")

    def test_requesting_none_leaves_everything_f32(self):
        self.assertEqual(set(self._export(None).values()), {"F32"})
