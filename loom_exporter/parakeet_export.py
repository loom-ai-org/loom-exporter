"""Parakeet TDT/RNN-T -- the NeMo leaf of the Transducer family (BACKLOG.md P4.0.17 step 2).

**What this replaces and why it is a different decomposition.** The ASR family's other export
(`nemo_asr_export.ASRNemoEncoderExportConfig`) is `Flattened`: it traces the encoder and nothing else,
because a Transducer's prediction network is an `nn.LSTM` and ggml has no LSTM op. Everything after the
encoder therefore lived in hand-derived topologies written by `tools/convert_nemo/convert_parakeet_tdt.py`
into five separate GGUFs, driven by `src/core/tdt_decoder.cpp`. That left the MIL export of these two
checkpoints *unreachable*: nothing but its own test could run it, since the artifact held only half a
model and carried a driver (the causal-LM one) that raises when called.

**Everything that was general moved out** (P4.2). The four phases, their shapes, the blank-id
derivation, the joint/duration cross-check and the whole component list are
`transducer_export.BaseTransducerExportConfig`; what stays here is what is actually Parakeet's --
`ASRModel.restore_from`, where NeMo keeps the three post-encoder modules, and the two `.nemo`
recognizers. GigaAM v3 is the second leaf and shares the rest verbatim, which is the check on whether
that split was drawn in the right place.

**The live door is NeMo's buffered streaming** (loom.cpp ADR-122), the loop of NeMo's
`speech_to_text_streaming_infer_rnnt.py`: every step runs the one-pass `encoder` phase over a buffer of up
to `left + chunk + right` seconds (the script's defaults, 10 + 2 + 2) and decodes the chunk's rows with
the stateful greedy decode. `transducer_driver/11_buffered_stream.lua` is that loop; no graph is added,
because the script runs a full-attention model's own forward over each buffer. Its ids are the script's,
which are not the one pass's.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .nemo_asr_export import extract_nemo_tokenizer_dir, load_model, prepare_nemo_environment
from .transducer_export import BaseTransducerExportConfig, TransducerParts, stream_decode_fragment

# A stream's default left context, chunk and right context, in seconds: NeMo's streaming script's own
# (`speech_to_text_streaming_infer_rnnt.py`; its docstring's "streaming with 4s latency: 10-2-2").
STREAM_SECONDS = (10.0, 2.0, 2.0)
# Before `phases()` has read them off a checkpoint (a catalogue build, which never traces): 16 kHz, a
# 10 ms hop, 8x subsampling -- every NeMo FastConformer's.
DEFAULT_STREAM_FACTS = {"SAMPLE_RATE": 16000, "FRAME_SAMPLES": 1280, "FEATURES_PER_SEC": 100.0,
                        "SUBSAMPLING": 8}


@dataclass
class ASRParakeetExportConfig(BaseTransducerExportConfig):
    """Parakeet TDT or plain RNN-T: a `.nemo` archive restored through NeMo's own loader."""

    architecture: str = "parakeet_tdt"
    output_path: str = "parakeet.gguf"

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def encoder_phase(self, model):
        phase, n_embd = super().encoder_phase(model)
        self.stream_facts = stream_facts(model, int(model.cfg.preprocessor.sample_rate))
        return phase, n_embd

    # -- the live door (ADR-122) ---------------------------------------------------------------------

    def buffered_stream_constants(self) -> dict:
        """What differs between the leaves that stream this way: the step phase, whether it takes a
        language prompt and the chunked attention mask's sizes, the default sizes, and the name errors
        carry. Parakeet's step is its one-pass encoder."""
        # No prompt: `S_DEFAULT_PROMPT` is bound anyway, so every name the fragment reads is (-1 is no row).
        return {"S_STREAM_NAME": self.architecture, "S_STEP_PHASE": "encoder", "S_STEP_PROMPT": 0,
                "S_DEFAULT_PROMPT": -1, "S_STEP_MASK": 0, "S_DEFAULT_LEFT_SECONDS": STREAM_SECONDS[0],
                "S_DEFAULT_CHUNK_SECONDS": STREAM_SECONDS[1], "S_DEFAULT_RIGHT_SECONDS": STREAM_SECONDS[2]}

    def buffered_step_inputs(self) -> tuple:
        """The inputs `11_buffered_stream.lua` fills on the step phase it names by `S_STEP_PHASE`."""
        return ("waveform", "length")

    def driver_components(self) -> List:
        """The template's one-pass decode, and beside it the live door: the shared live decode
        (`transducer_driver/10_stream_decode.lua`) and NeMo's buffered loop (`11_buffered_stream.lua`),
        top level so their constants are too."""
        from .driver_components import ComputedCall, ExportConstants, LuaFragment
        from .nemo_asr_export import MAX_SECONDS

        facts = dict(getattr(self, "stream_facts", None) or DEFAULT_STREAM_FACTS)
        constants = {"S_" + k: v for k, v in facts.items()}
        constants["S_MAX_BUFFER_SAMPLES"] = int(MAX_SECONDS * facts["SAMPLE_RATE"])
        constants.update(self.stream_decode_constants(blank_default=8192))
        constants.update(self.buffered_stream_constants())
        here = Path(__file__).resolve().parent / "transducer_driver"
        return [
            ExportConstants(values=constants, top_level=True),
            stream_decode_fragment(self.num_pred_layers),
            # The step phase is named by a constant, so the call site is declared, not parsed.
            LuaFragment(here / "11_buffered_stream.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish"),
                        drives=(ComputedCall(topologies=(constants["S_STEP_PHASE"],),
                                             inputs=self.buffered_step_inputs(), written="S_STEP_PHASE"),)),
        ] + super().driver_components()

    def load_model(self):
        return load_model(self)

    def transducer_parts(self, model) -> TransducerParts:
        """NeMo's layout: the prediction network under `decoder.prediction`, the joint beside it.

        `model_defaults.tdt_durations` is the checkpoint's own authority on its duration set -- present
        for TDT, absent entirely for plain RNN-T. Read off the RESTORED model rather than by re-opening
        the archive in `build_config`, so a config can be built (and its components counted by
        `component_registry.usage`) without a checkpoint on disk.
        """
        defaults = getattr(model.cfg, "model_defaults", None) or {}
        pred = model.decoder.prediction
        return TransducerParts(
            embed=pred.embed,
            lstm=pred.dec_rnn.lstm,
            joint=model.joint,
            durations=tuple(defaults.get("tdt_durations") or ()),
            # NeMo states the joint's total width a second time. It is NOT the token count -- see
            # `BaseTransducerExportConfig.phases`, where the two readings are compared.
            declared_joint_width=int(model.joint.num_classes_with_blank),
        )

    def tokenizer_dir(self):
        """A `.nemo` archive keeps its SentencePiece proto under a content-hashed name inside the
        tarball, so this one has to unpack it; `extract_nemo_tokenizer_dir` is that adapter."""
        return extract_nemo_tokenizer_dir(self.checkpoint)


def stream_facts(model, sample_rate: int) -> dict:
    """The numbers the buffered loop's arithmetic is in, as NeMo's streaming script derives them: the
    hop in samples (which it rounds to a multiple of the subsampling, a no-op here, refused otherwise),
    an encoder frame in samples, and the features per second its seconds-to-frames rounding uses."""
    stride = float(model.cfg.preprocessor.window_stride)
    factor = int(model.encoder.subsampling_factor)
    hop = int(sample_rate * stride)
    if hop % factor:
        raise ValueError(f"the hop ({hop} samples) is not a multiple of the subsampling ({factor}); NeMo's "
                         "streaming script would round it and the driver does not")
    return {"SAMPLE_RATE": sample_rate, "FRAME_SAMPLES": hop * factor, "FEATURES_PER_SEC": 1.0 / stride,
            "SUBSAMPLING": factor}
