"""NeMo CTC ASR (Conformer-CTC, Citrinet) as three traced phases: the one pass, and NeMo's buffered stream.

**`ctc` is the one pass, unchanged**: `ASRNemoEncoderExportConfig`'s trace (preprocessor, encoder, CTC head
over the whole waveform, cut to `encoded_len`), decoded by the same greedy CTC components the `Flattened`
export synthesized. It is not called `main_topology` as it was: a phase of that name writes its weights
unprefixed, the `Flattened` convention, and its anonymous constants (`_11`) then share names with the other
phases' intermediate values, which the merge rightly refuses. Until loom.cpp ADR-123 this leaf WAS that `Flattened` export;
it is multi-phase only because the live door needs two more graphs, and a `Flattened` export writes one.

**The live door is NeMo's `FrameBatchASR`** (`speech_to_text_buffered_infer_ctc.py`; `nemo_ctc_driver/
01_stream.lua` is its loop). It does not run the model on audio: it computes the clip's raw log-mel once,
slides it a chunk at a time into a mel buffer, normalises each buffer per feature over itself and runs the
model on that. So the stream needs the halves of the one pass apart:

  * `raw_mel` -- the preprocessor with normalisation off, `(waveform, length) -> [1, T, n_mels]`. NeMo's
    `FrameBatchASR` builds this preprocessor from the config with `normalize = "None"`, dither 0, no pad;
    here it is the model's own featurizer with normalisation shadowed off while the phase traces (its
    filterbank and window are the same buffers).
  * `ctc_buffer` -- `(features [1, T, n_mels], length) -> log-probs [1, T_enc, V]`: `FeatureFrameBufferer.
    normalize_frame_buffers` (per-feature mean and population std over the buffer, `+ 1e-5`), then the
    model's `forward(processed_signal=...)` -- encoder and CTC head. Every frame the tensor has, as NeMo
    decodes them (a whole buffer's frame count is a multiple of the subsampling, so no extra frame).

Both phases read the one pass's weights; nothing is duplicated.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .nemo_asr_export import (
    ASREncoderWrapper, ASRNemoEncoderExportConfig, EncoderOutput, build_trace, extract_nemo_tokenizer_dir,
    load_model, prepare_nemo_environment,
)
from .spec_protocol import Unchecked

# The script's defaults (`chunk_len_in_secs`, `total_buffer_in_secs`).
STREAM_SECONDS = (1.6, 4.0)
# Mel frames a `raw_mel` window reaches past the frames kept, each side. A frame reads n_fft samples about its
# centre (+-256 at 16 kHz) and pre-emphasis one more; 4 frames is 640 samples. Checked at export against the
# checkpoint's own n_fft.
MEL_MARGIN = 4
# The longest buffer `ctc_buffer` is traced for, in mel frames (20 s at a 10 ms hop).
MAX_BUFFER_FRAMES = 2000


class RawMel(nn.Module):
    """`(waveform [1, n], length [1]) -> [1, T, n_mels]`: the preprocessor with normalisation off, as
    `FrameBatchASR.raw_preprocessor` runs it. The featurizer's `normalize` is read by a Python branch, so it is
    shadowed while this module runs and the trace bakes the unnormalised path; the one pass is not touched."""

    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, waveform, length):
        featurizer = self.preprocessor.featurizer
        saved = featurizer.normalize
        featurizer.normalize = None
        try:
            features, _ = self.preprocessor(input_signal=waveform, length=length)
        finally:
            featurizer.normalize = saved
        return features.transpose(1, 2)


class CtcBuffer(nn.Module):
    """`(features [1, T, n_mels], length [1]) -> log-probs [1, T_enc, V]`: one `FrameBatchASR` buffer --
    normalised per feature over itself (`FeatureFrameBufferer.normalize_frame_buffers`: population std, the
    constant added to the std) -- through the model from its processed signal on."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, length):
        x = features.transpose(1, 2)                                       # [1, n_mels, T]
        # Sums over the frames divided by their count, the `length` input: a mean over a run-time count on
        # this axis has no lowering (REDUCE_SUM + SCALE wants the count at export time).
        count = length.to(torch.float32).view(1, 1, 1)
        mean = x.sum(dim=-1, keepdim=True) / count
        std = torch.sqrt(((x - mean) ** 2).sum(dim=-1, keepdim=True) / count)
        normed = (x - mean) / (std + 1e-5)
        log_probs, _, _ = self.model(processed_signal=normed, processed_signal_length=length)
        return log_probs


@dataclass(kw_only=True)
class ASRNemoCtcExportConfig(BaseMultiPhaseModelExportConfig):
    """A NeMo CTC checkpoint (`EncDecCTCModel(BPE)`): the one pass, and `FrameBatchASR`'s two graphs."""

    checkpoint: str
    architecture: str = "conformer_ctc"
    output_path: str = "nemo_ctc.gguf"
    root_axis: str = "n_samples"
    # What the one pass returns: fixed by this leaf, as the transducer leaves fix theirs, and still checked
    # against the real outputs inside the traced wrapper (`EncoderOutput.validate`).
    output: EncoderOutput = EncoderOutput.CTC_LOG_PROBS
    driver_script_path: Path = Path(__file__).resolve().parent / "nemo_ctc_driver"
    # Read off the checkpoint in `phases()`.
    num_classes: Optional[int] = field(default=None, init=False, repr=False)
    stream_facts: Optional[dict] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        **BaseMultiPhaseModelExportConfig.__unchecked__,
        "checkpoint": Unchecked("the .nemo archive, established by the recognizer's detect(); load_model "
                                "raises on anything it cannot restore"),
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "output_path": Unchecked("where to write; a caller's choice"),
        "root_axis": Unchecked("checked by each phase's own Axis link"),
        "output": Unchecked("fixed at CTC_LOG_PROBS by this leaf; EncoderOutput.validate checks it against "
                            "what the model returns, inside the traced wrapper"),
        "num_classes": Unchecked("READ off the checkpoint (`decoder.num_classes_with_blank`) in phases()"),
        "stream_facts": Unchecked("READ off the checkpoint's preprocessor and encoder in phases()"),
    }

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        return load_model(self)

    def _flattened(self) -> ASRNemoEncoderExportConfig:
        """The `Flattened` config this leaf was: its checks and its trace are the one pass's."""
        return ASRNemoEncoderExportConfig(checkpoint=self.checkpoint, output=self.output,
                                          architecture=self.architecture, output_path=self.output_path)

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        flat = self._flattened()
        sample_rate = flat.validate_against_model(model)
        self.num_classes = flat.num_classes
        wrapper, dummy_inputs, mil_inputs = build_trace(flat, model, sample_rate)
        self.stream_facts = _stream_facts(model, sample_rate)
        n_mels = self.stream_facts["N_MEL"]

        rate = sample_rate
        n = rate + 777                                       # neither a whole hop nor a whole chunk
        buffer = 400
        return [
            ExportPhase(name="ctc", wrapper=wrapper, dummy_inputs=dummy_inputs, mil_inputs=mil_inputs,
                        root_axis=self.root_axis),
            ExportPhase(
                name="raw_mel", wrapper=RawMel(model).eval(),
                dummy_inputs=(torch.randn(1, n) * 0.1, torch.tensor([n], dtype=torch.int64)),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, ct.RangeDim(rate // 10, 20 * rate)),
                                          dtype=np.float32),
                            ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
                root_axis="n_samples"),
            ExportPhase(
                name="ctc_buffer", wrapper=CtcBuffer(model).eval(),
                dummy_inputs=(torch.randn(1, buffer, n_mels), torch.tensor([buffer], dtype=torch.int64)),
                mil_inputs=[ct.TensorType(name="features",
                                          shape=(1, ct.RangeDim(16, MAX_BUFFER_FRAMES), n_mels),
                                          dtype=np.float32),
                            ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
                root_axis="n_tokens"),
        ]

    def driver_components(self) -> List:
        """`infer`: the one pass and greedy CTC, the components the `Flattened` export synthesized. Beside it,
        top level, `FrameBatchASR`'s loop (`nemo_ctc_driver/01_stream.lua`) and its constants."""
        from .driver_components import CtcGreedyEpilogue, ExportConstants, LuaFragment, SubgraphCallComponent
        from .driver_ir import FieldAccess, Len, Var

        facts = dict(self.stream_facts or {"SAMPLE_RATE": 16000, "HOP": 160, "N_MEL": 80,
                                           "FEATURE_STRIDE": 0.01, "MODEL_STRIDE": 4, "BUFFER_FILL": 0.0})
        constants = {"S_" + k: v for k, v in facts.items()}
        blank = (self.num_classes - 1) if self.num_classes is not None else 1024
        constants.update({
            "S_STREAM_NAME": self.architecture, "S_BLANK_ID": blank, "S_MEL_MARGIN": MEL_MARGIN,
            "S_MAX_BUFFER_FRAMES": MAX_BUFFER_FRAMES,
            "S_DEFAULT_CHUNK_SECONDS": STREAM_SECONDS[0], "S_DEFAULT_TOTAL_SECONDS": STREAM_SECONDS[1],
        })
        here = self.driver_script_path
        return [
            ExportConstants(values=constants, top_level=True),
            LuaFragment(here / "01_stream.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish")),
            LuaFragment(here / "00_inputs.lua", defines=("_waveform",)),
            SubgraphCallComponent(topology="ctc", outputs=(), retain=True,
                                  inputs={"waveform": Var("_waveform"), "length": FieldAccess("inputs", "length")},
                                  length=Len("_waveform")),
            CtcGreedyEpilogue(retained_module="ctc", blank_id=blank),
        ]

    def backend_kwargs(self) -> dict:
        # Each phase's weights under its own prefix, as every multi-phase export writes them (the writer
        # hashes them, so the stream's phases cost no second copy of the one pass's weights); plus the CTC
        # leaves' two vocabulary kinds.
        kwargs = dict(root_axis=self.root_axis, hparams=self.hparams())
        tokenizer_dir = extract_nemo_tokenizer_dir(self.checkpoint)
        if tokenizer_dir is not None:
            kwargs["tokenizer_dir"] = tokenizer_dir
            kwargs["tokenizer_family"] = ("sentencepiece_proto" if (Path(tokenizer_dir) / "tokenizer.model").is_file()
                                          else "wordpiece")
        return kwargs


def _stream_facts(model, sample_rate: int) -> dict:
    """The numbers `FrameBatchASR`'s arithmetic is in, read where NeMo reads them.

    `MODEL_STRIDE` is the encoder's subsampling: the script's `model_stride` (r2.3: "4 for Conformer, 8 for
    Citrinet"; from r2.4 it reads `asr_model.subsampling_factor`, which no CTC model has). `BUFFER_FILL` is
    `FeatureFrameBufferer`'s own test -- `-16.635` only when the PREPROCESSOR has a true `log`, which NeMo's
    `AudioToMelSpectrogramPreprocessor` does not (its featurizer does), so 0.0 in practice."""
    stride = float(model.cfg.preprocessor.window_stride)
    hop = int(sample_rate * stride)
    n_fft = int(model.preprocessor.featurizer.n_fft)
    if MEL_MARGIN * hop < n_fft // 2 + 1:
        raise ValueError(f"a raw_mel window's margin ({MEL_MARGIN} hops of {hop}) does not cover a frame's "
                         f"reach (n_fft {n_fft} / 2 + 1)")
    preprocessor = model.preprocessor
    fill = -16.635 if (hasattr(preprocessor, "log") and preprocessor.log) else 0.0
    return {"SAMPLE_RATE": sample_rate, "HOP": hop, "N_MEL": int(model.cfg.preprocessor.features),
            "FEATURE_STRIDE": stride, "MODEL_STRIDE": int(model.encoder.subsampling_factor),
            "BUFFER_FILL": fill}
