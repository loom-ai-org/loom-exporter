"""The neural-audio-codec family (EXPORT-ROADMAP.md's family 11, P5): discrete codes in, a waveform
out.

**It is the connector for family 10, which is why it comes next.** An AR codec-token LM (parler, dia,
csm, orpheus, qwen3-tts -- ~20 models whose LM half `ModularExportSpec` already exports) emits
integers and is silent on its own; this is what makes them audible. Family 11 is ~11 models in its own
right, so it pays twice.

**The DECODE half only.** `encode` is audio-in/codes-out, a different contract with a different
modality pair, and no family-10 model ever calls it -- so exporting it would be weight in every GGUF
for a door nothing opens. A codec that is genuinely wanted both ways is two exports, not one with two
entry points.

Three things about this family are worth stating up front, because each is a place where the obvious
version is subtly wrong:

* **The RVQ loop is a GRAPH fact, not a config fact, and it unrolls.** `DacResidualVectorQuantize.
  from_codes` is a Python `for i in range(n_codebooks)` over codebook lookups and 1x1 projections,
  summed. Under tracing that becomes N `GET_ROWS` + N convolutions + N-1 `ADD`s in the graph, which is
  *correct*: the codebook count is a property of the checkpoint, not of the input, exactly as a token
  classifier's label count is. So there is no hparam the driver reads, no Lua loop, and no engine
  primitive -- the same finding family 12 produced, one family over.
* **Codes arrive frame-major, `[1, n_frames, n_codebooks]`, and are transposed inside the wrapper.**
  The model's own layout is `[1, n_codebooks, n_frames]`, and declaring it that way breaks the driver:
  `apply_monolithic_export` derives `n_tokens` as `Len(first_input) / shape[2]`, so a trailing axis
  that is the DYNAMIC one leaves the divisor at 1 and the driver counts `n_codebooks * n_frames`
  frames. Frame-major is also the better caller contract -- an AR LM emits all N codes for frame *t*
  together, so that is the order a flat array arrives in anyway.
* **The delay pattern is not here.** An AR LM emits codebook *k* offset by *k* steps (MusicGen's
  convention, inherited by Parler and Dia); undoing it is index arithmetic over a small array and it is
  a property of the LM, not of the codec -- DAC knows nothing about it. It belongs to family 10's
  driver, and putting it in this contract would make every codec carry a fact only some of its callers
  have. See [ADR-020].

The modality pair is `audio_codes -> audio`, NOT `token_ids -> audio`: ADR-020 argues that at length,
and the short version is that a file declaring `token_ids` here resolves to `text2speech` and gets
handed a sentence.
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import coremltools as ct

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .spec_protocol import Unchecked


# What stops EnCodec from exporting today, raised at `load` rather than discovered 200 MB into a trace.
#
# THIS TEXT IS THE FINDING, kept here rather than in a doc because here is where the next person meets
# it. Both blockers were confirmed on `facebook/encodec_32khz` (MusicGen's codec) after the DAC leaf
# was already working, and neither is a gap in this exporter:
#
#  1. coremltools' own torch frontend refuses EnCodec's convolution padding once the length is
#     genuinely dynamic -- `NotImplementedError: Dynamic padding for n-dimensional tensors is not
#     supported`. `EncodecConv1d` pads by a LENGTH-DERIVED amount
#     (`_get_extra_padding_for_conv1d`: `ideal_length - length`). It is the same wall Supertonic hit
#     and the reason that model's text axis is static (see `supertonic_export`'s own docstring).
#     **It looks tractable**: for a stride-1 convolution the extra padding works out to exactly 0
#     (`n_frames = L - k + padding_total + 1 = L`, so `ideal_length = L`), and every convolution on
#     the DECODE path is stride 1. Patching it to a constant 0 should be sound, but it has to be
#     proved per stage rather than assumed -- a wrong pad is a silent output shift, not an error.
#
#  2. `EncodecDecoder` contains `LSTM(1024, 1024, num_layers=2)` OVER THE TIME AXIS, which DAC has
#     nothing like -- DAC's decoder is purely convolutional. A flattened trace unrolls it at the
#     traced length and bakes it, so this is a `ScriptedLoop`/`run_recurrent` export rather than the
#     four-line `Flattened` one DAC gets. The machinery exists (StyleTTS2's BiLSTM goes through it,
#     `recurrent.py` + `bilstm_stepper.h`); what is missing is wiring this family to it.
#
# Everything ELSE about EnCodec is already handled: `CodecFamily.decode` knows its chunked
# `(audio_codes, audio_scales)` signature and `[chunks, batch, n_q, frames]` layout, and `geometry`
# knows its config spellings. Those are exercised by nothing today, which is why the recognizer raises
# THIS instead of letting a caller find out inside coremltools -- an unused member is fine, a live
# branch selecting an unusable path is how the next person loses a morning.
ENCODEC_BLOCKERS = (
    "EnCodec cannot be exported yet, and the two reasons are specific rather than a general gap.\n"
    "  1. coremltools refuses its length-derived convolution padding once the frame axis is dynamic "
    "(`Dynamic padding for n-dimensional tensors is not supported`) -- the same limitation that keeps "
    "Supertonic's text axis static. Every decode-path convolution is stride 1, where the extra "
    "padding is provably 0, so patching it to a constant looks sound but must be proved per stage.\n"
    "  2. Its decoder contains a 2-layer LSTM over the time axis, which a flattened trace bakes. That "
    "needs the ScriptedLoop/run_recurrent path StyleTTS2's BiLSTM uses, not this family's Flattened "
    "one.\n"
    "`CodecFamily.decode`/`geometry` already know EnCodec's signature and config spellings, so the "
    "work is the two items above. See loom.cpp docs/epics/epic-03-model-coverage.md."
)


class CodecFamily(Enum):
    """Which codec this is, as the three things that genuinely differ between them.

    THE SECOND LEAF IS WHAT MADE THIS A TYPE. With DAC alone the wrapper was four lines and naming the
    class inline was honest; EnCodec has the same shape underneath -- an RVQ sum feeding a
    transposed-convolution decoder -- and none of the same spelling. Its `decode` takes
    `(audio_codes, audio_scales)` with codes shaped `[chunks, batch, n_q, frames]` rather than DAC's
    `[batch, n_q, frames]`, and its geometry lives under different config names.

    So what is shared is stated by the parts that are NOT here: the frame-major caller layout, the
    dynamic frame axis, the driver, the contract, and the fact that neither needed an engine
    primitive. A third codec adds a member, not a module.
    """

    DAC = "dac"
    ENCODEC = "encodec"

    def load(self, model_dir: str):
        import transformers

        if self is CodecFamily.ENCODEC:
            raise NotImplementedError(ENCODEC_BLOCKERS)
        cls = {CodecFamily.DAC: "DacModel", CodecFamily.ENCODEC: "EncodecModel"}[self]
        return getattr(transformers, cls).from_pretrained(model_dir, dtype=torch.float32).eval()

    def decode(self, model, codes):
        """`codes` is `[1, n_codebooks, n_frames]`; returns the waveform, batch axis included.

        EnCodec's public `decode` is CHUNKED -- its first axis is a chunk index, and its Python loop
        over chunks unrolls to one iteration at trace time, which is what a caller that hands over a
        whole clip wants. `audio_scales=[None]` is the un-normalised path; `config.normalize` is False
        for every checkpoint this targets, and a normalised one would need the scale as a second input
        rather than a constant, which is a different contract.
        """
        if self is CodecFamily.DAC:
            return model.decode(audio_codes=codes).audio_values
        if not getattr(model.config, "normalize", False):
            return model.decode(audio_codes=codes[None], audio_scales=[None]).audio_values
        raise NotImplementedError(
            f"{model.config.model_type} declares normalize=True, so its decode needs the per-clip "
            f"scale its encoder produced. That is a second input and a different contract; this "
            f"family exports the un-normalised path only."
        )

    def geometry(self, config) -> dict:
        """`{n_codebooks, codebook_size, sample_rate, hop_length}`, read off the checkpoint.

        The two codecs spell every one of these differently except `codebook_size`, which is the
        reason this is a method rather than four attribute reads in `load_model`.
        """
        if self is CodecFamily.DAC:
            return dict(n_codebooks=int(config.n_codebooks), codebook_size=int(config.codebook_size),
                        sample_rate=int(config.sampling_rate), hop_length=int(config.hop_length))
        # EnCodec's `num_quantizers` is the count for the bandwidth the checkpoint was configured at,
        # which for the 32 kHz model is the 4 MusicGen emits. A checkpoint offering several bandwidths
        # would need the caller to name one -- it is a property of the EXPORT, not of the file -- and
        # that is why this reads the config rather than a maximum.
        return dict(n_codebooks=int(config.num_quantizers), codebook_size=int(config.codebook_size),
                    sample_rate=int(config.sampling_rate), hop_length=int(config.hop_length))


class _CodecDecodeWrapper(torch.nn.Module):
    """Reduces a codec to `(codes) -> waveform`, taking codes frame-major.

    The transpose is the whole of the caller-facing adaptation and it is one op; see the module
    docstring for why the caller's layout is the transpose of the model's own.
    """

    def __init__(self, model, family: "CodecFamily"):
        super().__init__()
        self.model = model
        self.family = family

    def forward(self, codes):
        # [1, n_frames, n_codebooks] -> [1, n_codebooks, n_frames], which is what `decode` expects.
        waveform = self.family.decode(self.model, codes.transpose(1, 2))
        # EnCodec returns [batch, channels, samples] where DAC returns [batch, samples]; one squeeze
        # rather than two shapes reaching the topology, so the driver and the contract stay identical.
        return waveform.reshape(1, -1)


@dataclass(kw_only=True)
class AudioCodecExportConfig(LoomExportConfig):
    """A neural audio codec's decode half -> Loom GGUF.

    One leaf today (DAC). The fields below are the ones that genuinely vary between codecs; everything
    else -- the trace length, the dynamic bounds, the contract -- is derived from the checkpoint's own
    config, because a codec states its geometry there and a spec that restated it could only disagree.
    """

    architecture: Optional[str] = None
    model_dir: str
    # Which codec this is. A declaration rather than something sniffed inside `load_model`, so the
    # recognizer that matched a directory and the loader that reads it cannot disagree.
    family: CodecFamily = CodecFamily.DAC
    decomposition: Decomposition = field(default_factory=Flattened)
    # EXPORT-ROADMAP.md R1's axis vocabulary: `n_codes` was declared in `axes.py` for this family and
    # has had no user until now -- its docstring says so. A codec's root axis is a count of CODEC
    # FRAMES, which is neither a subword-token count nor a raw sample count, and reusing `n_tokens`
    # for it is exactly the collapse that vocabulary exists to prevent.
    root_axis: str = "n_codes"
    # Frames the trace runs at. The dynamic range is declared separately through `ct.convert`'s own
    # `inputs=`, as in every other family.
    n_frames: int = 16
    max_frames: int = 4096
    # Read off the checkpoint by `load_model`, never declared: see `__unchecked__`.
    _resolved_architecture: Optional[str] = None
    _n_codebooks: Optional[int] = None
    _codebook_size: Optional[int] = None
    _sample_rate: Optional[int] = None
    _hop_length: Optional[int] = None

    __unchecked__ = {
        "family": Unchecked(
            "which codec this is, stamped by the recognizer that matched the directory. There is no "
            "second authority to check it against: the recognizer reads `model_type` off the same "
            "config.json the loader then loads through, so a mismatch is not expressible."
        ),
        "model_dir": Unchecked(
            "path to the HF directory. The recognizer's detect() already read its config.json, and "
            "the loader raises on anything it cannot load."
        ),
        "root_axis": Unchecked(
            "`axes.py`'s own name for this quantity, declared there before any model used it. The "
            "Axis link checks membership in that vocabulary; that a codec's frames are what it counts "
            "is what this module is."
        ),
        "n_frames": Unchecked(
            "the concrete length torch.jit.trace runs at. The dynamic range is declared separately, "
            "so this constrains nothing the checkpoint could disagree with."
        ),
        "max_frames": Unchecked(
            "the ct.RangeDim upper bound. Unlike a learned position table there is no ceiling in the "
            "checkpoint to check it against -- a convolutional decoder is length-agnostic -- so this "
            "is a declaration about the export, not a claim about the model."
        ),
        "_resolved_architecture": Unchecked("load_model()'s output, cached so export_architecture() "
                                            "can read it back. A field only because this is a dataclass"),
        "_n_codebooks": Unchecked(
            "READ off the checkpoint's own config during load_model, not declared -- it is the width "
            "of the matrix a caller passes, and the checkpoint is the only authority on it."
        ),
        "_codebook_size": Unchecked("same: the checkpoint's own config"),
        "_sample_rate": Unchecked("same"),
        "_hop_length": Unchecked(
            "same. The frame rate the contract declares is `sample_rate / hop_length`, derived rather "
            "than declared, because a codec states the two and never their quotient."
        ),
    }

    def load_model(self):
        print(f"Loading {self.family.value} codec from {self.model_dir}...")
        model = self.family.load(self.model_dir)
        geometry = self.family.geometry(model.config)
        self._resolved_architecture = self.architecture or getattr(model.config, "model_type", None)
        self._n_codebooks = geometry["n_codebooks"]
        self._codebook_size = geometry["codebook_size"]
        self._sample_rate = geometry["sample_rate"]
        self._hop_length = geometry["hop_length"]
        return model

    def export_architecture(self) -> str:
        return self._resolved_architecture or self.architecture

    def build_trace(self, model):
        """`Flattened`'s hook. One input, one symbolic axis.

        The codebook axis is declared as the checkpoint's own N rather than as a range: it is a
        property of the model, and a graph that accepted a different width would be accepting codes
        from a different codec.
        """
        print(f"Tracing the codec decoder (dummy n_frames={self.n_frames})...")
        dummy = (torch.zeros((1, self.n_frames, self._n_codebooks), dtype=torch.long),)
        frames = ct.RangeDim(1, self.max_frames)
        mil_inputs = [
            ct.TensorType(name="codes", shape=(1, frames, self._n_codebooks), dtype=np.int32),
        ]
        return _CodecDecodeWrapper(model, self.family), dummy, mil_inputs

    def synthesized_builder_key(self) -> str:
        """The third family to override this, and the reason is the one P4.0.17 recorded: a
        `Flattened` export's orchestration is not implied by its decomposition. Here the output IS the
        answer -- a waveform -- so there is no reduction at all, and `ArgmaxEpilogue` would argmax it.
        """
        return "CodecDecode"

    def hparams(self) -> dict:
        """What a caller cannot build the input, or interpret the output, without.

        All four are the HOST's half of `hparams()`'s own split: `n_codebooks` is the width of the
        matrix a caller passes, `codebook_size` bounds the ids in it, `frame_rate` is how a caller
        sizes a clip, and `sample_rate` is what the returned floats mean. None of them is read by the
        driver, which is handed the codes and needs no geometry to pass them on.
        """
        if self._n_codebooks is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        return {
            "codec.n_codebooks": self._n_codebooks,
            "codec.codebook_size": self._codebook_size,
            "codec.frame_rate": float(self._sample_rate) / float(self._hop_length),
            "sample_rate": self._sample_rate,
        }

    def contract(self) -> dict:
        return super().contract()

    def backend_kwargs(self) -> dict:
        return dict(
            flat_namespace=True,
            root_axis=self.root_axis,
            driver_builder=self.synthesized_builder_key(),
            hparams=self.hparams(),
        )


def _hf_config(path: Path) -> Optional[dict]:
    cfg_path = path / "config.json"
    if not path.is_dir() or not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _is_dac(path: Path) -> bool:
    """An HF directory declaring `model_type == "dac"`.

    Specific rather than generic, unlike family 12's single recognizer, and the difference is real:
    `*ForTokenClassification` is a claim the CHECKPOINT makes about which `AutoModelFor*` class loads
    it, so one check covered every member. There is no `AutoModelForAudioCodec`, the codec classes are
    unrelated (`DacModel`, `EncodecModel`, `MimiModel`, and SNAC's own package), and each has a
    different `decode` signature. A generic recognizer here would claim checkpoints this wrapper
    cannot drive. The second leaf is what shows where the shared half really is.
    """
    cfg = _hf_config(path)
    return cfg is not None and cfg.get("model_type") == "dac"


def _is_encodec(path: Path) -> bool:
    """An HF directory declaring `model_type == "encodec"` -- MusicGen's codec, and the second leaf.

    Registered even though the export raises, deliberately: detection working is what makes the
    failure message reachable. Without this recognizer an EnCodec directory is "no family recognizes
    this checkpoint", which is the wrong answer -- this family recognizes it fine and cannot yet trace
    it, and `ENCODEC_BLOCKERS` says exactly why.
    """
    cfg = _hf_config(path)
    return cfg is not None and cfg.get("model_type") == "encodec"


def _build_dac(path: Path, output_path: str) -> LoomExportConfig:
    return AudioCodecExportConfig(architecture=None, output_path=output_path, model_dir=str(path),
                                  family=CodecFamily.DAC)


def _build_encodec(path: Path, output_path: str) -> LoomExportConfig:
    return AudioCodecExportConfig(architecture=None, output_path=output_path, model_dir=str(path),
                                  family=CodecFamily.ENCODEC)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="audio-codec",
        config_class=AudioCodecExportConfig,
        recognizers=[
            ModelRecognizer(name="dac", detect=_is_dac, build_config=_build_dac),
            ModelRecognizer(name="encodec", detect=_is_encodec, build_config=_build_encodec),
        ],
    ))
