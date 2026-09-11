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
* **Codes arrive frame-major, `[1, n_frames, codes_per_frame]`, and each codec adapts that to its own
  layout.** DAC's own layout is `[1, n_codebooks, n_frames]` -- one transpose -- and declaring THAT as
  the contract breaks the driver:
  `apply_monolithic_export` derives `n_tokens` as `Len(first_input) / shape[2]`, so a trailing axis
  that is the DYNAMIC one leaves the divisor at 1 and the driver counts `n_codebooks * n_frames`
  frames. Frame-major is also the better caller contract -- an AR LM emits all N codes for frame *t*
  together, so that is the order a flat array arrives in anyway.
* **The delay pattern is not here.** An AR LM emits codebook *k* offset by *k* steps (MusicGen's
  convention, inherited by Parler and Dia); undoing it is index arithmetic over a small array and it is
  a property of the LM, not of the codec -- DAC knows nothing about it. It belongs to family 10's
  driver, and putting it in this contract would make every codec carry a fact only some of its callers
  have. See [ADR-020].

**The third leaf is what the layout claim was waiting for, and the claim holds.** SNAC's
`vq_strides = [4, 2, 1]` puts its three codebooks at three DIFFERENT frame rates -- codebook 0 emits
one code where codebook 2 emits four -- which is the one thing "codes in, frame-major" had never been
asked. It survives, with the row read as the COARSEST codebook's frame: a row is
`sum(coarse // stride)` codes wide -- 7 for SNAC, and exactly `n_codebooks` for a uniform codec, which
is why ONE formula covers both -- laid out level-major, and the wrapper slices the row back into one
tensor per codebook. It is also the layout the caller already has: an AR LM over SNAC emits those same
7 ids per step. The alternative -- one input per codebook -- is expressible (`declared_axes` would
carry the 2x and 4x, as Kokoro's vocoder phase does) and worse: three arrays for a caller to keep in
step, for a codec whose own `decode` takes a list only because Python has lists.

**SNAC's decode is STOCHASTIC, and the noise is an INPUT.** `NoiseBlock` computes
`x + randn(B, 1, T) * linear(x)` at four points in the decoder, so the reference model returns a
different waveform for the same codes on every call. A trace BAKES that `randn` as a constant at the
traced length, and no node in a topology can draw a fresh one either, because a topology is a pure
graph `GraphBuilder` builds once and reuses (`topology_ops._op_random` says exactly this). So the
noise is hoisted: four graph inputs at lengths that are exact multiples of the root axis
(32/256/1024/2048 times `n_codes`, one per decoder upsampling stage), drawn per call by the DRIVER
through the same `loom.seed_rng`/`loom.gaussian_array` host RNG every stochastic model here already
uses. The caller's contract is untouched -- still one `codes` array in -- and the caller may hand the
noise in instead, which is what keeps this family's oracle exact rather than distributional.

**It shipped without the noise first, and that was wrong.** Dropping the term leaves exactly
`E[output]`, 2.4% away in relative RMS where the reference's own seed-to-seed spread is 3.1%, with the
ASR oracle reading 22/22 either way and every spectral difference more than 20 dB down. A listener
called the mean "less sharp, slightly more artificial" on the first hearing. ADR-029 has both halves
and Retro-043 the lesson.

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


SNAC_MISSING = (
    "SNAC is not a transformers architecture -- it ships as its own MIT-licensed package, which this "
    "family imports lazily so that the other leaves need no extra dependency. Install it with "
    "`pip install --no-deps snac` (its unpinned `huggingface-hub` requirement will otherwise drag the "
    "export venv to 1.x, which transformers 4.x refuses to import against)."
)


# The noise tensors the wrapper is currently handing the decoder, one per `NoiseBlock`, in decoder
# order. A module-level handoff because `NoiseBlock.forward` is patched on the CLASS and the tensors
# arrive at the wrapper: the alternative is threading a fifth argument through four `nn.Sequential`s
# that were not written to carry one. Only ever read during the wrapper's own forward.
_SNAC_NOISE: list = []


def _patch_snac(layers) -> None:
    """The two class-level rewrites SNAC's decode needs to trace. Applied at `load`, not at import,
    because the package is optional -- see `SNAC_MISSING`.

    **`Snake1d`, because `snake` is `@torch.jit.script`.** Its body reshapes through
    `x.reshape(shape[0], shape[1], -1)`, and a scripted function's `x.shape[i]` survives inlining as a
    real `__getitem__` on a shape the converter only knows symbolically: `AssertionError: Item
    selection is supported only on python list/tuple objects`, at the first of the decoder's 29
    snakes. The reshape is a NO-OP for the rank-3 input every one of them gets -- `alpha` is
    `[1, C, 1]` and broadcasts against `[B, C, T]` directly -- so the patch is the same arithmetic
    with the two reshapes removed, not an approximation of it.

    **`NoiseBlock`, because `torch.randn` is not traceable and the noise is not disposable.** The draw
    becomes a CONSTANT at the traced length, so it cannot stay; this reads the tensor the wrapper was
    handed instead, which makes the noise a graph INPUT and leaves the arithmetic untouched. The
    driver draws it per call -- see `noise_multiples`.

    **This is a class-level patch, so a `snac` model loaded in this process decodes through it too.**
    Deliberate, and what makes an oracle possible at all: hand the reference the same noise and the
    comparison is exact rather than distributional.
    """
    import torch as _torch

    layers.Snake1d.forward = (
        lambda self, x: x + (self.alpha + 1e-9).reciprocal() * _torch.sin(self.alpha * x).pow(2)
    )
    layers.NoiseBlock.forward = (
        lambda self, x: x + _SNAC_NOISE[self._loom_noise_index] * self.linear(x)
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

    THE THIRD LEAF MOVED ONE THING ACROSS THE LINE. `decode` used to take the model's own
    `[1, n_codebooks, n_frames]` and the wrapper owned the transpose, which read as "the adaptation is
    one op, shared". SNAC's is a slice per codebook, so the adaptation is per-codec after all: `decode`
    now takes the CALLER's frame-major matrix and each member states how its own model wants it. The
    wrapper keeps only what is genuinely common -- the flattening of the returned waveform.
    """

    DAC = "dac"
    ENCODEC = "encodec"
    SNAC = "snac"

    def load(self, model_dir: str):
        if self is CodecFamily.ENCODEC:
            raise NotImplementedError(ENCODEC_BLOCKERS)
        if self is CodecFamily.SNAC:
            try:
                from snac import SNAC
                from snac import layers as snac_layers
            except ImportError as exc:                       # pragma: no cover - env-dependent
                raise ImportError(SNAC_MISSING) from exc
            _patch_snac(snac_layers)
            # `from_pretrained` takes a local directory (it branches on `os.path.isdir`) and reads the
            # `config.json` + `pytorch_model.bin` pair this checkpoint ships. There is no safetensors
            # variant on the Hub, and no `dtype` argument: the package builds at F32 and stays there.
            model = SNAC.from_pretrained(model_dir).eval()
            for index, block in enumerate(
                    m for m in model.decoder.modules() if isinstance(m, snac_layers.NoiseBlock)):
                block._loom_noise_index = index
            return model

        import transformers

        cls = {CodecFamily.DAC: "DacModel", CodecFamily.ENCODEC: "EncodecModel"}[self]
        return getattr(transformers, cls).from_pretrained(model_dir, dtype=torch.float32).eval()

    def decode(self, model, codes):
        """`codes` is the CALLER's `[1, n_frames, codes_per_frame]`; returns the waveform with its
        batch axis still on.

        EnCodec's public `decode` is CHUNKED -- its first axis is a chunk index, and its Python loop
        over chunks unrolls to one iteration at trace time, which is what a caller that hands over a
        whole clip wants. `audio_scales=[None]` is the un-normalised path; `config.normalize` is False
        for every checkpoint this targets, and a normalised one would need the scale as a second input
        rather than a constant, which is a different contract.
        """
        if self is CodecFamily.SNAC:
            # A row is one COARSEST-codebook frame, level-major: codebook 0's single id, then
            # codebook 1's `coarse // stride` ids for that span, and so on. The reshape is what turns
            # the level's columns back into consecutive frames -- reading a `[1, n_frames, k]` slice
            # row-major gives `f0s0 f0s1 f1s0 f1s1 ...`, which is exactly that level's own sequence.
            # `-1` rather than an arithmetic expression so the frame axis stays symbolic.
            strides, out, column = model.vq_strides, [], 0
            for stride in strides:
                width = strides[0] // stride
                out.append(codes[:, :, column:column + width].reshape(1, -1))
                column += width
            # `from_codes` then does the repeat_interleave back up to the finest rate itself. Left to
            # the model rather than lifted into this slicing: it is the model's arithmetic, and it
            # converts (MIL `tile`, one per non-unit stride) without help.
            return model.decode(out)
        codes = codes.transpose(1, 2)                        # -> [1, n_codebooks, n_frames]
        if self is CodecFamily.DAC:
            return model.decode(audio_codes=codes).audio_values
        if not getattr(model.config, "normalize", False):
            return model.decode(audio_codes=codes[None], audio_scales=[None]).audio_values
        raise NotImplementedError(
            f"{model.config.model_type} declares normalize=True, so its decode needs the per-clip "
            f"scale its encoder produced. That is a second input and a different contract; this "
            f"family exports the un-normalised path only."
        )

    def noise_multiples(self, model) -> list:
        """How many noise samples each stochastic leaf needs per unit of the ROOT AXIS, in decoder
        order -- `[32, 256, 1024, 2048]` for SNAC, and empty for a deterministic codec.

        Every one is a fixed ratio because a `NoiseBlock` sits immediately after its stage's
        transposed convolution, so its length is the coarse frame count times that stage's cumulative
        upsampling: `coarse * prod(decoder_rates[:i+1])`. That is the form `declared_axes` takes and
        the form the driver multiplies `n_codes` by, which is why it is computed once, here, rather
        than twice in two spellings.

        Read off the real modules rather than off `config.noise`: the config says whether the blocks
        were BUILT, and what this needs is which stages actually have one.
        """
        if self is not CodecFamily.SNAC:
            return []                                    # DAC and EnCodec decode deterministically
        from snac.layers import DecoderBlock, NoiseBlock

        multiples, cumulative = [], max(model.vq_strides)
        rates = iter(model.decoder_rates)
        for module in model.decoder.model:
            if not isinstance(module, DecoderBlock):
                continue
            cumulative *= int(next(rates))
            if any(isinstance(inner, NoiseBlock) for inner in module.block):
                multiples.append(cumulative)
        return multiples

    def geometry(self, model) -> dict:
        """`{n_codebooks, codebook_size, sample_rate, hop_length, vq_strides}`, read off the
        checkpoint.

        No two of these codecs spell any of it the same way except `codebook_size`, which is why this
        is a method rather than four attribute reads in `load_model` -- and SNAC does not even keep
        them in the same PLACE, having no `config` object at all: the package builds the geometry onto
        the module in `__init__`. So this takes the loaded model, not a config.

        `vq_strides` is the per-codebook downsampling factor, and it is what makes SNAC's rows wider
        than its codebook count. A uniform codec reports `[1] * n_codebooks`, which is not a special
        case anywhere downstream: every derived quantity falls out of the same formula.
        """
        if self is CodecFamily.SNAC:
            return dict(n_codebooks=len(model.vq_strides), codebook_size=int(model.codebook_size),
                        sample_rate=int(model.sampling_rate), hop_length=int(model.hop_length),
                        vq_strides=[int(s) for s in model.vq_strides])
        config = model.config
        if self is CodecFamily.DAC:
            n_codebooks = int(config.n_codebooks)
        else:
            # EnCodec's `num_quantizers` is the count for the bandwidth the checkpoint was configured
            # at, which for the 32 kHz model is the 4 MusicGen emits. A checkpoint offering several
            # bandwidths would need the caller to name one -- it is a property of the EXPORT, not of
            # the file -- and that is why this reads the config rather than a maximum.
            n_codebooks = int(config.num_quantizers)
        return dict(n_codebooks=n_codebooks, codebook_size=int(config.codebook_size),
                    sample_rate=int(config.sampling_rate), hop_length=int(config.hop_length),
                    vq_strides=[1] * n_codebooks)


class _CodecDecodeWrapper(torch.nn.Module):
    """Reduces a codec to `(codes) -> waveform`, taking codes frame-major.

    Down to one op now that the per-codec half of the adaptation lives on `CodecFamily.decode`; see
    that method for why the caller's layout is not any of these models' own.
    """

    def __init__(self, model, family: "CodecFamily"):
        super().__init__()
        self.model = model
        self.family = family

    def forward(self, codes, *noise):
        # The handoff `NoiseBlock.forward` reads: see `_patch_snac`. Assigned rather than appended so
        # a second forward through the same wrapper cannot see the first one's tensors.
        global _SNAC_NOISE
        _SNAC_NOISE = list(noise)
        waveform = self.family.decode(self.model, codes)
        # EnCodec and SNAC return [batch, channels, samples] where DAC returns [batch, samples]; one
        # reshape rather than two shapes reaching the topology, so the driver and the contract stay
        # identical across the family.
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
    _vq_strides: Optional[list] = None
    _noise_multiples: Optional[list] = None

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
            "READ off the checkpoint's own config during load_model, not declared -- it is how many "
            "codebooks the quantizer has, and the checkpoint is the only authority on it."
        ),
        "_codebook_size": Unchecked("same: the checkpoint's own config"),
        "_sample_rate": Unchecked("same"),
        "_hop_length": Unchecked(
            "same. The frame rate the contract declares is `sample_rate / hop_length / coarsest "
            "stride`, derived rather than declared, because a codec states the parts and never their "
            "quotient."
        ),
        "_vq_strides": Unchecked(
            "same: the per-codebook downsampling factors, `[1] * n_codebooks` for a uniform codec. "
            "`codes_per_frame` and `frame_rate` are both derived from it, so it is the one field here "
            "a wrong value would corrupt silently -- which is why it is read rather than declared."
        ),
        "_noise_multiples": Unchecked(
            "walked off the real decoder's own NoiseBlocks during load_model. Empty for a codec that "
            "decodes deterministically, which is every other member. A wrong ratio here is NOT silent: "
            "the driver would hand the graph an array of the wrong length and the call would fail on "
            "the shape, which is why one function computes it for both the axis declaration and the "
            "driver."
        ),
    }

    @property
    def _noise_inputs(self) -> dict:
        """`{input name: samples per root-axis unit}` -- the one table both halves read.

        The export declares these ratios to coremltools as `declared_axes` and to the driver as
        `noise_inputs`, and the two have to agree exactly: one says what shape the graph accepts, the
        other how long an array the driver draws.
        """
        return {f"noise_{i}": multiple for i, multiple in enumerate(self._noise_multiples or [])}

    @property
    def _coarse_stride(self) -> int:
        """The coarsest codebook's stride, which is what one ROW of the caller's matrix spans.

        `vq_strides` is descending in every checkpoint that has one, but `max` rather than `[0]`: the
        row's span is a property of the set, and nothing here depends on the order.
        """
        return max(self._vq_strides)

    @property
    def _codes_per_frame(self) -> int:
        """The width of the caller's matrix: how many ids belong to one coarsest-codebook frame.

        `sum(coarse // stride)`, which is `n_codebooks` exactly when every stride is 1 -- so a uniform
        codec is not a branch here, it is the same formula with a factor of one.
        """
        return sum(self._coarse_stride // stride for stride in self._vq_strides)

    def load_model(self):
        print(f"Loading {self.family.value} codec from {self.model_dir}...")
        model = self.family.load(self.model_dir)
        geometry = self.family.geometry(model)
        config = getattr(model, "config", None)
        self._resolved_architecture = self.architecture or getattr(config, "model_type", None)
        self._n_codebooks = geometry["n_codebooks"]
        self._codebook_size = geometry["codebook_size"]
        self._sample_rate = geometry["sample_rate"]
        self._hop_length = geometry["hop_length"]
        self._vq_strides = geometry["vq_strides"]
        self._noise_multiples = self.family.noise_multiples(model)
        if self._noise_multiples:
            print(f"  {len(self._noise_multiples)} stochastic leaves, at "
                  f"{self._noise_multiples} samples per frame -- drawn by the driver")
        return model

    def export_architecture(self) -> str:
        return self._resolved_architecture or self.architecture

    def build_trace(self, model):
        """`Flattened`'s hook. One input, one symbolic axis.

        The code axis is declared as the checkpoint's own width rather than as a range: it is a
        property of the model, and a graph that accepted a different width would be accepting codes
        from a different codec.
        """
        width = self._codes_per_frame
        print(f"Tracing the codec decoder (dummy n_frames={self.n_frames}, {width} codes/frame)...")
        dummy = [torch.zeros((1, self.n_frames, width), dtype=torch.long)]
        frames = ct.RangeDim(1, self.max_frames)
        mil_inputs = [
            ct.TensorType(name="codes", shape=(1, frames, width), dtype=np.int32),
        ]
        # One `ct.RangeDim` INSTANCE PER NOISE INPUT, deliberately not shared: coremltools gives two
        # inputs that share an instance the same symbol, which is right for lengths that are equal and
        # wrong for lengths that are four different multiples of one root. Each gets its own symbol
        # here and `declared_axes` (via `backend_kwargs`) says what each one is in terms of `n_codes`.
        for name, multiple in self._noise_inputs.items():
            dummy.append(torch.zeros((1, 1, self.n_frames * multiple), dtype=torch.float32))
            mil_inputs.append(ct.TensorType(
                name=name, shape=(1, 1, ct.RangeDim(1, self.max_frames * multiple)),
                dtype=np.float32))
        return _CodecDecodeWrapper(model, self.family), tuple(dummy), mil_inputs

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

        **`codec.n_codebooks` is CODE STREAMS PER FRAME, not the quantizer count**, and the two part
        company for the first time here: SNAC's three codebooks put 7 ids in a row. The key keeps the
        meaning it was given and documented with -- it is the pairing check between a family-10 LM and
        its codec (`loom-py`'s `tests/gate/test_codec_pair.py` asserts the two files agree on it, and
        uses it as the row width), and an LM over SNAC emits 7 per step. A key that switched to 3 here
        would break that pair while still reading true.

        **The stride list itself is NOT written, and the reason is the writer's own rule.** `hparams()`
        writes GGUF scalars a host reads with `hparam_u32`/`hparam_f32`; `[4, 2, 1]` is structured, and
        `write_gguf` refuses it by design. Nothing needs it: the four keys above are what a caller
        builds the matrix and interprets the output with, `codec.frame_rate` already carries the row
        rate (11.72 Hz for SNAC, the COARSEST codebook's), and the driver reads none of them. What the
        strides describe is the order of the columns WITHIN a row, which is documentation for whoever
        rearranges an LM's output into it -- the model card's job, not a number's.
        """
        if self._n_codebooks is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        hparams = {
            "codec.n_codebooks": self._codes_per_frame,
            "codec.codebook_size": self._codebook_size,
            "codec.frame_rate": (float(self._sample_rate) / float(self._hop_length)
                                 / float(self._coarse_stride)),
            "sample_rate": self._sample_rate,
        }
        return hparams

    def contract(self) -> dict:
        return super().contract()

    def backend_kwargs(self) -> dict:
        return dict(
            flat_namespace=True,
            root_axis=self.root_axis,
            driver_builder=self.synthesized_builder_key(),
            hparams=self.hparams(),
            # Both readings of `_noise_inputs`: what shape the graph takes, and how long an array the
            # driver draws. Empty dicts for a deterministic codec, which is what DAC has always
            # emitted -- so its GGUF does not move.
            declared_axes={name: {2: f"{multiple}*{self.root_axis}"}
                           for name, multiple in self._noise_inputs.items()},
            noise_inputs=dict(self._noise_inputs),
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


def _is_snac(path: Path) -> bool:
    """A SNAC checkpoint, which declares no `model_type` at all -- so this reads the SHAPE of the
    config instead.

    `config.json` here is the kwargs of `SNAC.__init__` dumped verbatim, because that is exactly what
    `SNAC.from_config` passes back: no `architectures`, no `model_type`, nothing naming the class.
    `vq_strides` is the discriminating key -- no HF codec config has it, and no SNAC config lacks it
    -- and `encoder_rates`/`decoder_rates` are required alongside so that a future package reusing the
    name does not get loaded through the wrong loader. The `model_type` check is what keeps this from
    claiming a checkpoint one of the other recognizers owns: a config with both would be a config for
    a model this leaf cannot drive.
    """
    cfg = _hf_config(path)
    return (cfg is not None and "model_type" not in cfg
            and all(key in cfg for key in ("vq_strides", "encoder_rates", "decoder_rates")))


def _build_dac(path: Path, output_path: str) -> LoomExportConfig:
    return AudioCodecExportConfig(architecture=None, output_path=output_path, model_dir=str(path),
                                  family=CodecFamily.DAC)


def _build_encodec(path: Path, output_path: str) -> LoomExportConfig:
    return AudioCodecExportConfig(architecture=None, output_path=output_path, model_dir=str(path),
                                  family=CodecFamily.ENCODEC)


def _build_snac(path: Path, output_path: str) -> LoomExportConfig:
    # The one leaf that must be TOLD its architecture: `load_model` reads `config.model_type` for it,
    # and a SNAC model has no `config` at all.
    return AudioCodecExportConfig(architecture="snac", output_path=output_path, model_dir=str(path),
                                  family=CodecFamily.SNAC)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="audio-codec",
        config_class=AudioCodecExportConfig,
        recognizers=[
            ModelRecognizer(name="dac", detect=_is_dac, build_config=_build_dac),
            ModelRecognizer(name="encodec", detect=_is_encodec, build_config=_build_encodec),
            ModelRecognizer(name="snac", detect=_is_snac, build_config=_build_snac),
        ],
    ))
