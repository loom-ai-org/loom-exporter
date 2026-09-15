"""The SANM / FunASR family (`EXPORT-ROADMAP.md`'s family 5, P5): a kaldi-fbank front end, a
low-frame-rate stack, an SANM encoder -- self-attention with a depthwise FSMN memory block beside it --
and, on this first leaf, one linear CTC head.

**The roadmap scoped this family with the same sentence it scoped family 4 with, and the correction
family 4 wrote down is the one that applies here too.** "Family-1-shaped once the encoder template
generalizes past NeMo" was half right there, and it is half right again: the CTC *epilogue* is family
1's, reused unchanged (`CtcGreedyBuilder`, `loom.argmax_rows`, `ctc_blank_id`), and the *encoder
template* is shared with nothing. Family 1's `build_trace` is a NeMo `(input_signal,
input_signal_length)` pair around NeMo's own mel front end; family 4 takes a raw waveform and no
length; this family takes a raw waveform and a second, non-audio input. Three siblings, one epilogue.

What this family costs, and each item is a place the export would otherwise be silently wrong:

* **THE FRONT END IS KALDI'S, NOT `torch.stft`'S, AND IT IS NOT DIFFERENTIABLE-SHAPED CODE.** FunASR's
  `WavFrontend` calls `torchaudio.compliance.kaldi.fbank` and then `apply_lfr`, and both are written
  against Python-level shapes -- `as_strided` with strides computed from `.shape[0]`, and an end
  padding whose LENGTH depends on the frame count modulo the LFR stride. Neither traces. It is
  rebuilt here in ops that do, and the rebuild is exact rather than approximate: see
  `KaldiFbankLfrCmvn`, whose whole design is that kaldi's per-frame work is LINEAR and therefore
  foldable into one convolution kernel.
* **THE REFERENCE FRONT END IS STOCHASTIC BY DEFAULT.** `WavFrontend`'s `dither` defaults to `1.0` --
  kaldi's own default -- so `funasr`'s own pipeline adds Gaussian noise to the waveform before every
  fbank and does not transcribe the same file the same way twice at the bit level. A graph has no
  dither, so the exported model IS the `dither=0` model, and the oracle must disable it on the
  reference side or the comparison grades noise. This is the front-end half of
  [Retro-032](../../loom.cpp/docs/retros/retro-032-one-seed-is-not-an-asr-oracle.md)'s rule arriving
  one family later and one stage earlier.
* **THE MODEL'S FIRST FOUR FRAMES ARE A PROMPT, AND TWO OF THEM ARE KNOBS.** `SenseVoiceSmall.inference`
  prepends four rows of a 16-entry embedding table to the features: a language id, the two fixed
  event/emotion queries, and a text-normalization id. Language selects between auto-detection and six
  named languages; text-normalization selects between raw lowercase output and one with casing,
  punctuation and digits. Baking them would ship a model that can never punctuate, so `prompt_ids` is
  a graph INPUT -- the second one, after the waveform, so the root-axis expression still reads the
  waveform's own length. It is also the family's answer to `HIGH-LEVEL-API.md`'s canonical-name
  question -- and the answer is NOT to reuse Whisper's `asr.language_ids`, which are decoder prompt
  tokens for a mechanism this family does not have. The languages go in `loom.text.languages`, which
  states what the model speaks without claiming how; the name -> row tables go under a `sanm.` prefix;
  and the driver defaults the whole vector, so a caller who wants `transcribe(audio)` never learns the
  input exists. See `contract()`.
* **The attention mask is omitted, deliberately.** Every `mask` argument in FunASR's SANM stack is
  `sequence_mask(ilens)` over a single unpadded sequence, which is all ones -- and all ones is what
  `forward_fsmn`'s two multiplies and `forward_attention`'s `masked_fill` each do nothing with. Passing
  `None` takes the same branch the reference takes for a batch of one and removes the one tensor built
  from a Python-level length. loom.cpp ADR-019, a third modality over.
* **The sinusoidal position encoding is rebuilt from the TENSOR.** FunASR's own
  `SinusoidalPositionEncoder.forward` unpacks `batch_size, timesteps, input_dim = x.size()` and calls
  `torch.arange(1, timesteps + 1)`; `timesteps` is a Python int by the time `arange` sees it. The
  positions are derived from `x.shape[1]` here instead, which is the same number and stays symbolic.
  Note the `+ 1`: these positions are ONE-BASED, and a zero-based range is the silent-and-plausible
  failure [Retro-039](../../loom.cpp/docs/retros/retro-039-position-zero-was-not-row-zero.md) records
  one family over.

And one thing that is free and was not expected to be: **the vocabulary.** The CTC head's 25,055 rows
are exactly the 25,055 pieces of the checkpoint's own SentencePiece BPE protobuf, in id order, and no
`tokenizer.json` sits beside it -- so ADR-027's seam sends this down the same `sentencepiece_proto`
path family 1's NeMo checkpoints take, byte for byte. The blank is id 0 rather than the last class,
which `ctc_blank_id` has been a parameter for since family 4.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .spec_protocol import Axis, Unchecked

# The dummy trace length and the dynamic sample-axis bounds, in SECONDS, resolved against the
# checkpoint's own declared rate. The floor is this family's real one: the fbank needs a full
# `frame_length` window (400 samples at 16 kHz) to produce even one frame, and the LFR stack then needs
# one frame to produce one row, so anything shorter is a graph with nothing in it. 30 s at the top is
# what SenseVoice's own card states it was trained on; FunASR's pipeline segments longer audio with a
# VAD rather than feeding it whole.
TRACE_SECONDS = 1.0
MIN_SECONDS = 0.1
MAX_SECONDS = 30.0

# `torchaudio.compliance.kaldi`'s own floor under the log, transcribed rather than rounded:
# `numeric_limits<float>::epsilon()`. It is what stands between `log` and `-inf` on a silent frame.
LOG_FLOOR = float(np.finfo(np.float32).eps)


def _frame_matrix(window_size: int, padded_size: int, window: torch.Tensor,
                  preemphasis: float, remove_dc_offset: bool) -> torch.Tensor:
    """The `(padded_size, window_size)` linear map kaldi applies to each raw frame, as one matrix.

    **This is the whole reason the front end fits in a graph.** Kaldi does four things to a frame before
    the FFT -- subtract its mean, apply a first-order pre-emphasis filter, multiply by a window, and
    zero-pad to the FFT size -- and every one of them is linear, so their composition is a single
    constant matrix and the framing itself becomes one strided convolution whose kernel that matrix is.
    Built at float64 and cast by the caller, because the hamming window and the `1/n` of the DC term are
    the only places the arithmetic here is not exact.

    The pre-emphasis row for column 0 is `1 - preemphasis`, not `1`: kaldi pads the frame on the left by
    REPLICATION before differencing, so the first sample is differenced against itself.
    """
    n = window_size
    dc = torch.eye(n, dtype=torch.float64)
    if remove_dc_offset:
        dc = dc - torch.full((n, n), 1.0 / n, dtype=torch.float64)
    pre = torch.eye(n, dtype=torch.float64)
    if preemphasis != 0.0:
        pre[0, 0] = 1.0 - preemphasis
        rows = torch.arange(1, n)
        pre[rows, rows - 1] = -preemphasis
    folded = torch.diag(window.double()) @ pre @ dc
    out = torch.zeros(padded_size, n, dtype=torch.float64)
    out[:n, :] = folded
    return out


class KaldiFbankLfrCmvn(nn.Module):
    """`WavFrontend` -- kaldi fbank, low frame rate stacking, and CMVN -- in ops that trace and stay
    dynamic in the sample count.

    Four constants carry the whole of it, and all four are computed from the checkpoint's own frontend
    configuration rather than declared:

    * `frame_w`, the `(n_fft, 1, window_size)` convolution kernel `_frame_matrix` folds. Convolving the
      waveform with it at `stride=hop` IS kaldi's framing, DC removal, pre-emphasis, windowing and
      zero-padding, in one op with one dynamic output axis.
    * `dft_cos` / `dft_sin`, the real DFT written as two matrices. `torch.fft.rfft` would need a
      complex-typed intermediate, which is the same thing that blocks `torch.stft` one family over; the
      power spectrum needs only `|X|^2 = (Cx)^2 + (Sx)^2`, so the complex number never has to exist.
    * `mel_banks`, kaldi's triangular filterbank, taken from `torchaudio` rather than rebuilt so there
      is one authority on it. Kaldi drops the Nyquist bin, which is the trailing zero column.
    * `cmvn_add` / `cmvn_mul`, the checkpoint's `am.mvn` as the affine it already is.

    **The end padding is constant here and data-dependent in the reference, and that is not an
    approximation.** `apply_lfr` pads the frame sequence at the end by an amount that depends on the
    frame count modulo `lfr_n` -- between -2 and 3 rows for `lfr_m=7, lfr_n=6` -- which under tracing
    would bake the trace length's own remainder. Padding by the MAXIMUM instead is exact: with
    `lfr_m - 1 - left_pad` rows of tail, the window count below is already `ceil(n_frames / lfr_n)` and
    the surplus is consumed by the last window rather than read separately. Checked against the
    reference at every length rather than argued.

    **`lfr_stack` is `lfr_m` depthwise convolutions and a concatenation, and the shape of it is the
    point.** The obvious spelling is one strided slice per window position -- `padded[j::lfr_n]`, seven
    of them -- and it is correct in torch and unexportable: MIL does not propagate algebra through a
    strided slice with a symbolic end, it mints a fresh opaque symbol, and what reaches the topology is
    then the root axis or the slice's own `end`. The first export of this family declared
    `6*floor(floor((n_samples - 400)/160)/6) + 10` rows where the truth was 187 of them, and aborted
    inside `ggml_repeat` at the position encoding -- the same failure, one family over, that
    loom.cpp's *a shape-derived slice kills the algebra* rule was written for.

    A convolution has none of that problem: its output length is `floor((P - lfr_m)/lfr_n) + 1`, which
    is arithmetic the exporter already derives exactly (it is how the fbank framing above gets its own
    axis), and it is identically `ceil(n_frames / lfr_n)` for every `n_frames`. So window position `j`
    becomes a DEPTHWISE convolution whose kernel is one-hot at tap `j` -- "take frame `lfr_n*i + j`",
    written as an op whose shapes are inferable -- and concatenating the `lfr_m` of them along the
    feature axis lays the result out window-position-major, mel-bin-minor, which is exactly
    `apply_lfr`'s own order. It costs `lfr_m * n_mels * lfr_m` weights (3,920 here) against the
    `n_mels * lfr_m * n_mels * lfr_m` a single dense convolution would have needed, and it removes the
    `t_lfr` arithmetic from the graph entirely rather than correcting it.
    """

    def __init__(self, cmvn: torch.Tensor, *, n_mels: int = 80, sample_rate: int = 16000,
                 frame_length: float = 25.0, frame_shift: float = 10.0, window_type: str = "hamming",
                 low_freq: float = 20.0, high_freq: float = 0.0, preemphasis: float = 0.97,
                 remove_dc_offset: bool = True, lfr_m: int = 7, lfr_n: int = 6,
                 upscale_samples: bool = True):
        super().__init__()
        import torchaudio.compliance.kaldi as kaldi

        self.hop = int(sample_rate * frame_shift * 0.001)
        self.window_size = int(sample_rate * frame_length * 0.001)
        # kaldi's `round_to_power_of_two`, which is on by default and is what makes the FFT 512 wide for
        # a 400-sample window. Transcribed from `_next_power_of_2` rather than assumed to be 512.
        self.n_fft = 1 if self.window_size == 0 else 2 ** (self.window_size - 1).bit_length()
        self.lfr_m, self.lfr_n = lfr_m, lfr_n
        self.left_pad = (lfr_m - 1) // 2
        self.right_pad = lfr_m - 1 - self.left_pad
        # FunASR's `upsacle_samples` (sic): these checkpoints were trained on int16-scaled audio, so a
        # float waveform in [-1, 1] is 32768x too quiet for them. Silent, like every scale error.
        self.scale = float(1 << 15) if upscale_samples else 1.0

        window = kaldi._feature_window_function(window_type, self.window_size, 0.42,
                                                torch.device("cpu"), torch.float64)
        frame_w = _frame_matrix(self.window_size, self.n_fft, window, preemphasis, remove_dc_offset)
        self.register_buffer("frame_w", frame_w.reshape(self.n_fft, 1, self.window_size).float())

        col = torch.arange(self.n_fft, dtype=torch.float64)
        row = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64).unsqueeze(1)
        angle = 2.0 * math.pi * row * col / self.n_fft
        self.register_buffer("dft_cos", torch.cos(angle).float())
        self.register_buffer("dft_sin", torch.sin(angle).float())

        banks, _ = kaldi.get_mel_banks(n_mels, self.n_fft, sample_rate, low_freq, high_freq,
                                       100.0, -500.0, 1.0)
        self.register_buffer("mel_banks", F.pad(banks, (0, 1), value=0.0).float())

        if cmvn.shape[0] != 2 or cmvn.shape[1] != n_mels * lfr_m:
            raise ValueError(
                f"the checkpoint's CMVN is {tuple(cmvn.shape)}, but this frontend produces "
                f"{n_mels * lfr_m} features per row ({n_mels} mel bins stacked {lfr_m} deep) -- a "
                f"(2, {n_mels * lfr_m}) shift/scale pair is what `am.mvn` has to hold.")
        self.register_buffer("cmvn_add", cmvn[0:1, :].float())
        self.register_buffer("cmvn_mul", cmvn[1:2, :].float())

        # One depthwise kernel per window position: `(n_mels, 1, lfr_m)`, one-hot at tap j. Registered
        # as `lfr_m` separate buffers rather than one stacked tensor because they are `lfr_m` separate
        # convolutions -- a single grouped one would lay the output out mel-bin-major, which is the
        # transpose of what `apply_lfr` produces and of what the CMVN vector and the encoder's first
        # linear are indexed by.
        for j in range(lfr_m):
            tap = torch.zeros(n_mels, 1, lfr_m)
            tap[:, 0, j] = 1.0
            self.register_buffer(f"lfr_tap_{j}", tap)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """`(1, n_samples)` -> `(1, n_rows, n_mels * lfr_m)`."""
        framed = F.conv1d((waveform * self.scale).unsqueeze(1), self.frame_w, stride=self.hop)
        frames = framed[0]                                          # (n_fft, n_frames)
        real = torch.matmul(self.dft_cos, frames)
        imag = torch.matmul(self.dft_sin, frames)
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_banks, power)
        mel = torch.clamp(mel, min=LOG_FLOOR).log()                 # (n_mels, n_frames)
        return self.stack_and_normalize(mel)

    def stack_and_normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """`apply_lfr` then `apply_cmvn`, as `lfr_m` depthwise convolutions rather than an `as_strided`.

        Takes `mel` as `(n_mels, n_frames)` -- the layout the mel matmul above already produces -- and
        stays in it until the stacking is done. Row `i` of the output is frames
        `lfr_n*i .. lfr_n*i + lfr_m - 1` of the padded sequence, laid out mel-bin-fastest. See the class
        docstring for why this is convolutions and not slices, and note that nothing here computes a row
        count: the convolution's own output length IS it.

        **The edge padding repeats along the FRAME axis rather than along a transposed copy of it, and
        that is a ggml constraint rather than a preference.** `ggml_repeat` requires its source to be
        contiguous in the fastest-varying dimension (`nb00 == sizeof(float)`), and a slice of a PERMUTED
        tensor is not -- so the natural torch spelling, transposing to `(n_frames, n_mels)` first and
        repeating row 0, produces a graph that converts, exports, and then aborts inside
        `ggml_compute_forward_repeat` at run time. Slicing one frame out of the `(n_mels, n_frames)`
        layout is the same tensor by value and a legal repeat source.
        """
        channels = mel.unsqueeze(0)                               # (1, n_mels, n_frames)
        padded = torch.cat((channels[:, :, :1].expand(-1, -1, self.left_pad), channels,
                            channels[:, :, -1:].expand(-1, -1, self.right_pad)), dim=2)
        taps = [F.conv1d(padded, getattr(self, f"lfr_tap_{j}"), stride=self.lfr_n,
                         groups=channels.shape[1]) for j in range(self.lfr_m)]
        stacked = torch.cat(taps, dim=1)[0].transpose(0, 1)       # (n_rows, n_mels * lfr_m)
        return ((stacked + self.cmvn_add) * self.cmvn_mul).unsqueeze(0)


def sinusoidal_positions(x: torch.Tensor) -> torch.Tensor:
    """FunASR's `SinusoidalPositionEncoder.encode`, with the positions COUNTED OUT OF the tensor rather
    than built from its length.

    One-based (`1 .. timesteps`), which is the checkpoint's convention and not a detail: a zero-based
    range shifts every row of the table by one and produces output that is wrong everywhere and
    plausible everywhere.

    **`cumsum` of ones, not `arange`, and the difference is the whole of why this function exists.**
    `torch.arange(1, x.shape[1] + 1)` is the reference's own spelling and it is correct in torch; it is
    also a SHAPE QUERY, and MIL answers a shape query over a dynamic axis with a fresh opaque symbol
    that has no algebraic relation to the axis it came from. What then reaches the topology is the root
    axis -- `n_samples` -- so the position table claims one row per audio SAMPLE, and the engine aborts
    in `ggml_repeat` broadcasting it against 187 rows. Measured directly: converting this function both
    ways, `arange` produces `range_1d` under a new symbol and the output under a third, while the form
    below carries the encoder's own symbol from the convolution all the way to the sum.

    A cumulative sum of ones has the same VALUES and is not a shape query: `ones_like` is elementwise
    and `mean` over the static feature axis reduces one axis and preserves the other, so the positions
    inherit the row axis instead of re-deriving it.

    The squeeze/unsqueeze pair around the `cumsum` is `ggml_cumsum`'s, which only ever accumulates over
    the fastest-varying dimension -- and it is a pair of RESHAPES rather than the two transposes that
    read more naturally, because moving a length-1 axis past the row axis is pure metadata on a
    contiguous tensor while a transpose is a strided view, and the broadcast below it lowers to a
    `ggml_repeat` that requires a contiguous source.
    """
    depth = x.shape[2]
    increment = math.log(10000.0) / (depth / 2 - 1)
    inv = torch.exp(torch.arange(depth // 2, dtype=x.dtype, device=x.device) * -increment)
    ones = torch.ones_like(x).mean(dim=2, keepdim=True)            # (1, rows, 1)
    flat = torch.cumsum(ones.squeeze(2).unsqueeze(1), dim=-1)      # (1, 1, rows)
    pos = flat.squeeze(1).unsqueeze(2)                             # (1, rows, 1) = 1 .. rows
    scaled = pos * inv.reshape(1, 1, -1)
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=2)


class _SenseVoiceWrapper(nn.Module):
    """Reduces `SenseVoiceSmall` to `(waveform, prompt_ids) -> ctc logits`, front end included.

    Re-states `SenseVoiceEncoderSmall.forward` rather than calling it, and the three differences are all
    the module docstring's: no mask, positions from the tensor, and the front end in front. Everything
    between is the checkpoint's own layers, called in the checkpoint's own order.
    """

    def __init__(self, model, frontend: KaldiFbankLfrCmvn):
        super().__init__()
        self.model = model
        self.frontend = frontend

    def forward(self, waveform, prompt_ids):
        encoder = self.model.encoder
        x = torch.cat((self.model.embed(prompt_ids), self.frontend(waveform)), dim=1)
        # `sqrt(output_size)`, applied to the 560-wide INPUT rather than to the 512-wide hidden state.
        # That is what the reference does; it reads like a bug and is the trained scale.
        x = x * (encoder.output_size() ** 0.5)
        x = x + sinusoidal_positions(x)
        for layer in encoder.encoders0:
            x = layer(x, None)[0]
        for layer in encoder.encoders:
            x = layer(x, None)[0]
        x = encoder.after_norm(x)
        for layer in encoder.tp_encoders:
            x = layer(x, None)[0]
        return self.model.ctc.ctc_lo(encoder.tp_norm(x))


def _funasr_config(path: Path) -> Optional[dict]:
    """A FunASR checkpoint directory's own `config.yaml`, parsed, or `None` if `path` isn't one.

    Never raises: `TaskRegistry.detect` runs every recognizer against every path by construction, so
    "not one of mine" has to be an answer rather than an exception.
    """
    cfg_path = path / "config.yaml"
    if not path.is_dir() or not cfg_path.exists() or not (path / "model.pt").exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
    except Exception:  # a YAML file that will not parse is not this family's checkpoint
        return None
    return cfg if isinstance(cfg, dict) else None


def stage_spm_protobuf(model_dir: str) -> Optional[str]:
    """Copies the checkpoint's SentencePiece protobuf into a temp dir as `tokenizer.model`.

    The same adapter `extract_nemo_tokenizer_dir` is, for the same reason and against the same reader:
    `_write_tokenizer`'s `sentencepiece_proto` branch looks for one of three fixed filenames in a
    directory, and this family's checkpoint names its protobuf after the languages it covers
    (`chn_jpn_yue_eng_ko_spectok.bpe.model`). The NAME is read out of `configuration.json`, which is
    where the checkpoint states it, rather than globbed for `*.model` -- a glob would also match
    `model.pt`'s neighbours in a checkpoint that ships two.

    `None` for a directory with no protobuf, because a config can be built and introspected without a
    checkpoint on disk: `component_registry.usage()` does exactly that for every registered recognizer,
    and `backend_kwargs()` is on that path.
    """
    import json
    import shutil
    import tempfile

    root = Path(model_dir)
    meta_path = root / "configuration.json"
    if not meta_path.exists():
        return None
    try:
        metas = json.loads(meta_path.read_text()).get("file_path_metas") or {}
    except (json.JSONDecodeError, OSError):
        return None
    name = (metas.get("tokenizer_conf") or {}).get("bpemodel")
    if not name or not (root / name).exists():
        return None
    out = Path(tempfile.mkdtemp(prefix="loom_funasr_tokenizer_"))
    shutil.copyfile(root / name, out / "tokenizer.model")
    return str(out)


@dataclass(kw_only=True)
class SANMAsrExportConfig(LoomExportConfig):
    """A FunASR `SenseVoiceSmall` checkpoint directory -> Loom GGUF.

    Everything that could be declared is read off the checkpoint instead -- the frontend geometry, the
    CMVN, the sample rate, the blank id, the language table. What is left is the path, which is the
    whole spec.
    """

    architecture: Optional[str] = None
    model_dir: str
    decomposition: Decomposition = None
    # `EXPORT-ROADMAP.md` R1: a waveform's own axis is raw audio samples, never a token count -- the
    # same declaration families 1 and 4 make, checked the same way.
    root_axis: str = "n_samples"
    # All resolved from the checkpoint during `load_model()` / `build_trace()`.
    _sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    _blank_id: Optional[int] = field(default=None, init=False, repr=False)
    _languages: dict = field(default_factory=dict, init=False, repr=False)
    _textnorms: dict = field(default_factory=dict, init=False, repr=False)
    _prompt_default: tuple = field(default=(), init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the FunASR checkpoint directory. The recognizer's detect() already read its "
            "config.yaml and found `model.pt` beside it -- that is how it claimed the checkpoint at "
            "all -- and funasr's own AutoModel raises on anything it cannot build."
        ),
        "_sample_rate": Unchecked(
            "READ off the checkpoint's own frontend during build_trace, never declared -- it is what "
            "the trace length and the dynamic-axis bounds are derived from, so a caller value would be "
            "a second authority over the same number."
        ),
        "_blank_id": Unchecked(
            "READ off the restored model (`SenseVoiceSmall.blank_id`). This family's blank is row 0, "
            "not the last row, so the class count cannot supply it and nothing else in the artifact "
            "states it."
        ),
        "_languages": Unchecked(
            "the checkpoint's own `lid_dict`, copied into the contract so a host can name a language "
            "instead of knowing an embedding row. There is no second authority to check it against; "
            "what would falsify it is the model declining an id, and every id here comes from it."
        ),
        "_textnorms": Unchecked("the checkpoint's own `textnorm_dict`, for the same reason."),
        "_prompt_default": Unchecked(
            "assembled from the two dicts above plus the two fixed event/emotion rows, which is the "
            "same vector `SenseVoiceSmall.inference` builds for its own default arguments."
        ),
    }

    def __post_init__(self):
        # Structural, not chosen: the front end, the encoder and the CTC head are one graph with no
        # boundary a caller could name.
        if self.decomposition is None:
            self.decomposition = Flattened()

    def load_model(self):
        """The whole `AutoModel`, not just `.model` -- the frontend beside it is where the CMVN and the
        fbank geometry live, and both are needed by `build_trace`.

        `disable_update=True` because an exporter must not reach the network to decide what a local
        checkpoint is; FunASR's default checks ModelScope for a newer version on every construction.
        """
        from funasr import AutoModel

        print(f"Loading FunASR model from {self.model_dir}...")
        auto = AutoModel(model=self.model_dir, device="cpu", disable_update=True)
        auto.model.eval()
        self._auto = auto
        return auto.model

    def export_architecture(self) -> str:
        return self.architecture or "sense-voice-small"

    def build_trace(self, model):
        """`Flattened`'s hook: the wrapper, one clip of dummy audio and one prompt, and the two MIL
        input declarations.

        The frontend is read here rather than in `load_model` because it answers the sample rate the
        trace length is built from, and that is needed exactly at this point.
        """
        import coremltools as ct

        frontend = self._auto.kwargs["frontend"]
        if frontend.cmvn is None:
            raise ValueError(
                f"{self.model_dir} declares no `cmvn_file`, so its frontend has no mean/variance "
                f"normalization to fold into the graph. Every checkpoint in this family ships one "
                f"(`am.mvn`), and a graph without it feeds the encoder features at the wrong scale.")
        self._sample_rate = int(frontend.fs)
        self._blank_id = int(model.blank_id)
        self._languages = dict(model.lid_dict)
        self._textnorms = dict(model.textnorm_dict)
        # The vector `SenseVoiceSmall.inference` builds for `language="auto", use_itn=False`: the
        # language query, the two fixed event/emotion queries (rows 1 and 2, which the reference
        # hardcodes), and the text-normalization query.
        self._prompt_default = (int(self._languages["auto"]), 1, 2, int(self._textnorms["woitn"]))

        wrapped = _SenseVoiceWrapper(model, KaldiFbankLfrCmvn(
            frontend.cmvn, n_mels=frontend.n_mels, sample_rate=self._sample_rate,
            frame_length=float(frontend.frame_length), frame_shift=float(frontend.frame_shift),
            window_type=frontend.window, lfr_m=frontend.lfr_m, lfr_n=frontend.lfr_n,
            upscale_samples=bool(frontend.upsacle_samples)))

        n_samples = int(TRACE_SECONDS * self._sample_rate)
        dummy_inputs = (torch.randn(1, n_samples, dtype=torch.float32),
                        torch.tensor([self._prompt_default], dtype=torch.int64))
        print(f"Tracing the complete PyTorch graph (dummy n_samples={n_samples}, "
              f"prompt={list(self._prompt_default)})...")
        seq_dim = ct.RangeDim(int(MIN_SECONDS * self._sample_rate),
                              int(MAX_SECONDS * self._sample_rate))
        mil_inputs = [
            # The waveform FIRST and not by preference: `apply_monolithic_export` derives the driver's
            # root-axis expression from the first declared input, so a prompt-first graph would measure
            # the utterance in prompt rows.
            ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
            ct.TensorType(name="prompt_ids", shape=(1, len(self._prompt_default)), dtype=np.int32),
        ]
        return wrapped, dummy_inputs, mil_inputs

    def synthesized_builder_key(self) -> str:
        """Family 1's answer, for family 4's reason: this is a `Flattened` export and what differs from
        every other one is entirely what the host does with the single output."""
        return "CtcGreedy"

    def hparams(self) -> dict:
        return {} if self._sample_rate is None else {"sample_rate": self._sample_rate}

    # The two `lid_dict` entries that are not languages: `auto` asks the model to detect one, and
    # `nospeech` is a verdict it can return. Both are real prompt rows and neither belongs in the
    # "which languages does this model speak" list a host filters on.
    _NON_LANGUAGE_PROMPTS = ("auto", "nospeech")

    def contract(self) -> dict:
        """The task pair, the languages this checkpoint speaks, and this family's prompt tables.

        **The prompt ids are NOT published under `loom.asr.language_ids`, and resisting that was the
        point.** `language` is a recurring ASR role, so reaching for the keys Whisper already publishes
        is the obvious move -- but those keys are read into `AsrDecodeTable`, whose ids are DECODER
        PROMPT TOKENS that `transcribe.cpp` pushes into a cross-attention prompt. This family's ids
        index a 16-row embedding table prepended to the FEATURES instead. The two are the same concept
        for a caller and different objects for the engine, and putting one where the other is read is
        how a table gets used for the wrong mechanism. Today it would be inert -- that path is gated on
        a declared clip length and this file declares none -- which is exactly the kind of accident
        that stops being inert later.

        What IS shared is the mechanism-free half: `loom.text.languages`, which `ModelContract` reads
        and which the engine already uses to refuse a language a file cannot serve. The name -> row
        tables go under this family's own prefix, where only something that knows what a SANM prompt is
        will look for them, and the model card names them.
        """
        contract = super().contract()
        if self._languages:
            names = sorted(self._languages, key=self._languages.get)
            contract["text.languages"] = [n for n in names if n not in self._NON_LANGUAGE_PROMPTS]
            contract["sanm.language_names"] = names
            contract["sanm.language_ids"] = [int(self._languages[n]) for n in names]
        if self._textnorms:
            names = sorted(self._textnorms, key=self._textnorms.get)
            contract["sanm.textnorm_names"] = names
            contract["sanm.textnorm_ids"] = [int(self._textnorms[n]) for n in names]
        if self._prompt_default:
            contract["sanm.prompt_default"] = [int(i) for i in self._prompt_default]
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(
            flat_namespace=True,
            root_axis=self.root_axis,
            driver_builder=self.synthesized_builder_key(),
            hparams=self.hparams(),
        )
        tokenizer_dir = stage_spm_protobuf(self.model_dir)
        if tokenizer_dir is not None:
            kwargs["tokenizer_dir"] = tokenizer_dir
            kwargs["tokenizer_family"] = "sentencepiece_proto"
        # Omitted rather than raised before the trace has run, because `component_registry.usage()`
        # builds every registered config without ever tracing. The export path cannot slip through: the
        # exporter raises when asked for the CTC builder without a blank id.
        if self._blank_id is not None:
            kwargs["ctc_blank_id"] = self._blank_id
        if self._prompt_default:
            # The one non-waveform input, and the whole reason it can stay invisible to a caller who
            # does not want it. See `driver_components.DEFAULTED`.
            kwargs["defaulted_inputs"] = {"prompt_ids": [int(i) for i in self._prompt_default]}
        return kwargs


def _is_funasr_sensevoice(path: Path) -> bool:
    """A FunASR checkpoint directory whose `config.yaml` declares `model: SenseVoiceSmall`.

    Named rather than structural, unlike families 4 and 12, and the difference is in the checkpoint
    rather than in the taste: an HF directory declares `architectures`, a list of classes whose SUFFIX
    is the contract (`*ForCTC`), while a FunASR `config.yaml` declares one `model:` key that is a class
    name in FunASR's own registry with no shared suffix to match on. `Paraformer` sits in the same
    directory layout with the same `encoder: SANMEncoder` and a CIF predictor and a non-autoregressive
    decoder after it -- so a structural check on the encoder alone would claim a checkpoint this
    template cannot export. The second leaf gets the second recognizer.
    """
    cfg = _funasr_config(path)
    return bool(cfg) and cfg.get("model") == "SenseVoiceSmall"


def _build_funasr_sensevoice(path: Path, output_path: str) -> LoomExportConfig:
    return SANMAsrExportConfig(architecture=None, output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=SANMAsrExportConfig,
        recognizers=[
            ModelRecognizer(name="funasr-sensevoice", detect=_is_funasr_sensevoice,
                            build_config=_build_funasr_sensevoice),
        ],
    ))
