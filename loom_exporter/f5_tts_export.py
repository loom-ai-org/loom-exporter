"""Export F5-TTS (`SWivid/F5-TTS`, the `F5TTS_v1_Base` DiT checkpoint) -- family 9's third leaf, and
the first one whose sampler is not plain uniform Euler over a single evaluation.

F5-TTS is a flow-matching mel decoder with no duration model and no phonemizer: you give it a
reference clip, the text that clip says, and the text you want said, and it in-fills the rest of one
mel spectrogram. That shape is why the four phases below are what they are.

Four phases:
  - `mel`:        waveform -> the reference clip's log-mel, FRAME-major (`[T, 100]`), the layout the
                  estimator's own `cond` input wants. `torchaudio.transforms.MelSpectrogram` with
                  `power=1`, then `clamp(1e-5).log()` -- `get_vocos_mel_spectrogram` line for line,
                  with the filterbank READ off the real transform rather than recomputed.
  - `text_embed`: character ids -> `[n, 512]`, the DiT's own `TextEmbedding` (embedding + a
                  sinusoidal table + four ConvNeXtV2 blocks). Called TWICE per synthesis, on the real
                  ids and on the dropped ones; see the `keep` input below for why that is one
                  topology and not two.
  - `estimator`:  one velocity evaluation of the 22-layer DiT. Called twice per ODE stage under
                  classifier-free guidance, by the ENGINE rather than by the driver.
  - `vocoder`:    Vocos (`charactr/vocos-mel-24khz`), mel -> waveform. A separate checkpoint from a
                  separate repository, which is why `F5TTSExportConfig` takes two paths.

**The estimator's sequence length is the WHOLE utterance, reference included.** F5-TTS conditions by
in-filling: `cond` is the reference mel padded with zeros out to the target duration, the text is the
reference transcript concatenated with the text to speak, and the generated audio is the tail of the
result. So there is exactly one dynamic axis across `text_embed` and `estimator` -- `n_tokens`, the
total frame count -- and the driver slices the reference's frames off the answer before the vocoder
sees it. Nothing here needs a second symbol, which is what keeps it inside `GraphBuilder`'s one-
dynamic-length-per-topology limit.

**Two engine-visible things this leaf needed, and one it did not.** It did not need a new primitive:
`loom.run_ode` already integrated a learned vector field with the state on the C++ side (ADR-031).
What it did need is that the integrator learned CLASSIFIER-FREE GUIDANCE -- two evaluations of one
graph per stage, combined as `v_cond + scale * (v_cond - v_uncond)` -- because F5-TTS's velocity field
is not the estimator's output, it is that combination, and running the pair in Lua would put the whole
mel spectrogram across the boundary four times per step for arithmetic with no decision in it (which
is exactly the measurement ADR-031 was written from). And `FlowMatchingSpec` learned that the time
schedule can be the CALLER's: F5-TTS integrates a "sway"-reparameterised linspace,
`t + coef*(cos(pi/2 * t) - 1 + t)` with `coef = -1`, not `k/n_steps`. Both are declarations on the
existing template rather than a bespoke sampler -- the LOOP is unchanged, which is the test of whether
a template still fits.

Trace-friendliness patches, all of them the same category as every prior MIL export here:
  - **The text mask is an INPUT, not a comparison.** `TextEmbedding.forward` derives `text_mask` from
    the ids (`text == 0`) and then, for the unconditional branch, REPLACES those ids with zeros. Both
    branches must mask with the CONDITIONAL mask -- a graph that recomputed `ids == 0` would mask
    everything on the unconditional call and return a different tensor than the reference does. So
    `keep` is handed in beside the ids, and the two branches are one topology called twice.
  - **RoPE is a constant table sliced by the sequence length**, not
    `RotaryEmbedding.forward_from_seq_len`'s `arange` + `einsum`. Identical values by construction
    (the table is built by calling the real method once, at the maximum length), and it keeps a
    dynamic `arange` out of the graph. The text embedding's own `freqs_cis` buffer is already written
    this way by the reference and is left alone.
  - **`seq_len` is an int, so `valid_pos_mask` is None and the reference's own batch path never
    runs.** This project's single-unpadded-utterance convention again; the batched branch builds a
    comparison against a per-sample length tensor that has no meaning for one sequence.
  - `f5_tts.model.__init__` imports `Trainer`, which imports wandb/accelerate/ema_pytorch. Stubbed in
    `sys.modules` before the real package is touched -- the same stand-in pattern
    `matcha_export.py` uses for `matcha.utils`, and for the same reason.

**What is NOT in the GGUF, and is a real limitation rather than an oversight.** F5-TTS's text front end
is `convert_char_to_pinyin`, which runs `rjieba` word segmentation and `pypinyin` before it ever
reaches a vocabulary: Chinese characters become toned pinyin syllables, and a space is inserted before
a multi-character segment whose predecessor did not end in one. The segmentation half cannot be a
table. What ships is the character table (`tokenizer.ggml.model == "f5"`), and `F5Vocab` implements the
ASCII half of that function exactly -- which for ordinary English text is the identity on characters.
Text containing CJK is refused by name rather than tokenized wrongly. Same boundary the phoneme-input
families draw around g2p.

Usage:
  loom-export /path/to/F5TTS_v1_Base -o f5_tts.gguf --task text-to-speech --model f5-tts \\
      --vocoder /path/to/vocos-mel-24khz
"""
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .export_config import LoomExportConfig
from .f5_tokenizer_export import f5_vocab_size
from .decomposition import Decomposition, MultiPhase
from .flow_matching_export import FlowMatchingSpec
from .multi_phase_export import ExportPhase, BaseMultiPhaseModelExportConfig
from .paths import driver_dir
from .spec_protocol import Axis, Unchecked

# Where the F5-TTS checkout lives. `f5_tts` is a PyPI package, but installing it drags in gradio,
# wandb, datasets and bitsandbytes for four `nn.Module`s -- the same trade `matcha_export.py` records,
# and the same resolution: a git clone on `sys.path`.
F5_REPO = "/home/flavio/Dev/F5-TTS/src"

# The mel geometry, `F5TTS_v1_Base.yaml`'s `mel_spec` block. Constants rather than config reads
# because there is no config file beside the checkpoint -- the release is a bare `.safetensors` plus a
# `vocab.txt`, and these five numbers are what the vocoder was trained against.
SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MEL = 100

# The DiT's own geometry, the same file's `arch` block.
DIT_ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True,
                conv_layers=4, qk_norm=None, pe_attn_head=None, attn_backend="torch",
                attn_mask_enabled=False, checkpoint_activations=False)

# How far the two position tables reach. The reference's own `precompute_max_pos` is 8192 frames
# (~87 s at 24 kHz / hop 256) and the rope table is built to match, so a longer utterance is refused by
# the graph's own declared range rather than reading past a table. Both are CONSTANTS in the GGUF:
# 8192*512 and 8192*64 floats is 18 MB against a 1.3 GB model, and the alternative -- an `arange` over
# a dynamic axis feeding an `outer` -- is the shape the exporter's own walk is worst at.
MAX_POS = 8192

# `infer_batch_process`'s own defaults, which are the numbers every published F5-TTS sample was
# produced with. `sway_sampling_coef = -1` is not a tuning knob in disguise: the reference passes it
# unconditionally and the checkpoint was evaluated under it.
DEFAULT_STEPS = 32
DEFAULT_CFG = 2.0
DEFAULT_SWAY = -1.0

# Where the mel vocoder is expected to sit inside the model directory. F5-TTS ships none -- the
# release is the DiT and a `vocab.txt` -- and every published sample was vocoded by
# `charactr/vocos-mel-24khz`, so the two files that repository holds (`config.yaml` and
# `pytorch_model.bin`) go here.
VOCODER_SUBDIR = "vocos-mel-24khz"

# The trace length, and the range the estimator's graph may be built over. 32 frames is long enough
# that every convolution here (kernel 31, dilated kernel 7) sees a real interior; the ceiling is
# MAX_POS, where the tables run out.
TRACE_FRAMES = 64
MIN_FRAMES = 32
TRACE_SAMPLES = SAMPLE_RATE  # one second of reference audio, for the mel phase's own trace


def _find_checkpoint(model_dir: Path) -> Path:
    """The release's weight file. One `.safetensors`, or a named one if several are present."""
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"{model_dir} holds no `.safetensors` -- an F5-TTS release directory is a weight file "
            f"plus a `vocab.txt`.")
    if len(files) > 1:
        raise ValueError(
            f"{model_dir} holds {len(files)} `.safetensors` files ({[f.name for f in files]}); this "
            f"export takes a directory with exactly one, so which checkpoint it converted is not a "
            f"guess. Point it at a directory holding the one you mean.")
    return files[0]


def _import_f5():
    """Import the F5-TTS checkout, with the training half stubbed out.

    `f5_tts.model.__init__` does `from f5_tts.model.trainer import Trainer`, and that module imports
    wandb at module scope. Registering the stub BEFORE the package is first imported is what makes the
    real `f5_tts.model.cfm`/`backbones.dit` modules importable without it -- the import system finds
    the already-present entry in `sys.modules` rather than executing the file."""
    if "f5_tts.model.trainer" not in sys.modules:
        stub = types.ModuleType("f5_tts.model.trainer")
        stub.Trainer = object
        sys.modules["f5_tts.model.trainer"] = stub
    if F5_REPO not in sys.path:
        sys.path.insert(0, F5_REPO)
    from f5_tts.model.backbones.dit import DiT  # noqa: F401  (imported for its side effect on callers)
    return DiT


class F5MelFrontend(nn.Module):
    """`get_vocos_mel_spectrogram`, as one traceable module returning FRAME-major mel.

    The filterbank and the window are READ off a real `torchaudio.transforms.MelSpectrogram` built
    with the reference's own arguments rather than recomputed here: `norm=None` and the default
    `mel_scale="htk"` are both easy to get subtly wrong, and a wrong filterbank is a plausible-looking
    spectrogram that conditions the model on the wrong voice.

    Returns `[1, T, 100]` rather than the reference's `[1, 100, T]`. The transpose is here rather than
    in the driver because the estimator's `cond` input is frame-major, and because a bare `.transpose()`
    as a traced graph's declared output is a live GGML permute view that `ggml_backend_tensor_get`
    silently ignores -- the bug `vits_export.py`'s `TextWrapper` already paid for. Ending on a real op
    (the `log`) and transposing INSIDE the trace avoids it.
    """

    def __init__(self):
        super().__init__()
        import torchaudio

        t = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
            n_mels=N_MEL, power=1, center=True, normalized=False, norm=None)
        self.register_buffer("window", t.spectrogram.window.clone())
        # `mel_scale.fb` is stored `(n_freq, n_mels)`; the matmul below wants `(n_mels, n_freq)`.
        self.register_buffer("filters", t.mel_scale.fb.T.contiguous())

    def forward(self, waveform):                       # (1, n_samples)
        stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                          window=self.window, center=True, return_complex=True)
        # `power=1` is the MAGNITUDE, not the power spectrum -- Whisper's frontend squares here and
        # this one must not.
        mel = torch.matmul(self.filters, stft.abs())   # (1, n_mels, T)
        mel = torch.clamp(mel, min=1e-5).log()
        return mel.transpose(1, 2)                     # (1, T, n_mels)


class F5TextEmbedPhase(nn.Module):
    """`DiT.text_embed` for one unpadded sequence, with the filler mask supplied rather than derived.

    See the module docstring: the reference computes the mask from the ids and then replaces the ids
    for the unconditional branch, so mask and ids are genuinely independent inputs and this is one
    topology the driver calls twice.
    """

    def __init__(self, text_embed):
        super().__init__()
        self.te = text_embed
        # The reference registers `freqs_cis` non-persistently and slices it; keeping a copy here is
        # what puts it in the GGUF rather than leaving it to be recomputed.
        self.register_buffer("freqs", text_embed.freqs_cis[:MAX_POS].clone())

    def forward(self, ids, keep):                      # (1, n) int32, (1, n) f32
        x = self.te.text_embed(ids)
        x = x + self.freqs[: ids.shape[1], :]
        k = keep.unsqueeze(-1)
        # `masked_fill(text_mask, 0.0)` where `text_mask` is the FILLER positions, written as a
        # multiply by its complement. Same values, and no comparison op over an integer tensor.
        x = x * k
        for block in self.te.text_blocks:
            x = block(x)
            x = x * k
        return x                                       # (1, n, 512)


def _rope_pair_swap(dim: int) -> torch.Tensor:
    """The constant matrix `S` with `t @ S == rotate_half(t)`, for x_transformers' INTERLEAVED
    convention.

    `rotate_half` is written as `rearrange(x, '... (d r) -> ... d r', r=2)`, an unbind, a stack and a
    rearrange back. On a `(batch, heads, tokens, head_dim)` query that middle form is **five
    dimensions**, and ggml has four -- the export converts and the engine refuses it at run time with
    `VIEW 'shape' attribute must have 1-4 entries, got 5`.

    The map itself is linear and fixed: `out[2i] = -x[2i+1]` and `out[2i+1] = x[2i]`, i.e. a
    block-diagonal matrix of `[[0, 1], [-1, 0]]` blocks. One `(64, 64)` constant and one matmul per
    call site replaces the whole rearrange-unbind-stack sequence -- exact, four-dimensional, and
    ~0.06% of the estimator's own arithmetic (44 sites x n x 64 x 64 against a 22-layer 1024-wide
    transformer).
    """
    if dim % 2 != 0:
        raise ValueError(f"rope pair swap needs an even head dim, got {dim}")
    swap = torch.zeros(dim, dim, dtype=torch.float32)
    for i in range(0, dim, 2):
        swap[i + 1, i] = -1.0        # out[2i]   = -x[2i+1]
        swap[i, i + 1] = 1.0         # out[2i+1] =  x[2i]
    return swap


def _apply_rope_precomputed(t, freqs, scale=1.0):
    """`apply_rotary_pos_emb` with the re-slice removed, cos/sin handed in, and `rotate_half` as a
    matmul. Three substitutions, each forced by something the reference does that does not export.

    **The re-slice.** x_transformers' function opens with `freqs = freqs[:, -seq_len:, :]`, a defensive
    trim for a table longer than the sequence. Here the table has already been cut to exactly
    `seq_len`, so it is the identity -- but a NEGATIVE begin over a dynamic axis is a slice shape the
    exporter's walk renders arithmetically instead of normalising against the source length. It emits
    `VIEW shape=[64, 2*n_tokens, 1] offset=-256*n_tokens`, twice the rows at a negative offset, once
    per query and per key in all 22 blocks. The graph converts, writes and loads; only RUNNING fails
    (loom.cpp Retro-051).

    **`rotate_half`.** Five-dimensional on a four-dimensional query -- see `_rope_pair_swap`.

    **cos/sin rather than the angles.** A real saving rather than tidiness: the reference recomputes
    `freqs.cos()` and `freqs.sin()` inside every one of those 44 call sites, over a table that changes
    neither across blocks nor across sampling steps. Precomputed, they are two constants sliced once,
    and the exported estimator has ONE `COS` node in it (the timestep embedding's) instead of 45.

    Restricted to a FULL rotation (`rot_dim == head_dim`), which is what this checkpoint has
    (`RotaryEmbedding(dim_head)`): the partial-rotary split the reference carries would leave an empty
    `t[..., rot_dim:]` to concatenate. Both restrictions raise rather than being silently assumed.
    """
    cos, sin, swap = freqs
    if cos.shape[-1] != t.shape[-1]:
        raise NotImplementedError(
            f"F5-TTS rope: the table is {cos.shape[-1]} wide and the head is {t.shape[-1]} -- this "
            f"substitution assumes a full rotation, which is what `RotaryEmbedding(dim_head)` builds. "
            f"A partial one needs the reference's own split and concatenate.")
    if scale != 1.0:
        raise NotImplementedError(
            "F5-TTS rope: xpos scaling is not part of this checkpoint (`use_xpos=False`), so it is "
            "refused rather than ignored.")
    if t.ndim == 4 and cos.ndim == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return t * cos + torch.matmul(t, swap) * sin


class F5EstimatorPhase(nn.Module):
    """One velocity evaluation of the DiT: `v = f(x, cond, text_embed, t)`.

    This is `DiT.forward` with the text-embedding half already done (it is the `text_embed` phase, and
    it is invariant across every step) and with the reference's own `cfg_infer` packing removed --
    the two guided runs are two CALLS here, not a doubled batch, because the engine's integrator runs
    them and a doubled batch would double the sequence axis the whole graph is sized by.
    """

    def __init__(self, transformer):
        super().__init__()
        import f5_tts.model.modules as f5_modules

        self.t = transformer
        freqs, _ = transformer.rotary_embed.forward_from_seq_len(MAX_POS)
        # cos/sin rather than the angles: see `_apply_rope_precomputed`. Built by calling the REAL
        # `forward_from_seq_len` once at the ceiling, so the values are the reference's own and only
        # the length is this export's decision.
        self.register_buffer("rope_cos", freqs.cos())  # (1, MAX_POS, dim_head)
        self.register_buffer("rope_sin", freqs.sin())
        self.register_buffer("rope_swap", _rope_pair_swap(freqs.shape[-1]))
        # `AttnProcessor` resolves `apply_rotary_pos_emb` as a module global at call time, so this is
        # where the substitution has to land. Idempotent, and scoped to a wrapper that exists only to
        # be traced -- the same arrangement `supertonic_export`'s own ConvNext patch uses, and the
        # reason idempotence matters here is that patching the same name twice would nest the
        # replacement inside itself.
        if getattr(f5_modules.apply_rotary_pos_emb, "__name__", "") != "_apply_rope_precomputed":
            f5_modules.apply_rotary_pos_emb = _apply_rope_precomputed

    def forward(self, x, cond, text_embed, t):         # (1,n,100) (1,n,100) (1,n,512) (1,)
        time = self.t.time_embed(t)
        h = self.t.input_embed.proj(torch.cat((x, cond, text_embed), dim=-1))
        h = self.t.input_embed.conv_pos_embed(h) + h
        n = x.shape[1]
        # `None` where the reference puts its xpos scale, which is what makes `AttnProcessor` pass
        # `scale = 1.0` -- this checkpoint's `RotaryEmbedding` has `use_xpos=False`.
        rope = ((self.rope_cos[:, :n, :], self.rope_sin[:, :n, :], self.rope_swap), None)
        for block in self.t.transformer_blocks:
            # `mask=None` throughout: one unpadded sequence, so every attention is full and no mask is
            # constructed at all. loom.cpp ADR-019's reasoning, one modality over.
            h = block(h, time, mask=None, rope=rope)
        h = self.t.norm_out(h, time)
        return self.t.proj_out(h)                      # (1, n, 100)


class F5VocoderPhase(nn.Module):
    """Vocos: mel -> waveform. Channel-major in, because that is `Vocos.decode`'s own convention.

    **`ISTFTHead` is the one module here that cannot be traced as written**, and for the reason
    `istft.py` was written down: it builds a COMPLEX tensor (`mag * (cos + 1j*sin)`) and hands it to
    `torch.istft`, which has no coremltools torch-frontend handler at all -- the failure is inside
    TorchScript, before any MIL graph exists for a pass to fix. So the head's last two lines are
    replaced by the same real/imaginary pair fed to this project's own traceable `ISTFT`, which is
    already what Kokoro and StyleTTS2 synthesise through. Everything above that line -- the linear
    projection, the exp, the clip, the cos/sin -- is the real module's, unchanged.

    `center=True` because `vocos-mel-24khz`'s config declares `padding: center`, which is the branch
    of `vocos.spectral_ops.ISTFT` that delegates to `torch.istft`. A `padding: same` checkpoint would
    need the other branch's trim and is refused rather than silently centred.
    """

    def __init__(self, vocos):
        super().__init__()
        from .istft import ISTFT

        head = vocos.head
        if getattr(head.istft, "padding", None) != "center":
            raise NotImplementedError(
                f"this Vocos head pads with {getattr(head.istft, 'padding', None)!r}; only 'center' "
                f"is implemented here, and 'same' trims by `(win_length - hop_length) // 2` rather "
                f"than by `n_fft // 2` -- a different signal, not a different spelling.")
        self.backbone = vocos.backbone
        self.out = head.out
        self.istft = ISTFT(n_fft=head.istft.n_fft, hop_length=head.istft.hop_length,
                           win_length=head.istft.win_length, center=True)

    def forward(self, mel):                            # (1, m, 100) -- FRAME-major
        # **The transpose is in the GRAPH, and that is the decision rather than a convenience.** The
        # estimator's output is frame-major (`ne = [100, n_tokens]`, 100 contiguous floats per frame),
        # which is the layout its own `x`/`cond` inputs are in; `Vocos.decode`'s convention is
        # channel-major. Something has to convert, and the producer cannot -- a bare `.transpose()` as
        # a traced graph's declared output is a live GGML permute view that `ggml_backend_tensor_get`
        # silently ignores (`vits_export.py`'s `TextWrapper` paid for that one). So the CONSUMER takes
        # the producer's layout and transposes on the way in, where it is an ordinary interior op.
        #
        # The alternative -- converting in Lua -- is what ADR-031 and ADR-032 exist to refuse: it is
        # 86,800 elements through the boundary per synthesis for a reindexing no host decision depends
        # on. It is also what this export did until the ASR oracle caught it: every phase graded clean
        # tensor-for-tensor and the audio was "(chimes ringing)", because a frame-major array read as
        # channel-major is still a plausible spectrogram.
        x = self.backbone(mel.transpose(1, 2))
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)      # the reference's own overflow guard
        return self.istft(mag * torch.cos(p), mag * torch.sin(p))


def _load_vocos(vocoder_dir: str):
    """The Vocos mel vocoder, from a local `charactr/vocos-mel-24khz` checkout.

    `Vocos.from_pretrained` would reach the Hub; this takes the two files it downloads. The
    `EncodecFeatures` branch `load_vocoder` carries is deliberately not reproduced -- this config
    refuses a vocoder whose feature extractor is not the mel one, because an EnCodec-featured Vocos
    decodes a different input entirely and would load happily and synthesise noise.
    """
    import torch
    from vocos import Vocos
    from vocos.feature_extractors import MelSpectrogramFeatures

    d = Path(vocoder_dir)
    model = Vocos.from_hparams(str(d / "config.yaml"))
    if not isinstance(model.feature_extractor, MelSpectrogramFeatures):
        raise ValueError(
            f"{vocoder_dir} is a Vocos checkpoint whose feature extractor is "
            f"{type(model.feature_extractor).__name__}, not MelSpectrogramFeatures. F5-TTS's mel is "
            f"what this export feeds it, and an EnCodec-featured Vocos takes codec embeddings -- it "
            f"would load and synthesise noise.")
    state = torch.load(str(d / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@dataclass(kw_only=True)
class F5TTSExportConfig(BaseMultiPhaseModelExportConfig):
    """An `F5TTS_v1_Base` checkpoint directory plus a local Vocos vocoder -> one Loom GGUF.

    Two paths because it is two checkpoints from two repositories: F5-TTS ships the DiT and a
    `vocab.txt` and nothing else, and every published F5-TTS sample was vocoded by
    `charactr/vocos-mel-24khz`. Bundling them is what makes the artifact synthesise on its own -- the
    same call `matcha_export` makes about HiFi-GAN v1.
    """

    architecture: str = "f5-tts"
    model_dir: str
    # Defaulted to a subdirectory of the model directory rather than taken as a CLI flag, which is
    # `matcha_export`'s own arrangement: its config requires `generator_v1` to sit beside
    # `matcha_ljspeech.ckpt`. A release that needs a second repository's weights needs the two
    # assembled into one directory somewhere, and making that the EXPORT's requirement means the
    # recognizer can check it -- a `--vocoder` flag would be a fifth argument that only one family in
    # the registry understands, checked by nothing until the trace failed.
    vocoder_dir: str = ""
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = driver_dir("convert_f5_tts", "f5_tts_driver")
    _vocab_size: Optional[int] = field(default=None, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked(
            "the GGUF's own architecture string; it names this export rather than describing the "
            "checkpoint, which ships no config file at all -- the same reading `paraformer_export` "
            "gives the same field."
        ),
        "model_dir": Unchecked(
            "path to the F5-TTS release directory; the recognizer's detect() already found the "
            "`vocab.txt` and a `.safetensors` beside it whose keys are the DiT's."
        ),
        "vocoder_dir": Unchecked(
            "path to a local `charactr/vocos-mel-24khz` checkout. `_load_vocos` raises naming the "
            "feature extractor if it is the wrong kind of Vocos, which is the failure that would "
            "otherwise be silent."
        ),
        "decomposition": Unchecked("MultiPhase by construction -- four graphs, and the sampler runs "
                                    "between two of them"),
        "_vocab_size": Unchecked(
            "READ off the checkpoint's own `vocab.txt` during phases(); it sizes the text embedding "
            "table and the trace's dummy ids, and the embedding's own weight is what would disagree."
        ),
    }

    def __post_init__(self):
        if not self.vocoder_dir:
            self.vocoder_dir = str(Path(self.model_dir) / VOCODER_SUBDIR)
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()

    def load_model(self):
        """The DiT, with its weights out of the release's `.safetensors`.

        The release is an EMA checkpoint whose keys are `ema_model.*` plus two optimizer scalars, so
        the prefix is stripped and `strict=False` absorbs `initted`/`step`. A file that has no
        `ema_model.` keys at all is loaded as-is, which is what the non-EMA releases are.
        """
        from safetensors.torch import load_file

        _import_f5()
        from f5_tts.model.backbones.dit import DiT

        # The same reader the vocabulary writer uses, so the width this sizes the embedding by and the
        # table that ships beside it are one fact. Loading the checkpoint `strict=True` below is what
        # would catch a disagreement -- the embedding's own row count is `vocab_size + 1`.
        self._vocab_size = f5_vocab_size(self.model_dir)

        ckpt = _find_checkpoint(Path(self.model_dir))
        print(f"Loading F5-TTS from {ckpt} (vocab {self._vocab_size})...")
        model = DiT(mel_dim=N_MEL, text_num_embeds=self._vocab_size, **DIT_ARCH)
        state = load_file(str(ckpt))
        ema = {k[len("ema_model.transformer."):]: v for k, v in state.items()
               if k.startswith("ema_model.transformer.")}
        if not ema:
            ema = {k[len("transformer."):]: v for k, v in state.items()
                   if k.startswith("transformer.")}
        if not ema:
            raise ValueError(
                f"{ckpt} holds no `ema_model.transformer.*` or `transformer.*` tensors -- it is not "
                f"an F5-TTS release checkpoint. Its top-level key prefixes are "
                f"{sorted({k.split('.')[0] for k in state})}.")
        model.load_state_dict(ema, strict=True)
        return model.eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        vocos = _load_vocos(self.vocoder_dir)

        frame_dim = ct.RangeDim(MIN_FRAMES, MAX_POS)
        sample_dim = ct.RangeDim(MIN_FRAMES * HOP_LENGTH, MAX_POS * HOP_LENGTH)

        text_phase = F5TextEmbedPhase(model.text_embed).eval()
        est_phase = F5EstimatorPhase(model).eval()

        dummy_ids = torch.randint(1, self._vocab_size, (1, TRACE_FRAMES), dtype=torch.int32)
        dummy_keep = torch.ones(1, TRACE_FRAMES, dtype=torch.float32)
        dummy_text = torch.randn(1, TRACE_FRAMES, DIT_ARCH["text_dim"])
        return [
            ExportPhase(
                name="mel",
                wrapper=F5MelFrontend().eval(),
                dummy_inputs=(torch.randn(1, TRACE_SAMPLES, dtype=torch.float32),),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, sample_dim),
                                           dtype=np.float32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="text_embed",
                wrapper=text_phase,
                dummy_inputs=(dummy_ids, dummy_keep),
                mil_inputs=[
                    ct.TensorType(name="ids", shape=(1, frame_dim), dtype=np.int32),
                    ct.TensorType(name="keep", shape=(1, frame_dim), dtype=np.float32),
                ],
                root_axis="n_tokens",
                # **A stream, not a copy.** The driver runs this topology twice per synthesis -- on the
                # real ids and on the dropped ones -- and BOTH results are alive at once: the sampler
                # feeds one to the conditional evaluation and the other to the unconditional one, at
                # every step. One retained buffer would have the second call overwrite the first.
                extra_streams=("text_embed_uncond",),
            ),
            ExportPhase(
                name="estimator",
                wrapper=est_phase,
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.randn(1, TRACE_FRAMES, N_MEL),
                              dummy_text,
                              torch.tensor([0.3])),
                mil_inputs=[
                    ct.TensorType(name="x", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="cond", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="text_embed",
                                   shape=(1, frame_dim, DIT_ARCH["text_dim"]), dtype=np.float32),
                    ct.TensorType(name="t", shape=(1,), dtype=np.float32),
                ],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="vocoder",
                wrapper=F5VocoderPhase(vocos).eval(),
                # FRAME-major in, matching what the estimator retains -- see the wrapper's own note on
                # why the conversion is in this graph rather than in the driver.
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL),),
                mil_inputs=[ct.TensorType(name="mel", shape=(1, frame_dim, N_MEL),
                                           dtype=np.float32)],
                # `n_enc_frames`, not `n_tokens`: this phase counts the ACOUSTIC frames a decoder
                # consumes, and it is a genuinely different number from the estimator's own axis --
                # the reference's frames have been sliced off by the time the vocoder runs, so the two
                # differ by however long the prompt was. axes.py names this quantity.
                root_axis="n_enc_frames",
            ),
        ]

    def samplers(self) -> List[FlowMatchingSpec]:
        return [FlowMatchingSpec(
            func_name="sample_estimator",
            estimator="estimator",
            carried_input="x",
            time_input="t",
            fixed_inputs=["cond", "text_embed"],
            schedule="caller",
            guidance=True,
            # The initial state is the caller's when it supplies one. Not a convenience: torch's RNG
            # and the engine's are different algorithms, so the same seed is not the same noise, and
            # without this the export can only be graded distributionally -- which for a generative
            # model is not a grade at all. Absent a value the engine still draws.
            caller_noise=True,
            note="Euler over F5-TTS's DiT vector field, on the CALLER's sway-reparameterised\n"
                 "schedule and under classifier-free guidance: the estimator runs twice per step,\n"
                 "on the reference conditioning and on a dropped one, and the engine integrates\n"
                 "`v_cond + cfg_scale * (v_cond - v_uncond)`.",
        )]

    def driver_components(self) -> List:
        from .driver_components import (
            CALLER, DriverInputs, DriverReturn, ExportConstants, FlowMatchingSampler, LuaFragment,
            SubgraphCallComponent,
        )
        from .lua_library import LuaLibrary
        from .driver_ir import BinOp, FieldAccess, Len, OutputRef, Var

        fragment = self.driver_script_path
        n_frames, n_gen = Var("n_frames"), Var("n_gen")
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True,
                        defines=("opt_scalar", "sway_times", "f5_text_arrays", "f5_step_cond")),
            ExportConstants(values={
                "SAMPLE_RATE": SAMPLE_RATE,
                "HOP_LENGTH": HOP_LENGTH,
                "N_MEL": N_MEL,
                # `infer_batch_process`'s own four defaults, in one place rather than repeated in the
                # fragments that read them.
                "TARGET_RMS": 0.1,
                "DEFAULT_STEPS": DEFAULT_STEPS,
                "DEFAULT_CFG": DEFAULT_CFG,
                "DEFAULT_SWAY": DEFAULT_SWAY,
                "DEFAULT_SPEED": 1.0,
            }),
            LuaLibrary(uses=("array_slice",)),
            DriverInputs(bindings=(("waveform", CALLER), ("text_ids", CALLER)),
                         n_tokens=Len("waveform")),
            LuaFragment(fragment / "01_reference.lua",
                        reads=("waveform", "TARGET_RMS"),
                        defines=("ref_wave", "rms_gain")),
            SubgraphCallComponent(
                topology="mel", outputs=("cond_mel",), extra_outputs=("cond_shape",),
                length=Len("ref_wave"), inputs={"waveform": Var("ref_wave")},
                note="--- The reference clip's log-mel, frame-major. Its FRAME COUNT is the\n"
                     "    conditioning length, which is why the shape is captured too. ---"),
            LuaFragment(fragment / "02_plan.lua",
                        reads=("cond_shape", "cond_mel", "text_ids", "waveform", "HOP_LENGTH",
                               "N_MEL", "DEFAULT_STEPS", "DEFAULT_CFG", "DEFAULT_SWAY",
                               "DEFAULT_SPEED"),
                        defines=("cond_len", "ref_frames", "n_frames", "n_gen", "ids", "uncond_ids",
                                 "keep", "step_cond", "zero_cond", "times", "cfg_scale")),
            SubgraphCallComponent(
                topology="text_embed", outputs=(), retain=True, length=n_frames,
                inputs={"ids": Var("ids"), "keep": Var("keep")},
                note="--- TextEmbedding on the real ids. RETAINED: the sampler hands it to every\n"
                     "    conditional evaluation and no host arithmetic touches it. ---"),
            SubgraphCallComponent(
                topology="text_embed_uncond", outputs=(), retain=True, length=n_frames,
                inputs={"ids": Var("uncond_ids"), "keep": Var("keep")},
                note="--- The same graph on the DROPPED ids, as its own stream. The mask is still\n"
                     "    the real text's -- see the export's own note on `keep`. ---"),
            FlowMatchingSampler(
                spec=self.samplers()[0], result="_mel", length=n_frames,
                n_elems=BinOp("*", n_frames, Var("N_MEL")),
                n_steps=None, times=Var("times"),
                step_inputs={"cond": Var("step_cond"),
                             "text_embed": OutputRef("text_embed")},
                uncond_inputs={"cond": Var("zero_cond"),
                               "text_embed": OutputRef("text_embed_uncond")},
                guidance_scale=Var("cfg_scale"),
                # `inputs.noise` or nil -- the driver does not draw, it forwards. The engine's own
                # fallback is what runs when a caller names none, which keeps `infer(text)` working.
                state=FieldAccess("inputs", "noise"),
                note="--- The whole spectrogram, in-filled. See sample_estimator above. ---"),
            LuaFragment(fragment / "03_mel_tail.lua",
                        reads=("ref_frames", "cond_len", "n_gen", "step_cond", "N_MEL"),
                        defines=("mel_tail",), retains=("estimator",)),
            SubgraphCallComponent(
                topology="vocoder", outputs=("wave_raw",), length=n_gen,
                inputs={"mel": Var("mel_tail")},
                note="--- Vocos: the generated frames -> waveform. ---"),
            LuaFragment(fragment / "04_output.lua", reads=("wave_raw", "rms_gain"),
                        defines=("waveform_out",)),
            DriverReturn(values=("waveform_out",)),
        ]

    def hparams(self) -> dict:
        # NOT `sample_rate`: `contract()` already writes `loom.sample_rate`, and declaring it here as
        # well made the writer log a duplicate-key overwrite. One fact, one writer.
        return {"n_mel": N_MEL, "hop_length": HOP_LENGTH}

    def contract(self) -> dict:
        contract = super().contract()
        # F5-TTS takes TEXT, not phoneme ids: its vocabulary is a character table and it ships in the
        # GGUF. Same exception Supertonic is, and declaring the task default instead would close the
        # door this model actually has.
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        contract["tts.default_steps"] = DEFAULT_STEPS
        return contract

    def backend_kwargs(self) -> dict:
        return dict(
            flat_namespace=False,
            root_axis=self.root_axis,
            hparams=self.hparams(),
            # Named rather than auto-detected, for the reason "ctc", "funasr" and "supertonic" are:
            # `vocab.txt` is a bare newline-separated list, which is also how several other schemes
            # write themselves down.
            tokenizer_dir=self.model_dir,
            tokenizer_family="f5",
        )


def _is_f5_tts(path: Path) -> bool:
    """An F5-TTS release directory: a `vocab.txt` and an assembled `vocos-mel-24khz/` beside exactly
    one `.safetensors` whose tensor names are the DiT's.

    The name check is the discriminator and it has to be, because the layout alone is not one: a
    `vocab.txt` plus a `.safetensors` is also what several other releases here look like. Reading the
    safetensors HEADER answers it without loading a gigabyte -- the format's first 8 bytes are the
    JSON length, and `transformer.transformer_blocks.0.attn.to_q.weight` is a name no other checkpoint
    in this zoo carries.

    The vocoder is part of the check rather than of the failure: a directory without it is an F5-TTS
    release this export cannot finish, and detection claiming it would turn a missing file into a
    traceback two minutes into a conversion.
    """
    if not path.is_dir() or not (path / "vocab.txt").is_file():
        return False
    vocoder = path / VOCODER_SUBDIR
    if not all((vocoder / n).is_file() for n in ("config.yaml", "pytorch_model.bin")):
        return False
    files = sorted(path.glob("*.safetensors"))
    if len(files) != 1:
        return False
    try:
        import json
        import struct

        with files[0].open("rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if length > (1 << 24):          # a sane header; this format's is kilobytes
                return False
            header = json.loads(f.read(length))
    except (OSError, ValueError, struct.error):
        return False
    names = set(header)
    return any(n.endswith("transformer.transformer_blocks.0.attn.to_q.weight") for n in names) and \
        any("transformer.text_embed.text_embed.weight" in n for n in names)


def _build_f5_tts(path: Path, output_path: str) -> LoomExportConfig:
    return F5TTSExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=F5TTSExportConfig,
        recognizers=[
            ModelRecognizer(name="f5-tts", detect=_is_f5_tts, build_config=_build_f5_tts),
        ],
    ))
