"""tiny-voca (n4ze3m, Apache-2.0): a 4.5M-parameter English TTS with no neural vocoder, exported as four
graphs and the host arithmetic between them (loom.cpp ADR-133).

Upstream (`tiny_voca.model.TinyVoca.synthesize`) for one utterance:

  phonemes + stress + voice -> transformer encoder (4 blocks) -> duration and pause predictor
      -> tokens repeated by their (integer) durations, plus a per-frame position feature
      -> frame transformer A (2 blocks) -> log-F0, voicing, energy  (vowels forced voiced)
      -> frame transformer B (5 blocks) -> per-frame complex cepstra and gains
      -> source-filter head: a band-limited pulse train at the F0 and white noise, each shaped by its
         time-varying filter in the STFT domain -> 24 kHz

The split is where the data forces it:

  encode    ids, stress, voice -> the encoder's states and the duration head's two outputs per token.
  (host)    durations: `round(expm1(clamp(d, 0, log1p 400)) / speed)`, at least 1, zero for a pause the
            presence logit drops -- then each frame's owner token and its position features.
  prosody   owner rows gathered, features and voice added -> transformer A -> prosody head.
  (host)    vowels kept voiced, the pitch range widened, the hard conditioning clamps, the F0.
  acoustic  transformer B -> the harmonic and noise heads' per-frame parameters.
  (host)    the pulse train, in DOUBLE precision (upstream's float64 phase), and the noise draw.
  synth     sub-frame envelopes, cepstrum -> response (DFT matrices), STFT, multiply, inverse STFT.

**The pulse train leaves the graph entirely.** Upstream computes it in float64 ("Float64 keeps the phase
exact"): a log-linear F0 per sample, its running sum, and the closed-form band-limited sum of harmonics.
ggml has no float64, so the driver (Lua numbers are doubles) computes it sample by sample and hands the
graph float32 samples -- the same rounding upstream ends with (`.float()`).

**Every STFT is a convolution with constant kernels**, the construction upstream's own ONNX export uses
(`tiny_voca.export.Exportable`): analysis kernels are the windowed DFT basis, the inverse is a transposed
convolution with the windowed inverse basis, and the window-square envelope is the same transposed
convolution of ones. The crops are static from both ends, so no slice depends on a shape.

**Single utterance, so every mask is all ones** and the wrappers drop them; the rotary tables are built
from the positions the driver passes (`rotary_table` reads `x.shape[1]`, a traced constant). The
interleaved rotation (`x[..., 0::2]`) is a strided slice no lowering takes: it is spelled as a reshape to
pairs and a static index.
"""
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked

DRIVER_DIR = Path(__file__).resolve().parent / "tiny_voca_driver"

TINY_VOCA_MISSING = (
    "tiny-voca's model code is the `tiny_voca` package (github.com/n4ze3m/tiny-voca, Apache-2.0). Put a "
    "clone's `src/` on PYTHONPATH (or LOOM_TINY_VOCA_REPO=<clone>): it needs only torch."
)


def load_tiny_voca(model_dir: Path):
    repo = os.environ.get("LOOM_TINY_VOCA_REPO")
    if repo:
        src = str(Path(repo) / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        from tiny_voca.config import CHECKPOINT_FORMAT, Config
        from tiny_voca.model import TinyVoca
    except ImportError as exc:                       # pragma: no cover - env-dependent
        raise ImportError(TINY_VOCA_MISSING) from exc
    state = torch.load(Path(model_dir) / "tiny-voca.pt", map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"tiny-voca checkpoint format {state.get('format_version')}; this export reads "
                         f"{CHECKPOINT_FORMAT}")
    model = TinyVoca(Config.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state


def _rotate(x, cos, sin):
    """`tiny_voca.model.rotate` on `[heads, T, w]`: pairs (2i, 2i+1) rotated, spelled without a strided
    slice and within four dimensions (the engine's PERMUTE/RESHAPE ceiling)."""
    heads, t, w = x.shape
    pairs = x.reshape(heads, t, w // 2, 2)
    # Slices reshaped back to three axes: an integer index (`pairs[..., 0]`) lowered with its unit axis
    # kept, and the [T, w/2] tables then did not broadcast against it.
    a = pairs[..., 0:1].reshape(heads, t, w // 2)
    b = pairs[..., 1:2].reshape(heads, t, w // 2)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).reshape(heads, t, w)


class _Stack(nn.Module):
    """`tiny_voca.model.Stack` for one unpadded utterance: the same blocks, no masks.

    The rotation's pair spelling leaves many small nodes (copies over 24 channels, a stack over a size-2
    axis). Reordering each head's q/k rows evens-then-odds so it becomes two halves was measured on the
    Pi 4 (2026-11-17): 664 ms against 667 ms per synthesis, the same -- the per-node profile that had
    blamed those nodes for a third of the time was its own per-node overhead. Not worth the reorder."""

    def __init__(self, stack):
        super().__init__()
        self.blocks = stack.blocks
        self.norm = stack.norm
        width = stack.width
        inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
        self.register_buffer("inverse", inverse)

    def forward(self, x, positions):
        angle = positions.unsqueeze(-1) * self.inverse.unsqueeze(0)                # [T, w/2]
        cos, sin = angle.cos(), angle.sin()
        for block in self.blocks:
            att = block.attention
            _, t, c = x.shape
            heads, width = att.heads, c // att.heads
            # One utterance: the batch axis dropped, so no tensor here passes four dimensions. The qkv
            # rows are [3][heads][width], as upstream's view(b, t, 3, heads, width) reads them.
            q, k, v = att.qkv(block.norm1(x))[0].chunk(3, dim=-1)                  # [T, C] each
            q, k, v = (z.reshape(t, heads, width).transpose(0, 1) for z in (q, k, v))
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
            scores = torch.matmul(q * (width ** -0.5), k.transpose(-1, -2))
            y = torch.matmul(torch.softmax(scores, dim=-1), v)                       # [heads, T, w]
            x = x + att.out(y.transpose(0, 1).reshape(t, c)).unsqueeze(0)
            y = block.depthwise(block.norm2(x).transpose(1, 2)).transpose(1, 2)
            x = x + block.project(F.gelu(block.expand(y)))
        return self.norm(x)


def _predictor(p, x):
    x = p.conv1(x.transpose(1, 2)).transpose(1, 2)
    x = p.norm1(F.gelu(x))
    x = p.conv2(x.transpose(1, 2)).transpose(1, 2)
    x = p.norm2(F.gelu(x))
    return p.out(x)


class _EncodeWrapper(nn.Module):
    """`(tokens, stress, voice, positions) -> (states [1,N,C], duration head [N,2])`."""

    def __init__(self, model):
        super().__init__()
        self.tokens, self.stress, self.voices = model.tokens, model.stress, model.voices
        self.encoder = _Stack(model.encoder)
        self.duration = model.duration

    def forward(self, tokens, stress, voice, positions):
        x = self.tokens(tokens) + self.stress(stress) + self.voices(voice).unsqueeze(1)
        hidden = self.encoder(x, positions)
        return hidden, _predictor(self.duration, hidden)[0]


class _ProsodyWrapper(nn.Module):
    """`(states, owner, features, voice, positions) -> (frame states [1,T,C], prosody head [T,3])`: the
    expansion (`TinyVoca.expand`) with the driver's owner index and features, then transformer A."""

    def __init__(self, model):
        super().__init__()
        self.frame_position, self.frame_voices = model.frame_position, model.frame_voices
        self.stack = _Stack(model.prosody_stack)
        self.head = model.prosody_head

    def forward(self, hidden, owner, features, voice, positions):
        expanded = hidden[0].index_select(0, owner).unsqueeze(0)
        expanded = expanded + self.frame_position(features).unsqueeze(0) + self.frame_voices(voice).unsqueeze(1)
        states = self.stack(expanded, positions)
        return states, self.head(states)[0]


class _AcousticWrapper(nn.Module):
    """`(frame states, condition [T,3], positions) -> (harmonic head [T,H], noise head [T,M])`."""

    def __init__(self, model):
        super().__init__()
        self.projection = model.prosody_projection
        self.stack = _Stack(model.acoustic_stack)
        self.harmonic, self.noise = model.vocoder.harmonic, model.vocoder.noise

    def forward(self, states, condition, positions):
        acoustic = self.stack(states + self.projection(condition).unsqueeze(0), positions)[0]
        return self.harmonic(acoustic), self.noise(acoustic)


def _cepstrum_matrices(positions, fft):
    buffer = torch.zeros(len(positions), fft, dtype=torch.float64)
    buffer[torch.arange(len(positions)), torch.tensor(positions)] = 1
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return spectrum.real.float(), spectrum.imag.float()


def _analysis_kernels(window):
    fft = len(window)
    spectrum = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
    return spectrum.real.T[:, None].float().contiguous(), spectrum.imag.T[:, None].float().contiguous()


def _synthesis_kernel(window):
    fft = len(window)
    bins = fft // 2 + 1
    unit = torch.eye(bins, dtype=torch.complex128)
    real = torch.fft.irfft(unit, n=fft, dim=-1)
    imaginary = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
    return (torch.cat((real, imaginary)) * window.double())[:, None].float().contiguous()


class _SynthWrapper(nn.Module):
    """`(heads, sub-frame indices, pulses, noise) -> waveform`: `SourceFilter.forward` after its linear
    heads, for one utterance. Index inputs are the driver's (`_subframes`), so nothing here reads a shape."""

    def __init__(self, vocoder, config):
        super().__init__()
        c = config
        self.cep_causal, self.cep_anti, self.noise_cep = c.cep_causal, c.cep_anti, c.noise_cep
        self.synth_fft, self.synth_hop, self.noise_fft, self.noise_hop = c.synth_fft, c.synth_hop, c.noise_fft, c.noise_hop
        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        for name, (real, imaginary) in (("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
                                        ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft))):
            self.register_buffer(f"{name}_real", real)
            self.register_buffer(f"{name}_imaginary", imaginary)
        for name, window in (("synth", vocoder.synth_window), ("noise", vocoder.noise_window)):
            real, imaginary = _analysis_kernels(window)
            self.register_buffer(f"{name}_analysis_real", real)
            self.register_buffer(f"{name}_analysis_imaginary", imaginary)
            self.register_buffer(f"{name}_synthesis", _synthesis_kernel(window))
            self.register_buffer(f"{name}_envelope", (window.double() ** 2).float()[None, None].contiguous())
        self.register_buffer("lifter", torch.cat((vocoder.causal_lifter, vocoder.anti_lifter)).float())
        self.register_buffer("noise_lifter", vocoder.noise_lifter.float())

    def _filter(self, source, real, imaginary, name, fft, hop):
        # `filtered(F.pad(source, (0, fft//2)), ...)` with torch.stft(center=True) = a constant pad of
        # fft//2 in front and fft (the centring half plus that trailing half) behind.
        padded = F.pad(source.unsqueeze(1), (fft // 2, fft))
        s_real = F.conv1d(padded, getattr(self, f"{name}_analysis_real"), stride=hop)
        s_imaginary = F.conv1d(padded, getattr(self, f"{name}_analysis_imaginary"), stride=hop)
        real, imaginary = real.t().unsqueeze(0), imaginary.t().unsqueeze(0)
        product = torch.cat((s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real), dim=1)
        wave = F.conv_transpose1d(product, getattr(self, f"{name}_synthesis"), stride=hop)
        envelope = F.conv_transpose1d(s_real[:, :1] * 0.0 + 1.0, getattr(self, f"{name}_envelope"), stride=hop)
        # The transposed convolution makes S + fft//2 + fft samples; upstream keeps [fft//2, fft//2 + S).
        return (wave / envelope)[0, 0, fft // 2: -fft]

    @staticmethod
    def _response(cepstrum, real, imaginary):
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def forward(self, h, n, here, following, h_sub, n_owner, n_sub, pulses, noise):
        split = self.cep_causal + self.cep_anti
        cep = h[:, :split]
        envelope = 0.5 * (cep.index_select(0, here) + cep.index_select(0, following)) * self.lifter
        gain = h[:, split:].reshape(-1, 1).index_select(0, h_sub)
        envelope = torch.cat((envelope[:, :1] + gain, envelope[:, 1:]), dim=1)
        h_real, h_imag = self._response(envelope, self.harmonic_real, self.harmonic_imaginary)
        shape = n[:, : self.noise_cep].index_select(0, n_owner) * self.noise_lifter
        n_gain = n[:, self.noise_cep:].reshape(-1, 1).index_select(0, n_sub)
        shape = torch.cat((shape[:, :1] + n_gain, shape[:, 1:]), dim=1)
        n_real, n_imag = self._response(shape, self.noise_real, self.noise_imaginary)
        return (self._filter(pulses, h_real, h_imag, "synth", self.synth_fft, self.synth_hop)
                + self._filter(noise, n_real, n_imag, "noise", self.noise_fft, self.noise_hop)).unsqueeze(0)


@dataclass(kw_only=True)
class TinyVocaExportConfig(BaseMultiPhaseModelExportConfig):
    """An `n4ze3m/tiny-voca` directory (`tiny-voca.pt` + `config.json`) -> one Loom GGUF."""

    architecture: str = "tiny-voca"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = DRIVER_DIR
    trace_tokens: int = 13
    trace_frames: int = 37
    _config: Optional[object] = field(default=None, init=False, repr=False)
    _symbols: Optional[list] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string"),
        "model_dir": Unchecked("path to the checkpoint; the recognizer found tiny-voca.pt in it"),
        "root_axis": Unchecked("encode's token axis"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "driver_script_path": Unchecked("hand-written fragments, checked against the topologies"),
        "trace_tokens": Unchecked("the concrete N the trace runs at"),
        "trace_frames": Unchecked("the concrete T the trace runs at"),
        "_config": Unchecked("read off the checkpoint in phases()"),
        "_symbols": Unchecked("upstream's SYMBOLS, read in phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model, _ = load_tiny_voca(Path(self.model_dir))      # puts the clone on sys.path first
        from tiny_voca.text import SYMBOLS
        c = model.config
        self._config, self._symbols = c, list(SYMBOLS)
        if len(SYMBOLS) != c.vocab_size:
            raise ValueError(f"tiny-voca: {c.vocab_size} embedding rows, {len(SYMBOLS)} symbols")
        n, t = self.trace_tokens, self.trace_frames
        width = c.channels
        tokens_axis = ct.RangeDim(2, c.max_tokens)
        frames_axis = ct.RangeDim(2, c.max_frames)
        hsub_axis = ct.RangeDim(7, 2 * c.max_frames + 3)
        nsub_axis = ct.RangeDim(11, 4 * c.max_frames + 3)
        samples_axis = ct.RangeDim(2 * c.hop, c.max_frames * c.hop)
        g = torch.Generator().manual_seed(0)
        h_width = c.cep_causal + c.cep_anti + c.hop // c.synth_hop
        n_width = c.noise_cep + c.hop // c.noise_hop
        steps_h, steps_n = c.hop // c.synth_hop, c.hop // c.noise_hop
        sub_h = torch.minimum(torch.arange(steps_h * t + 3), torch.tensor(steps_h * t - 1))
        sub_n = torch.minimum(torch.arange(steps_n * t + 3), torch.tensor(steps_n * t - 1))
        return [
            ExportPhase(
                name="encode", wrapper=_EncodeWrapper(model).eval(),
                dummy_inputs=(torch.randint(5, c.vocab_size, (1, n), generator=g), torch.randint(0, 3, (1, n), generator=g),
                              torch.zeros(1, dtype=torch.long), torch.arange(n, dtype=torch.float32)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, tokens_axis), dtype=np.int32),
                            ct.TensorType(name="stress", shape=(1, tokens_axis), dtype=np.int32),
                            ct.TensorType(name="voice", shape=(1,), dtype=np.int32),
                            ct.TensorType(name="positions", shape=(tokens_axis,), dtype=np.float32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="prosody", wrapper=_ProsodyWrapper(model).eval(),
                dummy_inputs=(torch.randn(1, n, width, generator=g), torch.randint(0, n, (t,), generator=g),
                              torch.rand(t, 3, generator=g), torch.zeros(1, dtype=torch.long),
                              torch.arange(t, dtype=torch.float32)),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, tokens_axis, width), dtype=np.float32),
                            ct.TensorType(name="owner", shape=(frames_axis,), dtype=np.int32),
                            ct.TensorType(name="features", shape=(frames_axis, 3), dtype=np.float32),
                            ct.TensorType(name="voice", shape=(1,), dtype=np.int32),
                            ct.TensorType(name="positions", shape=(frames_axis,), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"hidden": {1: "n_tokens"}},
            ),
            ExportPhase(
                name="acoustic", wrapper=_AcousticWrapper(model).eval(),
                dummy_inputs=(torch.randn(1, t, width, generator=g), torch.randn(t, 3, generator=g),
                              torch.arange(t, dtype=torch.float32)),
                mil_inputs=[ct.TensorType(name="states", shape=(1, frames_axis, width), dtype=np.float32),
                            ct.TensorType(name="condition", shape=(frames_axis, 3), dtype=np.float32),
                            ct.TensorType(name="positions", shape=(frames_axis,), dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="synth", wrapper=_SynthWrapper(model.vocoder, c).eval(),
                dummy_inputs=(torch.randn(t, h_width, generator=g) * 0.1, torch.randn(t, n_width, generator=g) * 0.1,
                              sub_h // steps_h, torch.minimum((sub_h + 1) // steps_h, torch.tensor(t - 1)), sub_h,
                              sub_n // steps_n, sub_n,
                              torch.randn(1, t * c.hop, generator=g), torch.randn(1, t * c.hop, generator=g)),
                mil_inputs=[ct.TensorType(name="h", shape=(frames_axis, h_width), dtype=np.float32),
                            ct.TensorType(name="n", shape=(frames_axis, n_width), dtype=np.float32),
                            ct.TensorType(name="here", shape=(hsub_axis,), dtype=np.int32),
                            ct.TensorType(name="following", shape=(hsub_axis,), dtype=np.int32),
                            ct.TensorType(name="h_sub", shape=(hsub_axis,), dtype=np.int32),
                            ct.TensorType(name="n_owner", shape=(nsub_axis,), dtype=np.int32),
                            ct.TensorType(name="n_sub", shape=(nsub_axis,), dtype=np.int32),
                            ct.TensorType(name="pulses", shape=(1, samples_axis), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(1, samples_axis), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"here": {0: f"{steps_h}*n_enc_frames+3"},
                               "following": {0: f"{steps_h}*n_enc_frames+3"},
                               "h_sub": {0: f"{steps_h}*n_enc_frames+3"},
                               "n_owner": {0: f"{steps_n}*n_enc_frames+3"},
                               "n_sub": {0: f"{steps_n}*n_enc_frames+3"},
                               "pulses": {1: "n_samples"}, "noise": {1: "n_samples"}},
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import DriverReturn, ExportConstants, LuaFragment
        from .lua_library import LuaLibrary

        return [
            LuaLibrary(uses=("round_half_to_even", "to_f32", "aten_randn")),
            ExportConstants(values=self._constants(), top_level=True),
            LuaFragment(DRIVER_DIR / "00_voca.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish")),
            LuaFragment(DRIVER_DIR / "01_generate.lua", defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def _constants(self) -> dict:
        c, symbols = self._config, self._symbols
        if c is None:
            return {}
        from tiny_voca.text import PHONE_IDS, SYMBOL_TO_ID, VOWELS

        is_phone = [1 if i in PHONE_IDS else 0 for i in range(len(symbols))]
        is_vowel = [1 if symbols[i] in VOWELS else 0 for i in range(len(symbols))]
        return {
            "VOCAB": len(symbols), "HOP": c.hop, "SAMPLE_RATE": c.sample_rate, "SYNTH_HOP": c.synth_hop,
            "NOISE_HOP": c.noise_hop, "PITCH_REFERENCE_HZ": c.pitch_reference_hz, "F0_MIN": c.f0_min,
            "F0_MAX": c.f0_max, "HARMONIC_LIMIT": c.harmonic_limit, "MAX_TOKENS": c.max_tokens,
            "MAX_FRAMES": c.max_frames, "PEAK": 0.99,
            "PAD_ID": SYMBOL_TO_ID["<pad>"], "BOS_ID": SYMBOL_TO_ID["<bos>"], "EOS_ID": SYMBOL_TO_ID["<eos>"],
            "UNK_ID": SYMBOL_TO_ID["<unk>"], "PERIOD_ID": SYMBOL_TO_ID["."],
            # 0-based id -> flag, as Lua tables keyed by id (`IS_PHONE[id]`).
            "IS_PHONE_LIST": is_phone, "IS_VOWEL_LIST": is_vowel,
        }

    def phoneme_table(self) -> dict:
        """Upstream's 76 symbols, then every phone again with primary (`ˈ`) and secondary (`ˌ`) stress at
        `VOCAB + k` and `2 VOCAB + k`: a dizione string carries stress as a mark before the phone, and
        longest match then finds the marked entry. The driver splits the id back into (k, stress). No
        assembly: dizione's own `<bos>`/`<eos>` are in the string."""
        symbols = list(self._symbols or [])
        from tiny_voca.text import PHONE_IDS

        v = len(symbols)
        table, ids = list(symbols), list(range(v))
        for mark, base in (("ˈ", v), ("ˌ", 2 * v)):
            for k in sorted(PHONE_IDS):
                table.append(mark + symbols[k])
                ids.append(base + k)
        return {"symbols": table, "ids": ids, "bos": -1, "eos": -1, "blank": -1, "interleave_blank": False}

    def hparams(self) -> dict:
        return {"tts.phoneme_style": "dizione"}

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "phonemes"
        contract["sample_rate"] = int(self._config.sample_rate) if self._config else 24000
        contract["text.phoneme_alphabet"] = "ipa"
        contract["text.languages"] = ["en-US"]
        return contract


def _is_tiny_voca(path: Path) -> bool:
    return path.is_dir() and (path / "tiny-voca.pt").is_file() and (path / "config.json").is_file()


def _build_tiny_voca(path: Path, output_path: str) -> TinyVocaExportConfig:
    return TinyVocaExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech", config_class=TinyVocaExportConfig,
        recognizers=[ModelRecognizer(name="tiny-voca", detect=_is_tiny_voca, build_config=_build_tiny_voca)],
    ))
