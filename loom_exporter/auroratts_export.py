"""AuroraTTS-100K (`AuroraAI-Research/AuroraTTS-100K`, Apache-2.0): a 95,638-parameter English TTS with no
neural vocoder, exported as four graphs, a BiGRU swept by the engine, and the host arithmetic between them.

Upstream is one `TinyTTS` module (`modeling_auroratts.py`, transformers remote code) whose `infer` runs:

  ids -> embedding -> a conv-GLU/FFN block applied 4x -> BiGRU context -> duration + pitch predictors
      -> durations rounded on the host -> Gaussian soft upsampling -> dilated depthwise decoder
      -> per-frame log-F0, voicing, a 40-band harmonic and a 24-band noise envelope
      -> HNSynth: 80 harmonics + STFT-shaped white noise + a 65-tap FIR -> waveform at 22.05 kHz.

The split is where the data forces it:

  encode    ids -> the encoder's output, and the BiGRU's normalised input, one row per token.
  ctx_gru   the BiGRU, as two per-timestep cells the engine sweeps (`RecurrentPhase`): a traced
            `nn.GRU` reaches MIL as an unrolled loop, so the cells are read off the module.
  predict   residual + GRU -> log-durations (to the host) and the pitch-conditioned encoder output.
  (host)    durations: `round((exp(logd) - 1) / speed)`, at least 1, then the Gaussian centres.
  decode    upsampling, decoder and heads -> f0 (to the host), voicing logit and the two envelopes.
  (host)    the harmonic phase at each frame boundary, in double precision, and the noise draw.
  synth     the DSP: harmonics, shaped noise, FIR.

**The phase is the one thing that has to leave the graph.** Upstream accumulates `f0 * 2*pi / sr` with a
float64 cumsum over every sample ("to avoid drift over long utterances"); at float32 the error at 5 s
is ~1e-3 rad on the fundamental and 80x that on the top harmonic. ggml has no float64, so the driver
(Lua numbers are doubles) carries the phase from frame to frame and hands the graph each frame's start,
already reduced mod 2*pi. Inside a frame the sample-rate f0 is a straight line, so its running sum is a
closed-form quadratic -- a transposed convolution of the frame-rate f0, below.

**Every per-sample interpolation is a transposed convolution.** Upstream's `F.interpolate(..., size=L+1,
mode="linear", align_corners=True)[..., :L]` over `T` frames samples frame `s / 256` at sample `s`, i.e.
a lerp between frames `j` and `j+1`. A stride-256 transposed convolution with a 512-tap kernel and
padding 256 emits exactly `L = (T-1)*256` samples, each the sum of frame `j` times `kernel[r+256]` and
frame `j+1` times `kernel[r]` (`r = s - 256j`): the lerp with the triangle, the phase's quadratic with
its two halves, and a per-frame constant with a box. No slicing of a dynamic axis anywhere.

**The harmonic envelope lookup is a hat-function sum, not a gather.** Upstream reads `logAh` at the
fractional band `pos = mel(h*f0)` by `gather` + lerp; `sum_k logAh[k] * relu(1 - |pos - k|)` is the
same number for every `pos` in `[0, kh-1]` (the clamp guarantees it) without a data-dependent index.

Noise: upstream draws `torch.randn(1, L)` from the global generator, so `torch.manual_seed(s)` followed
by `synthesize` renders exactly what this driver renders with `seed = s` (`aten_randn`, ATen's own CPU
stream). Batch size is 1 and every mask in upstream is all ones for one unpadded utterance, so the
wrappers drop them.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase
from .spec_protocol import Unchecked

DRIVER_DIR = Path(__file__).resolve().parent / "auroratts_driver"

# The symbol table `modeling_auroratts.py` builds (`SYMBOLS`), in id order: padding, silence, the 24
# ARPAbet consonants, the 15 vowels x stress 0/1/2, then punctuation and the word boundary. Restated
# rather than imported because upstream ships it as remote code; `phases()` checks the embedding's row
# count against it. ONE spelling differs: the word boundary is `|` here where upstream's list has a
# space (id 71). The ids are upstream's; the spelling is how a caller writes a phoneme string, and
# ARPAbet phonemes are written space-separated (CMUdict, g2p_en's output joined) -- with the boundary
# spelled as a space too, `D HH` and `DH` would read the same. The engine skips the separating spaces
# (no symbol holds one), so `"DH AH0 | K W IH1 K"` encodes to upstream's ids.
_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
SYMBOLS = (["<pad>", "<sil>"] + _CONSONANTS + [v + s for v in _VOWELS for s in "012"]
           + ["|", ",", ".", "?", "!", ";", ":", "-"])
SIL_ID = 1

HOP = 256


def load_tiny_tts(model_dir: Path):
    """`(TinyTTS, Config)` from the checkpoint directory: upstream's own module, imported from the
    directory's `modeling_auroratts.py` (it is remote code, not a package), with `model.safetensors`
    loaded strictly. The DSP's three buffers are not persistent -- `HNSynth.__init__` rebuilds them,
    which is all `AuroraTTSModel._ensure_buffers` does too."""
    import importlib.util
    import sys
    import types

    from safetensors.torch import load_file

    package = "_loom_auroratts"
    if f"{package}.modeling_auroratts" not in sys.modules:
        pkg = types.ModuleType(package)
        pkg.__path__ = [str(model_dir)]
        sys.modules[package] = pkg
        for name in ("configuration_auroratts", "modeling_auroratts"):
            spec = importlib.util.spec_from_file_location(f"{package}.{name}", model_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    upstream = sys.modules[f"{package}.modeling_auroratts"]
    config = upstream.Config.from_dict(json.loads((model_dir / "config.json").read_text()))
    net = upstream.TinyTTS(config).eval()
    state = {k[len("net."):]: v for k, v in load_file(str(model_dir / "model.safetensors")).items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("dsp.") or k.split(".")[-1] not in ("harm", "win", "interp")
                         for k in missing):
        raise ValueError(f"AuroraTTS: unexpected={unexpected} missing={missing}")
    return net, config


def _cln(norm, x):
    """`CLN.forward`: LayerNorm over the channels of a `(1, C, T)` tensor."""
    return F.layer_norm(x.transpose(1, 2), (x.shape[1],), norm.w, norm.b).transpose(1, 2)


def _var_pred(vp, x):
    """`VarPred.forward` with its all-ones mask dropped."""
    y = vp.dw(_cln(vp.n, x))
    y = F.relu(vp.p1(y))
    y = F.relu(vp.p2(y))
    return vp.o(y)                                                     # (1, 1, N)


class _EncodeWrapper(nn.Module):
    """`TinyTTS.encode` up to the BiGRU: the embedding and `LoopedEncoder` (one block, its norms per
    pass), then the GRU's input -- `ContextGRU.norm(x)` -- emitted one row per token, the layout the
    engine's sweep reads. The encoder output is the GRU's residual."""

    def __init__(self, net):
        super().__init__()
        self.emb, self.enc, self.ctx_norm = net.emb, net.enc, net.ctx.norm

    def forward(self, tokens):
        x = self.emb(tokens).transpose(1, 2)                           # (1, d, N)
        enc = self.enc
        for i in range(enc.loops):
            x = x + F.glu(enc.glu(enc.dw(_cln(enc.n1[i], x))), dim=1)
            x = x + enc.ffn(_cln(enc.n2[i], x))
        rows = F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.ctx_norm.w, self.ctx_norm.b)
        return x, rows                                                 # (1, d, N), (1, N, d)


class _PredictWrapper(nn.Module):
    """The rest of `encode` and the pitch conditioning of `infer`: the GRU's residual, both
    predictors, and `x + pit_emb(pit)`."""

    def __init__(self, net):
        super().__init__()
        self.dur_pred, self.pit_pred, self.pit_emb = net.dur_pred, net.pit_pred, net.pit_emb

    def forward(self, x, gru_rows):
        x = x + gru_rows.transpose(1, 2)
        logd = _var_pred(self.dur_pred, x)
        pit = _var_pred(self.pit_pred, x)
        return logd.reshape(-1), x + self.pit_emb(pit)                 # (N,), (1, d, N)


class _DecodeWrapper(nn.Module):
    """`gaussian_upsample`, `Decoder` and `TinyTTS.heads`, then f0 as `infer` computes it. The host
    supplies the durations (`dur`, already at least 1, so `dsafe` is `dur`), their Gaussian centres and
    the frame positions `t + 0.5`."""

    def __init__(self, net, config):
        super().__init__()
        self.dec, self.head = net.dec, net.head
        self.kh = config.kh
        self.f0_mean, self.f0_std = float(config.f0_mean), float(config.f0_std)
        with torch.no_grad():
            # `2 * sigma ** 2` with sigma = softplus(sigma_raw), formed in float32 as upstream forms it.
            sigma = F.softplus(net.sigma_raw.detach().float())
            self.register_buffer("two_sigma_sq", (2 * sigma ** 2).reshape(1))

    def forward(self, x, centers, dur, frame_pos, pitch_shift):
        # `t - centre` for every (frame, token). ggml broadcasts only an op's SECOND operand, so the
        # outer difference starts from the frames tiled across the tokens -- a K=1 product with ones,
        # exact -- rather than from two one-sided broadcasts.
        frames = torch.matmul(frame_pos[None, :, None], torch.ones_like(centers)[:, None, :])
        diff = frames - centers[:, None, :]                            # (1, T, N)
        w = (-(diff ** 2) / self.two_sigma_sq).softmax(-1)
        up = torch.bmm(x, w.transpose(1, 2))                           # (1, d, T)
        pos = (w * (diff / dur[:, None, :])).sum(-1)
        ld = (w * torch.log(dur)[:, None, :]).sum(-1)
        h = torch.cat([up, pos[:, None], ld[:, None]], 1)
        dec = self.dec
        h = dec.inp(h)
        for i in range(dec.L):
            h = h + dec.ffn[i % dec.G](_cln(dec.ln[i], dec.dw[i](h)))
        o = self.head(_cln(dec.out_ln, h))                             # (1, 2 + kh + kn, T)
        f0 = torch.exp(o[:, 0] * self.f0_std + self.f0_mean + pitch_shift).clamp(60, 600)
        log_ah = o[:, 2:2 + self.kh].clamp(-14, 2)
        log_an = o[:, 2 + self.kh:].clamp(-14, 2)
        return f0, o[:, 1], log_ah, log_an                             # (1,T), (1,T), (1,kh,T), (1,kn,T)


def _frame_kernels() -> dict:
    """The three 512-tap kernels of the per-sample interpolations (module docstring), float64 values
    rounded once. `r = m` in the first half weights frame `j+1`, `r = m - 256` in the second weights
    frame `j`: a sample `s = 256j + r` sees `in[j] * k[r+256] + in[j+1] * k[r]`."""
    r = np.arange(HOP, dtype=np.float64)
    quad = r * (r + 1) / (2 * HOP)              # sum_{i<=r} i/256: the f0 slope's share of the phase
    return {
        "lerp": np.concatenate([r / HOP, 1 - r / HOP]),
        "ramp": np.concatenate([r / HOP, np.zeros(HOP)]),         # the lerp's weight on frame j+1
        "phase": np.concatenate([quad, (r + 1) - quad]),
        "box_next": np.concatenate([np.ones(HOP), np.zeros(HOP)]),
    }


class _SynthWrapper(nn.Module):
    """`HNSynth.forward`, frame-rate in and sample-rate out, with the phase's frame starts from the
    host (`phase0[j+1]` = the phase at the first sample of frame `j`, mod 2*pi; `phase0[0]` is unread)
    and the noise drawn by the host."""

    def __init__(self, dsp, config):
        super().__init__()
        self.kh, self.sr, self.n_fft, self.taps = config.kh, config.sr, config.n_fft, config.fir_taps
        self.nyq_mel = float(dsp.nyq_mel)
        kernels = _frame_kernels()
        for name in ("phase", "box_next", "ramp"):
            self.register_buffer(f"k_{name}", torch.tensor(kernels[name], dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("harm", dsp.harm.detach().clone().view(1, -1, 1))
        self.register_buffer("theta", dsp.theta.detach().clone().view(1, -1, 1))
        self.register_buffer("bands", torch.arange(config.kh, dtype=torch.float32).view(1, 1, -1, 1))
        self.register_buffer("interp", dsp.interp.detach().clone())   # (kn, bins)
        self.register_buffer("window", dsp.win.detach().clone())
        self.register_buffer("fir", dsp.fir.detach().clone().view(1, 1, -1))
        from .istft import ISTFT
        self.istft = ISTFT(config.n_fft, config.hop, center=True)
        # ISTFT bakes its own periodic Hann into the synthesis kernels, formed at f64; HNSynth's is
        # `torch.hann_window` at f32, which rounds to within 2e-7 of it. Anything further is another window.
        if not torch.allclose(self.istft.window.double(), dsp.win.detach().double(), atol=1e-6, rtol=0):
            raise ValueError("AuroraTTS: HNSynth's window is not the periodic Hann ISTFT synthesises with")

    @staticmethod
    def _per_sample(frames, kernel):
        """`(1, C, T)` frame values -> `(1, C, (T-1)*256)` samples through one of the kernels above.
        The padding is a crop of `HOP` samples off each end, written as the slice it is: the exporter
        folds a depthwise transposed convolution only with no padding of its own."""
        full = F.conv_transpose1d(frames, kernel, stride=HOP, groups=frames.shape[1])   # (T+1)*256
        return full[:, :, HOP:-HOP]

    @staticmethod
    def _lerp(frames, ramp):
        """The `lerp` kernel's result for many channels at once without a kernel per channel (80 x 512
        floats, a quarter of the file): each sample's frame `j` and `j+1` values are the
        nearest-neighbour upsampling read 0 and 256 samples on, and `ramp` is the weight on `j+1`."""
        held = F.interpolate(frames, scale_factor=HOP, mode="nearest")          # (1, C, T*256)
        here, after = held[:, :, :-HOP], held[:, :, HOP:]
        return here + (after - here) * ramp

    def forward(self, f0, vuv, log_ah, log_an, phase0, noise):
        voiced = (vuv > 0).float()                       # sigmoid(vuv) > 0.5
        # Outer products as K=1 matmuls, exact, for the reason `_DecodeWrapper` gives.
        freqs = torch.matmul(self.harm, f0[:, None, :])                         # (1, H, T)
        alive = (freqs < 0.95 * self.sr / 2).float()
        mel = 2595.0 * torch.log10(1.0 + freqs / 700.0)
        pos = (mel / self.nyq_mel).clamp(0, 1) * (self.kh - 1)
        hat = F.relu(1 - (pos[:, :, None, :].repeat(1, 1, self.kh, 1) - self.bands).abs())  # (1, H, kh, T)
        envelope = (hat * log_ah[:, None, :, :]).sum(2)                          # (1, H, T)
        amp = envelope.exp() * alive * voiced[:, None, :]
        amp = self._lerp(amp, self._per_sample(torch.ones_like(f0)[:, None], self.k_ramp))   # (1, H, L)
        phase = (self._per_sample(phase0[:, None], self.k_box_next)
                 + self._per_sample(f0[:, None], self.k_phase) * (2 * math.pi / self.sr))[:, 0]  # (1, L)
        phase = phase - (2 * math.pi) * torch.floor(phase / (2 * math.pi))
        harmonic = (amp * torch.sin(torch.matmul(self.harm, phase[:, None, :]) + self.theta)).sum(1)
        spec = torch.stft(noise, self.n_fft, HOP, window=self.window, center=True, return_complex=True)
        mag = torch.matmul(log_an.transpose(1, 2), self.interp).exp().transpose(1, 2)   # (1, bins, T)
        shaped = self.istft(spec.real * mag, spec.imag * mag)                   # (1, L)
        y = harmonic + shaped
        return F.conv1d(y[:, None], self.fir, padding=self.taps // 2)[:, 0].clamp(-1, 1).reshape(-1)


@dataclass(kw_only=True)
class TTSAuroraExportConfig(BaseMultiPhaseModelExportConfig):
    """One AuroraTTS checkpoint directory -> one GGUF."""

    model_dir: str
    architecture: str = "auroratts"
    driver_script_path: Path = DRIVER_DIR
    trace_tokens: int = 24
    trace_frames: int = 96
    max_tokens: int = 2048
    max_frames: int = 65536
    config: Optional[object] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked("the checkpoint directory; `phases()` loads it strictly"),
        "driver_script_path": Unchecked("the directory of this family's Lua fragments"),
        "trace_tokens": Unchecked("the token count the phases are traced at; every axis is dynamic"),
        "trace_frames": Unchecked("same, for the frame axis"),
        "max_tokens": Unchecked("the RangeDim ceiling on the token axis"),
        "max_frames": Unchecked("the RangeDim ceiling on the frame axis"),
        "config": Unchecked("READ off the checkpoint by phases()"),
    }

    def phases(self) -> list:
        import coremltools as ct

        net, config = load_tiny_tts(Path(self.model_dir))
        self.config = config
        if net.emb.num_embeddings != len(SYMBOLS):
            raise ValueError(f"AuroraTTS: {net.emb.num_embeddings} embedding rows; the table has {len(SYMBOLS)}")
        if config.hop != HOP:
            raise ValueError(f"AuroraTTS: hop {config.hop}; this export's interpolation kernels are built for {HOP}")
        n, t = int(self.trace_tokens), int(self.trace_frames)
        d = config.d
        tokens_axis, frames_axis = ct.RangeDim(1, self.max_tokens), ct.RangeDim(2, self.max_frames)
        samples_axis = ct.RangeDim(HOP, (self.max_frames - 1) * HOP)
        g = torch.Generator().manual_seed(0)
        dur = torch.full((1, n), float(t) / n)
        centers = dur.cumsum(1) - dur / 2
        return [
            ExportPhase(
                name="encode", wrapper=_EncodeWrapper(net).eval(),
                dummy_inputs=(torch.randint(1, len(SYMBOLS), (1, n), generator=g),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, tokens_axis), dtype=np.int32)],
                root_axis="n_tokens",
            ),
            RecurrentPhase(name="ctx_gru", module=net.ctx.gru),
            ExportPhase(
                name="predict", wrapper=_PredictWrapper(net).eval(),
                dummy_inputs=(torch.randn(1, d, n, generator=g), torch.randn(1, n, d, generator=g)),
                mil_inputs=[ct.TensorType(name="x", shape=(1, d, tokens_axis), dtype=np.float32),
                            ct.TensorType(name="gru", shape=(1, tokens_axis, d), dtype=np.float32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="decode", wrapper=_DecodeWrapper(net, config).eval(),
                dummy_inputs=(torch.randn(1, d, n, generator=g), centers, dur,
                              torch.arange(t, dtype=torch.float32) + 0.5, torch.zeros(1)),
                mil_inputs=[ct.TensorType(name="x", shape=(1, d, tokens_axis), dtype=np.float32),
                            ct.TensorType(name="centers", shape=(1, tokens_axis), dtype=np.float32),
                            ct.TensorType(name="dur", shape=(1, tokens_axis), dtype=np.float32),
                            ct.TensorType(name="frame_pos", shape=(frames_axis,), dtype=np.float32),
                            ct.TensorType(name="pitch_shift", shape=(1,), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"x": {2: "n_tokens"}, "centers": {1: "n_tokens"}, "dur": {1: "n_tokens"}},
            ),
            ExportPhase(
                name="synth", wrapper=_SynthWrapper(net.dsp, config).eval(),
                dummy_inputs=(torch.rand(1, t, generator=g) * 200 + 80, torch.randn(1, t, generator=g),
                              torch.randn(1, config.kh, t, generator=g) - 5,
                              torch.randn(1, config.kn, t, generator=g) - 5,
                              torch.rand(1, t, generator=g) * 6, torch.randn(1, (t - 1) * HOP, generator=g)),
                mil_inputs=[ct.TensorType(name="f0", shape=(1, frames_axis), dtype=np.float32),
                            ct.TensorType(name="vuv", shape=(1, frames_axis), dtype=np.float32),
                            ct.TensorType(name="log_ah", shape=(1, config.kh, frames_axis), dtype=np.float32),
                            ct.TensorType(name="log_an", shape=(1, config.kn, frames_axis), dtype=np.float32),
                            ct.TensorType(name="phase0", shape=(1, frames_axis), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(1, samples_axis), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"noise": {1: "n_samples"}},
            ),
        ]

    def hparams(self) -> dict:
        """The rate, and the conventions of the phoneme table: ARPAbet with stress digits, g2p_en's
        (loom.cpp ADR-071's `tts.phoneme_style`; loom-py folds the text door's IPA to it)."""
        return {"sample_rate": int(self.config.sr) if self.config else 22050,
                "tts.phoneme_style": "arpabet"}

    def contract(self) -> dict:
        contract = super().contract()
        contract["sample_rate"] = self.hparams()["sample_rate"]
        contract["text.frontend"] = "phonemes"
        contract["text.phoneme_alphabet"] = "ipa"
        contract["text.languages"] = ["en-US"]
        return contract

    def phoneme_table(self) -> dict:
        """Upstream's `text_to_ids` frames a sentence `[<sil>, p1, ..., pn, <sil>]`; the engine's
        assembly writes the same with `<sil>` as both BOS and EOS."""
        return {"symbols": list(SYMBOLS), "ids": list(range(len(SYMBOLS))),
                "bos": SIL_ID, "eos": SIL_ID, "blank": -1, "interleave_blank": False}

    def driver_input_aliases(self) -> dict:
        return {"token_ids": "tokens"}

    def driver_components(self) -> List:
        from .driver_components import (
            BiRecurrentCall, DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import BinOp, Lit, OutputRef, Var
        from .lua_library import LuaLibrary

        config = self.config
        d = int(config.d) if config else 56
        gru_h = int(config.gru_h) if config else 28
        fragment = self.driver_script_path
        n, frames = Var("_n"), Var("_n_frames")
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            LuaLibrary(uses=("round_half_to_even", "to_f32", "aten_randn")),
            ExportConstants(values={"HOP": HOP, "SAMPLE_RATE": int(config.sr) if config else 22050}),
            LuaFragment(fragment / "01_tokens.lua", defines=("_n", "_tokens")),
            SubgraphCallComponent(
                topology="encode", outputs=(), retain=True, length=n, inputs={"tokens": Var("_tokens")},
                note="Embedding and the looped encoder; both outputs stay in the store."),
            BiRecurrentCall(
                forward_topology="ctx_gru_fwd", backward_topology="ctx_gru_bwd", out_var="_gru_gen",
                sequence=OutputRef("encode", index=2), seq_len=n, input_dim=d, hidden_dim=gru_h,
                note="The BiGRU over the tokens, swept in C++: one [h_fwd | h_bwd] row per token."),
            SubgraphCallComponent(
                topology="predict", outputs=(), retain=True, length=n,
                inputs={"x": OutputRef("encode", index=1), "gru": OutputRef("ctx_gru_fwd")},
                note="Duration and pitch predictors; the log-durations come back for the rounding."),
            LuaFragment(fragment / "02_durations.lua", reads=("_n",),
                        defines=("_dur", "_centers", "_frame_pos", "_n_frames", "_pitch_shift")),
            SubgraphCallComponent(
                topology="decode", outputs=(), retain=True,
                inputs={"x": OutputRef("predict", index=2), "centers": Var("_centers"), "dur": Var("_dur"),
                        "frame_pos": Var("_frame_pos"), "pitch_shift": Var("_pitch_shift")},
                axes={"n_enc_frames": frames, "n_tokens": n, "n_past": Lit(0)},
                note="Gaussian upsampling, the decoder and the heads."),
            LuaFragment(fragment / "03_phase.lua", reads=("_n_frames", "HOP", "SAMPLE_RATE"),
                        defines=("_phase0", "_noise")),
            SubgraphCallComponent(
                topology="synth", outputs=("waveform",),
                inputs={"f0": OutputRef("decode", index=1), "vuv": OutputRef("decode", index=2),
                        "log_ah": OutputRef("decode", index=3), "log_an": OutputRef("decode", index=4),
                        "phase0": Var("_phase0"), "noise": Var("_noise")},
                axes={"n_enc_frames": frames, "n_samples": BinOp("*", BinOp("-", frames, Lit(1)), Var("HOP")),
                      "n_past": Lit(0)},
                note="Harmonics, shaped noise and the output FIR."),
            DriverReturn(values=("waveform",)),
        ]


def _is_auroratts(path: Path) -> bool:
    path = Path(path)
    config = path / "config.json"
    if not (path.is_dir() and config.is_file() and (path / "modeling_auroratts.py").is_file()):
        return False
    try:
        return json.loads(config.read_text()).get("model_type") == "auroratts"
    except (OSError, ValueError):
        return False


def _build_auroratts(path: Path, output_path: str) -> TTSAuroraExportConfig:
    return TTSAuroraExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=BaseMultiPhaseModelExportConfig,
        recognizers=[ModelRecognizer(name="auroratts", detect=_is_auroratts, build_config=_build_auroratts)],
    ))
