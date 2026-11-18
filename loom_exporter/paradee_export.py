"""Paradee (sahilmahendrakar, Apache-2.0): a 7.88M-parameter TTS distilled from Kokoro-82M, eight of Kokoro's
voices in English, Hindi, French and Spanish, 24 kHz. Exported from the PyTorch halves of a release
(`pytorch/text_side.pt`, `pytorch/decoder.pt`, 1.3 and later; 1.1 is the same graph with one voice).

Upstream (`training/scripts/export_paradee_v1_2.py`'s `Paradee2.forward`, the graph its ONNX files hold):

  ids, voice -> ALBERT + bert_encoder -> DurationEncoder (3x BiLSTM + AdaLayerNorm) -> BiLSTM
      -> durations: `round(sum(sigmoid(duration_proj)) / speed)`, at least 1
      -> the prosody states repeated by the durations -> F0Ntrain (BiLSTM, 2x3 AdainResBlk1d) -> F0, N
  ids -> TextEncoder (CNN + BiLSTM) -> asr_proj (Linear, GELU, Linear to the teacher's 512 channels)
      -> repeated by the durations
  asr, F0, N, voice -> Kokoro's AdaIN decoder trunk at width 256 (80 frames a second)
      -> the source-filter head: ConvNeXt blocks -> per-frame gains on 80 mel-spaced points; a sine bank
         at the F0 and its 64 multiples, STFT-filtered by a smooth gain and phase; white noise and noise
         pulsing with the pitch, STFT-filtered by their own gains; a 3 ms-window noise path -> 24 kHz.

**Everything before the head is Kokoro's own modules at smaller widths** (`student.TextStudent` builds
`kokoro.modules.CustomAlbert`, `ProsodyPredictor` and `TextEncoder`; the trunk is `kokoro.istftnet`'s
`AdainResBlk1d`), so it is exported by `kokoro_export`'s phases unchanged: ALBERT as one graph, the six
BiLSTMs as cells the engine sweeps, the AdaLayerNorms, the duration head and the F0/N stacks as graphs, and
the driver fragments that wire them are Kokoro's (`02_duration_encoder.lua`, `04_f0n.lua`). Two things are
new: `asr_proj`, between the text encoder's BiLSTM and the frame expansion, and the head, which replaces
Kokoro's upsampling iSTFTNet generator.

**The voice is a row of two learned tables**, one per half (`TextStudent.style` [voices, 32] for the
predictor's AdaLayerNorms and F0Ntrain, `VDecoder.style` [voices, 16] for the trunk's AdaIN), carried in
the GGUF and picked by the driver's `voice` input. Unlike Kokoro's packs the row does not depend on the
phoneme count. The names are `config.json`'s `voices`, in the tables' order.

**The sine bank's running phase leaves the graph**, as tiny-voca's and AuroraTTS's do: upstream
accumulates `f0 * 300 / 24000` cycles per frame in float64 and wraps it per frame (`vhead.sine_bank`),
because harmonic 64 multiplies any error by 64. ggml has no float64, so the driver (Lua numbers are
doubles) reads the F0 curve back, does that sum, and hands the graph each frame's wrapped start and its
advance to the next frame centre. Inside the graph the per-sample phase is upstream's linear
interpolation between frame centres, built as `[start, advance] @ [1, ramp]` per frame and cropped by 150
samples at each end, which is where the frame centres sit; nothing slices a dynamic axis.

**Every STFT is a convolution with constant kernels**, upstream's own ONNX construction (`ConvSTFT`: the
windowed DFT basis, reflect padding, a transposed convolution with the windowed inverse basis divided by
the window-square envelope). **Outer products are matmuls with K = 1** (`a[:, None] @ b[None]`): ggml's
elementwise ops broadcast only their second operand, and a product of two numbers is exact either way.

**Noise is torch's**: upstream draws `torch.randn_like` from the global generator, so the driver draws
`aten_randn(seed, samples)` (loom.cpp ADR-133) and the graph takes it, and its reverse for the 3 ms path.
"""
import json
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

DRIVER_DIR = Path(__file__).resolve().parent / "paradee_driver"

PARADEE_MISSING = (
    "Paradee's model code is its training scripts (github.com/sahilmahendrakar/paradee, Apache-2.0, "
    "`training/scripts/`), which build Kokoro's modules: point LOOM_PARADEE_REPO at a clone and install "
    "`kokoro` (pip install kokoro)."
)
# A clone of upstream, never vendored: the training scripts ARE the architecture, so the export names the
# commit it was verified against (a moved script is a different export).
DEFAULT_REPO = os.path.expanduser("~/Dev/paradee")
PARADEE_COMMIT = "2de547e"

SAMPLE_RATE = 24000
FRAME_HOP = 300          # samples per F0 frame (80 a second), `vhead.FHOP`
DURATION_HOP = 600       # samples per duration frame (40 a second): the trunk upsamples by two
TEXT_STYLE = "loom.paradee.text_style"
DECODER_STYLE = "loom.paradee.decoder_style"
# The trunk's widths and the head's, as upstream's export builds them (`VDecoder(head="v2", dim=160,
# blocks=4)` over decoder preset "A"); checked against the checkpoint's shapes when it loads.
HEAD_DIM, HEAD_BLOCKS = 160, 4


def _strip(module):
    """Weight norm folded into plain weights, as upstream's export does before it traces (`strip`)."""
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    for m in module.modules():
        try:
            remove_weight_norm(m)
        except (ValueError, AttributeError):
            pass
        if is_parametrized(m, "weight"):
            remove_parametrizations(m, "weight", leave_parametrized=True)


def load_paradee(model_dir: Path):
    """(TextStudent, VDecoder, config) from a release directory, weight norm folded."""
    repo = os.environ.get("LOOM_PARADEE_REPO", DEFAULT_REPO)
    scripts = str(Path(repo) / "training" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        from student import TextStudent
        from vhead import VDecoder
    except ImportError as exc:                       # pragma: no cover - env-dependent
        raise ImportError(PARADEE_MISSING) from exc
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    text = torch.load(model_dir / "pytorch" / "text_side.pt", map_location="cpu", weights_only=True)["model"]
    decoder = torch.load(model_dir / "pytorch" / "decoder.pt", map_location="cpu", weights_only=True)["model"]
    voices = list(config.get("voices") or ["af_heart"])
    for name, state in (("text_side", text), ("decoder", decoder)):
        if state["style"].shape[0] != len(voices):
            raise ValueError(f"paradee: {name}.pt has {state['style'].shape[0]} voice rows, config.json "
                             f"names {len(voices)} voices")
    ts = TextStudent("s+mlp", n_token=int(config.get("n_token", 178)), n_voices=len(voices)).eval()
    ts.load_state_dict(text, strict=True)
    vd = VDecoder(head="v2", dim=HEAD_DIM, blocks=HEAD_BLOCKS, n_voices=len(voices)).eval()
    vd.load_state_dict(decoder, strict=True)
    if not vd.head.burst or vd.head.mod != 2 or vd.head.K != 64:
        raise ValueError("paradee: this export traces the 1.1+ head (two pulsing-noise pairs, 64 harmonics, "
                         "the 3 ms noise path); this checkpoint's head differs")
    _strip(ts)
    _strip(vd)
    return ts, vd, config


class _AsrProjWrapper(nn.Module):
    """`TextStudent.asr_proj` over the text encoder's BiLSTM rows: `(T, 2H) -> (T, 512)`. Applied per token
    before the expansion, which is the same numbers as upstream's `asr_tok @ aln` (a repeat)."""

    def __init__(self, proj):
        super().__init__()
        self.proj = proj

    def forward(self, x):
        return self.proj(x)


def _dft_bases(n):
    """`ConvSTFT`'s analysis basis `[2K, 1, n]`, inverse basis `[2K, 1, n]` and window square `[1, 1, n]`,
    built exactly as upstream's export builds them (float64, then float32)."""
    w = torch.hann_window(n, dtype=torch.float64)
    k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
    ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n
    c = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64)
    c[0] = c[-1] = 1
    fwd = (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w).float()[:, None]
    inv = (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w).float()[:, None]
    return fwd.contiguous(), inv.contiguous(), (w ** 2).float()[None, None].contiguous()


def phase_ramp() -> torch.Tensor:
    """`[2, FRAME_HOP]`: a frame's samples as `[start, advance] @ ramp` -- the start, plus the advance times
    each sample's position past the frame centre (`(t + 0.5) / FRAME_HOP - 0.5`, upstream's interpolation
    coordinate, minus the half frame the driver's table is offset by)."""
    rr = (torch.arange(FRAME_HOP, dtype=torch.float64) + 0.5) / FRAME_HOP
    return torch.stack((torch.ones(FRAME_HOP, dtype=torch.float64), rr))


class _SynthWrapper(nn.Module):
    """`(asr [1, 512, F], f0 [1, 2F], n [1, 2F], style [16], phase [2F + 1, 2], noise [600 F],
    noise_rev [600 F]) -> waveform [1, 600 F]`: `Paradee2.trunk` and `HeadX.forward` (upstream's export) for one utterance.

    `phase` is the driver's: row j + 1 is F0 frame j's wrapped running phase (cycles) and the advance to
    frame j + 1's centre; row 0 repeats frame 0's start with no advance, which is the clamp upstream's
    `floor(p).clamp(0)` applies to the first half frame."""

    def __init__(self, vd):
        super().__init__()
        self.F0_conv, self.N_conv, self.encode, self.decode, self.asr_res = (
            vd.F0_conv, vd.N_conv, vd.encode, vd.decode, vd.asr_res)
        h = vd.head
        self.inp, self.norm0, self.blocks, self.norm1, self.out, self.out_b = (
            h.inp, h.norm0, h.blocks, h.norm1, h.out, h.out_b)
        self.up, self.C, self.G, self.Gb, self.nb, self.n_fft, self.hop, self.mod = (
            h.up, h.C, h.G, h.Gb, h.nb, h.n_fft, h.hop, h.mod)
        self.register_buffer("M_T", h.M.T.contiguous())
        self.register_buffer("Mb_T", h.Mb.T.contiguous())
        for name, n in (("main", h.n_fft), ("burst", 150)):
            fwd, inv, w2 = _dft_bases(n)
            self.register_buffer(f"{name}_fwd", fwd)
            self.register_buffer(f"{name}_inv", inv)
            self.register_buffer(f"{name}_w2", w2)
        k = torch.arange(1, h.K + 1).float()[:, None]                         # [K, 1]
        self.register_buffer("k", k)
        # `2 * torch.pi * self.k` as upstream evaluates it (left to right, a float32 tensor times a double).
        self.register_buffer("k2pi", 2 * torch.pi * k)
        # Frequency of every main-STFT bin, `torch.linspace(0, SR / 2, nb)`, as a column.
        self.register_buffer("fr", torch.linspace(0, SAMPLE_RATE / 2, h.nb)[:, None].contiguous())
        self.register_buffer("ramp", phase_ramp().float())
        self.register_buffer("ones_hop", torch.ones(1, FRAME_HOP))
        self.register_buffer("ones_up", torch.ones(1, h.up))
        self.register_buffer("ones_k", torch.ones(1, h.K))

    @staticmethod
    def _stft(x, fwd, n, hop):
        y = F.conv1d(F.pad(x[None, None], (n // 2, n // 2), mode="reflect"), fwd, stride=hop)[0]
        k = n // 2 + 1
        return y[:k], y[k:]

    @staticmethod
    def _istft(re, im, inv, w2, n, hop):
        y = F.conv_transpose1d(torch.cat([re, im], 0)[None], inv, stride=hop)[0, 0]
        env = F.conv_transpose1d(re[:1][None] * 0.0 + 1.0, w2, stride=hop)[0, 0]
        return (y / env.clamp_min(1e-8))[n // 2: -(n // 2)]

    def forward(self, asr, f0, n, style, phase, noise, noise_rev):
        # The layouts Kokoro's decoder takes, which are what the retained F0/N projections and the
        # frame-expanded `asr_proj` already are.
        s = style[None]
        F0 = self.F0_conv(f0[:, None])
        Nn = self.N_conv(n[:, None])
        f0 = f0[0]
        x = self.encode(torch.cat([asr, F0, Nn], 1), s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nn], 1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        # ---- the head (`SourceFilterHead2.forward` / upstream's `HeadX`) ----
        z = self.norm0(self.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            z = b(z)
        hn = self.norm1(z.transpose(1, 2))[0]                                # [n, dim]
        frames = hn.shape[0]
        width = self.C * self.G + 1
        o = self.out(hn).reshape(frames * self.up, width)                     # [T, C G + 1]
        o = torch.cat([o, o[-1:]], 0)                                          # centred STFT: T + 1 frames
        g = (o[:, : self.C * self.G].reshape(frames * self.up + 1, self.C, self.G) @ self.M_T)
        g = g.permute(1, 2, 0)                                                 # [C, nb, T + 1]
        gh = torch.exp(g[0].clamp(max=6.0))
        gn = torch.exp(g[3].clamp(max=6.0))
        um = torch.sqrt(g[1] ** 2 + g[2] ** 2).clamp_min(1e-4)
        ur, ui = g[1] / um, g[2] / um
        # Pulse shift: `2 pi (bin frequency / f0) * 0.5 tanh(o[-1])`, f0 per STFT frame (each F0 frame twice,
        # then the last again), at least 60 Hz.
        f0f = (f0[:, None] @ self.ones_up).reshape(frames * self.up)
        f0f = torch.cat([f0f, f0f[-1:]], 0).clamp_min(60.0)
        ratio = (self.fr @ (f0f[None] * 0.0 + 1.0)) / f0f[None]               # [nb, T + 1]
        ang = 2 * torch.pi * ratio * (0.5 * torch.tanh(o[:, -1]))[None]
        cos_a, sin_a = torch.cos(ang), torch.sin(ang)
        Hr = gh * (ur * cos_a - ui * sin_a)
        Hi = gh * (ur * sin_a + ui * cos_a)

        # ---- the sources (`vhead.sine_bank` and the noise) ----
        cyc = (phase @ self.ramp).reshape(-1)[FRAME_HOP // 2: -(FRAME_HOP // 2)]   # [S], cycles
        f0s = (f0[:, None] @ self.ones_hop).reshape(-1)                           # each F0 frame's value
        uv = (f0s > 10.0).float()
        amp = ((self.k @ f0s[None]) < 11000.0).float() * uv[None]                   # [K, S]
        e = (self.ones_k @ (torch.sin(self.k2pi @ cyc[None]) * amp))[0] * 0.1
        n_fft, hop = self.n_fft, self.hop
        Er, Ei = self._stft(e, self.main_fwd, n_fft, hop)
        Nr, Ni = self._stft(noise, self.main_fwd, n_fft, hop)
        j = 0
        for m in range(1, self.mod + 1):
            for fn in (torch.cos, torch.sin):
                a = g[4 + j]
                mr, mi = self._stft(noise * uv * fn(2 * torch.pi * m * cyc), self.main_fwd, n_fft, hop)
                Nr = Nr + a * mr
                Ni = Ni + a * mi
                j += 1
        y = self._istft(Er * Hr - Ei * Hi + Nr * gn, Er * Hi + Ei * Hr + Ni * gn,
                        self.main_inv, self.main_w2, n_fft, hop)
        # The 3 ms path: gains at 4 points per trunk frame, on the reversed noise.
        gb = torch.exp((self.out_b(hn).reshape(frames * 4, self.Gb) @ self.Mb_T).clamp(max=6.0)).transpose(0, 1)
        gb = torch.cat([gb, gb[:, -1:]], 1)
        br, bi = self._stft(noise_rev, self.burst_fwd, 150, 75)
        y = y + self._istft(br * gb, bi * gb, self.burst_inv, self.burst_w2, 150, 75)
        return y[None]


@dataclass(kw_only=True)
class ParadeeExportConfig(BaseMultiPhaseModelExportConfig):
    """A Paradee release directory (`config.json` + `pytorch/{text_side,decoder}.pt`) -> one Loom GGUF."""

    architecture: str = "paradee"
    model_dir: str
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = DRIVER_DIR
    trace_frames: int = 12
    _config: Optional[dict] = field(default=None, init=False, repr=False)
    _styles: Optional[dict] = field(default=None, init=False, repr=False)
    _dims: Optional[dict] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string"),
        "model_dir": Unchecked("path to the release; the recognizer found config.json + pytorch/ in it"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "driver_script_path": Unchecked("hand-written fragments, checked against the topologies"),
        "trace_frames": Unchecked("the concrete duration-frame count the synth trace runs at"),
        "_config": Unchecked("config.json, read in phases()"),
        "_styles": Unchecked("the two style tables, read in phases()"),
        "_dims": Unchecked("prosody_dims of the real modules, read in phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct
        from . import kokoro_export as ke

        ke.load_kokoro()            # Kokoro's trace-friendly AdainResBlk1d, before anything is traced
        ts, vd, config = load_paradee(Path(self.model_dir))
        self._config = config
        self._styles = {TEXT_STYLE: ts.style.detach().float().numpy().copy(),
                        DECODER_STYLE: vd.style.detach().float().numpy().copy()}
        self._dims = ke.prosody_dims(ts.text_encoder, ts.predictor)
        seq = ct.RangeDim(1, 2000)
        g = torch.Generator().manual_seed(0)
        frames = self.trace_frames
        f0_axis = ct.RangeDim(2, 4000)
        asr_dim = ts.asr_proj[-1].out_features
        hidden = 2 * self._dims["hidden_per_dir"]
        vd_style = vd.style.shape[1]
        synth = _SynthWrapper(vd).eval()
        f0 = torch.rand(2 * frames, generator=g) * 200 + 80
        dummy = (torch.randn(1, asr_dim, frames, generator=g), f0[None], torch.randn(1, 2 * frames, generator=g),
                 torch.randn(vd_style, generator=g), torch.rand(2 * frames + 1, 2, generator=g),
                 torch.randn(DURATION_HOP * frames, generator=g), torch.randn(DURATION_HOP * frames, generator=g))
        with torch.no_grad():
            synth(*dummy)                  # eager sanity check before tracing
        return [
            ke.build_albert_bert_encoder_phase(ts.bert, ts.bert_encoder,
                                               vocab_size=ts.bert.config.vocab_size),
            *ke.build_prosody_phases(ts.text_encoder, ts.predictor),
            ExportPhase(
                name="asr_proj", wrapper=_AsrProjWrapper(ts.asr_proj).eval(),
                dummy_inputs=(torch.randn(7, hidden, generator=g),),
                mil_inputs=[ct.TensorType(name="x", shape=(seq, hidden), dtype=np.float32)],
            ),
            ExportPhase(
                name="synth", wrapper=synth, dummy_inputs=dummy,
                mil_inputs=[ct.TensorType(name="asr", shape=(1, asr_dim, ct.RangeDim(1, 2000)), dtype=np.float32),
                            ct.TensorType(name="f0", shape=(1, f0_axis), dtype=np.float32),
                            ct.TensorType(name="n", shape=(1, f0_axis), dtype=np.float32),
                            ct.TensorType(name="style", shape=(vd_style,), dtype=np.float32),
                            ct.TensorType(name="phase", shape=(ct.RangeDim(3, 4001), 2), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(ct.RangeDim(600, 1200000),), dtype=np.float32),
                            ct.TensorType(name="noise_rev", shape=(ct.RangeDim(600, 1200000),), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"asr": {2: "n_enc_frames"}, "f0": {1: "2*n_enc_frames"}, "n": {1: "2*n_enc_frames"},
                               "phase": {0: "2*n_enc_frames+1"},
                               "noise": {0: "600*n_enc_frames"}, "noise_rev": {0: "600*n_enc_frames"}},
            ),
        ]

    def backend_kwargs(self) -> dict:
        kwargs = super().backend_kwargs()
        if self._styles:
            kwargs["driver_weights"] = dict(self._styles)
        # ALBERT stays float under `--quantize`. Measured 2026-11-18, whisper-small on 10 held-out sentences
        # (af_heart, misaki): every weight in Q8_0_PC 35.2% WER against F32's 4.0%; ALBERT float 4.0%, the
        # five BiLSTMs float instead 4.0% at 2 MB more. Either one recovers F32's rate, so the error needs
        # both: ALBERT applies one shared layer six times (its weights appear once), and the engine's int8
        # GEMMs round each activation as well as the weights, into BiLSTMs that carry it from step to step.
        # ALBERT is the smaller to keep (0.8 MB in int8).
        kwargs["float_phases"] = ("albert_bert_encoder",)
        return kwargs

    def driver_components(self) -> List:
        """One top-level fragment holding the per-sentence pipeline and the live door, and the body that
        speaks every sentence. The pipeline's topology names are literals or the helpers' (declared
        below), so every call is checked against the topologies."""
        from .driver_components import ComputedCall, DriverReturn, ExportConstants, HelperCall, LuaFragment
        from .lua_library import LuaLibrary

        dims = self._dims or {"style_dim": 0, "d_model": 0, "hidden_per_dir": 0}
        voices = list((self._config or {}).get("voices") or [])
        return [
            LuaLibrary(uses=("array_slice", "array_sum", "run_bi_lstm", "run_resblk_stack", "run_proj1x1",
                             "predict_durations", "aten_randn")),
            ExportConstants(values={
                "STYLE_DIM": dims["style_dim"], "D_MODEL": dims["d_model"],
                "HIDDEN_PER_DIR": dims["hidden_per_dir"], "N_VOICES": len(voices),
                "DECODER_STYLE_DIM": int(self._styles[DECODER_STYLE].shape[1]) if self._styles else 0,
                "FRAME_HOP": FRAME_HOP, "SAMPLE_RATE": SAMPLE_RATE,
            }, top_level=True),
            LuaFragment(DRIVER_DIR / "00_paradee.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish"),
                        drives=(
                            HelperCall("run_bi_lstm", tuple(f"duration_lstm_{i}" for i in range(3)),
                                       written='"duration_lstm_" .. i'),
                            ComputedCall(topologies=tuple(f"duration_adaln_{i}" for i in range(3)),
                                         inputs=("x", "style"), written='"duration_adaln_" .. i'),
                            HelperCall("run_bi_lstm", "top_lstm"),
                            HelperCall("run_bi_lstm", "text_encoder_lstm"),
                            HelperCall("run_bi_lstm", "f0n_shared_lstm"),
                            HelperCall("run_resblk_stack", "f0n_f0"),
                            HelperCall("run_resblk_stack", "f0n_n"),
                            HelperCall("run_proj1x1", "f0n_f0_proj"),
                            HelperCall("run_proj1x1", "f0n_n_proj"),
                        )),
            LuaFragment(DRIVER_DIR / "01_generate.lua", defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def hparams(self) -> dict:
        """The rate and the phoneme style: English as misaki writes it, which is what Paradee was distilled
        on (Kokoro's own G2P)."""
        return {"sample_rate": SAMPLE_RATE, "tts.phoneme_style": "kokoro"}

    def phoneme_table(self) -> dict:
        """Kokoro's vocabulary (`config.json`'s `vocab`, phoneme -> id), each id wrapped in a 0 at both ends
        as `KModel.forward` and upstream's runtime (`[0] + ids + [0]`) do."""
        vocab = (self._config or {}).get("vocab") or {}
        symbols = sorted(vocab, key=lambda s: vocab[s])
        return {"symbols": symbols, "ids": [vocab[s] for s in symbols], "bos": 0, "eos": 0, "blank": -1,
                "interleave_blank": False}

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "phonemes"
        contract["sample_rate"] = SAMPLE_RATE
        contract["text.phoneme_alphabet"] = "ipa"
        # The voices the file carries, in the style tables' order: the driver's `voice` is a row index, and
        # the first is what it speaks when none is named. Each voice speaks one language (upstream picks the
        # G2P by voice), declared beside it so a host can phonemize for the voice it was asked for; the
        # languages in first-voice order, so the first is the default voice's.
        voices = list((self._config or {}).get("voices") or [])
        per_voice = [_bcp47(((self._config or {}).get("languages") or {}).get(v, "en-us")) for v in voices]
        contract["tts.voices"] = voices
        contract["tts.voice_languages"] = per_voice
        contract["text.languages"] = list(dict.fromkeys(per_voice)) or ["en-US"]
        return contract


def _bcp47(espeak: str) -> str:
    """`config.json`'s eSpeak language codes as BCP 47 (`en-us` -> `en-US`, `fr-fr` -> `fr-FR`)."""
    lang, _, region = espeak.partition("-")
    return f"{lang}-{region.upper()}" if region else lang


def _is_paradee(path: Path) -> bool:
    if not (path.is_dir() and (path / "pytorch" / "text_side.pt").is_file()
            and (path / "pytorch" / "decoder.pt").is_file()):
        return False
    try:
        return json.loads((path / "config.json").read_text()).get("model_type") == "paradee"
    except (OSError, ValueError):
        return False


def _build_paradee(path: Path, output_path: str) -> ParadeeExportConfig:
    return ParadeeExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech", config_class=ParadeeExportConfig,
        recognizers=[ModelRecognizer(name="paradee", detect=_is_paradee, build_config=_build_paradee)],
    ))
