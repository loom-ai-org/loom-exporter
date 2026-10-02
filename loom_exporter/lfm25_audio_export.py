"""Export LFM2.5-Audio-1.5B's speech-to-text door -- family 3's third leaf, and its first hybrid LM.

`LiquidAI/LFM2.5-Audio-1.5B` (liquid-audio 1.3) is an audio encoder, an adapter and LFM2.5-1.2B:

    16 kHz audio -> NeMo log-mel (128 bins) -> FastConformer (canary-180m-flash, x8)
                 -> adapter MLP (LayerNorm, 512 -> 2048 -> GELU -> 2048) -> one LM row per 80 ms
    "<|startoftext|><|im_start|>system\\nPerform ASR.<|im_end|>\\n<|im_start|>user\\n" [audio rows]
    "<|im_end|>\\n<|im_start|>assistant\\n" -> greedy text until <|im_end|>

which is `generate_sequential` with the README's fixed ASR system prompt. The model also SPEAKS (a
depthformer over 8 Mimi codebooks and an LFM2-based detokenizer); that door is not this export.

**The encoder is NeMo's own.** liquid-audio vendors NeMo's `ConformerEncoder` and
`AudioToMelSpectrogramPreprocessor`, and the checkpoint's 692 conformer tensors load key for key into
NeMo's class -- so this builds NeMo's modules from the checkpoint's `encoder`/`preprocessor` configs
and uses Canary's encoder path, whose masks were already fixed against NeMo (Retro-065). Dither is
training-only in both.

**Not the family-3 template, though it is a family-3 model.** `BaseSpeechLMExportConfig` holds an
encoder to "rows grow linearly with whole chunks of samples"; NeMo's centred STFT gives
`floor(n / 160) + 1` frames and the x8 subsampling `ceil(frames / 8)` rows, so k whole chunks become
k + 1 rows and the template's geometry check fails by construction. So the encoder takes the whole
waveform and its length, as Canary's does (`transcribe`'s one-pass branch), and the driver walks the
prompt in segments as family 3's does. The embed, decoder and head wrappers are family 3's.

**The LM is a hybrid**: 10 short-convolution blocks and 6 attention blocks. The decoder phase fuses
both (`fuse_attention`, `fuse_conv`), so the file carries a KV cache and a conv-state cache.

Phases: `encoder` (waveform, length -> rows), `embed` (ids -> embeddings), `decoder` (embeddings ->
hidden, cached), `lm_head` (hidden -> logits, the tied embedding).

Usage:
  loom-export ~/Dev/models/lfm2.5-audio-1.5b -o lfm25_audio_asr.gguf \\
      --task automatic-speech-recognition --model lfm2.5-audio
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

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .speech_lm_export import _DecoderWrapper, _EmbedWrapper, _LMHeadWrapper, causal_mask
from .spec_protocol import Axis, Unchecked

# The README's fixed ASR system prompt, and the turn scaffolding `ChatState` adds around it.
ASR_SYSTEM_PROMPT = "Perform ASR."
PROMPT_HEAD = ["<|startoftext|>", "<|im_start|>system\n", ASR_SYSTEM_PROMPT, "<|im_end|>\n",
               "<|im_start|>user\n"]
PROMPT_TAIL = ["<|im_end|>\n", "<|im_start|>assistant\n"]
END_OF_TURN = "<|im_end|>"
# `generate_sequential(max_new_tokens=512)`, the README's ASR call.
MAX_NEW_TOKENS = 512
# The KV cache: the prompt is ~20 text positions plus 12.5 audio rows a second, then the transcript.
MAX_SEQ_LEN = 4096
# The encoder's sample axis: 280 s at 16 kHz is 3500 audio rows, which with the ~19 prompt positions
# and a 512-id transcript fits the 4096-position cache above (300 s would not: 3750 + 19 + 512).
MAX_SECONDS = 280

TRACE_SAMPLES = 16000 * 3 + 173
TRACE_TOKENS = 7


def load_model(model_dir: str):
    """`(preprocessor, encoder, adapter, lfm, config)`: NeMo's preprocessor and conformer and the
    adapter MLP, with the checkpoint's weights, and the LFM2 backbone as transformers' `Lfm2Model`."""
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
    from safetensors.torch import load_file
    from transformers import Lfm2Config, Lfm2Model

    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    weights = load_file(str(d / "model.safetensors"))
    pre = cfg["preprocessor"]
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=pre["sample_rate"], normalize=pre["normalize"], window_size=pre["window_size"],
        window_stride=pre["window_stride"], window=pre["window"], features=pre["features"],
        n_fft=pre["n_fft"], log=pre["log"], frame_splicing=pre["frame_splicing"], dither=pre["dither"],
        pad_to=pre["pad_to"], pad_value=pre["pad_value"]).eval()
    encoder = ConformerEncoder(**cfg["encoder"]).eval()
    encoder.load_state_dict({k[len("conformer."):]: v.float() for k, v in weights.items()
                             if k.startswith("conformer.")})
    hidden = int(cfg["lfm"]["hidden_size"])
    adapter = nn.Sequential(nn.LayerNorm(encoder._feat_out), nn.Linear(encoder._feat_out, hidden), nn.GELU(),
                            nn.Linear(hidden, hidden)).eval()
    adapter.load_state_dict({k[len("audio_adapter.model."):]: v.float() for k, v in weights.items()
                             if k.startswith("audio_adapter.model.")})
    lfm_cfg = dict(cfg["lfm"])
    lfm_cfg["torch_dtype"] = "float32"
    lfm = Lfm2Model(Lfm2Config(**lfm_cfg)).eval()
    lfm.load_state_dict({k[len("lfm."):]: v.float() for k, v in weights.items() if k.startswith("lfm.")})
    return preprocessor, encoder, adapter, lfm, cfg


class EncoderPhase(nn.Module):
    """`(waveform, length) -> audio rows [n_rows, hidden]`: NeMo's log-mel and FastConformer over the
    real samples, then the adapter -- `LFM2AudioModel._prefill`'s audio half, which keeps the
    `audio_in_len` rows the conformer reports."""

    def __init__(self, preprocessor, encoder, adapter):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, waveform, length):
        features, feature_len = self.preprocessor(input_signal=waveform, length=length)
        # `ChatState.add_audio` ignores the length NeMo returns, `floor(n / 160)`, and hands the
        # conformer EVERY frame the centred STFT made -- one more. Measured on jfk.wav: with NeMo's
        # length the last frame is masked and the rows differ by 0.32 (absmax 3.4); with this one they
        # are identical.
        encoded, encoded_len = self.encoder(audio_signal=features, length=feature_len + 1)
        return self.adapter(encoded.permute(0, 2, 1))[0, :encoded_len[0]]


class _TiedHead(nn.Module):
    """`nn.functional.linear(hidden, embed_tokens.weight)`: the text logits `generate_sequential`
    reads, through the input embedding (LFM2 ties them)."""

    def __init__(self, embed_tokens):
        super().__init__()
        self.weight = embed_tokens.weight

    def forward(self, hidden):
        return nn.functional.linear(hidden, self.weight)


def prompt_ids(model_dir: str):
    """`(head, tail, end_of_turn)` ids: each piece of the scaffolding tokenized ON ITS OWN, as
    `ChatState.add_text` tokenizes it (`add_special_tokens=False`)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    head = [i for piece in PROMPT_HEAD for i in enc(piece)]
    tail = [i for piece in PROMPT_TAIL for i in enc(piece)]
    eot = enc(END_OF_TURN)
    if len(eot) != 1:
        raise ValueError(f"{END_OF_TURN!r} is {len(eot)} ids; the driver stops on one")
    return head, tail, eot[0]


@dataclass(kw_only=True)
class Lfm25AudioAsrExportConfig(BaseMultiPhaseModelExportConfig):
    """An LFM2.5-Audio checkpoint directory -> one Loom GGUF for its speech-to-text door."""

    architecture: str = "lfm2.5-audio-asr"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "lfm25_audio_driver"
    _facts: dict = field(default_factory=dict, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the checkpoint directory; the recognizer found its "
                               "`Lfm2AudioForConditionalGeneration` config.json"),
        "decomposition": Unchecked("MultiPhase by construction -- four graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragment is parsed and checked against the "
                                         "traced topologies by LuaFragment"),
        "_facts": Unchecked("the checkpoint's numbers, read during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        preprocessor, encoder, adapter, lfm, cfg = load_model(self.model_dir)
        head, tail, eot = prompt_ids(self.model_dir)
        hidden = int(lfm.config.hidden_size)
        self._facts = {"head": head, "tail": tail, "eot": eot, "hidden": hidden,
                       "sample_rate": int(cfg["preprocessor"]["sample_rate"])}
        n_samples = ct.RangeDim(400, self._facts["sample_rate"] * MAX_SECONDS)
        token_axis = ct.RangeDim(1, MAX_SEQ_LEN)
        return [
            ExportPhase(
                name="encoder",
                wrapper=EncoderPhase(preprocessor, encoder, adapter).eval(),
                dummy_inputs=(torch.randn(1, TRACE_SAMPLES) * 0.1, torch.tensor([TRACE_SAMPLES])),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, n_samples), dtype=np.float32),
                            ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="embed",
                wrapper=_EmbedWrapper(lfm).eval(),
                dummy_inputs=(torch.randint(0, 1000, (1, TRACE_TOKENS), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32)],
            ),
            ExportPhase(
                name="decoder",
                wrapper=_DecoderWrapper(lfm).eval(),
                dummy_inputs=(torch.randn(1, TRACE_TOKENS, hidden),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, token_axis, hidden), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
                ],
                fuse_attention=True,
                fuse_conv=True,
                kv_cache_size=MAX_SEQ_LEN,
            ),
            ExportPhase(
                name="lm_head",
                wrapper=_LMHeadWrapper(_TiedHead(lfm.embed_tokens)).eval(),
                dummy_inputs=(torch.randn(1, 1, hidden),),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, ct.RangeDim(1, MAX_SEQ_LEN), hidden),
                                          dtype=np.float32)],
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        f = self._facts
        constants = {"PROMPT_HEAD": f.get("head", [1]), "PROMPT_TAIL": f.get("tail", [7]),
                     "END_OF_TURN": f.get("eot", 7), "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                     "MAX_SEQ_LEN": MAX_SEQ_LEN}
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("waveform", CALLER), ("length", CALLER)), n_tokens=Len("waveform")),
            LuaFragment(self.driver_script_path / "01_transcribe.lua",
                        reads=("waveform", "length") + tuple(constants), defines=("ids",)),
            DriverReturn(values=("ids",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = int(self._facts.get("sample_rate", 16000))
        return contract

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=False, root_axis=self.root_axis, tokenizer_dir=self.model_dir,
                    eos_token_id=int(self._facts.get("eot", 7)))


def _is_lfm25_audio(path: Path) -> bool:
    cfg = path / "config.json"
    if not (path.is_dir() and cfg.is_file() and (path / "model.safetensors").is_file()):
        return False
    try:
        c = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    return "Lfm2AudioForConditionalGeneration" in c.get("architectures", [])


def _build_lfm25_audio(path: Path, output_path: str) -> LoomExportConfig:
    return Lfm25AudioAsrExportConfig(output_path=output_path, model_dir=str(path))


# ------------------------------------------------------------------------------------- text to speech --
#
# **The TTS door is a second file of the same checkpoint**, `generate_sequential` with a "Perform TTS."
# system prompt: the LM reads the text, emits `<|audio_start|>`, then one audio frame per step -- its
# last hidden row through `depth_linear` into a 6-layer DEPTHFORMER that draws the frame's 8 codes one
# after another -- until a frame's first code is end-of-audio (2048); the frames go through the
# LFM2-based DETOKENIZER and an ISTFT to 24 kHz. liquid-audio needs Python >= 3.12 and the export runs in
# 3.11, so the depthformer and the detokenizer are re-spelled from their weights here (the oracle,
# `loom.cpp/scripts/lfm25_audio_tts_reference.py`, runs liquid-audio itself).

TTS_SAMPLE_RATE = 24000
N_CODEBOOKS = 8
AUDIO_VOCAB = 2049            # 2048 codes + end-of-audio
END_OF_AUDIO = 2048
AUDIO_START = "<|audio_start|>"
# The README's four voices, each a system prompt; the first is the file's own.
TTS_VOICES = {
    "us_male": "Perform TTS. Use the US male voice.",
    "us_female": "Perform TTS. Use the US female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
}
DEFAULT_VOICE = "us_male"
# The README's TTS call: `audio_temperature=0.8, audio_top_k=64`; text is greedy.
TTS_AUDIO_TEMPERATURE = 0.8
TTS_AUDIO_TOP_K = 64
# Frames of audio per call: 512 steps of `generate_sequential` (the README's max_new_tokens) is ~41 s.
TTS_MAX_NEW_TOKENS = 1024
DETOK_UPSAMPLE = 6
DETOK_WINDOW = 30
# The ISTFT: n_fft 1280, hop 320, "same" padding -- (1280 - 320) // 2 trimmed at each end.
ISTFT_N_FFT, ISTFT_HOP = 1280, 320

TRACE_FRAMES = 7


def _rms(x, eps):
    """liquid-audio's `RMSNorm._norm`, `x * rsqrt(mean(x^2) + eps)`: x against a `[..., 1]` column, which
    broadcasts one way."""
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


def _rope_interleaved(x, cos, sin):
    """liquid-audio's `apply_rotary_emb`: consecutive PAIRS as complex numbers, `x * e^{i theta}`."""
    from .pocket_tts_export import _rotate_pairs

    return x * cos + _rotate_pairs(x) * sin


class _DepthBlock(nn.Module):
    """liquid-audio's `StandardBlock(MHA)`: `h = x + attn(norm(x))`, `out = h + swiglu(norm(h))`. The
    attention is GQA with 32 query heads over 8 K/V heads, RMS-normed per head, interleaved RoPE, causal.
    The K/V projections are DUPLICATED to 32 heads here (`repeat_interleave`, which pairs query head h
    with K/V head h // 4 as SDPA's `enable_gqa` does), so no repeat traces."""

    def __init__(self, w: dict, prefix: str, dim: int, heads: int, kv_heads: int, eps: float):
        super().__init__()
        hd = dim // heads
        qkv = w[prefix + "operator.qkv_proj.weight"]
        q, k, v = qkv.split([dim, hd * kv_heads, hd * kv_heads], dim=0)
        rep = heads // kv_heads
        widen = lambda t: t.view(kv_heads, hd, -1).repeat_interleave(rep, dim=0).reshape(heads * hd, -1)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        self.q.weight.data.copy_(q)
        self.k.weight.data.copy_(widen(k))
        self.v.weight.data.copy_(widen(v))
        self.o.weight.data.copy_(w[prefix + "operator.out_proj.weight"])
        self.q_norm = nn.Parameter(w[prefix + "operator.bounded_attention.q_layernorm.weight"].clone())
        self.k_norm = nn.Parameter(w[prefix + "operator.bounded_attention.k_layernorm.weight"].clone())
        self.attn_norm = nn.Parameter(w[prefix + "operator_norm.weight"].clone())
        self.ffn_norm = nn.Parameter(w[prefix + "ffn_norm.weight"].clone())
        ff = w[prefix + "feed_forward.w1.weight"].shape[0]
        self.w1, self.w3 = nn.Linear(dim, ff, bias=False), nn.Linear(dim, ff, bias=False)
        self.w2 = nn.Linear(ff, dim, bias=False)
        for name in ("w1", "w2", "w3"):
            getattr(self, name).weight.data.copy_(w[prefix + f"feed_forward.{name}.weight"])
        self.heads, self.hd, self.eps = heads, hd, eps

    def forward(self, x, cos, sin, mask):
        b, t, d = x.shape
        a = _rms(x, self.eps) * self.attn_norm
        q = self.q(a).view(b, t, self.heads, self.hd)
        k = self.k(a).view(b, t, self.heads, self.hd)
        v = self.v(a).view(b, t, self.heads, self.hd)
        q = _rope_interleaved(_rms(q, self.eps) * self.q_norm, cos, sin)
        k = _rope_interleaved(_rms(k, self.eps) * self.k_norm, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q * (1.0 / math.sqrt(self.hd)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(b, t, d)
        h = x + self.o(ctx)
        f = _rms(h, self.eps) * self.ffn_norm
        return h + self.w2(F.silu(self.w1(f)) * self.w3(f))


class DepthPhase(nn.Module):
    """`(hidden, prev) -> logits [8, 2049]`: one depthformer pass over all 8 rows of a frame.

    `_sample_audio_frame` runs it cached, one row per code: row j is `depth_linear(hidden)[j]` plus the
    PREVIOUS code's embedding in table j - 1 (nothing for row 0). Here every call recomputes all 8 rows
    -- causal, so row j's output depends on rows <= j only, and rows past the one being drawn may hold
    anything (`prev` is zero there). That keeps the graph one fixed shape instead of a prefix sliced by a
    length, and it is 8 positions of a 1024-wide stack. Row j's logits come from codebook j's own table
    (tied: `to_logits` is its embedding) after its own RMSNorm, all eight as one batched matmul; the
    driver draws row j's.

    The same table serves the input embedding (`depth_embeddings[j-1](code)`) and the logits, which is
    why it is gathered by offset from ONE concatenated table."""

    def __init__(self, w: dict, layers: int, dim: int, eps: float = 1e-5, heads: int = 32, kv_heads: int = 8,
                 theta: float = 1_000_000.0):
        super().__init__()
        self.depth_linear = nn.Linear(w["depth_linear.weight"].shape[1], w["depth_linear.weight"].shape[0])
        self.depth_linear.weight.data.copy_(w["depth_linear.weight"])
        self.depth_linear.bias.data.copy_(w["depth_linear.bias"])
        tables = [w[f"depth_embeddings.{i}.embedding.weight"] for i in range(N_CODEBOOKS)]
        self.register_buffer("tables", torch.stack(tables))                         # [8, 2049, dim]
        self.embed = nn.Embedding(N_CODEBOOKS * AUDIO_VOCAB, dim)
        self.embed.weight.data.copy_(torch.cat(tables, 0))
        self.register_buffer("out_norms", torch.stack(
            [w[f"depth_embeddings.{i}.embedding_norm.weight"] for i in range(N_CODEBOOKS)]))  # [8, dim]
        # Row j embeds the previous code in table j - 1; row 0 embeds nothing.
        self.register_buffer("offsets", (torch.arange(N_CODEBOOKS).clamp(min=1) - 1).to(torch.int32) * AUDIO_VOCAB)
        self.register_buffer("row_mask", (torch.arange(N_CODEBOOKS) > 0).float().view(N_CODEBOOKS, 1))
        self.blocks = nn.ModuleList(_DepthBlock(w, f"depthformer.layers.{i}.", dim, heads, kv_heads, eps)
                                    for i in range(layers))
        from .pocket_tts_export import _rope_tables
        cos, sin = _rope_tables(torch.arange(N_CODEBOOKS).view(1, -1), dim // heads, theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("mask", causal_mask(N_CODEBOOKS))
        self.dim, self.eps = dim, eps

    def forward(self, hidden, prev):                       # (1, 1, 2048), (8,) i32
        x = self.depth_linear(hidden).view(N_CODEBOOKS, self.dim)
        x = x + self.embed(prev + self.offsets) * self.row_mask
        x = x.view(1, N_CODEBOOKS, self.dim)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, self.mask)
        n = _rms(x[0], self.eps) * self.out_norms                                   # (8, dim)
        return torch.matmul(self.tables, n.view(N_CODEBOOKS, self.dim, 1)).view(N_CODEBOOKS, AUDIO_VOCAB)


class AudioEmbedPhase(nn.Module):
    """`(codes [1, 8]) -> [1, 1, hidden]`: `audio_embedding(codes + codebook_offsets).sum(0)`, the LM's
    input row for a frame it drew (including the all-end-of-audio frame that closes the speech)."""

    def __init__(self, w: dict):
        super().__init__()
        table = w["audio_embedding.embedding.weight"]
        self.embed = nn.Embedding(table.shape[0], table.shape[1])
        self.embed.weight.data.copy_(table)
        self.register_buffer("offsets", (torch.arange(N_CODEBOOKS) * AUDIO_VOCAB).to(torch.int32).view(1, -1))

    def forward(self, codes):
        return self.embed(codes + self.offsets).sum(dim=1, keepdim=True)


def detokenizer_window_mask(positions: torch.Tensor, window: int) -> torch.Tensor:
    """0 where `0 <= q - k < window`, else a large negative: the detokenizer's own
    `d_idx <= 0 and d_idx > -window`, built from the positions as `pocket_tts_export`'s Mimi mask is."""
    pos = positions.to(torch.float32)[0]
    ones = torch.ones_like(pos)
    delta = torch.matmul(pos.view(-1, 1), ones.view(1, -1)) - torch.matmul(ones.view(-1, 1), pos.view(1, -1))
    outside = torch.relu(-delta) + torch.relu(delta - (window - 1))
    return (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, pos.shape[0], pos.shape[0])


class DetokenizerPhase(nn.Module):
    """`(codes [1, T, 8], positions [1, 6T]) -> waveform`: `LFM2AudioDetokenizer.forward`.

    The codes' embeddings averaged over the 8 codebooks; `nearest-exact` x6 upsampling, which at an
    integer factor is each frame repeated -- done by the DRIVER, which hands every frame's codes over six
    times (the average of a repeated frame's embeddings is that frame's embedding, exactly). In the graph
    a transposed convolution did the same arithmetic, and the shape walk lost its x6: the attention saw
    65 rows where the positions said 390. With the repetition outside, the codes and the positions are
    one axis. Then the LFM2 backbone under a causal 30-position window (its `sliding_attention` layers loaded as
    full, the window being the mask, as liquid-audio does); the 1282-wide head, half log-magnitude, half
    angle; and the "same"-padded ISTFT, which is `istft.py` uncentred and trimmed by `(n_fft - hop) / 2`.
    Its `irfft` ignores the DC and Nyquist bins' imaginary parts; `istft.py`'s sine basis is zero there."""

    def __init__(self, emb_weight, lfm, lin):
        super().__init__()
        from .istft import ISTFT

        self.embed = nn.Embedding(emb_weight.shape[0], emb_weight.shape[1])
        self.embed.weight.data.copy_(emb_weight)
        self.register_buffer("offsets", (torch.arange(N_CODEBOOKS) * (emb_weight.shape[0] // N_CODEBOOKS))
                             .to(torch.int32).view(1, 1, -1))
        self.lfm = lfm
        self.lin = lin
        self.istft = ISTFT(n_fft=ISTFT_N_FFT, hop_length=ISTFT_HOP, center=False)
        self.trim = (ISTFT_N_FFT - ISTFT_HOP) // 2

    def forward(self, codes, positions):                  # (1, 6T, 8) i32, each frame six times; (1, 6T) i32
        x = self.embed(codes + self.offsets).sum(dim=2) * (1.0 / N_CODEBOOKS)     # (1, 6T, 512)
        mask = detokenizer_window_mask(positions, DETOK_WINDOW)
        h = self.lfm(inputs_embeds=x, position_ids=positions, attention_mask=mask,
                     use_cache=False).last_hidden_state
        y = self.lin(h).transpose(1, 2)
        log_abs, angle = y.chunk(2, dim=1)
        mag = torch.exp(log_abs)
        wave = self.istft(mag * torch.cos(angle), mag * torch.sin(angle))
        return wave[:, self.trim:-self.trim]


def load_detokenizer(model_dir: str) -> DetokenizerPhase:
    from safetensors.torch import load_file
    from transformers import Lfm2Config, Lfm2Model

    d = Path(model_dir) / "audio_detokenizer"
    cfg = json.loads((d / "config.json").read_text())
    cfg["layer_types"] = ["full_attention" if t == "sliding_attention" else t for t in cfg["layer_types"]]
    if int(cfg.get("sliding_window", DETOK_WINDOW)) != DETOK_WINDOW:
        raise ValueError(f"the detokenizer's window is {cfg['sliding_window']}, not {DETOK_WINDOW}")
    cfg["torch_dtype"] = "float32"
    w = {k: v.float() for k, v in load_file(str(d / "model.safetensors")).items()}
    lfm = Lfm2Model(Lfm2Config(**cfg)).eval()
    lfm.load_state_dict({k[len("lfm."):]: v for k, v in w.items() if k.startswith("lfm.")})
    lin = nn.Linear(w["lin.weight"].shape[1], w["lin.weight"].shape[0])
    lin.weight.data.copy_(w["lin.weight"])
    lin.bias.data.copy_(w["lin.bias"])
    return DetokenizerPhase(w["emb.emb.weight"], lfm, lin).eval()


def tts_prompt_ids(model_dir: str) -> dict:
    """The TTS prompt's pieces as ids, each tokenized on its own as `ChatState.add_text` does: what goes
    before the voice's system prompt, between it and the user's text, and after the text; every voice's
    prompt; `<|audio_start|>` and `<|im_end|>`."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    def one(s):
        ids = enc(s)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} ids; the driver compares one")
        return ids[0]

    return {
        "pre": enc("<|startoftext|>") + enc("<|im_start|>system\n"),
        "mid": enc("<|im_end|>\n") + enc("<|im_start|>user\n"),
        "tail": enc("<|im_end|>\n") + enc("<|im_start|>assistant\n"),
        "voices": {name: enc(text) for name, text in TTS_VOICES.items()},
        "audio_start": one(AUDIO_START),
        "end_of_turn": one(END_OF_TURN),
    }


@dataclass(kw_only=True)
class Lfm25AudioTtsExportConfig(BaseMultiPhaseModelExportConfig):
    """An LFM2.5-Audio checkpoint directory -> one Loom GGUF for its text-to-speech door."""

    architecture: str = "lfm2.5-audio-tts"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "lfm25_audio_driver"
    _facts: dict = field(default_factory=dict, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the checkpoint directory; the recognizer found its "
                               "`Lfm2AudioForConditionalGeneration` config.json"),
        "decomposition": Unchecked("MultiPhase by construction -- seven graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragment is parsed and checked against the "
                                         "traced topologies by LuaFragment"),
        "_facts": Unchecked("the checkpoint's numbers, read during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct
        from safetensors.torch import load_file

        _, _, _, lfm, cfg = load_model(self.model_dir)
        w = {k: v.float() for k, v in load_file(str(Path(self.model_dir) / "model.safetensors")).items()
             if k.startswith(("depth", "audio_embedding"))}
        depth = DepthPhase(w, int(cfg["depthformer"]["layers"]), int(cfg["depthformer"]["dim"])).eval()
        if int(cfg["codebooks"]) != N_CODEBOOKS:
            raise NotImplementedError(f"{cfg['codebooks']} codebooks; the driver draws {N_CODEBOOKS}")
        self._facts = dict(tts_prompt_ids(self.model_dir), hidden=int(lfm.config.hidden_size))
        hidden = self._facts["hidden"]
        token_axis = ct.RangeDim(1, MAX_SEQ_LEN)
        # The detokenizer's rows: every frame six times (the driver repeats them), one axis for both inputs.
        row_axis = ct.RangeDim(DETOK_UPSAMPLE, DETOK_UPSAMPLE * TTS_MAX_NEW_TOKENS)
        return [
            ExportPhase(
                name="embed", wrapper=_EmbedWrapper(lfm).eval(),
                dummy_inputs=(torch.randint(0, 1000, (1, TRACE_TOKENS), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32)],
            ),
            ExportPhase(
                name="audio_embed", wrapper=AudioEmbedPhase(w).eval(),
                dummy_inputs=(torch.randint(0, 2048, (1, N_CODEBOOKS), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="codes", shape=(1, N_CODEBOOKS), dtype=np.int32)],
            ),
            ExportPhase(
                name="decoder", wrapper=_DecoderWrapper(lfm).eval(),
                dummy_inputs=(torch.randn(1, TRACE_TOKENS, hidden),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, token_axis, hidden), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
                ],
                fuse_attention=True, fuse_conv=True, kv_cache_size=MAX_SEQ_LEN,
            ),
            ExportPhase(
                name="lm_head", wrapper=_LMHeadWrapper(_TiedHead(lfm.embed_tokens)).eval(),
                dummy_inputs=(torch.randn(1, 1, hidden),),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, ct.RangeDim(1, MAX_SEQ_LEN), hidden),
                                          dtype=np.float32)],
            ),
            ExportPhase(
                name="depth", wrapper=depth,
                dummy_inputs=(torch.randn(1, 1, hidden), torch.randint(0, 2048, (N_CODEBOOKS,), dtype=torch.int32)),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, 1, hidden), dtype=np.float32),
                            ct.TensorType(name="prev", shape=(N_CODEBOOKS,), dtype=np.int32)],
            ),
            ExportPhase(
                name="detokenizer", wrapper=load_detokenizer(self.model_dir),
                dummy_inputs=(torch.randint(0, 2048, (1, DETOK_UPSAMPLE * TRACE_FRAMES, N_CODEBOOKS),
                                            dtype=torch.int32),
                              torch.arange(DETOK_UPSAMPLE * TRACE_FRAMES, dtype=torch.int32).view(1, -1)),
                mil_inputs=[
                    ct.TensorType(name="codes", shape=(1, row_axis, N_CODEBOOKS), dtype=np.int32),
                    ct.TensorType(name="positions", shape=(1, row_axis), dtype=np.int32),
                ],
                root_axis="n_codes",
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        f = self._facts
        voices = f.get("voices", {})
        constants = {
            "PROMPT_PRE": f.get("pre", [1]), "PROMPT_MID": f.get("mid", [7]), "PROMPT_TAIL": f.get("tail", [7]),
            "BOS": f.get("pre", [1])[0],
            "DEFAULT_VOICE": voices.get(DEFAULT_VOICE, [1]),
            "AUDIO_START": f.get("audio_start", 128), "END_OF_TURN": f.get("end_of_turn", 7),
            "END_OF_AUDIO": END_OF_AUDIO, "N_CODEBOOKS": N_CODEBOOKS,
            "DETOK_UPSAMPLE": DETOK_UPSAMPLE, "MAX_NEW_TOKENS": TTS_MAX_NEW_TOKENS, "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "DEFAULT_TEMPERATURE": TTS_AUDIO_TEMPERATURE, "DEFAULT_TOP_K": TTS_AUDIO_TOP_K,
        }
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(self.driver_script_path / "02_speak.lua",
                        reads=("tokens",) + tuple(constants), defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = TTS_SAMPLE_RATE
        # The voice the file carries, and what a voice file must match to replace it (ADR-045): a voice
        # here is a system prompt's ids, so the fingerprint is the tokenizer's.
        contract["tts.voices"] = [DEFAULT_VOICE]
        tokenizer = Path(self.model_dir) / "tokenizer.json"
        if tokenizer.is_file():
            from .lfm25_audio_voices import tokenizer_fingerprint

            contract["voice.compat"] = tokenizer_fingerprint(self.model_dir)
        return contract

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=False, root_axis=self.root_axis, tokenizer_dir=self.model_dir)


def _build_lfm25_audio_tts(path: Path, output_path: str) -> LoomExportConfig:
    return Lfm25AudioTtsExportConfig(output_path=output_path, model_dir=str(path))



def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=Lfm25AudioAsrExportConfig,
        recognizers=[ModelRecognizer(name="lfm2.5-audio", detect=_is_lfm25_audio, build_config=_build_lfm25_audio)],
    ))
    # The same checkpoint's other door: one directory, two tasks, so an untasked export names neither
    # and the registry asks for `--task`.
    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=Lfm25AudioTtsExportConfig,
        recognizers=[ModelRecognizer(name="lfm2.5-audio-tts", detect=_is_lfm25_audio,
                                     build_config=_build_lfm25_audio_tts)],
    ))
