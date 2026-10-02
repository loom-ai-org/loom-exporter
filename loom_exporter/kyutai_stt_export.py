"""Export Kyutai STT (`kyutai/stt-1b-en_fr`, the moshi-format release) -- a decoder-only streaming ASR
over Mimi codec codes, run the way Kyutai's own `moshi` runs it.

    24 kHz audio, + 1.5 s of trailing silence (`stt_config`: the 0.5 s text delay and 1 s more)
      -> Mimi ENCODER: causal SEANet (x960) -> 8-layer transformer (25 Hz) -> causal x2 downsample
         -> split RVQ (1 semantic + 31 acoustic codebooks): 32 codes per 80 ms frame
      -> 16-layer LM, one step per frame: text token + 32 codes, embedded and summed -> greedy text id

**The reference is `moshi`, not the transformers port** (the user's decision, 2026-10-02). The
`-trfs` port differs in two ways that are its own: it re-encodes the first frame (its window starts at
`[0, 0]`), so the codec's streaming state sees frame 0 twice and every later frame reaches the LM one
step late; and its conversion hard-codes a 375-position window, which is the 2.6B model's -- this
checkpoint's `context` is 750. Same weights; only the driver differs.

**moshi's alignment.** `run_inference` steps the FIRST frame's codes twice: step 0 sees the initial
tokens (text `text_card`, audio `card`) whatever it is handed, so the first frame would be lost; the
second step sees it. So step 0 is (initial, initial), step k >= 1 is (step k-1's text id, frame k-1's
codes), and the text of steps 1..n is the transcript. Ids 0 (`<unk>`) and 3 (`<pad>`) are dropped, as
`run_inference` drops them.

**Two sliding windows, both rings in moshi, and that decides both masks.**

* The LM attends to the last 750 positions through a `RingKVCache` of 750 -- one token per step, so
  that is exactly a 750-key window, and the engine runs it as a RING KV cache of 750 cells (loom.cpp
  ADR-066): unbounded audio, a fixed cache.
* The Mimi encoder's transformer streams TWO positions per 80 ms frame into a ring of 250. Both are
  written before either attends, so the first of each pair has already lost its oldest key: its window
  is 249, the second's 250. Measured, not reasoned: a one-call encode with that parity mask reproduces
  moshi's streamed codes 156/156 on jfk.wav, where a plain 250 window gets 144 and a causal one 127.
  The mask is built in the graph from the positions, so the driver hands over integers only.

**The encoder runs in chunks with left context, exactly.** Every output position depends on at most
8 layers x 249 positions of the transformer before it, plus the convolutions' few frames -- a finite
receptive field -- so a chunk encoded together with `CONTEXT_FRAMES` frames before it (whose own codes
are discarded) is the streamed result, at a memory cost bounded by the chunk rather than the clip.
Positions stay ABSOLUTE across chunks: RoPE is exact either way in real arithmetic, and absolute is
what moshi computes in floats.

Phases:
  - `mimi_encode`:  `(waveform, positions) -> rows`, the quantizer's input per frame.
  - `rvq_project_semantic` / `rvq_project_acoustic`, `rvq_step`: the split RVQ, one stage per call,
    the driver taking each stage's argmax (`qwen3_tts_export`'s wrappers, unchanged).
  - `lm`:           `(tokens [n, 33], position_ids, attention_mask) -> text logits` for the last row,
                    KV-cached in a 750-cell ring.

Usage:
  loom-export ~/Dev/models/kyutai-stt-1b-en-fr -o kyutai_stt.gguf \\
      --task automatic-speech-recognition --model kyutai-stt
"""
import json
import math
import sys
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
from .pocket_tts_export import _rope_tables, _rotate_pairs
from .spec_protocol import Axis, Unchecked

# A checkout of github.com/kyutai-labs/moshi; the package is its `moshi/` subdirectory.
MOSHI_REPO = "/home/flavio/Dev/moshi"

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
# The encoder transformer runs at 25 Hz: two positions per 12.5 Hz frame.
ENCODER_POSITIONS_PER_FRAME = 2
# Frames encoded per call, and the frames of left context each call carries (their codes discarded).
# The context must cover the receptive field: 8 layers x 249 transformer positions = 996 frames, plus
# the convolutions'. Checked against a one-call encode in the export's tests.
CHUNK_FRAMES = 250
CONTEXT_FRAMES = 1008
MAX_CHUNK_FRAMES = CHUNK_FRAMES + CONTEXT_FRAMES

TRACE_FRAMES = 5
TRACE_TOKENS = 7


def import_moshi():
    path = str(Path(MOSHI_REPO) / "moshi")
    if path not in sys.path:
        sys.path.insert(0, path)


def load_reference(model_dir: str):
    """`(mimi, lm, info)` through moshi's own loader, f32 on CPU, from a moshi-format directory."""
    import_moshi()
    from moshi.models import loaders

    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    info = loaders.CheckpointInfo.from_hf_repo(
        "kyutai/stt-1b-en_fr", moshi_weights=d / "model.safetensors", mimi_weights=d / config["mimi_name"],
        tokenizer=d / config["tokenizer_name"], config_path=d / "config.json")
    mimi = info.get_mimi(device="cpu").eval()
    lm = info.get_moshi(device="cpu", dtype=torch.float32).eval()
    return mimi, lm, info


def _inner_conv(streaming_conv) -> nn.Conv1d:
    """moshi's `StreamingConv1d` -> `NormConv1d` -> the `nn.Conv1d` (norm is "none" at inference)."""
    conv = streaming_conv.conv
    return conv.conv if hasattr(conv, "conv") else conv


def _causal_conv(streaming_conv, x: torch.Tensor) -> torch.Tensor:
    """`StreamingConv1d` over a whole, frame-aligned signal: `kernel - stride` columns of left padding
    and no extra padding on the right, which is zero for every layer when the input is a whole number
    of 1920-sample frames (each stride divides it).

    The padding is the stream's state before its first chunk: zeros for `pad_mode` "constant", and for
    "replicate" (the x2 downsample) the stream's FIRST column repeated -- written as a concatenation of
    that column, which needs no replicate-mode pad op."""
    conv = _inner_conv(streaming_conv)
    k, d, s = conv.kernel_size[0], conv.dilation[0], conv.stride[0]
    pad = (k - 1) * d + 1 - s
    if not pad:
        return conv(x)
    if streaming_conv.pad_mode == "constant":
        return conv(F.pad(x, (pad, 0)))
    if streaming_conv.pad_mode == "replicate":
        first = x[..., :1]
        return conv(torch.cat([first] * pad + [x], dim=-1))
    raise NotImplementedError(f"pad_mode {streaming_conv.pad_mode!r}")


class _RMSNorm(nn.Module):
    """moshi's `RMSNorm` (`rms_norm_f32`), spelled out: its `_rms_norm` is `torch.compile`-wrapped,
    which `torch.jit.trace` refuses ("using FX to torch.jit.trace a dynamo-optimized function"). The
    arithmetic is the same -- `x * (alpha * rsqrt(eps + mean(x^2)))` -- and already in f32."""

    def __init__(self, norm):
        super().__init__()
        self.alpha = nn.Parameter(norm.alpha.detach().float().view(-1).clone())
        self.eps = float(norm.eps)

    def forward(self, x):
        # `alpha * rsqrt(var)` broadcasts BOTH ways ([dim] against [1, n, 1]), which ggml's MUL cannot.
        # It went unseen while every LM call was one token (n = 1 makes it one-sided), and a 40-token
        # teacher-forced prefill failed on it -- Retro-070's lesson, met again. `expand_as` does not
        # help: MIL folds a broadcast away. `r + x * 0` carries x's shape by ARITHMETIC (exact for
        # finite x), so each multiply broadcasts one operand, in moshi's order.
        r = torch.rsqrt(self.eps + torch.mean(x * x, dim=-1, keepdim=True)) + x * 0.0
        return x * (self.alpha * r)


def _norm(norm) -> nn.Module:
    """A trace-safe stand-in for a moshi norm: `RMSNorm` re-spelled, `nn.LayerNorm` as it is."""
    return _RMSNorm(norm) if type(norm).__name__ == "RMSNorm" else norm


class _Attention(nn.Module):
    """moshi's `StreamingMultiheadAttention` (one in-projection, no per-step weights, no GQA) spelled
    for the trace: Q/K/V as three projections of the fused `in_projs[0]`, interleaved RoPE at the
    given positions, then `Q @ K^T + mask`, softmax, `@ V` -- the window `fuse_loom_attention`
    recognises. `past` is a torch-side verification hook, never traced."""

    def __init__(self, attn, max_period: float):
        super().__init__()
        if attn.kv_repeat != 1 or attn.weights_per_step or attn.cross_attention:
            raise NotImplementedError("GQA, per-step weights or cross-attention")
        w = attn.in_projs[0].weight.detach()
        e = attn.embed_dim
        self.q, self.k, self.v = (nn.Linear(e, e, bias=False) for _ in range(3))
        self.q.weight.data.copy_(w[:e])
        self.k.weight.data.copy_(w[e:2 * e])
        self.v.weight.data.copy_(w[2 * e:])
        if attn.in_projs[0].bias is not None or attn.out_projs[0].bias is not None:
            raise NotImplementedError("biased attention projections")
        self.out_proj = attn.out_projs[0]
        self.heads = attn.num_heads
        self.head_dim = e // attn.num_heads
        self.max_period = max_period

    def forward(self, x, cos, sin, mask, past=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim)
        k = self.k(x).view(b, t, self.heads, self.head_dim)
        v = self.v(x).view(b, t, self.heads, self.head_dim)
        q = q * cos + _rotate_pairs(q) * sin
        k = k * cos + _rotate_pairs(k) * sin
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past is not None:
            if past[0] is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            past[0], past[1] = k, v
        scores = torch.matmul(q * (1.0 / math.sqrt(self.head_dim)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out_proj(ctx.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


class _Layer(nn.Module):
    """moshi's `StreamingTransformerLayer`: pre-norm attention, then either the gated FFN (the LM:
    `linear_in` split into gate and value, SiLU) or a plain GELU MLP (Mimi: exact GELU, moshi's
    default activation), each optionally layer-scaled."""

    def __init__(self, layer, max_period: float):
        super().__init__()
        self.attn = _Attention(layer.self_attn, max_period)
        self.norm1, self.norm2 = _norm(layer.norm1), _norm(layer.norm2)
        self.gating = layer.gating
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        if self.gating is not None and isinstance(self.gating, nn.ModuleList):
            raise NotImplementedError("per-step gating weights")
        self.scale1 = getattr(layer.layer_scale_1, "scale", None)
        self.scale2 = getattr(layer.layer_scale_2, "scale", None)

    def forward(self, x, cos, sin, mask, past=None):
        update = self.attn(self.norm1(x), cos, sin, mask, past)
        x = x + (update if self.scale1 is None else self.scale1 * update)
        h = self.norm2(x)
        if self.gating is not None:
            gated = self.gating.linear_in(h)
            gate, value = gated.chunk(2, dim=-1)
            update = self.gating.linear_out(F.silu(gate) * value)
        else:
            update = self.linear2(F.gelu(self.linear1(h)))
        return x + (update if self.scale2 is None else self.scale2 * update)


def ring_pair_mask(positions: torch.Tensor, context: int) -> torch.Tensor:
    """The encoder transformer's mask as moshi's streaming computes it: query q attends to key k iff
    `0 <= q - k < w(q)`, with `w = context - 1` for the first position of each 80 ms frame (even q)
    and `context` for the second -- the pair is written into a `context`-slot ring before either
    reads it. Built from the positions in the graph: `q - k` as two outer products (ggml's SUB
    broadcasts one operand only), and the parity from a floor division, all exact on integers < 2^24."""
    pos = positions.to(torch.float32)[0]
    ones = torch.ones_like(pos)
    delta = torch.matmul(pos.view(-1, 1), ones.view(1, -1)) - torch.matmul(ones.view(-1, 1), pos.view(1, -1))
    odd = pos - torch.floor(pos * 0.5) * 2.0                         # 1 for the second of a pair
    window = (context - 1.0) + odd.view(-1, 1)                       # per query row
    outside = torch.relu(-delta) + torch.relu(delta - window + 1.0)
    return (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, pos.shape[0], pos.shape[0])


class MimiEncodePhase(nn.Module):
    """`(waveform, positions) -> rows`: SEANet, the encoder transformer, the x2 downsample -- the
    quantizer's input, frame-major `[1, n_frames, 512]`. `positions` are the transformer's ABSOLUTE
    positions, two per frame."""

    def __init__(self, mimi):
        super().__init__()
        self.encoder = mimi.encoder
        tr = mimi.encoder_transformer
        # Projections exist only when the transformer's width differs from the codec's; at equal
        # widths `input_proj` is None and each output projection an `nn.Identity`.
        if getattr(tr, "input_proj", None) is not None or not all(
                isinstance(p, nn.Identity) for p in getattr(tr, "output_projs", [])):
            raise NotImplementedError("a projected Mimi transformer")
        inner = tr.transformer
        if inner.positional_embedding != "rope":
            raise NotImplementedError(f"positional embedding {inner.positional_embedding!r}")
        self.layers = nn.ModuleList(_Layer(l, inner.max_period) for l in inner.layers)
        self.context = int(inner.layers[0].self_attn.context)
        self.head_dim = self.layers[0].attn.head_dim
        self.max_period = float(inner.max_period)
        self.downsample = mimi.downsample.conv

    def forward(self, waveform, positions):                      # (1, n * 1920), (1, 2n) i32
        x = waveform.unsqueeze(1)
        for block in self.encoder.model:
            name = type(block).__name__
            if name == "StreamingConv1d":
                x = _causal_conv(block, x)
            elif name == "SEANetResnetBlock":
                v = x
                for sub in block.block:
                    v = _causal_conv(sub, v) if type(sub).__name__ == "StreamingConv1d" else sub(v)
                x = x + v
            else:
                x = block(x)
        cos, sin = _rope_tables(positions, self.head_dim, self.max_period)
        mask = ring_pair_mask(positions, self.context)
        h = x.transpose(1, 2)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        x = _causal_conv(self.downsample, h.transpose(1, 2))
        return x.transpose(1, 2)


class LMPhase(nn.Module):
    """`(tokens, position_ids, attention_mask) -> text logits` for the LAST row. `tokens` is
    `[1, n, 33]`: the text id, then the 32 audio codes, each looked up in its own table and summed --
    one gather from the tables concatenated, offset per stream."""

    def __init__(self, lm):
        super().__init__()
        tables = [lm.text_emb.weight] + [e.weight for e in lm.emb]
        offsets, total = [], 0
        for t in tables:
            offsets.append(total)
            total += t.shape[0]
        self.embed = nn.Embedding(total, tables[0].shape[1])
        with torch.no_grad():
            self.embed.weight.copy_(torch.cat([t.detach() for t in tables], dim=0))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32).view(1, 1, -1))
        inner = lm.transformer
        self.layers = nn.ModuleList(_Layer(l, inner.max_period) for l in inner.layers)
        self.head_dim = self.layers[0].attn.head_dim
        self.max_period = float(inner.max_period)
        self.out_norm = _norm(lm.out_norm)
        self.text_linear = lm.text_linear

    def embed_tokens(self, tokens):
        return self.embed(tokens + self.offsets).sum(dim=2)

    def forward(self, tokens, position_ids, attention_mask, pasts=None):
        x = self.embed_tokens(tokens)
        cos, sin = _rope_tables(position_ids, self.head_dim, self.max_period)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attention_mask, None if pasts is None else pasts[i])
        return self.text_linear(self.out_norm(x[:, -1:]))


def concatenated_codebooks(mimi) -> torch.Tensor:
    """Every codebook in stage order -- the semantic one, then the 31 acoustic -- for `rvq_step`."""
    q = mimi.quantizer
    layers = list(q.rvq_first.vq.layers) + list(q.rvq_rest.vq.layers)
    return torch.cat([l._codebook.embedding.detach().float() for l in layers], dim=0)


def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


@dataclass(kw_only=True)
class KyutaiSttExportConfig(BaseMultiPhaseModelExportConfig):
    """A moshi-format Kyutai STT directory (`config.json` with `model_type: stt`, the LM and Mimi
    safetensors, the SentencePiece model) -> one Loom GGUF."""

    architecture: str = "kyutai-stt"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "kyutai_stt_driver"
    _facts: dict = field(default_factory=dict, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the moshi-format directory; the recognizer found `config.json` "
                               "with `model_type: stt` in it"),
        "decomposition": Unchecked("MultiPhase by construction -- five graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragment is parsed and checked against the "
                                         "traced topologies by LuaFragment"),
        "_facts": Unchecked("the checkpoint's numbers, read during phases()"),
    }

    def config_json(self) -> dict:
        return json.loads((Path(self.model_dir) / "config.json").read_text())

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct
        from .qwen3_tts_export import _RvqProjectWrapper, _RvqStepWrapper

        mimi, lm, info = load_reference(self.model_dir)
        cfg = self.config_json()
        if any(cfg["delays"]):
            raise NotImplementedError(f"delays {cfg['delays']}: the driver aligns streams with no delay")
        if cfg.get("dep_q", 0):
            raise NotImplementedError("a depformer: this is the speech-to-speech shape, not STT")
        stt = cfg.get("stt_config", {})
        codebooks = concatenated_codebooks(mimi)
        n_q = int(cfg["n_q"])
        self._facts = {
            "context": int(lm.transformer.layers[0].self_attn.context),
            "n_q": n_q,
            "card": int(cfg["card"]),
            "text_initial": int(lm.text_initial_token_id),
            "audio_initial": int(lm.initial_token_id),
            "pad_left": int(stt.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE),
            "pad_right": int((stt.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE),
            "text_pad": int(cfg.get("existing_text_padding_id", 3)),
        }
        if codebooks.shape[0] != n_q * self._facts["card"]:
            raise ValueError(f"{codebooks.shape[0]} codebook rows for {n_q} x {self._facts['card']}")
        width = int(mimi.quantizer.rvq_first.input_proj.weight.shape[1])
        frame_dim = ct.RangeDim(1, MAX_CHUNK_FRAMES)
        seq_dim = ct.RangeDim(1, self._facts["context"])
        hidden = int(lm.text_emb.weight.shape[1])
        return [
            ExportPhase(
                name="mimi_encode",
                wrapper=MimiEncodePhase(mimi).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES * FRAME_SAMPLES) * 0.1,
                              torch.arange(2 * TRACE_FRAMES, dtype=torch.int32).view(1, -1) + 6),
                mil_inputs=[
                    ct.TensorType(name="waveform", shape=(1, ct.RangeDim(FRAME_SAMPLES, FRAME_SAMPLES * MAX_CHUNK_FRAMES)),
                                  dtype=np.float32),
                    ct.TensorType(name="positions", shape=(1, ct.RangeDim(2, 2 * MAX_CHUNK_FRAMES)), dtype=np.int32),
                ],
                root_axis="n_codes",
                declared_axes={"waveform": {1: f"{FRAME_SAMPLES} * n_codes"},
                               "positions": {1: f"{ENCODER_POSITIONS_PER_FRAME} * n_codes"}},
            ),
            ExportPhase(
                name="rvq_project_semantic",
                wrapper=_RvqProjectWrapper(mimi.quantizer.rvq_first).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, width),),
                mil_inputs=[ct.TensorType(name="rows", shape=(1, frame_dim, width), dtype=np.float32)],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="rvq_project_acoustic",
                wrapper=_RvqProjectWrapper(mimi.quantizer.rvq_rest).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, width),),
                mil_inputs=[ct.TensorType(name="rows", shape=(1, frame_dim, width), dtype=np.float32)],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="rvq_step",
                wrapper=_RvqStepWrapper(codebooks).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, codebooks.shape[1]),
                              torch.zeros(TRACE_FRAMES, dtype=torch.int32),
                              torch.arange(self._facts["card"], dtype=torch.int32),
                              torch.arange(self._facts["card"], dtype=torch.int32) + self._facts["card"],
                              torch.tensor([1.0])),
                mil_inputs=[
                    ct.TensorType(name="rows", shape=(1, frame_dim, codebooks.shape[1]), dtype=np.float32),
                    ct.TensorType(name="prev_ids", shape=(frame_dim,), dtype=np.int32),
                    ct.TensorType(name="prev_codebook", shape=(self._facts["card"],), dtype=np.int32),
                    ct.TensorType(name="next_codebook", shape=(self._facts["card"],), dtype=np.int32),
                    ct.TensorType(name="subtract", shape=(1,), dtype=np.float32),
                ],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="lm",
                wrapper=LMPhase(lm).eval(),
                dummy_inputs=(torch.randint(0, 2000, (1, TRACE_TOKENS, n_q + 1), dtype=torch.int32),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="tokens", shape=(1, seq_dim, n_q + 1), dtype=np.int32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim), dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=self._facts["context"],
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        f = self._facts
        constants = {
            "FRAME_SAMPLES": FRAME_SAMPLES, "CHUNK_FRAMES": CHUNK_FRAMES, "CONTEXT_FRAMES": CONTEXT_FRAMES,
            "N_Q": f.get("n_q", 32), "CARD": f.get("card", 2048), "LM_CONTEXT": f.get("context", 750),
            "TEXT_INITIAL": f.get("text_initial", 8000), "AUDIO_INITIAL": f.get("audio_initial", 2048),
            "PAD_LEFT": f.get("pad_left", 0), "PAD_RIGHT": f.get("pad_right", 36000),
            "TEXT_PAD": f.get("text_pad", 3), "TEXT_UNK": 0,
        }
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
        contract["sample_rate"] = SAMPLE_RATE
        return contract

    def backend_kwargs(self) -> dict:
        cfg = self.config_json() if (Path(self.model_dir) / "config.json").is_file() else {}
        return dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                    tokenizer_dir=self.model_dir, tokenizer_family="sentencepiece_proto",
                    tokenizer_proto_name=cfg.get("tokenizer_name"), kv_cache_ring=True)


def _is_kyutai_stt(path: Path) -> bool:
    """A moshi-format STT release: `config.json` declaring `model_type: stt` beside its weights."""
    cfg = path / "config.json"
    if not (path.is_dir() and cfg.is_file() and (path / "model.safetensors").is_file()):
        return False
    try:
        c = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    return c.get("model_type") == "stt" and "mimi_name" in c and "tokenizer_name" in c


def _build_kyutai_stt(path: Path, output_path: str) -> LoomExportConfig:
    return KyutaiSttExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=KyutaiSttExportConfig,
        recognizers=[ModelRecognizer(name="kyutai-stt", detect=_is_kyutai_stt, build_config=_build_kyutai_stt)],
    ))
