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

The live door (loom.cpp ADR-108), beside the one pass, which it leaves unchanged:
  - `mimi_encode_step`: `(waveform, positions, key_positions, first, state_1..27) -> (rows, state')`, the
                    encoder a few frames per call with moshi's streaming state carried (every causal conv's
                    last inputs, the transformer's last 250 keys and values per layer); `mimi_encode_init`
                    makes a fresh state on the device. The one-call encoder to 1e-15 at f64.
  - `rvq_encode`:   `rows -> codes`, the split RVQ in one graph, its argmax on the device -- the 34 calls of
                    the chain cost more than a frame lasts when a push is a frame or two.

Usage:
  loom-export ~/Dev/models/kyutai-stt-1b-en-fr -o kyutai_stt.gguf \\
      --task automatic-speech-recognition --model kyutai-stt
"""
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
# The live door's encoder step (loom.cpp ADR-108) takes up to this many frames (2 s) per call; the driver
# splits a longer push.
MAX_STEP_FRAMES = 25

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


def _conv_pad(conv: nn.Conv1d) -> int:
    """The left padding a causal conv takes, and so the input columns its streaming state keeps."""
    return (conv.kernel_size[0] - 1) * conv.dilation[0] + 1 - conv.stride[0]


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


def ring_pair_key_mask(positions: torch.Tensor, key_positions: torch.Tensor, context: int) -> torch.Tensor:
    """`ring_pair_mask` with the keys at positions of their own: the carried window first, then the call's
    own. A key at a negative position is a slot no frame has written, and is masked, so a stream's first
    call reads what a fresh encoder reads. Square and over the same positions, this is `ring_pair_mask`."""
    q = positions.to(torch.float32)[0]
    k = key_positions.to(torch.float32)[0]
    delta = (torch.matmul(q.view(-1, 1), torch.ones_like(k).view(1, -1))
             - torch.matmul(torch.ones_like(q).view(-1, 1), k.view(1, -1)))
    odd = q - torch.floor(q * 0.5) * 2.0
    window = (context - 1.0) + odd.view(-1, 1)
    outside = torch.relu(-delta) + torch.relu(delta - window + 1.0) + torch.relu(-k).view(1, -1)
    return (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, q.shape[0], k.shape[0])


# The streaming state's conv step, shared with every decoder that streams (`stream_steps`).
from .stream_steps import conv_step as _conv_step  # noqa: E402


class MimiEncodeStepPhase(nn.Module):
    """`(waveform, positions, key_positions, first, state_1..state_N) -> (rows, state_1'..state_N')`: the
    next `n` frames of `MimiEncodePhase`, with the encoder's streaming state carried in and out -- what
    moshi's `mimi.streaming()` keeps between `encode` calls (loom.cpp ADR-108).

    The states, in the order `forward` meets them: each SEANet convolution's last `(k - 1) d + 1 - s`
    input samples; each transformer layer's last `context` keys (after RoPE) and values,
    `[1, heads, context, head_dim]`; the x2 downsample's last inputs. `positions` are the `2 n` new
    transformer positions, absolute; `key_positions` the `context + 2 n` the keys sit at, the state's
    first.

    `first` is 1 on a stream's first call and 0 after. The downsample pads by REPLICATE: moshi's streaming
    conv overwrites its state with the first input column on its first call (`state.first`). So the
    downsample's previous columns are `state * (1 - first) + x[:, :, :1] * first`, exact for `first` 0 or
    1. The constant-padded SEANet convolutions start from zeros, which the fresh state is."""

    def __init__(self, encoder: MimiEncodePhase):
        super().__init__()
        self.e = encoder
        self.context = encoder.context

    def _stateful(self) -> List[nn.Conv1d]:
        """The SEANet convolutions that carry a state, in the order `forward` meets them."""
        out = []
        for block in self.e.encoder.model:
            name = type(block).__name__
            if name == "StreamingConv1d":
                out.append(_inner_conv(block))
            elif name == "SEANetResnetBlock":
                out += [_inner_conv(sub) for sub in block.block if type(sub).__name__ == "StreamingConv1d"]
        return [c for c in out if _conv_pad(c) > 0]

    def state_shapes(self) -> List[Tuple[int, ...]]:
        shapes = [(1, c.in_channels, _conv_pad(c)) for c in self._stateful()]
        for layer in self.e.layers:
            kv = (1, layer.attn.heads, self.context, layer.attn.head_dim)
            shapes += [kv, kv]
        down = _inner_conv(self.e.downsample)
        shapes.append((1, down.in_channels, _conv_pad(down)))
        return shapes

    def _attend(self, attn: _Attention, x, cos, sin, mask, k_past, v_past):
        b, t, _ = x.shape
        q = attn.q(x).view(b, t, attn.heads, attn.head_dim)
        k = attn.k(x).view(b, t, attn.heads, attn.head_dim)
        v = attn.v(x).view(b, t, attn.heads, attn.head_dim)
        q = q * cos + _rotate_pairs(q) * sin
        k = k * cos + _rotate_pairs(k) * sin
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = torch.cat([k_past, k], dim=2)
        v = torch.cat([v_past, v], dim=2)
        scores = torch.matmul(q * (1.0 / math.sqrt(attn.head_dim)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = attn.out_proj(ctx.transpose(1, 2).reshape(b, t, attn.heads * attn.head_dim))
        return out, k[:, :, -self.context:], v[:, :, -self.context:]

    def forward(self, waveform, positions, key_positions, first, *states):   # (1, 1920n), (1, 2n), (1, ctx+2n), (1,)
        e = self.e
        states = list(states)
        out_states = []
        stateful = {id(c) for c in self._stateful()}

        def conv(streaming_conv, v):
            c = _inner_conv(streaming_conv)
            if id(c) not in stateful:
                return c(v)
            y, s = _conv_step(c, v, states.pop(0))
            out_states.append(s)
            return y

        x = waveform.unsqueeze(1)
        for block in e.encoder.model:
            name = type(block).__name__
            if name == "StreamingConv1d":
                x = conv(block, x)
            elif name == "SEANetResnetBlock":
                v = x
                for sub in block.block:
                    v = conv(sub, v) if type(sub).__name__ == "StreamingConv1d" else sub(v)
                x = x + v
            else:
                x = block(x)
        cos, sin = _rope_tables(positions, e.head_dim, e.max_period)
        mask = ring_pair_key_mask(positions, key_positions, self.context)
        h = x.transpose(1, 2)
        for layer in e.layers:
            update, k, v = self._attend(layer.attn, layer.norm1(h), cos, sin, mask, states.pop(0), states.pop(0))
            h = h + (update if layer.scale1 is None else layer.scale1 * update)
            h2 = layer.norm2(h)
            if layer.gating is not None:
                gate, value = layer.gating.linear_in(h2).chunk(2, dim=-1)
                update = layer.gating.linear_out(F.silu(gate) * value)
            else:
                update = layer.linear2(F.gelu(layer.linear1(h2)))
            h = h + (update if layer.scale2 is None else layer.scale2 * update)
            out_states += [k, v]
        x = h.transpose(1, 2)
        if e.downsample.pad_mode != "replicate":
            raise NotImplementedError(f"downsample pad_mode {e.downsample.pad_mode!r}")
        previous = states.pop(0) * (1.0 - first) + x[..., :1] * first
        x, s = _conv_step(_inner_conv(e.downsample), x, previous)
        out_states.append(s)
        return (x.transpose(1, 2), *out_states)


# `mimi_encode_step`'s fresh state, made on the device from one number (`stream_steps.StateInit`).
from .stream_steps import StateInit as MimiEncodeInitPhase  # noqa: E402


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


class RvqEncodePhase(nn.Module):
    """`rows -> codes [1, n, n_q]`, as floats: the split RVQ in ONE graph, for the live door (ADR-108).

    The one pass runs a stage per call (`rvq_step`), the driver taking each argmax: 34 calls a chunk, which
    cost nothing over 250 frames. A stream pushes a frame or two at a time, and there those calls were ~80 ms a
    push on the dev box (per-call overhead, not compute) against an 80 ms frame -- more than the Q8_0 LM step
    (68 ms). So here the argmax stays in the graph (the engine's ARGMAX, P4.53) and each stage's centroid is
    gathered by it: the same `2 r.e - |e|^2` scores `rvq_step` computes, the same subtraction. The ids come
    out as floats (exact below 2^24), frame-major, the order the LM reads them.

    Ties: ggml's CPU ARGMAX keeps the last tied index where `argmax_rows` and torch keep the first. No
    codebook of this checkpoint has a repeated row, and the gate compares this graph's codes with the chain's
    on every clip."""

    def __init__(self, quantizer):
        super().__init__()
        self.first = _RvqProject(quantizer.rvq_first)
        self.rest = _RvqProject(quantizer.rvq_rest)
        layers = list(quantizer.rvq_first.vq.layers) + list(quantizer.rvq_rest.vq.layers)
        self.books = nn.ModuleList()
        for l in layers:
            e = nn.Embedding(*l._codebook.embedding.shape)
            with torch.no_grad():
                e.weight.copy_(l._codebook.embedding.detach().float())
            self.books.append(e)

    @staticmethod
    def _draw(residual, book: nn.Embedding):
        embed = book.weight
        scores = residual @ embed.t() * 2.0 - (embed * embed).sum(dim=-1).view(1, 1, -1)
        return torch.argmax(scores, dim=-1)                              # [1, n]

    def forward(self, rows):                                            # (1, n, width)
        ids = [self._draw(self.first(rows), self.books[0])]
        residual = self.rest(rows)
        for q in range(1, len(self.books)):
            if q > 1:
                residual = residual - self.books[q - 1](ids[-1])
            ids.append(self._draw(residual, self.books[q]))
        return torch.stack([i.to(torch.float32) for i in ids], dim=-1)


class _RvqProject(nn.Module):
    """One RVQ's kernel-1 input projection as a matmul on rows (`qwen3_tts_export._RvqProjectWrapper`)."""

    def __init__(self, rvq):
        super().__init__()
        self.input_proj = rvq.input_proj

    def forward(self, rows):
        if self.input_proj is None:
            return rows
        return self.input_proj(rows.transpose(1, 2)).transpose(1, 2)


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
        encode = MimiEncodePhase(mimi).eval()
        step = MimiEncodeStepPhase(encode).eval()
        step_shapes = step.state_shapes()
        self._facts["mimi_states"] = len(step_shapes)
        self._facts["mimi_context"] = step.context
        ctx = step.context
        return [
            ExportPhase(
                name="mimi_encode",
                wrapper=encode,
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
            # The live door (loom.cpp ADR-108): the encoder a few frames per call, its streaming state
            # carried in and out as `state_1..state_N`, which the driver keeps on the device;
            # `mimi_encode_init` makes a fresh one there. `mimi_encode` above is unchanged.
            ExportPhase(
                name="mimi_encode_init",
                wrapper=MimiEncodeInitPhase(step_shapes).eval(),
                dummy_inputs=(torch.zeros(1),),
                mil_inputs=[ct.TensorType(name="zero", shape=(1,), dtype=np.float32)],
            ),
            ExportPhase(
                name="mimi_encode_step",
                wrapper=step,
                dummy_inputs=(torch.randn(1, TRACE_FRAMES * FRAME_SAMPLES) * 0.1,
                              torch.arange(2 * TRACE_FRAMES, dtype=torch.int32).view(1, -1) + 6,
                              torch.arange(ctx + 2 * TRACE_FRAMES, dtype=torch.int32).view(1, -1) + 6 - ctx,
                              torch.tensor([0.0]))
                             + tuple(torch.randn(sh) * 0.1 for sh in step_shapes),
                mil_inputs=[
                    ct.TensorType(name="waveform",
                                  shape=(1, ct.RangeDim(FRAME_SAMPLES, FRAME_SAMPLES * MAX_STEP_FRAMES)),
                                  dtype=np.float32),
                    ct.TensorType(name="positions", shape=(1, ct.RangeDim(2, 2 * MAX_STEP_FRAMES)), dtype=np.int32),
                    ct.TensorType(name="key_positions", shape=(1, ct.RangeDim(ctx + 2, ctx + 2 * MAX_STEP_FRAMES)),
                                  dtype=np.int32),
                    ct.TensorType(name="first", shape=(1,), dtype=np.float32),
                ] + [ct.TensorType(name=f"state_{i + 1}", shape=sh, dtype=np.float32)
                     for i, sh in enumerate(step_shapes)],
                root_axis="n_codes",
                declared_axes={"waveform": {1: f"{FRAME_SAMPLES} * n_codes"},
                               "positions": {1: f"{ENCODER_POSITIONS_PER_FRAME} * n_codes"},
                               "key_positions": {1: f"{ENCODER_POSITIONS_PER_FRAME} * n_codes + {ctx}"}},
            ),
            ExportPhase(
                name="rvq_encode",
                wrapper=RvqEncodePhase(mimi.quantizer).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, width),),
                mil_inputs=[ct.TensorType(name="rows", shape=(1, ct.RangeDim(1, MAX_STEP_FRAMES), width),
                                          dtype=np.float32)],
                root_axis="n_codes",
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
        # The live door (loom.cpp ADR-108), beside the one pass. Top level, so its constants are too; the
        # one-pass block keeps its own locals.
        stream_constants = {"S_" + k: v for k, v in constants.items()}
        stream_constants.update({"MIMI_STATES": f.get("mimi_states", 0), "MIMI_CONTEXT": f.get("mimi_context", 250),
                                 "MAX_STEP_FRAMES": MAX_STEP_FRAMES})
        return [
            ExportConstants(values=stream_constants, top_level=True),
            LuaFragment(self.driver_script_path / "00_stream.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish")),
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
