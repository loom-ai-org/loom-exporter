"""Export Pocket-TTS (`kyutai/pocket-tts`, the English model) -- family 9's fifth leaf, and the first
whose autoregressive loop carries CONTINUOUS latents rather than tokens.

Pocket-TTS is a flow language model over the latent space of a Mimi codec:

    text -> SentencePiece ids -> [voice KV | text] prefill of a 6-layer transformer
         -> per step: last hidden row -> EOS logit, and a one-step flow head (LSD) over a Gaussian
                      draw -> the next 32-d latent, which is ALSO the next step's input
         -> Mimi decoder (latents -> 12.5 Hz x 16 upsample -> 2-layer transformer -> SEANet) -> 24 kHz

Nothing samples a token and nothing takes an argmax. The loop's state is a float vector, so its
shape is family 10's KV-cached decoder with the sampler replaced by one flow-head call; the Lua is
hand-written, as Dia's and Chatterbox's are.

Five phases:
  - `text_embed`:  the text's ids -> the conditioner's lookup-table rows, the prefill's embeddings.
  - `lm`:          the flow LM's transformer, KV-cached, returning the LAST row after `out_norm` and
                   its EOS logit. The voice is NOT an input: it is a precomputed KV cache (below).
  - `step_embed`:  one latent -> `input_linear(latent)`, the next step's input row. Step 0's latent
                   is the checkpoint's `bos_emb`, which the reference substitutes for a NaN.
  - `flow_head`:   `SimpleMLPAdaLN(c, s, t, x)`, one velocity. The released models integrate it over
                   ONE step (`lsd_decode` with `s = 0, t = 1`), which the driver does in Lua.
  - `mimi_decoder`: every latent in one call -> the waveform. The reference decodes in streamed chunks
                   through causal convolutions and a windowed attention; one call computes the same
                   thing (`loom.cpp/scripts/pocket_tts_reference.py` checks it: 3.4e-06).

**A voice is a KV cache.** The reference's predefined voices are the flow LM's streaming state after
prefilling `[bos_before_voice | speaker_proj(mimi_encoder(audio))]`, saved as safetensors
(`export-voice` writes the same format for a user's own clip). Those rows cannot be recovered from the
cache, so the export ships the CACHE: the default voice's K and V for every layer as a driver weight,
which the driver writes into `lm`'s cache with `loom.seed_kv` before the text prefill (loom.cpp
ADR-043). The other voices are separate voice files (`pocket_tts_voices.py`), which the model accepts
because its contract declares the same weights fingerprint they carry (`loom.voice.compat`). Encoding a
new voice from audio needs the Mimi encoder and is not in this export.

Usage:
  loom-export ~/Dev/models/pocket-tts/languages/english_2026-09 -o pocket_tts.gguf \\
      --task text-to-speech --model pocket-tts
"""
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .pocket_tts_tokenizer_export import (
    LONG_CHUNK_FRAMES_AFTER_EOS, SHORT_CHUNK_FRAMES_AFTER_EOS, SHORT_CHUNK_MAX_WORDS,
)
from .spec_protocol import Axis, Unchecked

# Where the reference checkout lives: `pocket-tts` on PyPI would do too, but the export must read the
# checkpoint's YAML by name, and the clone is what `pocket_tts_reference.py` runs. `LOOM_POCKET_TTS_REPO`
# points elsewhere: the small English models (`english_{18m,7m,2m}_refl_2026-10`) need v4.0.1 or later,
# whose configs and modules (grouped-query attention, a low-rank text table, `z_proj`, one-step flow heads,
# a separate Mimi decoder) v3.2.0 does not have.
POCKET_TTS_REPO = os.environ.get("LOOM_POCKET_TTS_REPO", "/home/flavio/Dev/pocket-tts")

SAMPLE_RATE = 24000
# Mimi's latent rate and its decoder's: 12.5 Hz latents, upsampled x16 to the SEANet's 200 Hz.
MIMI_UPSAMPLE = 16
# Waveform samples per latent: 16 x the SEANet's hop (6 * 5 * 4).
SAMPLES_PER_FRAME = 1920
# `TTSModel._TOKENS_PER_SECOND_ESTIMATE`, `_GEN_SECONDS_PADDING`, `_MIN_FRAMES_BEFORE_EOS`, and
# `default_parameters.py`'s generation defaults: the numbers every published sample used.
TOKENS_PER_SECOND_ESTIMATE = 3.0
GEN_SECONDS_PADDING = 2.0
MIN_FRAMES_BEFORE_EOS = 6
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_DECODE_STEPS = 1
# The longest text the reference feeds one generation: longer inputs are split into sentences first.
MAX_TOKEN_PER_CHUNK = 50
# `generate_audio_stream`: `frames_after_eos_guess += 2` on top of `prepare_text_prompt`'s guess.
FRAMES_AFTER_EOS_PADDING = 2
# How many positions `lm`'s KV cache holds: the voice (126 rows for every predefined voice, 30 s = 376
# at most for a cloned one), a 50-token chunk, and the reference's own frame budget for it,
# `ceil((50 / 3 + 2) * 12.5) = 234`. 1024 covers the longest voice with room to spare, and costs
# 6 layers x 2 x 1024 x 1024 x 4 bytes = 48 MB.
LM_MAX_POSITIONS = 1024

# The flow heads the driver runs. `lsd` (the full-size models) takes two time conditions and is
# integrated over `n_steps`; `drifting` and `pointwise` (the small 2026-10 models) take none and are one
# step by construction: `head(noise)` and `noise + head(noise)`. `flow_matching` needs >= 16 Euler steps
# and no released checkpoint uses it.
FLOW_TYPES = ("lsd", "drifting", "pointwise")

# Trace lengths: odd and distinct from every static dimension (the Qwen3-TTS lesson about length 8).
TRACE_TOKENS = 13
TRACE_STEPS = 7
TRACE_FRAMES = 5


def import_pocket_tts():
    if POCKET_TTS_REPO not in sys.path:
        sys.path.insert(0, POCKET_TTS_REPO)


# ------------------------------------------------------------------------------------ shared parts --

def _attn_dims(attn) -> Tuple[int, int, int]:
    """`(attn_dim, num_kv_heads, head_dim)`. v4's attention has queries `attn_dim` wide and keys and
    values on `num_kv_heads` heads; before v4 (and in every full-size model) both are the defaults,
    `embed_dim` and `num_heads`, and v3.2.0's module has neither attribute."""
    attn_dim = getattr(attn, "attn_dim", attn.embed_dim)
    return attn_dim, getattr(attn, "num_kv_heads", attn.num_heads), attn_dim // attn.num_heads


def _split_in_proj(attn) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`in_proj`'s weight as Q `[attn_dim, embed]`, K and V `[kv_heads * head_dim, embed]`. The reference
    slices Q off the front and views the rest as `(b, t, 2, kv_heads, head_dim)` -- before v4, all of it as
    `(b, t, 3, heads, head_dim)`, the same rows -- and three 4-D projections replace a 5-D view that ggml
    has no rank for."""
    w = attn.in_proj.weight.detach()
    attn_dim, kv_heads, head_dim = _attn_dims(attn)
    kv = kv_heads * head_dim
    if w.shape[0] != attn_dim + 2 * kv:
        raise ValueError(f"in_proj has {w.shape[0]} rows, not attn_dim + 2 * kv = {attn_dim + 2 * kv}")
    return w[:attn_dim].clone(), w[attn_dim:attn_dim + kv].clone(), w[attn_dim + kv:].clone()


def _rope_tables(positions: torch.Tensor, head_dim: int, max_period: float):
    """`[1, t, 1, head_dim]` cos and sin for `apply_rope`'s INTERLEAVED pairs (`(x[2i], x[2i+1])`).

    Each frequency appears twice, once per member of its pair. The angle is `freq * position` in f32,
    the product the reference forms from its own `ds` and `ts`."""
    ds = torch.arange(head_dim // 2, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / head_dim))
    freqs = torch.repeat_interleave(freqs, 2)
    angles = positions.to(torch.float32)[0].view(1, -1, 1, 1) * freqs.view(1, 1, 1, head_dim)
    return torch.cos(angles), torch.sin(angles)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """`(x0, x1) -> (-x1, x0)` for every interleaved pair of a `[b, t, heads, head_dim]` tensor, in
    four dimensions: the pairs are the last axis of a `[b, t, heads * head_dim / 2, 2]` view."""
    b, t, h, d = x.shape
    pairs = x.reshape(b, t, h * d // 2, 2)
    return torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(b, t, h, d)


class _Attention(nn.Module):
    """`StreamingMultiheadAttention` re-spelled for the trace, and exact against it.

    RoPE: `q * cos + rotate_pairs(q) * sin` is `apply_rope`'s `(qr*cos - qi*sin, qr*sin + qi*cos)`
    term for term. The attention is HF's spelling -- Q scaled, `Q @ K^T + mask`, softmax, `@ V`,
    transpose, reshape -- which is the window `fuse_loom_attention` recognises, so `lm`'s layers become
    ATTENTION nodes with a cache, and the Mimi decoder's (not fused) stay plain ops.

    `past` is a torch-side verification hook, never traced: `(k, v)` rows to attend over before this
    call's own, returned updated, which is what the engine's cache does between calls."""

    def __init__(self, attn, max_period: float):
        super().__init__()
        wq, wk, wv = _split_in_proj(attn)
        attn_dim, self.kv_heads, self.head_dim = _attn_dims(attn)
        self.q = nn.Linear(attn.embed_dim, attn_dim, bias=False)
        self.k = nn.Linear(attn.embed_dim, wk.shape[0], bias=False)
        self.v = nn.Linear(attn.embed_dim, wv.shape[0], bias=False)
        self.q.weight.data.copy_(wq)
        self.k.weight.data.copy_(wk)
        self.v.weight.data.copy_(wv)
        self.out_proj = attn.out_proj
        self.heads = attn.num_heads
        self.max_period = max_period

    def _repeat_kv(self, x):
        """`(b, kv_heads, t, d) -> (b, heads, t, d)`, query head `h` reading KV head `h // n_rep`: the
        reference's `repeat_interleave(n_rep, dim=1)`, in HF's `repeat_kv` spelling, which
        `fuse_gqa_repeat_kv` recognises -- so the ATTENTION node gets the un-repeated K and V and the
        cache stores `kv_heads` heads, the rows a voice state holds."""
        n_rep = self.heads // self.kv_heads
        if n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)

    def forward(self, x, cos, sin, mask, past=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim)
        k = self.k(x).view(b, t, self.kv_heads, self.head_dim)
        v = self.v(x).view(b, t, self.kv_heads, self.head_dim)
        q = q * cos + _rotate_pairs(q) * sin
        k = k * cos + _rotate_pairs(k) * sin
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past is not None:
            if past[0] is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            past[0], past[1] = k, v
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        scores = torch.matmul(q * (1.0 / math.sqrt(self.head_dim)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        ctx = ctx.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.out_proj(ctx)


class _Layer(nn.Module):
    """`StreamingTransformerLayer`: pre-norm attention and a tanh-GELU MLP, each optionally scaled."""

    def __init__(self, layer, max_period: float):
        super().__init__()
        self.attn = _Attention(layer.self_attn, max_period)
        self.norm1, self.norm2 = layer.norm1, layer.norm2
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.scale1 = getattr(layer.layer_scale_1, "scale", None)
        self.scale2 = getattr(layer.layer_scale_2, "scale", None)

    def forward(self, x, cos, sin, mask, past=None):
        update = self.attn(self.norm1(x), cos, sin, mask, past)
        x = x + (update if self.scale1 is None else self.scale1 * update)
        update = self.linear2(F.gelu(self.linear1(self.norm2(x)), approximate="tanh"))
        return x + (update if self.scale2 is None else self.scale2 * update)


# ----------------------------------------------------------------------------------------- flow LM --

class TextEmbedPhase(nn.Module):
    """`(text_ids) -> conditioner(text_ids)`, `[1, n, d_model]`: the table's rows, through the bias-free
    `proj` when the table is low-rank (v4's `lookup_table.rank`, the 7m and 2m models)."""

    def __init__(self, flow_lm):
        super().__init__()
        self.embed = flow_lm.conditioner.embed
        self.proj = getattr(flow_lm.conditioner, "proj", None)

    def forward(self, text_ids):                                        # (1, n) i32
        x = self.embed(text_ids)
        return x if self.proj is None else self.proj(x)


class LMPhase(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> (hidden, eos_logit)` for the LAST row.

    `FlowLMModel.backbone` + `out_eos`: the transformer, `out_norm`, the last row, v4's bias-free
    `z_proj` when the config sets `z_dim` (the small models: 64 wide), and the EOS head whose logit the
    driver compares with the threshold. `hidden` is what the flow head is conditioned on. The reference
    computes the comparison in the graph; returning the logit keeps the threshold a caller's knob."""

    def __init__(self, flow_lm, max_period: float):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(l, max_period) for l in flow_lm.transformer.layers)
        self.out_norm = flow_lm.out_norm
        self.z_proj = getattr(flow_lm, "z_proj", None)
        self.out_eos = flow_lm.out_eos
        self.head_dim = self.layers[0].attn.head_dim
        self.max_period = max_period

    def forward(self, inputs_embeds, position_ids, attention_mask, pasts=None):
        cos, sin = _rope_tables(position_ids, self.head_dim, self.max_period)
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attention_mask, None if pasts is None else pasts[i])
        hidden = self.out_norm(x)[:, -1:]
        if self.z_proj is not None:
            hidden = self.z_proj(hidden)
        return hidden, self.out_eos(hidden)


class StepEmbedPhase(nn.Module):
    """`(latent) -> input_linear(latent)`, one `[1, 1, 1024]` row."""

    def __init__(self, flow_lm):
        super().__init__()
        self.input_linear = flow_lm.input_linear

    def forward(self, latent):                                          # (1, 1, 32)
        return self.input_linear(latent)


def _layer_norm(x, weight, bias, eps):
    """The flow head's own `LayerNorm` (biased variance), spelled with means so no `var` op traces."""
    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    y = centred / torch.sqrt((centred * centred).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight + bias


def _rms_norm(x, alpha, eps):
    """The flow head's `RMSNorm`, which divides by the UNBIASED variance (`x.var()`'s default), not by
    the mean square: `x * alpha / sqrt(eps + var(x))`, with `n - 1` spelled out."""
    n = x.shape[-1]
    centred = x - x.mean(dim=-1, keepdim=True)
    var = (centred * centred).sum(dim=-1, keepdim=True) / (n - 1)
    return x * (alpha * torch.rsqrt(var + eps))


class FlowHeadPhase(nn.Module):
    """`(c, s, t, x, x_scale, n_steps) -> x' + SimpleMLPAdaLN(c, s, t, x') / n_steps` with
    `x' = x * x_scale`: one `lsd_decode` step, `[1, 32]`.

    Two time embeddings, averaged, because the released models are LSD-distilled: the head is told
    where the step starts (`s`) and ends (`t`). The scale and the update are IN the graph rather than
    in the driver because they are f32 arithmetic in the reference -- `normal_(std=sqrt(temp))` is
    `z * std` and the update is `current + flow_dir / num_steps` -- and the driver's Lua (LuaJIT, 5.1)
    has doubles only. The first step passes the unit draw with `x_scale = sqrt(temperature)`, every
    later one its own output with 1."""

    def __init__(self, flow_net):
        super().__init__()
        self.net = flow_net

    def _time(self, emb, t):
        args = t * emb.freqs
        h = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        lin1, _, lin2, rms = emb.mlp
        h = lin2(F.silu(lin1(h)))
        return _rms_norm(h, rms.alpha, rms.eps)

    def forward(self, c, s, t, x, x_scale, n_steps):         # (1,1024) (1,1) (1,1) (1,32) (1,1) (1,1)
        x = x * x_scale
        return x + self.velocity(c, s, t, x) / n_steps

    def velocity(self, c, s, t, x):
        net = self.net
        x = net.input_proj(x)
        y = net.cond_embed(c)
        y = y + (self._time(net.time_embed[0], s) + self._time(net.time_embed[1], t)) / 2
        return self.blocks(x, y)

    def blocks(self, x, y):
        """`SimpleMLPAdaLN.forward` after the time conditions, on the PROJECTED input: the residual
        blocks and the final layer. (`input_proj` stays first in each caller, the op order the published
        full-size export traced, so a re-export names its constants as before.)"""
        net = self.net
        for block in net.res_blocks:
            shift, scale, gate = block.adaLN_modulation(y).chunk(3, dim=-1)
            h = _layer_norm(x, block.in_ln.weight, block.in_ln.bias, block.in_ln.eps)
            h = block.mlp(h * (1 + scale) + shift)
            x = x + gate * h
        final = net.final_layer
        shift, scale = final.adaLN_modulation(y).chunk(2, dim=-1)
        x = _layer_norm(x, None, None, final.norm_final.eps) * (1 + scale) + shift
        return final.linear(x)


class OneStepFlowHeadPhase(FlowHeadPhase):
    """`(c, x, x_scale) -> head(x')` (`drifting`) or `x' + head(x')` (`pointwise`), `x' = x * x_scale`:
    the small models' heads, which take no time condition and are one step by construction
    (`drifting_decode`, `pointwise_decode`). `y` is `cond_embed(c)` alone -- the reference adds the time
    embeddings' mean only `if self.num_time_conds`."""

    def __init__(self, flow_net, flow_type: str):
        super().__init__(flow_net)
        if flow_type not in ("drifting", "pointwise") or len(flow_net.time_embed) != 0:
            raise ValueError(f"a one-step head is drifting or pointwise with no time embedding, not "
                             f"{flow_type!r} with {len(flow_net.time_embed)}")
        self.residual = flow_type == "pointwise"

    def forward(self, c, x, x_scale):                                   # (1, cond) (1, 32) (1, 1)
        x = x * x_scale
        h = self.net.input_proj(x)
        update = self.blocks(h, self.net.cond_embed(c))
        return x + update if self.residual else update


# -------------------------------------------------------------------------------------------- Mimi --

def _causal_conv(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """`StreamingConv1d` on its first call: the state it prepends is zeros of `(k - 1) * d + 1 - s`."""
    k, d, s = conv.kernel_size[0], conv.dilation[0], conv.stride[0]
    pad = (k - 1) * d + 1 - s
    return conv(F.pad(x, (pad, 0)) if pad else x)


def _causal_conv_transpose(convtr: nn.ConvTranspose1d, x: torch.Tensor) -> torch.Tensor:
    """`StreamingConvTranspose1d` over the whole sequence: the last `k - s` samples are the partial
    overlap the reference carries into its NEXT call, and no call follows the last one."""
    y = convtr(x)
    tail = convtr.kernel_size[0] - convtr.stride[0]
    return y[..., :-tail] if tail else y


class MimiDecoderPhase(nn.Module):
    """`(latents, positions) -> waveform`, every frame in one call.

    `MimiModel.decode_from_latent` after `* emb_std + emb_mean`: the quantizer's 1x1 projection, the
    depthwise x16 transposed convolution, the 2-layer transformer (RoPE, layer scale, a causal window
    of `context` frames), and the SEANet decoder. `positions` is `0 .. 16n - 1`, handed over by the
    driver so the window mask is built from a graph input rather than from a shape."""

    def __init__(self, flow_lm, mimi):
        super().__init__()
        self.register_buffer("emb_std", flow_lm.emb_std.detach().clone().float())
        self.register_buffer("emb_mean", flow_lm.emb_mean.detach().clone().float())
        self.quantizer = mimi.quantizer.output_proj
        self.upsample = mimi.upsample.convtr.convtr
        transformer = mimi.decoder_transformer.transformer
        self.layers = nn.ModuleList(_Layer(l, transformer.max_period) for l in transformer.layers)
        self.head_dim = self.layers[0].attn.head_dim
        self.max_period = transformer.max_period
        self.context = transformer.layers[0].self_attn.context
        self.decoder = mimi.decoder

    def window_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """0 where `0 <= q - k < context`, else a large negative. The reference's is a boolean mask
        handed to SDPA (-inf); every row keeps its own diagonal, so both underflow to exactly 0."""
        pos = positions.to(torch.float32)[0]
        # `q - k` as two outer products rather than `pos[:, None] - pos[None, :]`: ggml's SUB broadcasts
        # its second operand only, and this is a MUTUAL broadcast. A K=1 product of integers below 2^24
        # is exact.
        ones = torch.ones_like(pos)
        delta = torch.matmul(pos.view(-1, 1), ones.view(1, -1)) - torch.matmul(ones.view(-1, 1),
                                                                               pos.view(1, -1))
        outside = torch.relu(-delta) + torch.relu(delta - (self.context - 1))
        return (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, pos.shape[0], pos.shape[0])

    def forward(self, latents, positions):                              # (1, n, 32), (1, 16n) i32
        x = latents * self.emb_std + self.emb_mean
        x = self.quantizer(x.transpose(1, 2))
        x = _causal_conv_transpose(self.upsample, x)
        cos, sin = _rope_tables(positions, self.head_dim, self.max_period)
        mask = self.window_mask(positions)
        h = x.transpose(1, 2)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        x = h.transpose(1, 2)
        for block in self.decoder.model:
            name = type(block).__name__
            if name == "StreamingConv1d":
                x = _causal_conv(block.conv, x)
            elif name == "StreamingConvTranspose1d":
                x = _causal_conv_transpose(block.convtr, x)
            elif name == "SEANetResnetBlock":
                v = x
                for sub in block.block:
                    v = _causal_conv(sub.conv, v) if type(sub).__name__ == "StreamingConv1d" else sub(v)
                x = x + v
            else:
                x = block(x)
        return x[:, 0]


# ------------------------------------------------------------------------------------------ export --

def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `fuse_loom_attention` expects to find added to `Q @ K^T`."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def load_reference(model_dir: str):
    """The reference `TTSModel` for a `languages/<name>` checkpoint directory, read through the
    reference's own `config/<name>.yaml` with its `hf://` paths pointed at the directory."""
    import tempfile

    import yaml

    import_pocket_tts()
    from pocket_tts.models.tts_model import TTSModel

    model_dir = Path(model_dir)
    config_path = Path(POCKET_TTS_REPO) / "pocket_tts" / "config" / f"{model_dir.name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} does not exist. A pocket-tts checkpoint is paired with its YAML by the "
            f"directory's name (languages/english_2026-09 <-> config/english_2026-09.yaml), and the "
            f"hyperparameters (flow type, heads, Mimi's shape) are written nowhere else.")
    config = yaml.safe_load(config_path.read_text())
    config["weights_path"] = str(model_dir / "model.safetensors")
    config.pop("weights_path_without_voice_cloning", None)
    config["flow_lm"]["lookup_table"]["tokenizer"] = "sentencepiece"
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(model_dir / "tokenizer.model")
    if config.get("pad_with_spaces_for_short_inputs"):
        raise NotImplementedError(f"{config_path.name} pads short inputs with spaces, which this "
                                  f"export's text front end does not do")
    if config["flow_lm"]["flow"].get("type", "lsd") not in FLOW_TYPES:
        raise NotImplementedError(f"{config_path.name}'s flow head is "
                                  f"{config['flow_lm']['flow']['type']!r}; the driver runs "
                                  f"{', '.join(FLOW_TYPES)} (one step each, no Euler integration)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / config_path.name
        path.write_text(yaml.safe_dump(config))
        tts = TTSModel.load_model(config=str(path))
    return tts.eval(), config


def reference_fade_in(tts) -> Optional[np.ndarray]:
    """The ramp the reference multiplies each chunk's first samples by, or None if it applies none.

    v4's `_decode_audio_worker` fades the first 5 ms of every chunk in (`linspace(0, 1, sr // 200)`): a
    fresh Mimi decoder state starts with a small step, heard as a click. v3.2.0 has no fade, so the ramp
    is shipped only when the reference in use applies it -- read off its source rather than assumed, so a
    re-export of a published model against the reference it was gated on keeps its waveform."""
    import inspect

    if "fade_in" not in inspect.getsource(type(tts)._decode_audio_worker):
        return None
    return torch.linspace(0, 1, SAMPLE_RATE // 200).numpy().astype(np.float32)


def read_voice(path: Path, n_layers: int) -> Tuple[np.ndarray, int]:
    """A saved voice state -> the flat array `loom.seed_kv` takes: per layer, K `[n, embed]` then V.

    The file is `export_model_state`'s: `transformer.layers.<i>.self_attn/cache` of shape
    `(2, 1, n, heads, head_dim)` and its `offset`. The cache's rows ARE the voice -- `offset` of them,
    with no padding -- and the head axes flatten in the order an ATTENTION node writes a row."""
    from safetensors import safe_open

    parts, n_rows = [], None
    with safe_open(str(path), "pt") as f:
        for i in range(n_layers):
            cache = f.get_tensor(f"transformer.layers.{i}.self_attn/cache").float()
            offset = int(f.get_tensor(f"transformer.layers.{i}.self_attn/offset").view(-1)[0])
            if cache.shape[2] != offset:
                raise ValueError(f"{path.name} layer {i}: {cache.shape[2]} cached rows but offset "
                                 f"{offset}; a padded state would seed rows the voice never wrote")
            if n_rows not in (None, offset):
                raise ValueError(f"{path.name}: layers disagree on the voice's length")
            n_rows = offset
            if torch.isnan(cache).any():
                raise ValueError(f"{path.name} layer {i} holds NaN")
            parts += [cache[0, 0].reshape(offset, -1), cache[1, 0].reshape(offset, -1)]
    return torch.cat(parts, dim=0).reshape(-1).numpy().astype(np.float32), n_rows


def word_start_flags(sp) -> np.ndarray:
    """Per id: 1 when the piece opens a word (`▁` and more), 2 for a lone `▁`, else 0 -- what the
    driver's `pocket_count_words` reads to reproduce `prepare_text_prompt`'s `len(text.split())`."""
    flags = np.zeros(sp.vocab_size(), dtype=np.float32)
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        if piece == "\u2581":
            flags[i] = 2
        elif piece.startswith("\u2581") and not (sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i)):
            flags[i] = 1
    return flags


@dataclass(kw_only=True)
class PocketTTSExportConfig(BaseMultiPhaseModelExportConfig):
    """A pocket-tts `languages/<name>` directory -> one Loom GGUF."""

    architecture: str = "pocket-tts"
    model_dir: str
    voice: str = "alba"
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "pocket_tts_driver"
    _driver_weights: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _temperature: float = field(default=0.7, init=False, repr=False)
    _eos_threshold: float = field(default=DEFAULT_EOS_THRESHOLD, init=False, repr=False)
    _flow_type: str = field(default="lsd", init=False, repr=False)
    _fade_in: int = field(default=0, init=False, repr=False)
    _voice_rows: int = field(default=0, init=False, repr=False)
    _voice_compat: Optional[str] = field(default=None, init=False, repr=False)
    _chunk_headers: Tuple[int, int] = field(default=(-1, -1), init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to a `languages/<name>` directory; the recognizer found "
                               "`model.safetensors`, `tokenizer.model` and `embeddings/` in it"),
        "voice": Unchecked("the built-in voice's file stem under `embeddings/`; read and "
                           "shape-checked by `read_voice`"),
        "decomposition": Unchecked("MultiPhase by construction -- five graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                         "against the traced topologies by LuaFragment"),
        "_driver_weights": Unchecked("READ off the checkpoint during phases() and shipped as driver "
                                      "weights"),
        "_temperature": Unchecked("the config's `default_temperature`, read during phases()"),
        "_eos_threshold": Unchecked("the config's `default_eos_threshold` (v4; -4 before), read during "
                                    "phases()"),
        "_flow_type": Unchecked("the config's `flow.type`, read during phases(); load_reference "
                                "refuses one the driver does not run"),
        "_fade_in": Unchecked("the reference's per-chunk fade-in length (0 for none), read during "
                              "phases()"),
        "_voice_rows": Unchecked("the built-in voice's length, read during phases()"),
        "_voice_compat": Unchecked("the flow LM weights' fingerprint, read once by contract()"),
        "_chunk_headers": Unchecked("the SentencePiece `<s>`/`</s>` ids, read during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        tts, config = load_reference(self.model_dir)
        flow_lm, mimi = tts.flow_lm, tts.mimi
        self._temperature = float(tts.temp)
        # v4 reads the threshold off the config (`default_eos_threshold`, -1 for the small models); before
        # v4 it is `DEFAULT_EOS_THRESHOLD`, which v4's config default also is.
        self._eos_threshold = float(config.get("default_eos_threshold", DEFAULT_EOS_THRESHOLD))
        self._flow_type = config["flow_lm"]["flow"].get("type", "lsd")
        sp = flow_lm.conditioner.tokenizer.sp
        self._chunk_headers = (int(sp.bos_id()), int(sp.eos_id()))
        n_layers = len(flow_lm.transformer.layers)
        voice, self._voice_rows = read_voice(Path(self.model_dir) / "embeddings" / f"{self.voice}.safetensors",
                                             n_layers)
        self._driver_weights = {
            "voice.kv": voice,
            # Step 0's input: the reference feeds a NaN latent and `FlowLMModel.forward` swaps in this.
            "bos_emb": flow_lm.bos_emb.detach().float().numpy(),
            "word_start": word_start_flags(flow_lm.conditioner.tokenizer.sp),
        }
        fade_in = reference_fade_in(tts)
        if fade_in is not None:
            self._driver_weights["fade_in"] = fade_in
        self._fade_in = 0 if fade_in is None else len(fade_in)
        dim, ldim = flow_lm.dim, flow_lm.ldim
        # What the flow and EOS heads read: `z_proj`'s output when the config has one, else the backbone's.
        cond = flow_lm.z_proj.out_features if getattr(flow_lm, "z_proj", None) is not None else dim
        max_period = float(config["flow_lm"]["transformer"]["max_period"])
        text_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        seq_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        frame_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        return [
            ExportPhase(
                name="text_embed",
                wrapper=TextEmbedPhase(flow_lm).eval(),
                dummy_inputs=(torch.randint(4, flow_lm.conditioner.embed.num_embeddings - 1,
                                            (1, TRACE_TOKENS), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="text_ids", shape=(1, text_dim), dtype=np.int32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="lm",
                wrapper=LMPhase(flow_lm, max_period).eval(),
                dummy_inputs=(torch.randn(1, TRACE_STEPS, dim),
                              torch.arange(TRACE_STEPS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_STEPS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, seq_dim, dim), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="step_embed",
                wrapper=StepEmbedPhase(flow_lm).eval(),
                dummy_inputs=(torch.randn(1, 1, ldim),),
                mil_inputs=[ct.TensorType(name="latent", shape=(1, 1, ldim), dtype=np.float32)],
            ),
            self._flow_head_phase(flow_lm, cond, ldim),
            ExportPhase(
                name="mimi_decoder",
                wrapper=MimiDecoderPhase(flow_lm, mimi).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, ldim),
                              torch.arange(TRACE_FRAMES * MIMI_UPSAMPLE, dtype=torch.int32).view(1, -1)),
                mil_inputs=[
                    ct.TensorType(name="latents", shape=(1, frame_dim, ldim), dtype=np.float32),
                    ct.TensorType(name="positions",
                                  shape=(1, ct.RangeDim(MIMI_UPSAMPLE, MIMI_UPSAMPLE * LM_MAX_POSITIONS)),
                                  dtype=np.int32),
                ],
                root_axis="n_codes",
                # 16 decoder positions per latent -- one symbol, not two.
                declared_axes={"positions": {1: f"{MIMI_UPSAMPLE} * n_codes"}},
            ),
        ]

    def _flow_head_phase(self, flow_lm, cond: int, ldim: int) -> ExportPhase:
        import coremltools as ct

        if self._flow_type == "lsd":
            return ExportPhase(
                name="flow_head",
                wrapper=FlowHeadPhase(flow_lm.flow_net).eval(),
                dummy_inputs=(torch.randn(1, cond), torch.tensor([[0.0]]), torch.tensor([[1.0]]),
                              torch.randn(1, ldim), torch.tensor([[0.5]]), torch.tensor([[1.0]])),
                mil_inputs=[ct.TensorType(name="c", shape=(1, cond), dtype=np.float32),
                            ct.TensorType(name="s", shape=(1, 1), dtype=np.float32),
                            ct.TensorType(name="t", shape=(1, 1), dtype=np.float32),
                            ct.TensorType(name="x", shape=(1, ldim), dtype=np.float32),
                            ct.TensorType(name="x_scale", shape=(1, 1), dtype=np.float32),
                            ct.TensorType(name="n_steps", shape=(1, 1), dtype=np.float32)],
            )
        # No time inputs at all, rather than ones the graph ignores: an input nothing reads would be a
        # binding the engine is asked for and a trace could drop.
        return ExportPhase(
            name="flow_head",
            wrapper=OneStepFlowHeadPhase(flow_lm.flow_net, self._flow_type).eval(),
            dummy_inputs=(torch.randn(1, cond), torch.randn(1, ldim), torch.tensor([[0.5]])),
            mil_inputs=[ct.TensorType(name="c", shape=(1, cond), dtype=np.float32),
                        ct.TensorType(name="x", shape=(1, ldim), dtype=np.float32),
                        ct.TensorType(name="x_scale", shape=(1, 1), dtype=np.float32)],
        )

    def driver_components(self) -> List:
        from .driver_components import (
            CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment,
        )
        from .driver_ir import Len

        fragment = self.driver_script_path
        return [
            ExportConstants(values={
                # The ids `loom::PocketTtsVocab` opens each chunk with (`<s>`, `</s>`), and the tails
                # they stand for: `prepare_text_prompt`'s guess, plus `generate_audio_stream`'s 2.
                "CHUNK_HEADER_SHORT": self._chunk_headers[0],
                "CHUNK_HEADER_LONG": self._chunk_headers[1],
                "SHORT_CHUNK_MAX_WORDS": SHORT_CHUNK_MAX_WORDS,
                "SHORT_CHUNK_FRAMES_AFTER_EOS": SHORT_CHUNK_FRAMES_AFTER_EOS,
                "LONG_CHUNK_FRAMES_AFTER_EOS": LONG_CHUNK_FRAMES_AFTER_EOS,
                "FRAMES_AFTER_EOS_PADDING": FRAMES_AFTER_EOS_PADDING,
                "LATENT_DIM": 32,
                "LM_MAX_POSITIONS": LM_MAX_POSITIONS,
                "MIMI_UPSAMPLE": MIMI_UPSAMPLE,
                "FRAME_RATE": SAMPLE_RATE / SAMPLES_PER_FRAME,
                "TOKENS_PER_SECOND_ESTIMATE": TOKENS_PER_SECOND_ESTIMATE,
                "GEN_SECONDS_PADDING": GEN_SECONDS_PADDING,
                "MIN_FRAMES_BEFORE_EOS": MIN_FRAMES_BEFORE_EOS,
                # The checkpoint's own `default_temperature` (0.3 for English, where the reference's
                # human evals preferred it), and the reference's other generation defaults.
                "DEFAULT_TEMPERATURE": self._temperature,
                "DEFAULT_EOS_THRESHOLD": self._eos_threshold,
                "DEFAULT_DECODE_STEPS": DEFAULT_DECODE_STEPS,
                # Samples of each chunk faded in by the `fade_in` driver weight; 0 for none.
                "FADE_IN_SAMPLES": self._fade_in,
            }),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "00_header.lua", top_level=True,
                        defines=("pocket_count_words", "pocket_split_chunks")),
            # The one `flow_head` call this export's graph takes: LSD's timed one or a one-step head's.
            LuaFragment(fragment / ("00_flow_head_lsd.lua" if self._flow_type == "lsd"
                                    else "00_flow_head_one_step.lua"),
                        top_level=True, defines=("pocket_flow_head",)),
            LuaFragment(fragment / "01_flow_lm.lua",
                        reads=("tokens", "CHUNK_HEADER_SHORT", "CHUNK_HEADER_LONG", "SHORT_CHUNK_MAX_WORDS",
                               "SHORT_CHUNK_FRAMES_AFTER_EOS", "LONG_CHUNK_FRAMES_AFTER_EOS",
                               "FRAMES_AFTER_EOS_PADDING", "LATENT_DIM", "LM_MAX_POSITIONS",
                               "MIMI_UPSAMPLE", "FRAME_RATE", "TOKENS_PER_SECOND_ESTIMATE",
                               "GEN_SECONDS_PADDING", "MIN_FRAMES_BEFORE_EOS", "DEFAULT_TEMPERATURE",
                               "DEFAULT_EOS_THRESHOLD", "DEFAULT_DECODE_STEPS", "FADE_IN_SAMPLES"),
                        defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        # What a voice file must match to be loaded into this model (`pocket_tts_voices`, loom.cpp
        # ADR-045): the fingerprint of the weights every voice state is a function of. A fact about
        # THESE weights, so it is read only when there are weights to read -- an architecture-only
        # query (test_tts_text_door's nonexistent path) opens nothing.
        from .pocket_tts_voices import weights_fingerprint

        weights = Path(self.model_dir) / "model.safetensors"
        if self._voice_compat is None and weights.is_file():
            self._voice_compat = weights_fingerprint(weights)
        if self._voice_compat is not None:
            contract["voice.compat"] = self._voice_compat
        # The voice the file carries, which is what `infer` uses when the caller names none.
        contract["tts.voices"] = [self.voice]
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                      tokenizer_dir=self.model_dir, tokenizer_family="pocket_tts")
        if self._driver_weights is not None:
            kwargs["driver_weights"] = dict(self._driver_weights)
        return kwargs


def _is_pocket_tts(path: Path) -> bool:
    """A pocket-tts language directory: the bundled weights, the SentencePiece model and the voices,
    with a `mimi.` and a `flow_lm.` half in the weights (what tells it from any other safetensors)."""
    if not (path.is_dir() and (path / "model.safetensors").is_file()
            and (path / "tokenizer.model").is_file() and (path / "embeddings").is_dir()):
        return False
    from safetensors import safe_open

    # `detect()` runs against unidentified paths by construction: a file that is not safetensors is a
    # "no", not a traceback.
    try:
        with safe_open(str(path / "model.safetensors"), "pt") as f:
            keys = set(f.keys())
    except Exception:
        return False
    return "flow_lm.bos_emb" in keys and "mimi.quantizer.output_proj.weight" in keys


def _build_pocket_tts(path: Path, output_path: str) -> LoomExportConfig:
    return PocketTTSExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=PocketTTSExportConfig,
        recognizers=[
            ModelRecognizer(name="pocket-tts", detect=_is_pocket_tts, build_config=_build_pocket_tts),
        ],
    ))
