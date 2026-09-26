"""Export VoxCPM2 (`openbmb/VoxCPM2`) -- family 9's sixth leaf, and the second whose autoregressive loop
carries continuous latents: a 2.3B diffusion-autoregressive TTS over the latents of its own AudioVAE.

    text -> BPE ids + <|audio_start|> -> base LM (MiniCPM-4, 28 layers) + residual LM (8 layers, no RoPE)
         -> per step: a local DiT (12 layers) integrates one PATCH of 4 x 64-d latents over 10 guided
                      Euler steps; the patch is re-encoded by a local encoder (12 layers) into the next
                      step's input row, and a stop head on the base LM's row ends the loop
         -> AudioVAE V2's causal decoder: 25 Hz latents -> x1920 -> 48 kHz

Five phases, each a re-spelling of the reference module it replaces (`VoxCPM2Model._inference`):
  - `feat_encode`:  patches `[1, T, 4, 64]` -> `enc_to_lm_proj(feat_encoder(patches))`, `[1, T, 2048]`. The
                    prefill encodes the prompt's patches (zeros where the prompt is text), and every
                    step encodes the patch it just generated.
  - `base_lm`:      the text-semantic LM, KV-cached. Its input row is `text_mask * embed(ids) +
                    audio_mask * feat_embed`, which is the reference's prefill embedding and, at
                    `(0, 1)`, a step's `curr_embed` exactly. Returns every row after the FSQ bottleneck
                    (applied where `audio_mask` is set, as the reference applies it), the last row
                    alone, and the stop head's two logits for that row.
  - `residual_lm`:  the residual acoustic LM, KV-cached, over `fusion_concat_proj([enc, audio_mask *
                    feat_embed])`. Returns its last row.
  - `dit_step`:     ONE Euler step of `UnifiedCFM.solve_euler`: the local DiT on the conditional and the
                    unconditional sequence at once (batch 2), CFG-Zero*'s projection and the guidance
                    combination, and `x - dt * v`. The whole update is in the graph because it is f32
                    arithmetic in the reference and the driver's LuaJIT has doubles only (the Pocket-TTS
                    lesson), and because the state is one patch: 256 floats, so the hand-written loop
                    costs none of the crossings `loom.run_ode` exists to avoid (loom.cpp ADR-031). It is
                    not `run_ode` for a second reason: CFG-Zero* scales the unconditional velocity by a
                    dot product of the two, which ADR-040's `v_c + s (v_c - v_u)` cannot express.
  - `vae_decode`:   every latent in one call -> the 48 kHz waveform. The decoder is causal, so one call
                    computes what the reference's `decode` does.

Voice cloning (a reference clip through the AudioVAE's encoder) is not in this export: the loop is
zero-shot, and a voice is DESIGNED in the text -- "(A young woman, gentle and sweet voice)Hello" -- which
needs no graph of its own.

Usage:
  loom-export ~/Dev/models/voxcpm2 -o voxcpm2.gguf --task text-to-speech --model voxcpm2
"""
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Axis, Unchecked

# The reference checkout (OpenBMB/VoxCPM, Apache-2.0). `voxcpm` on PyPI would do too; the clone is what
# `loom.cpp/scripts/voxcpm2_reference.py` runs, so the two read the same code.
VOXCPM_REPO = "/home/flavio/Dev/VoxCPM/src"

# The AudioVAE's OUTPUT rate: it encodes at 16 kHz and decodes at 48 kHz, super-resolving on the way.
SAMPLE_RATE = 48000
# Waveform samples per latent frame: `math.prod(decoder_rates)`, 8 * 6 * 5 * 2 * 2 * 2.
SAMPLES_PER_LATENT = 1920
# `VoxCPM.generate`'s defaults, which every published sample used.
DEFAULT_CFG = 2.0
DEFAULT_TIMESTEPS = 10
MIN_LEN = 2
MAX_LEN = 4096
# `retry_badcase_ratio_threshold`: the loop's budget is `ratio * len(target ids) + 10` patches, and a
# generation that uses all of it is the reference's "badcase".
BADCASE_RATIO = 6.0
# `UnifiedCFM.forward`'s sway-sampling coefficient and CFG-Zero*'s zero-init fraction.
SWAY_COEF = 1.0
ZERO_INIT_FRACTION = 0.04
# The special ids `VoxCPM2Model.__init__` hard-codes.
AUDIO_START_TOKEN = 101
# How many positions the two cached LMs hold: a prompt and its patches. The reference's own ceiling is
# `max_length = 8192`; 4096 is a 250-token text at the badcase ratio with room to spare, and costs
# (28 + 8) layers x 2 x 4096 x 256 x 4 bytes = 302 MB with the checkpoint's 2 K/V heads.
LM_MAX_POSITIONS = 4096

# Trace lengths: odd, and distinct from every static dimension -- the 2 K/V heads, the 8-fold GQA
# repeat, the 16 heads, the 4-latent patch and the 5- and 11-token local sequences.
TRACE_TOKENS = 13
TRACE_PATCHES = 7
TRACE_LATENTS = 9


def import_voxcpm():
    if VOXCPM_REPO not in sys.path:
        sys.path.insert(0, VOXCPM_REPO)


# ------------------------------------------------------------------------------------ shared parts --

def _rms_norm(norm, x):
    """`MiniCPMRMSNorm`: `x * rsqrt(mean(x^2) + eps) * weight`, in the reference's order."""
    variance = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + norm.variance_epsilon) * norm.weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x, n_rep: int):
    """HF's `repeat_kv`, spelled as HF spells it, because that spelling is what `passes.py`'s GQA fusion
    recognises and strips: the cache then stores the checkpoint's 2 K/V heads, not 16."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class _Attention(nn.Module):
    """`MiniCPMAttention`, in one of two spellings, both exact against it.

    CACHED (the two LMs): HF's -- Q scaled, `Q @ K^T + mask`, softmax, `@ V`, transpose, reshape --
    with `repeat_kv`, which is the window `fuse_loom_attention` turns into a cached ATTENTION node.

    GROUPED (the local encoder and the DiT, which have no cache and no mask): each K/V head serves
    `n_rep` ADJACENT query heads, so the queries are regrouped as `[b, n_kv, n_rep * s, d]` and attend
    to the un-repeated K/V directly. Query head `j` lands in group `j // n_rep`, the head `repeat_kv`
    would have paired it with, and no tile is traced at all.

    `past` is a torch-side verification hook, never traced: `(k, v)` rows to attend over before this
    call's own, returned updated, which is what the engine's cache does between calls."""

    def __init__(self, attn, cached: bool):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        self.heads, self.kv_heads, self.head_dim = attn.num_heads, attn.num_key_value_heads, attn.head_dim
        self.n_rep = self.heads // self.kv_heads
        self.cached = cached

    def forward(self, x, cos, sin, mask, past=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        if cos is not None:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        if past is not None:
            if past[0] is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            past[0], past[1] = k, v
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.cached:
            k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
            scores = torch.matmul(q * scale, k.transpose(-1, -2))
            if mask is not None:
                scores = scores + mask
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        else:
            q = q.reshape(b, self.kv_heads, self.n_rep * s, self.head_dim)
            scores = torch.matmul(q * scale, k.transpose(-1, -2))
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
            ctx = ctx.reshape(b, self.heads, s, self.head_dim)
        ctx = ctx.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o_proj(ctx)


class _Layer(nn.Module):
    """`MiniCPMDecoderLayer` with `use_mup` off, which is what the checkpoint declares: pre-norm
    attention and a SiLU-gated MLP, both residual, neither scaled."""

    def __init__(self, layer, cached: bool):
        super().__init__()
        if layer.use_mup:
            raise NotImplementedError("a MiniCPM layer with use_mup scales its residuals by "
                                      "scale_depth / sqrt(n_layers); VoxCPM2's checkpoint does not")
        self.attn = _Attention(layer.self_attn, cached)
        self.input_layernorm, self.post_attention_layernorm = layer.input_layernorm, layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, x, cos, sin, mask, past=None):
        x = x + self.attn(_rms_norm(self.input_layernorm, x), cos, sin, mask, past)
        h = _rms_norm(self.post_attention_layernorm, x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))


class _Stack(nn.Module):
    """`MiniCPMModel.forward` without its embedding: the layers and the final norm."""

    def __init__(self, model, cached: bool):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(l, cached) for l in model.layers)
        self.norm = model.norm

    def forward(self, x, cos, sin, mask, pasts=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, None if pasts is None else pasts[i])
        return _rms_norm(self.norm, x)


def _rope_rows(model, n: int):
    """The first `n` rows of a stack's own `cos_cached`/`sin_cached` -- the reference's table, LongRoPE
    factors and all, rather than a recomputation of it."""
    rope = model.rope_emb
    if rope is None:
        return None, None
    return rope.cos_cached[:n].detach().clone().float(), rope.sin_cached[:n].detach().clone().float()


# ---------------------------------------------------------------------------------------- phases --

class FeatEncodePhase(nn.Module):
    """`(patches) -> enc_to_lm_proj(feat_encoder(patches))`, `[1, T, 4, 64] -> [1, T, 2048]`.

    `VoxCPMLocEnc`: each patch is a 5-token sequence -- the learned special token, then the 4 projected
    latents -- through a bidirectional 12-layer stack, and the special token's output row is the
    patch's embedding. The special token is broadcast by ARITHMETIC (`x[..., :1, :] * 0 + token`)
    rather than `expand(B, T, 1, -1)`, whose target shape would be a traced read of `T`."""

    def __init__(self, model):
        super().__init__()
        enc = model.feat_encoder
        self.in_proj, self.special_token = enc.in_proj, enc.special_token
        self.stack = _Stack(enc.encoder, cached=False)
        self.out = model.enc_to_lm_proj
        cos, sin = _rope_rows(enc.encoder, model.patch_size + 1)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, patches):                                        # (1, T, 4, 64)
        x = self.in_proj(patches)
        special = x[:, :, :1, :] * 0.0 + self.special_token
        x = torch.cat([special, x], dim=2)[0]                           # (T, 5, 1024)
        h = self.stack(x, self.cos, self.sin, None)
        return self.out(h[:, 0, :]).unsqueeze(0)


class BaseLMPhase(nn.Module):
    """`(text_ids, feat_embed, text_mask, audio_mask, position_ids, attention_mask) ->
    (enc, last, stop_logits)`.

    `enc` is every row of `fsq(h) * audio_mask + h * text_mask` -- the rows the residual LM's prefill
    reads, and at a step the one row the DiT and the next residual step read. `last` is its final row
    and `stop_logits` the stop head on it; the reference takes the argmax, so the driver compares the
    two. RoPE's cos/sin are ROWS of the reference's own table, gathered by position: exact, and a table
    of `LM_MAX_POSITIONS` rows."""

    def __init__(self, model):
        super().__init__()
        lm = model.base_lm
        self.embed = lm.embed_tokens
        self.stack = _Stack(lm, cached=True)
        self.fsq = model.fsq_layer
        self.stop_proj, self.stop_head = model.stop_proj, model.stop_head
        cos, sin = _rope_rows(lm, LM_MAX_POSITIONS)
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)
        self.fsq_scale = float(model.fsq_layer.scale)

    def quantize(self, h):
        """`ScalarQuantizationLayer.forward` in eval: `round(tanh(in_proj(h)) * scale) / scale`, then out."""
        z = torch.tanh(self.fsq.in_proj(h))
        return self.fsq.out_proj(torch.round(z * self.fsq_scale) / self.fsq_scale)

    def forward(self, text_ids, feat_embed, text_mask, audio_mask, position_ids, attention_mask, pasts=None):
        x = text_mask * self.embed(text_ids) + audio_mask * feat_embed
        cos = F.embedding(position_ids, self.cos_table).unsqueeze(1)    # (1, 1, n, 128)
        sin = F.embedding(position_ids, self.sin_table).unsqueeze(1)
        h = self.stack(x, cos, sin, attention_mask, pasts)
        enc = self.quantize(h) * audio_mask + h * text_mask
        last = enc[:, -1:, :]
        return enc, last, self.stop_head(F.silu(self.stop_proj(last)))


class ResidualLMPhase(nn.Module):
    """`(enc, feat_embed, audio_mask, attention_mask) -> last row`, the residual acoustic LM over
    `fusion_concat_proj([enc, audio_mask * feat_embed])`. No positional encoding at all
    (`residual_lm_no_rope`): the causal mask and the cache are its only sense of order."""

    def __init__(self, model):
        super().__init__()
        if model.residual_lm.rope_emb is not None:
            raise NotImplementedError("this checkpoint's residual LM has RoPE; the phase has no positions input")
        self.fusion = model.fusion_concat_proj
        self.stack = _Stack(model.residual_lm, cached=True)

    def forward(self, enc, feat_embed, audio_mask, attention_mask, pasts=None):
        x = self.fusion(torch.cat([enc, audio_mask * feat_embed], dim=-1))
        return self.stack(x, None, None, attention_mask, pasts)[:, -1:, :]


class DiTStepPhase(nn.Module):
    """`(lm_hidden, residual_hidden, cond, x, t, dt, cfg) -> x - dt * v`, one guided Euler step.

    Patches are PATCH-MAJOR here, `[1, 4, 64]` -- the layout the driver keeps and the feature encoder
    reads. The reference's `x` and `cond` are channel-major `[b, 64, 4]` and its DiT transposes both on
    entry, so this phase starts one transpose later and ends one earlier: the same numbers.

    The DiT sequence is `[mu_0, mu_1, t, cond_0..3, x_0..3]`, 11 tokens, bidirectional, RoPE at 0..10;
    the unconditional run replaces the two `mu` rows with zeros (`solve_euler` zeroes `mu_in[b:]`), and
    both run as one batch. `dt` is ZERO inside the DiT (`mean_mode` is off, so `solve_euler` zeroes
    `dt_in`), which makes the delta-time embedding a constant, computed here once by the reference's
    own modules. The `dt` input is the Euler step's width, which is a different quantity."""

    def __init__(self, model):
        super().__init__()
        cfm = model.feat_decoder
        if cfm.mean_mode:
            raise NotImplementedError("a mean-mode CFM feeds dt to the DiT; this checkpoint's does not")
        est = cfm.estimator
        self.lm_to_dit, self.res_to_dit = model.lm_to_dit_proj, model.res_to_dit_proj
        self.in_proj, self.cond_proj, self.out_proj = est.in_proj, est.cond_proj, est.out_proj
        self.time_mlp = est.time_mlp
        self.stack = _Stack(est.decoder, cached=False)
        self.hidden = est.config.hidden_size
        half = self.hidden // 2
        # `SinusoidalPosEmb.forward`'s frequencies, in f32 as it forms them.
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000) / (half - 1)))
        self.register_buffer("freqs", freqs)
        with torch.no_grad():
            dt_emb = est.delta_time_mlp(est.time_embeddings(torch.zeros(1)))
        self.register_buffer("dt_emb", dt_emb.detach().clone().float())
        n = 2 + 1 + 2 * model.patch_size
        cos, sin = _rope_rows(est.decoder, n)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.prefix = 2 + 1 + model.patch_size

    def time_embedding(self, t):                                         # (1, 1) -> (1, 1024)
        args = (t * 1000.0) * self.freqs
        return self.time_mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1)) + self.dt_emb

    def velocity(self, lm_hidden, residual_hidden, cond, x, t, cfg):
        mu = torch.cat([self.lm_to_dit(lm_hidden), self.res_to_dit(residual_hidden)], dim=-1)
        mu = mu.reshape(1, 2, self.hidden)
        temb = self.time_embedding(t).unsqueeze(1)
        tail = torch.cat([temb, self.cond_proj(cond), self.in_proj(x)], dim=1)
        seq = torch.cat([torch.cat([mu, tail], dim=1), torch.cat([mu * 0.0, tail], dim=1)], dim=0)
        v = self.out_proj(self.stack(seq, self.cos, self.sin, None)[:, self.prefix:, :])
        positive, negative = v[0:1], v[1:2]
        # CFG-Zero*: the unconditional velocity scaled by its projection onto the conditional one. Each
        # sum over the patch's 256 values is two single-axis sums, which is what the lowering takes.
        dot = (positive * negative).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        norm = (negative * negative).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        st = dot / (norm + 1e-8)
        return negative * st + cfg * (positive - negative * st)

    def forward(self, lm_hidden, residual_hidden, cond, x, t, dt, cfg):
        # (1,1,2048) (1,1,2048) (1,4,64) (1,4,64) (1,1) (1,1) (1,1)
        return x - dt * self.velocity(lm_hidden, residual_hidden, cond, x, t, cfg)


def _snake(snake, x):
    """`Snake1d` without its scripted body's reshapes, which are no-ops for the rank-3 input (DAC's
    note in `audio_codec_export`)."""
    return x + (snake.alpha + 1e-9).reciprocal() * torch.sin(snake.alpha * x).pow(2)


def _causal_conv(conv, x):
    """`CausalConv1d.forward`: left-pad by `2 * padding - output_padding`, then convolve."""
    pad = conv._CausalConv1d__padding * 2 - conv._CausalConv1d__output_padding
    return F.conv1d(F.pad(x, (pad, 0)) if pad else x, conv.weight, conv.bias, conv.stride, 0,
                    conv.dilation, conv.groups)


def _causal_conv_transpose(convtr, x):
    """`CausalTransposeConv1d.forward`: the full transposed convolution, less its last
    `2 * padding - output_padding` samples."""
    y = F.conv_transpose1d(x, convtr.weight, convtr.bias, convtr.stride, 0, 0, convtr.groups, convtr.dilation)
    trim = convtr._CausalTransposeConv1d__padding * 2 - convtr._CausalTransposeConv1d__output_padding
    return y[..., :-trim] if trim else y


def _residual_unit(unit, x):
    snake1, conv1, snake2, conv2 = unit.block
    y = _causal_conv(conv2, _snake(snake2, _causal_conv(conv1, _snake(snake1, x))))
    return x + y


class VAEDecodePhase(nn.Module):
    """`(latents) -> waveform`, `[1, n, 64] -> [1, n * 1920]`: `AudioVAEV2.decode` at its default output
    rate, every frame in one call.

    Latents arrive FRAME-major -- the patches the loop produced, flattened -- and the decoder wants
    channel-major `[1, 64, n]`, which is the reference's own `rearrange('b t p d -> b d (t p)')`.
    The sample-rate condition is a constant: `decode` without `sr_cond` is `bucketize(48000)`, the last
    of the four buckets, so each block's scale and bias are rows the export reads off the embeddings.
    Weight norm is folded (the parametrizations are removed before the trace)."""

    def __init__(self, vae):
        super().__init__()
        dec = vae.decoder
        if dec.sr_bin_boundaries is None:
            raise NotImplementedError("an AudioVAE without sample-rate conditioning")
        bucket = int(torch.bucketize(torch.tensor([vae.out_sample_rate], dtype=torch.int32),
                                     dec.sr_bin_boundaries).item())
        self.model = dec.model
        self.scales, self.biases = [], []
        for i, cond in enumerate(dec.sr_cond_model):
            if cond is None:
                self.scales.append(None)
                self.biases.append(None)
                continue
            if cond.cond_type not in ("scale_bias", "scale_bias_init") or not isinstance(cond.out_layer, nn.Identity):
                raise NotImplementedError(f"sample-rate condition {cond.cond_type!r} with an out layer")
            self.register_buffer(f"sr_scale_{i}", cond.scale_embed.weight[bucket].detach().clone().view(1, -1, 1))
            self.register_buffer(f"sr_bias_{i}", cond.bias_embed.weight[bucket].detach().clone().view(1, -1, 1))
            self.scales.append(f"sr_scale_{i}")
            self.biases.append(f"sr_bias_{i}")

    def forward(self, latents):                                          # (1, n, 64)
        x = latents.transpose(1, 2)
        for i, layer in enumerate(self.model):
            if self.scales[i] is not None:
                x = x * getattr(self, self.scales[i]) + getattr(self, self.biases[i])
            name = type(layer).__name__
            if name == "CausalConv1d":
                x = _causal_conv(layer, x)
            elif name == "CausalDecoderBlock":
                for sub in layer.block:
                    sub_name = type(sub).__name__
                    if sub_name == "Snake1d":
                        x = _snake(sub, x)
                    elif sub_name == "CausalTransposeConv1d":
                        x = _causal_conv_transpose(sub, x)
                    elif sub_name == "CausalResidualUnit":
                        x = _residual_unit(sub, x)
                    else:
                        raise NotImplementedError(f"decoder block member {sub_name}")
            elif name == "Snake1d":
                x = _snake(layer, x)
            elif name == "Tanh":
                x = torch.tanh(x)
            else:
                raise NotImplementedError(f"decoder layer {name}")
        return x[:, 0]


# ------------------------------------------------------------------------------------------ export --

def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `fuse_loom_attention` expects to find added to `Q @ K^T`."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(1, 1, seq_len, seq_len)


def fold_weight_norm(module: nn.Module) -> None:
    """Remove every weight norm in place, keeping the weight it computes -- in either spelling.

    The AudioVAE uses the OLD one (`torch.nn.utils.weight_norm`, a forward pre-hook over `weight_g` and
    `weight_v`), and that is the dangerous one: `weight` is a plain attribute the hook refreshes on
    every forward, so a module whose checkpoint was loaded but which has not run yet holds the weight
    computed from its INITIALISATION. A wrapper reading `conv.weight` would trace that, and a check run
    after the reference's own forward would see the refreshed value and pass."""
    from torch.nn.utils import parametrize, remove_weight_norm
    from torch.nn.utils.weight_norm import WeightNorm

    for sub in module.modules():
        if any(isinstance(hook, WeightNorm) for hook in sub._forward_pre_hooks.values()):
            remove_weight_norm(sub)
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)


def load_reference(model_dir: str):
    """The reference `VoxCPM2Model` in f32 on the CPU, with its two KV caches rebuilt at f32 (they are
    allocated at the checkpoint's bf16 in `__init__`, before the cast)."""
    import_voxcpm()
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(str(model_dir), optimize=False, device="cpu")
    model.config.dtype = "float32"
    model = model.float().eval()
    for lm in (model.base_lm, model.residual_lm):
        lm.setup_cache(1, model.config.max_length, "cpu", torch.float32)
    return model


def euler_schedule(n_timesteps: int) -> np.ndarray:
    """`(t, dt)` for each of `solve_euler`'s steps, as the reference forms them in f32: the swayed
    linspace, then `t -= dt` and `dt = t - t_span[step + 1]`, both accumulated. Shipped as a driver
    weight (`[n, 2]`, row-major) for the default step count, because the driver's doubles cannot
    reproduce the f32 accumulation."""
    t_span = torch.linspace(1, 0, n_timesteps + 1, dtype=torch.float32)
    t_span = t_span + SWAY_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    t, dt = t_span[0], t_span[0] - t_span[1]
    rows = []
    for step in range(1, len(t_span)):
        rows.append((float(t), float(dt)))
        t = t - dt
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def zero_init_steps(n_timesteps: int) -> int:
    """`solve_euler`'s `max(1, int(len(t_span) * 0.04))`: the leading steps whose velocity is zero."""
    return max(1, int((n_timesteps + 1) * ZERO_INIT_FRACTION))


@dataclass(kw_only=True)
class VoxCPM2ExportConfig(BaseMultiPhaseModelExportConfig):
    """An `openbmb/VoxCPM2` checkpoint directory -> one Loom GGUF."""

    architecture: str = "voxcpm2"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "voxcpm2_driver"
    _driver_weights: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to a VoxCPM2 directory; the recognizer found `config.json` "
                               "(architecture voxcpm2), `model.safetensors` and `audiovae.pth` in it"),
        "decomposition": Unchecked("MultiPhase by construction -- five graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                         "against the traced topologies by LuaFragment"),
        "_driver_weights": Unchecked("computed from the reference during phases() and shipped as "
                                      "driver weights"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = load_reference(self.model_dir)
        fold_weight_norm(model.audio_vae)
        self._driver_weights = {"euler_schedule": euler_schedule(DEFAULT_TIMESTEPS)}
        dim = model.config.lm_config.hidden_size
        patch, ldim = model.patch_size, model.feat_dim
        seq_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        return [
            ExportPhase(
                name="feat_encode",
                wrapper=FeatEncodePhase(model).eval(),
                dummy_inputs=(torch.randn(1, TRACE_PATCHES, patch, ldim),),
                mil_inputs=[ct.TensorType(name="patches", shape=(1, ct.RangeDim(1, LM_MAX_POSITIONS), patch, ldim),
                                          dtype=np.float32)],
            ),
            ExportPhase(
                name="base_lm",
                wrapper=BaseLMPhase(model).eval(),
                dummy_inputs=(torch.randint(4, 1000, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.randn(1, TRACE_TOKENS, dim),
                              torch.ones(1, TRACE_TOKENS, 1), torch.zeros(1, TRACE_TOKENS, 1),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="text_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="feat_embed", shape=(1, seq_dim, dim), dtype=np.float32),
                    ct.TensorType(name="text_mask", shape=(1, seq_dim, 1), dtype=np.float32),
                    ct.TensorType(name="audio_mask", shape=(1, seq_dim, 1), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim), dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="residual_lm",
                wrapper=ResidualLMPhase(model).eval(),
                dummy_inputs=(torch.randn(1, TRACE_TOKENS, dim), torch.randn(1, TRACE_TOKENS, dim),
                              torch.zeros(1, TRACE_TOKENS, 1), causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="enc", shape=(1, seq_dim, dim), dtype=np.float32),
                    ct.TensorType(name="feat_embed", shape=(1, seq_dim, dim), dtype=np.float32),
                    ct.TensorType(name="audio_mask", shape=(1, seq_dim, 1), dtype=np.float32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim), dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="dit_step",
                wrapper=DiTStepPhase(model).eval(),
                dummy_inputs=(torch.randn(1, 1, dim), torch.randn(1, 1, dim), torch.randn(1, patch, ldim),
                              torch.randn(1, patch, ldim), torch.tensor([[0.7]]), torch.tensor([[0.1]]),
                              torch.tensor([[2.0]])),
                mil_inputs=[ct.TensorType(name="lm_hidden", shape=(1, 1, dim), dtype=np.float32),
                            ct.TensorType(name="residual_hidden", shape=(1, 1, dim), dtype=np.float32),
                            ct.TensorType(name="cond", shape=(1, patch, ldim), dtype=np.float32),
                            ct.TensorType(name="x", shape=(1, patch, ldim), dtype=np.float32),
                            ct.TensorType(name="t", shape=(1, 1), dtype=np.float32),
                            ct.TensorType(name="dt", shape=(1, 1), dtype=np.float32),
                            ct.TensorType(name="cfg", shape=(1, 1), dtype=np.float32)],
            ),
            ExportPhase(
                name="vae_decode",
                wrapper=VAEDecodePhase(model.audio_vae).eval(),
                dummy_inputs=(torch.randn(1, TRACE_LATENTS, ldim),),
                mil_inputs=[ct.TensorType(name="latents", shape=(1, ct.RangeDim(1, patch * LM_MAX_POSITIONS), ldim),
                                          dtype=np.float32)],
                root_axis="n_codes",
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        fragment = self.driver_script_path
        constants = {
            "AUDIO_START_TOKEN": AUDIO_START_TOKEN,
            "PATCH_SIZE": 4,
            "LATENT_DIM": 64,
            "LM_MAX_POSITIONS": LM_MAX_POSITIONS,
            "DEFAULT_CFG": DEFAULT_CFG,
            "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
            "ZERO_INIT_FRACTION": ZERO_INIT_FRACTION,
            "SWAY_COEF": SWAY_COEF,
            "MIN_LEN": MIN_LEN,
            "MAX_LEN": MAX_LEN,
            "BADCASE_RATIO": BADCASE_RATIO,
        }
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "00_header.lua", top_level=True, defines=("voxcpm_schedule",)),
            LuaFragment(fragment / "01_generate.lua", reads=("tokens",) + tuple(constants), defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                      tokenizer_dir=self.model_dir, tokenizer_family="voxcpm2")
        if self._driver_weights is not None:
            kwargs["driver_weights"] = dict(self._driver_weights)
        return kwargs


def _is_voxcpm2(path: Path) -> bool:
    """A VoxCPM2 directory: `config.json` declaring `architecture: voxcpm2`, the LM weights and the
    AudioVAE beside them."""
    if not (path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file()
            and (path / "audiovae.pth").is_file()):
        return False
    import json

    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return config.get("architecture") == "voxcpm2"


def _build_voxcpm2(path: Path, output_path: str) -> LoomExportConfig:
    return VoxCPM2ExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=VoxCPM2ExportConfig,
        recognizers=[ModelRecognizer(name="voxcpm2", detect=_is_voxcpm2, build_config=_build_voxcpm2)],
    ))
