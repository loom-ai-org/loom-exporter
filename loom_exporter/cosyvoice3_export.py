"""Export Fun-CosyVoice3-0.5B-2512 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) -- family 9's seventh leaf.

CosyVoice3 is three models in sequence, Chatterbox's shape with every stage swapped for a sibling:

    text -> LM (Qwen2-0.5B, sampled) -> FSQ speech tokens (25 Hz)
         -> flow (token embedding + a 2-conv look-ahead -> a guided 10-step ODE over a 22-layer DiT,
            in-filled after the voice's prompt frames) -> mel (50 Hz)
         -> CausalHiFT (NSF source + iSTFTNet) -> 24 kHz waveform

Four phases:
  - `lm`:           the Qwen2 decoder, KV-cached, embedding its OWN inputs: each row is either a text
                    id (Qwen's `embed_tokens`) or a speech id (the LM's `speech_embedding`, which also
                    holds the `sos` and `task_id` rows), chosen by `speech_mask`. So the prefill
                    `[sos, prompt text + text, task_id, prompt speech tokens]` and every decode step are
                    the same graph and no embedding ever crosses into Lua. Returns `llm_decoder`'s logits
                    for the LAST row, all 6761 of them: the reference samples over the whole head and
                    stops on any id at or above 6561.
  - `flow_encoder`: prompt tokens + generated tokens -> `mu` (`[2n, 80]`, frame-major) and the projected
                    speaker vector `spks`. `input_embedding`, `PreLookaheadLayer`, and the 2x
                    repeat_interleave.
  - `estimator`:    one velocity evaluation of the DiT. The ENGINE runs it twice per step under guidance
                    (loom.cpp ADR-040), on the cosine schedule -- Chatterbox's `FlowMatchingSpec` as is.
  - `vocoder`:      CausalHiFT: F0 predictor, the NSF sine source (at the FRAME rate -- see
                    `CausalHiftVocoderPhase`), and the iSTFTNet decoder. Its one live random draw is an
                    input.

**The LM samples with `ras_sampling`**, which `loom.sample_row` expresses with two options this leaf
added (loom.cpp ADR-047): a nucleus whose top-p mass is measured over the WHOLE softmax rather than
over the top-k survivors, and a banned-id set. The repetition-aware second draw is the driver's.

**The default voice is computed here, not shipped by the checkpoint.** The release has no speaker table
(`spk2info.pt` is absent) and every usage example clones `asset/zero_shot_prompt.wav` from the GitHub
checkout. Cloning needs two more models -- the S3 speech tokenizer v3 and CAMPPlus -- which the release
ships only as ONNX. This export runs them ONCE, in Python, through the reference's own frontend, on the
checkout's prompt clip and transcript, and ships the result (prompt text ids, prompt speech tokens,
prompt mel, speaker embedding) as `driver_weights`: Pocket-TTS's arrangement (loom.cpp ADR-043/045),
where a voice is DATA and neither ONNX model reaches the engine.

Usage:
  loom-export ~/Dev/models/fun-cosyvoice3-0.5b-2512 -o cosyvoice3.gguf --task text-to-speech \
      --model cosyvoice3
"""
import math
import sys
import tempfile
import types
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .flow_matching_export import FlowMatchingSpec
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Axis, Unchecked

# Where the reference checkouts live. The `cosyvoice` package is not on PyPI, and its requirements pin
# torch 2.3, transformers 4.51, deepspeed and tensorrt for a handful of `nn.Module`s -- the trade
# `chatterbox_export` records, with the same resolution: git clones on `sys.path`. Matcha-TTS is the
# checkout's own `third_party` submodule (its mel front end builds the voice's prompt features).
COSYVOICE_REPO = "/home/flavio/Dev/CosyVoice"
MATCHA_REPO = "/home/flavio/Dev/Matcha-TTS"
# The voice every usage example in the release's README clones, and its transcript. The
# `You are a helpful assistant.<|endofprompt|>` prefix is part of the PROMPT TEXT the LM reads -- the
# reference asserts `<|endofprompt|>` (151646) is somewhere in prompt text + text.
DEFAULT_VOICE_WAV = "asset/zero_shot_prompt.wav"
DEFAULT_VOICE_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
END_OF_PROMPT = 151646

SAMPLE_RATE = 24000
N_MEL = 80
TOKEN_MEL_RATIO = 2
# Output samples per mel frame: CausalHiFT's upsample rates (8, 5, 3) times its iSTFT hop (4).
SAMPLES_PER_FRAME = 480
NSF_HARMONICS = 9
SPEECH_VOCAB = 6561
# `CosyVoice3LM`'s speech-side rows: `sos = speech_token_size + 0`, `task_id = + 2`. Its `llm_decoder`
# and `speech_embedding` are both `speech_token_size + 200` wide, and EVERY id at or above 6561 is a
# stop id (`stop_token_ids`).
SOS, TASK_ID = SPEECH_VOCAB + 0, SPEECH_VOCAB + 2
LM_HEAD = SPEECH_VOCAB + 200
# `cosyvoice3.yaml`'s `ras_sampling` partial and `Qwen2LM.inference`'s defaults.
RAS_TOP_K, RAS_TOP_P, RAS_WIN, RAS_TAU = 25, 0.8, 10, 0.1
MIN_TOKEN_TEXT_RATIO, MAX_TOKEN_TEXT_RATIO = 2, 20
# `CosyVoice3Model.silent_tokens` and `llm_job`'s cap: a run of these longer than 5 is cut to 5 on the
# way to the flow (the LM and the RAS window still see all of them).
SILENT_TOKENS = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
MAX_SILENT_RUN = 5
FLOW_STEPS = 10
FLOW_CFG_RATE = 0.7
# The LM's KV cache. A prefill is 2 + the prompt text + the text + the prompt's speech tokens (87 for
# the default voice, at most 750 for the reference's 30 s prompt cap), and a decode is at most 20 x the
# text. With GQA's 2 K/V heads the cache is 24 layers x 2 x 128 x 4096 x 4 bytes = 100 MB.
LM_MAX_POSITIONS = 4096
# The flow's frame budget: twice every token the LM could hold. The DiT's rope table is this long.
FLOW_MAX_FRAMES = TOKEN_MEL_RATIO * LM_MAX_POSITIONS

# Trace lengths: odd and distinct from every static dimension (2 K/V heads, 14 heads, 9 harmonics, 16
# groups, 25 top-k...), so no fusion can confuse a sequence axis with a head or channel axis.
TRACE_TOKENS = 23
TRACE_FRAMES = 2 * TRACE_TOKENS
TRACE_STEPS = 7


def import_cosyvoice() -> None:
    """Put the reference checkouts on `sys.path`, with `modelscope` stubbed: `cosyvoice/cli` imports it
    for `snapshot_download`, which a local checkpoint never calls."""
    if "modelscope" not in sys.modules:
        stub = types.ModuleType("modelscope")

        def _no_download(*args, **kwargs):
            raise RuntimeError("cosyvoice3_export loads a local checkpoint; nothing is downloaded")

        stub.snapshot_download = _no_download
        sys.modules["modelscope"] = stub
    for path in (MATCHA_REPO, COSYVOICE_REPO):
        if path not in sys.path:
            sys.path.insert(0, path)


def install_patches() -> None:
    """The reference's causal convolutions build their zero padding as a traced
    `torch.zeros(x.shape[0], x.shape[1], p)` and concatenate it -- a dynamic-shape constant this
    pipeline cannot carry. `F.pad` with the same width on the same side is exact and constant-shaped.
    Only the `cache`-less call (the non-streaming one, the only kind this export makes) is rewritten;
    a streaming call with a real cache keeps the reference's path."""
    from cosyvoice.transformer import convolution

    conv = convolution.CausalConv1d
    if getattr(conv.forward, "_loom_padded", False):
        return
    real = conv.forward

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        if cache.size(2) != 0:
            return real(self, x, cache)
        pad = (self.causal_padding, 0) if self.causal_type == "left" else (0, self.causal_padding)
        return nn.Conv1d.forward(self, F.pad(x, pad))

    forward._loom_padded = True
    conv.forward = forward


def fold_weight_norm(module: nn.Module) -> None:
    """Every `parametrizations.weight_norm` folded into a plain weight -- see
    `chatterbox_export.fold_weight_norm`. CausalHiFT and its F0 predictor use the new-style
    parametrization, which carries the folded value in its `original0`/`original1` pair, so
    `leave_parametrized=True` is exact here (the old-style trap of loom.cpp Retro-056 does not apply)."""
    from torch.nn.utils import parametrize

    for sub in module.modules():
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)


# ------------------------------------------------------------------------------------------- LM --

def _rms_norm(norm, x):
    """`Qwen2RMSNorm`: `weight * (x * rsqrt(mean(x^2) + eps))`."""
    variance = (x * x).mean(dim=-1, keepdim=True)
    return norm.weight * (x * torch.rsqrt(variance + norm.variance_epsilon))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x, n_rep: int):
    """HF's `repeat_kv`, spelled as HF spells it -- the spelling `passes.py`'s GQA fusion recognises
    and strips, so the cache holds the checkpoint's 2 K/V heads and not 14 (voxcpm2_export's
    `_repeat_kv`, for the same reason)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class _Qwen2Layer(nn.Module):
    """`Qwen2DecoderLayer` in HF's own attention order (q/k/v with bias, rope, scaled `Q @ K^T + mask`,
    softmax, `@ V`, `o_proj`) -- the window `fuse_loom_attention` turns into a cached ATTENTION node.

    Written out rather than called because HF's `Qwen2Attention` dispatches through
    `ALL_ATTENTION_FUNCTIONS` and takes its rope as `position_embeddings`; spelling it is what keeps the
    traced graph the one the fusion pass reads."""

    def __init__(self, layer, config):
        super().__init__()
        attn = layer.self_attn
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_rep = self.heads // self.kv_heads
        self.input_layernorm, self.post_attention_layernorm = layer.input_layernorm, layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, x, cos, sin, mask):
        b, s, _ = x.shape
        h = _rms_norm(self.input_layernorm, x)
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        scores = torch.matmul(q * (1.0 / math.sqrt(self.head_dim)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        x = x + self.o_proj(ctx.transpose(1, 2).reshape(b, s, self.heads * self.head_dim))
        h = _rms_norm(self.post_attention_layernorm, x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))


class LMPhase(nn.Module):
    """`(text_ids, speech_ids, speech_mask, position_ids, attention_mask) -> logits of the LAST row`,
    `(1, 1, 6761)`.

    Each input row is `speech_embedding(speech_id)` where `speech_mask` is 1 and
    `embed_tokens(text_id)` where it is 0 -- a blend by arithmetic, the unused id being 0 (a valid row
    of both tables). That one rule covers the prefill (`sos` and `task_id` are speech rows) and every
    decode step (one speech row), so `Qwen2LM.inference`'s concatenation happens in the driver's
    id arrays and not on embeddings.

    RoPE's cos/sin are rows of the reference's own `Qwen2RotaryEmbedding` table, gathered by position.
    The head is `llm_decoder` (bias-free), NOT Qwen's tied `lm_head`, which the reference never calls."""

    def __init__(self, llm):
        super().__init__()
        qwen = llm.llm.model
        config = qwen.config
        self.embed_tokens = qwen.model.embed_tokens
        self.speech_embedding = llm.speech_embedding
        self.layers = nn.ModuleList(_Qwen2Layer(layer, config) for layer in qwen.model.layers)
        self.norm = qwen.model.norm
        self.head = llm.llm_decoder
        positions = torch.arange(LM_MAX_POSITIONS).view(1, -1)
        cos, sin = qwen.model.rotary_emb(torch.zeros(1, dtype=torch.float32), positions)
        self.register_buffer("cos_table", cos[0].detach().clone().float())   # (LM_MAX_POSITIONS, 64)
        self.register_buffer("sin_table", sin[0].detach().clone().float())

    def forward(self, text_ids, speech_ids, speech_mask, position_ids, attention_mask):
        x = self.embed_tokens(text_ids) * (1.0 - speech_mask) + self.speech_embedding(speech_ids) * speech_mask
        cos = F.embedding(position_ids, self.cos_table).unsqueeze(1)          # (1, 1, n, 64)
        sin = F.embedding(position_ids, self.sin_table).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return self.head(_rms_norm(self.norm, x)[:, -1:])


# --------------------------------------------------------------------------------------- flow --

class FlowEncoderPhase(nn.Module):
    """`(tokens, embedding) -> (mu, spks)`: `CausalMaskedDiffWithDiT.inference` up to the decoder, for
    one unpadded utterance.

    `tokens` is the voice's prompt tokens followed by the generated ones, one sequence, as the reference
    concatenates them. The pad mask it multiplies by is all ones and is dropped; `repeat_interleave(2)`
    along time is a nearest-neighbour 2x upsample, which is the same map and traces to a fixed-factor
    resize rather than to an index gather. `mu` comes back FRAME-major, `(1, 2n, 80)`, the ODE state's
    own layout (loom.cpp Retro-052: one layout at every join)."""

    def __init__(self, flow):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.spk_affine = flow.spk_embed_affine_layer
        self.pre = flow.pre_lookahead_layer
        if flow.token_mel_ratio != TOKEN_MEL_RATIO:
            raise NotImplementedError(f"token_mel_ratio {flow.token_mel_ratio}; this export assumes 2")

    def forward(self, tokens, embedding):                                     # (1, n) i32, (1, 192)
        spks = self.spk_affine(F.normalize(embedding, dim=1))                 # (1, 80)
        x = self.input_embedding(tokens)                                      # (1, n, 80)
        # `PreLookaheadLayer.forward` with no context, spelled: its default `context` is a
        # `torch.zeros(0, 0, 0)` that would trace a transpose and a copy of an empty constant. Right-pad
        # by the look-ahead, conv, leaky ReLU, left-pad causally, conv, residual.
        pre = self.pre
        h = F.pad(x.transpose(1, 2), (0, pre.pre_lookahead_len))
        h = F.leaky_relu(pre.conv1(h))
        h = pre.conv2(F.pad(h, (pre.conv2.kernel_size[0] - 1, 0)))
        h = h.transpose(1, 2) + x
        h = F.interpolate(h.transpose(1, 2), scale_factor=float(TOKEN_MEL_RATIO), mode="nearest")
        return h.transpose(1, 2), spks                                        # (1, 2n, 80), (1, 80)


def _apply_rope_first_head(t, freqs, scale=1.0):
    """x_transformers' `apply_rotary_pos_emb` as CosyVoice3's DiT calls it: on the PROJECTED query and
    key, `(1, n, 1024)`, BEFORE they are split into heads -- so with a 64-wide table the function's
    partial-rotary branch rotates channels `[0, 64)`, which is head 0, and passes the other 15 heads
    through untouched. That is the reference's arithmetic (the early F5-TTS `AttnProcessor` it copied
    applied rope before the head split; later F5-TTS moved it after), and the checkpoint was trained
    with it, so it is reproduced rather than corrected.

    `f5_tts_export._apply_rope_precomputed`'s other two substitutions carry over unchanged: cos/sin
    precomputed and sliced once, and the interleaved `rotate_half` as a constant `(64, 64)` pair-swap
    matmul (a 5-D rearrange otherwise). No re-slice (loom.cpp Retro-051)."""
    cos, sin, swap = freqs
    if scale != 1.0:
        raise NotImplementedError("CosyVoice3 rope: xpos scaling is not part of this checkpoint")
    rot = cos.shape[-1]
    head, rest = t[..., :rot], t[..., rot:]
    return torch.cat([head * cos + torch.matmul(head, swap) * sin, rest], dim=-1)


class EstimatorPhase(nn.Module):
    """`(x, mu, spks, cond, t) -> dx/dt`, all frame-major: `DiT.forward` for one unpadded utterance.

    F5-TTS's DiT with a causal position convolution and a speaker column. Its rope rotates head 0 only
    (see `_apply_rope_first_head`, which is also where `f5_tts_export`'s three rope substitutions are
    re-applied). `static_chunk_size` only matters
    when `streaming=True`; the non-streaming reference builds an all-ones mask, so no mask is built.
    `spks` is broadcast over time by ADDITION to a zero tensor rather than einops' `repeat` (a dynamic
    `tile`)."""

    def __init__(self, dit):
        super().__init__()
        from cosyvoice.flow.DiT import modules as dit_modules
        from .f5_tts_export import _rope_pair_swap

        self.d = dit
        freqs, _ = dit.rotary_embed.forward_from_seq_len(FLOW_MAX_FRAMES)
        self.register_buffer("rope_cos", freqs.cos())
        self.register_buffer("rope_sin", freqs.sin())
        self.register_buffer("rope_swap", _rope_pair_swap(freqs.shape[-1]))
        # `AttnProcessor` resolves `apply_rotary_pos_emb` as a module global at call time; idempotent.
        dit_modules.apply_rotary_pos_emb = _apply_rope_first_head
        if dit.long_skip_connection is not None:
            raise NotImplementedError("this DiT has a long skip connection; CosyVoice3's does not")

    def forward(self, x, mu, spks, cond, t):                                  # frame-major, (1,)
        d = self.d
        time = d.time_embed(t)
        spk = torch.zeros_like(x) + spks.unsqueeze(1)
        h = d.input_embed.proj(torch.cat([x, cond, mu, spk], dim=-1))
        h = d.input_embed.conv_pos_embed(h) + h
        n = x.shape[1]
        rope = ((self.rope_cos[:, :n, :], self.rope_sin[:, :n, :], self.rope_swap), None)
        for block in d.transformer_blocks:
            h = block(h, time, mask=None, rope=rope)
        h = d.norm_out(h, time)
        return d.proj_out(h)                                                  # (1, T, 80)


# ------------------------------------------------------------------------------------ vocoder --

class CausalHiftVocoderPhase(nn.Module):
    """`(mel, nsf_noise) -> waveform`: `CausalHiFTGenerator.inference(finalize=True)`.

    **The sine source runs at the FRAME rate**, which is what the causal `SineGen2` computes once its
    interpolations are read: it takes `(f0 * h / sr) mod 1` at the sample rate, downsamples it by 480
    with a LINEAR interpolation that lands exactly between two samples of the same (nearest-upsampled)
    frame -- so it returns each frame's own value, exactly -- takes `2 pi * cumsum` over frames, scales
    by 480, and upsamples the PHASE by nearest. The sine is therefore constant within each frame. This
    wrapper computes the same numbers without the 480x round trip, in the same order (`cumsum * 2 *
    pi`, then `* 480`), because the phase reaches ~1e6 rad where an f32 ulp is 0.06 rad and a
    reordered product is a different waveform.

    The causal initial phase `rand_ini` is dropped: it is added to sample 0 only, and the downsample
    never reads sample 0 (max|d| 0.0 with it zeroed). **`nsf_noise` is the one live draw**, a UNIFORM
    `[0, 1)` value per harmonic per sample (`SineGen2.sine_waves`, a construction-time `torch.rand`),
    harmonic-major `(1, 9, n_samples)`.

    Three substitutions as in `chatterbox_export.HiftVocoderPhase`, each exact: `% 1` as
    `x - floor(x)`, `f0 > 10` as `clamp((f0 - 10) * 1e30, 0, 1)`, and the harmonic multiply as a
    concatenation. The F0 predictor runs at f32; the reference runs it at f64, and the gate's tolerance
    is measured with that difference in it.
    """

    def __init__(self, hift):
        super().__init__()
        from .istft import ISTFT

        self.h = hift
        sg = hift.m_source.l_sin_gen
        self.harmonics = sg.harmonic_num + 1
        self.sine_amp, self.noise_std = float(sg.sine_amp), float(sg.noise_std)
        self.voiced_threshold, self.sr = float(sg.voiced_threshold), float(sg.sampling_rate)
        self.upsample_scale = int(sg.upsample_scale)
        if self.upsample_scale != SAMPLES_PER_FRAME or not sg.causal:
            raise NotImplementedError("CausalHiFT with a 480-sample frame and the causal SineGen2 only")
        n_fft, hop = hift.istft_params["n_fft"], hift.istft_params["hop_len"]
        self.n_fft, self.hop = n_fft, hop
        window = hift.stft_window.to(torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        angle = 2 * torch.pi * k * n / n_fft
        self.register_buffer("stft_re", (window * torch.cos(angle)).float()[:, None, :])
        self.register_buffer("stft_im", (-window * torch.sin(angle)).float()[:, None, :])
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)

    def source(self, f0, nsf_noise):
        """`SourceModuleHnNSF.forward` -> `(1, 1, n_samples)`."""
        f0 = f0.unsqueeze(1)                                                    # (1, 1, m)
        rad = torch.cat([(f0 * float(i + 1)) / self.sr for i in range(self.harmonics)], dim=1)  # (1, 9, m)
        rad = rad - torch.floor(rad)
        phase = torch.cumsum(rad, dim=-1) * 2 * np.pi
        phase = F.interpolate(phase * float(self.upsample_scale), scale_factor=float(self.upsample_scale),
                              mode="nearest")                                   # (1, 9, n)
        sine = torch.sin(phase) * self.sine_amp
        f0_up = F.interpolate(f0, scale_factor=float(self.upsample_scale), mode="nearest")
        uv = torch.clamp((f0_up - self.voiced_threshold) * 1e30, 0.0, 1.0)     # (1, 1, n)
        noise_amp = uv * self.noise_std + (1 - uv) * (self.sine_amp / 3)
        sine = sine * uv + noise_amp * nsf_noise
        lin = self.h.m_source.l_linear
        return torch.tanh(F.conv1d(sine, lin.weight.unsqueeze(-1), lin.bias))  # (1, 1, n)

    def forward(self, mel, nsf_noise):
        # mel (1, m, 80) frame-major; nsf_noise (1, 9, 480 m)
        h = self.h
        x = mel.transpose(1, 2)                                                # (1, 80, m)
        f0 = h.f0_predictor(x, finalize=True)                                  # (1, m)
        s = self.source(f0, nsf_noise).squeeze(1)                             # (1, n)
        pad = self.n_fft // 2
        s = F.pad(s.unsqueeze(1), (pad, pad), mode="reflect")
        s_stft = torch.cat([F.conv1d(s, self.stft_re, stride=self.hop),
                            F.conv1d(s, self.stft_im, stride=self.hop)], dim=1)
        x = h.conv_pre(x)
        for i in range(h.num_upsamples):
            x = F.leaky_relu(x, h.lrelu_slope)
            x = h.ups[i](x)
            if i == h.num_upsamples - 1:
                x = h.reflection_pad(x)
            x = x + h.source_resblocks[i](h.source_downs[i](s_stft))
            xs = None
            for j in range(h.num_kernels):
                r = h.resblocks[i * h.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / h.num_kernels
        x = F.leaky_relu(x)
        x = h.conv_post(x)
        half = self.n_fft // 2 + 1
        magnitude = torch.clip(torch.exp(x[:, :half, :]), max=1e2)
        phase = torch.sin(x[:, half:, :])
        wave = self.istft(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        return torch.clamp(wave, -h.audio_limit, h.audio_limit).reshape(1, -1)


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask (`chatterbox_export.causal_mask`'s form)."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


# ------------------------------------------------------------------------------- the reference --

def build_reference(model_dir: str):
    """`cosyvoice3.yaml`'s `llm`, `flow` and `hift`, constructed with the yaml's own arguments, then
    loaded the way `CosyVoice3Model.load` loads them.

    Not through `hyperpyyaml`: its 1.2.3 (the reference's pin) does not load under ruamel.yaml 0.19,
    and the yaml's module graph is three constructors deep, so it is written out. The loader's
    `strict=True` is what says the constructor arguments are the checkpoint's."""
    from omegaconf import DictConfig
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    llm = CosyVoice3LM(llm_input_size=896, llm_output_size=896, speech_token_size=SPEECH_VOCAB,
                       length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15],
                       llm=Qwen2Encoder(pretrain_path=f"{model_dir}/CosyVoice-BlankEN"),
                       sampling=partial(ras_sampling, top_p=RAS_TOP_P, top_k=RAS_TOP_K,
                                        win_size=RAS_WIN, tau_r=RAS_TAU))
    estimator = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=N_MEL, mu_dim=N_MEL,
                    spk_dim=N_MEL, out_channels=N_MEL, static_chunk_size=25 * TOKEN_MEL_RATIO,
                    num_decoding_left_chunks=-1)
    cfm = CausalConditionalCFM(
        in_channels=240, n_spks=1, spk_emb_dim=N_MEL, estimator=estimator,
        cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                               "training_cfg_rate": 0.2, "inference_cfg_rate": FLOW_CFG_RATE,
                               "reg_loss_type": "l1"}))
    flow = CausalMaskedDiffWithDiT(
        input_size=N_MEL, output_size=N_MEL, spk_embed_dim=192, output_type="mel",
        vocab_size=SPEECH_VOCAB, input_frame_rate=25, only_mask_loss=True,
        token_mel_ratio=TOKEN_MEL_RATIO, pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=N_MEL, channels=1024, pre_lookahead_len=3),
        decoder=cfm)
    hift = CausalHiFTGenerator(
        in_channels=N_MEL, base_channels=512, nb_harmonics=8, sampling_rate=SAMPLE_RATE, nsf_alpha=0.1,
        nsf_sigma=0.003, nsf_voiced_threshold=10, upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7], istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3,
        source_resblock_kernel_sizes=[7, 7, 11], source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=N_MEL, cond_channels=512))
    load = partial(torch.load, map_location="cpu", weights_only=True)
    llm.load_state_dict(load(f"{model_dir}/llm.pt"), strict=True)
    flow.load_state_dict(load(f"{model_dir}/flow.pt"), strict=True)
    hift.load_state_dict({k.replace("generator.", ""): v for k, v in load(f"{model_dir}/hift.pt").items()},
                         strict=True)
    return llm.eval(), flow.eval(), hift.eval()


def build_frontend(model_dir: str):
    """The reference's `CosyVoiceFrontEnd` with `cosyvoice3.yaml`'s tokenizer and mel extractor. It
    opens the two ONNX models (S3 tokenizer v3, CAMPPlus); they run here and nowhere else."""
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
    from matcha.utils.audio import mel_spectrogram

    feat = partial(mel_spectrogram, n_fft=1920, num_mels=N_MEL, sampling_rate=SAMPLE_RATE, hop_size=480,
                   win_size=1920, fmin=0, fmax=None, center=False)
    tok = partial(get_qwen_tokenizer, token_path=f"{model_dir}/CosyVoice-BlankEN",
                  skip_special_tokens=True, version="cosyvoice3")
    return CosyVoiceFrontEnd(tok, feat, f"{model_dir}/campplus.onnx",
                             f"{model_dir}/speech_tokenizer_v3.onnx", f"{model_dir}/spk2info.pt", "all")


def compute_voice(frontend, wav_path: str, prompt_text: str) -> Dict[str, np.ndarray]:
    """`CosyVoiceFrontEnd.frontend_zero_shot`'s voice half -> the four driver weights.

    Everything `add_zero_shot_spk` would store in `spk2info` except the text: prompt text ids, prompt
    speech tokens (trimmed with the prompt mel to exactly 2 frames per token, which the reference does
    for any 24 kHz model), the prompt mel FRAME-major, and CAMPPlus's 192-d embedding. One embedding
    serves both the LM (`llm_embedding`, which `CosyVoice3LM` ignores) and the flow."""
    if END_OF_PROMPT not in frontend._extract_text_token(prompt_text)[0][0].tolist():
        raise ValueError(f"the voice's prompt text has no <|endofprompt|>; CosyVoice3's LM asserts one "
                         f"in prompt text + text, and the prompt is where the reference puts it")
    mi = frontend.frontend_zero_shot("", prompt_text, wav_path, SAMPLE_RATE, "")
    n_tokens = mi["flow_prompt_speech_token"].shape[1]
    n_frames = mi["prompt_speech_feat"].shape[1]
    if n_frames != TOKEN_MEL_RATIO * n_tokens:
        raise ValueError(f"prompt is {n_tokens} tokens and {n_frames} frames; the flow in-fills after "
                         f"exactly {TOKEN_MEL_RATIO} frames per token")
    return {
        "voice.prompt_text": mi["prompt_text"][0].numpy().astype(np.float32),
        "voice.prompt_speech_tokens": mi["flow_prompt_speech_token"][0].numpy().astype(np.float32),
        "voice.prompt_feat": mi["prompt_speech_feat"][0].reshape(-1).numpy().astype(np.float32),
        "voice.embedding": mi["flow_embedding"][0].numpy().astype(np.float32),
    }


def stage_tokenizer(model_dir: str, staging: str) -> int:
    """`CosyVoice3Tokenizer`, written out as a `tokenizer.json` the BPE writer reads.

    The checkpoint's `CosyVoice-BlankEN` is a plain Qwen2 tokenizer (vocab.json + merges.txt, three
    added tokens); the reference adds ~290 MORE at load time -- `<|endofprompt|>`, the paralinguistic
    tags (`[breath]`, `<laughter>`), and CMU/pinyin phoneme tokens for pronunciation in-painting -- with
    `add_special_tokens`, so their ids are whatever transformers assigns in list order. The ids are
    taken from transformers doing exactly that, not recomputed, and saved; the writer then treats them
    as any other added token (split out of the raw text before BPE). Returns the chunk header's id."""
    from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer

    from .cosyvoice3_tokenizer_export import CHUNK_HEADER

    tok = CosyVoice3Tokenizer(token_path=f"{model_dir}/CosyVoice-BlankEN").tokenizer
    if tok.convert_tokens_to_ids("<|endofprompt|>") != END_OF_PROMPT:
        raise ValueError("CosyVoice3Tokenizer did not put <|endofprompt|> at 151646, the id the LM checks")
    tok.save_pretrained(staging)
    return int(tok.convert_tokens_to_ids(CHUNK_HEADER))


@dataclass(kw_only=True)
class CosyVoice3ExportConfig(BaseMultiPhaseModelExportConfig):
    """A `Fun-CosyVoice3-0.5B-2512` directory -> one Loom GGUF."""

    architecture: str = "cosyvoice3"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "cosyvoice3_driver"
    voice_wav: Optional[str] = None
    voice_text: str = DEFAULT_VOICE_TEXT
    _voice: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _chunk_header: Optional[int] = field(default=None, init=False, repr=False)
    _staging: Optional[tempfile.TemporaryDirectory] = field(default=None, init=False, repr=False)
    _voice_compat: Optional[str] = field(default=None, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked(
            "path to the checkpoint directory; the recognizer found `cosyvoice3.yaml`, `llm.pt`, "
            "`flow.pt`, `hift.pt` and the two ONNX front-end models in it"),
        "decomposition": Unchecked("MultiPhase by construction -- four graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                         "against the traced topologies by LuaFragment"),
        "voice_wav": Unchecked("the default voice's clip; None means the checkout's zero_shot_prompt.wav"),
        "voice_text": Unchecked("the default voice's prompt text, <|endofprompt|> checked in compute_voice"),
        "_voice": Unchecked("computed from the clip during phases() and shipped as driver weights"),
        "_chunk_header": Unchecked("the chunk separator's id, read off the staged tokenizer in phases()"),
        "_staging": Unchecked("a temporary directory holding the staged tokenizer.json"),
        "_voice_compat": Unchecked("the LM and flow weights' fingerprint, read once by contract()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        import_cosyvoice()
        install_patches()
        llm, flow, hift = build_reference(self.model_dir)
        fold_weight_norm(hift)
        frontend = build_frontend(self.model_dir)
        self._voice = compute_voice(frontend, self.voice_wav or f"{COSYVOICE_REPO}/{DEFAULT_VOICE_WAV}",
                                    self.voice_text)
        self._staging = tempfile.TemporaryDirectory(prefix="cosyvoice3_tok_")
        self._chunk_header = stage_tokenizer(self.model_dir, self._staging.name)
        del frontend

        hidden = llm.llm_input_size
        seq_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        code_dim = ct.RangeDim(4, LM_MAX_POSITIONS)
        frame_dim = ct.RangeDim(8, FLOW_MAX_FRAMES)
        gen_dim = ct.RangeDim(2, FLOW_MAX_FRAMES)
        trace_mask = torch.zeros(1, TRACE_STEPS, 1)
        trace_mask[0, 0] = trace_mask[0, -2:] = 1.0
        return [
            ExportPhase(
                name="lm",
                wrapper=LMPhase(llm).eval(),
                dummy_inputs=(torch.randint(0, 151000, (1, TRACE_STEPS), dtype=torch.int32),
                              torch.randint(0, SPEECH_VOCAB, (1, TRACE_STEPS), dtype=torch.int32),
                              trace_mask,
                              torch.arange(TRACE_STEPS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_STEPS)),
                mil_inputs=[
                    ct.TensorType(name="text_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="speech_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="speech_mask", shape=(1, seq_dim, 1), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim), dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="flow_encoder",
                wrapper=FlowEncoderPhase(flow).eval(),
                dummy_inputs=(torch.randint(0, SPEECH_VOCAB, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.randn(1, 192)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, code_dim), dtype=np.int32),
                            ct.TensorType(name="embedding", shape=(1, 192), dtype=np.float32)],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="estimator",
                wrapper=EstimatorPhase(flow.decoder.estimator).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL), torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.randn(1, N_MEL), torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.tensor([0.3])),
                mil_inputs=[
                    ct.TensorType(name="x", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="mu", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="spks", shape=(1, N_MEL), dtype=np.float32),
                    ct.TensorType(name="cond", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="t", shape=(1,), dtype=np.float32),
                ],
                # `n_tokens`: `FlowMatchingSampler` binds its length under that name (Chatterbox's and
                # F5-TTS's estimators are declared the same way).
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="vocoder",
                wrapper=CausalHiftVocoderPhase(hift).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.rand(1, NSF_HARMONICS, TRACE_FRAMES * SAMPLES_PER_FRAME)),
                mil_inputs=[
                    ct.TensorType(name="mel", shape=(1, gen_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="nsf_noise",
                                  shape=(1, NSF_HARMONICS, ct.RangeDim(2 * SAMPLES_PER_FRAME,
                                                                       FLOW_MAX_FRAMES * SAMPLES_PER_FRAME)),
                                  dtype=np.float32),
                ],
                root_axis="n_enc_frames",
                # The noise's sample axis is the mel's frame axis times 480 -- one symbol, not two.
                declared_axes={"nsf_noise": {2: f"{SAMPLES_PER_FRAME} * n_enc_frames"}},
            ),
        ]

    def samplers(self) -> List[FlowMatchingSpec]:
        return [FlowMatchingSpec(
            func_name="sample_estimator",
            estimator="estimator",
            carried_input="x",
            time_input="t",
            fixed_inputs=["mu", "spks", "cond"],
            # `1 - cos(pi/2 * t)` over a linspace: `CausalConditionalCFM.forward`'s cosine schedule.
            schedule="caller",
            # `(1 + rate) * v_cond - rate * v_uncond`, the unconditional run seeing mu, spks and cond
            # zeroed: `solve_euler`'s batch of two, ADR-040's form exactly.
            guidance=True,
            # The reference's noise is a slice of one construction-time tensor; a waveform is only
            # reproducible from it, so a caller may hand it over.
            caller_noise=True,
            note="Euler over CosyVoice3's DiT on the cosine schedule, under classifier-free\n"
                 "guidance: the unconditional evaluation sees mu, spks and cond zeroed.",
        )]

    def driver_components(self) -> List:
        from .driver_components import (
            CALLER, DriverInputs, DriverReturn, ExportConstants, FlowMatchingSampler, LuaFragment,
            SubgraphCallComponent,
        )
        from .driver_ir import BinOp, FieldAccess, Len, OutputRef, Var

        fragment = self.driver_script_path
        n_frames, n_gen = Var("n_frames"), Var("n_gen_frames")
        constants = {
            # What `loom::CosyVoice3Vocab` opens each chunk with (`<|endoftext|>`, read off the tokenizer).
            "CHUNK_HEADER": self._chunk_header,
            "SOS": SOS, "TASK_ID": TASK_ID, "END_OF_PROMPT": END_OF_PROMPT,
            "LM_MAX_POSITIONS": LM_MAX_POSITIONS,
            "RAS_TOP_K": RAS_TOP_K, "RAS_TOP_P": RAS_TOP_P, "RAS_WIN": RAS_WIN, "RAS_TAU": RAS_TAU,
            "MIN_TOKEN_TEXT_RATIO": MIN_TOKEN_TEXT_RATIO, "MAX_TOKEN_TEXT_RATIO": MAX_TOKEN_TEXT_RATIO,
            "MAX_SILENT_RUN": MAX_SILENT_RUN,
            "TOKEN_MEL_RATIO": TOKEN_MEL_RATIO, "N_MEL": N_MEL, "SAMPLES_PER_FRAME": SAMPLES_PER_FRAME,
            "NSF_HARMONICS": NSF_HARMONICS, "FLOW_STEPS": FLOW_STEPS, "FLOW_CFG_RATE": FLOW_CFG_RATE,
        }
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True,
                        defines=("cosyvoice3_voice", "cosyvoice3_cosine_times", "cosyvoice3_zeros",
                                 "COSYVOICE3_SILENT", "cosyvoice3_split_chunks")),
            ExportConstants(values=constants),
            # The text's ids are the one required input; the voice, the knobs and the draws are optional.
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "00_chunks.lua", reads=("tokens", "CHUNK_HEADER")),
            LuaFragment(fragment / "01_lm.lua",
                        reads=("tokens", "SOS", "TASK_ID", "END_OF_PROMPT",
                               "LM_MAX_POSITIONS", "RAS_TOP_K", "RAS_TOP_P", "RAS_WIN", "RAS_TAU",
                               "MIN_TOKEN_TEXT_RATIO", "MAX_TOKEN_TEXT_RATIO", "MAX_SILENT_RUN"),
                        defines=("speech_tokens",)),
            LuaFragment(fragment / "02_flow_plan.lua",
                        reads=("speech_tokens", "TOKEN_MEL_RATIO", "N_MEL", "FLOW_STEPS", "FLOW_CFG_RATE"),
                        defines=("flow_tokens", "flow_embedding", "n_codes", "n_frames",
                                 "n_prompt_frames", "n_gen_frames", "step_cond", "zero_cond",
                                 "zero_spks", "times", "flow_cfg")),
            SubgraphCallComponent(
                topology="flow_encoder", outputs=(), retain=True, length=Var("n_codes"),
                inputs={"tokens": Var("flow_tokens"), "embedding": Var("flow_embedding")},
                note="--- mu (frame-major) and the projected speaker vector, both RETAINED: the\n"
                     "    sampler hands them to every conditional evaluation. ---"),
            FlowMatchingSampler(
                spec=self.samplers()[0], result="_mel", length=n_frames,
                n_elems=BinOp("*", n_frames, Var("N_MEL")),
                n_steps=None, times=Var("times"),
                step_inputs={"mu": OutputRef("flow_encoder", index=1),
                             "spks": OutputRef("flow_encoder", index=2),
                             "cond": Var("step_cond")},
                uncond_inputs={"mu": Var("zero_cond"), "spks": Var("zero_spks"),
                               "cond": Var("zero_cond")},
                guidance_scale=Var("flow_cfg"),
                state=FieldAccess("inputs", "noise"),
                note="--- The whole mel grid, in-filled after the voice's prompt frames. ---"),
            LuaFragment(fragment / "03_mel_tail.lua",
                        reads=("n_prompt_frames", "n_gen_frames", "N_MEL"),
                        defines=("mel_tail",), retains=("estimator",)),
            LuaFragment(fragment / "04_nsf.lua",
                        reads=("n_gen_frames", "NSF_HARMONICS", "SAMPLES_PER_FRAME"),
                        defines=("nsf_noise",)),
            SubgraphCallComponent(
                topology="vocoder", outputs=("wave",), length=n_gen,
                inputs={"mel": Var("mel_tail"), "nsf_noise": Var("nsf_noise")},
                note="--- CausalHiFT: the generated frames -> 24 kHz waveform. ---"),
            DriverReturn(values=("wave",)),
        ]

    def hparams(self) -> dict:
        return {"n_mel": N_MEL}

    def contract(self) -> dict:
        contract = super().contract()
        # Text, not phoneme ids: the LM reads its own Qwen2 BPE ids, and the table ships in the GGUF.
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        contract["tts.default_steps"] = FLOW_STEPS
        # What a voice file must match to be loaded into this model (`cosyvoice3_voices`, loom.cpp
        # ADR-045): the LM and flow weights, the two that read a voice's arrays. A fact about THESE
        # weights, so it is read only when there are weights to read -- an architecture-only query
        # (test_tts_text_door's nonexistent path) opens nothing.
        from .cosyvoice3_voices import DEFAULT_VOICE_NAME, weights_fingerprint

        if self._voice_compat is None and Path(self.model_dir).is_dir():
            self._voice_compat = weights_fingerprint(self.model_dir)
        if self._voice_compat is not None:
            contract["voice.compat"] = self._voice_compat
        # The voice the file carries, which is what `infer` uses when the caller names none.
        contract["tts.voices"] = [DEFAULT_VOICE_NAME]
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams())
        if self._staging is not None:
            # A byte-level Qwen2 BPE with CosyVoice3's added tokens, under the reference's text path
            # (`cosyvoice3_tokenizer_export`): numbers spelled out, paragraphs split into chunks.
            kwargs["tokenizer_dir"] = self._staging.name
            kwargs["tokenizer_family"] = "cosyvoice3"
        if self._voice is not None:
            kwargs["driver_weights"] = dict(self._voice)
        return kwargs


def _is_cosyvoice3(path: Path) -> bool:
    """A Fun-CosyVoice3 release: the yaml, the three torch checkpoints, the Qwen tokenizer directory and
    the two ONNX front-end models the default voice is computed with. `cosyvoice3.yaml` is the
    discriminator -- CosyVoice 1 and 2 ship `cosyvoice.yaml` / `cosyvoice2.yaml` and neither is this
    export."""
    return path.is_dir() and all((path / n).exists() for n in (
        "cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "CosyVoice-BlankEN",
        "campplus.onnx", "speech_tokenizer_v3.onnx"))


def _build_cosyvoice3(path: Path, output_path: str) -> LoomExportConfig:
    return CosyVoice3ExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=CosyVoice3ExportConfig,
        recognizers=[
            ModelRecognizer(name="cosyvoice3", detect=_is_cosyvoice3, build_config=_build_cosyvoice3),
        ],
    ))
