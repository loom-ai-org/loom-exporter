"""MOSS-Audio-Tokenizer's decode as a stream (loom.cpp ADR-103): one graph per transformer stage, each
over the engine's RING KV cache of its own.

The decoder is six causal transformers with plain reshapes between them -- no convolution anywhere -- at 1,
2, 4, 8, 16 and 32 positions per 25 Hz code frame, each attending to a window of 250, 500 or 800 positions.
Its whole streaming state is those windows' keys and values, 338 MiB at full context: carried as graph
inputs and outputs (`codec_stream_export`'s shape for Qwen3-TTS and Audio8) every push would copy all of it.
So each stage is a phase whose fused, cached ATTENTION nodes write into a ring of their own
(`ExportPhase.own_kv_cache`) of `window + positions per push` cells -- the reference's streaming
attention appends a call's keys to the window before attending and trims after, so the ring needs room for
both -- and the driver masks it with `loom.causal_mask(n, n_past, window, ring)`.

`StreamStage` is `moss_audio_tokenizer_export.BlockedStage`'s arithmetic -- the same Q, K, V, interleaved
RoPE at absolute positions, layer scales -- with the attention in the spelling `fuse_loom_attention`
recognises (scaled Q, `Q @ K^T + mask`, softmax, `@ V`). Its `past` argument is the torch-side stand-in for
the cache, never traced: `check_stream` runs the stages that way against the one-call decoder.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn

from .moss_audio_tokenizer_export import MossCodecDecoder, _outer, _patch_decode

# The most code frames one push decodes: what sizes every stage's ring past its window.
MAX_PUSH_FRAMES = 8
NEG = -1e30


class StreamStage(nn.Module):
    """One stage: `(x [1, n, C_in], position_ids [1, n], attention_mask [1, 1, n, n_kv]) -> [1, n * patch,
    C_out / patch]`, the stage's transformer then the patch reshape after it. Stage 0 takes the codes
    `[1, n, n_q]` instead of `x`, and runs the quantizer's table first."""

    def __init__(self, blocked, patch: int, positions_per_frame: int, codes: Optional[MossCodecDecoder] = None):
        super().__init__()
        self.input_proj, self.output_proj = blocked.input_proj, blocked.output_proj
        self.layers, self.qkv = blocked.layers, blocked.qkv
        self.H, self.C, self.Dh, self.W = blocked.H, blocked.C, blocked.Dh, blocked.W
        self.register_buffer("inv_freq", blocked.inv_freq.clone())
        self.patch = patch
        self.positions_per_frame = positions_per_frame
        self.ring = self.W + MAX_PUSH_FRAMES * positions_per_frame
        self.codes = codes

    def _rope(self, x, T, cos, sin):
        pairs = x.reshape(1, T, self.H * self.Dh // 2, 2)
        rotated = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(1, T, self.H, self.Dh)
        return x * cos + rotated * sin

    def forward(self, x, position_ids, attention_mask, past: Optional[List] = None):
        if self.codes is not None:
            d = self.codes
            n, n_q = x.shape[1], x.shape[2]
            index = (x.float() + d.offsets).to(torch.int32).reshape(n * n_q)
            rows = d.table[index]
            x = d.output_proj(rows.reshape(1, n, n_q, rows.shape[-1]).sum(dim=2))
        x = self.input_proj(x)
        T = x.shape[1]
        angle = _outer(position_ids.to(torch.float32)[0], self.inv_freq).reshape(1, T, 1, self.Dh)
        cos, sin = torch.cos(angle), torch.sin(angle)
        for index, (layer, qkv) in enumerate(zip(self.layers, self.qkv)):
            h = layer.norm1(x)
            q = self._rope(qkv[0](h).reshape(1, T, self.H, self.Dh), T, cos, sin).transpose(1, 2)
            k = self._rope(qkv[1](h).reshape(1, T, self.H, self.Dh), T, cos, sin).transpose(1, 2)
            v = qkv[2](h).reshape(1, T, self.H, self.Dh).transpose(1, 2)
            if past is not None:
                if past[index] is not None:
                    k = torch.cat([past[index][0], k], dim=2)
                    v = torch.cat([past[index][1], v], dim=2)
                past[index] = (k, v)
            scores = torch.matmul(q * (1.0 / math.sqrt(self.Dh)), k.transpose(-1, -2)) + attention_mask
            out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(1, T, self.C)
            x = x + layer.layer_scale_1(layer.self_attn.out_proj(out))
            x = x + layer.layer_scale_2(layer.ffn(layer.norm2(x)))
        return _patch_decode(self.output_proj(x), self.patch)


def stream_stages(decoder: MossCodecDecoder) -> List[StreamStage]:
    """The decoder's stages in order, each with the patch that follows it and its rate in positions per
    code frame. The plan alternates stage and patch and ends on a patch (240, to samples)."""
    stages, rate = [], 1
    plan = decoder.plan
    for i, (kind, arg) in enumerate(plan):
        if kind != "stage":
            continue
        kind_next, patch = plan[i + 1]
        if kind_next != "patch":
            raise NotImplementedError("a MOSS decoder stage not followed by a patch")
        stages.append(StreamStage(decoder.stages[arg], patch, rate, decoder if not stages else None))
        rate *= patch
    return stages


def window_mask(q_positions: torch.Tensor, k_positions: torch.Tensor, window: int) -> torch.Tensor:
    delta = q_positions.view(-1, 1) - k_positions.view(1, -1)
    return torch.where((delta >= 0) & (delta < window), 0.0, NEG).view(1, 1, q_positions.numel(), -1)


def check_stream(decoder: MossCodecDecoder, stages: List[StreamStage], n_frames: int, n_q: int, pushes,
                 atol: float = 1e-3) -> float:
    """The stages run push by push -- each layer's past keys kept and trimmed to the window, as the
    engine's ring does -- against the one-call decoder on the same random codes. Run before tracing."""
    torch.manual_seed(0)
    block = decoder.stages[0].B                       # the one-call decoder needs whole blocks
    n_frames = (n_frames + block - 1) // block * block
    codes = torch.randint(0, decoder.size - 1, (1, n_frames, n_q))
    with torch.no_grad():
        whole = decoder(codes).reshape(-1)
        pasts = [[None] * len(s.layers) for s in stages]
        got, at, i = [], 0, 0
        while at < n_frames:
            m = min(pushes[i % len(pushes)], n_frames - at)
            i += 1
            x = codes[:, at:at + m]
            for s, past in zip(stages, pasts):
                p = s.positions_per_frame
                q = torch.arange(at * p, (at + m) * p)
                kept = 0 if past[0] is None else past[0][0].shape[2]
                k = torch.arange(at * p - kept, (at + m) * p)
                x = s(x, q.view(1, -1).to(torch.int32), window_mask(q, k, s.W), past)
                for j, kv in enumerate(past):
                    past[j] = (kv[0][:, :, -s.W:], kv[1][:, :, -s.W:])
            got.append(x.reshape(-1))
            at += m
    delta = float((torch.cat(got) - whole).abs().max())
    if delta > atol:
        raise ValueError(f"the streamed stages disagree with the one-call decoder by {delta:.3g} (> {atol})")
    return delta
