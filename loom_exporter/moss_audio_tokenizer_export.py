"""MOSS-Audio-Tokenizer-v2's decode half (family 11): 32 residual-LFQ codebooks at 12.5 Hz in, 48 kHz
STEREO out, through six stacks of causal sliding-window transformers and no convolution at all.

It is the codec MOSS-TTS emits codes for, which is why it is here. Three things about it are new to
this family, and each is a place where the obvious export is wrong.

**The window is exact or it is nothing, so the graph is one call over the whole sequence.** Every
stack is a causal transformer whose attention sees the last `context` positions (125 frames at 12.5 Hz
growing to 400 positions at 400 Hz), and the reference's streaming decode -- a ring KV cache per layer,
which is what MOSS-TTS's processor calls, `chunk_duration=8` -- is the same function as one
whole-sequence pass: 1.1e-06 apart on 30 s of speech. What it is NOT is ADR-034's shape. A chunk that
re-decodes 100 frames (8 s) of left context and drops them is still 48% away in relative RMS, and
25 frames of context is 58%: 92 layers of stacked windows reach far past any context a chunk could
afford, so the Qwen3-TTS codec's driver loop would be a wrong answer here, not an approximate one.

**So the attention is BLOCKED rather than masked-dense, which is what makes one call affordable.** A
dense `T x T` score matrix at the 400 Hz stack is 12 heads x T^2 floats -- 0.8 GB per layer for 10 s,
7 GB for 30 s. Instead the sequence is cut into blocks of `m` codec frames (`m * 2**stage` positions),
and the queries of block b attend to the keys of blocks `b - p .. b` only, `p = ceil((W - 1) / B)`.
Relative to that span the window test `0 <= q - k < W` does not depend on b at all, so it is one
CONSTANT `[B, (p+1)B]` mask; the only b-dependent fact is that blocks before 0 do not exist, which is
a second, `[nb, (p+1)B]` mask built in-graph. The keys are gathered with ONE row gather per layer from
a clamped block index. Memory is linear in the clip, and every op is one the MIL path already lowers.
Verified against the reference's own decode at 1.6e-06 max over 30 s -- see `BlockedStage`.

**The driver pads the frame count to a multiple of `m`**, because the blocks must tile the sequence
and the graph cannot pad a dynamic axis (coremltools refuses dynamic padding). That is exact rather
than approximate for the same reason the window is: the decoder is causal end to end -- per-frame
lookups, per-position projections, causal attention, and patch reshapes that never mix frames -- so
nothing appended after the last real frame reaches any sample before it. The padding is trimmed off
the output.

**The quantizer is FOLDED, and each codebook gets an ABSENT row.** Decode is
`output_proj(sum_i out_proj_i(codebook_i[code_i]))`: 32 lookups of 8-wide codes, each through its own
1x1 projection to 512. Every `out_proj_i` is linear, so it is applied to its codebook ahead of time,
bias included, and the whole sum becomes one gather from a `[32 * 1025, 512]` table and a sum over the
codebook axis. Row 1024 of each codebook's block is all ZERO -- bias too -- so an id of 1024 contributes
exactly nothing. That is what a residual quantizer's prefix decode means (the reference's
`decode(num_quantizers=12)` sums the first 12 and stops), and it lets ONE file serve an LM that emits
fewer codebooks than the codec has: MOSS-TTS emits 12, and its rows are padded with 1024 -- which is
already MOSS's own `audio_pad_code`.

**The output is interleaved stereo.** The codec models `L R L R ...` as one 96 kHz stream
(`enable_channel_interleave`), so its raw output IS the interleaved waveform; the reference's final
`view(-1, 2).transpose` only de-interleaves it. The export hands back the interleaved floats and
declares `channels = 2` beside `sample_rate`.
"""
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# The reference's own additive "masked" value would be -inf through SDPA's boolean mask; a finite
# large negative gives exactly 0 after softmax because every query row keeps its own diagonal.
NEG = -1e30


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a[:, None] * b[None, :]` as a K=1 matmul. ggml's binary ops broadcast their SECOND operand
    only, so a mutual broadcast has to be spelled as a product -- the same spelling
    `pocket_tts_export.MimiDecoderPhase.window_mask` and `audio_codec_export._qwen3_tts_sliding_
    causal_mask` use. Exact for the small integers it is ever given."""
    return a.reshape(-1, 1) @ b.reshape(1, -1)


def fold_quantizer(quantizer) -> tuple:
    """`(table [n_q * (size + 1), rvq_dim], output_proj)` from a `MossAudioTokenizerResidualLFQ`.

    `table[i * (size + 1) + c] = out_proj_i(codebook_i[c])`, bias included, and
    `table[i * (size + 1) + size] = 0`. Reading `.weight` off a `parametrizations.weight_norm` module
    COMPUTES the normalised weight, so this is the weight the reference's forward uses.
    """
    blocks = []
    for lfq in quantizer.quantizers:
        codebook = lfq.codebook.weight.detach().float()                    # [size, dim]
        w = lfq.out_proj.weight.detach().float()[:, :, 0]                  # [rvq_dim, dim]
        b = lfq.out_proj.bias.detach().float()
        rows = codebook @ w.t() + b
        blocks.append(torch.cat([rows, torch.zeros(1, rows.shape[1])], 0))
    table = torch.cat(blocks, 0)
    out = quantizer.output_proj
    proj = nn.Linear(out.weight.shape[1], out.weight.shape[0])
    with torch.no_grad():
        proj.weight.copy_(out.weight.detach().float()[:, :, 0])
        proj.bias.copy_(out.bias.detach().float())
    return table, proj


class BlockedStage(nn.Module):
    """One `MossAudioTokenizerProjectedTransformer`, `[T, C_in] -> [T, C_out]`, attention blocked.

    Everything but the attention is the reference's own modules, called as they are. The attention is
    its arithmetic in a different order -- the same Q, K, V, the same interleaved-pair RoPE at the same
    absolute positions, the same window -- with the keys of each block gathered instead of the whole
    row masked.
    """

    def __init__(self, projected, frames_per_block: int, positions_per_frame: int):
        super().__init__()
        self.input_proj = projected.input_proj
        self.output_proj = projected.output_proj
        transformer = projected.transformer
        self.layers = transformer.layers
        attn = transformer.layers[0].self_attn
        if not attn.causal or attn.context is None:
            raise NotImplementedError(
                "every MOSS-Audio-Tokenizer stack in the released checkpoints is causal and windowed; "
                "this one is not, and the blocked attention below assumes both")
        self.H = attn.num_heads
        self.C = attn.embed_dim
        self.Dh = self.C // self.H
        self.W = int(attn.context)
        self.B = frames_per_block * positions_per_frame
        self.p = math.ceil((self.W - 1) / self.B)
        B, p, W = self.B, self.p, self.W
        # delta = q - k for a query at offset i of its block against key j of the (p+1)-block span
        # that starts p blocks earlier: (p*B + i) - j, which does not depend on the block.
        delta = torch.arange(B).view(B, 1) + p * B - torch.arange((p + 1) * B).view(1, -1)
        self.register_buffer("window_mask",
                             torch.where((delta >= 0) & (delta < W), 0.0, NEG).float())
        half = torch.arange(self.Dh // 2, dtype=torch.float32)
        inv_freq = torch.exp(half * (-math.log(transformer.rope.max_period) * 2 / self.Dh))
        # Each frequency twice, once per member of its pair -- `pocket_tts_export._rope_tables`.
        self.register_buffer("inv_freq", torch.repeat_interleave(inv_freq, 2))
        self.register_buffer("key_block_offsets", torch.arange(p + 1, dtype=torch.float32) - p)
        # `in_proj` split into Q, K and V, rows [0, C), [C, 2C), [2C, 3C) of its weight -- what its own
        # `reshape(..., 3, H, Dh)` selects. Split because selecting `qkv[:, 0]` is an integer index,
        # which lowers to a view that KEEPS its unit axis and then fails to broadcast against the RoPE
        # tables; `pocket_tts_export` split Mimi's `in_proj` for the same reason.
        self.qkv = nn.ModuleList()
        for layer in self.layers:
            weight = layer.self_attn.in_proj.weight.detach()
            parts = nn.ModuleList()
            for i in range(3):
                proj = nn.Linear(self.C, self.C, bias=False)
                with torch.no_grad():
                    proj.weight.copy_(weight[i * self.C:(i + 1) * self.C])
                parts.append(proj)
            self.qkv.append(parts)

    def _before_zero(self, n_blocks) -> tuple:
        """`(key block index [nb * (p+1)], mask [nb, 1, 1, (p+1)B])` for this call's block count.

        Block `b - p + k` is clamped to 0 when negative -- a real row, so the gather never reads
        outside the tensor -- and masked out, which is the only place a block's absolute position
        matters.
        """
        p, B = self.p, self.B
        # `.float()` after, not `dtype=` inside: coremltools' `arange` over a shape read ignores the
        # dtype and yields int32, which then meets a float weight in the outer product below.
        b = torch.arange(n_blocks).float()                                      # 0 .. nb-1
        rel = _outer(b, torch.ones_like(self.key_block_offsets)) + \
            _outer(torch.ones_like(b), self.key_block_offsets)                 # b - p + k
        index = torch.relu(rel).to(torch.int32).reshape(n_blocks * (p + 1))
        absent = torch.clamp(-rel, 0.0, 1.0)                                   # 1 where b - p + k < 0
        mask = _outer(absent.reshape(n_blocks * (p + 1)), torch.ones(B)) * NEG
        return index, mask.reshape(n_blocks, 1, 1, (p + 1) * B)

    def _rope(self, x: torch.Tensor, T, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """The reference's `apply_rope` on `[1, T, H, Dh]`: channel pairs (2i, 2i+1) rotated as one
        complex number. Not `rotate_half` -- the halves are INTERLEAVED here, as in Mimi.

        `x * cos + rotate_pairs(x) * sin`, with the pairs' members sliced `1:2`/`0:1` rather than
        indexed: an index `[..., 0]` lowers to a rank-4 view with its unit axis in the wrong place and
        the multiply against `cos` fails to broadcast. `pocket_tts_export._rotate_pairs` is this."""
        pairs = x.reshape(1, T, self.H * self.Dh // 2, 2)
        rotated = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(1, T, self.H, self.Dh)
        return x * cos + rotated * sin

    def _attention(self, attn, qkv, x, T, n_blocks, cos, sin, index, mask):
        B, p, H, Dh, C = self.B, self.p, self.H, self.Dh, self.C
        q = self._rope(qkv[0](x).reshape(1, T, H, Dh), T, cos, sin)
        k = self._rope(qkv[1](x).reshape(1, T, H, Dh), T, cos, sin)
        v = qkv[2](x)
        q = q.reshape(n_blocks, B, H, Dh).permute(0, 2, 1, 3)                            # [nb,H,B,Dh]
        k = k.reshape(n_blocks, B * C)[index].reshape(n_blocks, (p + 1) * B, H, Dh).permute(0, 2, 3, 1)
        v = v.reshape(n_blocks, B * C)[index].reshape(n_blocks, (p + 1) * B, H, Dh).permute(0, 2, 1, 3)
        scores = (q @ k) * (1.0 / math.sqrt(Dh)) + self.window_mask + mask
        out = torch.softmax(scores, dim=-1) @ v                                          # [nb,H,B,Dh]
        return attn.out_proj(out.permute(0, 2, 1, 3).reshape(1, T, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[1, T, C_in] -> [1, T, C_out]`.

        **The leading 1 is load-bearing.** Every length here is read off the SEQUENCE axis, and that
        axis must not be axis 0: `value_facts.gather_shape_value` reads `x.shape[0]` of a rank>=2
        tensor as a BATCH size whenever it resolves to the root axis, and answers 1. In a flat
        `[T, C]` layout the first stack's length is exactly that, so its positions were
        `arange(1)` -- one position's rotation broadcast over every frame, a graph that ran and was
        0.1% off end to end and 65% off at the rotated Q. GigaAM's rotary crop is the same trap.

        Every reshape that a later shape read depends on is written with its length (`T`,
        `n_blocks`) rather than `-1`, which reaches the topology as a literal ([Retro-047]).
        """
        x = self.input_proj(x)
        T = x.shape[1]
        n_blocks = T // self.B
        # Absolute positions from 0, which is what the reference's whole-sequence pass uses and what
        # its streaming pass reaches by carrying an offset.
        pos = torch.arange(T).float()
        angle = _outer(pos, self.inv_freq).reshape(1, T, 1, self.Dh)
        cos, sin = torch.cos(angle), torch.sin(angle)
        index, mask = self._before_zero(n_blocks)
        for layer, qkv in zip(self.layers, self.qkv):
            x = x + layer.layer_scale_1(self._attention(
                layer.self_attn, qkv, layer.norm1(x), T, n_blocks, cos, sin, index, mask))
            x = x + layer.layer_scale_2(layer.ffn(layer.norm2(x)))
        return self.output_proj(x)


def _patch_decode(x: torch.Tensor, patch: int) -> torch.Tensor:
    """`MossAudioTokenizerPatchedPretransform.decode` in `[1, T, C]` layout: channel `d * patch + j`
    of row t becomes channel d of row `t * patch + j`. The new length is written as `T * patch`, not
    `-1`, because the next stack reads it."""
    T, C = x.shape[1], x.shape[2]
    return x.reshape(1, T, C // patch, patch).permute(0, 1, 3, 2).reshape(1, T * patch, C // patch)


class MossCodecDecoder(nn.Module):
    """`codes [1, n_frames, n_q] -> [1, channels * samples]`, interleaved, `n_frames % m == 0`."""

    def __init__(self, model, frames_per_block: int):
        super().__init__()
        table, self.output_proj = fold_quantizer(model.quantizer)
        self.register_buffer("table", table)
        self.size = int(model.quantizer.codebook_size) + 1
        n_q = len(model.quantizer.quantizers)
        # FLOAT, and the sum cast back: a weight is written to the GGUF as F32 whatever its torch dtype,
        # so an int offsets buffer meets the int codes in an ADD that comes out float, and
        # `ggml_get_rows` asserts an I32 index. Exact below 2^24; the largest id here is 32799.
        self.register_buffer("offsets", torch.arange(n_q, dtype=torch.float32) * self.size)
        self.plan = []
        stages, rate = [], 1
        for module in model.decoder:
            if type(module).__name__.endswith("PatchedPretransform"):
                self.plan.append(("patch", module.patch_size))
                rate *= module.patch_size
            else:
                self.plan.append(("stage", len(stages)))
                stages.append(BlockedStage(module, frames_per_block, rate))
        self.stages = nn.ModuleList(stages)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        n, n_q = codes.shape[1], codes.shape[2]
        index = (codes.float() + self.offsets).to(torch.int32).reshape(n * n_q)
        rows = self.table[index]                                             # [n * n_q, rvq_dim]
        x = rows.reshape(1, n, n_q, rows.shape[-1]).sum(dim=2)              # [1, n, rvq_dim]
        x = self.output_proj(x)
        for kind, arg in self.plan:
            x = _patch_decode(x, arg) if kind == "patch" else self.stages[arg](x)
        return x.reshape(1, -1)


def is_moss_audio_tokenizer(cfg: Optional[dict]) -> bool:
    return cfg is not None and cfg.get("model_type") == "moss-audio-tokenizer"


def load(model_dir: str):
    """The reference model, at F32 and on the SDPA path.

    It is remote code shipped in the checkpoint (`modeling_moss_audio_tokenizer.py`, Apache-2.0), so
    `trust_remote_code` loads it from the directory itself -- nothing is fetched. `compute_dtype` is
    forced to fp32 because the checkpoint declares bf16, which on CPU turns on `torch.autocast` inside
    `decode`: the reference the export is graded against would otherwise be a bf16 one. The encoder
    is dropped at once: this family exports the decode half only, and it is half the weights.
    """
    import transformers

    model = transformers.AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    model.set_attention_implementation("sdpa")
    model.set_compute_dtype("fp32")
    model.encoder = nn.ModuleList()
    return model


def geometry(model) -> dict:
    config = model.config
    n_q = len(model.quantizer.quantizers)
    return dict(n_codebooks=n_q, codebook_size=int(model.quantizer.codebook_size),
                sample_rate=int(config.sampling_rate),
                # Samples PER CHANNEL per frame: 3840 at 48 kHz is the 12.5 Hz the codes run at. The
                # interleaved stream the graph emits carries `channels` times as many floats.
                hop_length=int(config.downsample_rate), vq_strides=[1] * n_q,
                channels=int(config.number_channels) if config.enable_channel_interleave else 1)
