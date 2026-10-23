"""A declared attention window (P4.41, loom.cpp ADR-085): `ExportPhase.attention_window` fuses a phase's SDPA
blocks into UNCACHED `ATTENTION` nodes that carry the window, so the engine computes only the causal band.

The window is a promise about the mask that only the wrapper building it can make, so what is pinned here is
the plumbing and its two refusals; the mask itself is checked against the declaration by the family that makes
it (Pocket-TTS: `test_pocket_tts_export.test_mimis_attention_window_is_the_masks_own_band`).
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

from loom_exporter.multi_phase_export import ExportPhase
from loom_exporter.passes import apply_loom_mil_passes
from loom_exporter.phase_conversion import convert_phase

N_HEAD, HEAD_DIM, SEQ = 2, 4, 12


class _SelfAttention(nn.Module):
    """HF's spelling, the window `fuse_loom_attention` matches; `scale_scores` moves the scale onto the
    scores, which the pattern does not take."""

    def __init__(self, scale_scores=False):
        super().__init__()
        torch.manual_seed(0)
        self.qkv = nn.Linear(N_HEAD * HEAD_DIM, 3 * N_HEAD * HEAD_DIM, bias=False)
        self.scale_scores = scale_scores

    def forward(self, x, mask):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, N_HEAD, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)
        s = 1.0 / math.sqrt(HEAD_DIM)
        scores = (torch.matmul(q, k.transpose(-1, -2)) * s if self.scale_scores
                  else torch.matmul(q * s, k.transpose(-1, -2))) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        return ctx.transpose(1, 2).reshape(b, t, N_HEAD * HEAD_DIM)


def _phase(window=None, kv_cache_size=None, scale_scores=False):
    import coremltools as ct

    seq = ct.RangeDim(1, 64)
    return ExportPhase(
        name="attend",
        wrapper=_SelfAttention(scale_scores).eval(),
        dummy_inputs=(torch.randn(1, SEQ, N_HEAD * HEAD_DIM), torch.zeros(1, 1, SEQ, SEQ)),
        mil_inputs=[ct.TensorType(name="x", shape=(1, seq, N_HEAD * HEAD_DIM), dtype=np.float32),
                    ct.TensorType(name="mask", shape=(1, 1, seq, seq), dtype=np.float32)],
        attention_window=window,
        kv_cache_size=kv_cache_size,
    )


def _attention(result):
    return [n for n in result.topologies["attend"]["nodes"] if n["op"] == "ATTENTION"]


def test_a_declared_window_is_an_uncached_attention_carrying_it():
    result = convert_phase(_phase(window=5))
    (node,) = _attention(result)
    assert node["attrs"] == {"layer": 0, "scale": pytest.approx(0.5), "kv_cache": False, "window": 5}
    # Nothing to size a cache from: a windowed block stores nothing between calls.
    assert result.geometry.get("attention", []) == []


def test_without_a_window_a_fused_block_is_cached_as_before():
    """The two new inputs are only set when windowed, so every cached export's op and attrs are unchanged."""
    from coremltools.converters.mil.mil import Builder as mb, types

    specs = [mb.TensorSpec(shape=(1, N_HEAD, SEQ, HEAD_DIM), dtype=types.fp32) for _ in range(3)]
    specs.append(mb.TensorSpec(shape=(1, 1, SEQ, SEQ), dtype=types.fp32))

    @mb.program(input_specs=specs)
    def prog(q, k, v, mask):
        probs = mb.softmax(x=mb.add(x=mb.matmul(x=mb.mul(x=q, y=np.float32(0.5)), y=k, transpose_y=True),
                                    y=mask), axis=-1)
        ctx = mb.transpose(x=mb.matmul(x=probs, y=v), perm=[0, 2, 1, 3])
        return mb.reshape(x=ctx, shape=[1, SEQ, N_HEAD * HEAD_DIM])

    apply_loom_mil_passes(prog, fuse_attention=True)
    (op,) = [o for o in prog.functions["main"].operations if o.op_type == "loom_fused_attention"]
    assert bool(op.inputs["kv_cache"].val) is True and int(op.inputs["window"].val) == 0


def test_a_window_that_fused_nothing_is_refused():
    """The dense form would run instead: right output, the whole speed-up gone without a word."""
    with pytest.raises(ValueError, match="fused none of its attention blocks"):
        convert_phase(_phase(window=5, scale_scores=True))


def test_a_window_and_a_cache_are_refused_together():
    with pytest.raises(ValueError, match="both a KV cache and an attention window"):
        convert_phase(_phase(window=5, kv_cache_size=64))
