"""Two lowerings OmniVoice's unmasking step needed (loom.cpp ADR-131).

* `ExportPhase.dense_attention`: a phase's SDPA blocks fused into UNCACHED, DENSE `ATTENTION` nodes, for a
  bidirectional stack re-run whole every call -- no cache to size, no window, and the mask free to be
  computed in the graph (a cached node needs it as a declared input; this one does not).
* `reduce_max` over the static LAST axis: one `POOL_1D` whose kernel and stride are that axis, then a
  `RESHAPE` to the input's other extents. (A per-row maximum; every other partial reduction is still
  refused.)
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

from loom_exporter.multi_phase_export import ExportPhase
from loom_exporter.phase_conversion import convert_phase

N_HEAD, HEAD_DIM, SEQ = 2, 4, 12


class _BlockMaskedAttention(nn.Module):
    """Self-attention whose mask is built IN the graph from a per-position segment vector -- the outer
    products OmniVoice's step uses -- so the fused node's mask is not a declared input."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.qkv = nn.Linear(N_HEAD * HEAD_DIM, 3 * N_HEAD * HEAD_DIM, bias=False)

    def forward(self, x, segment):
        b, t, _ = x.shape
        col = segment.transpose(0, 1)
        same = col @ segment + (1.0 - col) @ (1.0 - segment)
        mask = ((same - 1.0) * 1e9).unsqueeze(0).unsqueeze(0)
        q, k, v = self.qkv(x).view(b, t, 3, N_HEAD, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(q * (1.0 / math.sqrt(HEAD_DIM)), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        return ctx.transpose(1, 2).reshape(b, t, N_HEAD * HEAD_DIM)


def _attention_phase(**kwargs):
    import coremltools as ct

    seq = ct.RangeDim(2, 64)
    segment = torch.cat([torch.ones(7), torch.zeros(SEQ - 7)]).view(1, -1)
    return ExportPhase(
        name="attend",
        wrapper=_BlockMaskedAttention().eval(),
        dummy_inputs=(torch.randn(1, SEQ, N_HEAD * HEAD_DIM), segment),
        mil_inputs=[ct.TensorType(name="x", shape=(1, seq, N_HEAD * HEAD_DIM), dtype=np.float32),
                    ct.TensorType(name="segment", shape=(1, seq), dtype=np.float32)],
        **kwargs,
    )


def test_dense_attention_is_an_uncached_unwindowed_node_over_an_in_graph_mask():
    result = convert_phase(_attention_phase(dense_attention=True))
    (node,) = [n for n in result.topologies["attend"]["nodes"] if n["op"] == "ATTENTION"]
    assert node["attrs"] == {"layer": 0, "scale": pytest.approx(0.5), "kv_cache": False}
    # Nothing for the file's KvCache to size: the block stores nothing between calls.
    assert result.geometry.get("attention", []) == []


def test_dense_attention_refuses_a_cache_or_a_window():
    with pytest.raises(ValueError, match="dense_attention with a KV cache or a window"):
        convert_phase(_attention_phase(dense_attention=True, kv_cache_size=64))
    with pytest.raises(ValueError, match="dense_attention with a KV cache or a window"):
        convert_phase(_attention_phase(dense_attention=True, attention_window=4))


class _RowMax(nn.Module):
    def __init__(self, keepdim):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x):
        # A permute ahead of the max, OmniVoice's order: the pool's input is then a non-contiguous view,
        # which the ENGINE's op_pool_1d makes contiguous (loom.cpp test_pool_1d_permuted_input).
        return torch.amax(x.permute(1, 0, 2), dim=-1, keepdim=self.keepdim) * 2.0


def _max_phase(keepdim):
    import coremltools as ct

    rows = ct.RangeDim(1, 64)
    return ExportPhase(
        name="rowmax",
        wrapper=_RowMax(keepdim).eval(),
        dummy_inputs=(torch.randn(5, 3, 11),),
        mil_inputs=[ct.TensorType(name="x", shape=(rows, 3, 11), dtype=np.float32)],
    )


@pytest.mark.parametrize("keepdim", [False, True])
def test_a_last_axis_max_is_a_pool_spanning_that_axis(keepdim):
    nodes = convert_phase(_max_phase(keepdim)).topologies["rowmax"]["nodes"]
    (pool,) = [n for n in nodes if n["op"] == "POOL_1D"]
    assert pool["attrs"] == {"op": "max", "k0": 11, "s0": 11, "p0": 0}
    (reshape,) = [n for n in nodes if n["op"] == "RESHAPE" and n["inputs"] == pool["outputs"]]
    # ne order: the rows (dynamic) then the static 3 -- the INPUT's extents, not a fresh symbol.
    shape = reshape["attrs"]["shape"]
    assert shape[-1] == 3 and isinstance(shape[-2], str) and (shape[0] == 1) == keepdim


class _MidMax(nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=1)


def test_a_max_over_another_axis_is_still_refused():
    import coremltools as ct

    phase = ExportPhase(name="midmax", wrapper=_MidMax().eval(), dummy_inputs=(torch.randn(4, 3, 11),),
                        mil_inputs=[ct.TensorType(name="x", shape=(ct.RangeDim(1, 64), 3, 11),
                                                  dtype=np.float32)])
    with pytest.raises(NotImplementedError, match="does not reduce every axis"):
        convert_phase(phase)
