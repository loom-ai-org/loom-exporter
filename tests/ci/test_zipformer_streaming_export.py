"""Zipformer streaming's pieces (`zipformer_streaming_export`), each against the reference arithmetic it
replaces, plus the strided slice its export found the lowering ignoring. None of it needs the checkpoint or
the `pkufool/zipformer` clone: the whole model is graded by loom.cpp's gate test
`test_e2e_zipformer_streaming_mil_export` against upstream's released jit file.

Measured on `pkufool/zipformer-small-streaming` (2026-11-08): the wrapper equals upstream's
`StreamingEncoderWrapper` to 8.9e-6 on every chunk, and the exported GGUF decodes the 73 LibriSpeech-dummy
clips token for token as upstream's jit file does.
"""
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import pytest
import torch
import torch.nn as nn

from loom_exporter.exporter import LoomGGUFExporter
from loom_exporter.zipformer_export import _KaldiFbank, _is_zipformer
from loom_exporter.zipformer_streaming_export import (
    CHUNK_SIZE, FEATURE_TAIL, LEFT_CONTEXT, LOG_EPS, PAD_LENGTH, TAIL, _Features, _StreamingAttentionWeights,
    _is_zipformer_streaming, key_positions, read_streaming_architecture,
)


def _reference_streaming_attention(x, pos_emb, cached_key, left, key_padding_mask, heads, query_dim, pos_dim,
                                   linear_pos):
    """`RelPositionMultiheadAttentionWeights.streaming_forward`'s traced branch: the `gather`."""
    seq_len, batch = x.shape[0], x.shape[1]
    width = query_dim * heads
    q = x[..., :width].reshape(seq_len, batch, heads, query_dim).permute(2, 1, 0, 3)
    k = torch.cat([cached_key, x[..., width : 2 * width]], dim=0)
    k_len = k.shape[0]
    k = k.reshape(k_len, batch, heads, query_dim).permute(2, 1, 3, 0)
    p = x[..., 2 * width :].reshape(seq_len, batch, heads, pos_dim).permute(2, 1, 0, 3)
    scores = torch.matmul(q, k)
    pos = linear_pos(pos_emb).reshape(-1, 2 * seq_len - 1 + left, heads, pos_dim).permute(2, 0, 3, 1)
    pos_scores = torch.matmul(p, pos)
    n = pos_scores.shape[-1]
    rows = torch.arange(seq_len - 1, -1, -1).repeat(batch * heads).unsqueeze(-1)
    gathered = torch.gather(pos_scores.reshape(-1, n), 1, rows + torch.arange(k_len))
    scores = scores + gathered.reshape(heads, batch, seq_len, k_len)
    return scores.masked_fill(key_padding_mask.unsqueeze(1), -1000).softmax(dim=-1)


@pytest.mark.parametrize("n, left", [(1, 4), (2, 8), (4, 16), (16, 64), (32, 128)])
def test_the_relative_shift_with_a_left_context_is_the_references_gather(n, left):
    torch.manual_seed(n + left)
    heads, query_dim, pos_dim, dim = 4, 8, 3, 16
    in_proj = nn.Linear(dim, (2 * query_dim + pos_dim) * heads)
    linear_pos = nn.Linear(12, heads * pos_dim, bias=False)
    x, pos_emb = torch.randn(n, 1, dim), torch.randn(1, 2 * n - 1 + left, 12)
    cached_key = torch.randn(left, 1, heads * query_dim)
    mask = torch.rand(1, left + n) < 0.3
    mask[:, left:] = False
    weights = SimpleNamespace(in_proj=in_proj, linear_pos=linear_pos, num_heads=heads,
                              query_head_dim=query_dim, pos_head_dim=pos_dim)
    with torch.no_grad():
        want = _reference_streaming_attention(in_proj(x), pos_emb, cached_key, left, mask, heads, query_dim,
                                              pos_dim, linear_pos)
        got, new_key = _StreamingAttentionWeights(weights).streaming_forward(x, pos_emb, cached_key, left, mask)
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)
    torch.testing.assert_close(new_key, torch.cat([cached_key, in_proj(x)[..., heads * query_dim :
                                                                         2 * heads * query_dim]])[-left:])


@pytest.mark.parametrize("ds", [1, 2, 4, 8])
@pytest.mark.parametrize("processed", [0, 1, 32, 64, 127, 128, 129, 4096])
def test_key_positions_are_the_references_mask(ds, processed):
    """`StreamingEncoderWrapper`'s `(processed_lens <= arange(L)).flip(1)`, the chunk's own unmasked keys
    after it, and `Zipformer2.streaming_forward`'s `[..., ::ds]`."""
    lens = torch.tensor([processed])
    cache = (lens.unsqueeze(1) <= torch.arange(LEFT_CONTEXT).expand(1, LEFT_CONTEXT)).flip(1)
    want = torch.cat([cache, torch.zeros(1, CHUNK_SIZE, dtype=torch.bool)], dim=1)[..., ::ds]
    got = (key_positions(ds) >= lens.to(torch.float32)).unsqueeze(0)
    assert torch.equal(got, want)


def test_the_features_end_in_the_padding_upstream_reads():
    x = torch.randn(1, 8000, generator=torch.Generator().manual_seed(0)) * 0.1
    got = _Features()(x)
    head = _KaldiFbank()(x)[0]
    assert got.shape == (head.shape[0] + FEATURE_TAIL, 80)
    assert torch.equal(got[: head.shape[0]], head)
    assert torch.all(got[head.shape[0] :] == torch.tensor(LOG_EPS))
    # The last chunk starts before T + PAD_LENGTH and reads TAIL rows; all of them must exist.
    assert FEATURE_TAIL >= PAD_LENGTH + TAIL - 1


def _streaming_state():
    """A two-stack chunk-causal checkpoint's tensors, under the names `_read_architecture` reads."""
    state = {"joiner.output_linear.weight": torch.zeros(100, 16), "decoder.embedding.weight": torch.zeros(100, 32),
             "decoder.conv.weight": torch.zeros(32, 8, 2), "encoder.encoders.1.downsample.bias": torch.zeros(2)}
    for i, (root, d, k) in enumerate((("encoder.encoders.0.", 4, 31), ("encoder.encoders.1.encoder.", 8, 15))):
        prefix = f"{root}layers.0."
        state[prefix + "bypass_scale"] = torch.zeros(d)
        state[prefix + "conv_module1.depthwise_conv.causal_conv.weight"] = torch.zeros(d, 1, (k + 1) // 2)
        state[prefix + "conv_module1.depthwise_conv.chunkwise_conv.weight"] = torch.zeros(d, 1, k)
        state[prefix + "self_attn_weights.linear_pos.weight"] = torch.zeros(16, 48)
        state[prefix + "feed_forward2.out_proj.weight"] = torch.zeros(d, 24)
    return state


def test_the_architecture_reads_the_chunkwise_kernel_and_fixes_the_latency():
    got = read_streaming_architecture(_streaming_state())
    assert got["cnn_module_kernel"] == "31,15" and got["encoder_dim"] == "4,8"
    assert got["causal"] is True
    assert (got["chunk_size"], got["left_context_frames"]) == (str(CHUNK_SIZE), str(LEFT_CONTEXT))


def test_each_leaf_claims_only_its_own_checkpoint(tmp_path):
    def checkpoint(path, state):
        (path / "data").mkdir(parents=True)
        (path / "data" / "tokens.txt").write_text("<blk> 0\n", encoding="utf-8")
        torch.save({"model": state}, path / "model.pt")
        return path

    streaming = checkpoint(tmp_path / "streaming", _streaming_state())
    plain = checkpoint(tmp_path / "plain", {
        k.replace("depthwise_conv.chunkwise_conv.weight", "depthwise_conv.weight"): v
        for k, v in _streaming_state().items() if "causal_conv" not in k})
    assert _is_zipformer_streaming(streaming) and not _is_zipformer(streaming)
    assert _is_zipformer(plain) and not _is_zipformer_streaming(plain)
    assert not _is_zipformer_streaming(tmp_path / "absent")


class _EveryOther(nn.Module):
    def forward(self, x):
        return x[..., ::2] + 1.0


class _Prefix(nn.Module):
    def forward(self, x):
        return x[..., 1:5] + 1.0


def _topology(module, example):
    program = ct.convert(torch.jit.trace(module.eval(), example),
                         inputs=[ct.TensorType(name="x", shape=example[0].shape, dtype=np.float32)],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    return LoomGGUFExporter(program).generate_graph_topology(program.functions["main"], "main")["nodes"]


def test_a_strided_slice_is_refused_not_lowered_as_a_view():
    """The rule used to drop `stride` and declare the unstrided extent: `mask[..., ::2]` came out [160]
    where torch has [80]."""
    with pytest.raises(NotImplementedError, match="strided"):
        _topology(_EveryOther(), (torch.zeros(1, 8),))


def test_a_unit_stride_slice_is_still_a_view():
    view = next(n for n in _topology(_Prefix(), (torch.zeros(1, 8),)) if n["op"] == "VIEW")
    assert int(view["attrs"]["shape"][0]) == 4
