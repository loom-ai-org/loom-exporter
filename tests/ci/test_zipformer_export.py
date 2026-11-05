"""Zipformer's pieces (`zipformer_export`), each against the arithmetic it replaces, plus the two lowering
bugs its export found. None of it needs the checkpoint or the `pkufool/zipformer` clone: the whole model is
graded by loom.cpp's gate test `test_e2e_zipformer_mil_export` against the reference's own token ids.

Measured on `pkufool/zipformer-small` (2026-11-05): the wrapper equals `AsrModel.forward_encoder` to 1.2e-5,
the exported encoder equals the wrapper to 1e-5 at 15 clip lengths from 1500 samples to 16.7 s, and the GGUF
transcribes the repo's English and Chinese clips token for token on the released rc20 wheel.
"""
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from loom_exporter.exporter import LoomGGUFExporter
from loom_exporter.zipformer_export import (
    _AttentionWeights, _Downsample, _KaldiFbank, _Positions, _Stack, _convert_channels, _is_zipformer,
    _read_architecture, _read_tokens,
)
from loom_exporter.zipformer_tokenizer_export import read_zipformer_layout


def _topology_nodes(module, example, name="x", shape=None):
    program = ct.convert(torch.jit.trace(module.eval(), example),
                         inputs=[ct.TensorType(name=name, shape=shape or example[0].shape, dtype=np.float32)],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    return LoomGGUFExporter(program).generate_graph_topology(program.functions["main"], "main")["nodes"]


@pytest.mark.parametrize("n", [1500, 4000, 4001, 8159, 8160, 8239, 8240, 16077])
def test_the_front_end_is_kaldis_with_snip_edges_off(n):
    x = torch.randn(1, n, generator=torch.Generator().manual_seed(n)) * 0.1
    want = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, sample_frequency=16000, dither=0,
                                             snip_edges=False, high_freq=-400)
    got = _KaldiFbank()(x)
    assert got.shape == (1, want.shape[0], 80)
    torch.testing.assert_close(got[0], want, rtol=0, atol=2e-3)


def _reference_positions(n: int, embed_dim: int, length_factor: float) -> torch.Tensor:
    """`CompactRelPositionalEncoding.extend_pe` and its slice, for a table built long and read short."""
    T = 2 * n
    x = torch.arange(-(T - 1), T).to(torch.float32).unsqueeze(1)
    freqs = 1 + torch.arange(embed_dim // 2)
    compression = embed_dim ** 0.5
    compressed = compression * x.sign() * ((x.abs() + compression).log() - np.log(compression))
    scale = length_factor * embed_dim / (2.0 * np.pi)
    atan = (compressed / scale).atan()
    pe = torch.zeros(x.shape[0], embed_dim)
    pe[:, 0::2] = (atan * freqs).cos()
    pe[:, 1::2] = (atan * freqs).sin()
    pe[:, -1] = 1.0
    return pe[pe.size(0) // 2 - n + 1 : pe.size(0) // 2 + n].unsqueeze(0)


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_positions_equal_the_cached_table(n):
    got = _Positions(48, 1.0)(torch.zeros(n, 1, 4))
    assert got.shape == (1, 2 * n - 1, 48)
    torch.testing.assert_close(got, _reference_positions(n, 48, 1.0), rtol=0, atol=1e-6)


def _reference_downsample(src, bias):
    """`SimpleDownsample.forward`, transcribed: repeat the last frame to a multiple of `ds`."""
    ds = bias.shape[0]
    seq_len, batch, channels = src.shape
    d_seq_len = (seq_len + ds - 1) // ds
    pad = d_seq_len * ds - seq_len
    src = torch.cat((src, src[-1:].expand(pad, batch, channels)), dim=0)
    return (src.reshape(d_seq_len, ds, batch, channels) * bias.softmax(dim=0)[:, None, None]).sum(dim=1)


@pytest.mark.parametrize("ds", [2, 4, 8])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 16, 31])
def test_downsample_equals_the_reference_at_every_remainder(ds, n):
    bias = torch.randn(ds, generator=torch.Generator().manual_seed(ds))
    src = torch.randn(n, 1, 6, generator=torch.Generator().manual_seed(n))
    got = _Downsample(SimpleNamespace(downsample=ds, bias=bias), 6)(src)
    torch.testing.assert_close(got, _reference_downsample(src, bias), rtol=0, atol=2e-6)


@pytest.mark.parametrize("n", [1, 5, 8, 9, 17])
def test_a_stack_returns_every_frame_it_was_given(n):
    """`SimpleUpsample` and the slice back to `T`, on a stack that passes frames through: a frame's value
    has to come back at each of the `ds` positions it covers, and the bypass has to see them."""
    class Stack(nn.Module):
        downsample_factor = 4

        def __init__(self):
            super().__init__()
            self.downsample = SimpleNamespace(downsample=4, bias=torch.zeros(4))
            self.encoder = nn.Identity()
            self.out_combiner = lambda orig, new: new

    src = torch.randn(n, 1, 6, generator=torch.Generator().manual_seed(n))
    low = _reference_downsample(src, torch.zeros(4))
    want = low.unsqueeze(1).expand(low.shape[0], 4, 1, 6).reshape(low.shape[0] * 4, 1, 6)[:n]
    torch.testing.assert_close(_Stack(Stack(), 6)(src), want, rtol=0, atol=2e-6)


def _reference_attention(q_k_p, pos_emb, heads, query_dim, pos_dim, linear_pos):
    """The reference's traced path, `gather` included."""
    seq_len, batch = q_k_p.shape[:2]
    width = query_dim * heads
    q = q_k_p[..., :width].reshape(seq_len, batch, heads, query_dim).permute(2, 1, 0, 3)
    k = q_k_p[..., width : 2 * width].reshape(seq_len, batch, heads, query_dim).permute(2, 1, 3, 0)
    p = q_k_p[..., 2 * width :].reshape(seq_len, batch, heads, pos_dim).permute(2, 1, 0, 3)
    scores = torch.matmul(q, k)
    pos = linear_pos(pos_emb).reshape(-1, 2 * seq_len - 1, heads, pos_dim).permute(2, 0, 3, 1)
    pos_scores = torch.matmul(p, pos)
    n = pos_scores.shape[-1]
    rows = torch.arange(seq_len - 1, -1, -1).repeat(batch * heads).unsqueeze(-1)
    gathered = torch.gather(pos_scores.reshape(-1, n), 1, rows + torch.arange(seq_len))
    return (scores + gathered.reshape(heads, batch, seq_len, seq_len)).softmax(dim=-1)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 21])
def test_the_relative_shift_is_the_references_gather(n):
    torch.manual_seed(n)
    heads, query_dim, pos_dim, dim = 4, 8, 3, 16
    in_proj = nn.Linear(dim, (2 * query_dim + pos_dim) * heads)
    linear_pos = nn.Linear(12, heads * pos_dim, bias=False)
    x, pos_emb = torch.randn(n, 1, dim), torch.randn(1, 2 * n - 1, 12)
    weights = SimpleNamespace(in_proj=in_proj, linear_pos=linear_pos, num_heads=heads,
                              query_head_dim=query_dim, pos_head_dim=pos_dim)
    with torch.no_grad():
        want = _reference_attention(in_proj(x), pos_emb, heads, query_dim, pos_dim, linear_pos)
        got = _AttentionWeights(weights)(x, pos_emb)
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)


def test_channels_pad_with_zeros_and_truncate():
    x = torch.randn(3, 1, 4)
    assert torch.equal(_convert_channels(x, 2), x[..., :2])
    assert torch.equal(_convert_channels(x, 6), torch.cat((x, torch.zeros(3, 1, 2)), dim=-1))


def _state(dims=(4, 8), layers=(2, 1), kernels=(31, 15), factors=(1, 2), heads=(4, 8), feedforward=(16, 24)):
    """The tensors `_read_architecture` reads, under the names a checkpoint's stacks use."""
    state = {"joiner.output_linear.weight": torch.zeros(100, 16), "decoder.embedding.weight": torch.zeros(100, 32),
             "decoder.conv.weight": torch.zeros(32, 8, 2)}
    for i, (d, ls, k, f, h, ff) in enumerate(zip(dims, layers, kernels, factors, heads, feedforward)):
        root = f"encoder.encoders.{i}." + ("encoder." if f > 1 else "")
        if f > 1:
            state[f"encoder.encoders.{i}.downsample.bias"] = torch.zeros(f)
        for layer in range(ls):
            prefix = f"{root}layers.{layer}."
            state[prefix + "bypass_scale"] = torch.zeros(d)
            state[prefix + "conv_module1.depthwise_conv.weight"] = torch.zeros(d, 1, k)
            state[prefix + "self_attn_weights.linear_pos.weight"] = torch.zeros(h * 4, 48)
            state[prefix + "feed_forward2.out_proj.weight"] = torch.zeros(d, ff)
    return state


def test_the_architecture_is_read_off_the_weights():
    got = _read_architecture(_state())
    assert got["encoder_dim"] == "4,8" and got["num_encoder_layers"] == "2,1"
    assert got["cnn_module_kernel"] == "31,15" and got["downsampling_factor"] == "1,2"
    assert got["num_heads"] == "4,8" and got["feedforward_dim"] == "16,24"
    assert got["vocab_size"] == 100 and got["decoder_dim"] == 32 and got["joiner_dim"] == 16
    assert got["context_size"] == 2 and got["pos_dim"] == 48 and got["causal"] is False


def test_a_streaming_checkpoint_is_refused():
    state = {k.replace("depthwise_conv.weight", "depthwise_conv.chunkwise_conv.weight"): v
             for k, v in _state().items()}
    with pytest.raises(ValueError, match="streaming"):
        _read_architecture(state)


def _checkpoint(tmp_path, pieces, scores=None):
    data = tmp_path / "data"
    data.mkdir()
    (data / "tokens.txt").write_text("".join(f"{p} {i}\n" for i, p in enumerate(pieces)), encoding="utf-8")
    scores = scores if scores is not None else [0.0] * len(pieces)
    (data / "zh-en-5.vocab").write_text("".join(f"{p}\t{s}\n" for p, s in zip(pieces, scores)), encoding="utf-8")
    return tmp_path


PIECES = ["<blk>", "<sos>", "<eos>", "<pad>", "<unk>", "<0x00>", "▁his", "e", "称"]


def test_the_vocabulary_keeps_ids_scores_and_roles(tmp_path):
    layout = read_zipformer_layout(_checkpoint(tmp_path, PIECES, [0, 0, 0, 0, 0, 0, -5.5, -5.25, -9.5]))
    assert layout.pieces == PIECES
    assert layout.scores[6:] == [-5.5, -5.25, -9.5]
    assert (layout.unk_id, layout.bos_id, layout.eos_id, layout.pad_id) == (4, 1, 2, 3)
    assert layout.special_ids == {0, 1, 2, 3}
    assert layout.model_type == "bpe"


def test_a_vocab_that_disagrees_with_tokens_is_refused(tmp_path):
    model_dir = _checkpoint(tmp_path, PIECES)
    (model_dir / "data" / "zh-en-5.vocab").write_text("<blk>\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different pieces"):
        read_zipformer_layout(model_dir)


def test_tokens_must_number_their_rows(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "tokens.txt").write_text("<blk> 0\ne 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="number its rows"):
        _read_tokens(tmp_path)


def test_a_directory_without_the_layout_is_not_claimed(tmp_path):
    assert _is_zipformer(tmp_path) is False
    assert _is_zipformer(tmp_path / "absent") is False


def test_a_2d_convs_attrs_are_in_ggml_axis_order():
    """ggml's axis 0 is the width, torch's LAST spatial axis. Same on both axes, nothing shows."""
    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=(1, 2), padding=(0, 1), dilation=(1, 1))

        def forward(self, x):
            return self.conv(x)

    node = next(n for n in _topology_nodes(Conv(), (torch.zeros(1, 1, 16, 20),)) if n["op"] == "CONV_2D")
    assert (node["attrs"]["s0"], node["attrs"]["s1"]) == (2, 1)      # freq, time
    assert (node["attrs"]["p0"], node["attrs"]["p1"]) == (1, 0)


class _Log1p(nn.Module):
    def forward(self, x):
        return torch.log1p(x)


def test_log1p_adds_the_one_in_the_graph():
    nodes = _topology_nodes(_Log1p(), (torch.ones(1, 4),))
    log = next(n for n in nodes if n["op"] == "LOG")
    producer = next(n for n in nodes if log["inputs"][0] in n["outputs"])
    assert producer["op"] == "ADD"


def test_a_log_with_a_real_epsilon_is_refused():
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
    def main(x):
        return mb.log(x=x, epsilon=1.0, name="out")

    with pytest.raises(NotImplementedError, match="epsilon"):
        LoomGGUFExporter(main).generate_graph_topology(main.functions["main"], "main")


class _LogAddExp(nn.Module):
    def forward(self, x):
        return torch.logaddexp(torch.zeros_like(x), x - 4.0)


def test_logaddexp_lowers_to_its_formula():
    """The composition, in torch, is what the converter emits; its accuracy is checked here because the
    engine is not in this test's reach."""
    ops = {n["op"] for n in _topology_nodes(_LogAddExp(), (torch.ones(1, 4),))}
    assert {"MAXIMUM", "ABS", "EXP", "LOG"} <= ops
    x = torch.linspace(-30.0, 30.0, 601)
    y = x - 4.0
    composed = torch.maximum(torch.zeros_like(x), y) + torch.log(1.0 + torch.exp(-(torch.zeros_like(x) - y).abs()))
    torch.testing.assert_close(composed, torch.logaddexp(torch.zeros_like(x), y), rtol=0, atol=1e-6)
