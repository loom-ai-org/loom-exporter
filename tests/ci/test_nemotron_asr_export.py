"""Nemotron 3.5 ASR's encoder pieces (`nemotron_asr_export`), each against the reference arithmetic it
replaces, plus the SentencePiece-BPE vocabulary it writes. None of it needs the checkpoint or the
`transformers` classes (5.13+, absent from CI's 4.57): the whole model is graded by loom.cpp's gate test
`test_e2e_nemotron_asr_mil_export` against `transformers`' own token ids.

Measured on the real checkpoint (2026-10-10): the wrapper equals `transformers`' sdpa encoder to 4e-6,
the exported encoder to 7.6e-6 under four language prompts, and the GGUF transcribes JFK and three
synthetic clips (zh/de/es, auto and named) token for token on the released rc16 wheel.
"""
import json

import pytest
import torch

from loom_exporter.nemotron_asr_export import (
    _NemotronMel, _chunked_limited_mask, _language_tag_ids, _relative_positions,
)


class _FakeExtractor:
    """The attributes `_NemotronMel` reads, with a random positive filterbank: what is checked is the
    framing and arithmetic, not librosa's filters."""

    def __init__(self, n_mels=16):
        self.n_fft, self.hop_length, self.win_length, self.preemphasis = 512, 160, 400, 0.97
        self.mel_filters = torch.rand(n_mels, self.n_fft // 2 + 1, generator=torch.Generator().manual_seed(0))


def _reference_mel(x, fe):
    """`NemotronAsrStreamingFeatureExtractor.__call__` for one clip, transcribed: preemphasis, a centred
    zero-padded STFT, power, filterbank, log -- then only the `len // hop` VALID frames it keeps."""
    x = torch.cat([x[:1], x[1:] - fe.preemphasis * x[:-1]])
    window = torch.hann_window(fe.win_length, periodic=False)
    stft = torch.stft(x[None], fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                      window=window, return_complex=True, pad_mode="constant", center=True)
    power = torch.sqrt(torch.view_as_real(stft).pow(2).sum(-1)).pow(2)
    mel = torch.log(fe.mel_filters @ power + 2.0 ** -24).permute(0, 2, 1)
    valid = (len(x) + fe.n_fft // 2 * 2 - fe.n_fft) // fe.hop_length
    return mel[:, :valid]


@pytest.mark.parametrize("n", [16000, 16000 + 77, 48005, 1600 * 3 + 159])
def test_the_mel_front_end_is_the_extractors_valid_frames(n):
    fe = _FakeExtractor()
    x = torch.randn(n, generator=torch.Generator().manual_seed(n)) * 0.1
    want = _reference_mel(x, fe)
    got = _NemotronMel(fe)(x[None])
    assert got.shape == want.shape == (1, n // 160, 16)
    torch.testing.assert_close(got, want, rtol=0, atol=2e-5)


def _reference_chunked_mask(n, left, right):
    """transformers' `chunked_limited_mask_function(left, right)`, element by element."""
    chunk = right + 1
    left_chunks = left // chunk if left >= 0 else 10_000
    q = torch.arange(n)[:, None] // chunk
    k = torch.arange(n)[None, :] // chunk
    return (q - k >= 0) & (q - k <= left_chunks)


@pytest.mark.parametrize("n,left,right", [(139, 56, 3), (40, 56, 0), (97, 56, 6), (300, 56, 13), (61, 70, 13)])
def test_the_chunked_mask_is_transformers_predicate(n, left, right):
    chunk = right + 1
    got = _chunked_limited_mask(n, chunk, left // chunk)[0, 0]
    assert torch.equal(got, _reference_chunked_mask(n, left, right))


@pytest.mark.parametrize("n", [1, 7, 139])
def test_the_relative_positions_are_the_references(n):
    hidden = 32
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden, 2).float() / hidden))
    # The reference's own spelling, with the negative-step range this module avoids.
    pos = torch.arange(n - 1, -n, -1).float()
    freqs = (inv_freq[:, None] @ pos[None, :]).T
    want = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(1, 2 * n - 1, hidden)
    torch.testing.assert_close(_relative_positions(n, inv_freq), want, rtol=0, atol=0)


def test_the_language_tags_are_the_special_locale_tokens(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"added_tokens": [
        {"id": 0, "content": "<unk>", "special": True},
        {"id": 1, "content": "<bg-BG>", "special": True},
        {"id": 7, "content": "<en-US>", "special": True},
        {"id": 9, "content": "<pad>", "special": True},
        {"id": 11, "content": "<xx-YY>", "special": False},
    ]}))
    assert _language_tag_ids(path) == (1, 7)


def _spm_bpe_dir(tmp_path, pre="Metaspace"):
    """A tiny SentencePiece-BPE `tokenizer.json`: `<unk>` at 0, two byte pieces, a language tag."""
    vocab = {"<unk>": 0, "<en-US>": 1, "<0x41>": 2, "<0xE3>": 3, "▁": 4, "a": 5, "b": 6, "▁a": 7, "▁ab": 8}
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": [["▁", "a"], ["▁a", "b"]], "byte_fallback": True,
                  "unk_token": "<unk>"},
        "added_tokens": [{"id": 0, "content": "<unk>", "special": True},
                         {"id": 1, "content": "<en-US>", "special": True},
                         {"id": 9, "content": "<pad>", "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": pre, "replacement": "▁", "prepend_scheme": "always", "split": True},
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"unk_token": "<unk>", "pad_token": "<pad>"}))
    return tmp_path


def test_a_sentencepiece_bpe_tokenizer_json_is_read_as_one(tmp_path):
    from loom_exporter.spm_tokenizer_export import read_hf_id_layout

    tok_dir = _spm_bpe_dir(tmp_path)
    assert read_hf_id_layout(tok_dir) is None            # opt-in: the default reading is unchanged
    layout = read_hf_id_layout(tok_dir, allow_bpe=True)
    assert layout.model_type == "bpe"
    assert layout.pieces[:3] == ["<unk>", "<en-US>", "<0x41>"] and layout.pieces[9] == "<pad>"
    assert layout.scores == [-float(i) for i in range(10)]
    assert layout.unk_id == 0 and layout.pad_id == 9 and layout.add_dummy_prefix


def test_a_byte_level_bpe_is_not_read_as_sentencepiece(tmp_path):
    from loom_exporter.spm_tokenizer_export import read_hf_id_layout

    with pytest.raises(NotImplementedError, match="Metaspace"):
        read_hf_id_layout(_spm_bpe_dir(tmp_path, pre="ByteLevel"), allow_bpe=True)


def test_the_sentencepiece_bpe_is_written_as_llama_with_byte_pieces_typed(tmp_path):
    gguf = pytest.importorskip("gguf")
    from loom_exporter.spm_tokenizer_export import read_hf_id_layout, write_sentencepiece_vocab

    out = tmp_path / "vocab.gguf"
    writer = gguf.GGUFWriter(str(out), "test")
    write_sentencepiece_vocab(writer, None, hf_ids=read_hf_id_layout(_spm_bpe_dir(tmp_path), allow_bpe=True))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.close()
    fields = gguf.GGUFReader(str(out)).fields
    assert bytes(fields["tokenizer.ggml.model"].parts[-1]).decode() == "llama"
    types = [int(fields["tokenizer.ggml.token_type"].parts[i][0]) for i in fields["tokenizer.ggml.token_type"].data]
    assert types[0] == 2 and types[1] == 3 and types[9] == 3      # UNKNOWN, CONTROL tag, CONTROL pad
    assert types[2] == types[3] == 6                               # BYTE, by spelling
    assert types[4:9] == [1] * 5
    # Not written for a BPE vocabulary: loom::Vocab refuses it there (no BPE byte fallback).
    assert "tokenizer.ggml.byte_fallback" not in fields


# ---- the live door (loom.cpp ADR-109): `NemotronEncoderStepPhase` against the one-pass wrapper ---------
#
# CI has no `transformers` Nemotron classes, so the toy encoder below is built from stand-ins that
# transcribe `modeling_nemotron_asr_streaming`'s OFFLINE path (5.14): the causal Conv2D stages pad
# `(k - 1, s - 1)` on time and freq, the attention is its relative-position one with `_rel_shift`, the
# conv module left-pads its depthwise conv. The step must reproduce that path from carried state, which
# is the claim the real checkpoint was measured on (1.9e-14 at f64 on two clips; the gate grades the
# file).

import math
import types

import torch.nn as nn
import torch.nn.functional as F

from loom_exporter.nemotron_asr_export import (  # noqa: E402
    NemotronEncoderStepPhase, _NemotronEncoderWrapper, chunk_key_mask,
)


class _CausalConv2D(nn.Conv2d):
    """`NemotronAsrStreamingEncoderCausalConv2D` offline: k=3, s=2, padded (2, 1) on both axes."""

    def __init__(self, cin, cout, groups=1):
        super().__init__(cin, cout, 3, stride=2, groups=groups)

    freq_pad = (2, 1)

    def forward(self, x):
        return super().forward(F.pad(F.pad(x, self.freq_pad), (0, 0, 2, 1)))


class _SubLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.depthwise_conv = _CausalConv2D(c, c, groups=c)
        self.pointwise_conv = nn.Conv2d(c, c, 1)

    def forward(self, h, lengths):
        return self.pointwise_conv(self.depthwise_conv(h)), lengths


class _Subsampling(nn.Module):
    def __init__(self, c, n_mel, d):
        super().__init__()
        self.conv_in = _CausalConv2D(1, c)
        self.layers = nn.ModuleList([_SubLayer(c), _SubLayer(c)])
        freq = n_mel
        for _ in range(3):
            freq = (freq + 3 - 3) // 2 + 1
        self.linear = nn.Linear(c * freq, d)

    def forward(self, features, mask):
        h = torch.relu(self.conv_in(features.unsqueeze(1)))
        for layer in self.layers:
            h = torch.relu(layer(h, None)[0])
        return self.linear(h.transpose(1, 2).reshape(h.shape[0], h.shape[2], -1))


class _Attention(nn.Module):
    """`NemotronAsrStreamingEncoderAttention.forward` with no cache, its eager softmax in the input's
    dtype (the reference takes it in f32, which an f64 comparison cannot use)."""

    def __init__(self, d, heads):
        super().__init__()
        self.config = types.SimpleNamespace(num_attention_heads=heads)
        self.head_dim = d // heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d) for _ in range(4))
        self.relative_k_proj = nn.Linear(d, d, bias=False)
        self.bias_u = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.bias_v = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)

    def _rel_shift(self, s):
        b, h, q, p = s.shape
        s = F.pad(s, pad=(1, 0)).view(b, h, -1, q)
        return s[:, :, 1:].view(b, h, q, p)

    def forward(self, x, position_embeddings, attention_mask):
        b, t, _ = x.shape
        heads, dim = self.config.num_attention_heads, self.head_dim
        q = self.q_proj(x).view(b, t, -1, dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, -1, dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, dim).transpose(1, 2)
        rel = self.relative_k_proj(position_embeddings).view(b, -1, heads, dim)
        bd = (q + self.bias_v.view(1, heads, 1, dim)) @ rel.permute(0, 2, 3, 1)
        bd = self._rel_shift(bd)[..., :k.shape[2]] * self.scaling
        bd = bd.masked_fill(attention_mask.logical_not(), float("-inf"))
        w = torch.softmax(((q + self.bias_u.view(1, heads, 1, dim)) @ k.transpose(2, 3)) * self.scaling + bd, -1)
        return self.o_proj((w @ v).transpose(1, 2).reshape(b, t, -1))


class _ConvModule(nn.Module):
    def __init__(self, d, kernel=9):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, groups=d)
        self.depthwise_conv.left_pad = kernel - 1
        self.norm = nn.LayerNorm(d)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x):
        h = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        h = self.depthwise_conv(F.pad(h, (self.depthwise_conv.left_pad, 0)))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return self.pointwise_conv2(self.activation(h)).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        ff = lambda: nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))  # noqa: E731
        self.feed_forward1, self.feed_forward2 = ff(), ff()
        self.self_attn = _Attention(d, heads)
        self.conv = _ConvModule(d)
        (self.norm_feed_forward1, self.norm_self_att, self.norm_conv, self.norm_feed_forward2,
         self.norm_out) = (nn.LayerNorm(d) for _ in range(5))

    def forward(self, h, attention_mask, position_embeddings):
        h = h + 0.5 * self.feed_forward1(self.norm_feed_forward1(h))
        h = h + self.self_attn(self.norm_self_att(h), position_embeddings, attention_mask)
        h = h + self.conv(self.norm_conv(h))
        h = h + 0.5 * self.feed_forward2(self.norm_feed_forward2(h))
        return self.norm_out(h)


def _toy_wrapper(right=1, left=8, d=32, heads=4, n_layers=2, n_prompts=5):
    torch.manual_seed(0)
    encoder = nn.Module()
    encoder.subsampling = _Subsampling(8, 16, d)
    encoder.input_scale = 1.0
    encoder.encode_positions = types.SimpleNamespace(
        inv_freq=1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d)))
    encoder.layers = nn.ModuleList([_Block(d, heads) for _ in range(n_layers)])
    encoder._resolve_attn_context = lambda n: (left, n)
    model = nn.Module()
    model.encoder = encoder
    model.prompt_projector = nn.Module()
    model.prompt_projector.linear_1 = nn.Linear(d + n_prompts, 24)
    model.prompt_projector.act = nn.ReLU()
    model.prompt_projector.linear_2 = nn.Linear(24, 20)
    model.encoder_projector = nn.Linear(20, 12)
    model.config = types.SimpleNamespace(encoder_config=types.SimpleNamespace(hidden_size=d))
    return _NemotronEncoderWrapper(model, _FakeExtractor(), right).eval().double()


def _streamed(step, x, groups, prompt):
    """`01_stream.lua`'s arithmetic in Python: whole chunks as they become ready (`groups` chunks per
    call, cycled), then `stream_finish`'s rest of the clip."""
    hop, cm, lead_mel = step.mel.hop, step.chunk_mel, step.lead_mel
    pad_left, extra = step.mel.pad_left, step.overlap
    lead = lead_mel * hop + 1 + pad_left
    clip = torch.cat([torch.zeros(lead, dtype=x.dtype), x])            # clip[0] is sample -lead
    n = len(x)
    states = [torch.zeros(s, dtype=x.dtype) for s in step.state_shapes()]
    rows, c, i = [], 0, 0

    def call(frame0, n_mel, n_valid):
        a = frame0 * hop - 1 - pad_left
        length = hop * n_mel + extra
        w = torch.zeros(length, dtype=x.dtype)
        seg = clip[a + lead: a + lead + length]
        w[:len(seg)] = seg
        out = step(w.view(1, -1), torch.tensor([float(frame0)], dtype=x.dtype),
                   torch.tensor([float(n_valid)], dtype=x.dtype), prompt, *states)
        states[:] = out[1:]
        rows.append(out[0])

    ready = math.floor(((n - pad_left) / hop + lead_mel + 1) / cm)
    while ready > c:
        g = min(ready - c, groups[i % len(groups)])
        call(cm * c - lead_mel, cm * g, hop * cm * g + extra - 1)
        c, i = c + g, i + 1
    frame0 = cm * c - lead_mel
    if n // hop - frame0 > 0:
        call(frame0, n // hop - frame0, n - (frame0 * hop - 1 - pad_left) - 1)
    return torch.cat(rows, dim=1)


@pytest.mark.parametrize("n", [16000, 16000 + 77, 5200, 161, 1600 * 3 + 159])
@pytest.mark.parametrize("groups", [[1], [3, 1, 2], [100]])
def test_the_encoder_step_is_the_one_pass_encoder(n, groups):
    wrapper = _toy_wrapper()
    step = NemotronEncoderStepPhase(wrapper).eval()
    x = torch.randn(n, generator=torch.Generator().manual_seed(n), dtype=torch.float64) * 0.1
    prompt = torch.tensor([3])
    with torch.no_grad():
        want = wrapper(x.view(1, -1), prompt)
        got = _streamed(step, x, groups, prompt)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


def test_the_step_needs_its_rows_before_time_zero_zeroed(monkeypatch):
    """Sabotage: without the masks the first call's virtual frames reach the first rows."""
    import loom_exporter.nemotron_asr_export as ne

    wrapper = _toy_wrapper()
    step = NemotronEncoderStepPhase(wrapper).eval()
    monkeypatch.setattr(ne, "_position_valid", lambda first, n, like: torch.ones(n, dtype=like.dtype))
    x = torch.randn(16000, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.1
    with torch.no_grad():
        want = wrapper(x.view(1, -1), torch.tensor([3]))
        got = _streamed(step, x, [1], torch.tensor([3]))
    assert not torch.allclose(got, want, rtol=0, atol=1e-6)


def test_the_step_needs_the_clip_end_marked():
    """Sabotage: a last call that takes its zero padding for the clip's own samples ends elsewhere."""
    wrapper = _toy_wrapper()
    step = NemotronEncoderStepPhase(wrapper).eval()
    x = torch.randn(16000 + 77, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.1
    clip_end = torch.cat([x, torch.full((400,), 0.3, dtype=torch.float64)])
    with torch.no_grad():
        want = wrapper(x.view(1, -1), torch.tensor([3]))
        # The same frames from a clip that goes on: only `n_valid` makes the padding zeros.
        got = _streamed(step, clip_end, [1], torch.tensor([3]))[:, :want.shape[1]]
    assert not torch.allclose(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("chunk,left_chunks,start,n", [(4, 14, 0, 4), (4, 14, 36, 12), (2, 4, 6, 7), (4, 14, 400, 4)])
def test_the_carried_key_mask_is_a_block_of_the_whole_clips_mask(chunk, left_chunks, start, n):
    """The step's mask over a carried window of `chunk * left_chunks` keys is the one-pass mask's rows
    for those queries, at those keys, wherever the window reaches back before time 0."""
    left = chunk * left_chunks
    positions = torch.arange(start, start + n).float()
    keys = torch.arange(start - left, start + n).float()
    got = chunk_key_mask(positions, keys, chunk, left_chunks)[0, 0] == 0
    whole = _chunked_limited_mask(start + n, chunk, left_chunks)[0, 0]
    want = torch.zeros(n, left + n, dtype=torch.bool)
    for j, key in enumerate(keys.long().tolist()):
        if key >= 0:
            want[:, j] = whole[start:start + n, key]
    assert torch.equal(got, want)
