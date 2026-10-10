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
