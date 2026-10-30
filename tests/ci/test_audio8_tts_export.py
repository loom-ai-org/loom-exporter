"""Audio8-TTS's export (family 10) and its codec's voice files, on synthetic modules -- no checkpoint.

The places the export departs from the reference's own spelling are pinned here: the template's
encoding granularity, the de-interleaved rope rows (the reference rotates PAIRS), the embedding fold
(an absent code's zero row standing in for `where(semantic, ...)`), the trimmed head, and the driver's
sampler options. The real checkpoint is checked at export time (`check_equivalence`, greedy codes id for
id against `generate`), so that is not re-tested here.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from loom_exporter import audio8_tts_export as a8


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(c) for c in text]


def test_the_template_is_encoded_piece_by_piece_as_the_processor_does():
    tok = _RecordingTokenizer()
    seg = a8.prompt_segments(tok)
    assert tok.calls[:4] == a8.NO_REFERENCE_HEAD
    assert seg["no_reference_head"] == [ord(c) for c in "".join(a8.NO_REFERENCE_HEAD)]
    assert seg["tail"] == [ord(c) for c in "".join(a8.TAIL)]
    # The transcript is the voice file's own ids, so the reference head stops before the speaker tag.
    assert seg["reference_head"] == [ord(c) for c in "".join(a8.REFERENCE_HEAD)]
    assert seg["reference_speech"] == [ord(c) for c in "\n\nSpeech:\n"]


def _interleaved_rope(x, freqs):
    """`modeling_arktts._apply_rope`: x `[B, T, H, D]`, freqs `[T, D/2, 2]`, rotating pairs (2j, 2j+1)."""
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope = freqs[None, :, None]
    out = torch.stack((shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
                       shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1]), dim=-1)
    return out.flatten(3)


def _freqs(length, head_dim, base=1e6):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    phases = torch.outer(torch.arange(length).float(), inv)
    v = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((v.real, v.imag), dim=-1).to(torch.bfloat16)      # the reference's bf16 table


def test_the_deinterleaved_rows_are_the_references_pair_rotation():
    """Permuting each q/k head's rows evens-then-odds turns pair rotation into rotate-half, and the
    attention output -- GQA by `repeat_interleave`, causal -- is the reference's."""
    torch.manual_seed(0)
    H, KV, D, dim, T = 4, 2, 8, 16, 5
    attn = SimpleNamespace(n_head=H, n_local_heads=KV, head_dim=D, qk_norm=False,
                           wqkv=nn.Linear(dim, (H + 2 * KV) * D, bias=True), wo=nn.Linear(H * D, dim))
    ours = a8._Attention(attn)
    freqs = _freqs(T, D)
    cos, sin = a8.rope_tables(freqs)
    x = torch.randn(1, T, dim)
    with torch.no_grad():
        got = ours(x, cos[None], sin[None], a8.causal_mask(T))
        q, k, v = attn.wqkv(x).split((H * D, KV * D, KV * D), dim=-1)
        q = _interleaved_rope(q.view(1, T, H, D), freqs).transpose(1, 2)
        k = _interleaved_rope(k.view(1, T, KV, D), freqs).transpose(1, 2)
        v = v.view(1, T, KV, D).transpose(1, 2)
        k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
        mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
        want = attn.wo(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(1, T, H * D))
    assert torch.allclose(got, want, atol=1e-5)


def _fake_model(V=40, size=6, n=3, dim=8, begin=20, eos=5):
    config = SimpleNamespace(num_codebooks=n, codebook_size=size, dim=dim, semantic_begin_id=begin,
                             semantic_end_id=begin + size - 1, eos_token_id=eos, norm_fastlayer_input=True)
    return SimpleNamespace(config=config, embeddings=nn.Embedding(V, dim),
                           codebook_embeddings=nn.Embedding(n * size, dim), layers=[],
                           norm=nn.Identity(), freqs_cis=_freqs(8, 4))


def test_the_absent_codes_zero_row_is_the_references_semantic_mask():
    torch.manual_seed(0)
    model = _fake_model()
    embed = a8._EmbedWrapper(model)
    size, n, begin = 6, 3, 20
    # A text row (absent codes) and two semantic rows.
    rows = torch.tensor([[[7, size, size, size], [begin + 2, 2, 0, 5], [begin + 4, 4, 1, 3]]])
    got = embed(rows)
    ids, codes = rows[0, :, 0], rows[0, :, 1:]
    semantic = (ids >= begin) & (ids < begin + size)
    total = sum(model.codebook_embeddings(codes[:, i].clamp_max(size - 1) + i * size) for i in range(n))
    want = model.embeddings(ids) + torch.where(semantic[:, None], total, 0.0)
    assert torch.allclose(got[0], want, atol=1e-6)


def test_the_head_is_the_semantic_block_then_eos():
    model = _fake_model()
    slow = a8._SlowWrapper(model)
    W = model.embeddings.weight
    assert slow.head.weight.shape == (7, 8)
    assert torch.equal(slow.head.weight[:6], W[20:26]) and torch.equal(slow.head.weight[6], W[5])


def test_the_recognizers_split_the_directory_from_its_codec_file(tmp_path):
    from loom_exporter.audio_codec_export import _is_audio8_codec

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "arktts"}))
    (tmp_path / "codec.pth").write_bytes(b"")
    assert a8._is_audio8_tts(tmp_path) and not a8._is_audio8_tts(tmp_path / "codec.pth")
    assert _is_audio8_codec(tmp_path / "codec.pth") and not _is_audio8_codec(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moss_tts_local"}))
    assert not a8._is_audio8_tts(tmp_path) and not _is_audio8_codec(tmp_path / "codec.pth")


def _codec_pth(path, quantizer=1.0, encoder=1.0):
    path.mkdir(parents=True)
    torch.save({"quantizer.semantic_quantizer.quantizers.0.codebook.weight": torch.full((4, 2), quantizer),
                "quantizer.quantizer.quantizers.0.codebook.weight": torch.full((3, 2), quantizer),
                "quantizer.post_module.norm.weight": torch.full((2,), encoder),
                "encoder.block.0.conv.bias": torch.full((3,), encoder)}, str(path / "codec.pth"))
    return path


def test_the_fingerprint_follows_the_quantizers_and_ignores_the_rest(tmp_path):
    from loom_exporter.audio8_tts_voices import codec_fingerprint

    base = codec_fingerprint(_codec_pth(tmp_path / "a"))
    assert codec_fingerprint(_codec_pth(tmp_path / "b", encoder=2.0)) == base
    assert codec_fingerprint(_codec_pth(tmp_path / "c", quantizer=2.0)) != base


def test_a_written_voice_file_holds_the_transcript_and_the_frames(tmp_path):
    import gguf
    import numpy as np

    from loom_exporter.audio8_tts_voices import write_voice

    codes = (np.arange(4 * 10).reshape(4, 10) * 37) % 1024
    tokens = np.array([60, 91, 3, 7])
    out = tmp_path / "voices" / "me.gguf"
    assert write_voice({"reference_tokens": tokens, "reference_codes": codes}, out, name="me",
                       compat="ab" * 16, license="CC0-1.0", origin="o") == 4
    reader = gguf.GGUFReader(str(out))
    fields = {f.name: f.contents() for f in reader.fields.values() if f.name.startswith("loom.voice.")}
    assert fields["loom.voice.architecture"] == "audio8_tts" and fields["loom.voice.n_frames"] == 4
    tensors = {t.name: np.asarray(t.data) for t in reader.tensors}
    np.testing.assert_array_equal(tensors["reference_codes"], codes.reshape(-1))     # frame-major
    np.testing.assert_array_equal(tensors["reference_tokens"], tokens)


def test_your_own_voice_needs_its_transcript_and_the_recordings_licence(tmp_path):
    from loom_exporter.audio8_tts_voices import convert

    with pytest.raises(ValueError, match="--text"):
        convert(tmp_path, tmp_path / "out", wav=tmp_path / "me.wav", text="", name="me", license="CC0")
    with pytest.raises(ValueError, match="--license"):
        convert(tmp_path, tmp_path / "out", wav=tmp_path / "me.wav", text="hi", name="me", license="")


def test_the_driver_samples_with_the_references_rule():
    """Every draw names the ADR-095 options (an engine ignores a misspelled key, so a typo here would
    sample with `transformers'` rule unnoticed); the RAS window skips frame 0 as the reference's does."""
    lua = (Path(a8.__file__).parent / "audio8_tts_driver" / "01_generate.lua").read_text()
    assert "top_p_mass = 'row'" in lua
    assert "truncation = 'untempered'" in lua and "top_p_crossing = 'exclude'" in lua
    assert lua.count("_options(") >= 4                 # semantic, RAS redraw, fast, + the definition
    assert "if _step > 0 then" in lua
    # The reference prompt: transcript, then frames, then the user turn.
    assert lua.index("_text_rows(inputs.reference_tokens)") < lua.index("SEMANTIC_BEGIN + _ref[_at + 1]") \
        < lua.index("_text_rows(PROMPT_REFERENCE_TO_TEXT)")
