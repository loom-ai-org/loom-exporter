"""MOSS-TTS-Local's export (family 10), on synthetic modules and a recording tokenizer -- no checkpoint.

The three places the export departs from the reference's own spelling are pinned here: the prompt's
template (text AND encoding granularity), the embedding fold (the pad code's zero row standing in for
the reference's mask), and the local GPT-2 block's re-spelling (interleaved-pair RoPE, the split
`c_attn`). The global stack is transformers' own `Qwen3Model`, and `load_model` asserts that against
the checkpoint's at export time, so it is not re-tested here.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from loom_exporter import moss_tts_export as mt


def test_the_after_reference_text_is_the_processors_with_every_field_unset():
    """`_render_user_prompt_after_reference` verbatim; a drift here changes every prompt silently."""
    assert mt.after_reference(None) == (
        "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone"
        "\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n")
    assert mt.after_reference("French").endswith("\n- Language:\nFrench\n- Text:\n")


class _RecordingTokenizer:
    """Encodes each string to one id per character and records the calls, so the test can see the
    GRANULARITY: the processor encodes each template piece on its own."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(c) for c in text]


def test_the_template_is_encoded_piece_by_piece_as_the_processor_does():
    tok = _RecordingTokenizer()
    cfg = SimpleNamespace(im_start_token_id=1, im_end_token_id=2, audio_start_token_id=3)
    seg = mt.prompt_segments(tok, cfg)
    assert seg["head"][0] == 1 and seg["tail"][-1] == 3
    assert tok.calls[:3] == ["user\n", "<user_inst>\n- Reference(s):\n", "None"]
    # "None" is its own segment: a clone prompt puts the references exactly there, and the head the
    # two prompts share must end where `_user_prompt_prefix_ids` does.
    assert seg["head"] == [1] + [ord(c) for c in "user\n<user_inst>\n- Reference(s):\n"]
    assert seg["no_reference"] == [ord(c) for c in "None"]
    # One `after` per language plus the no-language one, each a single encode of the rendered text.
    assert len(seg["after"]) == len(mt.LANGUAGES) + 1
    assert seg["after"][0] == [ord(c) for c in mt.after_reference(None)]


def test_the_pad_codes_zero_row_is_the_references_mask():
    torch.manual_seed(0)
    text = nn.Embedding(50, 8)
    audio = nn.ModuleList(nn.Embedding(6, 8) for _ in range(3))
    wrapper = mt._EmbedWrapper(text, audio, codebook_size=6)
    rows = torch.tensor([[[7, 1, 6, 2], [9, 6, 6, 6], [11, 0, 5, 3]]])        # 6 is the pad code
    got = wrapper(rows)
    want = text(rows[..., 0])
    for g, emb in enumerate(audio):
        ids = rows[..., g + 1]
        valid = ids.ne(6)
        want = want + emb(ids.masked_fill(~valid, 0)) * valid.unsqueeze(-1)
    assert torch.allclose(got, want, atol=1e-6)


def _synthetic_local(D=16, H=4, inner=24, base=1e6):
    block = nn.Module()
    attn = nn.Module()
    attn.num_heads, attn.head_dim, attn.embed_dim = H, D // H, D
    attn.scale_attn_weights, attn.scale_attn_by_inverse_layer_idx = True, False
    attn.c_attn, attn.c_proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
    attn.rotary_emb = SimpleNamespace(base=base)
    block.attn = attn
    block.ln_1, block.ln_2 = nn.LayerNorm(D), nn.LayerNorm(D)
    block.mlp = nn.Sequential(nn.Linear(D, inner), nn.SiLU(), nn.Linear(inner, D))
    local = nn.Module()
    local.h = nn.ModuleList([block])
    local.ln_f = nn.LayerNorm(D)
    return local


def _reference_local(local, x):
    """`MossTTSNanoGPT2Model` for one block, uncached, causal: the reference's own arithmetic."""
    block, attn = local.h[0], local.h[0].attn
    L, D, H, Dh = x.shape[1], attn.embed_dim, attn.num_heads, attn.head_dim
    q, k, v = attn.c_attn(block.ln_1(x)).split(D, dim=-1)
    q, k, v = (t.view(1, L, H, Dh) for t in (q, k, v))
    inv = 1.0 / (attn.rotary_emb.base ** (torch.arange(0, Dh, 2, dtype=torch.float32) / Dh))
    freqs = torch.arange(L, dtype=torch.float32).view(-1, 1) * inv
    cos = freqs.cos().repeat_interleave(2, -1).view(1, L, 1, Dh)
    sin = freqs.sin().repeat_interleave(2, -1).view(1, L, 1, Dh)

    def rot(t):
        return torch.stack((-t[..., 1::2], t[..., ::2]), dim=-1).reshape_as(t)

    q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    x = x + attn.c_proj(out.transpose(1, 2).reshape(1, L, D))
    x = x + block.mlp(block.ln_2(x))
    return local.ln_f(x)[:, -1]


def test_the_local_block_respelling_is_the_references_arithmetic():
    torch.manual_seed(1)
    D, n_vq, V = 16, 3, 5
    local = _synthetic_local(D=D)
    embeddings = torch.randn(n_vq * V, D)
    heads = torch.randn(n_vq * V + 2, D)
    hidden = torch.randn(1, 1, D)
    rows = torch.tensor([[2, V + 4]])
    wrapper = mt._LocalWrapper(local, heads, embeddings, with_rows=True).eval()
    with torch.no_grad():
        got = wrapper(hidden, torch.arange(3).view(1, -1), mt.causal_mask(3), rows)
        x = torch.cat([hidden, embeddings[rows]], dim=1)
        want = _reference_local(local, x) @ heads.t()
    assert got.shape == (1, n_vq * V + 2)
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())


def test_the_driver_draws_n_vq_plus_one_per_frame_in_the_references_order():
    """Continue/stop first, then the codebooks: the layout `draws` pins, and the reference's order."""
    lua = (Path(mt.__file__).parent / "moss_tts_driver" / "01_generate.lua").read_text()
    assert "_step * (N_VQ + 1)" in lua
    assert lua.index("lo = _stop_lo") < lua.index("lo = _g * CODEBOOK_SIZE")
    assert "inputs.language or 0" in lua


def test_the_recognizer_claims_the_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moss_tts_local"}))
    assert mt._is_moss_tts_local(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moss-audio-tokenizer"}))
    assert not mt._is_moss_tts_local(tmp_path)


# -- voice files (loom.cpp ADR-045) ------------------------------------------------------------------

def _codec(path, quantizer=1.0, encoder=1.0, sharded=False):
    """A stand-in codec directory: a config and `quantizer.*` / `encoder.*` tensors."""
    from safetensors.numpy import save_file

    np = pytest.importorskip("numpy")
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "moss-audio-tokenizer"}))
    tensors = {"quantizer.quantizers.0.codebook.weight": np.full((4, 2), quantizer, np.float32),
               "encoder.0.w": np.full(3, encoder, np.float32)}
    if sharded:
        save_file({"quantizer.quantizers.0.codebook.weight": tensors.pop(
            "quantizer.quantizers.0.codebook.weight")}, str(path / "model-00001.safetensors"))
        save_file(tensors, str(path / "model-00002.safetensors"))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "quantizer.quantizers.0.codebook.weight": "model-00001.safetensors",
            "encoder.0.w": "model-00002.safetensors"}}))
    else:
        save_file(tensors, str(path / "model.safetensors"))
    return path


def test_the_fingerprint_follows_the_quantizer_and_ignores_the_rest(tmp_path):
    """A code is the quantizer's index: re-training the encoder around the same codebooks leaves what
    a voice means unchanged, and any codebook change must change the fingerprint."""
    from loom_exporter.moss_tts_voices import codec_fingerprint

    base = codec_fingerprint(_codec(tmp_path / "a"))
    assert codec_fingerprint(_codec(tmp_path / "b", encoder=2.0)) == base
    assert codec_fingerprint(_codec(tmp_path / "c", quantizer=2.0)) != base
    assert codec_fingerprint(_codec(tmp_path / "d", sharded=True)) == base


def test_the_codec_is_found_beside_the_checkpoint_by_the_name_its_config_gives(tmp_path):
    from loom_exporter.moss_tts_voices import find_codec

    model = tmp_path / "moss-tts-local"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(
        {"audio_tokenizer_name_or_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"}))
    with pytest.raises(FileNotFoundError, match="MOSS-Audio-Tokenizer-v2"):
        find_codec(model)
    codec = _codec(tmp_path / "moss-audio-tokenizer-v2")
    assert find_codec(model) == codec
    assert find_codec(model, _codec(tmp_path / "elsewhere")) == tmp_path / "elsewhere"


def test_a_written_voice_file_holds_every_reference_and_their_lengths(tmp_path):
    import gguf
    import numpy as np

    from loom_exporter.moss_tts_voices import voice_arrays, write_voice

    refs = [np.arange(3 * 12).reshape(3, 12) % 1024, np.arange(2 * 12).reshape(2, 12) + 500]
    out = tmp_path / "voices" / "pair.gguf"
    n = write_voice(voice_arrays(refs), out, name="pair", compat="ab" * 16, license="CC0-1.0", origin="o")
    assert n == 5
    reader = gguf.GGUFReader(str(out))
    fields = {f.name: f.contents() for f in reader.fields.values() if f.name.startswith("loom.voice.")}
    assert fields == {"loom.voice.architecture": "moss_tts_local", "loom.voice.compat": "ab" * 16,
                      "loom.voice.name": "pair", "loom.voice.license": "CC0-1.0",
                      "loom.voice.origin": "o", "loom.voice.n_references": 2, "loom.voice.n_frames": 5}
    tensors = {t.name: np.asarray(t.data) for t in reader.tensors}
    # Frame-major, reference after reference: the order the driver reads them in.
    np.testing.assert_array_equal(tensors["reference_codes"], np.concatenate([r.reshape(-1) for r in refs]))
    np.testing.assert_array_equal(tensors["reference_frames"], [3, 2])


def test_every_reference_must_be_frames_by_codebooks():
    import numpy as np

    from loom_exporter.moss_tts_voices import voice_arrays

    with pytest.raises(ValueError, match="at least one"):
        voice_arrays([])
    with pytest.raises(ValueError, match="one n_vq"):
        voice_arrays([np.zeros((2, 12)), np.zeros((2, 8))])


def test_your_own_voice_needs_a_name_and_the_recordings_licence(tmp_path):
    from loom_exporter.moss_tts_voices import convert

    with pytest.raises(ValueError, match="--license"):
        convert(tmp_path, tmp_path / "out", wavs=[tmp_path / "me.wav"], name="me", license="")


def test_the_contract_declares_the_codecs_fingerprint(tmp_path):
    from loom_exporter.moss_tts_voices import codec_fingerprint

    model = tmp_path / "moss-tts-local"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(
        {"audio_tokenizer_name_or_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"}))
    codec = _codec(tmp_path / "moss-audio-tokenizer-v2")
    config = mt.TextToCodesMossTTSExportConfig(model_dir=str(model), output_path=str(tmp_path / "o.gguf"))
    config.task = "text-to-codes"
    assert config.contract()["voice.compat"] == codec_fingerprint(codec)


def test_the_driver_puts_the_references_where_the_template_says_none():
    """The reference's direct clone path: head, then per reference `<audio_start>`, USER-slot rows and
    `<audio_end>` with no separator, then the rest; "None" only when there is no reference."""
    lua = (Path(mt.__file__).parent / "moss_tts_driver" / "01_generate.lua").read_text()
    head = lua.index("_text_rows(PROMPT_HEAD)")
    assert head < lua.index("_text_rows(PROMPT_NO_REFERENCE)") < lua.index("_text_rows(PROMPT_AFTER")
    assert lua.index("_text_rows({AUDIO_START_ID})") < lua.index("USER_SLOT_ID") < \
        lua.index("_text_rows({AUDIO_END_ID})") < lua.index("_text_rows(PROMPT_AFTER")
    assert "inputs.reference_frames or {#_ref / N_VQ}" in lua
