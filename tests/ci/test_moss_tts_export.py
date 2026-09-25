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
