"""Moonshine Streaming (family 2) -- the hermetic half.

Each rewrite the export makes of transformers' modules, against transformers' own formula restated here
(CI's transformers predates `moonshine_streaming`), and the three exporter changes the model needed,
through the real compiler where the change is a lowering:

* a `range_1d` indexing a gather reached GET_ROWS as F32, which ggml refuses (the encoder-position
  table, `pos_emb(arange(n))`);
* the window mask and the RoPE angle were both broadcasts of BOTH operands, which the engine's
  elementwise ops refuse -- the angle only on a prefill longer than one token, which the decode loop
  never makes;
* a SentencePiece-converted `tokenizer.json` with a dummy prefix decoded with every space dropped until
  the vocabulary said so (`tokenizer.ggml.add_space_prefix`).

The real checkpoints are the gate half: loom vs transformers on 73 LibriSpeech utterances is identical
in ids and text for both sizes, and the tensor numbers are in loom.cpp's Epic-03.
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import loom_exporter  # noqa: F401 -- registers the "loom" backend
from loom_exporter import moonshine_export as M


# -- the rewrites, against transformers' own formulas ----------------------------------------------------

def _hf_window_allowed(n, window):
    """`sliding_window_mask_function`, verbatim in effect."""
    left, right = window
    q = torch.arange(n).view(n, 1)
    k = torch.arange(n).view(1, n)
    dist = q - k
    return ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))


@pytest.mark.parametrize("window", [(16, 4), (16, 0), (3, 1), (1, 0)])
@pytest.mark.parametrize("n", [1, 2, 17, 40])
def test_the_window_mask_is_transformers_predicate(window, n):
    """One band, `-max(right, 1) < q - k < left`, and no `logical_or`; built as an outer product."""
    got = M.window_mask(n, window)[0, 0]
    allowed = _hf_window_allowed(n, window)
    assert torch.equal(got == 0, allowed)
    assert torch.all(got[~allowed] == M._MASKED_SCORE)
    # Every query keeps its own key, so no row is all-masked and the softmax is never 0/0.
    assert torch.all(allowed.diagonal())


def test_asinh_is_torchs_over_the_range_the_front_end_sees():
    x = torch.linspace(-60.0, 60.0, 100001, dtype=torch.float64)
    assert torch.allclose(M._asinh(x), torch.asinh(x), rtol=1e-14, atol=1e-15)
    assert M._asinh(torch.zeros(1, dtype=torch.float64)).item() == 0.0


def _hf_apply_rope(x, cos_full, sin_full):
    """transformers' `apply_rotary_pos_emb` for one tensor, with its own interleaved `rotate_half`."""
    cos = cos_full[..., : cos_full.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin_full[..., : sin_full.shape[-1] // 2].repeat_interleave(2, dim=-1)
    dim = cos.shape[-1]
    x_rot, x_pass = x[..., :dim], x[..., dim:]
    rotated = torch.stack((-x_rot[..., 1::2], x_rot[..., 0::2]), dim=-1).flatten(-2)
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)


@pytest.mark.parametrize("head_dim,rotary", [(40, 32), (64, 32)])
def test_the_interleaved_rope_is_transformers(head_dim, rotary):
    """Pre-interleaved frequencies and a +-1 permutation matmul: the same numbers, exactly, at f32."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary))
    rope = M._InterleavedRope(inv_freq)
    positions = torch.tensor([[0, 1, 5, 4095]])
    x = torch.randn(1, 8, 4, head_dim, generator=torch.Generator().manual_seed(0))

    freqs = (inv_freq[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    want = _hf_apply_rope(x, emb.cos().unsqueeze(1), emb.sin().unsqueeze(1))

    cos, sin = rope.cos_sin(positions, torch.float32)
    assert torch.equal(rope.apply(x, cos, sin), want)


# -- the lowerings, through the real compiler -----------------------------------------------------------

def _export(module, shape, dynamic_axis, tmp_path, name="toy"):
    """Trace `module(x)` with one dynamic axis; return {topology name: nodes}."""
    import coremltools as ct
    from gguf import GGUFReader

    x = torch.randn(*shape)
    traced = torch.jit.trace(module.eval(), (x,))
    dims = list(shape)
    dims[dynamic_axis] = ct.RangeDim(1, 4096)
    program = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=tuple(dims))],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    out = tmp_path / f"{name}.gguf"
    loom_exporter.LoomGGUFBackend()(program, output_path=str(out), architecture=name,
                                    root_axis="n_frames", flat_namespace=True)
    reader = GGUFReader(str(out))
    key = next(k for k in reader.fields if k.startswith("model.graph_topology"))
    return json.loads(reader.fields[key].contents())["nodes"]


class _PositionTableAndMask(nn.Module):
    """`pos_emb(arange(n))` beside a float use of the same range -- the encoder's shape."""

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(64, 4)

    def forward(self, x):
        n = x.shape[1]
        idx = torch.arange(n)
        return x + self.table(idx) + idx.to(x.dtype).unsqueeze(-1)


def test_a_range_indexing_a_gather_reaches_it_as_i32_and_nothing_else_does(tmp_path):
    nodes = _export(_PositionTableAndMask(), (1, 6, 4), 1, tmp_path)
    ranges = [n for n in nodes if n["op"] == "RANGE_1D"]
    assert len(ranges) == 1
    rng = ranges[0]["outputs"][0]
    (gather,) = [n for n in nodes if n["op"] == "GET_ROWS"]
    (cast,) = [n for n in nodes if n["op"] == "CAST" and n["outputs"][0] == gather["inputs"][1]]
    assert cast["inputs"] == [rng] and cast["attrs"] == {"dtype": "i32"}
    # The float reader keeps the F32 range itself.
    readers = [n for n in nodes if rng in n.get("inputs", []) and n is not cast]
    assert readers and all(n["op"] != "GET_ROWS" for n in readers)


class _WindowMaskOnly(nn.Module):
    def forward(self, x):
        return M.window_mask(x.shape[1], (16, 4))[0, 0] + x


def test_the_window_mask_lowers_without_a_two_sided_broadcast(tmp_path):
    """The `q - k` matrix is one MUL_MAT of `[n, 2] x [2, n]`, not a SUB of `[n, 1]` and `[1, n]`."""
    nodes = _export(_WindowMaskOnly(), (1, 6), 1, tmp_path)
    assert any(n["op"] == "MUL_MAT" for n in nodes)
    assert not any(n["op"] == "SUB" for n in nodes)


class _RopeAngle(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = M._InterleavedRope(torch.tensor([1.0, 0.5, 0.25]))

    def forward(self, x):
        cos, _ = self.rope.cos_sin(x, torch.float32)
        return cos


def test_the_rope_angle_is_an_outer_product(tmp_path):
    nodes = _export(_RopeAngle(), (1, 5), 1, tmp_path)
    assert any(n["op"] == "MUL_MAT" for n in nodes)


# -- the tokenizer: a dummy prefix the vocabulary declares ---------------------------------------------

def _spm_tokenizer_json(prepend, strip):
    normalizers = ([{"type": "Prepend", "prepend": "▁"}] if prepend else []) + \
                  [{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]
    decoders = [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"}]
    if strip:
        decoders.append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "▁": 3, "a": 4, "▁a": 5}
    return {
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": None,
        "decoder": {"type": "Sequence", "decoders": decoders},
        "added_tokens": [{"id": i, "content": p, "special": True} for p, i in list(vocab.items())[:3]],
        "model": {"type": "BPE", "byte_fallback": True, "vocab": vocab, "merges": ["▁ a"]},
    }


def _write_vocab(tmp_path, tokenizer_json, pre):
    from gguf import GGUFReader, GGUFWriter

    from loom_exporter.bpe_tokenizer_export import write_bpe_vocab

    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    out = tmp_path / "v.gguf"
    w = GGUFWriter(str(out), "t")
    write_bpe_vocab(w, str(tmp_path), pre_type=pre)
    w.add_tensor("x", np.zeros(1, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return GGUFReader(str(out)).fields


def test_a_prepended_space_is_declared_on_the_spm_shape(tmp_path):
    fields = _write_vocab(tmp_path, _spm_tokenizer_json(prepend=True, strip=True), "spm-byte-fallback")
    assert bool(fields["tokenizer.ggml.add_space_prefix"].contents())


def test_a_tokenizer_without_one_writes_no_key(tmp_path):
    """Gemma 3's shape: no prefix, so its file is byte-identical to before."""
    fields = _write_vocab(tmp_path, _spm_tokenizer_json(prepend=False, strip=False),
                          "granite-embed-multi-311m")
    assert "tokenizer.ggml.add_space_prefix" not in fields


def test_a_prefix_the_decoder_does_not_strip_is_refused(tmp_path):
    with pytest.raises(NotImplementedError, match="strip"):
        _write_vocab(tmp_path, _spm_tokenizer_json(prepend=True, strip=False), "spm-byte-fallback")


# -- recognizer -----------------------------------------------------------------------------------------

def test_the_recognizer_reads_the_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moonshine_streaming"}))
    assert M._is_moonshine_streaming(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moonshine"}))
    assert not M._is_moonshine_streaming(tmp_path), "v1 Moonshine is a different architecture"
    assert not M._is_moonshine_streaming(tmp_path / "missing")


def test_it_is_registered_for_asr():
    from loom_exporter.registry import default_registry

    registry = default_registry()
    names = {(rec.task, rec.name) for entry in registry._entries.values() for rec in entry.recognizers}
    assert ("automatic-speech-recognition", "moonshine-streaming") in names
