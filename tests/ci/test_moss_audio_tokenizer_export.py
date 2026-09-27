"""MOSS-Audio-Tokenizer-v2's decode export (family 11), on synthetic modules shaped like the checkpoint's
remote code -- so this runs with no checkpoint and no `trust_remote_code`.

What is under test is the three places the export departs from the reference's own spelling: the
BLOCKED attention (exact against dense windowed attention at every block size), the FOLDED quantizer
with its absent rows, and the shape facts the blocking relies on surviving the trace -- the last is
the one that failed silently, with every RANGE_1D over a stack's length written as `end: 1`.
"""
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from loom_exporter import moss_audio_tokenizer_export as moss


class _Attention(nn.Module):
    def __init__(self, dim, heads, context):
        super().__init__()
        self.embed_dim, self.num_heads, self.context, self.causal = dim, heads, context, True
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(dim) + 0.5)

    def forward(self, x):
        return self.scale * x


class _Layer(nn.Module):
    def __init__(self, dim, heads, context):
        super().__init__()
        self.self_attn = _Attention(dim, heads, context)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim, bias=False), nn.GELU(),
                                 nn.Linear(2 * dim, dim, bias=False))
        self.layer_scale_1, self.layer_scale_2 = _LayerScale(dim), _LayerScale(dim)


class _Projected(nn.Module):
    """`MossAudioTokenizerProjectedTransformer`'s attribute shape."""

    def __init__(self, c_in, c_out, dim=16, heads=2, layers=2, context=6):
        super().__init__()
        self.input_proj = nn.Linear(c_in, dim, bias=False)
        self.output_proj = nn.Linear(dim, c_out, bias=False)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(_Layer(dim, heads, context) for _ in range(layers))
        self.transformer.rope = SimpleNamespace(max_period=10000.0)


class FakePatchedPretransform(nn.Module):     # the class NAME is what the decoder plan reads
    def __init__(self, patch):
        super().__init__()
        self.patch_size = patch


def _dense_stage(stage: _Projected, x):
    """The reference's arithmetic, masked-dense: `[T, C_in] -> [T, C_out]`."""
    x = stage.input_proj(x)
    T = x.shape[0]
    for layer in stage.transformer.layers:
        attn = layer.self_attn
        H, Dh = attn.num_heads, attn.embed_dim // attn.num_heads
        h = layer.norm1(x)
        q, k, v = attn.in_proj(h).reshape(T, 3, H, Dh).permute(1, 2, 0, 3)
        half = torch.arange(Dh // 2, dtype=torch.float32)
        freqs = torch.exp(half * (-math.log(stage.transformer.rope.max_period) * 2 / Dh))
        ang = torch.arange(T, dtype=torch.float32).view(-1, 1) * freqs
        cos, sin = torch.cos(ang), torch.sin(ang)

        def rope(t):
            t = t.reshape(H, T, Dh // 2, 2)
            re, im = t[..., 0], t[..., 1]
            return torch.stack([re * cos - im * sin, re * sin + im * cos], -1).reshape(H, T, Dh)

        q, k = rope(q), rope(k)
        delta = torch.arange(T).view(-1, 1) - torch.arange(T).view(1, -1)
        allowed = (delta >= 0) & (delta < attn.context)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, allowed)
        x = x + layer.layer_scale_1(attn.out_proj(out.permute(1, 0, 2).reshape(T, -1)))
        x = x + layer.layer_scale_2(layer.ffn(layer.norm2(x)))
    return stage.output_proj(x)


@pytest.mark.parametrize("frames_per_block,rate,context", [
    (4, 1, 6),     # blocks narrower than the window: two blocks back
    (4, 2, 8),     # block == window
    (4, 4, 5),     # block wider than the window: one block back
    (1, 1, 7),     # one frame per block: p = window - 1
])
def test_blocked_attention_is_dense_windowed_attention(frames_per_block, rate, context):
    torch.manual_seed(0)
    stage = _Projected(12, 10, context=context).eval()
    blocked = moss.BlockedStage(stage, frames_per_block, rate).eval()
    T = frames_per_block * rate * 7
    x = torch.randn(T, 12)
    with torch.no_grad():
        want = _dense_stage(stage, x)
        got = blocked(x.unsqueeze(0))[0]
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4), float((got - want).abs().max())


def _quantizer(n_q=3, size=5, dim=4, rvq=6, out=8):
    q = nn.Module()
    q.codebook_size = size
    q.quantizers = nn.ModuleList()
    for _ in range(n_q):
        lfq = nn.Module()
        lfq.codebook = nn.Embedding(size, dim)
        lfq.out_proj = nn.Conv1d(dim, rvq, 1)
        q.quantizers.append(lfq)
    q.output_proj = nn.Conv1d(rvq, out, 1)
    return q


def _reference_rvq_decode(q, codes):                 # codes [n_q_used, T]
    emb = 0
    for i in range(codes.shape[0]):
        lfq = q.quantizers[i]
        emb = emb + lfq.out_proj(lfq.codebook(codes[i]).t().unsqueeze(0))
    return q.output_proj(emb)[0].t()                 # [T, out]


def test_an_absent_codebook_contributes_nothing_so_a_prefix_decodes_exactly():
    """The residual quantizer's prefix decode -- the first k codebooks summed and the rest never
    looked at -- is what an absent id reproduces, bias included."""
    torch.manual_seed(1)
    q = _quantizer()
    table, proj = moss.fold_quantizer(q)
    size = q.codebook_size + 1
    codes = torch.randint(0, q.codebook_size, (3, 9))
    for used in (1, 2, 3):
        row = codes.clone()
        row[used:] = q.codebook_size                                 # absent
        index = (row.t() + torch.arange(3) * size).reshape(-1)
        got = proj(table[index].reshape(9, 3, -1).sum(1))
        with torch.no_grad():
            want = _reference_rvq_decode(q, codes[:used])
        assert torch.allclose(got, want, atol=1e-5), used


def _synthetic_model():
    torch.manual_seed(2)
    model = nn.Module()
    model.quantizer = _quantizer(n_q=3, size=5, dim=4, rvq=6, out=8)
    model.decoder = nn.ModuleList([_Projected(8, 8, context=6), FakePatchedPretransform(2),
                                   _Projected(4, 4, context=6), FakePatchedPretransform(4)])
    model.config = SimpleNamespace(sampling_rate=48000, downsample_rate=4, number_channels=2,
                                   enable_channel_interleave=True)
    return model.eval()


def test_the_decoder_blocks_every_stack_and_emits_interleaved_samples():
    model = _synthetic_model()
    decoder = moss.MossCodecDecoder(model, frames_per_block=4)
    codes = torch.randint(0, 5, (1, 8, 3))
    with torch.no_grad():
        out = decoder(codes)
    # 8 frames x 2 x 4 positions x 1 channel of the final patch = 64 floats, L R L R ...
    assert out.shape == (1, 64)
    assert [s.B for s in decoder.stages] == [4, 8]


def test_geometry_counts_interleaved_channels():
    geometry = moss.geometry(_synthetic_model())
    assert geometry["channels"] == 2 and geometry["n_codebooks"] == 3
    assert geometry["hop_length"] == 4 and geometry["codebook_size"] == 5


def _export(tmp_path, monkeypatch):
    pytest.importorskip("coremltools")
    from gguf import GGUFReader
    from loom_exporter.audio_codec_export import AudioCodecExportConfig, CodecFamily

    monkeypatch.setattr(moss, "load", lambda model_dir: _synthetic_model())
    out = tmp_path / "moss.gguf"
    config = AudioCodecExportConfig(architecture="moss-audio-tokenizer", output_path=str(out),
                                    model_dir=str(tmp_path), family=CodecFamily.MOSS_AUDIO_TOKENIZER,
                                    frames_per_block=4, n_frames=16, max_frames=256)
    config.task = "audio-codec"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "topology": json.loads(reader.fields["model.graph_topology.main_topology"].contents()),
        "channels": reader.fields["loom.channels"].contents(),
        "absent": reader.fields["loom.codec.absent_code"].contents(),
        "n_codebooks": reader.fields["loom.codec.n_codebooks"].contents(),
    }


def test_every_length_the_blocking_reads_stays_symbolic(tmp_path, monkeypatch):
    """THE CHECK THIS FILE EXISTS FOR. In a flat `[T, C]` layout the first stack's length sat at axis
    0 and `value_facts.gather_shape_value` read it as a batch size: `arange(T)` was written as
    `end: 1`, every position got position zero's rotation, and the export ran 0.1% off with no error.
    Every RANGE_1D here must be an expression in the root axis."""
    exported = _export(tmp_path, monkeypatch)
    ranges = [n["attrs"] for n in exported["topology"]["nodes"] if n["op"] == "RANGE_1D"]
    assert ranges, "no RANGE_1D at all -- positions and block indices should both be one"
    literal = [r for r in ranges if "n_codes" not in str(r.get("end"))]
    assert not literal, f"RANGE_1D bounds that lost the root axis: {literal}"


def test_the_driver_pads_to_whole_blocks_with_the_absent_id_and_trims(tmp_path, monkeypatch):
    exported = _export(tmp_path, monkeypatch)
    driver = exported["driver"]
    assert exported["channels"] == 2 and exported["absent"] == 5 and exported["n_codebooks"] == 3
    assert "math.floor(#codes / 3)" in driver
    assert "= 5" in driver                                   # the pad id written into the copy
    assert "array_slice" in driver and "loom.run_subgraph('main_topology'" in driver
    # the trim is `n_frames * hop * channels` floats: 4 * 2 = 8 per frame
    assert "_n_frames * 8" in driver


def test_the_recognizer_claims_the_model_type_and_nothing_else(tmp_path):
    from loom_exporter.audio_codec_export import _is_moss_audio_tokenizer

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moss-audio-tokenizer"}))
    assert _is_moss_audio_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moss_tts_local"}))
    assert not _is_moss_audio_tokenizer(tmp_path)
