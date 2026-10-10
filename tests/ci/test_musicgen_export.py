"""Family 14 (P5): MusicGen -- a T5 prompt in, four delayed EnCodec code streams out.

**Every assertion here is about a way the export succeeds and the model is wrong**, because each was
the actual state of the first real export at some point, or is the next shape of it:

1. **The decoder must reach the KV cache.** MusicGen's eager attention scales the SCORES rather than Q,
   and its mask builder inverts a 4-D mask. Together they left `fuse_loom_attention` matching nothing:
   48 bare SOFTMAX nodes, no cached ATTENTION, a decoder attending to the current step alone. (With
   only the mask defect, the exporter's own guard refuses the graph, which is the loud version.)
2. **The positions are an input.** HF derives them from `past_key_values_length`, which a cache-free
   trace fixes at 0, so a traced-as-is decoder embeds every step at position 0.
3. **The traced lengths must not reach the graph**, checked by two exports that must be identical.

Everything runs against a real, randomly-initialised `MusicgenForConditionalGeneration` small enough to
trace in a unit test (family 12's reason for real over mocked).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_t5_export import _spiece_bytes  # noqa: E402  (T5's own vocabulary layout)

from loom_exporter.musicgen_export import (  # noqa: E402
    TextToCodesMusicgenExportConfig,
    _build_musicgen,
    _is_musicgen,
    install_scaled_query_attention,
)
from loom_exporter.registry import default_registry  # noqa: E402

_N_CODEBOOKS = 3
_N_LAYERS = 2


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection and the task ------------------------------------------------------------------------

def test_a_musicgen_directory_is_claimed(tmp_path):
    assert _is_musicgen(_hf_dir(tmp_path, "mg", {"model_type": "musicgen"}))


def test_melody_and_dia_are_not_claimed(tmp_path):
    """`musicgen_melody` conditions on a chromagram concatenated to the decoder input, not on
    cross-attended text; this wrapper would export the wrong graph for it."""
    assert not _is_musicgen(_hf_dir(tmp_path, "melody", {"model_type": "musicgen_melody"}))
    assert not _is_musicgen(_hf_dir(tmp_path, "dia", {"model_type": "dia"}))


def test_the_registry_resolves_a_synthetic_musicgen(tmp_path):
    recognizer = default_registry().detect(_hf_dir(tmp_path, "mg", {"model_type": "musicgen"}))
    assert recognizer.name == "musicgen"
    assert recognizer.task == "text-to-codes"


def test_hparams_are_empty_without_a_checkpoint(tmp_path):
    assert _build_musicgen(tmp_path, "/tmp/x.gguf").hparams() == {}


# -- the scaled-query patch ------------------------------------------------------------------------

def test_the_scaled_query_attention_is_bit_identical_at_musicgens_head_dim():
    """The patch moves `head_dim ** -0.5` from the scores onto Q. At head_dim 64 that factor is 1/8,
    a power of two, so the two orders round identically -- which is the claim the export relies on and
    `phases()` guards. Compared at the real head_dim, with values whose products do round."""
    torch = pytest.importorskip("torch")
    from transformers.models.musicgen import modeling_musicgen

    original = modeling_musicgen.eager_attention_forward
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 64, generator=gen) * 3 for _ in range(3))
    mask = torch.triu(torch.full((9, 9), float("-inf")), diagonal=1).view(1, 1, 9, 9)

    class _M(torch.nn.Module):
        training = False

    try:
        want, _ = original(_M(), q, k, v, mask, scaling=64 ** -0.5)
        install_scaled_query_attention()
        got, _ = modeling_musicgen.eager_attention_forward(_M(), q, k, v, mask, scaling=64 ** -0.5)
    finally:
        modeling_musicgen.eager_attention_forward = original
    assert torch.equal(want, got)


# -- the export ------------------------------------------------------------------------------------

def _tiny_musicgen(tmp_path: Path, *, hidden: int = 16, heads: int = 4, name: str = "tiny-mg") -> Path:
    pytest.importorskip("torch")
    import torch
    from transformers import (
        EncodecConfig, MusicgenConfig, MusicgenDecoderConfig, MusicgenForConditionalGeneration,
        T5Config,
    )

    # d_model 8 against a decoder hidden of 16, so `enc_to_dec_proj` exists, as it does at full size
    # (768 -> 1024).
    text = T5Config(vocab_size=64, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_heads=2,
                    relative_attention_num_buckets=8, relative_attention_max_distance=16,
                    n_positions=32, feed_forward_proj="relu", eos_token_id=1, pad_token_id=0)
    audio = EncodecConfig(target_bandwidths=[2.2], sampling_rate=32000, audio_channels=1,
                          num_filters=4, codebook_size=32, codebook_dim=8, hidden_size=8,
                          upsampling_ratios=[2, 2], num_residual_layers=1, num_lstm_layers=1)
    decoder = MusicgenDecoderConfig(vocab_size=32, max_position_embeddings=64,
                                    num_hidden_layers=_N_LAYERS, ffn_dim=32, num_attention_heads=heads,
                                    hidden_size=hidden, num_codebooks=_N_CODEBOOKS,
                                    pad_token_id=32, bos_token_id=32)
    config = MusicgenConfig.from_sub_models_config(text, audio, decoder)
    out = tmp_path / name
    with torch.device("cpu"):
        MusicgenForConditionalGeneration(config).eval().save_pretrained(out)
    (out / "generation_config.json").write_text(json.dumps({
        "do_sample": True, "guidance_scale": 3.0, "max_length": 40,
        "bos_token_id": 32, "pad_token_id": 32, "decoder_start_token_id": 32,
    }))
    (out / "spiece.model").write_bytes(_spiece_bytes(64))
    (out / "special_tokens_map.json").write_text(json.dumps({
        "unk_token": "<unk>", "pad_token": "<pad>", "eos_token": "</s>",
    }))
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    from gguf import GGUFReader

    config = TextToCodesMusicgenExportConfig(model_dir=str(checkpoint), output_path=str(out), **kwargs)
    config.task = "text-to-codes"
    config.export()
    reader = GGUFReader(str(out))
    topo = {name: json.loads(reader.fields[f"model.graph_topology.{name}"].contents())
            for name in ("encoder", "cross_kv", "cross_kv_uncond", "decoder", "decoder_uncond")}
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        **topo,
        "sampling": {k[len("loom.sampling."):]: reader.fields[k].contents()
                     for k in reader.fields if k.startswith("loom.sampling.")},
        "n_codebooks": reader.fields["loom.codec.n_codebooks"].contents(),
    }


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("coremltools")
    tmp = tmp_path_factory.mktemp("musicgen")
    return _export(_tiny_musicgen(tmp), tmp / "tiny.gguf")


def test_every_self_attention_block_is_cached_and_no_cross_attention_is(exported):
    """THE CHECK THIS FILE EXISTS FOR. One cached ATTENTION per decoder layer, and the cross-attention
    blocks left as plain softmaxes: fused, they would take KV-cache slots the self-attention addresses."""
    nodes = exported["decoder"]["nodes"]
    attention = [n for n in nodes if n["op"] == "ATTENTION"]
    assert len(attention) == _N_LAYERS, f"{len(attention)} fused blocks for {_N_LAYERS} layers"
    assert all(n.get("attrs", {}).get("kv_cache", True) for n in attention)
    assert sorted(n["attrs"]["layer"] for n in attention) == list(range(_N_LAYERS))
    assert sum(1 for n in nodes if n["op"] == "SOFTMAX") == _N_LAYERS   # the cross-attention
    # The mask reaches the fused blocks as the cache-wide `[n_kv, n_tokens]` input, not as something
    # HF's builder computed from it (the inverted 0/1 reading left a SUB here).
    shapes = {i["name"]: i["shape"] for i in exported["decoder"]["inputs"]}
    assert shapes["attention_mask"] == ["n_kv", "n_tokens"]
    assert not any(n["op"] == "SUB" for n in nodes)


def test_the_positions_are_read_at_the_callers_position_ids(exported):
    """The positional table is gathered at the `position_ids` INPUT. Gathered at a range the graph
    computed instead, every cached step would be embedded at position 0."""
    nodes = exported["decoder"]["nodes"]
    by_output = {n.get("output") or (n.get("outputs") or [None])[0]: n for n in nodes}
    gathers = [n for n in nodes if n["op"] == "GET_ROWS"
               and any("embed_positions" in str(i) for i in n["inputs"])]
    assert len(gathers) == 1, gathers

    # Walk the index operand back through shape-only ops; it must end at the graph input.
    index = gathers[0]["inputs"][1]
    for _ in range(8):
        node = by_output.get(index)
        if node is None or node["op"] not in ("RESHAPE", "VIEW", "CONT", "PERMUTE"):
            break
        index = node["inputs"][0]
    assert index == "position_ids", f"positions gathered at {index!r}, not the position_ids input"


def test_the_decoder_carries_two_independent_dynamic_axes(exported):
    shapes = {i["name"]: i["shape"] for i in exported["decoder"]["inputs"]}
    assert shapes["codes"] == [str(_N_CODEBOOKS), "n_tokens", "1"]
    for name in ("xk_0", "xv_0"):
        assert "n_enc_frames" in shapes[name] and "n_tokens" not in str(shapes[name])


def test_the_unconditional_streams_are_private_aliases(exported):
    assert exported["decoder_uncond"]["kv_cache_scope"] == "private"
    assert "kv_cache_scope" not in exported["decoder"]
    assert exported["decoder_uncond"]["nodes"] == exported["decoder"]["nodes"]
    assert exported["cross_kv_uncond"]["nodes"] == exported["cross_kv"]["nodes"]
    driver = exported["driver"]
    assert "loom.run_subgraph_and_retain('decoder_uncond'" in driver
    # MusicGen's unconditional K/V are zero, so the stream is fed zeros, not a T5 pass over id 0.
    assert "loom.run_subgraph_and_retain('cross_kv_uncond', {n_enc_frames = 1" in driver


def test_the_decoding_defaults_are_generate_s_not_the_files(exported):
    """`top_k` is absent from the generation config, and `transformers` then uses 50, not 0. The
    guidance scale is the standard uncond-centred form, so it is declared and used unconverted."""
    assert exported["sampling"]["top_k"] == 50
    assert exported["sampling"]["temperature"] == pytest.approx(1.0)
    assert exported["sampling"]["guidance_scale"] == pytest.approx(3.0)
    assert "scale = _guidance}" in exported["driver"]
    assert "_guidance + 1.0" not in exported["driver"]


def test_the_traced_lengths_do_not_reach_the_graph(tmp_path):
    pytest.importorskip("coremltools")
    checkpoint = _tiny_musicgen(tmp_path)
    short = _export(checkpoint, tmp_path / "a.gguf", trace_text_len=7, trace_steps=5)
    long = _export(checkpoint, tmp_path / "b.gguf", trace_text_len=13, trace_steps=9)
    for name in ("encoder", "cross_kv", "decoder", "driver"):
        assert short[name] == long[name], name


def test_a_head_dim_the_patch_cannot_keep_exact_is_refused(tmp_path):
    """head_dim 8 makes the scale 1/sqrt(8), not a power of two, so moving it onto Q would change the
    rounding. Refused rather than exported a ulp away from the reference."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_musicgen(tmp_path, hidden=16, heads=2)
    config = TextToCodesMusicgenExportConfig(model_dir=str(checkpoint),
                                             output_path=str(tmp_path / "x.gguf"))
    config.task = "text-to-codes"
    with pytest.raises(ValueError, match="power of four"):
        config.phases()


# -- the general guard -----------------------------------------------------------------------------

def test_a_cached_phase_that_fuses_nothing_fails_the_export():
    """The guard Retro-078 put in `convert_phase` for EVERY family, driven without any family's own
    count: one attention block written the way MusicGen's eager path writes it -- scores scaled after
    `Q @ K^T` -- declared with a KV cache. `fuse_loom_attention` cannot match it, and the phase must
    fail instead of writing a cached decoder with no cache in it. The same block scaled on Q fuses,
    which is what shows the guard is about the miss and not about the block."""
    torch = pytest.importorskip("torch")
    ct = pytest.importorskip("coremltools")
    import numpy as np

    from loom_exporter.multi_phase_export import ExportPhase
    from loom_exporter.phase_conversion import convert_phase

    class _Block(torch.nn.Module):
        def __init__(self, scale_on_q: bool):
            super().__init__()
            self.qkv = torch.nn.Linear(8, 24, bias=False)
            self.out = torch.nn.Linear(8, 8, bias=False)
            self.scale_on_q = scale_on_q

        def forward(self, x, mask):
            q, k, v = (t.view(1, -1, 2, 4).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
            if self.scale_on_q:
                scores = torch.matmul(q * 0.5, k.transpose(2, 3))
            else:
                scores = torch.matmul(q, k.transpose(2, 3)) * 0.5
            probs = torch.softmax(scores + mask, dim=-1)
            return self.out(torch.matmul(probs, v).transpose(1, 2).reshape(1, -1, 8))

    def phase(scale_on_q):
        axis = ct.RangeDim(1, 16)
        return ExportPhase(
            name="decoder", wrapper=_Block(scale_on_q).eval(),
            dummy_inputs=(torch.zeros(1, 4, 8),
                          torch.triu(torch.full((4, 4), float("-inf")), 1).view(1, 1, 4, 4)),
            mil_inputs=[ct.TensorType(name="x", shape=(1, axis, 8), dtype=np.float32),
                        ct.TensorType(name="attention_mask", shape=(1, 1, axis, axis),
                                      dtype=np.float32)],
            fuse_attention=True, kv_cache_size=16,
        )

    fused = convert_phase(phase(scale_on_q=True))
    assert any(n["op"] == "ATTENTION" for n in fused.topologies["decoder"]["nodes"])
    with pytest.raises(ValueError, match="fused no cached ATTENTION"):
        convert_phase(phase(scale_on_q=False))
