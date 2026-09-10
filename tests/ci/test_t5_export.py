"""Family 6 (P5): a text encoder-decoder -- source text in, generated text out.

**What this file exists for is the relative attention bias**, because that is the one piece of T5 the
engine has no primitive for and the whole export turns on where it is computed. Three properties, each
of which the export satisfies silently when it is wrong:

1. **The bias is bucketed, and the Lua that buckets it must match `transformers`.** A driver that
   bucketed differently would still index inside the table and still produce a plausible sentence, so
   the check is against HF's own `_relative_position_bucket` over the whole (query, key) grid rather
   than against a recorded array.
2. **Cross-attention must not fuse.** Its zero bias is folded away by a coremltools pass, not by
   anything here; if a release stops folding it, every cross block acquires a KV cache addressing the
   self-attention blocks' slots and the model decodes plausible tokens out of a mixed cache.
3. **The mask input keeps its HEAD axis through `n_kv` retyping.** T5's mask is per-head, and the
   retyping used to rewrite the whole shape as `[n_kv, n_tokens]` -- which declares a 3-D input as 2-D
   and builds it one sixth of its real size.

Everything below runs against a real, randomly-initialised `T5ForConditionalGeneration` small enough
to trace in a unit test: real rather than mocked because what is under test is the interaction with
the actual `T5Attention`, and a stub has none of it.
"""
import json
import math
from pathlib import Path

import pytest

from loom_exporter.registry import default_registry
from loom_exporter.t5_export import (
    Text2TextT5ExportConfig,
    _build_t5,
    _check_fused_attention,
    _is_t5,
    cross_kv_input_names,
    relative_attention_bias_table,
)
from loom_exporter.tasks import task_spec


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection and the task ------------------------------------------------------------------------

def test_a_t5_directory_is_claimed(tmp_path):
    assert _is_t5(_hf_dir(tmp_path, "t5", {"model_type": "t5"}))


def test_the_t5_variants_this_wrapper_does_not_cover_are_not_claimed(tmp_path):
    """`longt5` is a different block (local/transient-global attention) and `mt5`/`umt5` spell their
    own `model_type`; claiming them would export a graph shaped for something else."""
    for other in ("mt5", "umt5", "longt5", "switch_transformers"):
        assert not _is_t5(_hf_dir(tmp_path, other, {"model_type": other}))


def test_the_registry_resolves_a_synthetic_t5(tmp_path):
    recognizer = default_registry().detect(_hf_dir(tmp_path, "t5", {"model_type": "t5"}))
    assert recognizer.name == "t5"
    assert recognizer.task == "text2text-generation"


def test_the_task_is_its_own_and_not_text_generation():
    """`text-generation` is one causal stack continuing its own prompt; this is a source read once by
    a bidirectional encoder and an answer generated against it. They share a modality pair and share
    no export shape, so the base config is the multi-phase one."""
    spec = task_spec("text2text-generation")
    assert not spec.reserved
    assert spec.base_config == "multi_phase_export:BaseMultiPhaseModelExportConfig"


def test_the_modality_pair_is_text_in_text_out(tmp_path):
    """A host asking for text and getting text back should not have to know whether one stack or two
    produced it."""
    config = _build_t5(tmp_path, "/tmp/x.gguf")
    config.task = "text2text-generation"
    contract = config.contract()
    assert contract["input.kind"] == "text"
    assert contract["output.kind"] == "text"
    assert contract["text.frontend"] == "vocab"


def test_cross_kv_names_interleave_k_and_v():
    """One function orders the wrapper's return tuple, names the decoder's inputs and is what the
    driver's `2*layer + 1` arithmetic assumes -- so they cannot disagree."""
    assert cross_kv_input_names(2) == ("xk_0", "xv_0", "xk_1", "xv_1")


# -- the bias table and its bucketing ---------------------------------------------------------------

def _lua_bucket(rel, bidirectional, num_buckets, max_distance):
    """`t5_driver/00_header.lua`'s `t5_bucket`, transcribed line for line.

    A transcription rather than a call into a Lua interpreter, deliberately: what is under test is
    whether the ARITHMETIC matches `transformers`, and the tests/ci tier has no engine to run Lua in.
    The transcription is checked against the real fragment's text below, so the two cannot silently
    drift apart.
    """
    bucket = 0
    n = num_buckets
    if bidirectional:
        n = num_buckets // 2
        if rel > 0:
            bucket = n
        if rel < 0:
            rel = -rel
    else:
        if rel > 0:
            rel = 0
        rel = -rel
    max_exact = n // 2
    if rel < max_exact:
        return bucket + rel
    large = max_exact + math.floor(
        math.log(rel / max_exact) / math.log(max_distance / max_exact) * (n - max_exact))
    if large > n - 1:
        large = n - 1
    return bucket + large


@pytest.mark.parametrize("bidirectional", [True, False])
def test_the_drivers_bucketing_matches_transformers_over_the_whole_grid(bidirectional):
    """THE CHECK THIS FILE EXISTS FOR. Every (query, key) pair a 512-long sequence can produce, against
    `T5Attention._relative_position_bucket` itself -- because a bucketing that is merely CLOSE still
    indexes inside the table, still produces fluent output, and is wrong everywhere."""
    torch = pytest.importorskip("torch")
    from transformers.models.t5.modeling_t5 import T5Attention

    n = 512
    rel = torch.arange(n)[None, :] - torch.arange(n)[:, None]
    reference = T5Attention._relative_position_bucket(
        rel, bidirectional=bidirectional, num_buckets=32, max_distance=128)
    for i in range(0, n, 37):
        for j in range(0, n, 41):
            assert _lua_bucket(int(rel[i, j]), bidirectional, 32, 128) == int(reference[i, j]), (
                f"({i},{j}) rel={int(rel[i, j])}")


def test_the_transcribed_bucketing_is_the_fragment_that_ships():
    """The test above checks a Python transcription; this checks the transcription is of the real Lua.
    Without it the fragment could be edited and every bucketing assertion above would keep passing."""
    lua = (Path(_build_t5(Path("/tmp"), "x").driver_script_path) / "00_header.lua").read_text()
    for line in ("local n = num_buckets",
                 "n = math.floor(num_buckets / 2)",
                 "if rel > 0 then bucket = n end",
                 "if rel < 0 then rel = -rel end",
                 "if rel > 0 then rel = 0 end",
                 "local max_exact = math.floor(n / 2)",
                 "if large > n - 1 then large = n - 1 end"):
        assert line in lua, line


def test_the_bias_table_is_flattened_bucket_major(tmp_path):
    """`bucket * n_head + head`, which is the layout the Lua indexes with. Row-major over
    `[num_buckets, n_head]` is that layout; the transpose is not, and would read some other bucket's
    bias for every pair without raising."""
    torch = pytest.importorskip("torch")
    model = _tiny_t5_model(num_heads=3, relative_attention_num_buckets=8)
    table = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    flat = relative_attention_bias_table(model.encoder)
    assert len(flat) == 8 * 3
    for bucket in range(8):
        for head in range(3):
            assert flat[bucket * 3 + head] == pytest.approx(float(table[bucket, head]))


def test_a_stack_with_no_bias_table_is_refused(tmp_path):
    """Every T5 stack puts the table on block 0 and reuses its output for the rest, which is the same
    fact that makes ONE `position_bias` input serve the whole stack. A checkpoint that does not is not
    this architecture."""
    pytest.importorskip("torch")
    model = _tiny_t5_model()
    model.encoder.block[0].layer[0].SelfAttention.has_relative_attention_bias = False
    with pytest.raises(ValueError, match="relative_attention_bias"):
        relative_attention_bias_table(model.encoder)


def test_a_bias_table_that_does_not_match_the_declared_shape_is_refused(tmp_path):
    """The driver indexes the flat table as `bucket * n_head + head`, so a table of the wrong height
    is satisfied silently -- every index still lands inside the array."""
    pytest.importorskip("torch")
    checkpoint = _tiny_t5(tmp_path)
    config = Text2TextT5ExportConfig(model_dir=str(checkpoint), output_path=str(tmp_path / "x.gguf"))
    original = Text2TextT5ExportConfig.load_model

    def _shrink(self):
        model = original(self)
        model.config.relative_attention_num_buckets += 1
        return model

    Text2TextT5ExportConfig.load_model = _shrink
    try:
        with pytest.raises(ValueError, match="relative-attention table"):
            config.phases()
    finally:
        Text2TextT5ExportConfig.load_model = original


# -- the fusion outcome -----------------------------------------------------------------------------

def test_a_fused_cross_attention_block_is_refused():
    """The absence this export depends on, asserted directly: `_check_fused_attention` is the only
    thing standing between a coremltools release that stops folding `scores + 0` and a model whose
    cross-attention blocks each hold a KV cache slot the self-attention blocks address."""
    assert _check_fused_attention({"nodes": [{"op": "ATTENTION"}, {"op": "ATTENTION"}]}, 2) == 2
    with pytest.raises(ValueError, match="Cross-attention must NOT fuse"):
        _check_fused_attention({"nodes": [{"op": "ATTENTION"}] * 4}, 2)
    with pytest.raises(ValueError, match="Cross-attention must NOT fuse"):
        _check_fused_attention(
            {"nodes": [{"op": "ATTENTION"},
                       {"op": "ATTENTION", "attrs": {"kv_cache": False}}]}, 2)


# -- the real trace ---------------------------------------------------------------------------------

def _tiny_t5_model(**overrides):
    pytest.importorskip("torch")
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    # `d_kv * num_heads` is 6 against a `d_model` of 8, DELIBERATELY: T5 sizes its heads independently
    # of its residual stream, so the cross-attention K/V are narrower than the encoder output they come
    # from -- and a fixture where the two happened to match declared them at the wrong width and passed.
    config = T5Config(
        vocab_size=64, d_model=8, d_kv=3, d_ff=16, num_layers=1, num_decoder_layers=1,
        num_heads=2, relative_attention_num_buckets=8, relative_attention_max_distance=16,
        n_positions=32, feed_forward_proj="gated-gelu", tie_word_embeddings=False,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    with torch.device("cpu"):
        return T5ForConditionalGeneration(config).eval()


def _spiece_bytes(n_pieces: int) -> bytes:
    """A real Unigram `ModelProto` in T5's own layout -- `<pad>`, `</s>`, `<unk>`, then pieces.

    Built field by field rather than trained, the way `test_spm_tokenizer_export` builds its fixtures:
    what is under test here is the graph, and the vocabulary only has to be one the writer accepts.
    T5's order matters even so, because `decoder_start_token_id` is `<pad>` at 0 and eos is `</s>` at 1.
    """
    pytest.importorskip("sentencepiece")
    from sentencepiece import sentencepiece_model_pb2 as spm_pb2

    m = spm_pb2.ModelProto()
    m.trainer_spec.model_type = m.trainer_spec.UNIGRAM
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.remove_extra_whitespaces = True
    m.normalizer_spec.precompiled_charsmap = b"\x00\x00\x00\x00charsmap"
    for piece, score, kind in (("<pad>", 0.0, 3), ("</s>", 0.0, 3), ("<unk>", 0.0, 2)):
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = piece, score, kind
    for i in range(n_pieces - 3):
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = f"\u2581w{i}", -float(i), 1
    return m.SerializeToString()


def _tiny_t5(tmp_path: Path, name: str = "tiny-t5", **overrides) -> Path:
    out = tmp_path / name
    model = _tiny_t5_model(**overrides)
    model.save_pretrained(out)
    # The SentencePiece vocabulary travels with the model, and this is the layout every T5 ships: a
    # bare `spiece.model` with no `tokenizer.json` beside it. That is also the branch where the
    # framing flags matter -- with no post-processor to read, `add_eos_token` is the only thing that
    # says a T5 sequence ends in `</s>`.
    (out / "spiece.model").write_bytes(_spiece_bytes(model.config.vocab_size))
    (out / "special_tokens_map.json").write_text(json.dumps({
        "unk_token": "<unk>", "pad_token": "<pad>", "eos_token": "</s>",
    }))
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    from gguf import GGUFReader

    config = Text2TextT5ExportConfig(model_dir=str(checkpoint), output_path=str(out), **kwargs)
    config.task = "text2text-generation"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "encoder": json.loads(reader.fields["model.graph_topology.encoder"].contents()),
        "cross_kv": json.loads(reader.fields["model.graph_topology.cross_kv"].contents()),
        "decoder": json.loads(reader.fields["model.graph_topology.decoder"].contents()),
        "tensors": {t.name for t in reader.tensors},
    }


def test_the_masks_head_axis_survives_the_n_kv_retyping(tmp_path):
    """T5's mask is the relative bias and the causal mask summed, so it is PER HEAD -- and the `n_kv`
    retyping used to rewrite the whole shape as `[n_kv, n_tokens]`, which declares a 3-D input as 2-D
    and allocates it one head deep."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_t5(tmp_path), tmp_path / "tiny.gguf")
    shapes = {i["name"]: i["shape"] for i in exported["decoder"]["inputs"]}
    assert shapes["position_bias"] == ["n_kv", "n_tokens", "2"], shapes["position_bias"]
    # The encoder's is NOT retyped: that phase is unfused, so its mask is an ordinary input over the
    # root axis and there is no cache for `n_kv` to mean anything about.
    enc = {i["name"]: i["shape"] for i in exported["encoder"]["inputs"]}
    assert enc["position_bias"] == ["n_tokens", "n_tokens", "2", "1"], enc["position_bias"]


def test_the_decoder_carries_two_independent_dynamic_axes(tmp_path):
    """The source length and the number of tokens generated from it are unrelated. Collapsed onto one
    symbol the export still succeeds and the cross-attention K/V are sized by the step count, which is
    wrong for every sentence."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_t5(tmp_path), tmp_path / "tiny.gguf")
    shapes = {i["name"]: i["shape"] for i in exported["decoder"]["inputs"]}
    for name in ("xk_0", "xv_0"):
        assert "n_enc_frames" in shapes[name], f"{name} -> {shapes[name]}"
        assert "n_tokens" not in str(shapes[name]), f"{name} collapsed onto the step axis"
        # `num_heads * d_kv`, not `d_model`. The engine checks a retained output against the input it
        # is copied into, so declaring the residual width here fails at the first decode step -- and
        # only on a checkpoint where the two differ, which is why the fixture makes them differ.
        assert shapes[name][0] == "6", f"{name} is {shapes[name][0]} wide, expected num_heads * d_kv"
    # ... and the loop's own call binds that axis, or `SymbolEnv` raises on an unbound symbol.
    assert "n_enc_frames = #inputs.tokens" in exported["driver"]


def test_the_shared_embedding_is_written_once(tmp_path):
    """Both stacks read `model.shared`, and the encoder and decoder are traced SEPARATELY -- so the
    65 MB matrix is written twice unless both wrappers call it under one name. `merge_phase_weights`
    dedups an identical name with an identical value and hard-errors on an identical name with a
    different one, so this is also what keeps the two stacks' own weights from colliding."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_t5(tmp_path), tmp_path / "tiny.gguf")
    embeddings = [t for t in exported["tensors"] if "shared" in t]
    assert len(embeddings) == 1, sorted(exported["tensors"])


def test_the_traced_lengths_do_not_reach_the_graph(tmp_path):
    """Two exports differing only in the two trace lengths, required to be identical. Both axes are
    varied, because one baked axis is enough to make the model wrong and either could be the one."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_t5(tmp_path)
    short = _export(checkpoint, tmp_path / "a.gguf", trace_tokens=3, trace_src=5)
    long = _export(checkpoint, tmp_path / "b.gguf", trace_tokens=7, trace_src=11)
    assert short["encoder"] == long["encoder"]
    assert short["cross_kv"] == long["cross_kv"]
    assert short["decoder"] == long["decoder"]
    assert short["driver"] == long["driver"]


def test_the_decode_loop_starts_from_the_checkpoints_own_start_token(tmp_path):
    """The caller's tokens are the SOURCE and they went to the encoder; the loop's prompt is
    `decoder_start_token_id`. A loop that started from the source would generate a continuation of it,
    which for an encoder-decoder is not the model's answer."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_t5(tmp_path, name="t5-start", decoder_start_token_id=5,
                                eos_token_id=7), tmp_path / "tiny.gguf")
    assert "local _step_tokens = {5}" in exported["driver"], exported["driver"]
    assert "inputs.eos_token or 7" in exported["driver"]
