"""Family 4 (P5): CNN + transformer + CTC -- wav2vec 2.0, HuBERT, data2vec-audio.

The end-to-end tests here build REAL tiny `*ForCTC` models and trace them through the real compiler,
for family 12's reason and one of this family's own. The shared reason is that the whole of what this
family had to solve is what a trace bakes: `attention_mask` is omitted deliberately, and a test that
exports at one length and never looks at the graph's axis would pass against exactly that bug.

The reason that is this family's own is `groups`. A `Wav2Vec2PositionalConvEmbedding` is a GROUPED
convolution -- `groups=16` over 768 channels, neither dense nor depthwise -- and the exporter's rule
was `groups > 1 -> depthwise` for eight families before this one, because every convolution it had ever
seen sat at one end of that range or the other. So the emitted OP is asserted, not just the export
(loom.cpp Retro-046).

The detection half needs no torch at all and is where a new checkpoint's first failure shows up.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.ctc_asr_export import (
    ASRCtcExportConfig,
    _build_hf_ctc_asr,
    _is_hf_ctc_asr,
    read_ctc_vocab,
    strip_weight_norm,
)
from loom_exporter.registry import default_registry


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection -----------------------------------------------------------------------------------

def test_a_ctc_checkpoint_is_claimed(tmp_path):
    path = _hf_dir(tmp_path, "w2v", {"model_type": "wav2vec2",
                                     "architectures": ["Wav2Vec2ForCTC"]})
    assert _is_hf_ctc_asr(path)


@pytest.mark.parametrize("model_type,architecture", [
    ("hubert", "HubertForCTC"),
    ("data2vec-audio", "Data2VecAudioForCTC"),
])
def test_the_other_two_architectures_are_claimed_by_the_same_recognizer(tmp_path, model_type,
                                                                        architecture):
    """One generic recognizer, not one per architecture: `*ForCTC` is the checkpoint's own statement of
    which `AutoModelFor*` class it loads through, and that is the whole claim."""
    path = _hf_dir(tmp_path, model_type, {"model_type": model_type,
                                          "architectures": [architecture]})
    assert _is_hf_ctc_asr(path)
    assert default_registry().detect(path).name == "hf-ctc-asr"


def test_a_pretraining_checkpoint_is_not_claimed(tmp_path):
    """THE ONE THAT SITS BESIDE THESE ON DISK. `facebook/wav2vec2-large-xlsr-53` declares the same
    `model_type` and `Wav2Vec2ForPreTraining`; it has no CTC head and no tokenizer, so claiming it
    would export a GGUF whose `logits` are an encoder activation with nothing to decode them."""
    assert not _is_hf_ctc_asr(_hf_dir(tmp_path, "pre", {
        "model_type": "wav2vec2", "architectures": ["Wav2Vec2ForPreTraining"]}))


def test_the_architecture_half_is_load_bearing(tmp_path):
    """`TaskRegistry.detect` runs every recognizer against every path, so `model_type` alone would
    claim every other audio family's checkpoints."""
    assert not _is_hf_ctc_asr(_hf_dir(tmp_path, "bare", {"model_type": "wav2vec2"}))
    assert not _is_hf_ctc_asr(_hf_dir(tmp_path, "lm", {"model_type": "qwen3",
                                                        "architectures": ["Qwen3ForCausalLM"]}))
    assert not _is_hf_ctc_asr(_hf_dir(tmp_path, "seq", {
        "model_type": "wav2vec2", "architectures": ["Wav2Vec2ForSequenceClassification"]}))


def test_a_directory_that_is_not_an_hf_checkpoint_is_a_no_not_an_error(tmp_path):
    assert not _is_hf_ctc_asr(tmp_path / "nothing-here")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{not json")
    assert not _is_hf_ctc_asr(tmp_path / "broken")


def test_the_recognizer_is_a_fallback():
    """This family's task already has three SPECIFIC recognizers (the NeMo archives), so a generic one
    has to be consulted last or it would race them."""
    entry = default_registry()._entries["automatic-speech-recognition"]
    generic = [r for r in entry.recognizers if r.name == "hf-ctc-asr"]
    assert len(generic) == 1 and generic[0].fallback


def test_the_driver_builder_is_family_1s(tmp_path):
    """The reduction is the same one Conformer-CTC uses -- `loom.argmax_rows` plus the collapse -- which
    is the roadmap's estimate for this family ("needs no new head at all") holding."""
    config = _build_hf_ctc_asr(tmp_path, "/tmp/x.gguf")
    assert config.synthesized_builder_key() == "CtcGreedy"
    assert config.backend_kwargs()["driver_builder"] == "CtcGreedy"
    # No blank id before the trace, and that is deliberate: `component_registry.usage()` builds every
    # registered config without tracing, and the exporter raises where the number is actually needed.
    assert "ctc_blank_id" not in config.backend_kwargs()


# -- the vocabulary, and the two ids that are not derivable from it --------------------------------

def _write_ctc_tokenizer(directory: Path, vocab: dict, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(json.dumps(vocab))
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    return directory


ENGLISH_VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6}
ENGLISH_CONFIG = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>",
                  "eos_token": "</s>", "word_delimiter_token": "|"}

# The multilingual arrangement, and the reason the blank is RESOLVED rather than assumed: this
# checkpoint's `pad_token` is `<s>`, and it has a separate `<pad>` piece at id 1 that is not the blank.
# Reading "the id of the piece named by pad_token" gets 0; reading "the id of `<pad>`" gets 1, and a CTC
# decode that drops the wrong class returns a transcript made of nothing but blanks.
MULTILINGUAL_VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, " ": 4, "R": 5}
MULTILINGUAL_CONFIG = {"pad_token": "<s>", "unk_token": "<unk>", "bos_token": "<s>",
                       "eos_token": "</s>", "word_delimiter_token": " "}


def test_the_blank_is_the_pad_token_not_the_last_class(tmp_path):
    directory = _write_ctc_tokenizer(tmp_path / "en", ENGLISH_VOCAB, ENGLISH_CONFIG)
    read = read_ctc_vocab(str(directory))
    assert read["blank_id"] == 0
    assert read["blank_id"] != len(read["pieces"]) - 1   # NeMo's convention, which is not this one
    assert read["word_delimiter_id"] == 4
    assert read["pieces"][:5] == ["<pad>", "<s>", "</s>", "<unk>", "|"]


def test_the_blank_is_resolved_through_the_vocabulary_not_spelled(tmp_path):
    directory = _write_ctc_tokenizer(tmp_path / "ml", MULTILINGUAL_VOCAB, MULTILINGUAL_CONFIG)
    read = read_ctc_vocab(str(directory))
    assert read["blank_id"] == 0            # `<s>`, because that is what `pad_token` names
    assert read["pieces"][1] == "<pad>"     # and the piece SPELLED `<pad>` is a different class
    assert read["word_delimiter_id"] == 4   # a literal space, not "|"


def test_a_gap_in_the_vocabulary_is_refused(tmp_path):
    """A CTC vocabulary is indexed by the head's own row number, so a hole means a row with no
    spelling. Never seen; refused rather than filled, because a placeholder would hide it."""
    directory = _write_ctc_tokenizer(tmp_path / "gap", {"a": 0, "b": 2}, ENGLISH_CONFIG)
    with pytest.raises(ValueError, match="no piece for id"):
        read_ctc_vocab(str(directory))


def test_a_token_the_vocabulary_does_not_contain_is_refused(tmp_path):
    directory = _write_ctc_tokenizer(tmp_path / "bad", ENGLISH_VOCAB,
                                     {**ENGLISH_CONFIG, "pad_token": "<nope>"})
    with pytest.raises(ValueError, match="not a piece in vocab.json"):
        read_ctc_vocab(str(directory))


def test_the_vocab_is_written_under_its_own_tag(tmp_path):
    """A tag of its own rather than a "t5" file with the delimiter rewritten to U+2581, which would
    decode correctly and answer `id_to_piece` with a character the checkpoint never had."""
    pytest.importorskip("gguf")
    from gguf import GGUFReader, GGUFWriter

    from loom_exporter.ctc_tokenizer_export import write_ctc_vocab

    directory = _write_ctc_tokenizer(tmp_path / "en", ENGLISH_VOCAB, ENGLISH_CONFIG)
    out = tmp_path / "vocab.gguf"
    writer = GGUFWriter(str(out), "ctc-vocab-test")
    write_ctc_vocab(writer, str(directory))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()

    reader = GGUFReader(str(out))
    assert reader.fields["tokenizer.ggml.model"].contents() == "ctc"
    assert reader.fields["tokenizer.ggml.padding_token_id"].contents() == 0
    assert reader.fields["tokenizer.ggml.word_delimiter_id"].contents() == 4
    tokens = [reader.fields["tokenizer.ggml.tokens"].contents(i) for i in range(7)]
    assert tokens == ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T"]
    # The delimiter is NOT a control token: it decodes to a real space, and marking it control would
    # make it the one character a transcript is missing.
    types = [reader.fields["tokenizer.ggml.token_type"].contents(i) for i in range(7)]
    assert types == [3, 1, 1, 3, 1, 1, 1]


# -- the real trace ------------------------------------------------------------------------------

def _tiny_wav2vec2(**overrides):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    return Wav2Vec2ForCTC(Wav2Vec2Config(**_TINY_CONFIG, **overrides))


def _tiny_hubert(**overrides):
    """HuBERT is here because it is structurally different where this family could have needed a
    branch and does not: `HubertModel.feature_projection` returns a bare tensor where wav2vec2's
    returns a pair, and its conv stem is layer-normed rather than group-normed."""
    from transformers import HubertConfig, HubertForCTC

    return HubertForCTC(HubertConfig(**_TINY_CONFIG, **overrides))


# Small enough to trace in a unit test and still structurally real: the grouped positional convolution
# is present (`num_conv_pos_embedding_groups` divides `hidden_size`), and the convolutional stem keeps
# its stride-5 first layer so the frame count is a real subsample of the sample count.
_TINY_CONFIG = dict(
    vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
    intermediate_size=32, conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2),
    conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
    feat_extract_norm="layer", do_stable_layer_norm=True,
)

ARCHITECTURES = {"wav2vec2": _tiny_wav2vec2, "hubert": _tiny_hubert}


def _tiny_checkpoint(tmp_path: Path, architecture: str = "wav2vec2", do_normalize: bool = True) -> Path:
    torch = pytest.importorskip("torch")

    model = ARCHITECTURES[architecture]()
    out = tmp_path / f"tiny-{architecture}-ctc"
    model.save_pretrained(out)
    (out / "preprocessor_config.json").write_text(json.dumps({
        "feature_extractor_type": "Wav2Vec2FeatureExtractor", "feature_size": 1,
        "sampling_rate": 16000, "padding_value": 0.0, "do_normalize": do_normalize,
        "return_attention_mask": True,
    }))
    _write_ctc_tokenizer(out, {**ENGLISH_VOCAB, "A": 7}, ENGLISH_CONFIG)
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    from gguf import GGUFReader

    config = ASRCtcExportConfig(architecture=None, output_path=str(out),
                                model_dir=str(checkpoint), **kwargs)
    config.task = "automatic-speech-recognition"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "topology": json.loads(reader.fields["model.graph_topology.main_topology"].contents()),
        "task": reader.fields["loom.task"].contents(),
        "input_kind": reader.fields["loom.input.kind"].contents(),
        "output_kind": reader.fields["loom.output.kind"].contents(),
        "sample_rate": reader.fields["loom.sample_rate"].contents(),
        "tokenizer": reader.fields["tokenizer.ggml.model"].contents(),
        "blank": reader.fields["tokenizer.ggml.padding_token_id"].contents(),
    }


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_a_tiny_ctc_model_exports_with_a_dynamic_sample_axis(tmp_path, architecture):
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path, architecture=architecture)
    exported = _export(checkpoint, tmp_path / "tiny.gguf")

    # ONE input, and its axis is the symbolic sample count rather than the length the trace ran at.
    # A second input here would mean a mask or a length had leaked in -- which is what makes the graph
    # static, because every route `transformers` takes to build one reads a Python-level `.shape[1]`.
    shapes = {inp["name"]: inp["shape"] for inp in exported["topology"]["inputs"]}
    assert shapes == {"waveform": ["n_samples", "1"]}

    assert exported["task"] == "automatic-speech-recognition"
    assert (exported["input_kind"], exported["output_kind"]) == ("audio", "token_ids")
    assert exported["sample_rate"] == 16000
    assert exported["tokenizer"] == "ctc"
    assert exported["blank"] == 0

    # Family 1's driver, unchanged: one call, one reduction, the collapse against the blank the
    # tokenizer named.
    assert "loom.run_subgraph_and_retain('main_topology'" in exported["driver"]
    assert "loom.argmax_rows('main_topology')" in exported["driver"]
    assert "_ctc_prev = 0" in exported["driver"]


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_the_positional_convolution_is_grouped_not_depthwise(tmp_path, architecture):
    """THE ASSERTION THIS FILE ADDS TO FAMILY 12'S SHAPE.

    `groups > 1` was read as "depthwise" for eight families, because every convolution converted before
    this one was either dense or genuinely depthwise. A grouped kernel on `CONV_1D_DW` does not compute
    a wrong answer -- it aborts the engine inside `ggml_im2col`, naming neither the op nor the model --
    so the check is on the emitted op, and it is here rather than in the engine because this is where
    the decision is made.
    """
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path, architecture=architecture)
    exported = _export(checkpoint, tmp_path / "tiny.gguf")

    convs = [node for node in exported["topology"]["nodes"] if node["op"].startswith("CONV_1D")]
    grouped = [node for node in convs if (node.get("attrs") or {}).get("groups", 1) > 1]
    assert grouped, "the positional convolution should be grouped"
    assert all(node["op"] == "CONV_1D" for node in grouped), \
        f"a grouped convolution was emitted as {sorted({n['op'] for n in grouped})}"
    assert all((node.get("attrs") or {}).get("groups") == 4 for node in grouped)
    # And the stem stays dense, so the check above is not passing by making everything grouped.
    assert any((node.get("attrs") or {}).get("groups", 1) == 1 for node in convs)


def test_the_traced_length_does_not_reach_the_graph(tmp_path):
    """The same checkpoint traced at two clip lengths must produce the same topology.

    A weaker version -- export once, check it runs -- passes against a graph with the length baked in,
    because the length it was baked at is the length it is asked for.
    """
    pytest.importorskip("coremltools")
    from loom_exporter import ctc_asr_export

    checkpoint = _tiny_checkpoint(tmp_path)
    at_1s = _export(checkpoint, tmp_path / "a.gguf")
    original = ctc_asr_export.TRACE_SECONDS
    try:
        ctc_asr_export.TRACE_SECONDS = 2.0
        at_2s = _export(checkpoint, tmp_path / "b.gguf")
    finally:
        ctc_asr_export.TRACE_SECONDS = original
    assert at_1s["topology"] == at_2s["topology"]
    assert at_1s["driver"] == at_2s["driver"]


def test_the_waveform_normalization_is_inside_the_graph(tmp_path):
    """The feature extractor's `do_normalize` is part of the MODEL here, not something a host does in
    front of it -- family 1's rule, and leaving it out feeds a correctly-shaped graph audio at the wrong
    scale, which transcribes plausible nonsense rather than failing.

    Checked by exporting the same checkpoint twice with only that flag changed: the normalizing arm has
    a MEAN reduction over the raw waveform that the other does not.
    """
    pytest.importorskip("coremltools")
    normalizing = _export(_tiny_checkpoint(tmp_path / "on", do_normalize=True),
                          tmp_path / "on.gguf")
    plain = _export(_tiny_checkpoint(tmp_path / "off", do_normalize=False), tmp_path / "off.gguf")

    def leading_ops(exported):
        return [node["op"] for node in exported["topology"]["nodes"][:7]]

    assert leading_ops(normalizing)[:4] == ["MEAN", "SUB", "SQR", "MEAN"]
    assert "MEAN" not in leading_ops(plain)


def test_weight_norm_is_removed_before_tracing(tmp_path):
    """A `weight_norm` parametrization is a forward-time recomputation, so a trace records `g * v/||v||`
    over constants where the checkpoint has one kernel. The values are identical; what differs is that
    the artifact's tensors stop corresponding to the checkpoint's."""
    torch = pytest.importorskip("torch")
    import torch.nn.utils.parametrize as parametrize

    model = _tiny_wav2vec2()
    assert any(parametrize.is_parametrized(module, "weight") for module in model.modules())
    strip_weight_norm(model)
    assert not any(parametrize.is_parametrized(module, "weight") for module in model.modules())
