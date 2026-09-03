"""Family 12 (P5): BERT-family token classifiers, `text` in and one declared class per token out.

The end-to-end test here builds a REAL `BertForTokenClassification` from a tiny config rather than
mocking one, and traces it through the real compiler. That is not thoroughness for its own sake: the
whole of what this family had to solve is that three separate things in `transformers`' forward pass
bake the traced sequence length into the graph (the attention mask, `token_type_ids`' buffer slice, and
`position_ids`), and every one of them is silently harmless at the traced length. A test that exports at
one length and never checks the graph's own axis would pass against exactly the bug this family exists
to avoid -- so the assertion is on the emitted topology's declared shapes, and there is a second export
at a different length to make "it was baked" impossible to miss.

The detection half needs no torch at all and is where a new checkpoint's first failure shows up.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.export_config import LoomExportConfig
from loom_exporter.registry import default_registry
from loom_exporter.tasks import task_spec
from loom_exporter.token_classification_export import (
    TokenClassificationExportConfig,
    _build_hf_token_classifier,
    _is_hf_token_classifier,
    _read_labels,
)


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection -----------------------------------------------------------------------------------

def test_a_token_classifier_directory_is_claimed(tmp_path):
    path = _hf_dir(tmp_path, "ner", {"model_type": "bert",
                                     "architectures": ["BertForTokenClassification"]})
    assert _is_hf_token_classifier(path)


def test_the_architecture_half_is_load_bearing(tmp_path):
    """`model_type` alone is not enough, and this is the check that keeps this family from claiming
    every other family's checkpoints -- `TaskRegistry.detect` runs every recognizer against every path,
    and Whisper, Parakeet and the causal LMs all declare a `model_type` too."""
    assert not _is_hf_token_classifier(_hf_dir(tmp_path, "bare", {"model_type": "bert"}))
    assert not _is_hf_token_classifier(
        _hf_dir(tmp_path, "lm", {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    assert not _is_hf_token_classifier(
        _hf_dir(tmp_path, "seq", {"model_type": "bert",
                                  "architectures": ["BertForSequenceClassification"]}))


def test_a_directory_that_is_not_an_hf_checkpoint_is_a_no_not_an_error(tmp_path):
    """`detect()` runs against unidentified paths by construction."""
    assert not _is_hf_token_classifier(tmp_path / "nothing-here")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{not json")
    assert not _is_hf_token_classifier(tmp_path / "broken")


def test_the_registry_resolves_a_synthetic_checkpoint(tmp_path):
    path = _hf_dir(tmp_path, "ner", {"model_type": "bert",
                                     "architectures": ["BertForTokenClassification"]})
    recognizer = default_registry().detect(path)
    assert recognizer.name == "hf-token-classifier"
    assert recognizer.task == "token-classification"


def test_the_recognizer_is_a_fallback():
    """Generic by construction, so adding a specific recognizer later cannot make this ambiguous."""
    entry = default_registry()._entries["token-classification"]
    assert [r.fallback for r in entry.recognizers] == [True]


# -- the task, and what the file declares ---------------------------------------------------------

def test_the_task_declares_this_familys_base_config():
    assert task_spec("token-classification").base_config_class() is TokenClassificationExportConfig
    assert not task_spec("token-classification").reserved


def test_the_modality_pair_is_the_first_non_audio_one():
    config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)
    config.task = "token-classification"
    assert config.contract() == {"task": "token-classification", "input.kind": "text",
                                 "output.kind": "class"}


def test_labels_are_read_off_the_checkpoint_not_declared(tmp_path):
    class _Config:
        id2label = {"0": "O", "1": "B-PER"}

    assert _read_labels(_Config()) == ["O", "B-PER"]


def test_a_gap_in_id2label_keeps_its_id_rather_than_becoming_a_blank():
    """Two unnamed classes must stay distinguishable; empty strings would collide on lookup."""
    class _Config:
        id2label = {0: "O", 2: "B-PER"}

    assert _read_labels(_Config()) == ["O", "LABEL_1", "B-PER"]


def test_a_checkpoint_naming_no_classes_declares_no_labels():
    class _Config:
        id2label = {}

    assert _read_labels(_Config()) == []


def test_the_driver_builder_is_named_by_the_family(tmp_path):
    """The second family to override `synthesized_builder_key`, for the reason the first one records:
    this is a `Flattened` export like Qwen3, and what differs is what the host does with the output."""
    config = _build_hf_token_classifier(tmp_path, "/tmp/x.gguf")
    assert config.synthesized_builder_key() == "TokenLabels"
    assert config.backend_kwargs()["driver_builder"] == "TokenLabels"


# -- the real trace ------------------------------------------------------------------------------

def _tiny_checkpoint(tmp_path: Path, max_position_embeddings: int = 64) -> Path:
    """A real, randomly-initialised `BertForTokenClassification`, small enough to trace in a unit test.

    Real rather than mocked because what is under test is the interaction with `transformers`' own
    forward pass -- a stub would have none of the mask/`token_type_ids`/`position_ids` machinery this
    family exists to route around, so it could not fail.
    """
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForTokenClassification

    config = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=max_position_embeddings,
                        id2label={0: "O", 1: "B-PER", 2: "I-PER"},
                        label2id={"O": 0, "B-PER": 1, "I-PER": 2})
    model = BertForTokenClassification(config)
    out = tmp_path / "tiny-ner"
    model.save_pretrained(out)
    # A `vocab.txt` and a `special_tokens_map.json`, which is the CLASSIC BERT layout -- no
    # `tokenizer.json`, because several of this family's real checkpoints predate the fast tokenizer
    # and ship exactly this (dslim/bert-base-NER). It is also the one path the export takes that a
    # tokenizer-less fixture would leave untested.
    pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(59)]
    (out / "vocab.txt").write_text("\n".join(pieces) + "\n")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
        "cls_token": "[CLS]", "mask_token": "[MASK]",
    }))
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    """Exports and returns the GGUF's driver text and main topology, parsed."""
    from gguf import GGUFReader

    config = TokenClassificationExportConfig(architecture=None, output_path=str(out),
                                             model_dir=str(checkpoint), **kwargs)
    config.task = "token-classification"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "topology": json.loads(reader.fields["model.graph_topology.main_topology"].contents()),
        "labels": reader.fields["loom.labels"].contents(),
        "task": reader.fields["loom.task"].contents(),
        "output_kind": reader.fields["loom.output.kind"].contents(),
        "tokenizer": reader.fields["tokenizer.ggml.model"].contents(),
        "cls_id": reader.fields["tokenizer.ggml.bos_token_id"].contents(),
        "sep_id": reader.fields["tokenizer.ggml.seperator_token_id"].contents(),
    }


def test_a_tiny_token_classifier_exports_with_a_dynamic_token_axis(tmp_path):
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path)
    exported = _export(checkpoint, tmp_path / "tiny.gguf", seq_len=8)

    # THE ASSERTION THIS FILE EXISTS FOR. Both declared inputs carry the symbolic token axis, not the
    # 8 the trace ran at -- a literal here is the baked length, and it is invisible at seq_len=8.
    shapes = {inp["name"]: inp["shape"] for inp in exported["topology"]["inputs"]}
    assert shapes["tokens"] == ["n_tokens", "1"]
    assert shapes["position_ids"] == ["n_tokens"]
    # And no mask input at all: a single unpadded sequence needs none, and every route transformers
    # takes to build one bakes the length (see the module docstring).
    assert "attention_mask" not in shapes

    assert exported["task"] == "token-classification"
    assert exported["output_kind"] == "class"
    assert list(exported["labels"]) == ["O", "B-PER", "I-PER"]

    # The driver is the whole template: read the input, fill in the positions, one call, one reduction.
    assert "loom.argmax_rows('main_topology')" in exported["driver"]
    assert "loom.run_subgraph_and_retain('main_topology'" in exported["driver"]

    # The vocabulary came from `vocab.txt` alone, and the framing ids reached the file -- which is what
    # `loom::text::classify` strips on, so a checkpoint of this vintage silently missing them would
    # label [CLS] and [SEP] and nothing would report it.
    assert exported["tokenizer"] == "bert"
    assert exported["cls_id"] == 2 and exported["sep_id"] == 3


def test_the_traced_length_does_not_reach_the_graph(tmp_path):
    """The same checkpoint at two trace lengths must produce the same topology.

    A weaker version of this test -- export once, check it runs -- passes against a graph with the
    length baked in, because the shape it was baked at is the shape it is asked for. Two exports whose
    only difference is `seq_len` is what turns that into a real check.
    """
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path)
    at_8 = _export(checkpoint, tmp_path / "a.gguf", seq_len=8)
    at_32 = _export(checkpoint, tmp_path / "b.gguf", seq_len=32)
    assert at_8["topology"] == at_32["topology"]
    assert at_8["driver"] == at_32["driver"]


def test_a_range_longer_than_the_position_table_is_refused(tmp_path):
    """A learned position table has no rows past `max_position_embeddings`, so a longer range would
    index off the end of it rather than extrapolate -- unlike a RoPE decoder, where asking for more
    than the checkpoint declares is legitimate."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path, max_position_embeddings=64)
    config = TokenClassificationExportConfig(architecture=None, output_path=str(tmp_path / "x.gguf"),
                                             model_dir=str(checkpoint), max_seq_len=512)
    config.task = "token-classification"
    with pytest.raises(ValueError, match="max_position_embeddings"):
        config.export()
