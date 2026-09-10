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
    _TokenClassifierWrapper,
    _build_hf_token_classifier,
    _is_hf_token_classifier,
    _position_limit,
    _position_offset,
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


def test_a_second_architecture_is_claimed_by_the_same_recognizer(tmp_path):
    """One generic recognizer, not one per family: `*ForTokenClassification` is the checkpoint's own
    statement of which `AutoModelFor*` class it loads through, and that is the whole claim."""
    path = _hf_dir(tmp_path, "dner", {"model_type": "distilbert",
                                      "architectures": ["DistilBertForTokenClassification"]})
    assert _is_hf_token_classifier(path)
    assert default_registry().detect(path).name == "hf-token-classifier"


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

_LABELS = {"id2label": {0: "O", 1: "B-PER", 2: "I-PER"},
           "label2id": {"O": 0, "B-PER": 1, "I-PER": 2}}


def _tiny_bert(max_position_embeddings: int):
    from transformers import BertConfig, BertForTokenClassification

    return BertForTokenClassification(BertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=max_position_embeddings, **_LABELS))


def _tiny_distilbert(max_position_embeddings: int):
    """DistilBERT is in this file because it is structurally different in the THREE ways this family
    has to absorb, not because it is a second name: no token-type embeddings, no `position_ids`
    argument (its `Embeddings.forward` slices a registered buffer at the traced length instead), and
    `.transformer` where BERT has `.encoder`. A template that only ever saw BERT would pass every test
    above and fail on the first of these."""
    from transformers import DistilBertConfig, DistilBertForTokenClassification

    return DistilBertForTokenClassification(DistilBertConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, hidden_dim=64,
        max_position_embeddings=max_position_embeddings, **_LABELS))


def _tiny_xlm_roberta(max_position_embeddings: int):
    """The fairseq-derived shape, and the third structural difference this family absorbs: its position
    table declares a `padding_idx`, so position 0 is table ROW 2 and the first two rows are never
    addressed. BERT and DistilBERT number from zero; a template that only saw those two passes every
    other test in this file and reads the wrong two rows here."""
    from transformers import XLMRobertaConfig, XLMRobertaForTokenClassification

    return XLMRobertaForTokenClassification(XLMRobertaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=max_position_embeddings,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, type_vocab_size=1, **_LABELS))


ARCHITECTURES = {"bert": _tiny_bert, "distilbert": _tiny_distilbert,
                 "xlm-roberta": _tiny_xlm_roberta}

# The two architectures whose checkpoints are the CLASSIC BERT layout (`vocab.txt`, no fast tokenizer),
# which is what `_tiny_checkpoint` writes. XLM-R is not one of them -- its vocabulary is a SentencePiece
# protobuf, exercised against a real one in `test_spm_tokenizer_export.py` and against the real
# checkpoint in the gate sweep -- so the export-shaped tests below parametrize over these and the
# offset tests over all three.
WORDPIECE_ARCHITECTURES = ["bert", "distilbert"]


def _tiny_checkpoint(tmp_path: Path, max_position_embeddings: int = 64,
                     architecture: str = "bert") -> Path:
    """A real, randomly-initialised token classifier, small enough to trace in a unit test.

    Real rather than mocked because what is under test is the interaction with `transformers`' own
    forward pass -- a stub would have none of the mask/`token_type_ids`/`position_ids` machinery this
    family exists to route around, so it could not fail.
    """
    torch = pytest.importorskip("torch")

    model = ARCHITECTURES[architecture](max_position_embeddings)
    out = tmp_path / f"tiny-{architecture}-ner"
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


@pytest.mark.parametrize("architecture", WORDPIECE_ARCHITECTURES)
def test_a_tiny_token_classifier_exports_with_a_dynamic_token_axis(tmp_path, architecture):
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path, architecture=architecture)
    exported = _export(checkpoint, tmp_path / "tiny.gguf", seq_len=8)

    # THE ASSERTION THIS FILE EXISTS FOR. Both declared inputs carry the symbolic token axis, not the
    # 8 the trace ran at -- a literal here is the baked length, and it is invisible at seq_len=8.
    #
    # The SAME two inputs for both architectures, which is the claim that matters more than either of
    # them: DistilBERT reaches them through a pre-hook on its position table and BERT through a plain
    # kwarg, and the artifact cannot tell. A third input, or a missing `position_ids`, would mean the
    # difference had leaked out of the exporter and into the file.
    shapes = {inp["name"]: inp["shape"] for inp in exported["topology"]["inputs"]}
    assert shapes == {"tokens": ["n_tokens", "1"], "position_ids": ["n_tokens"]}

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


@pytest.mark.parametrize("architecture", WORDPIECE_ARCHITECTURES)
def test_the_traced_length_does_not_reach_the_graph(tmp_path, architecture):
    """The same checkpoint at two trace lengths must produce the same topology.

    A weaker version of this test -- export once, check it runs -- passes against a graph with the
    length baked in, because the shape it was baked at is the shape it is asked for. Two exports whose
    only difference is `seq_len` is what turns that into a real check.
    """
    pytest.importorskip("coremltools")
    checkpoint = _tiny_checkpoint(tmp_path, architecture=architecture)
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


# -- where position 0 lives ------------------------------------------------------------------------

@pytest.mark.parametrize("architecture,expected", [("bert", 0), ("distilbert", 0), ("xlm-roberta", 2)])
def test_the_position_offset_is_read_off_the_table_not_the_name(architecture, expected):
    """`nn.Embedding`'s own `padding_idx`, which the fairseq family sets and BERT/DistilBERT do not.

    A structural read for the same reason `_accepts` is one: the alternative is a list of `model_type`
    strings to keep in step with `transformers`.
    """
    pytest.importorskip("torch")
    model = ARCHITECTURES[architecture](64)
    assert _position_offset(model.base_model) == expected


@pytest.mark.parametrize("architecture,expected", [("bert", 64), ("distilbert", 64),
                                                   ("xlm-roberta", 62)])
def test_the_offset_shortens_the_length_the_family_can_serve(architecture, expected):
    """`max_position_embeddings` is the SIZE OF THE TABLE, not the length it can answer.

    Real numbers rather than this fixture's: XLM-R ships a 514-row table and serves 512 tokens. Reading
    the declared number as the cap would let a 514-token sequence index two rows past the end.
    """
    pytest.importorskip("torch")
    assert _position_limit(ARCHITECTURES[architecture](64)) == expected


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_the_wrapper_reproduces_the_models_own_forward(architecture):
    """THE ASSERTION THE OFFSET EXISTS FOR, and the one a token-level oracle would not have made.

    The wrapper is handed 0-based `position_ids` -- `loom.range(0, n_tokens)`, the driver's one contract
    for the whole family -- and has to produce what the model produces when it computes its own. On the
    real checkpoint, passing 0-based ids to XLM-R moves the logits by up to 7.0 while 24 of 27 argmaxes
    still agree, so this compares the TENSOR ([[feedback-tensor-oracle-not-token-oracle]]).
    """
    torch = pytest.importorskip("torch")
    model = ARCHITECTURES[architecture](64).eval()
    tokens = torch.arange(5, 17, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        # No `position_ids` and no mask: the model computes both, which is the behaviour being matched.
        expected = model(input_ids=tokens).logits
        got = _TokenClassifierWrapper(model)(tokens, torch.arange(tokens.shape[1], dtype=torch.long))
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()
