"""Family 9's third leaf (P5): F5-TTS -- a flow-matching mel decoder that clones a voice by in-filling.

Two things are this family's own and both are tested here rather than inferred from a shape:

* **the text front end's BOUNDARY.** `convert_char_to_pinyin` is `rjieba` + `pypinyin` before a single
  id is looked up, and what ships in the GGUF is the character table. The claim that a codepoint scan
  reproduces that function for ordinary prose is a MEASUREMENT, so it is measured -- against the real
  function when the F5-TTS checkout is importable, and skipped (not asserted around) when it is not.
* **the declaration that the sampler is guided and caller-scheduled.** The Lua is generated, so a
  wrong declaration is a driver that integrates the wrong field with nothing failing; the export's own
  spec links are what catch it, and they only run if the spec says what this model does.

Everything else here is either inherited (`FlowMatchingSpec`, covered by `test_flow_matching_export`)
or exercised end to end by the gate.
"""
import json
import struct
import sys
from pathlib import Path

import pytest

from loom_exporter.f5_tokenizer_export import f5_vocab_size, read_f5_vocab, write_f5_vocab
from loom_exporter.f5_tts_export import (
    DEFAULT_CFG, DEFAULT_STEPS, DEFAULT_SWAY, F5TTSExportConfig, VOCODER_SUBDIR, _build_f5_tts,
    _find_checkpoint, _is_f5_tts,
)
from loom_exporter.registry import default_registry

DIT_NAMES = [
    "ema_model.transformer.transformer_blocks.0.attn.to_q.weight",
    "ema_model.transformer.text_embed.text_embed.weight",
]


def _safetensors(path: Path, names) -> None:
    """A safetensors file with a real header and no tensor data. `_is_f5_tts` reads the header only --
    the point of doing it that way is not loading a gigabyte to answer a detection question."""
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0\0\0\0")


def _release(tmp_path: Path, name="F5TTS", *, names=None, vocoder=True, vocab=True) -> Path:
    d = tmp_path / name
    d.mkdir()
    if vocab:
        (d / "vocab.txt").write_text(" \n!\n\"\na\nan1\n")
    _safetensors(d / "model_1250000.safetensors", names if names is not None else DIT_NAMES)
    if vocoder:
        v = d / VOCODER_SUBDIR
        v.mkdir()
        (v / "config.yaml").write_text("feature_extractor:\n")
        (v / "pytorch_model.bin").write_bytes(b"")
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_release_directory_is_claimed(tmp_path):
    assert _is_f5_tts(_release(tmp_path))


def test_a_directory_without_the_vocoder_is_not_claimed(tmp_path):
    """The vocoder is part of the CHECK rather than of the failure.

    F5-TTS ships none -- the release is the DiT and a `vocab.txt` -- so a directory without an
    assembled `vocos-mel-24khz/` is one this export cannot finish. Claiming it would turn a missing
    file into a traceback two minutes into a conversion.
    """
    assert not _is_f5_tts(_release(tmp_path, vocoder=False))


def test_a_checkpoint_whose_tensors_are_not_the_dits_is_not_claimed(tmp_path):
    assert not _is_f5_tts(_release(tmp_path, names=["model.layers.0.self_attn.q_proj.weight"]))


def test_a_directory_without_a_vocab_is_not_claimed(tmp_path):
    assert not _is_f5_tts(_release(tmp_path, vocab=False))


def test_a_truncated_safetensors_header_is_declined_rather_than_raising(tmp_path):
    d = _release(tmp_path)
    (d / "model_1250000.safetensors").write_bytes(b"\x01\x02\x03")
    assert not _is_f5_tts(d)


def test_two_weight_files_are_refused_by_name(tmp_path):
    """Which checkpoint an export converted must not be a guess."""
    d = _release(tmp_path)
    _safetensors(d / "second.safetensors", DIT_NAMES)
    with pytest.raises(ValueError, match="exactly one"):
        _find_checkpoint(d)


def test_the_registry_routes_a_release_to_this_recognizer(tmp_path):
    recognizer = default_registry().detect(_release(tmp_path))
    assert recognizer.name == "f5-tts"


def test_the_vocoder_defaults_into_the_model_directory(tmp_path):
    d = _release(tmp_path)
    config = _build_f5_tts(d, str(tmp_path / "out.gguf"))
    assert Path(config.vocoder_dir) == d / VOCODER_SUBDIR


# -- the vocabulary --------------------------------------------------------------------------------

def test_the_vocab_is_read_the_way_get_tokenizer_reads_it(tmp_path):
    """`char[:-1]` per line, id = line number. The COUNT sizes the embedding, so it is the same
    reader on both sides rather than two that must agree."""
    d = tmp_path / "v"
    d.mkdir()
    (d / "vocab.txt").write_text(" \n!\nzhong1\n")
    assert read_f5_vocab(d) == [" ", "!", "zhong1"]
    assert f5_vocab_size(d) == 3


def test_a_repeated_piece_collapses_in_the_size(tmp_path):
    """The reference builds a DICT, so `len()` is of the map and not of the file."""
    d = tmp_path / "v"
    d.mkdir()
    (d / "vocab.txt").write_text("a\nb\na\n")
    assert len(read_f5_vocab(d)) == 3
    assert f5_vocab_size(d) == 2


def test_the_table_is_written_under_its_own_tag(tmp_path):
    from gguf import GGUFWriter

    d = tmp_path / "v"
    d.mkdir()
    (d / "vocab.txt").write_text(" \n!\nan1\n")
    out = tmp_path / "t.gguf"
    w = GGUFWriter(str(out), "f5-test")
    write_f5_vocab(w, str(d))
    w.add_tensor("x", __import__("numpy").zeros(1, dtype="float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    from gguf import GGUFReader

    r = GGUFReader(str(out))
    fields = {f.name: f for f in r.fields.values()}
    assert bytes(fields["tokenizer.ggml.model"].parts[-1]).decode() == "f5"
    # The filler offset is a property of the VOCABULARY's relationship to the embedding, not of the
    # sampler, so it travels in the file rather than only in the driver.
    assert int(fields["tokenizer.ggml.f5.filler_offset"].parts[-1][0]) == 1
    assert int(fields["tokenizer.ggml.unknown_token_id"].parts[-1][0]) == 0


# -- the sampler declaration -----------------------------------------------------------------------

def _config(tmp_path):
    return F5TTSExportConfig(output_path=str(tmp_path / "o.gguf"),
                             model_dir=str(_release(tmp_path)))


def test_the_sampler_declares_guidance_and_a_caller_schedule(tmp_path):
    """Both are what the Lua is GENERATED from, so a wrong declaration is a silent wrong model:
    unguided, F5-TTS's velocity field is not the one it was trained to integrate, and on a uniform
    schedule it spends its steps in the wrong places."""
    spec, = _config(tmp_path).samplers()
    assert spec.guidance is True
    assert spec.schedule == "caller"
    assert spec.method == "euler"
    assert spec.carried_input == "x"
    assert spec.fixed_inputs == ["cond", "text_embed"]


def test_the_contract_declares_a_text_door_and_this_models_own_defaults(tmp_path):
    contract = _config(tmp_path).contract()
    # The `text-to-speech` default is `phoneme_ids`; this family encodes graphemes itself, and
    # declaring the task default would close the door the model actually has (Supertonic's finding).
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == 24000
    assert contract["tts.default_steps"] == DEFAULT_STEPS


def test_the_sample_rate_is_declared_once(tmp_path):
    """It used to be in `hparams()` as well, which made the writer log a duplicate-key overwrite."""
    assert "sample_rate" not in _config(tmp_path).hparams()


def test_the_defaults_are_the_references_own(tmp_path):
    assert (DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_SWAY) == (32, 2.0, -1.0)


# -- the layout the phases hand each other ---------------------------------------------------------

MODEL_DIR = Path("/home/flavio/Dev/models/f5-tts/F5TTS_v1_Base")


def _phases():
    if not _is_f5_tts(MODEL_DIR):
        pytest.skip("no assembled F5-TTS checkpoint to trace")
    return {p.name: p for p in F5TTSExportConfig(
        output_path="/dev/null", model_dir=str(MODEL_DIR)).phases()}


def test_the_vocoder_takes_the_layout_the_estimator_produces():
    """**The join, which no per-phase check can see.**

    Every graph in this export graded clean against torch -- mel bit-identical, both text-embedding
    branches bit-identical, the estimator at cosine 1.000000000, the vocoder at 1.5e-05 -- and the
    audio was "(chimes ringing)". The estimator retains FRAME-major mel (`ne = [100, n_tokens]`) and
    `Vocos.decode`'s own convention is channel-major, so the driver was handing one straight to the
    other and the vocoder was reading a transposed spectrogram. A frame-major array read as
    channel-major is still a plausible spectrogram, which is why nothing raised.

    The fix is that the CONSUMER transposes, inside its own graph. This pins it as a relationship
    between two declarations rather than as a fact about one, because that is what was wrong.
    """
    phases = _phases()
    # `symbolic_shape` renders a RangeDim as its symbol, which is what makes two declarations
    # comparable at all -- two RangeDims with identical bounds are not `==`.
    est = {i.name: list(i.shape.symbolic_shape) for i in phases["estimator"].mil_inputs}
    voc = {i.name: list(i.shape.symbolic_shape) for i in phases["vocoder"].mil_inputs}
    # Torch order, (batch, frames, channels) for both: the estimator's `x` is the state the ODE
    # integrates and the vocoder's `mel` is a slice of it, so the CHANNEL axis has to be last on both
    # and the dynamic frame axis in the middle. Channel-major would put the 100 at index 1.
    assert len(est["x"]) == 3 and len(voc["mel"]) == 3, (est["x"], voc["mel"])
    assert est["x"][2] == 100 and voc["mel"][2] == 100, (est["x"], voc["mel"])
    assert voc["mel"][1] != 100, "the vocoder's mel went back to channel-major"
    # The frame axis is symbolic on both -- a literal there would mean one of them got pinned to a
    # traced length.
    assert not isinstance(est["x"][1], int) and not isinstance(voc["mel"][1], int)


def test_the_estimators_state_and_conditioning_share_one_layout():
    """`cond` is a slice of the mel front end's output and `x` is what the ODE carries; the sampler
    hands both to the same graph, so a disagreement would be silent in exactly the same way."""
    phases = _phases()
    shapes = {i.name: list(i.shape.symbolic_shape) for i in phases["estimator"].mil_inputs}
    assert shapes["x"] == shapes["cond"]
    # And the text embedding rides the same frame axis, at its own width.
    assert shapes["text_embed"][1] == shapes["x"][1]
    assert shapes["text_embed"][2] == 512


# -- the front-end boundary, measured against the real function ------------------------------------

F5_SRC = Path("/home/flavio/Dev/F5-TTS/src")


def _convert_char_to_pinyin():
    if not F5_SRC.is_dir():
        pytest.skip("no F5-TTS checkout to compare against")
    import types

    sys.modules.setdefault("f5_tts.model.trainer",
                           types.ModuleType("f5_tts.model.trainer"))
    sys.modules["f5_tts.model.trainer"].Trainer = object
    if str(F5_SRC) not in sys.path:
        sys.path.insert(0, str(F5_SRC))
    try:
        from f5_tts.model.utils import convert_char_to_pinyin
    except ImportError as exc:          # rjieba/pypinyin absent
        pytest.skip(f"F5-TTS's text utils are not importable: {exc}")
    return convert_char_to_pinyin


TRANS = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

ORDINARY = [
    "Some call me nature, others call me mother nature.",
    "I don't really care what you call me.",
    "The quick brown fox jumps over the lazy dog.",
    "Wait, what? No! Really?",
    "She said “hello” and left.",
    "It’s fine; really.",
    "one, two, three and four",
]

DIVERGENT = ["the end...", "dated 2026-09-18 exactly"]


@pytest.mark.parametrize("text", ORDINARY)
def test_a_codepoint_scan_IS_the_reference_for_ordinary_prose(text):
    """The claim `F5Vocab` is built on, checked against the function itself.

    Ordinary prose means words already separated by spaces -- which is when jieba's segmentation
    inserts nothing, so the reference is the identity on characters (after `custom_trans`).
    """
    assert _convert_char_to_pinyin()([text])[0] == list(text.translate(TRANS))


@pytest.mark.parametrize("text", DIVERGENT)
def test_the_two_divergent_classes_are_a_single_inserted_space(text):
    """The boundary, stated as a test rather than as a comment.

    A multi-character punctuation run and a hyphen-joined digit group are both jieba HAN blocks, and
    the reference inserts a space before them. A table cannot: the decision comes from a
    dictionary-and-HMM DAG. What is worth pinning is that the difference is exactly that -- one added
    space, always in the same direction -- so a host reading the docs knows what it is trading.
    """
    want = "".join(_convert_char_to_pinyin()([text])[0])
    got = "".join(list(text.translate(TRANS)))
    assert want != got
    assert want.replace(" ", "") == got.replace(" ", "")
    assert len(want) == len(got) + 1
