"""The raw-UTF-8-byte vocabulary writer, on both of its parameterisations.

**This file exists because the ByT5 half had no test at all**, and family 10 rewrote it. ByT5 was the
only member for as long as the writer assumed its constants, so "the constants are right" and "the
writer works" were the same statement and neither was checked. Dia made them different statements: byte
0 at id 0 rather than 3, no appended eos, and two added tokens sitting *inside* the byte range.

What is asserted here is what `loom::ByteVocab` reads back, because that is the only consumer — the
three KVs it takes (`byte_offset`, `add_eos_token`, `token_type`) plus the token list itself. The
engine's own side is `tests/ci/test_byte_vocab.cpp`, and the two meet on a real checkpoint in
`tests/gate/test_e2e_dia_mil_export.cpp`.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.byt5_tokenizer_export import write_byte_vocab

_BYTE_RANGE = 256
_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4


class _FakeWriter:
    """Records what `write_byte_vocab` writes. A fake rather than a real `GGUFWriter` because what is
    under test is the VALUES, and a real writer would need a file and give them back through a reader
    that has its own opinions about types."""

    def __init__(self):
        self.kv = {}

    def add_tokenizer_model(self, name): self.kv["model"] = name
    def add_token_list(self, tokens): self.kv["tokens"] = list(tokens)
    def add_token_types(self, types): self.kv["token_type"] = list(types)
    def add_pad_token_id(self, i): self.kv["pad"] = i
    def add_eos_token_id(self, i): self.kv["eos"] = i
    def add_unk_token_id(self, i): self.kv["unk"] = i
    def add_uint32(self, key, value): self.kv[key] = value
    def add_bool(self, key, value): self.kv[key] = value


def _write(tmp_path: Path, config: dict) -> dict:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    writer = _FakeWriter()
    write_byte_vocab(writer, str(tmp_path))
    return writer.kv


def _byt5_config(n_sentinels: int = 2) -> dict:
    added = {"0": {"content": "<pad>"}, "1": {"content": "</s>"}, "2": {"content": "<unk>"}}
    for i in range(n_sentinels):
        added[str(3 + _BYTE_RANGE + i)] = {"content": f"<extra_id_{i}>"}
    return {"tokenizer_class": "ByT5Tokenizer", "added_tokens_decoder": added}


def _dia_config() -> dict:
    return {
        "tokenizer_class": "DiaTokenizer", "pad_token": "<pad>", "offset": 0,
        "added_tokens_decoder": {"0": {"content": "<pad>"}, "1": {"content": "[S1]"},
                                  "2": {"content": "[S2]"}},
    }


# -- ByT5 ------------------------------------------------------------------------------------------

def test_byt5_writes_its_own_constants_rather_than_relying_on_the_defaults(tmp_path):
    """Both values are `loom::ByteVocab`'s defaults, so an ABSENT KV and a correct one behave
    identically -- which is exactly why the value is asserted rather than the key's presence. A file
    that states its own constants is readable without knowing which family produced it."""
    kv = _write(tmp_path, _byt5_config())
    assert kv["model"] == "byt5"
    assert kv["tokenizer.ggml.byte_offset"] == 3
    assert kv["tokenizer.ggml.add_eos_token"] is True
    assert (kv["pad"], kv["eos"], kv["unk"]) == (0, 1, 2)


def test_byt5s_vocab_is_three_specials_then_the_byte_range_then_the_sentinels(tmp_path):
    kv = _write(tmp_path, _byt5_config(n_sentinels=2))
    assert len(kv["tokens"]) == 3 + _BYTE_RANGE + 2
    assert kv["tokens"][:3] == ["<pad>", "</s>", "<unk>"]
    # The byte range carries no piece text: a byte >= 0x80 cannot round-trip through a GGUF string,
    # so `ByteVocab` computes those arithmetically. See that class's own doc comment.
    assert set(kv["tokens"][3:3 + _BYTE_RANGE]) == {""}
    assert kv["tokens"][3 + _BYTE_RANGE:] == ["<extra_id_0>", "<extra_id_1>"]


def test_byt5s_specials_and_sentinels_are_typed_and_the_byte_range_is_not(tmp_path):
    """`token_type` is what makes an id ADDED, and `ByteVocab` builds its longest-match scan from it.
    Typing the byte range would make every byte an added token and the scan meaningless."""
    kv = _write(tmp_path, _byt5_config(n_sentinels=1))
    types = kv["token_type"]
    assert types[:3] == [_CONTROL] * 3
    assert set(types[3:3 + _BYTE_RANGE]) == {_NORMAL}
    assert types[3 + _BYTE_RANGE:] == [_USER_DEFINED]


def test_non_contiguous_byt5_sentinels_are_refused(tmp_path):
    config = _byt5_config(n_sentinels=0)
    config["added_tokens_decoder"]["259"] = {"content": "<extra_id_0>"}
    config["added_tokens_decoder"]["261"] = {"content": "<extra_id_1>"}
    with pytest.raises(NotImplementedError, match="not contiguous"):
        _write(tmp_path, config)


def test_a_byt5_config_missing_a_special_is_refused(tmp_path):
    """Ids 0/1/2 are hardcoded by every real `ByT5Tokenizer.__init__`, so their absence means this is
    not the layout the writer was written against -- and defaulting them would put the wrong strings
    in the file rather than fail."""
    config = _byt5_config()
    del config["added_tokens_decoder"]["1"]
    with pytest.raises(ValueError, match="ids 0/1/2"):
        _write(tmp_path, config)


# -- Dia -------------------------------------------------------------------------------------------

def test_dia_puts_byte_zero_at_id_zero_and_appends_nothing(tmp_path):
    kv = _write(tmp_path, _dia_config())
    assert kv["model"] == "byt5"
    assert kv["tokenizer.ggml.byte_offset"] == 0
    assert kv["tokenizer.ggml.add_eos_token"] is False
    assert len(kv["tokens"]) == _BYTE_RANGE


def test_dias_eos_id_is_never_written_at_all(tmp_path):
    """Not merely `add_eos_token=False`. `ByteVocab` defaults `eos_id_` to 1, which under Dia's offset
    is `[S1]` -- so a file that said "do not append" while still naming 1 as its eos would be one flag
    away from ending every prompt with a speaker tag."""
    kv = _write(tmp_path, _dia_config())
    assert "eos" not in kv
    assert kv["pad"] == 0 and kv["unk"] == 0


def test_dias_speaker_tags_are_added_tokens_inside_the_byte_range(tmp_path):
    """The property that forced `ByteVocab` to grow an added-token scan: `[S1]` is id 1, which under
    an offset of 0 is also byte 0x01. Without the typing it would encode as five literal bytes and
    decode back as a control character."""
    kv = _write(tmp_path, _dia_config())
    assert kv["tokens"][:3] == ["<pad>", "[S1]", "[S2]"]
    assert kv["token_type"][:3] == [_CONTROL, _USER_DEFINED, _USER_DEFINED]
    assert set(kv["token_type"][3:]) == {_NORMAL}
    assert set(kv["tokens"][3:]) == {""}


# -- the dispatch ----------------------------------------------------------------------------------

def test_an_unknown_tokenizer_class_is_refused_naming_both(tmp_path):
    """A third byte-level family adds a branch with its own offset, eos behaviour and added set --
    not a widening of either existing one, which is what this message says."""
    with pytest.raises(ValueError, match="ByT5Tokenizer.*DiaTokenizer"):
        _write(tmp_path, {"tokenizer_class": "SomeOtherByteTokenizer"})
