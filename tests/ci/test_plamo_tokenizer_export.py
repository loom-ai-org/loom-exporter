"""`plamo_tokenizer_export`: PLaMo's `tokenizer.jsonl` into the keys loom::PlamoVocab reads (loom.cpp ADR-132).

The rules themselves are the engine's (loom.cpp tests/ci/test_plamo_vocab.cpp, and the gate against the
reference's encodings); this pins what the file carries: the pieces by id, the reference's own score
rounding, the types as UINT8, the two break thresholds and the special ids.
"""
import json

import pytest
from gguf import GGUFReader, GGUFWriter

from loom_exporter.plamo_tokenizer_export import is_plamo_tokenizer, read_plamo_vocab, write_plamo_vocab
from loom_exporter.tokenizer_detect import detect_vocab_family


def _tokenizer_dir(tmp_path, with_bytes=True):
    rows = [["<|plamo:unk|>", 0.0, "UNKNOWN"], ["<|plamo:bos|>", 0.0, "CONTROL"], ["<|plamo:tag|>", 0.0, "CONTROL"],
            ["a", -1.23456, "NORMAL"], ["ab", -2.00004, "NORMAL"]]
    if with_bytes:
        rows += [[f"<0x{b:02X}>", 0.0, "BYTE"] for b in range(256)]
    (tmp_path / "tokenizer.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Plamo3Tokenizer", "bos_token": "<|plamo:bos|>", "eos_token": "<|plamo:bos|>",
        "unk_token": "<|plamo:unk|>", "break_around_consecutive_spaces_threshold": 2,
        "break_around_repeated_chars_threshold": 4}), encoding="utf-8")
    return tmp_path


def test_it_is_detected_by_its_config_class(tmp_path):
    d = _tokenizer_dir(tmp_path)
    assert is_plamo_tokenizer(str(d)) and detect_vocab_family(str(d)) == "plamo"


def test_scores_are_the_references_integer_rounding(tmp_path):
    v = read_plamo_vocab(str(_tokenizer_dir(tmp_path)))
    assert v["scores"][3:5] == [-12346, -20000]          # round(score * 1e4), as AhoCorasick.build
    assert v["types"][:5] == [3, 1, 1, 0, 0] and v["types"][5] == 2
    assert (v["bos"], v["eos"], v["unk"], v["break_spaces"], v["break_repeats"]) == (1, 1, 0, 2, 4)


def test_the_file_carries_what_the_engine_reads(tmp_path):
    d = _tokenizer_dir(tmp_path)
    path = tmp_path / "v.gguf"
    w = GGUFWriter(str(path), "test")
    write_plamo_vocab(w, str(d))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    fields = GGUFReader(str(path)).fields
    assert bytes(fields["tokenizer.ggml.model"].parts[-1]).decode() == "plamo"
    types = fields["tokenizer.ggml.plamo.types"]
    assert types.parts[types.data[0]].dtype.name == "uint8"    # kv_arr_u8 refuses anything else
    scores = fields["tokenizer.ggml.plamo.scores"]
    assert scores.parts[scores.data[0]].dtype.name == "int32"


def test_a_missing_byte_piece_is_refused(tmp_path):
    with pytest.raises(ValueError, match="every byte"):
        read_plamo_vocab(str(_tokenizer_dir(tmp_path, with_bytes=False)))
