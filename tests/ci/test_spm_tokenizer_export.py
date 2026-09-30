"""The SentencePiece vocab writer, and the one thing a protobuf cannot tell you: what a piece's ID is.

Family 12's third checkpoint (P5) is where this stopped being theoretical. Every SentencePiece caller
before it was a NeMo ASR model, whose `.nemo` archive holds a bare `tokenizer.model` and whose ids are
the protobuf's own piece order. The fairseq-derived family -- XLM-R, and RoBERTa-style Unigram models
generally -- is not: `transformers`' converter drops the proto's leading `<unk>/<s>/</s>`, prepends
`<s>/<pad>/</s>/<unk>` at 0..3, shifts every remaining piece by one and appends `<mask>`. Writing the
proto verbatim for such a checkpoint yields a vocabulary that is off by one for every piece, indexes the
embedding table wrongly, and still loads, encodes and decodes to plausible text.

The fixtures here are REAL `ModelProto`s, built field by field rather than mocked, because what is under
test is the reading of one -- and the fairseq-shaped one is built to the same recipe the real converter
uses, so the remapping it exercises is the one that actually happens.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("sentencepiece")

from sentencepiece import sentencepiece_model_pb2 as spm_pb2  # noqa: E402

from loom_exporter.spm_tokenizer_export import (  # noqa: E402
    HfIdLayout,
    read_hf_id_layout,
    write_sentencepiece_vocab,
)

# The three pieces every SentencePiece protobuf opens with, and the reason the fairseq layout exists:
# its converter throws these away and re-declares four of its own, in a different order.
_PROTO_SPECIALS = [("<unk>", 0.0, 2), ("<s>", 0.0, 3), ("</s>", 0.0, 3)]
_PROTO_PIECES = [(",", -3.5, 1), ("▁the", -4.0, 1), ("s", -5.0, 1), ("<0x41>", 0.0, 6)]


def _proto_bytes(*, unigram: bool = True) -> bytes:
    m = spm_pb2.ModelProto()
    m.trainer_spec.model_type = (m.trainer_spec.UNIGRAM if unigram else m.trainer_spec.BPE)
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.remove_extra_whitespaces = True
    m.normalizer_spec.precompiled_charsmap = b"\x00\x00\x00\x00charsmap"
    for piece, score, kind in _PROTO_SPECIALS + _PROTO_PIECES:
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = piece, score, kind
    return m.SerializeToString()


def _fairseq_dir(tmp_path: Path) -> Path:
    """A checkpoint in the shape XLM-R ships: the protobuf under fairseq's own name for it, plus the
    `tokenizer.json` whose vocab is the converter's OUTPUT and therefore the id authority."""
    d = tmp_path / "xlmr"
    d.mkdir()
    (d / "sentencepiece.bpe.model").write_bytes(_proto_bytes())
    vocab = ([["<s>", 0.0], ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0]]
             + [[piece, score] for piece, score, _ in _PROTO_PIECES]
             + [["<mask>", 0.0]])
    (d / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "Unigram", "vocab": vocab},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        },
    }))
    (d / "special_tokens_map.json").write_text(json.dumps({
        "unk_token": "<unk>", "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>",
    }))
    return d


class _RecordingWriter:
    """Every `add_*` the writer calls, as `{key: value}`. A stand-in for `GGUFWriter` because what is
    under test is WHICH KVs get written with WHICH values, and a real writer would answer that only
    after a file round-trip."""

    def __init__(self):
        self.kv = {}

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)
        return lambda *args: self.kv.__setitem__(name[len("add_"):], args[0] if len(args) == 1 else args)


def _written(tokenizer_dir=None, **kwargs) -> dict:
    writer = _RecordingWriter()
    hf_ids = read_hf_id_layout(tokenizer_dir) if tokenizer_dir is not None else None
    write_sentencepiece_vocab(writer, _proto_bytes(), hf_ids=hf_ids, **kwargs)
    return writer.kv


# -- reading the id authority ----------------------------------------------------------------------

def test_a_directory_with_no_fast_tokenizer_has_no_second_authority(tmp_path):
    """Every SentencePiece caller that predates this one, and the reason they are unaffected: a NeMo
    archive holds a bare `tokenizer.model`, so there is nothing to read and the protobuf stands."""
    d = tmp_path / "nemo"
    d.mkdir()
    (d / "tokenizer.model").write_bytes(_proto_bytes())
    assert read_hf_id_layout(d) is None


def test_a_non_unigram_tokenizer_json_is_not_an_id_authority(tmp_path):
    """A BPE `tokenizer.json` has a `vocab` too, and it is a dict of piece->id rather than an
    id-ordered list of `[piece, score]` pairs. Returning None is what keeps this path off it."""
    d = tmp_path / "bpe"
    d.mkdir()
    (d / "tokenizer.model").write_bytes(_proto_bytes(unigram=False))
    (d / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}}}))
    assert read_hf_id_layout(d) is None


def test_the_fairseq_layout_is_read_in_id_order(tmp_path):
    layout = read_hf_id_layout(_fairseq_dir(tmp_path))
    assert layout.pieces == ["<s>", "<pad>", "</s>", "<unk>", ",", "▁the", "s", "<0x41>", "<mask>"]
    assert (layout.bos_id, layout.eos_id, layout.unk_id, layout.pad_id) == (0, 2, 3, 1)


def test_the_framing_comes_from_the_post_processor_not_the_special_token_map(tmp_path):
    """`special_tokens_map.json` names a `bos_token` for tokenizers that never add one -- T5's own map
    names `</s>` while its encode adds it and nothing else. What decides is the post-processor, which
    is the encode written down, so a checkpoint with a special-token map and no template frames with
    nothing."""
    d = _fairseq_dir(tmp_path)
    tokenizer_json = json.loads((d / "tokenizer.json").read_text())
    del tokenizer_json["post_processor"]
    (d / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    layout = read_hf_id_layout(d)
    assert (layout.bos_id, layout.eos_id) == (None, None)
    assert (layout.unk_id, layout.pad_id) == (3, 1)   # still named, still read


# -- writing it ------------------------------------------------------------------------------------

def test_without_an_id_authority_the_protobuf_is_written_verbatim():
    """The pre-existing behaviour, asserted so a change to the new branch cannot quietly move it: this
    is what every NeMo model in the sweep gets, and its artifact is recorded in the baseline."""
    kv = _written()
    assert kv["token_list"] == ["<unk>", "<s>", "</s>", ",", "▁the", "s", "<0x41>"]
    assert kv["token_types"] == [2, 3, 3, 1, 1, 1, 6]
    assert kv["unk_token_id"] == 0
    assert "bos_token_id" not in kv and "eos_token_id" not in kv
    assert "add_bos_token" not in kv and "add_eos_token" not in kv
    assert "pad_token_id" not in kv


def test_the_fairseq_ids_replace_the_protobufs_own(tmp_path):
    """THE TEST THIS FILE EXISTS FOR. Every piece the proto and the fast tokenizer share moves by one,
    and reading the proto's order instead would put `,` at 3 where the model has it at 4."""
    kv = _written(_fairseq_dir(tmp_path))
    assert kv["token_list"] == ["<s>", "<pad>", "</s>", "<unk>", ",", "▁the", "s", "<0x41>", "<mask>"]
    assert kv["token_list"].index(",") == 4
    assert kv["unk_token_id"] == 3


def test_the_types_still_come_from_the_protobuf(tmp_path):
    """The one thing `tokenizer.json` does not carry, and it is load-bearing: `loom::Vocab` puts only
    NORMAL/USER_DEFINED/UNUSED pieces in its match trie, so a `<0x41>` demoted to NORMAL would start
    matching the letter A in ordinary text, and a `<mask>` written NORMAL would be emitted by an encode
    of the literal string."""
    kv = _written(_fairseq_dir(tmp_path))
    types = dict(zip(kv["token_list"], kv["token_types"]))
    assert types["<0x41>"] == 6                        # BYTE, carried across the remap
    assert types[","] == 1 and types["▁the"] == 1  # NORMAL
    assert types["<unk>"] == 2                          # UNKNOWN, from the proto's own entry
    assert types["<s>"] == 3 and types["</s>"] == 3     # CONTROL, likewise
    # The two the converter INVENTED. The protobuf has no entry to copy, and CONTROL is the answer that
    # keeps them out of the trie -- the alternative default (NORMAL) is a silent encode bug.
    assert types["<pad>"] == 3 and types["<mask>"] == 3


def test_the_framing_reaches_the_file(tmp_path):
    """`<s> ... </s>`, which is what XLM-R's encode does and what `loom::text::classify` strips back
    off. A file naming the ids but not the two `add_*` flags would encode without them."""
    kv = _written(_fairseq_dir(tmp_path))
    assert (kv["bos_token_id"], kv["eos_token_id"], kv["pad_token_id"]) == (0, 2, 1)
    assert kv["add_bos_token"] is True and kv["add_eos_token"] is True


def test_an_explicit_kwarg_still_wins(tmp_path):
    """The ALBERT/XLNet door, which predates the id authority and must keep overriding it -- a caller
    naming a framing is making a claim the files do not."""
    kv = _written(_fairseq_dir(tmp_path), bos_token_id=7, eos_token_id=8)
    assert (kv["bos_token_id"], kv["eos_token_id"]) == (7, 8)


def test_the_normalizer_always_comes_from_the_protobuf(tmp_path):
    """`precompiled_charsmap` exists in no other file, and without it `loom::Vocab` normalizes nothing
    and segments the un-normalized string."""
    for kv in (_written(), _written(_fairseq_dir(tmp_path))):
        assert kv["precompiled_charsmap"] == b"\x00\x00\x00\x00charsmap"
        assert kv["add_space_prefix"] is True
        assert kv["remove_extra_whitespaces"] is True
        assert kv["tokenizer_model"] == "t5"


def test_a_layout_can_be_handed_in_directly():
    """`HfIdLayout` is a plain record, so a caller that reads its ids from somewhere other than a
    `tokenizer.json` needs no file on disk to say so."""
    writer = _RecordingWriter()
    write_sentencepiece_vocab(writer, _proto_bytes(),
                              hf_ids=HfIdLayout(pieces=["a", "b"], scores=[-1.0, -2.0], unk_id=1))
    assert writer.kv["token_list"] == ["a", "b"]
    assert writer.kv["unk_token_id"] == 1
    assert "add_bos_token" not in writer.kv


def test_no_protobuf_reproduces_the_protobuf_path(tmp_path):
    """A checkpoint with no `.model` exports the same vocabulary as one with it.

    `kredor/punctuate-all` is the family-12 case: XLM-R base, `tokenizer.json` only. Everything a
    SentencePiece GGUF needs is in that file -- pieces and scores in `model.vocab`, types from
    `added_tokens[].special`, UNIGRAM from `model.type`, and the charsmap from a `Precompiled`
    normalizer -- so requiring the protobuf was a restriction rather than a necessity.

    Built here as a synthetic pair rather than against a real checkpoint so the assertion is exact and
    the test is hermetic: the same `HfIdLayout` through both paths must give byte-identical KVs.
    """
    import base64
    from gguf import GGUFReader, GGUFWriter
    from sentencepiece import sentencepiece_model_pb2 as spm_pb2
    from loom_exporter.spm_tokenizer_export import read_hf_id_layout, write_sentencepiece_vocab

    pieces = ["<s>", "<pad>", "</s>", "<unk>", "▁hello", "▁world", "!"]
    charsmap = b"\x00\x01\x02\x03"

    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [{"id": i, "content": p, "special": True} for i, p in enumerate(pieces[:4])],
        "normalizer": {"type": "Precompiled",
                       "precompiled_charsmap": base64.b64encode(charsmap).decode()},
        "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "always"},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}},
                                      {"SpecialToken": {"id": "</s>"}}]},
        "model": {"type": "Unigram", "unk_id": 3,
                  "vocab": [[p, 0.0 if i < 4 else -float(i)] for i, p in enumerate(pieces)]},
    }), encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>", "pad_token": "<pad>"}), encoding="utf-8")

    # the protobuf a checkpoint like this WOULD have shipped, in fairseq's own order
    proto = spm_pb2.ModelProto()
    proto.trainer_spec.model_type = proto.trainer_spec.UNIGRAM
    proto.normalizer_spec.add_dummy_prefix = True
    proto.normalizer_spec.remove_extra_whitespaces = True
    proto.normalizer_spec.precompiled_charsmap = charsmap
    for piece, kind in (("<unk>", 2), ("<s>", 3), ("</s>", 3),
                        ("▁hello", 1), ("▁world", 1), ("!", 1), ("<pad>", 3)):
        entry = proto.pieces.add()
        entry.piece, entry.score, entry.type = piece, 0.0, kind

    layout = read_hf_id_layout(tmp_path)
    assert layout is not None and layout.precompiled_charsmap == charsmap

    def kvs(path, protobuf):
        writer = GGUFWriter(str(path), "cmp")
        write_sentencepiece_vocab(writer, protobuf, hf_ids=layout)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.close()
        reader = GGUFReader(str(path))
        return {f.name: [bytes(f.parts[i]) if f.parts[i].dtype.kind in "SU" else f.parts[i].tolist()
                         for i in f.data]
                for f in reader.fields.values() if f.name.startswith("tokenizer.")}

    assert kvs(tmp_path / "with.gguf", proto.SerializeToString()) == kvs(tmp_path / "without.gguf", None)


def test_no_protobuf_and_no_layout_is_refused(tmp_path):
    """Neither source is a caller error, not a default -- it used to be impossible to express."""
    from gguf import GGUFWriter
    from loom_exporter.spm_tokenizer_export import write_sentencepiece_vocab
    with pytest.raises(ValueError, match="protobuf"):
        write_sentencepiece_vocab(GGUFWriter(str(tmp_path / "x.gguf"), "x"), None)


# -- a CHAR model (SpeechT5's `spm_char.model`) ----------------------------------------------------

def _char_proto(pieces) -> bytes:
    m = spm_pb2.ModelProto()
    m.trainer_spec.model_type = m.trainer_spec.CHAR
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.precompiled_charsmap = b"\x00\x00\x00\x00charsmap"
    for piece, kind in [("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2)] + [(p, 1) for p in pieces]:
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = piece, -1.0, kind
    return m.SerializeToString()


def test_a_char_model_of_single_characters_is_written_as_unigram():
    """Every matchable piece is one character, so the Unigram lattice has exactly one path -- the
    CHAR model's split. SentencePiece merges an unknown run into one `<unk>` in both models."""
    writer = _RecordingWriter()
    write_sentencepiece_vocab(writer, _char_proto(["▁", "a", "b", "̄"]))
    assert writer.kv["tokenizer_model"] == "t5"
    assert writer.kv["token_list"] == ["<s>", "<pad>", "</s>", "<unk>", "▁", "a", "b", "̄"]


def test_the_char_split_is_the_unigram_split_in_sentencepiece_itself():
    """The claim the mapping rests on, measured on SentencePiece rather than argued: the same pieces
    re-typed UNIGRAM encode every string -- unknown runs included -- identically."""
    import random

    import sentencepiece as spm

    raw = _char_proto(list("▁abcde.,"))
    m = spm_pb2.ModelProto()
    m.ParseFromString(raw)
    m.normalizer_spec.precompiled_charsmap = b""          # identity normalisation for both
    char = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    m.trainer_spec.model_type = m.trainer_spec.UNIGRAM
    uni = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    rng = random.Random(0)
    alphabet = list("abcde., xyz09")
    for _ in range(2000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        assert char.encode(s) == uni.encode(s), repr(s)


def test_a_char_model_with_a_multi_character_piece_is_refused():
    """With a two-character piece the lattice has a second path, and Unigram would take it."""
    with pytest.raises(NotImplementedError, match="multi-character"):
        write_sentencepiece_vocab(_RecordingWriter(), _char_proto(["▁", "a", "ab"]))
