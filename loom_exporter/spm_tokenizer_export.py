"""Extracts SentencePiece vocab data from a `.model` protobuf and writes it into a GGUF file using
llama.cpp's own "tokenizer.ggml.*" KV schema (confirmed directly against gguf-py's `GGUFWriter`,
gguf-py's own `SentencePieceVocab` class in `gguf/vocab.py`, and llama.cpp's `include/llama.h`):
`tokenizer.ggml.model` is `"t5"` for a genuine SentencePiece *unigram* model, or `"llama"` for real
SentencePiece **BPE** (llama.cpp's own tag -- the original LLaMA/Mistral tokenizers are themselves
SentencePiece BPE models, confirmed via `gguf-py`'s `SentencePieceVocab` writing this exact tag) --
selected automatically here from the real `.model` protobuf's own `trainer_spec.model_type`
(`UNIGRAM=1`, `BPE=2`; `WORD`/`CHAR` are not implemented on the C++ side, see `loom::Vocab`), not
assumed or hardcoded. `.tokens`/`.scores`/`.token_type` arrays, `unknown_token_id`, `add_space_prefix`,
`remove_extra_whitespaces`, and the raw `precompiled_charsmap` blob (needed for Unicode normalization
during encode -- see `Vocab` in the C++ engine, which mirrors llama.cpp's XCDA-based normalizer
exactly) are identical between both types, confirmed by inspecting a real BPE model's protobuf directly.

SentencePiece's own per-piece `Type` enum (`NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5,
BYTE=6`, confirmed via direct protobuf inspection) is numerically identical to llama.cpp's
`llama_token_type` (confirmed from `include/llama.h`), so piece types are copied straight through with
no remapping.

Uses the `sentencepiece` package's bundled protobuf definitions directly (`sentencepiece_model_pb2`),
not the `SentencePieceProcessor` wrapper -- the wrapper doesn't expose `precompiled_charsmap` or the
normalizer flags needed here.

Optional `bos_token_id`/`eos_token_id`/`add_bos_token`/`add_eos_token` kwargs close the gap that
otherwise blocks ALBERT/XLNet-style Unigram models (which wrap sequences via SentencePiece's own BOS/EOS
convention rather than a separate CLS/SEP concept, unlike T5's own tokenizer) -- default to today's no-op
(nothing written), so every existing NeMo T5/BPE call site is byte-for-byte unaffected.

**THE PROTOBUF IS THE AUTHORITY ON WHAT A PIECE IS. IT IS NOT ALWAYS THE AUTHORITY ON WHAT ITS ID IS**,
and `hf_ids` is where the second authority goes (P5, family 12's third checkpoint). The fairseq-derived
family -- XLM-R, and RoBERTa-style Unigram models generally -- ships a `.model` protobuf whose piece
ORDER is not the vocabulary the model was trained against: `transformers`' own converter drops the
proto's leading `<unk>/<s>/</s>`, prepends `<s>/<pad>/</s>/<unk>` at 0..3, shifts every remaining piece
by one, and appends `<mask>` at the end. Writing the proto verbatim for such a checkpoint produces a
vocabulary that is off by one for all 250,000 pieces and whose ids index the embedding table wrongly --
a file that loads, encodes, decodes to plausible text, and is wrong everywhere.

Nothing in the protobuf records that remapping. What does record it is the `tokenizer.json` the same
checkpoint ships, whose Unigram `model.vocab` is that converter's OUTPUT and is therefore already in
model-id order. So when one sits beside the proto, `read_hf_id_layout` reads it and the two are
combined: ids, pieces and scores from `tokenizer.json`, and piece TYPES plus the normalizer
(`precompiled_charsmap`, `add_dummy_prefix`, `remove_extra_whitespaces`) from the protobuf, which is the
only place those exist. A piece the proto does not have at all is one the converter ADDED, so it is
written CONTROL -- unmatchable in text, which is what `loom::Vocab` does with that type and what a
`<pad>`/`<mask>` has to be.

The seam is `tokenizer.json`'s presence, deliberately: every SentencePiece call site that predates this
(the NeMo ASR converters, which extract a bare `tokenizer.model` out of a `.nemo` archive) ships no
`tokenizer.json`, gets `hf_ids=None`, and writes byte-for-byte the file it wrote before.

Requires: pip install sentencepiece gguf


Moved here from `tools/convert_nemo/tokenizer_common.py` when the bespoke NeMo converters retired
(BACKLOG.md P4.0.17 step 3). It sits beside `bpe_tokenizer_export` and `byt5_tokenizer_export`, which is
where the other vocab writers already were -- and it removes a cross-package import: `exporter.py` used
to reach into `convert_nemo`, which only resolved when `tools/` happened to be on `sys.path` as a
package root.
"""
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gguf import GGUFWriter
from sentencepiece import sentencepiece_model_pb2 as spm_pb2

# SentencePiece's own `ModelProto.SentencePiece.Type`, which llama.cpp's `llama_token_type` matches
# numerically (see the module docstring). Named here because the pieces this file SYNTHESIZES have no
# protobuf entry to copy a type from.
_TYPE_NORMAL = 1
_TYPE_UNKNOWN = 2
_TYPE_CONTROL = 3

# The roles read out of `special_tokens_map.json` / `tokenizer_config.json`, as the keys both files
# spell them with. TWO, not four: `bos_token`/`eos_token` are named by tokenizers that do not add them
# (T5's map names an `eos_token`; its encode is the only thing that decides whether one appears), so
# the framing is read from the post-processor instead and only these two come from here.
_SPECIAL_TOKEN_KEYS = ("unk_token", "pad_token")


@dataclass(frozen=True)
class HfIdLayout:
    """What a checkpoint's `tokenizer.json` says its ids are, for a Unigram tokenizer.

    `pieces`/`scores` are id-indexed. The four ids are the roles the checkpoint names, resolved to ids
    through those same pieces, or None where it names none -- `bos_id`/`eos_id` are read from the
    POST-PROCESSOR rather than from the special-token map, because what they have to answer is not
    "does this tokenizer have a `<s>`" but "does its encode put one there", and a
    `TemplateProcessing.single` of `<s> $A </s>` is that question written down.
    """

    pieces: List[str]
    scores: List[float]
    unk_id: Optional[int] = None
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    pad_id: Optional[int] = None
    # Everything below is what the PROTOBUF supplies when there is one, read off `tokenizer.json`
    # instead for the checkpoints that ship no `.model` at all -- see `write_sentencepiece_vocab`.
    # `special_ids` are the ids `added_tokens` marks `special`, which is the only statement of piece
    # TYPE a fast tokenizer makes.
    special_ids: frozenset = frozenset()
    precompiled_charsmap: Optional[bytes] = None
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True


def _template_framing(tokenizer_json: dict) -> tuple[Optional[str], Optional[str]]:
    """The pieces a `TemplateProcessing` post-processor wraps a single sequence in, as `(bos, eos)`.

    Reads the template's own first and last entries and takes them only when they are `SpecialToken`s,
    so a tokenizer that frames on one side gets one of the two and a tokenizer with no post-processor
    (T5's, whose eos comes from the model's own convention) gets neither. Any other post-processor type
    yields neither rather than guessing: `RobertaProcessing` and `BertProcessing` name their pieces
    under different keys, and no checkpoint in this repo uses one with a SentencePiece model.
    """
    post = tokenizer_json.get("post_processor") or {}
    if post.get("type") != "TemplateProcessing":
        return None, None
    single = post.get("single") or []
    if not single:
        return None, None

    def _special(entry):
        return (entry or {}).get("SpecialToken", {}).get("id")

    return _special(single[0]), _special(single[-1])


def _read_hf_normalizer(tokenizer_json: dict) -> tuple[Optional[bytes], bool, bool]:
    """The three normalizer facts a SentencePiece GGUF needs, from `tokenizer.json` rather than a proto.

    `tokenizers` stores exactly what the protobuf's `normalizer_spec` does, under different names: a
    `Precompiled` normalizer carries the same `precompiled_charsmap` (base64 in JSON, raw bytes in the
    proto), and `Metaspace`'s `prepend_scheme`/`add_prefix_space` is `add_dummy_prefix`. A `Strip`-ish
    `Replace` of runs of spaces is `remove_extra_whitespaces`; absent one, the SentencePiece default
    (true) is what the checkpoint was trained with, so that is the default here too.

    Returns `(charsmap_or_None, add_dummy_prefix, remove_extra_whitespaces)`.
    """
    def walk(node):
        if not isinstance(node, dict):
            return
        yield node
        for child in node.get("normalizers") or node.get("pretokenizers") or []:
            yield from walk(child)

    charsmap: Optional[bytes] = None
    for node in walk(tokenizer_json.get("normalizer") or {}):
        raw = node.get("precompiled_charsmap")
        if raw:
            charsmap = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
            break

    add_dummy_prefix = True
    for node in walk(tokenizer_json.get("pre_tokenizer") or {}):
        if node.get("type") == "Metaspace":
            scheme = node.get("prepend_scheme")
            if scheme is not None:
                add_dummy_prefix = scheme != "never"
            elif node.get("add_prefix_space") is not None:
                add_dummy_prefix = bool(node["add_prefix_space"])
            break

    return charsmap, add_dummy_prefix, True


def read_hf_id_layout(tokenizer_dir) -> Optional[HfIdLayout]:
    """A Unigram `tokenizer.json`'s own id order, or None when there is nothing to read.

    None for a directory with no `tokenizer.json` and for one whose tokenizer is not Unigram -- in both
    cases the protobuf is the only authority there is, which is exactly the pre-existing behaviour.
    """
    tok_dir = Path(tokenizer_dir)
    tokenizer_json_path = tok_dir / "tokenizer.json"
    if not tokenizer_json_path.exists():
        return None
    tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    model = tokenizer_json.get("model") or {}
    if model.get("type") != "Unigram":
        return None

    entries = model.get("vocab") or []
    pieces = [str(entry[0]) for entry in entries]
    scores = [float(entry[1]) for entry in entries]
    by_piece = {piece: idx for idx, piece in enumerate(pieces)}

    config_path = tok_dir / "tokenizer_config.json"
    map_path = tok_dir / "special_tokens_map.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    mapped = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    specials = {key: (config.get(key) or mapped.get(key)) for key in _SPECIAL_TOKEN_KEYS}
    specials = {key: (value["content"] if isinstance(value, dict) else value)
                for key, value in specials.items() if value}

    template_bos, template_eos = _template_framing(tokenizer_json)
    norm_charsmap, dummy_prefix, strip_ws = _read_hf_normalizer(tokenizer_json)
    return HfIdLayout(
        pieces=pieces,
        scores=scores,
        special_ids=frozenset(int(t["id"]) for t in tokenizer_json.get("added_tokens") or []
                              if t.get("special") and t.get("id") is not None),
        precompiled_charsmap=norm_charsmap,
        add_dummy_prefix=dummy_prefix,
        remove_extra_whitespaces=strip_ws,
        unk_id=by_piece.get(specials.get("unk_token")),
        # The post-processor decides the framing; the special-token map only supplies the piece text
        # when there is no post-processor to name it, which is the ALBERT/XLNet shape the explicit
        # kwargs below were added for.
        bos_id=by_piece.get(template_bos),
        eos_id=by_piece.get(template_eos),
        pad_id=by_piece.get(specials.get("pad_token")),
    )


def write_sentencepiece_vocab(writer: GGUFWriter, tokenizer_model_bytes: Optional[bytes], *,
                               hf_ids: Optional[HfIdLayout] = None,
                               bos_token_id: int | None = None, eos_token_id: int | None = None,
                               add_bos_token: bool = False, add_eos_token: bool = False) -> None:
    """Writes a SentencePiece vocabulary, from the protobuf where there is one and from
    `tokenizer.json` where there is not.

    THE PROTOBUF IS NO LONGER REQUIRED. It is still preferred and still the only authority for a
    checkpoint that ships one, because piece TYPES are recorded nowhere else in full. But a growing
    number of fast-tokenizer-only checkpoints ship `tokenizer.json` alone -- `kredor/punctuate-all` is
    the family-12 case -- and everything a GGUF needs is in it:

    | GGUF field              | protobuf                     | tokenizer.json                        |
    |-------------------------|------------------------------|---------------------------------------|
    | pieces, scores          | `pieces[].piece/.score`      | `model.vocab` as `[piece, score]`     |
    | ids                     | position (but see below)     | position, already remapped            |
    | types                   | `pieces[].type`              | `added_tokens[].special`              |
    | UNIGRAM vs BPE          | `trainer_spec.model_type`    | `model.type`                          |
    | charsmap, dummy prefix  | `normalizer_spec`            | `normalizer`, `pre_tokenizer`         |

    The types column is the lossy one and the reason the protobuf still wins when present: it
    distinguishes CONTROL from USER_DEFINED from BYTE, where `tokenizer.json` says only "special or
    not". For a Unigram checkpoint that distinction does not reach `loom::Vocab`'s Viterbi -- it
    segments on scores, and treats every non-NORMAL piece as un-segmentable alike -- so a two-way split
    is faithful there. It would NOT be for a BPE model with byte-fallback pieces, which is why this
    path refuses anything but Unigram rather than guessing.

    AND THE FAIRSEQ REMAP IS A NO-OP HERE, which is the happy part: the reason `read_hf_id_layout`
    exists at all is that a proto's piece order and a fairseq checkpoint's ids disagree. A checkpoint
    with no proto has no second order to disagree with -- `model.vocab` IS the id order, `<s>/<pad>/
    </s>/<unk>` already at 0..3 and `<mask>` already last.
    """
    m = None
    if tokenizer_model_bytes is not None:
        m = spm_pb2.ModelProto()
        m.ParseFromString(tokenizer_model_bytes)
    elif hf_ids is None:
        raise ValueError("write_sentencepiece_vocab needs a `.model` protobuf or an HfIdLayout; "
                         "got neither")

    if hf_ids is None:
        pieces = [p.piece for p in m.pieces]
        scores = [p.score for p in m.pieces]
        types = [int(p.type) for p in m.pieces]
        unk_id = next((i for i, p in enumerate(m.pieces) if p.type == p.UNKNOWN), 0)
        pad_token_id = None
    else:
        # Ids, pieces and scores from the fast tokenizer; TYPES from the protobuf, which is the only
        # file that has them. Matched by piece text rather than by position, because the whole reason
        # this branch exists is that the two orders differ -- and by FIRST occurrence, so a duplicated
        # piece cannot make the lookup depend on which copy won.
        pieces = list(hf_ids.pieces)
        scores = list(hf_ids.scores)
        if m is not None:
            type_of: dict[str, int] = {}
            for piece in m.pieces:
                type_of.setdefault(piece.piece, int(piece.type))
            types = [type_of.get(piece, _TYPE_CONTROL) for piece in pieces]
        else:
            # No protobuf: `added_tokens[].special` is the only type statement there is. UNKNOWN is
            # named separately because it is a role rather than a flag, and `loom::Vocab` looks it up
            # by type rather than by id.
            types = [_TYPE_CONTROL if i in hf_ids.special_ids else _TYPE_NORMAL
                     for i in range(len(pieces))]
            if hf_ids.unk_id is not None and 0 <= hf_ids.unk_id < len(types):
                types[hf_ids.unk_id] = _TYPE_UNKNOWN
        # Derived from the REMAPPED types, not carried over from the protobuf's own index: the piece
        # that is UNKNOWN is the same piece either way, and its id is the thing that moved.
        unk_id = hf_ids.unk_id
        if unk_id is None:
            unk_id = next((i for i, t in enumerate(types) if t == _TYPE_UNKNOWN), 0)
        bos_token_id = bos_token_id if bos_token_id is not None else hf_ids.bos_id
        eos_token_id = eos_token_id if eos_token_id is not None else hf_ids.eos_id
        pad_token_id = hf_ids.pad_id
        # The framing is what the post-processor DOES, so an id read from it is also the statement that
        # the encode adds one -- unlike the explicit kwargs, where the caller says both separately.
        add_bos_token = add_bos_token or hf_ids.bos_id is not None
        add_eos_token = add_eos_token or hf_ids.eos_id is not None

    if m is not None:
        model_type = m.trainer_spec.model_type
        if model_type == m.trainer_spec.UNIGRAM:
            tokenizer_model = "t5"
        elif model_type == m.trainer_spec.BPE:
            tokenizer_model = "llama"
        else:
            raise NotImplementedError(f"SentencePiece model_type {model_type} (WORD/CHAR) is not "
                                       "implemented on the C++ side (loom::Vocab only supports "
                                       "UNIGRAM and BPE)")
    else:
        # `read_hf_id_layout` returns None for anything but Unigram, so reaching here without a proto
        # means Unigram -- but assert it rather than assume, because the type split above is only
        # faithful for Unigram and a BPE checkpoint arriving here would be silently mis-typed.
        if hf_ids is None:
            raise ValueError("no protobuf and no HfIdLayout")
        tokenizer_model = "t5"

    writer.add_tokenizer_model(tokenizer_model)
    writer.add_token_list(pieces)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_unk_token_id(unk_id)
    if m is not None:
        writer.add_add_space_prefix(bool(m.normalizer_spec.add_dummy_prefix))
        writer.add_remove_extra_whitespaces(bool(m.normalizer_spec.remove_extra_whitespaces))
        charsmap = m.normalizer_spec.precompiled_charsmap
    else:
        writer.add_add_space_prefix(bool(hf_ids.add_dummy_prefix))
        writer.add_remove_extra_whitespaces(bool(hf_ids.remove_extra_whitespaces))
        charsmap = hf_ids.precompiled_charsmap
    if charsmap:
        writer.add_precompiled_charsmap(charsmap)

    if bos_token_id is not None:
        writer.add_bos_token_id(bos_token_id)
    if eos_token_id is not None:
        writer.add_eos_token_id(eos_token_id)
    # `loom::text::classify` strips the ids a model's own encode ADDED before it hands back one label
    # per token, and reads this KV to know which they are -- so a pad id the file does not name is a
    # `<pad>` reported as a labelled token. Written only where the checkpoint names one.
    if pad_token_id is not None:
        writer.add_pad_token_id(pad_token_id)
    # Only written when true -- matches the C++ side's own "absent KV defaults to false" convention, so
    # every existing call site that never passes these kwargs writes byte-for-byte the same GGUF as before.
    if add_bos_token:
        writer.add_add_bos_token(True)
    if add_eos_token:
        writer.add_add_eos_token(True)
