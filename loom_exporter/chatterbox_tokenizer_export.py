"""Writes Chatterbox's text front end into a GGUF, `tokenizer.ggml.model`="chatterbox".

A new tag, for [ADR-033](../../loom.cpp/docs/adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)'s
reason: `tokenizer.json` is a `tokenizers` BPE with NO byte-level mapping, no normalizer and a
`Whitespace` pre-tokenizer, which is none of the schemes the engine had. "gpt2" would be the wrong
answer twice over -- `BpeVocab` NFC-normalizes and byte-maps, and both change ids.

What ships, and why each piece is DATA rather than code in `loom::ChatterboxVocab`:

* **the table, the 265 merges and the added tokens** -- the tokenizer proper. The added tokens are
  matched literally before pre-tokenization, which is how `[SPACE]` works at all and how a caller's
  `[laughter]` reaches the model;
* **`punc_norm`'s rules** (`chatterbox/tts.py`): the replacement pairs in their order, the sentence
  enders, the terminal full stop and the reference's stand-in for an empty text. They are model
  constants (ADR-006), so they belong to the export; the engine implements only the SHAPE of the
  function -- capitalise, collapse whitespace, replace, strip, terminate. "Capitalise" is Python's
  own full case mapping shipped as a table, because `ß.upper()` is `SS`;
* **`word_chars`, the pre-tokenizer's `\\w` restricted to the table's own characters**, computed here by
  asking the reference's own `Whitespace` pre-tokenizer. That restriction is what makes it exact
  without a Unicode table in the engine: a character NOT in the table becomes its own `[UNK]` and can
  take part in no merge, so which side of a pre-token boundary it falls on cannot change an id.

Verified differentially against `EnTokenizer.text_to_tokens(punc_norm(text))`; see the export's
module docstring for the numbers.

Requires: pip install gguf tokenizers
"""
import json
from pathlib import Path
from typing import List, Tuple

from gguf import GGUFWriter

_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3

# `punc_norm`'s own table, verbatim and in order -- `str.replace` is applied pair by pair, so the
# order is part of the function (`...` must go before any single `.` rule could see it).
PUNC_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("...", ", "),
    ("…", ", "),
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("“", "\""),
    ("”", "\""),
    ("‘", "'"),
    ("’", "'"),
)
SENTENCE_ENDERS = (".", "!", "?", "-", ",")
TERMINAL = "."
EMPTY_TEXT = "You need to add some text for me to talk."
SPACE_TOKEN = "[SPACE]"


def read_tokenizer(tokenizer_dir: str) -> dict:
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing -- a Chatterbox release directory carries one.")
    spec = json.loads(path.read_text(encoding="utf-8"))
    model = spec.get("model", {})
    # Every assumption `ChatterboxVocab` makes, checked against the file rather than trusted: a
    # tokenizer that differs in any of these is a different scheme and must not load as this one.
    problems = []
    if model.get("type") != "BPE":
        problems.append(f"model.type is {model.get('type')!r}, not 'BPE'")
    if spec.get("normalizer") is not None:
        problems.append(f"it has a normalizer ({spec['normalizer']})")
    if spec.get("pre_tokenizer") != {"type": "Whitespace"}:
        problems.append(f"its pre_tokenizer is {spec.get('pre_tokenizer')}, not Whitespace")
    for key, want in (("continuing_subword_prefix", None), ("end_of_word_suffix", None),
                      ("fuse_unk", False), ("dropout", None)):
        if model.get(key, want) != want:
            problems.append(f"model.{key} is {model.get(key)!r}, not {want!r}")
    if model.get("byte_fallback"):
        problems.append("model.byte_fallback is set")
    if problems:
        raise ValueError(f"{path} is not the tokenizer `ChatterboxVocab` implements: " + "; ".join(problems))
    return spec


def word_chars(pieces: List[str]) -> List[str]:
    """The table's single-codepoint pieces that `Whitespace`'s `\\w+` would join to a letter."""
    from tokenizers.pre_tokenizers import Whitespace

    pre = Whitespace()
    out = []
    for piece in pieces:
        if len(piece) != 1:
            continue
        if not pre.pre_tokenize_str(piece):
            continue  # whitespace: `punc_norm` collapses it to a space before this ever runs
        joined = [p for p, _ in pre.pre_tokenize_str("a" + piece + "a")]
        if len(joined) == 1:
            out.append(piece)
    return out


def upper_case_table() -> List[Tuple[str, str]]:
    """`(c, c.upper())` for every codepoint `punc_norm`'s `text[0].islower()` would change."""
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        c = chr(cp)
        if c.islower() and c.upper() != c:
            out.append((c, c.upper()))
    return out


def write_chatterbox_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    spec = read_tokenizer(tokenizer_dir)
    model = spec["model"]
    vocab = model["vocab"]
    size = max(vocab.values()) + 1
    tokens = [""] * size
    for piece, i in vocab.items():
        tokens[i] = piece
    if any(t == "" for t in tokens):
        raise ValueError("tokenizer.json's vocab has holes in its id range")
    added = [a for a in spec.get("added_tokens", []) if a.get("special")]
    types = [_TOKEN_TYPE_NORMAL] * size
    for a in added:
        if tokens[a["id"]] != a["content"]:
            raise ValueError(f"added token {a['content']!r} disagrees with vocab row {a['id']}")
        types[a["id"]] = _TOKEN_TYPE_CONTROL
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    unk = model.get("unk_token")

    writer.add_tokenizer_model("chatterbox")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_unk_token_id(vocab[unk])
    p = "tokenizer.ggml.chatterbox."
    writer.add_array(p + "added_tokens", [a["content"] for a in added])
    writer.add_array(p + "word_chars", word_chars(tokens))
    writer.add_array(p + "replace_from", [f for f, _ in PUNC_REPLACEMENTS])
    writer.add_array(p + "replace_to", [t for _, t in PUNC_REPLACEMENTS])
    writer.add_array(p + "sentence_enders", list(SENTENCE_ENDERS))
    # `text[0].islower()` / `text[0].upper()` as a table: Python's FULL case mapping, which no
    # single-codepoint rule reproduces (`ß` -> `SS`). Every codepoint whose `islower()` holds and whose
    # uppercase differs; 1494 pairs.
    upper = upper_case_table()
    writer.add_array(p + "upper_from", [k for k, _ in upper])
    writer.add_array(p + "upper_to", [v for _, v in upper])
    # `EnTokenizer.decode` removes the stop token and `[UNK]` after joining.
    writer.add_array(p + "decode_drop", [tokens[0], unk])
    writer.add_string(p + "space_token", SPACE_TOKEN)
    writer.add_string(p + "terminal", TERMINAL)
    writer.add_string(p + "empty_text", EMPTY_TEXT)
