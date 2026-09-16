"""Writes a FunASR `CharTokenizer` token list into a GGUF, `tokenizer.ggml.model`="funasr".

A new tag, for the reason [ADR-033](../../loom.cpp/docs/adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)
gives: a tag answers "which scheme is this", and this scheme is not one the engine already had.

**The TABLE is the same shape as family 4's -- flat, decode-only, one piece per row. What differs is how
the pieces COMPOSE**, and that is not expressible by rewriting them into an existing tag:

    ["and","so","my","f@@","el@@","low"]  ->  "and so my fellow"   `@@` continues into the NEXT piece
    ["hello","你","好"]                    ->  "hello你好"           the space is REMOVED before CJK
    ["b","b","c","news"]                  ->  "BBC news"           letter runs collapse AND UPPERCASE
    ["<s>","and","</s>"]                  ->  "and"                control pieces drop

`@@` is a SUFFIX meaning "I continue", where SentencePiece's `▁` and WordPiece's `##` are PREFIXES
meaning "a word starts here". They are duals: whether a piece begins a word depends on its predecessor
under this convention, and the same piece string occurs in both roles, so no per-piece rewrite turns one
into the other. `loom::FunasrVocab` implements the assembly; this writes what it reads.

**The per-piece SCRIPT is computed here rather than in the engine, and that is the one decision in this
file.** The reference decides `isAllChinese`/`isAllAlpha` per CHARACTER using Python's Unicode
`isalpha()`. `paraformer-zh`'s 8,404 pieces contain exactly one character that is alphabetic and outside
the U+4E00-U+9FFF block the Chinese test uses (U+2B5AF, a CJK extension ideograph) -- so a plausible
"ASCII letters plus the CJK block" rule written in C++ is wrong on one row in 8,404, silently, and the
transcript it produces is still readable. Python knows the answer exactly, so Python states it. This is
ADR-027's principle one family over: whoever actually knows a fact is the one who writes it down.

Requires: pip install gguf
"""
import json
from pathlib import Path
from typing import List

from gguf import GGUFWriter

# gguf's own `TokenType` values, as every other writer here uses them.
_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3

# `loom::FunasrVocab::Script`.
SCRIPT_OTHER, SCRIPT_CJK, SCRIPT_LATIN = 0, 1, 2

# Exactly the pieces `sentence_postprocess` drops, matched by SPELLING because that is how the reference
# matches them; what reaches the file is the resulting ID set, so the engine never matches a string.
#
# **`<blank>` IS DELIBERATELY NOT HERE, and a differential test is what put it right.** It is row 0 and
# it looks like a control token, so the first version marked it one -- and that made the engine drop it
# where the reference prints it literally, which showed up as one mismatch in 5,011 random sequences.
# A Paraformer decoder cannot emit it (the blank belongs to the CTC-trained members of FunASR that share
# this vocabulary, not to a non-autoregressive one), so the divergence is unobservable in practice --
# which is exactly why it would have been a bad thing to leave in. Reproducing the reference does not
# get to stop at the rows we expect to see.
CONTROL_PIECES = ("<s>", "</s>", "<unk>", "<OOV>")


def _char_is_cjk(char: str) -> bool:
    """`isChinese` on one character.

    The CJK block, an ASCII DIGIT, or `@`. The last two look like mistakes in the reference and are
    load-bearing: `9@@` is every-character-"Chinese" by this test, so it is never treated as a subword
    continuation, and reproducing that is the difference between matching the reference and not.
    """
    return "一" <= char <= "鿿" or "0" <= char <= "9" or char == "@"


def piece_script(piece: str) -> int:
    """Which of `FunasrVocab::Script` a piece behaves as.

    Both tests are per-character and the order matters: a piece is CJK only if EVERY character is, and
    Latin only if every character is alphabetic-and-not-CJK or an apostrophe -- which is what makes
    `can't` a word rather than falling through to the literal branch.
    """
    if piece and all(_char_is_cjk(c) for c in piece):
        return SCRIPT_CJK
    if piece and all((c.isalpha() and not _char_is_cjk(c)) or c == "'" for c in piece):
        return SCRIPT_LATIN
    return SCRIPT_OTHER


def read_funasr_tokens(tokenizer_dir: str) -> List[str]:
    """A FunASR checkpoint's `tokens.json`: a plain JSON array, so the id IS the index.

    No `seg_dict`: it maps text to pieces and only the ENCODE side uses it, which this scheme does not
    have -- the ids come out of a decoder's argmax.
    """
    path = Path(tokenizer_dir) / "tokens.json"
    if not path.exists():
        raise ValueError(f"{path} does not exist; a FunASR CharTokenizer states its vocabulary there")
    tokens = json.loads(path.read_text())
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{path} is not a non-empty JSON array; the array IS the vocabulary here, "
                         f"indexed by id")
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"{path} contains a non-string entry; every row is a piece")
    return tokens


def write_funasr_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    tokens = read_funasr_tokens(tokenizer_dir)
    control = {i for i, piece in enumerate(tokens) if piece in CONTROL_PIECES}

    writer.add_tokenizer_model("funasr")
    writer.add_token_list(tokens)
    writer.add_token_types([_TOKEN_TYPE_CONTROL if i in control else _TOKEN_TYPE_NORMAL
                            for i in range(len(tokens))])
    # A project-specific key, like `tokenizer.ggml.word_delimiter_id` before it: llama.cpp's schema has
    # no concept of a per-piece script because no vocabulary type it writes composes this way.
    writer.add_array("tokenizer.ggml.piece_script", [piece_script(t) for t in tokens])
    if "<unk>" in tokens:
        writer.add_unk_token_id(tokens.index("<unk>"))
