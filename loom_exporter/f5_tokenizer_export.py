"""Writes F5-TTS's character vocabulary into a GGUF, `tokenizer.ggml.model`="f5".

A new tag, for [ADR-033](../../loom.cpp/docs/adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)'s
reason: a tag answers "which scheme is this", and this scheme is not one the engine already had.

**The table is a flat one-piece-per-row list, like "ctc"'s and "funasr"'s. What is new is that its
pieces are CHARACTERS, and that the reference's own text function is not a table at all.**
`convert_char_to_pinyin` runs `rjieba` word segmentation and `pypinyin` before a single id is looked
up: Chinese characters become toned pinyin syllables (`zhong1`), which is why the table has
multi-character rows, and a space is inserted before a multi-character SEGMENT whose predecessor did
not end in one -- a decision made by jieba's dictionary-and-HMM DAG, which no table and no rule
reproduces.

**So what ships is the half that IS a table, and its boundary was measured rather than assumed.**
`loom::F5Vocab` applies the reference's own five-character `custom_trans` substitution and then maps
codepoints through this table. Against `convert_char_to_pinyin` over generated English prose:

| input class                                   | identical |
|---|---|
| ordinary prose (words separated by spaces)    | 2000/2000 |
| with multi-character punctuation runs (`--`)  | 1100/2000 |
| with hyphen-joined digit groups (`2026-09-18`)| 1758/2000 |

The two divergent classes differ from the reference by ONE INSERTED SPACE and nothing else, always in
the same direction (the reference inserts, the table does not). That is the family's front-end
boundary, and it is the same one the phoneme-input families draw around g2p: a host that needs
reference-exact ids for text of those shapes runs `convert_char_to_pinyin` and passes ids, which the
driver's `text_ids` input takes directly. Text containing CJK is refused by name rather than mapped
character by character, because a per-character lookup of Chinese would find no row and silently
produce the unknown id for a whole sentence.

Requires: pip install gguf
"""
from pathlib import Path
from typing import List

from gguf import GGUFWriter

# gguf's own `TokenType` value, as every other flat writer here uses it.
_TOKEN_TYPE_NORMAL = 1


def read_f5_vocab(tokenizer_dir: str) -> List[str]:
    """`vocab.txt` as `get_tokenizer(..., "custom")` reads it: one piece per line, id = line number.

    `char[:-1]` in the reference strips the newline, and the file is opened in text mode so a CRLF
    line has already become one. Reproduced rather than imported because this is the file the ids will
    come from and the count it yields is what sizes the embedding table."""
    path = Path(tokenizer_dir) / "vocab.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing -- an F5-TTS release directory carries one.")
    with path.open("r", encoding="utf-8") as f:
        return [line[:-1] for line in f]


def f5_vocab_size(tokenizer_dir: str) -> int:
    """The number the reference's `text_num_embeds` is, which is `len()` of the MAP rather than of the
    list -- a repeated piece collapses. The embedding is one row wider than this (`+ 1` for the filler
    token), and loading the checkpoint is what would catch a disagreement."""
    return len({piece: i for i, piece in enumerate(read_f5_vocab(tokenizer_dir))})


def write_f5_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    tokens = read_f5_vocab(tokenizer_dir)

    writer.add_tokenizer_model("f5")
    writer.add_token_list(tokens)
    writer.add_token_types([_TOKEN_TYPE_NORMAL] * len(tokens))
    # **The filler-token offset, written down rather than left to the driver.** Every id the graph
    # sees is `table id + 1`, because the reference reserves embedding row 0 for "no character here"
    # (`text = text + 1` in `TextEmbedding.forward`). It is a property of the VOCABULARY's relationship
    # to the embedding, not of the sampler, so a host that builds ids itself needs to know it and the
    # driver is not the place to learn it from.
    writer.add_uint32("tokenizer.ggml.f5.filler_offset", 1)
    # Id 0 is the reference's own fallback for a character the table does not have
    # (`vocab_char_map.get(c, 0)`), and in this release row 0 is the SPACE. Declared under the standard
    # unknown-id key so a reader does not have to know that coincidence.
    writer.add_unk_token_id(0)
