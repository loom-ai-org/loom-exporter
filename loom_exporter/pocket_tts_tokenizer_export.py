"""Pocket-TTS's text front end as GGUF data: `tokenizer.ggml.model = "pocket_tts"`.

The vocabulary is an ordinary SentencePiece Unigram model with byte fallback, written by
`spm_tokenizer_export.write_sentencepiece_vocab` under this tag. What the tag adds is the reference's
text path around it -- `prepare_text_prompt` and `split_into_best_sentences` -- and every constant
that path reads is written here FROM the reference, so `loom::PocketTtsVocab` holds only its shape
(loom.cpp ADR-041's rule):

| key (`tokenizer.ggml.pocket_tts.`) | from                                                        |
|------------------------------------|-------------------------------------------------------------|
| `replace_from`/`replace_to`        | `prepare_text_prompt`'s `str.replace` calls, in order        |
| `char_replace_from`/`_to`          | the config's `replace_characters` (v4), when it has any      |
| `sentence_marks`/`stray_clause_marks` | the two classes of v4's `re.sub` after it, PROBED off      |
|                                    | `prepare_text_prompt` rather than restated                   |
| `decimal_period_rule`              | 1 or 2, PROBED off `_is_decimal_period_boundary` ("3. 14")   |
| `terminal`/`weak`/`closers`        | `text_chunking._TERMINAL_PUNCTUATION` / `_WEAK_` / `_CLOSERS` |
| `full_stop`                        | the "." `_ensure_terminal_punctuation` appends               |
| `upper_from`/`upper_to`            | `c.upper()` where `not c.isupper()` and it changes `c`       |
| `digits`                           | `str.isdigit()`, the decimal-period rule's test              |
| `sentence_end_ids`/`clause_end_ids`| `tokenizer(".!...?")[1:]` / `tokenizer(",;:")[1:]`, as there |
| `max_tokens_per_chunk`             | `default_parameters.MAX_TOKEN_PER_CHUNK`                     |
| `chunk_header_short`/`_long`       | `<s>` / `</s>`, never produced by an encode: each chunk opens |
|                                    | with one, saying which tail `prepare_text_prompt` guessed     |
| `short_chunk_max_words`            | that guess's threshold (`number_of_words <= 4`)               |
| the two flags                      | the checkpoint's YAML                                         |
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

from .spm_tokenizer_export import write_sentencepiece_vocab

PREFIX = "tokenizer.ggml.pocket_tts."
# `prepare_text_prompt`: `frames_after_eos_guess = 3 if number_of_words <= 4 else 1`. A literal in the
# reference's code, so it is restated here and pinned against the function by the CI test.
SHORT_CHUNK_MAX_WORDS = 4
SHORT_CHUNK_FRAMES_AFTER_EOS = 3
LONG_CHUNK_FRAMES_AFTER_EOS = 1


def _reference_config(model_dir: Path) -> dict:
    import yaml

    from .pocket_tts_export import POCKET_TTS_REPO

    path = Path(POCKET_TTS_REPO) / "pocket_tts" / "config" / f"{model_dir.name}.yaml"
    return yaml.safe_load(path.read_text())


def _case_table():
    """Python's `text[0].upper()` wherever `prepare_text_prompt` would apply it: every codepoint that
    is not upper case and whose upper case differs. Full mappings included (`ß` -> `SS`)."""
    src, dst = [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if not c.isupper() and c.upper() != c:
            src.append(c)
            dst.append(c.upper())
    return src, dst


def _probe_stray_clause_rule(text_chunking):
    """`(sentence_marks, clause_marks)` of v4's `re.sub(r"([.!?\u2026])\s*[,;:]", r"\1", ...)` inside
    `prepare_text_prompt`, read off the function's behaviour over every punctuation character: `c` is a
    sentence mark when "Ab<c> , cd" loses its comma, a clause mark when "Ab. <c> cd" loses `c`. The
    table passed is a no-op one (NUL -> NUL) so that the step runs at all -- it runs only for a
    non-empty table -- without rewriting a probe character first."""
    import inspect
    import unicodedata

    if "replace_characters" not in inspect.signature(text_chunking.prepare_text_prompt).parameters:
        return None
    prepare = lambda t: text_chunking.prepare_text_prompt(t, False, False, False, False, {"\x00": "\x00"})[0]
    punct = [chr(cp) for cp in range(0x20, 0x3000) if unicodedata.category(chr(cp)).startswith("P")]
    sentence = [c for c in punct if prepare(f"Ab{c} , cd") == f"Ab{c} cd"]
    clause = [c for c in punct if prepare(f"Ab. {c} cd") == "Ab. cd"]
    if not sentence or not clause:
        raise ValueError("prepare_text_prompt takes replace_characters but drops no stray clause mark; "
                         "the probe no longer describes the reference")
    return sentence, clause


def _probe_decimal_period_rule(text_chunking, sp) -> int:
    """1 when `_is_decimal_period_boundary` reads "I counted to 3. 14 people." as 3.14 (the whole prefix
    and suffix decoded: before v4), 2 when it sees the space (v4: the ids next to the cut only)."""

    class _Tok:
        def decode(self, ids):
            return sp.decode(list(ids))

    ids = sp.encode("I counted to 3. 14 people.", out_type=int)
    cut = next(i + 1 for i, t in enumerate(ids) if sp.id_to_piece(t) == "." and i + 1 < len(ids))
    return 1 if text_chunking._is_decimal_period_boundary(ids, cut, _Tok()) else 2


def write_pocket_tts_vocab(w: GGUFWriter, tokenizer_dir: str) -> None:
    import sentencepiece as spm

    from .pocket_tts_export import import_pocket_tts

    import_pocket_tts()
    from pocket_tts.default_parameters import MAX_TOKEN_PER_CHUNK
    from pocket_tts.models import text_chunking

    model_dir = Path(tokenizer_dir)
    proto = (model_dir / "tokenizer.model").read_bytes()
    write_sentencepiece_vocab(w, proto, tokenizer_model="pocket_tts")

    config = _reference_config(model_dir)
    if config.get("pad_with_spaces_for_short_inputs"):
        raise NotImplementedError("this checkpoint pads short inputs with spaces, which "
                                  "loom::PocketTtsVocab does not do")
    replace_from, replace_to = ["\n", "\r", "  "], [" ", " ", " "]
    if config.get("remove_semicolons"):
        replace_from.append(";")
        replace_to.append(",")
    w.add_array(PREFIX + "replace_from", replace_from)
    w.add_array(PREFIX + "replace_to", replace_to)
    # v4's per-character rewrites (`Config.replace_characters`), and the stray-clause-mark rule the
    # reference runs only when there are some. A config without any writes neither, which is also what
    # every file written before v4 looks like.
    char_table = config.get("replace_characters") or {}
    rule = _probe_stray_clause_rule(text_chunking)
    if char_table:
        if rule is None:
            raise ValueError(f"{model_dir.name}'s config lists replace_characters but this reference's "
                             f"prepare_text_prompt does not take them; it is older than the config")
        w.add_array(PREFIX + "char_replace_from", list(char_table))
        w.add_array(PREFIX + "char_replace_to", list(char_table.values()))
        w.add_array(PREFIX + "sentence_marks", rule[0])
        w.add_array(PREFIX + "stray_clause_marks", rule[1])
    w.add_array(PREFIX + "terminal", list(text_chunking._TERMINAL_PUNCTUATION))
    w.add_array(PREFIX + "weak", list(text_chunking._WEAK_PUNCTUATION))
    w.add_array(PREFIX + "closers", list(text_chunking._CLOSERS))
    w.add_array(PREFIX + "full_stop", ["."])
    upper_from, upper_to = _case_table()
    w.add_array(PREFIX + "upper_from", upper_from)
    w.add_array(PREFIX + "upper_to", upper_to)
    w.add_array(PREFIX + "digits", [chr(cp) for cp in range(0x110000)
                                     if not 0xD800 <= cp <= 0xDFFF and chr(cp).isdigit()])

    sp = spm.SentencePieceProcessor(model_proto=proto)
    w.add_int32(PREFIX + "decimal_period_rule", _probe_decimal_period_rule(text_chunking, sp))
    # `split_into_best_sentences` drops the FIRST id of each, whatever it is (the dummy prefix's piece
    # for this vocabulary); reproduced as written, not re-derived.
    w.add_array(PREFIX + "sentence_end_ids", sp.encode(".!...?", out_type=int)[1:])
    w.add_array(PREFIX + "clause_end_ids", sp.encode(",;:", out_type=int)[1:])
    w.add_int32(PREFIX + "max_tokens_per_chunk", MAX_TOKEN_PER_CHUNK)
    if sp.bos_id() < 0 or sp.eos_id() < 0:
        raise ValueError(f"{model_dir}/tokenizer.model lacks <s> or </s>, which the chunk headers use")
    # The chunk headers carry `prepare_text_prompt`'s tail guess, which the driver cannot recompute:
    # it is `len(text.split())` of the text BEFORE its terminal punctuation is fixed, and the driver
    # has ids, not text (loom.cpp ADR-044).
    w.add_int32(PREFIX + "chunk_header_short", sp.bos_id())
    w.add_int32(PREFIX + "chunk_header_long", sp.eos_id())
    w.add_int32(PREFIX + "short_chunk_max_words", SHORT_CHUNK_MAX_WORDS)
    w.add_bool(PREFIX + "capitalize_first_letter", bool(config.get("capitalize_first_letter", True)))
    w.add_bool(PREFIX + "append_terminal_punctuation",
               bool(config.get("append_terminal_punctuation", True)))
