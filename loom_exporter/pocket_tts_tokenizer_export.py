"""Pocket-TTS's text front end as GGUF data: `tokenizer.ggml.model = "pocket_tts"`.

The vocabulary is an ordinary SentencePiece Unigram model with byte fallback, written by
`spm_tokenizer_export.write_sentencepiece_vocab` under this tag. What the tag adds is the reference's
text path around it -- `prepare_text_prompt` and `split_into_best_sentences` -- and every constant
that path reads is written here FROM the reference, so `loom::PocketTtsVocab` holds only its shape
(loom.cpp ADR-041's rule):

| key (`tokenizer.ggml.pocket_tts.`) | from                                                        |
|------------------------------------|-------------------------------------------------------------|
| `replace_from`/`replace_to`        | `prepare_text_prompt`'s `str.replace` calls, in order        |
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
