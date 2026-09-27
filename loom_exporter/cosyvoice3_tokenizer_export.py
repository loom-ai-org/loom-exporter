"""CosyVoice3's text front end as GGUF data: `tokenizer.ggml.model = "cosyvoice3"`.

The vocabulary is the checkpoint's byte-level Qwen2 BPE with CosyVoice3's added tokens, written by
`bpe_tokenizer_export.write_bpe_vocab` under this tag -- so `loom::CosyVoice3Vocab` wraps an ordinary
`BpeVocab`. What the tag adds is the reference's text path around it, `CosyVoiceFrontEnd.text_normalize`
as the release runs it with no `ttsfrd` and no `wetext` installed (the rules path; the two
normalisation engines are a separate item), and every constant that path reads is written here, so the
engine holds only its shape (loom.cpp ADR-041's rule):

| key (`tokenizer.ggml.cosyvoice3.`) | from                                                          |
|------------------------------------|---------------------------------------------------------------|
| `markup_open`/`markup_close`       | `text_normalize`'s "skip when `<|` and `|>`" test                |
| `zh_ranges`                        | `frontend_utils.chinese_char_pattern`, as inclusive ranges     |
| `zh_pre_from`/`zh_pre_to`          | the `"\\n"` removal before `replace_blank`                       |
| `zh_replace_from`/`zh_replace_to`  | `replace_corner_mark`, the `.` and ` - ` rules, `remove_bracket` |
| `zh_trailing`/`zh_trailing_to`     | `re.sub(r'[，,、]+$', '。', text)`                               |
| `zh_enders`/`en_enders`/`closers`  | `split_paragraph`'s `pounc` lists and its closing quotes        |
| `zh_terminal`/`en_terminal`        | what `split_paragraph` appends to an unterminated text          |
| `token_max_n`/`token_min_n`/`merge_len` | `text_normalize`'s arguments to `split_paragraph`         |
| `digits`                           | `str.isdigit()`, the runs `spell_out_number` hands to inflect   |
| `decimal_from`/`decimal_value`     | the `\\d` digits inflect keeps, and `int()`'s value for each     |
| `num_*`                            | inflect's `number_to_words` word tables and default words      |
| `punct_ranges`                     | `regex`'s `[\\p{P}\\p{S}]`, `is_only_punctuation`'s class        |
| `chunk_header`                     | `<|endoftext|>`: opens every chunk `encode` returns (ADR-044)   |

**Why `<|endoftext|>` can open chunks.** A text holding `<|` and `|>` skips the whole front end and is
one chunk; any other text cannot contain the header's spelling, so no chunk's ids can hold it. The one
text that could -- a marked-up one that types `<|endoftext|>` itself -- is refused by the engine by name.

Requires: pip install gguf inflect regex
"""
import inspect
from pathlib import Path
from typing import List, Tuple

from gguf import GGUFWriter

from .bpe_tokenizer_export import write_bpe_vocab

PREFIX = "tokenizer.ggml.cosyvoice3."
CHUNK_HEADER = "<|endoftext|>"

# Literals in the reference's function bodies, restated here and pinned against the functions by
# tests/ci/test_cosyvoice3_tokenizer_export.py (which runs each of them on these tables).
MARKUP_OPEN, MARKUP_CLOSE = "<|", "|>"
ZH_PRE = [("\n", "")]
ZH_REPLACE = [
    # replace_corner_mark
    ("²", "平方"), ("³", "立方"),
    # text_normalize's own two
    (".", "。"), (" - ", "，"),
    # remove_bracket (its second backtick pair is the same character, a no-op kept for fidelity)
    ("（", ""), ("）", ""), ("【", ""), ("】", ""), ("`", ""), ("`", ""), ("——", " "),
]
ZH_TRAILING, ZH_TRAILING_TO = "，,、", "。"
ZH_ENDERS = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
EN_ENDERS = [".", "?", "!", ";", ":"]
CLOSERS = ['"', "”"]
ZH_TERMINAL, EN_TERMINAL = "。", "."
# `text_normalize`'s call: `token_max_n=80, token_min_n=60, merge_len=20, comma_split=False`.
TOKEN_MAX_N, TOKEN_MIN_N, MERGE_LEN = 80, 60, 20
# inflect's `hundfn` literal.
NUM_HUNDRED = " hundred"


def _ranges(predicate) -> List[int]:
    """Inclusive `[lo, hi, lo, hi, ...]` of the codepoints `predicate` holds for, surrogates skipped."""
    out: List[int] = []
    run = None
    for cp in range(0x110000):
        hit = not 0xD800 <= cp <= 0xDFFF and predicate(chr(cp))
        if hit and run is None:
            run = cp
        elif not hit and run is not None:
            out += [run, cp - 1]
            run = None
    if run is not None:
        out += [run, 0x10FFFF]
    return out


def _chinese_ranges() -> List[int]:
    from .cosyvoice3_export import import_cosyvoice

    import_cosyvoice()
    from cosyvoice.utils.frontend_utils import chinese_char_pattern

    return _ranges(lambda c: chinese_char_pattern.fullmatch(c) is not None)


def _decimals() -> Tuple[List[str], List[int]]:
    """Every character inflect's `\\d` keeps (`NON_DIGIT = re.compile(r"\\D")` removes the rest of an
    `isdigit()` run, so `"2²"` is "two"), with the value `int()` gives it."""
    import re

    chars, values = [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if re.fullmatch(r"\d", c):
            chars.append(c)
            values.append(int(c))
    return chars, values


def number_words() -> dict:
    """inflect's tables as `number_to_words` reads them with its default arguments."""
    import inflect

    params = inspect.signature(inflect.engine.number_to_words).parameters
    return {
        "num_units": list(inflect.unit),
        "num_teens": list(inflect.teen),
        "num_tens": list(inflect.ten),
        "num_scales": list(inflect.mill),
        "num_hundred": NUM_HUNDRED,
        "num_and": params["andword"].default,
        "num_zero": params["zero"].default,
        "num_one": params["one"].default,
    }


def write_cosyvoice3_vocab(w: GGUFWriter, tokenizer_dir: str) -> None:
    import regex
    from transformers import AutoTokenizer

    from .tokenizer_detect import detect_loom_pre_type

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    header = tok.convert_tokens_to_ids(CHUNK_HEADER)
    if header is None or header == tok.unk_token_id or tok.convert_ids_to_tokens(header) != CHUNK_HEADER:
        raise ValueError(f"{tokenizer_dir} has no {CHUNK_HEADER}, which opens each chunk")
    write_bpe_vocab(w, tokenizer_dir, pre_type=detect_loom_pre_type(tok), tokenizer_model="cosyvoice3")

    w.add_string(PREFIX + "markup_open", MARKUP_OPEN)
    w.add_string(PREFIX + "markup_close", MARKUP_CLOSE)
    w.add_array(PREFIX + "zh_ranges", _chinese_ranges())
    w.add_array(PREFIX + "zh_pre_from", [a for a, _ in ZH_PRE])
    w.add_array(PREFIX + "zh_pre_to", [b for _, b in ZH_PRE])
    w.add_array(PREFIX + "zh_replace_from", [a for a, _ in ZH_REPLACE])
    w.add_array(PREFIX + "zh_replace_to", [b for _, b in ZH_REPLACE])
    w.add_array(PREFIX + "zh_trailing", list(ZH_TRAILING))
    w.add_string(PREFIX + "zh_trailing_to", ZH_TRAILING_TO)
    w.add_array(PREFIX + "zh_enders", ZH_ENDERS)
    w.add_array(PREFIX + "en_enders", EN_ENDERS)
    w.add_array(PREFIX + "closers", CLOSERS)
    w.add_string(PREFIX + "zh_terminal", ZH_TERMINAL)
    w.add_string(PREFIX + "en_terminal", EN_TERMINAL)
    w.add_int32(PREFIX + "token_max_n", TOKEN_MAX_N)
    w.add_int32(PREFIX + "token_min_n", TOKEN_MIN_N)
    w.add_int32(PREFIX + "merge_len", MERGE_LEN)
    w.add_array(PREFIX + "digits", [chr(cp) for cp in range(0x110000)
                                     if not 0xD800 <= cp <= 0xDFFF and chr(cp).isdigit()])
    decimal_from, decimal_value = _decimals()
    w.add_array(PREFIX + "decimal_from", decimal_from)
    w.add_array(PREFIX + "decimal_value", decimal_value)
    for key, value in number_words().items():
        if isinstance(value, list):
            w.add_array(PREFIX + key, value)
        else:
            w.add_string(PREFIX + key, value)
    pattern = regex.compile(r"[\p{P}\p{S}]")
    w.add_array(PREFIX + "punct_ranges", _ranges(lambda c: pattern.fullmatch(c) is not None))
    w.add_int32(PREFIX + "chunk_header", int(header))
