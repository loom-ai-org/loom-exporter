"""A text front end's number speller, shipped as DATA for `loom::NumberSpeller` (loom.cpp ADR-059).

SpeechT5's SentencePiece vocabulary has no digits, so `2026` reaches the model as one `<unk>`. The
reference's answer is `SpeechT5Tokenizer(normalize=True)`, which runs transformers'
`EnglishNumberNormalizer` before tokenizing: it strips thousands separators and spells each number as
words (`$15,000.5` -> `fifteen thousand point five dollars`). This module reads that function's
DATA off the reference -- its word tables, its currency names, the order of the currency chain in its
pattern -- and writes it as `tokenizer.ggml.numbers.*` keys. The engine holds the function's SHAPE
(ADR-041's split), so the words stay the reference's and the C++ stays language-free.

Python's `re` decides two character classes by Unicode properties, and the engine must agree with it
exactly: `\\d` is category Nd (and `int()` reads any Nd digit), `\\w` is `str.isalnum()` or `_`. Both
are computed here from this interpreter's `unicodedata`, the same tables the reference runs on, and
shipped as ranges: `digit_zeros` (the zero of every Nd run of ten) and `word_ranges` (inclusive pairs).
"""
import sys
import unicodedata
from typing import Dict, List

SCHEME = "english_number_normalizer"

# The optional prefix chain in `EnglishNumberNormalizer.__call__`'s pattern, in the pattern's order:
# `-?\\$?\\€?\\£?\\¢?\\¥?\\₹?\\₽?\\฿?\\₺?\\₴?\\₣?\\₡?\\₱?\\₪?\\₮?\\₩?\\₦?\\₫?\\﷼?`. It is NOT the order of
# `currency_symbols` (which puts `﷼` sixth), and the two orders mean different things: the chain decides
# what a match may consume, the dict which name `convert` gives it. `pattern_chain()` checks this list
# against the reference's own source.
SYMBOL_CHAIN = ["-", "$", "€", "£", "¢", "¥", "₹", "₽", "฿", "₺", "₴", "₣", "₡", "₱", "₪", "₮", "₩", "₦", "₫", "﷼"]


def pattern_chain() -> List[str]:
    """The prefix chain as the reference's `__call__` spells it, read out of its source."""
    import inspect
    import re

    from transformers.models.speecht5.number_normalizer import EnglishNumberNormalizer

    source = inspect.getsource(EnglishNumberNormalizer.__call__)
    pattern = re.search(r'pattern = r"\(\?<!\\w\)\((.*?)\\d\+', source).group(1)
    return [m.group(1) for m in re.finditer(r"\\?(.)\?", pattern)]


def digit_zeros() -> List[int]:
    """The zero of every run of Nd digits. Every Nd run in Unicode is ten consecutive codepoints
    valued 0..9, which the engine relies on and this checks."""
    zeros = []
    for cp in range(sys.maxunicode + 1):
        c = chr(cp)
        if c.isdecimal() and unicodedata.decimal(c) == 0:
            for k in range(10):
                if not (chr(cp + k).isdecimal() and unicodedata.decimal(chr(cp + k)) == k):
                    raise ValueError(f"U+{cp:04X}: an Nd run that is not ten consecutive digits")
            zeros.append(cp)
    n_decimal = sum(1 for cp in range(sys.maxunicode + 1) if chr(cp).isdecimal())
    if n_decimal != 10 * len(zeros):
        raise ValueError(f"{n_decimal} Nd codepoints, but {len(zeros)} runs of ten")
    return zeros


def word_ranges() -> List[int]:
    """`re`'s `\\w` for a str pattern -- `c.isalnum() or c == '_'` -- as flat inclusive [lo, hi] pairs."""
    ranges, start = [], None
    for cp in range(sys.maxunicode + 2):
        inside = cp <= sys.maxunicode and (chr(cp).isalnum() or cp == 0x5F)
        if inside and start is None:
            start = cp
        elif not inside and start is not None:
            ranges += [start, cp - 1]
            start = None
    return ranges


def english_number_normalizer_kv() -> Dict[str, object]:
    """Every `tokenizer.ggml.numbers.*` value, read off the reference."""
    from transformers.models.speecht5.number_normalizer import EnglishNumberNormalizer

    ref = EnglishNumberNormalizer()
    chain = pattern_chain()
    if chain != SYMBOL_CHAIN:
        raise ValueError(f"EnglishNumberNormalizer's pattern chain changed: {chain}")
    if [len(ref.ones), len(ref.teens), len(ref.tens)] != [10, 10, 10] or len(ref.thousands) < 2:
        raise ValueError("EnglishNumberNormalizer's word tables changed shape")
    symbols = list(ref.currency_symbols)
    if any(len(s) != 1 for s in symbols + SYMBOL_CHAIN):
        raise ValueError("a currency symbol is not one codepoint; the engine matches single codepoints")
    return {
        "scheme": SCHEME,
        "ones": list(ref.ones),
        "teens": list(ref.teens),
        "tens": list(ref.tens),
        "scales": list(ref.thousands),
        "symbol_chain": SYMBOL_CHAIN,
        "currency_symbols": symbols,
        "currency_names": [ref.currency_symbols[s] for s in symbols],
        "digit_zeros": digit_zeros(),
        "word_ranges": word_ranges(),
    }


def write_number_normalizer(writer, kv: Dict[str, object]) -> None:
    for key, value in kv.items():
        name = f"tokenizer.ggml.numbers.{key}"
        if isinstance(value, str):
            writer.add_string(name, value)
        elif value and isinstance(value[0], str):
            writer.add_array(name, list(value))
        else:
            writer.add_array(name, [int(v) for v in value])
