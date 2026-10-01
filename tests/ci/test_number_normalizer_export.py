"""The number speller's data (loom.cpp ADR-059), read off transformers' `EnglishNumberNormalizer`.

The engine holds the function's shape and trusts these tables completely, so what is checked here is
that they ARE the reference's: the symbol chain against the pattern in the reference's own source, and
the two Unicode classes against Python's `re`, which is what the reference's regexes use.
"""
import re

import pytest

pytest.importorskip("transformers")

from loom_exporter.number_normalizer_export import (  # noqa: E402
    SYMBOL_CHAIN, digit_zeros, english_number_normalizer_kv, pattern_chain, word_ranges,
    write_number_normalizer,
)


def test_the_chain_is_the_patterns_own():
    assert pattern_chain() == SYMBOL_CHAIN


def test_the_digit_table_is_res_backslash_d():
    zeros = digit_zeros()
    digits = {z + k for z in zeros for k in range(10)}
    for cp in list(range(0x0, 0x3000)) + list(range(0xFF00, 0xFF20)) + [0x1D7CE, 0x1FBF0]:
        assert (cp in digits) == bool(re.fullmatch(r"\d", chr(cp))), hex(cp)


def test_the_word_table_is_res_backslash_w():
    flat = word_ranges()
    ranges = list(zip(flat[::2], flat[1::2]))

    def inside(cp):
        return any(lo <= cp <= hi for lo, hi in ranges)

    for cp in list(range(0x0, 0x3000)) + [0x5F, 0xA0, 0x2160, 0x1F600, 0x20000]:
        assert inside(cp) == bool(re.fullmatch(r"\w", chr(cp))), hex(cp)


def test_every_key_is_written_under_the_numbers_namespace():
    class Recorder:
        def __init__(self):
            self.kv = {}

        def add_string(self, k, v):
            self.kv[k] = v

        def add_array(self, k, v):
            self.kv[k] = v

    kv = english_number_normalizer_kv()
    rec = Recorder()
    write_number_normalizer(rec, kv)
    assert rec.kv["tokenizer.ggml.numbers.scheme"] == "english_number_normalizer"
    assert rec.kv["tokenizer.ggml.numbers.currency_names"][0] == " dollars"
    assert len(rec.kv["tokenizer.ggml.numbers.ones"]) == 10
    assert all(isinstance(v, int) for v in rec.kv["tokenizer.ggml.numbers.word_ranges"])
