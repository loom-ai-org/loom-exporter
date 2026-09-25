"""CosyVoice3's text front end as data (`cosyvoice3_tokenizer_export`).

The tables below are literals in the reference's function bodies, restated in the writer; each is pinned
here by RUNNING the reference function it came from, so an edit to either side goes red. The claim that
the engine applies them as the reference does -- 9000/9000 texts, ids identical per chunk -- is the
engine's (tests/ci/test_cosyvoice3_vocab.cpp names the harness).
"""
import re
from pathlib import Path

import pytest

from loom_exporter.cosyvoice3_export import COSYVOICE_REPO
from loom_exporter.cosyvoice3_tokenizer_export import (
    CLOSERS, EN_ENDERS, EN_TERMINAL, MARKUP_CLOSE, MARKUP_OPEN, MERGE_LEN, TOKEN_MAX_N, TOKEN_MIN_N, ZH_ENDERS,
    ZH_PRE, ZH_REPLACE, ZH_TERMINAL, ZH_TRAILING, ZH_TRAILING_TO, number_words,
)


@pytest.fixture(scope="module")
def utils():
    if not Path(COSYVOICE_REPO, "cosyvoice").is_dir():
        pytest.skip("no FunAudioLLM/CosyVoice checkout")
    from loom_exporter.cosyvoice3_export import import_cosyvoice

    import_cosyvoice()
    from cosyvoice.utils import frontend_utils

    return frontend_utils


def _apply(table, text):
    for a, b in table:
        text = text.replace(a, b)
    return text


def test_the_zh_replacement_table_is_the_references_sequence(utils):
    """`replace_corner_mark`, then `.` -> `。` and ` - ` -> `，` (in `text_normalize`), then
    `remove_bracket` -- on text holding every character any of them looks at."""
    probes = ["面积3²和4³", "a.b - c", "（括号）【方括号】`代码`——破折号", "x - y.z²（）", "——————", "² - ³."]
    for text in probes:
        expected = utils.remove_bracket(utils.replace_corner_mark(text).replace(".", "。").replace(" - ", "，"))
        assert _apply(ZH_REPLACE, text) == expected, text
    assert ZH_PRE == [("\n", "")]


def test_the_trailing_comma_rule_is_the_references_regex():
    for text in ["好，", "好,，、", "好，好", "好"]:
        expected = re.sub(r"[，,、]+$", "。", text)
        stripped = text.rstrip(ZH_TRAILING)
        assert (stripped + ZH_TRAILING_TO if stripped != text else text) == expected


def test_the_enders_closers_and_terminals_are_split_paragraphs(utils):
    """Every BMP character is asked: is it an ender (the text splits after it), a closer (it stays with
    the sentence before), and what does an unterminated text get?"""
    for lang, enders, terminal in (("en", EN_ENDERS, EN_TERMINAL), ("zh", ZH_ENDERS, ZH_TERMINAL)):
        # Zero budgets: every non-empty piece closes as the next sentence arrives, so the result is the
        # sentences themselves.
        split = lambda text: utils.split_paragraph(text, list, lang, token_max_n=0, token_min_n=0, merge_len=0)
        found = [chr(cp) for cp in range(0x20, 0x10000) if not 0xD800 <= cp <= 0xDFFF
                 and len(split("ab" + chr(cp) + "cd")) == 2]
        assert sorted(found) == sorted(enders), lang
        assert split("ab") == ["ab" + terminal]
        closers = [chr(cp) for cp in range(0x20, 0x10000) if not 0xD800 <= cp <= 0xDFFF
                   and chr(cp) not in enders and split("ab." + chr(cp) + "cd")[0] == "ab." + chr(cp)]
        assert sorted(closers) == sorted(CLOSERS), lang


def test_the_budgets_are_text_normalizes_arguments(utils):
    """`text_normalize` calls `split_paragraph(..., token_max_n=80, token_min_n=60, merge_len=20,
    comma_split=False)` for both languages -- read off its source, which is where they are."""
    import inspect

    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    source = inspect.getsource(CosyVoiceFrontEnd.text_normalize)
    calls = re.findall(r"token_max_n=(\d+),\s*token_min_n=(\d+), merge_len=(\d+), comma_split=(\w+)", source)
    assert calls == [(str(TOKEN_MAX_N), str(TOKEN_MIN_N), str(MERGE_LEN), "False")] * 2
    assert f"'{MARKUP_OPEN}' in text and '{MARKUP_CLOSE}' in text" in source


def test_the_number_words_reproduce_inflect():
    """A direct transcription of `enword` over the shipped tables agrees with inflect itself on the
    shapes the engine test pins -- a guard on the TABLES (the algorithm is the engine's)."""
    import inflect

    w = number_words()
    engine = inflect.engine()
    assert engine.number_to_words("1998") == "one thousand, nine hundred and ninety-eight"
    assert engine.number_to_words("2026") == f"two{w['num_scales'][1]} {w['num_and']} twenty-six"
    assert w["num_units"][1] == "one" and w["num_teens"][0] == "ten" and w["num_tens"][9] == "ninety"
    assert w["num_scales"][0] == " " and len(w["num_scales"]) == 12
    assert engine.number_to_words("0") == w["num_zero"] and engine.number_to_words("1") == w["num_one"]
    assert engine.number_to_words("100") == "one" + w["num_hundred"]
