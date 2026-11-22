"""kale-tts-1-tiny's export (kale_tts_export.py, kale_tts_tokenizer_export.py; loom.cpp ADR-138).

The Ettin wrapper is held to transformers' own ModernBERT on a small random encoder whose sequence outruns its
sliding window; the front end's rules to upstream's JavaScript sources, verbatim, when a release is on disk
(`LOOM_KALE_TTS_DIR`, else skipped); the table helpers to the JS semantics they restate. The model itself is
held to upstream's code by loom.cpp's test_e2e_kale_tts.
"""
import os
import re
import unittest
from pathlib import Path

import torch

from loom_exporter import kale_tts_export as K
from loom_exporter import kale_tts_tokenizer_export as T

RELEASE = Path(os.environ.get("LOOM_KALE_TTS_DIR", os.path.expanduser("~/Dev/models/kale-tts-1-tiny")))


class TestEttinWrapper(unittest.TestCase):
    def test_the_wrapper_is_modernberts_eager_forward(self):
        """Batch-free attention, the per-layer rotary tables and the sliding window, against HF's eager path,
        on a sequence three windows long so the local layers' mask is not all zeros."""
        from transformers import ModernBertConfig, ModernBertModel

        torch.manual_seed(0)
        cfg = ModernBertConfig(vocab_size=200, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                               num_attention_heads=2, global_attn_every_n_layers=3, local_attention=8,
                               global_rope_theta=160000.0, local_rope_theta=10000.0, pad_token_id=0,
                               attn_implementation="eager")
        model = ModernBertModel(cfg).eval()
        ids = torch.randint(1, 200, (24,))
        with torch.no_grad():
            hs = model(input_ids=ids[None], attention_mask=torch.ones_like(ids[None]), output_hidden_states=True).hidden_states
            want = ((hs[-1] + hs[-2]) / 2)[0]
            got = K._EttinWrapper(model).eval()(ids.int())
        self.assertLess(float((got - want).abs().max()), 1e-5)


class TestTables(unittest.TestCase):
    def test_read_table_lets_a_later_line_win_in_the_first_ones_place(self):
        t = T.read_table("# c\na\t1\nb\t2\r\n\na\t3\nnotab\n")
        self.assertEqual(list(t.items()), [("a", "3"), ("b", "2")])

    def test_unit_names_sort_longest_first_then_by_code_unit(self):
        self.assertEqual(T.unit_names({"km": 0, "m": 0, "mph": 0}, {"L": 0, "MW": 0}), "mph|MW|km|L|m")

    def test_js_escape_backslashes_the_syntax_characters(self):
        self.assertEqual(T.js_escape("a.b/c(d)"), r"a\.b\/c\(d\)")
        self.assertEqual(T.js_esc_terms("a.b/c"), r"a\.b/c")

    def test_the_terms_pattern_is_longest_first_and_stable(self):
        pattern, flags = T.terms_rule(["api", ".js", "aws", "nodejs"])
        self.assertEqual(flags, "gu")
        self.assertEqual(pattern, r"(?:(?<![\p{L}\p{N}_])nodejs|(?<![\p{L}\p{N}_])api|\.js|(?<![\p{L}\p{N}_])aws)(?![\p{L}\p{N}_])")


@unittest.skipUnless((RELEASE / "frontend" / "dizione" / "spoken.js").is_file(), "needs a kale-tts-1-tiny release (LOOM_KALE_TTS_DIR)")
class TestRulesAreUpstreams(unittest.TestCase):
    """Every rule written as a regex literal upstream is that literal, flags and all; every constant the
    built rules interpolate is upstream's constant."""

    LITERAL = ["dotted_names", "list_comma", "size_suffix", "size_times", "size_seconds", "seconds_tail", "dr_end",
               "one_word", "sentence_split", "token", "vowel", "camel", "camel_split", "gb_groups", "gb_silent_e",
               "gb_strut_r", "gb_ary", "gb_ile", "gb_isation", "vulgar", "ratio", "code", "slashed", "glued",
               "area", "area_plain", "letters", "word_start", "next_word", "next_word_or_digit", "model_before",
               "counted_word", "alnum_runs", "digits_letters", "digits_suffix", "address_pieces", "address_parts",
               "address_trail", "address_scheme", "address_slashes", "capital_before", "previous_word", "stop_small",
               "en_dash_digits", "dots3", "minus_eq", "dashes", "without", "with", "not_applicable", "cents",
               "dot_half", "dot_times", "seen_dates", "dated", "iso_date", "ymd_slash", "dot_date", "slash_date",
               "phone", "phone7", "version3", "version_v", "negative", "times", "chat", "decades", "ordinal",
               "resolution", "percent", "mixed", "fraction", "year_range", "number_range", "nx", "power",
               "hash_number", "hash_code", "hashtag", "about", "compare", "ti", "ref_no", "dotted_letters",
               "abbreviation", "number", "slash_words", "less_greater", "space_before_mark", "mark_before_mark"]

    @classmethod
    def setUpClass(cls):
        fe = RELEASE / "frontend"
        cls.src = "\n".join(p.read_text() for p in [fe / "dizione" / "spoken.js", fe / "words" / "kaleTts.js",
                                                     fe / "words" / "enGb.js", RELEASE / "js" / "kale-tts.mjs"])
        tables = fe / "dizione" / "tables.js"
        cls.rules = T.build_rules(T._js_module_value(tables, "UNITS"), T._js_module_value(tables, "EXACT_UNITS"))

    def test_each_literal_rule_is_upstreams_literal(self):
        for name in self.LITERAL:
            with self.subTest(rule=name):
                pattern, flags = self.rules[name]
                g = flags.replace("g", "")
                hits = [f"/{pattern}/{f}" for f in {flags, "g" + g, g, g + "g"}]
                self.assertTrue(any(h in self.src for h in hits), f"{name}: /{pattern}/{flags} is not in upstream's sources")

    def test_each_interpolated_constant_is_upstreams(self):
        for const in ("ZONES", "SIZES", "LENGTHS", "VERSIONED", "CURRENCY_CODES", "DAY_PATTERN", "GROUP_CORE", "HALF_DAY", "AMOUNT"):
            with self.subTest(constant=const):
                m = re.search(r"const " + const + r" = ((?:String\.raw`[^`]*`|\"[^\"]*\"(?:\s*\+\s*\"[^\"]*\")*))", self.src)
                self.assertIsNotNone(m, const)
                text = m.group(1)
                value = text[len("String.raw`"):-1] if text.startswith("String.raw") else "".join(re.findall(r"\"([^\"]*)\"", text))
                self.assertEqual(value, getattr(T, const))

    def test_the_month_pattern_is_upstreams(self):
        self.assertIn('MONTHS.join("|") + "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"', self.src)


if __name__ == "__main__":
    unittest.main()
