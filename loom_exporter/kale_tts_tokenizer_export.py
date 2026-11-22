"""kale-tts-1-tiny's text front end as GGUF data: `tokenizer.ggml.model = "kale_tts"` (loom.cpp ADR-142).

Upstream's front end is JavaScript (`frontend/`, run by every one of its runtimes, Python's through QuickJS):
text -> Kale's terms table -> dizione's text rules (numbers, money, dates, times, units, codes, addresses) ->
sentences -> words -> IPA (a CMUdict word list with hand-checked overrides, dizione's homograph picker and a GRU
for unknown words; British voices read British files and rules) -> Kokoro's symbols -> ids, plus the Ettin
sub-tokens of the sentence (the previous one as context) and each phoneme's word. `loom::KaleTtsVocab` holds
the SHAPE of that path -- the order of the steps, each rule's replacement logic, the G2P's lookups, the GRU --
and everything a step reads is written here, from the release's own files (ADR-041's rule):

| key (`tokenizer.ggml.kale.`)        | from                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
| `rules`                             | every regular expression the JS front end runs, by name: the pattern and its flags, built the way the JS builds them (`RULES` below, pinned against the sources by tests/ci/test_kale_tts_tokenizer_export.py) |
| `tables`                            | dizione's tables (`frontend/dizione/tables.js`), its homograph data (`homographs.data.js`), the model's symbol table (`tokenizer.json`'s vocab), the constants below |
| `{us,gb}.lexicon_keys` / `_values`  | `words/lexicon.tsv` read, then `override.public.tsv` over it (`readTable`), per accent |
| `{us,gb}.g2p`                       | `g2p.bin`, the guesser's weights as the release stores them (fp16, `KG2P` v1) |
| `terms_keys` / `terms_values` / `rules.terms` | `words/terms.tsv` and the pattern `termsReplacer` builds from it |
| `nfkc_from` / `nfkc_to`             | Python's NFKC of every codepoint it changes (the normalizer runs `String.normalize("NFKC")`); the engine composes after |
| `upper_from` / `upper_to`, `lower_from` / `lower_to` | `str.upper()` / `str.lower()` where they change a codepoint (`toUpperCase`, `toLowerCase`) |

The Ettin vocabulary is written by `bpe_tokenizer_export.write_bpe_vocab` under this tag (its byte-level BPE,
NFC, GPT-2's pre-tokenizer), so the engine wraps an ordinary `BpeVocab` for it.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PREFIX = "tokenizer.ggml.kale."
SPELL, END = "", ""

# ---------------------------------------------------------------------------------------------- constants
# Each restated from the JS (file: line), and pinned there by the CI test.
MONTHS = "January February March April May June July August September October November December".split()
MONTH_PATTERN = "|".join(MONTHS) + "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
DAY_PATTERN = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
CURRENCY_CODES = "USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY"
ZONES = "ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST"
SIZES = "XXXL|XXL|XXS|XL|XS|S|M|L"
LENGTHS = r"mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?"
VERSIONED = ("version|ver|release|firmware|patch|kernel|build|python|node|nodejs|ios|ipados|macos|watchos|tvos|android|"
             "chrome|firefox|safari|java|php|ruby|rust|swift|kotlin|golang|django|rails|react|angular|vue|docker|"
             "kubernetes|ubuntu|debian|fedora|windows|linux|postgres|postgresql|mysql|redis|numpy|pytorch|tensorflow|cuda|"
             "xcode|gcc|clang|unity|minecraft|typescript|bluetooth|usb|hdmi|opengl|vulkan|directx|upgrade[ds]?|update[ds]?|downgrade[ds]?")
HALF_DAY = r"(?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b)"
AMOUNT = r"(?<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?<fraction>\d+))?"
GROUP_CORE = r"[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+"
CLOCK = r"\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?"
SYMBOL_WORDS = {"&": "and", "@": "at", "+": "plus", "=": "equals", "%": "percent", "×": "times", "÷": "divided by",
                "±": "plus or minus", "©": "copyright", "§": "section", "°": "degrees", "≥": "greater than or equal to",
                "≤": "less than or equal to", "≠": "not equal to", "≈": "approximately", "∞": "infinity",
                "√": "square root of", "π": "pi"}
EMOJI = "[\\u{1F000}-\\u{1FAFF}☀-➿\\u{1F1E6}-\\u{1F1FF}←-⇿⬀-⯿️‍⃣]+"


def js_escape(text: str) -> str:
    """spoken.js `escape` (and kale-tts.mjs's `esc`): `[.*+?^${}()|[\\]\\\\/]` backslashed."""
    return re.sub(r"[.*+?^${}()|\[\]\\/]", lambda m: "\\" + m.group(0), text)


def js_esc_terms(text: str) -> str:
    """kale-tts.mjs termsReplacer's `esc`: the same set without `/`."""
    return re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), text)


def unit_names(units: dict, exact: dict) -> str:
    """`Normalizer`'s constructor: every unit spelling, escaped, longest first, then in code-unit order."""
    names = sorted({js_escape(k) for k in [*units, *exact]}, key=lambda s: (-len(s.encode("utf-16-le")) // 2, s.encode("utf-16-be")))
    return "|".join(names)


def build_rules(units: dict, exact: dict) -> Dict[str, Tuple[str, str]]:
    names = unit_names(units, exact)
    M, D = MONTH_PATTERN, DAY_PATTERN
    r = {
        # ---- kaleTts.js
        "dotted_names": (r"(?<![\p{L}\p{N}.])((?:[\p{L}\p{N}][\p{L}\p{N}_-]*\.)+[\p{Ll}][\p{L}\p{N}_-]*)", "gu"),
        "list_comma": (r",(\s+(?:and|or)\s)", "gi"),
        "clause_split": (r"[;:]", ""),
        "space_split": (r"\s+", ""),
        "size_suffix": (r"(?<![\p{L}\p{N}.])(\d+(?:\.\d+)?)([MBKk])(?![\p{L}\p{N}])", "gu"),
        "size_times": (r"(?<![\p{L}\p{N}.])(\d+(?:\.\d+)?)[x×](?![\p{L}\p{N}])", "gu"),
        "size_seconds": (r"(?<![\p{L}\p{N}.])(\d+(?: s|\.\d+ ?s))(?![\p{L}\p{N}'’])", "gu"),
        "seconds_tail": (r" ?s$", ""),
        "dr_end": (r"\bDr\.(?=\s*$)", ""),
        "one_word": (r"[\p{L}\p{N}]+(?:['’][\p{L}]+)*", "gu"),
        "sentence_split": (r"(?<=[.!?…])\s+|\n+", ""),
        "token": (r"[A-Za-z]+(?:['’][A-Za-z]+)*|[.,!?;:…—–]", "g"),
        "vowel": (r"eɪ|aɪ|aʊ|ɔɪ|oʊ|əʊ|ɑː|iː|uː|ɜː|ɔː|oː|[aeiouæɑɒɔəɚɛɜɪʊʌᵻɐ]", "g"),
        # ---- kale-tts.mjs termsReplacer (`terms` itself is built from the table, see write_kale_tts_vocab)
        "camel": (r"(?<![\p{L}\p{N}_])[a-z]+(?:[A-Z][a-z0-9]*)+(?![\p{L}\p{N}_])", "gu"),
        "camel_split": (r"(?<=[a-z0-9])(?=[A-Z])", "g"),
        # ---- enGb.js
        "gb_groups": (r"[aeiou]+|(?<=[^aeiou])y(?![aeiou])|^y(?![aeiou])", "g"),
        "gb_silent_e": (r"[^aeiou](e|es|ed|e's)$", ""),
        "gb_strut_r": (r"(?:^|[^aeiou])(?:urr|orr|our)(?=[aeiouy])", ""),
        "gb_nurse_r": (r"^(?:fur|pur|blur|slur|spur|whir|stir|sir|occur|recur|concur|incur|demur|prefer|confer|refer|infer|"
                       r"defer|deter|transfer|aver|inter|astir|chirr|shirr|err)", ""),
        "gb_ary": (r"(ary|ory|ery|aries|ories|eries)$", ""),
        "gb_ile": (r"[^aeiou]ile[sd]?$", ""),
        "gb_isation": (r"i[sz]ation", ""),
        # ---- spoken.js: module patterns
        "money": (r"(?:(?<![\w.])(?<minus>-)\s?)?(?<symbol>[$£€¥₹]|(?<![A-Za-z])(?:" + CURRENCY_CODES + r")(?![A-Za-z]))\s?"
                  r"(?<inner>-)?" + AMOUNT + r"(?:\s?(?<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?", "g"),
        "money_after": (r"(?<![\w.,])" + AMOUNT + r"\s?(?<symbol>" + CURRENCY_CODES + r")(?![A-Za-z])", "g"),
        "vulgar": (r"(?<![\d.,/])(?:(\d+)\s?)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])", "g"),
        "address": (r"(?<url>(?:https?:\/\/|www\.)[^\s<>" '"' r"']+)"
                    r"|(?<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
                    r"|(?<file>(?<![\w@./-])(?<folder>\/?(?:[\w~-]+\/)+)?(?<stem>[\w-]+(?:\.[\w-]+)*)\.(?<ending>[A-Za-z][A-Za-z0-9]{0,9})"
                    r"(?![\w-])(?<path>\/[^\s<>" '"' r"']*)?)", "g"),
        "day_range": (r"\b(" + D + "|" + M + r")\b(\.?)\s?[-–]\s?(" + D + "|" + M + r")\b(\.?)", "g"),
        "ratio": (r"(?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d)", "g"),
        "code": (r"(?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d)", "g"),
        "slashed": (r"(?<![\w.,/-])(\d+(?:\/\d+){2,})(?![\w/]|[.,]\d)", "g"),
        "grouped": (r"(?<![\w.,:/#+-])(\+)?(?:([A-Z]{2}) (" + GROUP_CORE + r")(?: ([A-Z]{2})(?!\w))?|(" + GROUP_CORE + r"))"
                    r"(?!\w|[.,:/-]\d|\s?[%°])", "g"),
        "glued": (r"(?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d)", "g"),
        "emoji": (EMOJI, "gu"),
        "emoticon": (r"(?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|<\/3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o\/|:'[()])(?![A-Za-z0-9])"
                     r"|[:;=]['^-]?[)(]+", "g"),
        # ---- Normalizer's constructor
        "unit": (r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-–](\d[\d,]*(?:\.\d+)?))?\s?(" + names + r")(?![A-Za-z])", "gi"),
        "area": (r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([²³])", "g"),
        "area_plain": (r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?!\w)", "g"),
        "dot_times_unit": (r"^\s?(?:%|percent|million|billion|" + names + r")(?![A-Za-z])", "i"),
        "grouped_unit": (r"^\s?(?:" + names + r")(?![A-Za-z])", "i"),
        # ---- helpers inside the methods
        "before_split": (r"[.!?;…]\s|[()" '"' r"]", ""),
        "letters": (r"[A-Za-z]+", "g"),
        "word_start": (r"^[A-Za-z']+", ""),
        "next_word": (r"^\s*([A-Za-z]+)", ""),
        "next_word_or_digit": (r"^\s*([A-Za-z]+|\d)", ""),
        "model_before": (r"(?<![A-Za-z])([A-Z]{2,5}) $", ""),
        "counted_word": (r"(?:[^s]s|people|men|women|children|feet|miles)$", ""),
        "alnum_runs": (r"[A-Za-z]+|\d+", "g"),
        "digits_letters": (r"^\d+[A-Za-z]+$", ""),
        "digits_suffix": (r"^\d+(?:s|st|nd|rd|th)$", ""),
        "address_pieces": (r"[A-Za-z]+|\d+|.", "gs"),
        "address_parts": (r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", "g"),
        "address_trail": (r"[.,;:!?)]+$", ""),
        "address_scheme": (r"^(?:https?:\/\/)?(?:www\.)?", ""),
        "address_slashes": (r"\/+$", ""),
        "capital_before": (r"[A-Z][a-z]+\s+$", ""),
        "previous_word": (r"([A-Za-z]+)\.?\s+$", ""),
        "stop_small": (r"^\.\s+[a-z]", ""),
        # ---- rewrite
        "en_dash_digits": (r"(?<=\d)–(?=\d)", "g"),
        "dots3": (r"\.{3,}", "g"),
        "minus_eq": (r"(?<=\d)\s-\s(?=\d[\d.,]*\s?=)", "g"),
        "dashes": (r"-{2,}|\s-\s", "g"),
        "without": (r"\bw\/o\b", "gi"),
        "with": (r"\bw\/(?=\s|$)", "gi"),
        "not_applicable": (r"\bn\/a\b", "gi"),
        # ---- figures
        "cents": (r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?¢", "g"),
        "zone": (r"(" + CLOCK + r")\s(" + ZONES + r")\b", "g"),
        "dot_half": (r"(?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b)", "g"),
        "dot_times": (r"(?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-—])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d)", "g"),
        "clock_range": (r"(" + CLOCK + r")\s?[-—]\s?(?=\d{1,2}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b))", "g"),
        "time": (r"(?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?" + HALF_DAY + r"?(?![\d:])", "g"),
        "hour_half": (r"(?<![\d.:,])(\d{1,2})" + HALF_DAY + r"(?![A-Za-z])", "g"),
        "seen_dates": (r"(?<![\d/.])(\d{1,2})\/(\d{1,2})(?:\/(?:\d{4}|\d{2}))?(?![\d/])", "g"),
        "dated": (r"(?<![\d/])\d{1,2}\/\d{1,2}\/(?:\d{4}|\d{2})(?![\d/])", ""),
        "iso_date": (r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])", "g"),
        "ymd_slash": (r"(?<![\d/])(\d{4})\/(\d{1,2})\/(\d{1,2})(?![\d/])", "g"),
        "dot_date": (r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d)", "g"),
        "slash_date": (r"(?<![\d/])(\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2})(?![\d/])", "g"),
        "month_days": (r"\b(" + M + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])", "g"),
        "days_month": (r"(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?\s+(" + M + r")\b(?:,?\s+(\d{4}))?", "g"),
        "month_day": (r"\b(" + M + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])", "g"),
        "day_month": (r"(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s+(" + M + r")\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{4}))?", "g"),
        "phone": (r"(?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-])", "g"),
        "phone7": (r"(?<![\w.,-])(\d{3})-(\d{4})(?![\w-])", "g"),
        "version3": (r"(?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?!\w)", "g"),
        "version_v": (r"(?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w)", "g"),
        "versioned": (r"\b(" + VERSIONED + r")(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?"
                      r"(?!\d|\.\d|\s?%|\s(?:percent|million|billion)\b)", "gi"),
        "negative": (r"(?<![\w.])-(?=\d)", "g"),
        "by": (r"(?:(?<=\d)\s?[xX×]|×)\s?(?=\d[\d.,]*\s?(?:" + LENGTHS + r")\b)", "g"),
        "times": (r"(?<=\d)\s[xX×]\s(?=\d)", "g"),
        "chat": (r"(?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/'])", "g"),
        "decades": (r"(?<![\w.])'?(\d{2}|\d{4})'?s\b", "g"),
        "ordinal": (r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b", "g"),
        "resolution": (r"(?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b", "g"),
        "percent": (r"(\d[\d,]*(?:\.\d+)?)\s?%", "g"),
        "mixed": (r"(?<![\d/.,-])(\d+) (\d)\/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d)", "g"),
        "fraction": (r"(?<![\d/.])(\d+)\/(\d+)(?![\d/]|\.\d)", "g"),
        "year_range": (r"(?<![\w.,-])(\d{4})[-–](\d{4})(?![\w.,-])", "g"),
        "number_range": (r"(?<![\w.,-])(\d+)[-–](\d+)(?![\w-]|[.,]\d)", "g"),
        "nx": (r"(?<![\w.])(\d+)x\b", "g"),
        "power": (r"(?<=\d)\^(?=\d)", "g"),
        # ---- signs
        "sizes": (r"(?<![\w-])(" + SIZES + r")-(" + SIZES + r")(?![\w-])", "g"),
        "hash_number": (r"#(?=\d)", "g"),
        "hash_code": (r"#(?=[A-Z]{1,4}-?\d)", "g"),
        "hashtag": (r"#(?=[A-Za-z])", "g"),
        "about": (r"~(?=\s?\d)", "g"),
        "compare": (r"(?<=[\w\s])(>=|<=|!=)(?=[\w\s])", "g"),
        "ti": (r"(?<=\d) Ti\b", "g"),
        "ref_no": (r"\b([Rr])ef\.\s?[Nn]o\.(?=\s)", "g"),
        "dotted_letters": (r"\b((?:[A-Za-z]\.){2,})", "g"),
        "abbreviation": (r"\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'’])", "g"),
        "number": (r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+", "g"),
        "slash_words": (r"(?<=[A-Za-z])\/(?=[A-Za-z])", "g"),
        "less_greater": (r"(?<=\d)\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s)", "g"),
        "symbols": ("[" + js_escape("".join(SYMBOL_WORDS)) + "]", "g"),
        # ---- spokenText
        "spelled": (SPELL + r"([A-Za-z ]+)" + END, "g"),
        "spaces": (r"\s+", "g"),
        "space_before_mark": (r" (?=[,.;:!?…])", "g"),
        "mark_before_mark": (r"([,.;:!?…])(?=[,.;:!?…])", "g"),
        # ---- ettin.js
        "word_split": (r"\s+", ""),
    }
    return r


def terms_rule(keys: List[str]) -> Tuple[str, str]:
    """kale-tts.mjs termsReplacer's pattern: the forms longest first (a stable sort, file order among equals),
    a form starting with '.' matching inside a word too."""
    forms = sorted(keys, key=lambda f: -len(f.encode("utf-16-le")) // 2)
    alts = [js_esc_terms(f) if f.startswith(".") else r"(?<![\p{L}\p{N}_])" + js_esc_terms(f) for f in forms]
    return "(?:" + "|".join(alts) + r")(?![\p{L}\p{N}_])", "gu"


def read_table(text: str, into: Dict[str, str] = None) -> Dict[str, str]:
    """`readTable`: `word<TAB>value` lines, `#` comments and empty lines skipped; a later line wins, its key
    keeping the position the first one gave it (a JS `Map`)."""
    out = {} if into is None else into
    for raw in text.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line or line.startswith("#"):
            continue
        tab = line.find("\t")
        if tab > 0:
            out[line[:tab]] = line[tab + 1:]
    return out


def _js_module_value(path: Path, name: str):
    """`export const NAME = <json>;` from one of dizione's generated table modules."""
    for line in path.read_text().splitlines():
        m = re.match(r"export const (\w+) = (.*);$", line)
        if m and m.group(1) == name:
            return json.loads(m.group(2))
    raise ValueError(f"{path}: no `export const {name}`")


def _homograph_data(path: Path) -> dict:
    text = path.read_text()
    at = text.index("export default ") + len("export default ")
    return json.loads(text[at:].strip().rstrip(";"))


def _case_tables(fn) -> Tuple[List[int], List[str]]:
    src, dst = [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        t = fn(c)
        if t != c:
            src.append(cp)
            dst.append(t)
    return src, dst


def _nfkc_table() -> Tuple[List[int], List[str]]:
    return _case_tables(lambda c: unicodedata.normalize("NFKC", c))


def write_kale_tts_vocab(writer, model_dir: str, ettin_dir: str) -> None:
    from .bpe_tokenizer_export import write_bpe_vocab
    from .tokenizer_detect import detect_loom_pre_type
    from transformers import AutoTokenizer

    model_dir, ettin_dir = Path(model_dir), Path(ettin_dir)
    fe = model_dir / "frontend"
    # The Ettin vocabulary, the release's copy (`ettin-tokenizer.json`, byte for byte the checkpoint's).
    write_bpe_vocab(writer, str(ettin_dir), pre_type=detect_loom_pre_type(AutoTokenizer.from_pretrained(str(ettin_dir))),
                    tokenizer_model="kale_tts")

    tables_js = fe / "dizione" / "tables.js"
    units, exact = _js_module_value(tables_js, "UNITS"), _js_module_value(tables_js, "EXACT_UNITS")
    rules = build_rules(units, exact)
    terms = read_table((fe / "words" / "terms.tsv").read_text())
    if terms:
        rules["terms"] = terms_rule(list(terms))
    names = sorted(rules)
    writer.add_array(PREFIX + "rule_names", names)
    writer.add_array(PREFIX + "rule_patterns", [rules[n][0] for n in names])
    writer.add_array(PREFIX + "rule_flags", [rules[n][1] for n in names])
    writer.add_array(PREFIX + "terms_keys", list(terms) or [""])
    writer.add_array(PREFIX + "terms_values", list(terms.values()) or [""])

    vocab = json.loads((model_dir / "tokenizer.json").read_text())["model"]["vocab"]
    tables = {
        name: _js_module_value(tables_js, name)
        for name in ("ABBREVIATIONS", "UNITS", "EXACT_UNITS", "CHAT", "ENDINGS", "HOMOGRAPH_RULES")
    }
    tables["HOMOGRAPHS"] = _homograph_data(fe / "dizione" / "homographs.data.js")
    tables["VOCAB"] = vocab
    tables["SYMBOL_WORDS"] = SYMBOL_WORDS
    writer.add_string(PREFIX + "tables", json.dumps(tables, ensure_ascii=False))

    for accent, sub in (("us", fe / "words"), ("gb", fe / "words" / "en-gb")):
        lex = read_table((sub / "lexicon.tsv").read_text())
        read_table((sub / "override.public.tsv").read_text(), lex)
        writer.add_array(PREFIX + accent + ".lexicon_keys", list(lex))
        writer.add_array(PREFIX + accent + ".lexicon_values", list(lex.values()))
        # `bytes` is what gguf-py writes as a UINT8 array.
        writer.add_array(PREFIX + accent + ".g2p", (sub / "g2p.bin").read_bytes())

    for name, (src, dst) in (("nfkc", _nfkc_table()), ("upper", _case_tables(str.upper)),
                             ("lower", _case_tables(str.lower))):
        writer.add_array(PREFIX + name + "_from", src)
        writer.add_array(PREFIX + name + "_to", dst)
