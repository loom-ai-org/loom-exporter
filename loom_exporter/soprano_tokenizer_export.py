"""Writes Soprano's text front end into a GGUF, `tokenizer.ggml.model`="soprano".

A new tag (ADR-033: a tag answers "which scheme") because two things are new at once:

* **the tokenizer** is a `tokenizers` BPE over plain characters -- no byte-level mapping, a `Lowercase`
  + whitespace-collapse normalizer, and an individual-digit split ahead of `\\s+|\\w+|[^\\w\\s]+` -- which is
  none of the engine's BPE shapes ("gpt2" would byte-map, Chatterbox's has no digit split or normalizer);
* **the text path around it** is `SopranoTTS._preprocess_text`: `clean_text` (tortoise-tts's English
  normaliser -- `unidecode`, numbers through inflect, abbreviations, symbols, punctuation clean-up),
  `split_and_recombine_text` into sentences, short sentences merged into a neighbour, and each sentence
  wrapped as `[STOP][TEXT]{sentence}[START]` -- one generation each.

**Every rule is the reference's own pattern string, shipped as data** (ADR-041), and run by the engine's
`loom::PyRegex`, a matcher for the subset of Python `re` these patterns use. That is the point of the
design: a hand-written scanner per pattern would be forty-odd re-derivations to get exactly right, where
this ships the strings the reference compiles. Three sources, each checked here:

* the module-level compiled regexes and tables of `soprano.utils.text_normalizer`, read as attributes;
* the patterns written INLINE in function bodies (`remove_unknown_characters`' class, the splitter's
  three...), which cannot be read as attributes -- each is spelled here as its source literal and the
  export refuses unless that literal appears verbatim in `inspect.getsource` of its function;
* inflect's number words and the four clean-up patterns `number_to_words` runs (module attributes).

Also data: `unidecode`'s table (every non-ASCII codepoint it maps to something, 41,379 of them, packed
as codepoints + offsets + one string) and the splitter's and merger's lengths, read off the signatures.

What stays in C++ is each callback's SHAPE (`_expand_time`, `_expand_dollars`, ...) with its few literal
words, which live in function bodies the export cannot read any more truthfully than the engine can.
"""
import ast
import inspect
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from gguf import GGUFWriter

_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3

P = "tokenizer.ggml.soprano."

# The patterns written inline in a function body, as (key, function name, pattern literal, replacement
# literal or None when the replacement is a callback). Each literal is checked against the function's
# source, then evaluated: what ships is exactly what the reference compiles.
INLINE_RULES: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("date_split", "_expand_date", "'[./-]'", None),
    ("phone_non_digit", "_expand_phone_number", "r'\\D'", "''"),
    ("paren_open", "_expand_parantheses", "r'[\\(\\[\\{]'", "', '"),
    ("paren_close_inner", "_expand_parantheses", "r'[\\)\\]\\}][^$.!?,]'", "', '"),
    ("paren_close", "_expand_parantheses", "r'[\\)\\]\\}]'", "''"),
    ("camel_part", "_split_mixedcase", "'[A-Z][a-z]*'", None),
    ("unknown_chars", "remove_unknown_characters",
     "r\"[^A-Za-z !\\$%&'\\*\\+,-./0123456789<>\\?_]\"", "\"\""),
    ("unknown_symbols", "remove_unknown_characters", "r\"[<>/_+]\"", "\"\""),
    ("whitespace", "collapse_whitespace", "r'\\s+'", "' '"),
    ("space_before_punct", "collapse_whitespace", "r' [.\\?!,]'", None),
    ("ellipsis", "dedup_punctuation", "r\"\\.\\.\\.+\"", "\"[ELLIPSIS]\""),
    ("commas", "dedup_punctuation", "r\",+\"", "\",\""),
    ("periods", "dedup_punctuation", "r\"[\\.,]*\\.[\\.,]*\"", "\".\""),
    ("exclamations", "dedup_punctuation", "r\"[\\.,!]*![\\.,!]*\"", "\"!\""),
    ("questions", "dedup_punctuation", "r\"[\\.,!\\?]*\\?[\\.,!\\?]*\"", "\"?\""),
    ("ellipsis_back", "dedup_punctuation", "r\"\\[ELLIPSIS\\]\"", "\"...\""),
    ("triple_letters", "collapse_triple_letters", "r'(\\w)\\1{2,}'", None),
    ("split_newlines", "split_and_recombine_text", "r'\\n\\n+'", "'\\n'"),
    ("split_whitespace", "split_and_recombine_text", "r'\\s+'", "' '"),
    ("split_quotes", "split_and_recombine_text", "r'[“”]'", "'\"'"),
    ("split_empty", "split_and_recombine_text", "r'^[\\s\\.,;:!?]*$'", None),
)

# The module-level regexes, by attribute; each callback's shape is in `loom::SopranoVocab`.
NAMED_RULES = (
    "_num_prefix_re", "_num_suffix_re", "_num_letter_split_re", "_comma_number_re", "_date_re",
    "_phone_number_re", "_time_re", "_pounds_re", "_dollars_re", "_decimal_number_re", "_multiply_re",
    "_divide_re", "_add_re", "_subtract_re", "_fraction_re", "_ordinal_re", "_number_re",
    "_link_header_re", "_dash_re", "_dot_re", "_parentheses_re", "_camelcase_re",
)
# `normalize_numbers`' one template replacement (`re.sub(_pounds_re, r'\1 pounds', text)`).
POUNDS_TEMPLATE = "r'\\1 pounds'"

# inflect's own clean-up patterns, run by `number_to_words` on each chunk.
INFLECT_RULES = ("NON_DIGIT", "WHITESPACES_COMMA", "COMMA_WORD", "WHITESPACES")


def import_reference():
    from .soprano_export import import_soprano

    import_soprano()
    from soprano.utils import text_normalizer, text_splitter
    from soprano import tts
    return text_normalizer, text_splitter, tts


def _function(modules, name):
    for module in modules:
        fn = getattr(module, name, None)
        if fn is not None:
            return fn
    raise ValueError(f"the reference has no function {name!r}")


def inline_rules(modules) -> List[Tuple[str, str, bool, Optional[str]]]:
    """`INLINE_RULES` checked against the reference's source and evaluated: (key, pattern, icase,
    template or None)."""
    out = []
    for key, fn_name, pattern_literal, repl_literal in INLINE_RULES:
        source = inspect.getsource(_function(modules, fn_name))
        for literal in (pattern_literal, repl_literal):
            if literal is not None and literal not in source:
                raise ValueError(f"{fn_name}'s source no longer spells {literal}; the reference changed, "
                                 f"and {key!r} must be re-read from it")
        out.append((key, ast.literal_eval(pattern_literal), False,
                    None if repl_literal is None else ast.literal_eval(repl_literal)))
    return out


def named_rules(text_normalizer) -> List[Tuple[str, str, bool]]:
    import re

    out = []
    for name in NAMED_RULES:
        rx = getattr(text_normalizer, name)
        if rx.flags & ~(re.IGNORECASE | re.UNICODE):
            raise ValueError(f"{name} carries flags {rx.flags!r}; only IGNORECASE is implemented")
        out.append((name.strip("_"), rx.pattern, bool(rx.flags & re.IGNORECASE)))
    return out


def table_rules(table) -> List[Tuple[str, bool, str]]:
    import re

    out = []
    for rx, repl in table:
        if rx.flags & ~(re.IGNORECASE | re.UNICODE):
            raise ValueError(f"{rx.pattern!r} carries flags {rx.flags!r}; only IGNORECASE is implemented")
        out.append((rx.pattern, bool(rx.flags & re.IGNORECASE), repl))
    return out


def unidecode_table() -> Tuple[List[int], List[int], str]:
    """`unidecode(chr(cp))` for every non-ASCII codepoint it maps to something, packed: codepoints,
    offsets into one string (n + 1 of them), and the string. `unidecode` is per character, so the
    table is the whole function; an absent codepoint is `''`, its `errors='ignore'` default."""
    from unidecode import unidecode

    cps, offsets, text = [], [0], []
    n = 0
    for cp in range(0x80, sys.maxunicode + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        r = unidecode(chr(cp))
        if not r:
            continue
        if not r.isascii():
            raise ValueError(f"unidecode(U+{cp:04X}) = {r!r} is not ASCII")
        cps.append(cp)
        text.append(r)
        n += len(r)
        offsets.append(n)
    return cps, offsets, "".join(text)


def read_tokenizer(tokenizer_dir: str) -> dict:
    spec = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text())
    model = spec["model"]
    problems = []
    if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("continuing_subword_prefix") \
            or model.get("end_of_word_suffix") or model.get("fuse_unk") or model.get("dropout"):
        problems.append(f"its model is not a plain character BPE ({ {k: v for k, v in model.items() if k not in ('vocab', 'merges')} })")
    norm = spec.get("normalizer") or {}
    if norm != {"type": "Sequence", "normalizers": [
            {"type": "Lowercase"}, {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}]}:
        problems.append(f"its normalizer is {norm}")
    pre = spec.get("pre_tokenizer") or {}
    if pre != {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "Split", "pattern": {"Regex": "\\s+|\\w+|[^\\w\\s]+"}, "behavior": "Isolated",
             "invert": False}]}:
        problems.append(f"its pre-tokenizer is {pre}")
    if spec.get("post_processor") is not None:
        problems.append("it has a post-processor")
    if problems:
        raise ValueError("Soprano's tokenizer.json is not the shape loom::SopranoVocab implements: " +
                         "; ".join(problems))
    return spec


def write_soprano_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    import inflect

    text_normalizer, text_splitter, tts = import_reference()
    modules = (text_normalizer, text_splitter, tts.SopranoTTS)

    spec = read_tokenizer(tokenizer_dir)
    model = spec["model"]
    vocab = model["vocab"]
    size = max(vocab.values()) + 1
    tokens = [""] * size
    for piece, i in vocab.items():
        tokens[i] = piece
    if any(t == "" for t in tokens):
        raise ValueError("tokenizer.json's vocab has holes in its id range")
    types = [_TOKEN_TYPE_NORMAL] * size
    for a in spec["added_tokens"]:
        if tokens[a["id"]] != a["content"]:
            raise ValueError(f"added token {a['content']!r} disagrees with vocab row {a['id']}")
        types[a["id"]] = _TOKEN_TYPE_CONTROL
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]

    # The prompt `_preprocess_text` builds, checked against its source rather than assumed.
    prompt = "f'[STOP][TEXT]{item[\"text\"]}[START]'"
    if prompt not in inspect.getsource(tts.SopranoTTS._preprocess_text):
        raise ValueError("_preprocess_text no longer builds f'[STOP][TEXT]{...}[START]'")

    writer.add_tokenizer_model("soprano")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_unk_token_id(vocab[model["unk_token"]])
    writer.add_int32(P + "stop_id", vocab["[STOP]"])
    writer.add_int32(P + "text_id", vocab["[TEXT]"])
    writer.add_int32(P + "start_id", vocab["[START]"])

    rules = [(k, p, ic) for k, p, ic in named_rules(text_normalizer)]
    templates = {}
    for key, pattern, icase, repl in inline_rules(modules):
        rules.append((key, pattern, icase))
        if repl is not None:
            templates[key] = repl
    templates["pounds_re"] = ast.literal_eval(POUNDS_TEMPLATE)
    if POUNDS_TEMPLATE not in inspect.getsource(text_normalizer.normalize_numbers):
        raise ValueError("normalize_numbers no longer substitutes _pounds_re with r'\\1 pounds'")
    for name in INFLECT_RULES:
        rules.append(("inflect_" + name.lower(), getattr(inflect, name).pattern, False))
    writer.add_array(P + "rule_names", [k for k, _, _ in rules])
    writer.add_array(P + "rule_patterns", [p for _, p, _ in rules])
    writer.add_array(P + "rule_icase", [int(ic) for _, _, ic in rules])
    writer.add_array(P + "template_names", list(templates))
    writer.add_array(P + "template_values", list(templates.values()))

    # The ordered tables, each a (pattern, icase, replacement) list applied rule by rule.
    for key, table in (("preunicode", text_normalizer._preunicode_special_characters),
                       ("abbreviations", text_normalizer._abbreviations + text_normalizer._cased_abbreviations),
                       ("special", text_normalizer._special_characters)):
        rows = table_rules(table)
        writer.add_array(P + key + "_patterns", [p for p, _, _ in rows])
        writer.add_array(P + key + "_icase", [int(ic) for _, ic, _ in rows])
        writer.add_array(P + key + "_repl", [r for _, _, r in rows])

    # inflect's words, from its module (the tables `engine.number_to_words` reads).
    writer.add_array(P + "inflect_unit", list(inflect.unit))
    writer.add_array(P + "inflect_teen", list(inflect.teen))
    writer.add_array(P + "inflect_ten", list(inflect.ten))
    writer.add_array(P + "inflect_mill", list(inflect.mill))
    writer.add_array(P + "inflect_ordinal_from", list(inflect.ordinal))
    writer.add_array(P + "inflect_ordinal_to", list(inflect.ordinal.values()))
    writer.add_array(P + "inflect_nth_suffixes", sorted(inflect.nth_suff))

    # The lengths: the splitter's defaults and `_preprocess_text`'s merge threshold.
    split = inspect.signature(text_splitter.split_and_recombine_text).parameters
    writer.add_int32(P + "desired_length", int(split["desired_length"].default))
    writer.add_int32(P + "max_length", int(split["max_length"].default))
    merge = inspect.signature(tts.SopranoTTS._preprocess_text).parameters
    writer.add_int32(P + "min_length", int(merge["min_length"].default))

    cps, offsets, text = unidecode_table()
    writer.add_array(P + "unidecode_cps", cps)
    writer.add_array(P + "unidecode_offsets", offsets)
    writer.add_string(P + "unidecode_text", text)
