"""Writes PLaMo's tokenizer into a GGUF, `tokenizer.ggml.model`="plamo" (loom::PlamoVocab).

`Plamo3Tokenizer` (pfnet, remote code beside the checkpoint) is a unigram model decoded by Viterbi, run
as an Aho-Corasick walk from the end of the text, with byte fallback and hard boundaries around
`<|plamo:...|>` tags and around long runs of spaces or of one character. None of the engine's other
schemes: not SentencePiece (no normalizer, no `▁`, integer scores with a longest-piece tie rule).

What ships, all of it DATA (the algorithm is `loom::PlamoVocab`'s):

* `tokenizer.ggml.tokens` -- every row of `tokenizer.jsonl`, by id;
* `tokenizer.ggml.plamo.scores` -- `round(score * 1e4)` as int32, the reference's own rounding
  (`AhoCorasick.build`); a non-finite score is the reference's INVALID_SCORE;
* `tokenizer.ggml.plamo.types` -- 0 NORMAL, 1 CONTROL, 2 BYTE, 3 UNKNOWN;
* `tokenizer.ggml.plamo.break_spaces` / `break_repeats` -- `tokenizer_config.json`'s two thresholds
  (`break_around_consecutive_spaces_threshold`, `..._repeated_chars_threshold`), 0 for none;
* the bos/eos/pad/unk ids, from the config's own token names.

Requires: pip install gguf
"""
import json
import math
from pathlib import Path

from gguf import GGUFWriter

TYPES = {"NORMAL": 0, "CONTROL": 1, "BYTE": 2, "UNKNOWN": 3}
INVALID_SCORE = -20000000


def is_plamo_tokenizer(tokenizer_dir: str) -> bool:
    d = Path(tokenizer_dir)
    try:
        config = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (d / "tokenizer.jsonl").is_file() and str(config.get("tokenizer_class", "")).startswith("Plamo")


def read_plamo_vocab(tokenizer_dir: str) -> dict:
    d = Path(tokenizer_dir)
    rows = [json.loads(line) for line in (d / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines() if line]
    config = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        if len(row) < 3 or not isinstance(row[0], str) or row[2] not in TYPES:
            raise ValueError(f"tokenizer.jsonl row {i} is {row!r}; loom::PlamoVocab reads [piece, score, type]")
    bytes_seen = {int(r[0][3:5], 16) for r in rows if r[2] == "BYTE"}
    if bytes_seen != set(range(256)):
        raise ValueError("tokenizer.jsonl does not hold a <0xNN> piece for every byte; the reference asserts it")
    ids = {r[0]: i for i, r in enumerate(rows)}

    def token_id(name):
        piece = config.get(name)
        return ids.get(piece, -1) if isinstance(piece, str) else -1

    return dict(
        pieces=[r[0] for r in rows],
        scores=[round(float(r[1]) * 1e4) if math.isfinite(float(r[1])) else INVALID_SCORE for r in rows],
        types=[TYPES[r[2]] for r in rows],
        break_spaces=int(config.get("break_around_consecutive_spaces_threshold") or 0),
        break_repeats=int(config.get("break_around_repeated_chars_threshold") or 0),
        bos=token_id("bos_token"), eos=token_id("eos_token"), pad=token_id("pad_token"),
        unk=token_id("unk_token"),
    )


def write_plamo_vocab(w: GGUFWriter, tokenizer_dir: str) -> dict:
    v = read_plamo_vocab(tokenizer_dir)
    w.add_tokenizer_model("plamo")
    w.add_token_list(v["pieces"])
    # Element types stated, not inferred: `add_array` would type a list of Python ints INT32 for both, and
    # loom::PlamoVocab reads the types as UINT8 (`kv_arr_u8`).
    from gguf import GGUFValueType

    w.add_key_value("tokenizer.ggml.plamo.scores", [int(x) for x in v["scores"]], GGUFValueType.ARRAY,
                    sub_type=GGUFValueType.INT32)
    w.add_key_value("tokenizer.ggml.plamo.types", [int(x) for x in v["types"]], GGUFValueType.ARRAY,
                    sub_type=GGUFValueType.UINT8)
    w.add_int32("tokenizer.ggml.plamo.break_spaces", v["break_spaces"])
    w.add_int32("tokenizer.ggml.plamo.break_repeats", v["break_repeats"])
    for key, value in (("bos", v["bos"]), ("eos", v["eos"]), ("padding", v["pad"]), ("unknown", v["unk"])):
        if value >= 0:
            w.add_uint32(f"tokenizer.ggml.{key}_token_id", value)
    return v
