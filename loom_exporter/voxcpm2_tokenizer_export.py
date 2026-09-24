"""Writes VoxCPM2's text front end into a GGUF, `tokenizer.ggml.model`="voxcpm2".

A new tag, for [ADR-033](../../loom.cpp/docs/adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)'s
reason: the ids the model was trained on are not what `tokenizer.json` alone produces. The reference
tokenizes through `mask_multichar_chinese_tokens` (`voxcpm/model/utils.py`), which splits every
multi-character Chinese piece back into single characters -- "你好" is two ids, never one -- and the
tokenizer proper is a `tokenizers` BPE that is none of the engine's other schemes: character-level (not
byte-level, so not "gpt2"), merged by RANK rather than by score (so not "llama"), with byte fallback.

What ships, all of it DATA rather than code in `loom::VoxCpmVocab`:

* **the table, the merges and the added tokens** -- the tokenizer proper. Added tokens are matched
  literally before normalization, as `tokenizers` extracts them, and they are the union of
  `tokenizer.json`'s and `tokenizer_config.json`'s (`added_tokens`);
* **the normalizer's two strings**: the `Prepend` and the `Replace` of `" "`, read from the file, with
  the file checked to be exactly that sequence -- anything else raises rather than exports a
  tokenizer the engine would apply differently;
* **the split table**: for every piece whose `▁`-stripped text is in the reference's own
  `multichar_tokens` set, the ids of its characters, as the wrapper converts them (an absent
  character is `<unk>`, as `convert_tokens_to_ids` makes it).

The reference's text preparation (`VoxCPM._generate`: newlines to spaces, runs of whitespace to one)
is the engine's, as Chatterbox's `" ".join(text.split())` is: a function's shape, not a model constant.

Requires: pip install gguf
"""
import json
from pathlib import Path
from typing import Dict, List

from gguf import GGUFWriter

_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_BYTE = 6


def read_tokenizer(tokenizer_dir: str) -> dict:
    spec = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    model = spec["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"VoxCPM2's tokenizer.json is a {model.get('type')!r} model, not BPE")
    if spec.get("pre_tokenizer") is not None:
        raise ValueError("VoxCPM2's tokenizer.json has a pre-tokenizer; loom::VoxCpmVocab merges the "
                         "whole normalized text as one word")
    for key in ("continuing_subword_prefix", "end_of_word_suffix", "dropout"):
        if model.get(key):
            raise ValueError(f"VoxCPM2's BPE sets {key}={model[key]!r}, which loom::VoxCpmVocab does not do")
    if not model.get("byte_fallback"):
        raise ValueError("VoxCPM2's BPE has no byte fallback; loom::VoxCpmVocab always falls back to bytes")
    if model.get("ignore_merges"):
        raise ValueError("VoxCPM2's BPE sets ignore_merges, which loom::VoxCpmVocab does not do")
    return spec


def normalizer_strings(spec: dict) -> Dict[str, str]:
    """`Sequence[Prepend(p), Replace(" " -> r)]`, the only normalizer the engine implements."""
    norm = spec.get("normalizer") or {}
    steps = norm.get("normalizers") if norm.get("type") == "Sequence" else None
    if (not steps or len(steps) != 2 or steps[0].get("type") != "Prepend" or steps[1].get("type") != "Replace"
            or steps[1].get("pattern") != {"String": " "}):
        raise ValueError(f"VoxCPM2's normalizer is {json.dumps(norm)[:200]}, not Prepend then Replace(' ')")
    return {"prepend": steps[0]["prepend"], "space": steps[1]["content"]}


def split_table(tokens: List[str], unk_id: int) -> Dict[int, List[int]]:
    """The reference wrapper's rule, over the table: a piece whose `▁`-stripped text is a multi-character
    run of CJK Unified Ideographs that is ITSELF a piece becomes that run's characters' ids."""
    piece_to_id = {t: i for i, t in enumerate(tokens)}
    multichar = {t for t in tokens if len(t) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in t)}
    table = {}
    for i, t in enumerate(tokens):
        clean = t.replace("\u2581", "")
        if clean in multichar:
            table[i] = [piece_to_id.get(c, unk_id) for c in clean]
    return table


def added_tokens(tokenizer_dir: str, spec: dict) -> List[dict]:
    """Every token `LlamaTokenizerFast.from_pretrained` extracts before the model sees the text: the
    `tokenizer.json`'s own added tokens AND `tokenizer_config.json`'s `added_tokens_decoder`, which
    registers 15 more -- `<|audio_start|>` (101) among them -- that the BPE table also holds as plain
    pieces. Measured without the second list: 70 of 198 texts that type a special token differed.
    Each must be a plain literal (no stripping, no normalizing, not whole-word-only), which is the
    only matching `loom::VoxCpmVocab` does.

    **Two of the config's entries are not the table's, and are left out.** It names ids 103 and 104
    `<|audio_prompt_start|>`/`<|audio_prompt_end|>`, but those rows of the table are
    `<|ref_audio_start|>`/`<|ref_audio_end|>` -- the names `VoxCPM2Model` uses. transformers resolves the
    clash by minting NEW ids for the config's spellings, 73448 and 73449, past the model's 73448-row
    embedding, so a text that types either one cannot run in the reference at all. Here it tokenizes as
    the characters it is made of."""
    by_id = {a["id"]: a for a in spec.get("added_tokens", [])}
    table = {i: piece for piece, i in spec["model"]["vocab"].items()}
    config = Path(tokenizer_dir) / "tokenizer_config.json"
    if config.is_file():
        for key, a in json.loads(config.read_text(encoding="utf-8")).get("added_tokens_decoder", {}).items():
            entry = dict(a, id=int(key))
            if entry["id"] in by_id and by_id[entry["id"]]["content"] != entry["content"]:
                raise ValueError(f"added token {entry['id']} is {by_id[entry['id']]['content']!r} in "
                                 f"tokenizer.json and {entry['content']!r} in tokenizer_config.json")
            if table.get(entry["id"], entry["content"]) != entry["content"]:
                continue
            by_id.setdefault(entry["id"], entry)
    for a in by_id.values():
        if a.get("lstrip") or a.get("rstrip") or a.get("single_word") or a.get("normalized"):
            raise ValueError(f"added token {a['content']!r} is not a plain literal")
    return [by_id[i] for i in sorted(by_id)]


def write_voxcpm2_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    spec = read_tokenizer(tokenizer_dir)
    model = spec["model"]
    vocab = model["vocab"]
    added = added_tokens(tokenizer_dir, spec)
    size = max(list(vocab.values()) + [a["id"] for a in added]) + 1
    tokens = [""] * size
    for piece, i in vocab.items():
        tokens[i] = piece
    types = [_TOKEN_TYPE_NORMAL] * size
    for a in added:
        if tokens[a["id"]] not in ("", a["content"]):
            raise ValueError(f"added token {a['content']!r} disagrees with vocab row {a['id']}")
        tokens[a["id"]] = a["content"]
        types[a["id"]] = _TOKEN_TYPE_CONTROL
    if any(t == "" for t in tokens):
        raise ValueError("tokenizer.json's vocab and added tokens leave holes in the id range")
    for i, t in enumerate(tokens):
        if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            types[i] = _TOKEN_TYPE_BYTE
    if sum(1 for t in types if t == _TOKEN_TYPE_BYTE) != 256:
        raise ValueError("byte fallback needs all 256 <0xNN> pieces")
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    unk_id = vocab[model["unk_token"]]
    norm = normalizer_strings(spec)
    table = split_table(tokens, unk_id)

    writer.add_tokenizer_model("voxcpm2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_unk_token_id(unk_id)
    p = "tokenizer.ggml.voxcpm2."
    writer.add_array(p + "added_tokens", [a["content"] for a in added])
    writer.add_string(p + "prepend", norm["prepend"])
    writer.add_string(p + "space", norm["space"])
    froms = sorted(table)
    writer.add_array(p + "split_from", froms)
    offsets, flat = [0], []
    for i in froms:
        flat.extend(table[i])
        offsets.append(len(flat))
    writer.add_array(p + "split_offsets", offsets)
    writer.add_array(p + "split_to", flat)
