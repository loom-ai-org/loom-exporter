"""Writes a Tekken vocabulary (`tekken.json`, Mistral's tiktoken format) into a GGUF as byte-level BPE:
`tokenizer.ggml.model`="gpt2", `tokenizer.ggml.pre`="tekken".

Tekken is what every Mistral checkpoint since Nemo ships -- here, Voxtral-4B-TTS's. `mistral_common`'s
`Tekkenizer` builds a `tiktoken.Encoding` from it, and this writes the same three things down:

* **the ids**: the file's `default_num_special_tokens` markers first (`<unk>`, `<s>`, `</s>`, `[INST]`,
  ..., `[AUDIO]` at 24, then `<SPECIAL_n>` fillers), then the first `default_vocab_size - n_special`
  RANKS, in rank order. A rank's id is `rank + n_special`, which is `Tekkenizer.encode`'s own offset, so
  an id here is the id the model reads;
* **each rank's bytes**, spelled through GPT-2's byte-to-unicode map as every "gpt2" vocabulary is --
  a rank is raw bytes (`token_bytes`, base64), often not UTF-8 on its own;
* **the regex** as the `tekken` pretokenizer shape, with the file's pattern CHECKED against the one
  `loom::BpeVocab` implements: a different pattern raises rather than exports a tokenizer the engine
  would split differently.

No merges are written. tiktoken merges the adjacent pair whose concatenation has the lowest rank; the
engine asks the rank directly (`BpeShape::kTekken`), which is that rule exactly, where a merge list
rebuilt from the ranks is it only when each token's canonical split is the pair that meets.

The markers are typed CONTROL, so a host can tell them from text when decoding -- and the engine does
NOT split input on them, because tiktoken is built with `special_tokens={}`: a typed "[AUDIO]" is text.

Requires: pip install gguf
"""
import base64
import json
from pathlib import Path

from gguf import GGUFWriter

_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3

# The one pattern `BpeShape::kTekken` scans (bpe_vocab.h spells it out). Checked, not assumed.
TEKKEN_PATTERN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
                  r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
                  r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _bytes_to_unicode() -> dict:
    """GPT-2's byte -> printable-codepoint map (the table `loom::BpeVocab` reverses on decode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def read_tekken(tokenizer_dir: str) -> dict:
    path = Path(tokenizer_dir) / "tekken.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    config = spec["config"]
    if config.get("pattern") != TEKKEN_PATTERN:
        raise ValueError(f"{path}'s pattern is {config.get('pattern')!r}, not the one loom::BpeVocab's "
                         "`tekken` shape implements")
    return spec


def tekken_ids(spec: dict):
    """`(pieces, types, n_special)`: the markers' own spellings, then each kept rank's bytes spelled
    through GPT-2's byte map -- `Tekkenizer.__init__`'s truncation to `default_vocab_size`, as it is."""
    config = spec["config"]
    n_special = int(config["default_num_special_tokens"])
    vocab_size = int(config["default_vocab_size"])
    specials = spec["special_tokens"]
    if len(specials) > n_special:
        raise ValueError(f"tekken.json lists {len(specials)} markers for {n_special} slots")
    pieces, types = [], []
    for i in range(n_special):
        # `Tekkenizer.from_file` fills the unlisted marker slots with `<SPECIAL_{i}>`.
        if i < len(specials):
            if specials[i]["rank"] != i:
                raise ValueError(f"marker {i} has rank {specials[i]['rank']}")
            pieces.append(specials[i]["token_str"])
        else:
            pieces.append(f"<SPECIAL_{i}>")
        types.append(_TOKEN_TYPE_CONTROL)
    byte_map = _bytes_to_unicode()
    ranks = spec["vocab"][:vocab_size - n_special]
    for i, entry in enumerate(ranks):
        if entry["rank"] != i:
            raise ValueError(f"rank entry {i} says rank {entry['rank']}")
        raw = base64.b64decode(entry["token_bytes"])
        if i < 256 and raw != bytes([i]):
            raise ValueError(f"rank {i} is {raw!r}, not the single byte tiktoken requires there")
        pieces.append("".join(byte_map[b] for b in raw))
        types.append(_TOKEN_TYPE_NORMAL)
    if len(set(pieces[n_special:])) != len(pieces) - n_special:
        raise ValueError("two ranks spell the same bytes")
    return pieces, types, n_special


def write_tekken_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    spec = read_tekken(tokenizer_dir)
    pieces, types, n_special = tekken_ids(spec)
    by_text = {p: i for i, p in enumerate(pieces[:n_special])}
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("tekken")
    writer.add_token_list(pieces)
    writer.add_token_types(types)
    writer.add_bos_token_id(by_text["<s>"])
    writer.add_eos_token_id(by_text["</s>"])
    writer.add_unk_token_id(by_text["<unk>"])
    # `Tekkenizer.encode(s, bos=False, eos=False)` is how a speech request encodes its text; a model
    # that wants the markers places them itself (Voxtral's driver builds its whole prompt).
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
