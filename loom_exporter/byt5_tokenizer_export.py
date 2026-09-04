"""Extracts vocab data for a real HF raw-UTF-8-byte tokenizer directory (`tokenizer_config.json` only --
neither family here has a `tokenizer.json`/`tokenizer.model` at all, see tokenizer_detect.py's own doc
comment for how they get detected in the first place) and writes it into a GGUF file,
`tokenizer.ggml.model`="byt5".

**Two families, one scheme, and the constants are read rather than assumed.** ByT5 was the first and
gave the tag its name; Dia's `DiaTokenizer` is the same idea parameterised differently, and the
difference is exactly the three KVs this module now writes:

|                 | `ByT5Tokenizer`                      | `DiaTokenizer`                        |
|-----------------|--------------------------------------|---------------------------------------|
| byte offset     | 3 (pad/eos/unk precede the range)    | 0 (`offset`, a real config field)     |
| appends eos     | yes, unconditionally                 | no -- it has no eos token at all      |
| extra ids       | `<extra_id_N>` above the range       | `[S1]`/`[S2]` at 1/2, INSIDE it       |
| vocab size      | 3 + 256 + extra_ids                  | 256                                   |

Everything ByT5-specific below was confirmed directly against a real `transformers.ByT5Tokenizer`
instance's actual saved `tokenizer_config.json`, not assumed from its docstring (which describes a
different, non-matching sentinel-ordering scheme). Concretely: `added_tokens_decoder` (a real, on-disk
id -> {"content": ...} mapping) has entries "0"/"1"/"2" for pad/eos/unk (fixed positions, hardcoded by
every real `ByT5Tokenizer.__init__`, not a per-checkpoint choice) and one entry per T5-style
span-corruption sentinel ("<extra_id_N>") at ids 259, 260, ... (sequential, right after the 256-entry
byte range) -- NOT reversed/counted-from-the-end as the upstream docstring claims. The top-level
`extra_ids` config field is unreliable (`ByT5Tokenizer.__init__` always passes `extra_ids=0` to its base
class, regardless of the real sentinel count) -- the real count is derived from `added_tokens_decoder`
directly instead.

**Dia's speaker tags are what made `tokenizer.ggml.token_type` necessary here.** `[S1]` and `[S2]` are
ids 1 and 2, which under Dia's own offset of 0 are also byte values -- so the file has to say that
those two ids are ADDED tokens rather than bytes, or `loom::ByteVocab` would spell `"[S1]"` as five
literal bytes on the way in and return byte 0x01 on the way out. The KV is the one `BpeVocab` already
reads for the identical purpose (P4.23), with the identical CONTROL/USER_DEFINED meaning, which is why
it is reused rather than invented.

Byte-range piece text is deliberately left as empty placeholders in the written token list --
`loom::ByteVocab` computes those arithmetically at load time rather than storing them (see that class's
own doc comment for why storing them as normal GGUF token strings would silently corrupt any byte >=
0x80). An added token that falls inside the range is the one exception, and it carries real text.

Requires: pip install gguf
"""
import json
from pathlib import Path

from gguf import GGUFWriter

_BYTE_RANGE_SIZE = 256

# gguf's own `TokenType` values, spelled out rather than imported for the same reason
# `bpe_tokenizer_export` spells them out: this module keeps working against a gguf release that moves
# the enum. 1 NORMAL, 3 CONTROL (a special marker), 4 USER_DEFINED (added but not special).
_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4

# ByT5 hardcodes these three positions in every real `__init__`; they are not a per-checkpoint choice.
_BYT5_PAD_ID, _BYT5_EOS_ID, _BYT5_UNK_ID = 0, 1, 2
_BYT5_BYTE_OFFSET = 3


def write_byte_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    """Dispatch on the checkpoint's own `tokenizer_class`, which is the only thing on disk that says
    which parameterisation this is."""
    tok_dir = Path(tokenizer_dir)
    config = json.loads((tok_dir / "tokenizer_config.json").read_text())
    tokenizer_class = config.get("tokenizer_class")
    if tokenizer_class == "ByT5Tokenizer":
        return _write_byt5(writer, config)
    if tokenizer_class == "DiaTokenizer":
        return _write_dia(writer, config)
    raise ValueError(
        f"write_byte_vocab: expected tokenizer_class in ('ByT5Tokenizer', 'DiaTokenizer'), got "
        f"{tokenizer_class!r}. Both are raw-UTF-8-byte schemes; a third would be added here with its "
        f"own byte offset, eos behaviour and added-token set rather than by widening either branch."
    )


def _added_tokens(config: dict) -> dict:
    """`{id: content}` from `added_tokens_decoder`, whose values are either a dict or a bare string."""
    decoder = config.get("added_tokens_decoder") or {}
    return {int(k): (v["content"] if isinstance(v, dict) else v) for k, v in decoder.items()}


def _write_byt5(writer: GGUFWriter, config: dict) -> None:
    added = _added_tokens(config)
    for required in (_BYT5_PAD_ID, _BYT5_EOS_ID, _BYT5_UNK_ID):
        if required not in added:
            raise ValueError(
                f"write_byte_vocab: a ByT5Tokenizer config must declare added_tokens_decoder entries "
                f"for ids 0/1/2 (pad/eos/unk); id {required} is missing"
            )

    # Every added_tokens_decoder entry at an id >= the byte range's end is a sentinel -- ByT5 never adds
    # any other kind of token beyond pad/eos/unk (ids 0-2) and sentinels.
    sentinel_ids = sorted(i for i in added if i >= _BYT5_BYTE_OFFSET + _BYTE_RANGE_SIZE)
    if sentinel_ids and sentinel_ids != list(range(sentinel_ids[0], sentinel_ids[0] + len(sentinel_ids))):
        raise NotImplementedError(
            f"write_byte_vocab: sentinel ids {sentinel_ids} are not contiguous -- unexpected ByT5 "
            "tokenizer layout, not supported")

    vocab_size = _BYT5_BYTE_OFFSET + _BYTE_RANGE_SIZE + len(sentinel_ids)
    tokens = [""] * vocab_size
    token_types = [_TOKEN_TYPE_NORMAL] * vocab_size
    for i in (_BYT5_PAD_ID, _BYT5_EOS_ID, _BYT5_UNK_ID):
        tokens[i] = added[i]
        token_types[i] = _TOKEN_TYPE_CONTROL
    for sid in sentinel_ids:
        tokens[sid] = added[sid]
        token_types[sid] = _TOKEN_TYPE_USER_DEFINED

    writer.add_tokenizer_model("byt5")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_pad_token_id(_BYT5_PAD_ID)
    writer.add_eos_token_id(_BYT5_EOS_ID)
    writer.add_unk_token_id(_BYT5_UNK_ID)
    # Both are ByT5's own values and both are `ByteVocab`'s defaults, so writing them changes nothing
    # about how this family loads. They are written anyway because a file that states its own constants
    # is readable without knowing which family produced it -- and because the day a reader sees a
    # `byte_offset` of 0 it should be because the file said 0, not because the KV was absent.
    writer.add_uint32("tokenizer.ggml.byte_offset", _BYT5_BYTE_OFFSET)
    writer.add_bool("tokenizer.ggml.add_eos_token", True)


def _write_dia(writer: GGUFWriter, config: dict) -> None:
    """`DiaTokenizer`: bytes at their own ordinal, no eos, `[S1]`/`[S2]` shadowing bytes 1 and 2.

    **The pad token shadows byte 0 and the speaker tags shadow bytes 1 and 2, and that is the
    checkpoint's design rather than a collision to resolve.** `DiaTokenizer._convert_token_to_id` maps
    `ord(c) + offset` with `offset` 0, so those three byte values have no other spelling -- and HF
    reaches the tags through `added_tokens_encoder` before the byte path ever runs, which is exactly
    the order `loom::ByteVocab::encode` applies. A literal 0x00-0x02 byte in someone's text is not
    representable either way, in HF or here.
    """
    offset = int(config.get("offset", 0))
    added = _added_tokens(config)
    vocab_size = offset + _BYTE_RANGE_SIZE
    if any(i >= vocab_size for i in added):
        raise NotImplementedError(
            f"write_byte_vocab: this DiaTokenizer declares added tokens at ids "
            f"{sorted(i for i in added if i >= vocab_size)}, past the end of its {vocab_size}-token "
            f"vocabulary. Dia's own tags all fall inside the byte range; a checkpoint that appends "
            f"tokens above it needs its vocab size derived from those ids instead."
        )

    tokens = [""] * vocab_size
    token_types = [_TOKEN_TYPE_NORMAL] * vocab_size
    for i, content in added.items():
        tokens[i] = content
        # The pad token is a marker the model emits about its own decode; the speaker tags are things a
        # caller WRITES. That is exactly the CONTROL/USER_DEFINED distinction, and `ByteVocab` treats
        # both as added -- so the split is recorded because it is true, not because it changes anything
        # here.
        token_types[i] = (_TOKEN_TYPE_CONTROL if content == config.get("pad_token")
                          else _TOKEN_TYPE_USER_DEFINED)

    pad_id = next((i for i, c in added.items() if c == config.get("pad_token")), 0)

    writer.add_tokenizer_model("byt5")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_pad_token_id(pad_id)
    # `unk_token` is `<pad>` for this tokenizer -- the same id, and that is what the config says rather
    # than a fallback chosen here.
    writer.add_unk_token_id(pad_id)
    writer.add_uint32("tokenizer.ggml.byte_offset", offset)
    # **No eos, and no `add_eos_token=False` alone would be enough.** This tokenizer has no eos token
    # to append, so the id is never written either: `ByteVocab` defaults `eos_id_` to 1, which under
    # this offset is `[S1]`, and a file that said "do not append" while still naming 1 as its eos would
    # be one flag away from ending every prompt with a speaker tag.
    writer.add_bool("tokenizer.ggml.add_eos_token", False)
