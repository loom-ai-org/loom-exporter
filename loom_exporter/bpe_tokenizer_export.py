"""Extracts byte-level-BPE vocab data from a real HF tokenizer directory (`tokenizer.json` +
`tokenizer_config.json`) and writes it into a GGUF file using llama.cpp's own "tokenizer.ggml.*" schema
for a "gpt2"-style vocab -- the same schema `tools/convert_qwen3/qwen3_tokenizer.py`'s `write_bpe_vocab`
writes and `loom::BpeVocab` (include/loom/core/bpe_vocab.h) reads back, generalized to also handle:

- tokenizer.json schema variants where `model.merges` is a list of `[a, b]` pairs (LFM2's own
  tokenizer.json) rather than pre-joined "a b" strings (Qwen3's own tokenizer.json) -- both normalized to
  the "a b" format `loom::BpeVocab::load` expects.
- an explicit `pre_type` ("qwen2" default, or "llama3" for LFM2's grouped-up-to-3-digit pretokenizer
  regex variant, `\\p{N}{1,3}` -- see bpe_vocab.h's own doc comment) written as `tokenizer.ggml.pre`,
  dispatched on by `loom::BpeVocab` at load time. Not auto-detected from the regex string: per
  EXPORT-BACKLOG.md item 4's own plan, tokenizer family/variant selection is a bounded, one-time choice
  made by each model's own export script, not a generic regex-sniffing framework.
- `tokenizer_config.json`'s `add_bos_token`, needed because LFM2 (unlike Qwen3) prepends a BOS token to
  every encoded sequence per its own `TemplateProcessing` post-processor.
- **`tokenizer.ggml.token_type` (P4.23)**, without which the file does not say which of its ids are
  ADDED tokens and `BpeVocab::encode` cannot emit one atomically -- `encode("<|im_start|>")` comes back
  as seven literal ids, and a chat template is not un-applied but unrepresentable.
- **the checkpoint's FULL end-of-sequence set (P4.23)**, read from `generation_config.json` rather than
  from the tokenizer config's single `eos_token`. gemma-3-270m-it declares `eos_token_id: [1, 106]` --
  `<eos>` and `<end_of_turn>` -- and a loop knowing only the first runs to `max_new_tokens` on every
  chat turn. `granite_speech_export.py` and `qwen3_asr_export.py` already read that file for exactly
  this; the causal-LM path was the one that did not.

Requires: pip install gguf
"""
import json
from pathlib import Path

from gguf import GGUFWriter

# gguf's own `TokenType` values, spelled out rather than imported so this module keeps working against a
# gguf release whose enum module moves. `loom::BpeVocab` reads back only CONTROL and USER_DEFINED (see
# `added_to_id_`); the rest are written because the KV is defined as parallel to the token list, and a
# partially-filled array is a worse thing to hand a reader than a complete one.
_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4
_TOKEN_TYPE_BYTE = 6


def _is_byte_piece(piece: str) -> bool:
    """`<0xNN>`, the SentencePiece byte-fallback spelling. Marked BYTE rather than NORMAL so that a
    reader can tell a fallback entry from a token whose text happens to look like one -- and so the
    added-token pre-pass never splits on it, which would turn the literal text "<0x41>" into byte 0x41.
    """
    return (len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")
            and all(c in "0123456789ABCDEFabcdef" for c in piece[3:5]))


def read_eos_token_ids(model_dir) -> list[int]:
    """`generation_config.json`'s `eos_token_id`, always as a list, or [] when there is none.

    The CHECKPOINT's statement of where generation stops, which is not the same question as which
    single token the tokenizer calls `eos_token` -- an instruction-tuned checkpoint ends a turn on a
    marker its base model never emitted. Read from the generation config because that is where the
    authors wrote it down, and because two other families in this repo already read it from there.
    """
    path = Path(model_dir) / "generation_config.json"
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text()).get("eos_token_id")
    except (json.JSONDecodeError, OSError):
        return []
    if value is None:
        return []
    return [int(v) for v in (value if isinstance(value, list) else [value])]


def read_sampling_defaults(model_dir) -> dict:
    """`generation_config.json`'s sampling knobs, normalized to the three the engine implements
    (P4.24): `{"temperature": float, "top_k": int, "top_p": float}`.

    **`do_sample: false`, or no generation config at all, becomes `temperature = 0.0`** -- the engine's
    own spelling of greedy, and the reason one number rather than a separate flag decides it. Greedy is
    also what an absent file gets, which keeps every existing baseline where it is: a checkpoint that
    never asked to be sampled is not sampled.

    `top_k = 0` and `top_p = 1.0` mean "no truncation" to the engine. **A knob the file leaves out
    gets the value `generate()` uses**, `_GENERATE_DEFAULTS`, and for `top_k` that is 50, not 0.
    This function used to map a missing `top_k` to 0, which shipped an untruncated sampler the
    reference never runs. MusicGen, `llama-3.2-1b`, `csm-1b` and `parler-tts` all set `do_sample`
    without naming `top_k`.

    The defaults are applied here rather than read off a resolved `GenerationConfig`, because the
    resolved object depends on which transformers version is installed. 4.x stores the defaults on
    the object; 5.x stores `None` and fills them in inside `generate()`. MusicGen's first export
    read `GenerationConfig.top_k` and would have shipped 0 under the 5.x venv. An explicit `null`
    is refused, because the two versions disagree on it: 4.x reads it as "no truncation", while
    5.x reads it as "unset" and applies the default.
    """
    path = Path(model_dir) / "generation_config.json"
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}
    if not cfg.get("do_sample"):
        return {"temperature": 0.0, "top_k": 0, "top_p": 1.0}
    nulls = [k for k in _GENERATE_DEFAULTS if k in cfg and cfg[k] is None]
    if nulls:
        raise ValueError(
            f"{path} sets do_sample with {nulls} = null. transformers 4.x reads that as 'no "
            f"truncation' and 5.x as 'unset, use {[_GENERATE_DEFAULTS[k] for k in nulls]}', so the "
            f"reference's own behaviour depends on its version. Decide which one the export follows "
            f"and pass it explicitly."
        )
    resolved = {k: cfg.get(k, default) for k, default in _GENERATE_DEFAULTS.items()}
    return {
        "temperature": float(resolved["temperature"]),
        "top_k": int(resolved["top_k"]),
        "top_p": float(resolved["top_p"]),
    }


# What `generate()` samples with when `generation_config.json` leaves a knob out. These are the
# same three values on transformers 4.x (`GenerationConfig.__init__`) and 5.x
# (`GenerationConfig._get_default_generation_params`); `test_sampling_defaults_match_transformers`
# checks them against the installed version.
_GENERATE_DEFAULTS = {"temperature": 1.0, "top_k": 50, "top_p": 1.0}


def _tokenizer_json_text(tok_dir: Path) -> str:
    """`tokenizer.json`'s text, MATERIALISING it from a `vocab.json` + `merges.txt` pair if that is
    all the checkpoint ships.

    The classic GPT-2 layout predates the fast-tokenizer file, and Qwen3-TTS still ships it where
    every other Qwen checkpoint in this tree ships `tokenizer.json`. The pair holds the same
    vocabulary and the same merges, but not in this reader's schema, and the difference is not only
    formatting: `added_tokens.json` and `tokenizer_config.json` carry the added tokens with their
    flags, and the pre-tokenizer regex lives in the config rather than beside the merges.

    So the conversion is `tokenizers`' own, through a round trip, rather than reimplemented here --
    which is the same argument [ADR-027] makes about not re-deriving a tokenizer's own id order.
    Verified against the reference's tokenization of a real prompt, id for id.
    """
    direct = tok_dir / "tokenizer.json"
    if direct.exists():
        return direct.read_text()

    import tempfile

    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as staging:
        AutoTokenizer.from_pretrained(str(tok_dir)).save_pretrained(staging)
        written = Path(staging) / "tokenizer.json"
        if not written.exists():                     # pragma: no cover - a slow tokenizer
            raise NotImplementedError(
                f"{tok_dir} has vocab.json + merges.txt but its tokenizer has no fast backend, so "
                "there is no tokenizer.json to write and nothing on disk states the pre-tokenizer "
                "regex this writer needs"
            )
        return written.read_text()


# The `tokenizer.ggml.pre` names loom.cpp reads as `BpeShape::kSpmByteFallback`: a SentencePiece model
# converted to `tokenizer.json` (literal UTF-8 pieces, U+2581 for a space, `<0xNN>` byte fallback).
SPM_BYTE_FALLBACK_PRES = frozenset({"granite-embed-multi-311m", "granite-embed-multi-97m",
                                    "spm-byte-fallback"})
_SPM_SPACE = "\u2581"


def _spm_dummy_prefix(tokenizer_json: dict) -> bool:
    """Whether a SentencePiece-converted `tokenizer.json` prepends U+2581 to the text -- the dummy
    prefix, which loom.cpp reads as `tokenizer.ggml.add_space_prefix` on this shape and nowhere else.

    The engine implements exactly the pair the converter writes: `Prepend("\u2581")` in the normalizer
    and `Strip(" ", 1, 0)` after `Fuse` in the decoder. A prefix without that strip would decode with a
    leading space the reference does not produce, so the two are required together."""
    def walk(node, key):
        if isinstance(node, dict):
            yield node
            for child in node.get(key) or []:
                yield from walk(child, key)

    prepends = [n for n in walk(tokenizer_json.get("normalizer"), "normalizers")
                if n.get("type") == "Prepend"]
    if not prepends:
        return False
    if [n.get("prepend") for n in prepends] != [_SPM_SPACE]:
        raise NotImplementedError(f"tokenizer.json prepends {[n.get('prepend') for n in prepends]}; "
                                  f"only a single U+2581 dummy prefix is implemented.")
    strips = [n for n in walk(tokenizer_json.get("decoder"), "decoders") if n.get("type") == "Strip"]
    if [(n.get("content"), n.get("start"), n.get("stop")) for n in strips] != [(" ", 1, 0)]:
        raise NotImplementedError("tokenizer.json prepends a U+2581 dummy prefix but its decoder does not "
                                  "strip exactly one leading space back off, which is the pair loom.cpp "
                                  "implements.")
    return True


def write_bpe_vocab(writer: GGUFWriter, tokenizer_dir: str, pre_type: str = "qwen2",
                    eos_token_ids: list[int] | None = None, tokenizer_model: str = "gpt2") -> None:
    """`tokenizer_model` names the scheme for a front end that WRAPS this byte-level BPE and so has a
    tag of its own ("cosyvoice3", whose `loom::CosyVoice3Vocab` reads everything below unchanged)."""
    tok_dir = Path(tokenizer_dir)
    tokenizer_json = json.loads(_tokenizer_json_text(tok_dir))
    config_path = tok_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
    raw_merges: list = tokenizer_json["model"]["merges"]
    # Normalize both tokenizer.json merges schemas ("a b" strings, or [a, b] pair lists) to "a b" strings.
    merges = [m if isinstance(m, str) else " ".join(m) for m in raw_merges]
    added_tokens: list[dict] = tokenizer_json.get("added_tokens", [])

    max_id = max([*vocab.values(), *(t["id"] for t in added_tokens)], default=-1)
    tokens = [""] * (max_id + 1)
    for piece, idx in vocab.items():
        tokens[idx] = piece
    for t in added_tokens:
        tokens[t["id"]] = t["content"]

    # The added set is `tokenizer.json`'s own `added_tokens`, ALL of it -- not just the `special: true`
    # entries. HF's `AddedVocabulary` splits the raw input on every added token regardless, which is why
    # Gemma 3's `\n\n\n` (id 109, `special: false`) comes back as one id from `AutoTokenizer.encode`
    # and would come back as three from a pre-pass that only knew about markers.
    #
    # `normalized` is not consulted, and that is a real narrowing worth naming: HF matches a
    # `normalized: true` added token against the text AFTER the normalizer instead of before it. Every
    # added token in every checkpoint exported here is `normalized: false` (checked: 6415 of 6415 on
    # Gemma 3, 17 of 17 on SmolLM2), so the distinction has never had a case, and inventing a
    # second matching pass for one that has never occurred is how an untested path ships.
    token_types = [_TOKEN_TYPE_NORMAL] * len(tokens)
    for i, piece in enumerate(tokens):
        if _is_byte_piece(piece):
            token_types[i] = _TOKEN_TYPE_BYTE
    for t in added_tokens:
        token_types[t["id"]] = _TOKEN_TYPE_CONTROL if t.get("special") else _TOKEN_TYPE_USER_DEFINED

    def _token_id(value) -> int:
        # tokenizer_config.json's bos_token/eos_token are either a bare string or an AddedToken-style dict.
        if value is None:
            return -1
        piece = value["content"] if isinstance(value, dict) else value
        for t in added_tokens:
            if t["content"] == piece:
                return t["id"]
        return vocab.get(piece, -1)

    bos_token_id = _token_id(config.get("bos_token"))
    # The generation config wins when the caller passed one, and the tokenizer config is the fallback.
    # In that order because the two disagree exactly where it matters: gemma-3-270m-it's tokenizer says
    # `<eos>` and its generation config says `[<eos>, <end_of_turn>]`, and the second is the one that
    # ends a chat turn.
    eos_ids = list(eos_token_ids or [])
    if not eos_ids:
        fallback = _token_id(config.get("eos_token"))
        eos_ids = [fallback] if fallback >= 0 else []

    writer.add_tokenizer_model(tokenizer_model)
    writer.add_tokenizer_pre(pre_type)
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    if bos_token_id >= 0:
        writer.add_bos_token_id(bos_token_id)
    if eos_ids:
        # The scalar stays FIRST and unchanged, because it is what every pre-P4.23 reader looks at --
        # `loom::text::generate`'s default stop token, `loom_cli`, loom-py. The array is additive: a
        # reader that knows about it stops on any of them, and one that does not behaves exactly as it
        # did.
        writer.add_eos_token_id(eos_ids[0])
        writer.add_array("tokenizer.ggml.eos_token_ids", eos_ids)
    writer.add_add_bos_token(bool(config.get("add_bos_token", False)))
    # Written only when true, so Gemma 3's file -- the same shape, no prefix -- is byte-identical.
    if pre_type in SPM_BYTE_FALLBACK_PRES and _spm_dummy_prefix(tokenizer_json):
        writer.add_add_space_prefix(True)
