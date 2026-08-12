"""Writes SupertonicTTS v2's grapheme text front-end into a model GGUF, `tokenizer.ggml.model`="supertonic".

The same two KVs `tools/convert_supertonic/convert_supertonic_text_vectorizer.py` already wrote into a
STANDALONE tokenizer-only GGUF -- this module exists so they can travel inside the model instead. The
engine class that reads them back (`loom::SupertonicTextVectorizer`) and its gate test both predate this
and are unchanged; what was missing was only that the one-GGUF MIL export never wrote them, so every
published Supertonic model reported no tokenizer at all.

There is no learned parameter here and no HF tokenizer directory either, which is why this does not go
through `tokenizer_detect.py`'s auto-detection like the four HF families do: the whole vocabulary is one
static asset, `assets/onnx/unicode_indexer.json`, a flat 65536-entry array whose `indexer[codepoint]` is a
vocab id (or -1 for a codepoint this model cannot say). Everything else about the front-end -- NFKD
normalization, emoji stripping, the replacement table, the `<lang>...</lang>` wrap -- is the engine's
port of the real `TextVectorizer`, not data written here.

**That split is the part worth not copying.** A second grapheme TTS model should not add a second C++
class; the table is data and generalizes, the preprocessing pipeline is per-model and belongs in exporter
data or driver Lua. See BACKLOG.md's "grapheme text front-ends" entry.

Requires: pip install gguf
"""
import json
from pathlib import Path
from typing import List

from gguf import GGUFWriter

# The asset's path relative to the SupertonicTTS repo root, as the real `TextVectorizer.__init__` itself
# resolves it when given no explicit `indexer_path`.
INDEXER_RELPATH = Path("assets") / "onnx" / "unicode_indexer.json"

# The real `tokenize_str`'s own default language. Written into the file rather than assumed by each host:
# `SupertonicTextVectorizer::load` falls back to this same string when the KV is absent, so an older GGUF
# and a new one tokenize identically, but only a file that STATES it can ever be exported at another one.
DEFAULT_LANG = "en"

# The real `TextVectorizer.AVAILABLE_LANGS`. Checked against, not written: the engine deliberately does
# not validate a language tag (neither does the real Python class), so this exists only to catch a typo in
# `DEFAULT_LANG` here, where it is a static mistake rather than a run-time one.
AVAILABLE_LANGS = ("en", "ko", "es", "pt", "fr")


def find_indexer(model_dir: Path) -> Path | None:
    """The `unicode_indexer.json` belonging to the checkpoint at `model_dir`, or None if it isn't there.

    `model_dir` is the `assets/pt` directory holding the four `.pt` files (that is what the recognizer
    detects and what `phases()` loads from), so the asset is `assets/onnx/`'s -- two spellings of the same
    place, tried in order: from the repo root two levels up, and as a direct sibling directory. The second
    is what makes a copied-out `assets/` tree work without the rest of the checkout around it.

    None rather than an exception: an export whose checkpoint directory lacks the asset is still a
    perfectly good export of the four traced graphs, it just has no text door. The caller warns.
    """
    for candidate in (model_dir.parent.parent / INDEXER_RELPATH,
                      model_dir.parent / "onnx" / "unicode_indexer.json"):
        if candidate.is_file():
            return candidate
    return None


def write_supertonic_vocab(writer: GGUFWriter, tokenizer_dir: str, default_lang: str = DEFAULT_LANG) -> None:
    """`tokenizer_dir` is the directory holding `unicode_indexer.json` -- a directory, like every other
    family's writer takes, even though this one reads a single file out of it (`find_indexer` is what
    locates that directory from a checkpoint path)."""
    if default_lang not in AVAILABLE_LANGS:
        raise ValueError(f"write_supertonic_vocab: default_lang={default_lang!r} is not one of the real "
                          f"TextVectorizer.AVAILABLE_LANGS {list(AVAILABLE_LANGS)}")

    indexer_path = Path(tokenizer_dir) / "unicode_indexer.json"
    if not indexer_path.is_file():
        raise FileNotFoundError(f"write_supertonic_vocab: no unicode_indexer.json in {tokenizer_dir}")

    table: List[int] = json.loads(indexer_path.read_text())
    if not any(v >= 0 for v in table):
        raise ValueError(f"write_supertonic_vocab: {indexer_path} maps no codepoint at all (every entry "
                          f"is negative) -- this is not a real unicode_indexer.json")

    writer.add_tokenizer_model("supertonic")
    writer.add_array("tokenizer.ggml.supertonic.codepoint_to_id", table)
    writer.add_string("tokenizer.ggml.supertonic.default_lang", default_lang)
