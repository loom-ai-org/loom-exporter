"""oruk-loop-153M's leaf, the parts that need no checkpoint: which recognizer claims the directory, and
the kept-vocabulary tokenizer the GGUF carries.

The detection half matters because the checkpoint declares the stock `model_type: qwen3_asr` in the
transformers layout -- exactly what `qwen3-asr` recognizes -- and loads into the stock class as a 1.7B
model with every layer missing. Both recognizers are asserted, not only the new one.

Run: ~/.venvs/piper/bin/python3 -m pytest tests/ci/test_oruk_loop_export.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom_exporter.oruk_loop_export import (  # noqa: E402
    _check_kept_tokenizer, _is_oruk_loop, _write_kept_tokenizer,
)
from loom_exporter.qwen3_asr_export import _is_qwen3_asr_hf  # noqa: E402

_STOCK = {"model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"],
          "audio_config": {}, "text_config": {}}
_LOOPED = dict(_STOCK, architectures=["OrukLoopForConditionalGeneration"], oruk_loop={"encoder_rank": 64},
               auto_map={"AutoModel": "modeling_oruk_loop.OrukLoopForConditionalGeneration"})


def _checkpoint(tmp_path: Path, name: str, config: dict) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_each_recognizer_claims_only_its_own_checkpoint(tmp_path):
    stock = _checkpoint(tmp_path, "stock", _STOCK)
    looped = _checkpoint(tmp_path, "looped", _LOOPED)
    assert _is_qwen3_asr_hf(stock) and not _is_oruk_loop(stock)
    assert _is_oruk_loop(looped) and not _is_qwen3_asr_hf(looped)


def test_the_block_alone_is_not_enough(tmp_path):
    """`oruk_loop` without the class that reads it is not something this leaf knows how to load."""
    config = dict(_LOOPED, architectures=["Qwen3ASRForConditionalGeneration"])
    assert not _is_oruk_loop(_checkpoint(tmp_path, "odd", config))


def _toy_tokenizer(path: Path) -> None:
    """A byte-level BPE with five pieces, two merges and one added token -- the shape of Qwen's file."""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers

    vocab = {"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4}
    tok = tokenizers.Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b"), ("ab", "c")]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|im_end|>"])
    path.mkdir()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))


def test_kept_tokenizer_renumbers_pieces_and_drops_dead_merges(tmp_path):
    src, out = tmp_path / "src", tmp_path / "kept"
    _toy_tokenizer(src)
    keep = np.array([0, 1, 2, 5])                    # a, b, ab, <|im_end|>; c and abc are dropped
    _write_kept_tokenizer(src, keep, out)
    _check_kept_tokenizer(src, keep, out)

    written = json.loads((out / "tokenizer.json").read_text())
    assert written["model"]["vocab"] == {"a": 0, "b": 1, "ab": 2}
    assert [t["id"] for t in written["added_tokens"]] == [3]
    # `ab c` needs a dropped piece on both sides of it, so only `a b` survives.
    merges = [m if isinstance(m, str) else " ".join(m) for m in written["model"]["merges"]]
    assert merges == ["a b"]
    assert json.loads((out / "tokenizer_config.json").read_text()) == {"eos_token": "<|im_end|>"}


def test_kept_tokenizer_refuses_a_kept_id_with_no_piece(tmp_path):
    src = tmp_path / "src"
    _toy_tokenizer(src)
    with pytest.raises(ValueError, match="exactly once"):
        _write_kept_tokenizer(src, np.array([0, 1, 9]), tmp_path / "kept")
