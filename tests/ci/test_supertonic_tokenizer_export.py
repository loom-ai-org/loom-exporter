"""SupertonicTTS's grapheme vocabulary, on its way into a model GGUF (`supertonic_tokenizer_export.py`).

Three things, in the order they can go wrong:

* the writer -- what lands in the file, checked by writing one and reading it back with a plain
  `GGUFReader` rather than by re-running the branch under test;
* the asset lookup -- which of the two real `assets/` layouts resolve, and that a checkpoint without the
  asset resolves to None instead of half-matching something;
* the wiring -- that `TTSSupertonicExportConfig.backend_kwargs()` actually asks for all this, which is
  the part that would rot silently, exactly as `test_export_hparams` says of the hook it guards.

None of this touches the real 65536-entry asset: a hand-built table makes every expected value arithmetic.
The real-asset fidelity check is loom.cpp's own `gate/test_supertonic_text_vectorizer.cpp`.
"""
import json
import tempfile
import unittest
from pathlib import Path

from loom_exporter.supertonic_tokenizer_export import (
    AVAILABLE_LANGS, DEFAULT_LANG, find_indexer, write_supertonic_vocab,
)

TABLE_SIZE = 65536


def _table():
    """Printable ASCII -> 0.., in the flat BMP-sized shape the real unicode_indexer.json has."""
    table = [-1] * TABLE_SIZE
    for cp in range(32, 127):
        table[cp] = cp - 32
    return table


def _write_asset(directory: Path, table=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "unicode_indexer.json"
    path.write_text(json.dumps(_table() if table is None else table))
    return path


def _write_and_read_back(tokenizer_dir, **kwargs):
    """Writes a GGUF carrying just this vocabulary and returns its `tokenizer.ggml.*` KVs."""
    from gguf import GGUFReader, GGUFWriter

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "vocab.gguf")
        w = GGUFWriter(out, "loom-supertonic-vocab-test")
        w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
        write_supertonic_vocab(w, str(tokenizer_dir), **kwargs)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        reader = GGUFReader(out)
        out_kvs = {}
        for field in reader.fields.values():
            if not field.name.startswith("tokenizer."):
                continue
            if field.name.endswith("codepoint_to_id"):
                out_kvs[field.name] = [int(field.parts[i][0]) for i in field.data]
            else:
                out_kvs[field.name] = str(bytes(field.parts[field.data[0]]), "utf-8")
        return out_kvs


class TestTheWriter(unittest.TestCase):
    def test_it_writes_the_tag_the_engine_dispatches_on_and_the_whole_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_asset(Path(tmp))
            kvs = _write_and_read_back(tmp)

        self.assertEqual(kvs["tokenizer.ggml.model"], "supertonic")
        table = kvs["tokenizer.ggml.supertonic.codepoint_to_id"]
        # The whole BMP-sized array, not just its mapped entries -- `SupertonicTextVectorizer` indexes it
        # by codepoint directly, so a compacted one would decode every character at the wrong id.
        self.assertEqual(len(table), TABLE_SIZE)
        self.assertEqual(table[ord("A")], ord("A") - 32)
        self.assertEqual(table[0x00E9], -1)  # "é" is not in THIS (ASCII-only) test table
        self.assertEqual(table[TABLE_SIZE - 1], -1)  # the last BMP entry survived the round trip

    def test_the_default_language_is_written_so_a_host_need_not_assume_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_asset(Path(tmp))
            self.assertEqual(_write_and_read_back(tmp)["tokenizer.ggml.supertonic.default_lang"],
                              DEFAULT_LANG)
            self.assertEqual(
                _write_and_read_back(tmp, default_lang="ko")["tokenizer.ggml.supertonic.default_lang"],
                "ko")

    def test_the_declared_default_matches_the_real_classs_own(self):
        """The real `tokenize_str(text, lang="en")`. Written down here because the engine's fallback for
        an absent KV is the same string in another repo, and the two agreeing is the point."""
        self.assertEqual(DEFAULT_LANG, "en")
        self.assertIn(DEFAULT_LANG, AVAILABLE_LANGS)

    def test_a_language_the_model_cannot_speak_is_refused_at_export(self):
        """A typo here would otherwise ship: neither the engine nor the real Python class validates a
        language tag, so `<de>` would just tokenize into a few characters the model was never trained on
        and produce quietly wrong audio."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_asset(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                _write_and_read_back(tmp, default_lang="de")
        self.assertIn("AVAILABLE_LANGS", str(ctx.exception))

    def test_a_missing_asset_names_the_directory_it_looked_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                _write_and_read_back(tmp)
        self.assertIn(tmp, str(ctx.exception))

    def test_a_table_that_maps_nothing_is_refused_rather_than_written(self):
        """An all-negative table reads as a valid JSON array of the right length and produces a GGUF whose
        tokenizer silently encodes every string to nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_asset(Path(tmp), table=[-1] * TABLE_SIZE)
            with self.assertRaises(ValueError):
                _write_and_read_back(tmp)


class TestTheAssetLookup(unittest.TestCase):
    def test_the_real_repo_layout_resolves(self):
        """assets/pt (the checkpoint) and assets/onnx (the asset), from the repo root."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assets" / "pt").mkdir(parents=True)
            expected = _write_asset(root / "assets" / "onnx")
            self.assertEqual(find_indexer(root / "assets" / "pt"), expected)

    def test_a_copied_out_assets_tree_resolves_too(self):
        """The same two directories as siblings, without the checkout around them -- what a checkpoint
        copied to `/some/models/supertonic/{pt,onnx}` looks like."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pt").mkdir()
            expected = _write_asset(root / "onnx")
            self.assertEqual(find_indexer(root / "pt"), expected)

    def test_a_checkpoint_without_the_asset_resolves_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pt").mkdir()
            self.assertIsNone(find_indexer(Path(tmp) / "pt"))


class TestTheConfigAsksForIt(unittest.TestCase):
    """The wiring. `backend_kwargs()` is per-family and hand-built, so this is the half that rots."""

    def _config(self, model_dir):
        from loom_exporter.supertonic_export import TTSSupertonicExportConfig
        return TTSSupertonicExportConfig(architecture="supertonic_mil", output_path="unused.gguf",
                                          model_dir=str(model_dir))

    def test_it_names_the_family_rather_than_leaving_it_to_detection(self):
        """`unicode_indexer.json` is not an HF tokenizer directory, so `detect_vocab_family` would raise
        on it -- naming the family is what keeps auto-detection from ever being asked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assets" / "pt").mkdir(parents=True)
            asset = _write_asset(root / "assets" / "onnx")
            kwargs = self._config(root / "assets" / "pt").backend_kwargs()

        self.assertEqual(kwargs["tokenizer_family"], "supertonic")
        self.assertEqual(Path(kwargs["tokenizer_dir"]), asset.parent)

    def test_it_still_carries_hparams_through(self):
        """The standing rule every `backend_kwargs()` override has to honour (test_export_hparams)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assets" / "pt").mkdir(parents=True)
            _write_asset(root / "assets" / "onnx")
            kwargs = self._config(root / "assets" / "pt").backend_kwargs()
        self.assertIn("txt_len", kwargs["hparams"])

    def test_a_checkpoint_without_the_asset_exports_anyway_and_says_so(self):
        """Warned and omitted, not raised: the four traced graphs are still a good export, and this method
        is also called by callers that never trace at all."""
        import io
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pt").mkdir()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                kwargs = self._config(Path(tmp) / "pt").backend_kwargs()

        self.assertNotIn("tokenizer_dir", kwargs)
        self.assertIn("hparams", kwargs)
        self.assertIn("warning", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
