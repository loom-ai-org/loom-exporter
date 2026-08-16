"""What every text-to-speech family must declare for its text door to exist and to sound right.

Written after a shipped Kokoro GGUF (1.0.0-rc4) turned out to speak unintelligibly, from four separate
declarations that were each individually plausible and each wrong (BACKLOG.md P4.12):

  1. `text.frontend` fell through to `"vocab"`, so the high-level door encoded the caller's ENGLISH
     SPELLING against the phoneme table and the model read the spelling aloud;
  2. the phoneme table declared `bos: -1, eos: -1`, so nobody applied `KModel.forward`'s `[0, *ids, 0]`
     wrap -- the driver's header said the caller does it, the table's docstring said the driver does it;
  3. `sample_rate` was undeclared, so every host guessed;
  4. the default voice was a `rng.normal(scale=0.3)` tensor copied out of a reference-forward dump.

Every one of them is a per-family CONSTANT that no other test looked at, and the sweep could not catch
any of them: it diffs each export against its own recorded baseline, so a value that has been wrong
since the first snapshot is exactly what it certifies as unchanged. These are the standing rules
instead, driven by the registry rather than by a list -- a new TTS family that forgets one fails here.

`contract()` is built against a checkpoint path that DOES NOT EXIST, on purpose. A contract is a
statement about the architecture, not a reading of one checkpoint; a family that cannot answer it
without opening files has put a per-checkpoint fact where a per-architecture one belongs, and this test
failing on a FileNotFoundError is the right way to find that out.
"""
import unittest
from pathlib import Path

from loom_exporter.registry import default_registry

# The families whose GGUF consumes PHONEME ids, and which therefore need a G2P step outside the engine.
# Named rather than detected: "does this checkpoint take phonemes" is a fact about the model, and a rule
# that inferred it from the same declaration it is checking would be circular.
PHONEME_INPUT = {"kokoro", "matcha", "styletts2", "vits"}
# ...and the one that does not. Supertonic encodes graphemes itself and its GGUF carries the codepoint
# table, so `"vocab"` is the true answer for it and the wrong one for the four above. Here so the rule
# cannot be satisfied by declaring `"phonemes"` everywhere, which would refuse Supertonic a real door.
GRAPHEME_INPUT = {"supertonic"}

# Families that still declare no `sample_rate`, with the reason it is an EXEMPTION rather than a pass.
# Each needs its rate taken off its own checkpoint the way Kokoro's and Supertonic's are -- Matcha's and
# StyleTTS2's from their configs, VITS's from the Piper voice JSON's `audio.sample_rate`, which is the
# only one of the three that is genuinely per-voice rather than per-architecture. Listed rather than
# skipped silently, because an empty rule and a satisfied rule look identical from the outside
# (BACKLOG.md P4.12 follow-up). Removing a name here without fixing the family fails the last test.
NO_SAMPLE_RATE_YET = {"matcha", "styletts2", "vits"}


def _tts_configs():
    """`{recognizer name: config instance}` for every registered text-to-speech family."""
    registry = default_registry()
    entry = registry._entries["text-to-speech"]
    return {rec.name: rec.build_config(Path("/nonexistent"), "/tmp/does-not-matter.gguf")
            for rec in entry.recognizers}


class TestTTSTextDoor(unittest.TestCase):
    def setUp(self):
        self.configs = _tts_configs()

    def test_the_scan_reaches_every_family_these_rules_are_about(self):
        """Each rule below passes vacuously on an empty dict, so name what must be found."""
        self.assertEqual(set(self.configs), PHONEME_INPUT | GRAPHEME_INPUT,
                         "a TTS family was added or renamed without being classified above -- which "
                         "half it belongs in is the whole question these rules ask")

    def test_a_phoneme_family_declares_a_phoneme_front_end(self):
        """`"vocab"` on a phoneme model is not a missing declaration, it is a WRONG one: it tells a host
        the model encodes text itself, so the host skips the G2P and encodes the spelling. That failure
        is silent -- the ids are all valid, the audio is well formed, and it comes out as gibberish."""
        for name in sorted(PHONEME_INPUT):
            with self.subTest(model=name):
                self.assertEqual(self.configs[name].contract().get("text.frontend"), "phonemes")

    def test_a_grapheme_family_declares_a_vocab_front_end(self):
        for name in sorted(GRAPHEME_INPUT):
            with self.subTest(model=name):
                self.assertEqual(self.configs[name].contract().get("text.frontend"), "vocab")

    def test_a_phoneme_family_declares_which_alphabet_it_wants(self):
        """A host has to pick a G2P. Picking it from a default rather than from the file means the day a
        second alphabet is registered, every model that declared nothing silently gets the wrong one."""
        for name in sorted(PHONEME_INPUT):
            with self.subTest(model=name):
                self.assertTrue(self.configs[name].contract().get("text.phoneme_alphabet"),
                                "declares no phoneme alphabet, so a host must guess which G2P to run")

    def test_a_tts_family_declares_its_sample_rate(self):
        """A rate cannot be recovered from a list of floats, and getting it wrong does not fail: 24 kHz
        played at 16 kHz is a slow voice, not an error. Undeclared, it sends a caller off trying rates
        one at a time -- which is how this whole class of bug got reported in the first place."""
        for name, config in sorted(self.configs.items()):
            if name in NO_SAMPLE_RATE_YET:
                continue
            with self.subTest(model=name):
                rate = config.contract().get("sample_rate")
                self.assertTrue(rate and int(rate) > 0,
                                "declares no sample rate, so every host that plays this guesses")

    def test_the_exemption_list_holds_only_families_that_still_need_it(self):
        """The other half of the exemption, and the reason it is safe to have one: a family that gains a
        rate and is not removed from `NO_SAMPLE_RATE_YET` fails here, so the list cannot quietly outlive
        the gap it describes."""
        stale = sorted(name for name in NO_SAMPLE_RATE_YET
                       if self.configs[name].contract().get("sample_rate"))
        self.assertEqual(stale, [], "these declare a sample rate now -- drop them from NO_SAMPLE_RATE_YET")


class TestKokoroPhonemeAssembly(unittest.TestCase):
    """Kokoro's edge wrap, the one piece of assembly no other family in the set has.

    `KModel.forward` is `input_ids = [[0, *input_ids, 0]]`. Not decoration: it is what the ALBERT encoder
    and the duration predictor were trained to see at the edges, and without it the last phoneme loses
    its duration -- whisper-small hears "hello world" come back as "Hello, worth".

    Unlike the contract, `phoneme_table()` genuinely reads the checkpoint's `config.json` (the vocabulary
    IS per-checkpoint), so this builds one from a literal table rather than requiring a real model dir.
    """

    def _table(self, vocab):
        import json
        import tempfile

        from loom_exporter.kokoro_export import TTSKokoroExportConfig

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"vocab": vocab}))
            config = TTSKokoroExportConfig(architecture="loom-kokoro-mil",
                                           output_path="/tmp/does-not-matter.gguf", model_dir=tmp)
            return config.phoneme_table()

    def test_bos_and_eos_are_the_pad_id(self):
        table = self._table({"h": 50, "ə": 83, "l": 54})
        self.assertEqual((table["bos"], table["eos"]), (0, 0))
        self.assertFalse(table["interleave_blank"])

    def test_the_pad_id_is_not_also_a_real_phoneme(self):
        """What makes 0 usable as both bos and eos is that `$` is not a symbol any phonemizer emits. A
        vocabulary that assigned id 0 to a real phoneme would make the wrap ambiguous, and the wrap is
        applied here rather than in the driver -- so this is where that assumption is checked."""
        table = self._table({"h": 50, "ə": 83, "l": 54})
        self.assertNotIn(0, table["ids"])
