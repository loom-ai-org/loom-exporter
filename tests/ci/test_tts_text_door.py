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

# The families whose driver DIVIDES BY a step count, so `n_steps` is required rather than optional and
# an undeclared default is a door that raises out of Lua. Named rather than detected for the same reason
# as above, and because the three do not share one marker: Matcha's and Supertonic's step counts come
# from a `samplers()` spec, StyleTTS2's from a hand-written diffusion driver that no spec describes.
NEEDS_STEPS = {"matcha", "styletts2", "supertonic"}


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

    def test_a_family_whose_sampler_needs_a_step_count_declares_one(self):
        """Supertonic shipped without this and its text door did not open at all: the high-level
        `_infer` passes `n_steps` only when the caller named one or the contract declares one, so an
        undeclared count reached Lua as nil and `infer` died dividing by it. Unlike the sample rate,
        this one cannot be exempted while the family still works -- there is no wrong-but-audible
        version of it, only a raise -- which is why it is a flat rule with no exemption list."""
        for name in sorted(NEEDS_STEPS):
            with self.subTest(model=name):
                steps = self.configs[name].contract().get("tts.default_steps")
                self.assertTrue(steps and int(steps) > 0,
                                "this driver divides by n_steps, so a caller who names none gets a Lua "
                                "error rather than audio -- declare the count the model's own inference "
                                "entry point uses")

    def test_a_family_that_needs_no_step_count_declares_none(self):
        """The other half, so `NEEDS_STEPS` cannot be satisfied by declaring a count everywhere. Kokoro
        and VITS run no sampler; a step count on either would be a number a host might pass to a driver
        that has no use for it."""
        for name in sorted(set(self.configs) - NEEDS_STEPS):
            with self.subTest(model=name):
                self.assertIsNone(self.configs[name].contract().get("tts.default_steps"))

    def test_the_exemption_list_holds_only_families_that_still_need_it(self):
        """The other half of the exemption, and the reason it is safe to have one: a family that gains a
        rate and is not removed from `NO_SAMPLE_RATE_YET` fails here, so the list cannot quietly outlive
        the gap it describes."""
        stale = sorted(name for name in NO_SAMPLE_RATE_YET
                       if self.configs[name].contract().get("sample_rate"))
        self.assertEqual(stale, [], "these declare a sample rate now -- drop them from NO_SAMPLE_RATE_YET")


class TestEveryPhonemeFamilyDeclaresItsAssembly(unittest.TestCase):
    """The rule the three per-family classes below are instances of.

    All four phoneme families wrap or interleave, and all four shipped declaring `-1/-1/-1/False` at
    some point -- Kokoro (P4.12), then StyleTTS2, then Matcha, each found the same way: a caller heard
    a word go missing. The per-family classes pin what each one's assembly IS; this pins that having
    one is the NORM, so the next family cannot inherit the default and look deliberate. `-1` four times
    is what an undeclared table looks like, and no phoneme model in this set actually wants it.

    Only VITS may answer `interleave_blank` without a wrap, and it is the one whose assembly came from
    the checkpoint rather than from a library: Piper's `phoneme_id_map` states it.
    """

    def test_no_phoneme_family_declares_an_empty_assembly(self):
        """`phoneme_table()` needs each family's own library, so this reads the DECLARATION SITE rather
        than calling it -- the four constants are literals in the source, and a family that has stopped
        writing them as literals is a family this rule should be rewritten for rather than one it should
        silently pass."""
        import inspect

        from loom_exporter import kokoro_export, matcha_export, styletts2_export, vits_export

        empty = '"bos": -1, "eos": -1, "blank": -1, "interleave_blank": False'
        for name, module in sorted({"kokoro": kokoro_export, "matcha": matcha_export,
                                    "styletts2": styletts2_export, "vits": vits_export}.items()):
            with self.subTest(model=name):
                self.assertNotIn(empty, inspect.getsource(module),
                                 "this is the undeclared default, and every family that shipped with it "
                                 "shipped audio with phonemes missing -- state the wrap or the "
                                 "interleave the model's own inference path applies")


class TestKokoroPhonemeAssembly(unittest.TestCase):
    """Kokoro's edge wrap.

    This docstring used to call it "the one piece of assembly no other family in the set has", and that
    sentence is why StyleTTS2 shipped with `bos: -1` for another release: StyleTTS2 reads the SAME
    178-symbol table and wraps too, just asymmetrically, and a rule written as one family's special case
    never looked. `TestStyleTTS2PhonemeAssembly` below is the other half.

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


class TestStyleTTS2PhonemeAssembly(unittest.TestCase):
    """StyleTTS2's edge wrap, which is Kokoro's minus the tail.

    `Demo/Inference_LJSpeech.ipynb` is `tokens.insert(0, 0)` and no append, while `meldataset.py` wraps
    both edges at training. The asymmetry is deliberate downstream: the driver's `pred_dur[-1] += 5` is
    the repo's own compensation for the trailing pad it stops sending at inference, so `eos: 0` here
    would lengthen the final phoneme twice.

    Declared as `bos: -1` until this test existed, on the reasoning that the driver header called the
    wrap "the caller's responsibility" -- true of a caller hand-building ids, false of the high-level
    text door, which applies exactly what the table declares. The audible result was the first phoneme
    landing at position 0 and being spent there: "hello world" synthesized as "llo world".

    `phoneme_table()` reads the symbol list out of the STYLETTS2 CLONE rather than the checkpoint, and a
    hermetic test may not have the clone (or torch, or kokoro) -- so the library is stubbed. What is
    under test is the four assembly constants, which are the exporter's own statement and not the
    clone's.
    """

    def _table(self, vocab):
        import sys
        import types
        from unittest import mock

        from loom_exporter import styletts2_export

        text_utils = types.ModuleType("text_utils")
        text_utils.TextCleaner = lambda: types.SimpleNamespace(word_index_dictionary=vocab)
        with mock.patch.object(styletts2_export, "load_styletts2", lambda: None), \
                mock.patch.dict(sys.modules, {"text_utils": text_utils}):
            config = styletts2_export.TTSStyleTTS2ExportConfig(
                architecture="loom-styletts2-mil", output_path="/tmp/does-not-matter.gguf",
                checkpoint_path="/nonexistent")
            return config.phoneme_table()

    def test_the_pad_leads_and_nothing_trails(self):
        table = self._table({"$": 0, "h": 50, "ɛ": 86, "l": 54})
        self.assertEqual((table["bos"], table["eos"]), (0, -1))
        self.assertFalse(table["interleave_blank"])

    def test_the_symbols_survive_the_wrap_being_declared(self):
        """The wrap is assembly, not vocabulary: declaring it must not remove `$` from the table, which
        is a real id the engine's `decode()` skips by comparing against `bos_id` rather than by the
        symbol being absent."""
        table = self._table({"$": 0, "h": 50, "ɛ": 86, "l": 54})
        self.assertEqual(table["symbols"][0], "$")
        self.assertIn(0, table["ids"])


class TestMatchaPhonemeAssembly(unittest.TestCase):
    """Matcha's blank interleave, which is neither of the other two shapes.

    `intersperse(seq, 0)` is `[0, p1, 0, p2, 0, ..., pn, 0]` -- a blank between every phoneme AND at
    both ends -- applied at training (`text_mel_datamodule.py:219`) and at inference (`cli.py:51`)
    alike. The engine builds `[BOS, p1, blank, ..., pn, blank, EOS]`, so a leading bos plus a
    per-phoneme trailing blank is the same sequence and a declared `eos` would append a second
    trailing 0.

    Undeclared, the model gets every phoneme at half its trained spacing; the duration predictor
    answers by collapsing most of them, which is heard as a clipped or half-swallowed sentence rather
    than as an error.

    The symbol list comes from the MATCHA CLONE, so it is stubbed here for the same reason StyleTTS2's
    is -- what is under test is the four assembly constants, which are the exporter's statement.
    """

    def _table(self, symbols):
        import sys
        import types
        from unittest import mock

        from loom_exporter import matcha_export

        pkg = types.ModuleType("matcha.text.symbols")
        pkg.symbols = symbols
        with mock.patch.object(matcha_export, "load_matcha", lambda: None), \
                mock.patch.dict(sys.modules, {"matcha.text.symbols": pkg}):
            config = matcha_export.TTSMatchaExportConfig(
                architecture="loom-matcha-mil", output_path="/tmp/does-not-matter.gguf",
                model_dir="/nonexistent")
            return config.phoneme_table()

    def test_a_blank_separates_every_phoneme_and_leads(self):
        table = self._table(["_", "h", "ə", "l"])
        self.assertEqual((table["bos"], table["eos"], table["blank"]), (0, -1, 0))
        self.assertTrue(table["interleave_blank"])

    def test_the_declaration_reproduces_interspersed_ids_exactly(self):
        """The one that would have caught this: build what the engine's own assembly produces from the
        declaration and compare it against `intersperse`, rather than asserting four constants that
        look plausible. `phoneme_vocab.cpp:83` is the assembly being modelled."""
        table = self._table(["_", "h", "ə", "l"])
        body = [1, 2, 3, 1]

        out = []
        if table["bos"] >= 0:
            out.append(table["bos"])
        for i in body:
            out.append(i)
            if table["interleave_blank"] and table["blank"] >= 0:
                out.append(table["blank"])
        if table["eos"] >= 0:
            out.append(table["eos"])

        # matcha.utils.utils.intersperse(body, 0), inlined so the clone is not needed to state it.
        expected = [0] * (len(body) * 2 + 1)
        expected[1::2] = body
        self.assertEqual(out, expected)
