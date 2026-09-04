"""The declared contract: that a task reaches the file, and that nothing declares one it cannot know.

The GGUF used to say `loom.architecture` and nothing about what the model was FOR, so every end-to-end
door a host offered had to be reached through a table of architecture names -- the per-architecture host
code all three repos forbid. These tests pin the three things that stop being true:

  1. the task travels from the recognizer that matched to the written file;
  2. the modality pair is declared with it, since that is what a host actually dispatches on;
  3. a config that does NOT know its task declares nothing rather than guessing.

The third is the one worth having a test for. An absent `loom.task` is something a host can detect and
fall back from; a wrong one is not.

Companion to loom.cpp's `tests/ci/test_model_contract.cpp`, which reads these keys back. The two ends
are joined by nothing but the spelling of the keys, so if one is changed without the other the result is
an absent key and no error anywhere -- which is why both sides are tested against literal names.
"""
import os
import tempfile
import unittest

from gguf import GGUFReader

from loom_exporter.export_config import LoomExportConfig
from loom_exporter.registry import ModelRecognizer, TaskRegistry, TaskRegistryEntry
from loom_exporter.causal_lm_export import LMCausalModelExportConfig


def _kv_string(reader, key):
    return reader.fields[key].parts[-1].tobytes().decode("utf-8")


class TestContractDefaults(unittest.TestCase):
    def test_a_config_with_no_task_declares_nothing(self):
        """Declaring nothing is the correct answer, not a gap to be filled with a plausible default."""
        config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)
        self.assertEqual(config.task, "")
        self.assertEqual(config.contract(), {})

    def test_the_pair_is_per_task(self):
        config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)

        config.task = "text-generation"
        self.assertEqual(config.contract(),
                         {"task": "text-generation", "input.kind": "text", "output.kind": "text"})

        config.task = "automatic-speech-recognition"
        self.assertEqual(config.contract()["input.kind"], "audio")
        self.assertEqual(config.contract()["output.kind"], "token_ids")

        # Four of the five TTS families take phoneme ids from a G2P step outside the engine. Supertonic
        # overrides this because it encodes graphemes itself -- the distinction is per-model, and is what
        # tells a host whether a text door exists at all.
        config.task = "text-to-speech"
        self.assertEqual(config.contract()["input.kind"], "phoneme_ids")
        self.assertEqual(config.contract()["output.kind"], "audio")

    def test_a_task_with_no_modality_pair_still_declares_its_name(self):
        """A task this table has never heard of must still write `loom.task` -- a host that does not
        recognise the name can say so, where a host handed nothing cannot tell that from an old export.

        Exercised with a name that is not in the vocabulary at all, because **every canonical task now
        has a pair**: this used to use `text-to-codes` while it was reserved, and family 10 claiming it
        left the branch with no real example. A synthetic name keeps the branch covered without waiting
        for the vocabulary to grow one."""
        config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)
        config.task = "some-future-task"
        self.assertEqual(config.contract(), {"task": "some-future-task"})

    def test_every_canonical_task_declares_a_modality_pair(self):
        """The other half, and the reason the test above had to change: a canonical name reaching the
        `pair is None` branch would ship a file saying what it is FOR without saying what it maps
        BETWEEN, which is half a contract and reads to a host exactly like an old export."""
        from loom_exporter.tasks import known_tasks

        for task in known_tasks():
            config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)
            config.task = task
            contract = config.contract()
            self.assertIn("input.kind", contract, f"{task} declares no modality pair")
            self.assertIn("output.kind", contract, f"{task} declares no modality pair")


class TestTaskReachesTheConfig(unittest.TestCase):
    def test_register_stamps_the_task_onto_every_recognizer(self):
        """A recognizer's task is the task it was registered under, by construction rather than by a
        family remembering to repeat itself."""
        registry = TaskRegistry()
        rec = ModelRecognizer(name="fake", detect=lambda p: False,
                              build_config=lambda p, o: None)
        self.assertEqual(rec.task, "")
        registry.register(TaskRegistryEntry(task="text-generation",
                                            config_class=LMCausalModelExportConfig,
                                            recognizers=[rec]))
        self.assertEqual(rec.task, "text-generation")


class TestContractIsWritten(unittest.TestCase):
    """That `contract()`'s dict becomes GGUF KVs, including the tables an ASR export declares."""

    def test_every_value_shape_the_contract_uses_is_written(self):
        from loom_exporter import LoomGGUFBackend
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb

        @mb.program(input_specs=[mb.TensorSpec(shape=(2,))])
        def prog(x):
            return mb.identity(x=x)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "contract.gguf")
            LoomGGUFBackend()(prog, output_path=out, architecture="contract_test", contract={
                "task": "automatic-speech-recognition",
                "input.kind": "audio",
                "output.kind": "token_ids",
                "asr.timestamp_first_id": 50364,
                "asr.timestamp_step_sec": 0.02,
                "asr.language_names": ["de", "en"],
                "asr.language_ids": [50261, 50259],
                # Empty tables are omitted rather than written empty: an English-only checkpoint has no
                # language tokens, and "no languages to name" must read differently from "old export".
                "asr.task_names": [],
            })

            reader = GGUFReader(out)
            self.assertEqual(_kv_string(reader, "loom.task"), "automatic-speech-recognition")
            self.assertEqual(_kv_string(reader, "loom.input.kind"), "audio")
            self.assertEqual(_kv_string(reader, "loom.output.kind"), "token_ids")
            self.assertEqual(int(reader.fields["loom.asr.timestamp_first_id"].parts[-1][0]), 50364)
            self.assertAlmostEqual(
                float(reader.fields["loom.asr.timestamp_step_sec"].parts[-1][0]), 0.02, places=5)
            self.assertIn("loom.asr.language_names", reader.fields)
            self.assertIn("loom.asr.language_ids", reader.fields)
            self.assertNotIn("loom.asr.task_names", reader.fields)

    def test_a_bool_is_refused(self):
        """Nothing in the contract is a flag: a capability is declared by its key being PRESENT, which
        is what lets a host tell 'this model cannot' from 'this export predates the key'. A bool would
        be a third state that means neither."""
        from loom_exporter import LoomGGUFBackend
        from coremltools.converters.mil import Builder as mb

        @mb.program(input_specs=[mb.TensorSpec(shape=(2,))])
        def prog(x):
            return mb.identity(x=x)

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeError):
                LoomGGUFBackend()(prog, output_path=os.path.join(d, "bad.gguf"),
                                  architecture="contract_test", contract={"timestamps": True})


if __name__ == "__main__":
    unittest.main()
