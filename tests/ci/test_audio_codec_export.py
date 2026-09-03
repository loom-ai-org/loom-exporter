"""Family 11 (P5): neural audio codec decoders -- codes in, a waveform out.

**The assertion this file exists for is that the output LENGTH is a function of the input length.**
That is not a formality: the first working export of this family produced correct audio and returned
one frame's worth of it for every input, because the dynamic-shape walk gave up on the RVQ's
rank-reducing slice and every transposed convolution downstream was cropped to a literal computed at
length 1. Nothing raised. The export ran, the GGUF loaded, the driver returned floats, and the only
symptom was a number of samples nobody had written a test for.

So the checks below are on the emitted crop shapes -- which must be expressions in the root axis, not
numbers -- and on two exports at different trace lengths producing the identical topology.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.audio_codec_export import (
    AudioCodecExportConfig,
    _build_dac,
    _is_dac,
)
from loom_exporter.export_config import LoomExportConfig
from loom_exporter.registry import default_registry
from loom_exporter.tasks import task_spec


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection and the task ------------------------------------------------------------------------

def test_a_dac_directory_is_claimed(tmp_path):
    assert _is_dac(_hf_dir(tmp_path, "dac", {"model_type": "dac"}))


def test_another_codec_is_not_claimed_by_dacs_recognizer(tmp_path):
    """Specific rather than generic, unlike family 12's single recognizer. There is no
    `AutoModelForAudioCodec`: `EncodecModel`, `MimiModel` and SNAC's own package are unrelated classes
    with different `decode` signatures, so a recognizer that claimed them would claim checkpoints this
    wrapper cannot drive."""
    assert not _is_dac(_hf_dir(tmp_path, "enc", {"model_type": "encodec"}))
    assert not _is_dac(_hf_dir(tmp_path, "mimi", {"model_type": "mimi"}))
    assert not _is_dac(tmp_path / "nothing-here")


def test_the_registry_resolves_a_synthetic_dac(tmp_path):
    recognizer = default_registry().detect(_hf_dir(tmp_path, "dac", {"model_type": "dac"}))
    assert recognizer.name == "dac"
    assert recognizer.task == "audio-codec"


def test_the_task_declares_this_familys_base_config():
    assert task_spec("audio-codec").base_config_class() is AudioCodecExportConfig
    assert not task_spec("audio-codec").reserved


def test_the_modality_pair_is_codes_in_audio_out():
    """`audio_codes`, not `token_ids` -- ADR-020. The latter folds onto "text" in the engine's
    `interface_side`, so this file would declare itself `text2speech` and be offered a text door it
    has no vocabulary for."""
    config = LoomExportConfig(architecture="x", output_path="/tmp/x.gguf", decomposition=None)
    config.task = "audio-codec"
    assert config.contract() == {"task": "audio-codec", "input.kind": "audio_codes",
                                 "output.kind": "audio"}


def test_the_driver_builder_is_named_by_the_family(tmp_path):
    """The third family to override `synthesized_builder_key`, and the first whose epilogue reduces
    NOTHING -- `ArgmaxEpilogue` here would argmax the audio."""
    config = _build_dac(tmp_path, "/tmp/x.gguf")
    assert config.synthesized_builder_key() == "CodecDecode"
    assert config.backend_kwargs()["driver_builder"] == "CodecDecode"
    assert config.backend_kwargs()["root_axis"] == "n_codes"


def test_hparams_are_empty_without_a_checkpoint(tmp_path):
    """`component_registry.usage()` builds every registered config without a model to attribute
    driver components; a family whose hparams read the checkpoint has to survive that."""
    assert _build_dac(tmp_path, "/tmp/x.gguf").hparams() == {}


# -- the real trace --------------------------------------------------------------------------------

def _tiny_codec(tmp_path: Path) -> Path:
    """A real, randomly-initialised `DacModel`, small enough to trace in a unit test.

    Real rather than mocked for the reason family 12's fixtures are: what is under test is the
    interaction with the actual RVQ + transposed-convolution stack, and a stub has neither.
    """
    pytest.importorskip("torch")
    from transformers import DacConfig, DacModel

    config = DacConfig(encoder_hidden_size=16, decoder_hidden_size=32, codebook_size=32,
                       codebook_dim=4, n_codebooks=3, hidden_size=16,
                       downsampling_ratios=[2, 2], upsampling_ratios=[2, 2], sampling_rate=16000)
    out = tmp_path / "tiny-dac"
    DacModel(config).save_pretrained(out)
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    from gguf import GGUFReader

    config = AudioCodecExportConfig(architecture=None, output_path=str(out),
                                    model_dir=str(checkpoint), **kwargs)
    config.task = "audio-codec"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "topology": json.loads(reader.fields["model.graph_topology.main_topology"].contents()),
        "n_codebooks": reader.fields["loom.codec.n_codebooks"].contents(),
        "sample_rate": reader.fields["loom.sample_rate"].contents(),
        "input_kind": reader.fields["loom.input.kind"].contents(),
    }


def test_the_output_length_is_a_function_of_the_input_length(tmp_path):
    pytest.importorskip("coremltools")
    exported = _export(_tiny_codec(tmp_path), tmp_path / "tiny.gguf", n_frames=8)
    topo = exported["topology"]

    assert {i["name"]: i["shape"] for i in topo["inputs"]} == {"codes": ["3", "n_codes", "1"]}
    assert exported["input_kind"] == "audio_codes"
    assert exported["n_codebooks"] == 3 and exported["sample_rate"] == 16000

    # THE CHECK THIS FILE EXISTS FOR. Every transposed convolution's crop must be sized in the ROOT
    # AXIS. A literal here is the whole bug: the export succeeds, the audio is right, and the model
    # returns the traced number of samples forever.
    crops = [n["attrs"]["shape"] for n in topo["nodes"] if n["op"] == "VIEW"]
    assert crops, "no crop VIEWs at all -- the conv_transpose padding composition did not run"
    upsampling = [c for c in crops if any("n_codes" in str(d) for d in c)]
    assert len(upsampling) == len(crops), (
        f"crop shapes with no dynamic axis in them: "
        f"{[c for c in crops if not any('n_codes' in str(d) for d in c)]}"
    )

    # The driver divides by the codebook count to recover the frame count, which is why the caller's
    # layout is frame-major and not the model's own.
    assert "math.floor(#codes / 3)" in exported["driver"]
    assert "loom.run_subgraph('main_topology'" in exported["driver"]


def test_the_traced_length_does_not_reach_the_graph(tmp_path):
    """Two exports differing only in `n_frames`. A single export passes against a graph with the
    length baked in, because the shape it was baked at is the shape it is asked for."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_codec(tmp_path)
    at_8 = _export(checkpoint, tmp_path / "a.gguf", n_frames=8)
    at_32 = _export(checkpoint, tmp_path / "b.gguf", n_frames=32)
    assert at_8["topology"] == at_32["topology"]
    assert at_8["driver"] == at_32["driver"]
