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
    CodecFamily,
    _build_dac,
    _build_snac,
    _is_dac,
    _is_snac,
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


def test_encodec_is_recognized_and_refuses_with_its_reasons(tmp_path):
    """The second leaf is DETECTED but not exportable, and detection is what makes that sayable.

    Without the recognizer an EnCodec directory would be "no family recognizes this checkpoint",
    which is the wrong answer: this family recognizes it fine and cannot yet trace it. The two
    blockers are real and specific (coremltools' dynamic-pad limitation and a 2-layer LSTM over the
    time axis), so the message names them rather than saying "unsupported".
    """
    path = _hf_dir(tmp_path, "enc", {"model_type": "encodec"})
    assert default_registry().detect(path).name == "encodec"
    with pytest.raises(NotImplementedError, match="Dynamic padding"):
        CodecFamily.ENCODEC.load(str(path))
    with pytest.raises(NotImplementedError, match="LSTM"):
        CodecFamily.ENCODEC.load(str(path))


def test_encodecs_geometry_and_decode_are_already_written(tmp_path):
    """The half that IS done, pinned so it does not rot while the blockers are open: EnCodec's config
    spellings differ from DAC's in every field but `codebook_size`, and that mapping is what a future
    unblocking builds on."""
    class _EncodecConfig:
        num_quantizers, codebook_size, sampling_rate, hop_length = 4, 2048, 32000, 640

    class _Encodec:
        config = _EncodecConfig()

    assert CodecFamily.ENCODEC.geometry(_Encodec()) == {
        "n_codebooks": 4, "codebook_size": 2048, "sample_rate": 32000, "hop_length": 640,
        "vq_strides": [1, 1, 1, 1],
    }


def test_another_codec_is_not_claimed_by_dacs_recognizer(tmp_path):
    """Specific rather than generic, unlike family 12's single recognizer. There is no
    `AutoModelForAudioCodec`: `EncodecModel`, `MimiModel` and SNAC's own package are unrelated classes
    with different `decode` signatures, so a recognizer that claimed them would claim checkpoints this
    wrapper cannot drive."""
    assert not _is_dac(_hf_dir(tmp_path, "enc2", {"model_type": "encodec"}))
    assert not _is_dac(_hf_dir(tmp_path, "mimi", {"model_type": "mimi"}))
    assert not _is_dac(tmp_path / "nothing-here")


SNAC_CONFIG = {
    "sampling_rate": 24000, "encoder_dim": 8, "encoder_rates": [2, 2], "decoder_dim": 16,
    "decoder_rates": [2, 2], "attn_window_size": None, "codebook_size": 16, "codebook_dim": 4,
    "vq_strides": [4, 2, 1], "noise": True, "depthwise": True,
}


def test_a_snac_directory_is_claimed_by_its_shape(tmp_path):
    """SNAC's `config.json` is `SNAC.__init__`'s kwargs dumped verbatim: no `model_type`, no
    `architectures`, nothing naming the class. So detection reads the SHAPE of the config, and
    `vq_strides` is the key no HF codec config has."""
    assert _is_snac(_hf_dir(tmp_path, "snac", SNAC_CONFIG))
    assert default_registry().detect(_hf_dir(tmp_path, "snac2", SNAC_CONFIG)).name == "snac"


def test_snacs_recognizer_defers_to_a_named_architecture(tmp_path):
    """A config carrying BOTH `model_type` and `vq_strides` belongs to whichever recognizer owns that
    `model_type`, not to this one -- a future HF port of SNAC would be loaded through a different
    class with a different `decode`, and claiming it here would run the wrong loader on it."""
    assert not _is_snac(_hf_dir(tmp_path, "named", {**SNAC_CONFIG, "model_type": "snac"}))
    assert not _is_snac(_hf_dir(tmp_path, "dac", {"model_type": "dac"}))
    assert not _is_snac(_hf_dir(tmp_path, "partial", {"vq_strides": [4, 2, 1]}))
    assert not _is_dac(_hf_dir(tmp_path, "snac3", SNAC_CONFIG))


def test_a_row_is_one_coarsest_frame_and_that_is_one_formula(tmp_path):
    """`sum(coarse // stride)`, which is where a multi-rate codec stops agreeing with its own codebook
    count -- and where a uniform one still does, with no branch.

    This is the whole of what SNAC tested about "codes in, frame-major": 3 codebooks, 7 ids in a row.
    """
    config = _build_snac(tmp_path, "/tmp/x.gguf")
    config._vq_strides = [4, 2, 1]
    assert (config._coarse_stride, config._codes_per_frame) == (4, 7)
    config._vq_strides = [1, 1, 1, 1]
    assert (config._coarse_stride, config._codes_per_frame) == (1, 4)
    config._vq_strides = [8, 4, 2, 1]
    assert (config._coarse_stride, config._codes_per_frame) == (8, 15)


def test_the_declared_frame_rate_is_the_rate_of_the_ROWS(tmp_path):
    """A caller sizes a clip in rows, so `codec.frame_rate` has to be the rate of a row -- for SNAC
    the COARSEST codebook's, one quarter of the codec's own finest.

    And `codec.n_codebooks` stays the row WIDTH, which is the meaning it was given: it is the pairing
    check between a family-10 LM and its codec (loom-py's `test_codec_pair`), and an LM over SNAC
    emits 7 ids per step. Reporting the quantizer count here would read true and break that pair.
    """
    config = _build_snac(tmp_path, "/tmp/x.gguf")
    config._vq_strides, config._n_codebooks = [4, 2, 1], 3
    config._codebook_size, config._sample_rate, config._hop_length = 4096, 24000, 512
    assert config.hparams() == {
        "codec.n_codebooks": 7,
        "codec.codebook_size": 4096,
        "codec.frame_rate": 24000 / 512 / 4,
        "sample_rate": 24000,
    }


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

    kwargs.setdefault("architecture", None)
    config = AudioCodecExportConfig(output_path=str(out), model_dir=str(checkpoint), **kwargs)
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


def _tiny_snac(tmp_path: Path) -> Path:
    """A real, randomly-initialised `SNAC`, small enough to trace in a unit test.

    `pytest.importorskip` rather than a vendored stub: SNAC is an optional dependency of this family
    (`pip install --no-deps snac`), and what is under test is its own multi-rate `from_codes`.
    """
    pytest.importorskip("torch")
    snac = pytest.importorskip("snac")
    import torch

    out = tmp_path / "tiny-snac"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(SNAC_CONFIG))
    torch.save(snac.SNAC(**SNAC_CONFIG).state_dict(), out / "pytorch_model.bin")
    return out


def _snac_noise(model, frames):
    """One tensor per stochastic leaf, at the length that leaf's own stage needs."""
    import torch

    return [torch.randn((1, 1, frames * m))
            for m in CodecFamily.SNAC.noise_multiples(model)]


def test_the_wrapper_slices_the_row_into_the_codebooks_the_model_expects(tmp_path):
    """The wrapper owes the tensor it took over: its `[1, n_frames, 7]` in must decode to exactly what
    the package's own `decode` returns for the three-tensor list a caller would have built by hand.

    Byte-identical rather than close -- it is the same arithmetic on the same weights, and the only
    question is whether the slicing put each id in the right codebook at the right rate. A level-major
    layout read as anything else still has the right shape, the right length and audio in it.

    The noise is held FIXED across the two, which is what makes the comparison exact rather than
    distributional -- the whole reason it is an input.
    """
    pytest.importorskip("torch")
    import torch

    from loom_exporter.audio_codec_export import _CodecDecodeWrapper, _SNAC_NOISE
    import loom_exporter.audio_codec_export as module

    model = CodecFamily.SNAC.load(str(_tiny_snac(tmp_path)))
    frames = 5
    rows = torch.randint(0, SNAC_CONFIG["codebook_size"], (1, frames, 7))
    noise = _snac_noise(model, frames)
    by_hand = [rows[:, :, 0:1].reshape(1, -1), rows[:, :, 1:3].reshape(1, -1),
               rows[:, :, 3:7].reshape(1, -1)]
    assert [tuple(c.shape) for c in by_hand] == [(1, 5), (1, 10), (1, 20)]
    with torch.no_grad():
        module._SNAC_NOISE = list(noise)
        expected = model.decode(by_hand).reshape(1, -1)
        got = _CodecDecodeWrapper(model, CodecFamily.SNAC)(rows, *noise)
    assert torch.equal(got, expected)


def test_the_noise_reaches_the_decoder_and_changes_the_waveform(tmp_path):
    """A noise input nothing reads is the failure this catches, and it is a quiet one: the export
    still runs, the driver still draws, and the audio is the mean decode forever.

    Two draws through the same codes must differ, and by the right ORDER -- a `NoiseBlock` whose
    tensor was broadcast from the wrong stage would still move the output.
    """
    pytest.importorskip("torch")
    import torch

    from loom_exporter.audio_codec_export import _CodecDecodeWrapper

    model = CodecFamily.SNAC.load(str(_tiny_snac(tmp_path)))
    frames = 5
    rows = torch.randint(0, SNAC_CONFIG["codebook_size"], (1, frames, 7))
    wrapper = _CodecDecodeWrapper(model, CodecFamily.SNAC)
    with torch.no_grad():
        a = wrapper(rows, *_snac_noise(model, frames))
        b = wrapper(rows, *_snac_noise(model, frames))
        zeros = [torch.zeros_like(t) for t in _snac_noise(model, frames)]
        mean_a = wrapper(rows, *zeros)
        mean_b = wrapper(rows, *zeros)
    assert not torch.equal(a, b), "two draws gave the same waveform -- the noise is not being read"
    assert torch.equal(mean_a, mean_b), "zero noise is not deterministic"
    # Zero noise IS the mean decode, which is what the deterministic export used to ship: it must sit
    # between the two draws rather than off to one side, or the noise is entering with a bias.
    spread = (a - b).abs().mean()
    assert (a - mean_a).abs().mean() < spread and (b - mean_a).abs().mean() < spread


def test_a_deterministic_codec_declares_no_noise_and_no_axes(tmp_path):
    """DAC's export must not move. The noise machinery is per-checkpoint -- walked off the real
    decoder's own `NoiseBlock`s -- so a codec that has none declares empty, and empty is what its
    GGUF has always carried."""
    config = _build_dac(tmp_path, "/tmp/x.gguf")
    config._noise_multiples = []
    assert config._noise_inputs == {}
    assert config.backend_kwargs()["declared_axes"] == {}
    assert config.backend_kwargs()["noise_inputs"] == {}


def test_a_multi_rate_codec_keeps_the_length_in_the_root_axis(tmp_path):
    """The same assertion the DAC test makes, on the leaf that could plausibly have lost it: three
    slices at three rates, each of which has to stay an expression in `n_codes`.

    The three level crops are the check that matters -- `['1', n_codes, '1']`, `['2', ...]`,
    `['4', ...]` -- because a slice that baked its length would still decode, at the traced number of
    frames, forever.
    """
    pytest.importorskip("coremltools")
    pytest.importorskip("snac")
    exported = _export(_tiny_snac(tmp_path), tmp_path / "tiny-snac.gguf", n_frames=8,
                       family=CodecFamily.SNAC, architecture="snac")
    topo = exported["topology"]

    assert {i["name"]: i["shape"] for i in topo["inputs"]} == {
        "codes": ["7", "n_codes", "1"],
        # Each stochastic leaf at its own stage's multiple of the root axis, never a literal: a baked
        # length here is a call that fails on the shape at any other frame count.
        "noise_0": ["8*n_codes", "1", "1"], "noise_1": ["16*n_codes", "1", "1"],
    }
    assert exported["n_codebooks"] == 7 and exported["sample_rate"] == 24000
    assert "math.floor(#codes / 7)" in exported["driver"]
    # The driver draws them, seeded, and lets a caller hand them in instead -- which is what keeps
    # this family's oracle exact rather than distributional.
    assert "loom.seed_rng((inputs.seed or 1234))" in exported["driver"]
    assert ("local noise_0 = (inputs.noise_0 or loom.gaussian_array((8 * math.floor(#codes / 7))))"
            in exported["driver"])

    crops = [n["attrs"]["shape"] for n in topo["nodes"] if n["op"] == "VIEW"]
    assert [c for c in crops if c[1:] == ["n_codes", "1"]][:3] == [
        ["1", "n_codes", "1"], ["2", "n_codes", "1"], ["4", "n_codes", "1"]
    ], f"the per-level slices are not the three rates: {crops}"
    assert all(any("n_codes" in str(d) for d in c) for c in crops), (
        f"crop shapes with no dynamic axis in them: "
        f"{[c for c in crops if not any('n_codes' in str(d) for d in c)]}"
    )


def test_the_traced_length_does_not_reach_the_graph(tmp_path):
    """Two exports differing only in `n_frames`. A single export passes against a graph with the
    length baked in, because the shape it was baked at is the shape it is asked for."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_codec(tmp_path)
    at_8 = _export(checkpoint, tmp_path / "a.gguf", n_frames=8)
    at_32 = _export(checkpoint, tmp_path / "b.gguf", n_frames=32)
    assert at_8["topology"] == at_32["topology"]
    assert at_8["driver"] == at_32["driver"]
