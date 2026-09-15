"""Family 5 (P5): SANM / FunASR encoders -- SenseVoiceSmall, and the Paraformer leaves after it.

Three things are tested here and the split is deliberate.

**The front end is tested against `torchaudio` at many LENGTHS, because it is a rewrite.** Kaldi's
fbank and FunASR's `apply_lfr` are written with `as_strided` and Python-level shape arithmetic and
neither traces; `KaldiFbankLfrCmvn` rebuilds both out of convolutions and matmuls. A rewrite that is
right at one length and wrong at another is exactly what this family's end padding could have produced
-- the reference pads by an amount that depends on the frame count modulo the LFR stride, and the graph
pads by a constant -- so the sweep walks every residue rather than checking one clip.

**The emitted shape EXPRESSIONS are asserted, not just the export**, which is
[Retro-044](../../../loom.cpp/docs/retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)'s
own takeaway and what [Retro-048](../../../loom.cpp/docs/retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md)
cost to relearn. Four separate producers -- a `fill` from `ones_like`, `loom_scale` from a lowered
`reduce_mean`, `loom_broadcast_to` from an inserted broadcast, and a `keep_dims=True` `reduce_sum` --
each stopped the shape walk and each silently substituted the root axis, producing a graph that
converts, loads, builds, and declares one row per audio SAMPLE. None of them raises. The only thing
that catches it is reading the expression back.

**Detection needs no torch and no funasr at all**, and is where a new checkpoint's first failure shows.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.sanm_asr_export import (
    MAX_SECONDS,
    MIN_SECONDS,
    SANMAsrExportConfig,
    _is_funasr_sensevoice,
    stage_spm_protobuf,
)
from loom_exporter.registry import default_registry

CMVN_DIM = 560


def _funasr_dir(tmp_path: Path, name: str, config: str, *, with_weights: bool = True) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.yaml").write_text(config)
    if with_weights:
        (d / "model.pt").write_bytes(b"")
    return d


SENSEVOICE_YAML = "model: SenseVoiceSmall\nencoder: SenseVoiceEncoderSmall\n"
PARAFORMER_YAML = "model: Paraformer\nencoder: SANMEncoder\npredictor: CifPredictorV2\n"


# -- detection -----------------------------------------------------------------------------------

def test_a_sensevoice_checkpoint_is_claimed(tmp_path):
    assert _is_funasr_sensevoice(_funasr_dir(tmp_path, "sv", SENSEVOICE_YAML))


def test_a_paraformer_checkpoint_is_not_claimed(tmp_path):
    """The second leaf of this family, and NOT this template's: it shares the SANM encoder and adds a
    CIF predictor and a non-autoregressive decoder that nothing here exports. Claiming it structurally
    -- on `encoder: SANMEncoder`, which both declare -- would produce a GGUF missing two thirds of the
    model, so the recognizer names the model class instead."""
    assert not _is_funasr_sensevoice(_funasr_dir(tmp_path, "pf", PARAFORMER_YAML))


def test_a_config_without_weights_beside_it_is_not_claimed(tmp_path):
    assert not _is_funasr_sensevoice(
        _funasr_dir(tmp_path, "sv", SENSEVOICE_YAML, with_weights=False))


def test_a_directory_that_is_not_a_funasr_checkpoint_is_a_no_not_an_error(tmp_path):
    """`TaskRegistry.detect` runs every recognizer against every path by construction, so "not mine"
    has to be an answer rather than an exception -- including for a `config.yaml` that will not parse."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert not _is_funasr_sensevoice(plain)
    broken = _funasr_dir(tmp_path, "broken", "model: [unclosed\n")
    assert not _is_funasr_sensevoice(broken)


def test_the_registry_routes_a_sensevoice_directory_here(tmp_path):
    path = _funasr_dir(tmp_path, "sv", SENSEVOICE_YAML)
    assert default_registry().detect(path).name == "funasr-sensevoice"


# -- the tokenizer adapter -----------------------------------------------------------------------

def test_the_protobuf_is_staged_under_the_name_the_writer_looks_for(tmp_path):
    """`_write_tokenizer`'s `sentencepiece_proto` branch reads one of three fixed filenames out of a
    directory, and this family's checkpoint names its protobuf after the languages it covers. The name
    is read out of `configuration.json` rather than globbed, so a checkpoint shipping two `.model`
    files cannot have the wrong one picked."""
    d = _funasr_dir(tmp_path, "sv", SENSEVOICE_YAML)
    (d / "chn_jpn_yue_eng_ko_spectok.bpe.model").write_bytes(b"proto-bytes")
    (d / "configuration.json").write_text(json.dumps({
        "file_path_metas": {"tokenizer_conf": {"bpemodel": "chn_jpn_yue_eng_ko_spectok.bpe.model"}}}))

    staged = stage_spm_protobuf(str(d))
    assert (Path(staged) / "tokenizer.model").read_bytes() == b"proto-bytes"


def test_a_checkpoint_with_no_protobuf_stages_nothing_rather_than_raising(tmp_path):
    """`component_registry.usage()` builds every registered config without a checkpoint on disk, and
    `backend_kwargs()` is on that path -- so "no protobuf here" is a normal answer."""
    assert stage_spm_protobuf(str(_funasr_dir(tmp_path, "sv", SENSEVOICE_YAML))) is None


# -- what the config declares --------------------------------------------------------------------

def test_the_driver_builder_is_family_1s():
    """The CTC epilogue is inherited whole -- the roadmap's one accurate prediction about this family."""
    config = SANMAsrExportConfig(architecture=None, output_path="x.gguf", model_dir="d")
    assert config.synthesized_builder_key() == "CtcGreedy"


def test_the_blank_and_the_prompt_are_absent_before_the_trace():
    """Both are READ off the checkpoint during `build_trace`, and `backend_kwargs` runs for callers
    that never trace. Omitted rather than guessed: the exporter raises when it is asked for the CTC
    builder without a blank id, which is the moment the number is needed."""
    kwargs = SANMAsrExportConfig(architecture=None, output_path="x.gguf",
                                 model_dir="d").backend_kwargs()
    assert "ctc_blank_id" not in kwargs
    assert "defaulted_inputs" not in kwargs
    assert kwargs["driver_builder"] == "CtcGreedy"


def test_the_prompt_ids_are_not_published_as_whisper_language_ids():
    """The ids index a 16-row embedding table prepended to the FEATURES. `loom.asr.language_ids` is
    read into `AsrDecodeTable`, whose ids are decoder prompt TOKENS pushed into a cross-attention
    prompt. Same concept for a caller, different object for the engine."""
    config = SANMAsrExportConfig(architecture=None, output_path="x.gguf", model_dir="d")
    config.task = "automatic-speech-recognition"
    config._languages = {"auto": 0, "zh": 3, "en": 4, "nospeech": 13}
    config._textnorms = {"withitn": 14, "woitn": 15}
    contract = config.contract()

    assert "asr.language_ids" not in contract
    assert "asr.language_names" not in contract
    # `auto` and `nospeech` are prompt rows, not languages a caller can ask a model to speak.
    assert contract["text.languages"] == ["zh", "en"]
    assert contract["sanm.language_names"] == ["auto", "zh", "en", "nospeech"]
    assert contract["sanm.language_ids"] == [0, 3, 4, 13]
    assert contract["sanm.textnorm_ids"] == [14, 15]


def test_the_dynamic_bounds_bracket_the_trace_length():
    assert MIN_SECONDS < 1.0 <= MAX_SECONDS


# -- the front end, against the reference it replaces ---------------------------------------------

def _reference_frontend(samples, cmvn):
    """FunASR's own `WavFrontend` arithmetic, with dither forced to 0.

    The default is kaldi's, `1.0`, so the reference adds Gaussian noise to the waveform and does not
    produce the same features twice. A graph has no dither; comparing against a dithered reference
    grades noise.
    """
    import torch
    from funasr.frontends.wav_frontend import apply_cmvn, apply_lfr
    from funasr.utils import fbank as kaldi

    mat = kaldi.fbank(torch.as_tensor(samples)[None, :] * (1 << 15), num_mel_bins=80,
                      frame_length=25, frame_shift=10, dither=0.0, energy_floor=0.0,
                      window_type="hamming", sample_frequency=16000, snip_edges=True)
    return apply_cmvn(apply_lfr(mat, 7, 6), cmvn)


@pytest.fixture
def cmvn():
    torch = pytest.importorskip("torch")
    # A real shift/scale pair rather than an identity: an affine that happens to be the identity would
    # pass whether or not the two halves were applied in the right order or at all.
    g = torch.Generator().manual_seed(11)
    return torch.stack((torch.randn(CMVN_DIM, generator=g) * 0.5,
                        torch.rand(CMVN_DIM, generator=g) + 0.5))


@pytest.mark.parametrize("n_samples", [1600, 1601, 1602, 1603, 1604, 1605, 7777, 16000, 40321])
def test_the_rebuilt_front_end_matches_kaldi_at_every_length(cmvn, n_samples):
    """Every residue of the frame count modulo `lfr_n`, plus three longer clips.

    The lengths are not decorative. `apply_lfr` pads the frame sequence at the end by an amount that
    depends on `n_frames % lfr_n` -- between -2 and 3 rows here -- and the graph pads by the maximum
    unconditionally, because a data-dependent pad bakes the trace length's own remainder. The claim
    that over-padding is unobservable is only checkable by walking the residues.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("funasr")
    from loom_exporter.sanm_asr_export import KaldiFbankLfrCmvn

    g = torch.Generator().manual_seed(n_samples)
    samples = torch.randn(n_samples, generator=g) * 0.1

    with torch.no_grad():
        reference = _reference_frontend(samples, cmvn)
        got = KaldiFbankLfrCmvn(cmvn)(samples[None, :])[0]

    assert got.shape == reference.shape
    # RELATIVE to the reference's own magnitude, not absolute: white noise at this amplitude produces
    # log-mel values several times larger than speech does, so an absolute bound would encode the test
    # signal's loudness rather than the rewrite's accuracy.
    #
    # The residual is f32 accumulation in a 512-term DFT written as a matmul rather than an FFT. At f64
    # the two agree to 3.7e-07 on a scale of 4.7 -- and on real speech `torchaudio`'s OWN f32-vs-f64
    # spread (4.7e-05) is larger than this rewrite's disagreement with it, which is what says the gap is
    # accumulation and not algebra.
    error = (got - reference).abs().max() / reference.abs().max()
    assert error < 1e-4, error


def test_the_positions_are_one_based(cmvn):
    """`SinusoidalPositionEncoder` numbers from 1, and a zero-based range shifts every row of the table
    by one -- output that is wrong everywhere and plausible everywhere (loom.cpp Retro-039, one family
    over)."""
    torch = pytest.importorskip("torch")
    from loom_exporter.sanm_asr_export import sinusoidal_positions

    x = torch.zeros(1, 5, 8)
    table = sinusoidal_positions(x)
    # depth 8 -> inv_timescales[0] == 1, so channel 0 is sin(position) for position 1..5.
    assert torch.allclose(table[0, :, 0], torch.sin(torch.arange(1, 6, dtype=torch.float32)),
                          atol=1e-6)


# -- the emitted shape expressions ----------------------------------------------------------------

def _position_graph_topology():
    """A module with this family's SHAPE -- a strided convolution for the dynamic axis, then the
    position derivation on top of it -- converted and lowered exactly as a real export does.

    Deliberately not the real checkpoint: 70 SANM blocks take five minutes to convert and prove nothing
    the eleven ops here do not. What is being tested is the shape WALK, and every producer it has to
    cross is present: a `conv` with a real stride formula, `ones_like` (a `fill` shaped by `shape(x)`),
    `reduce_mean` (which `lower_reduce_mean` rewrites into `reduce_sum` + `loom_scale`), `cumsum`, and
    the `loom_broadcast_to` pair `insert_explicit_broadcasts` splices in front of the multiply.
    """
    import coremltools as ct
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from loom_exporter.exporter import LoomGGUFExporter
    from loom_exporter.passes import apply_loom_mil_passes
    from loom_exporter.sanm_asr_export import sinusoidal_positions

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 1, 7))

        def forward(self, waveform):
            x = F.conv1d(waveform.unsqueeze(1), self.weight, stride=6).transpose(1, 2)
            return x + sinusoidal_positions(x)

    traced = torch.jit.trace(Model().eval(), (torch.randn(1, 601),), check_trace=False)
    program = ct.convert(
        traced, inputs=[ct.TensorType(name="waveform", shape=(1, ct.RangeDim(100, 9000)),
                                       dtype=np.float32)],
        convert_to="milinternal", compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17)
    apply_loom_mil_passes(program)
    exporter = LoomGGUFExporter(program, flat_namespace=True, root_axis="n_samples")
    return exporter.generate_graph_topology(program.functions["main"], "main_topology")


def test_the_position_table_is_not_declared_one_row_per_sample():
    """The regression guard for all four shape-walk gaps (loom.cpp Retro-048).

    Each of `fill`, `loom_scale`, `loom_broadcast_to` and a `keep_dims=True` `reduce_sum` stopped the
    walk, which then substituted the root axis. The resulting graph converts, exports and loads; it
    fails only when the engine tries to broadcast a 176,000-row position table against 187 rows of
    encoder state, and on a shorter clip it might not fail at all. Reading the expression back is the
    only thing that catches it, so the expression is what is asserted.
    """
    pytest.importorskip("coremltools")
    pytest.importorskip("torch")
    import sympy

    topology = _position_graph_topology()

    frames = sympy.sympify("floor((n_samples - 7)/6) + 1")   # conv, kernel 7 stride 6

    # REPEAT is where this fails, and scoping to it is the point rather than convenience: every one of
    # the four gaps ended in a REPEAT target (a `fill`/`fill_like` materialising the ones, and the
    # broadcast pair in front of the multiply), and a REPEAT is also the node ggml checks hardest, so a
    # wrong extent here is the difference between a working model and an abort. `n_samples` elsewhere
    # is not a defect -- the waveform's own reshape genuinely has one element per sample.
    repeats = [n for n in topology["nodes"] if n["op"] == "REPEAT"]
    assert repeats, "the position broadcast should emit REPEAT nodes; this test proves nothing without"
    for node in repeats:
        for dim in node["attrs"]["shape"]:
            if not isinstance(dim, str) or dim.isdigit():
                continue
            assert sympy.sympify(dim).equals(frames), (
                f"REPEAT declares {dim!r}, which is not the frame count {frames} -- the shape walk "
                f"fell back to the root axis somewhere in this chain")
