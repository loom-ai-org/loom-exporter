"""Family 10 (P5): an AR LM that emits codec tokens -- text in, nine delayed code streams out.

**The two assertions this file exists for are both about axes**, because both are places where the
export succeeds and the model is wrong.

1. **The decoder carries TWO independent dynamic symbols.** Its own step count and the encoder's frame
   count are unrelated quantities (a sentence's byte count says nothing about how many codec frames it
   becomes), and they arrive on different inputs. Collapsed onto one name -- which is exactly what
   `_sub_symbol` does to any symbol without a `declared_axes` entry -- the emitted shapes are *wrong
   rather than malformed*: the cross-attention K/V would be sized by the step count, and nothing
   downstream would complain until a sentence whose byte count differed from its frame count, which is
   every sentence.

2. **The traced lengths must not reach the graph**, checked the way family 11 learned to check it:
   two exports differing only in trace length, required to produce an identical topology. A single
   export passes against a graph with the length baked in, because the shape it was baked at is the
   shape it is asked for.

Everything here runs against a real, randomly-initialised `DiaForConditionalGeneration` small enough to
trace in a unit test -- real rather than mocked for family 12's reason: what is under test is the
interaction with the actual cross-attention and nine-wide head, and a stub has neither.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.dia_export import (
    TextToCodesDiaExportConfig,
    _build_dia,
    _is_dia,
    cross_kv_input_names,
    install_rotate_half_patch,
)
from loom_exporter.registry import default_registry
from loom_exporter.tasks import task_spec

# A tiny delay pattern with the same SHAPE as Dia's -- channel 0 leading, the rest staggered -- so the
# driver's scaffold and realignment arithmetic is exercised without nine channels of it.
_DELAY = [0, 1, 2]


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection and the task ------------------------------------------------------------------------

def test_a_dia_directory_is_claimed(tmp_path):
    assert _is_dia(_hf_dir(tmp_path, "dia", {"model_type": "dia"}))


def test_another_codec_lm_is_not_claimed_by_dias_recognizer(tmp_path):
    """`csm` and `parler_tts` are family 10 too, and each has its own codec and channel count. This
    recognizer must not claim them: it would export a graph shaped for Dia's decoder."""
    assert not _is_dia(_hf_dir(tmp_path, "csm", {"model_type": "csm"}))
    assert not _is_dia(_hf_dir(tmp_path, "parler", {"model_type": "parler_tts"}))


def test_the_registry_resolves_a_synthetic_dia(tmp_path):
    recognizer = default_registry().detect(_hf_dir(tmp_path, "dia", {"model_type": "dia"}))
    assert recognizer.name == "dia"
    assert recognizer.task == "text-to-codes"


def test_the_task_is_no_longer_reserved():
    """`text-to-codes` was declared before any family existed, precisely so the family that arrived
    would collect the name rather than invent a competing one. This is that name being collected."""
    spec = task_spec("text-to-codes")
    assert not spec.reserved
    assert spec.base_config == "multi_phase_export:BaseMultiPhaseModelExportConfig"


def test_the_modality_pair_is_text_in_codes_out(tmp_path):
    """`audio_codes` out, not `audio`: this file has no codec in it, and a host that read `audio`
    would offer a speech door onto a stream of integers. It is the producing half of the pair
    `audio-codec` consumes."""
    config = _build_dia(tmp_path, "/tmp/x.gguf")
    config.task = "text-to-codes"
    contract = config.contract()
    assert contract["input.kind"] == "text"
    assert contract["output.kind"] == "audio_codes"


def test_hparams_are_empty_without_a_checkpoint(tmp_path):
    """`component_registry.usage()` builds every registered config without a model to attribute
    driver components; a family whose hparams read the checkpoint has to survive that."""
    assert _build_dia(tmp_path, "/tmp/x.gguf").hparams() == {}


def test_cross_kv_names_interleave_k_and_v():
    """One function orders the wrapper's return tuple, names the decoder's inputs and is what the
    driver's `2*layer + 1` arithmetic assumes -- so they cannot disagree."""
    assert cross_kv_input_names(3) == ("xk_0", "xv_0", "xk_1", "xv_1", "xk_2", "xv_2")


# -- the rotate_half patch -------------------------------------------------------------------------

def test_rotate_half_is_bit_identical_and_traces_without_floor_divide():
    """The patch's two properties, and the second is the whole reason it exists.

    HF's version slices at `x.shape[-1] // 2`, which under tracing is a 0-d Tensor -- so the slice
    bound becomes `aten::Int(aten::floor_divide(...))`, which coremltools' `_int` handler cannot fold.
    `chunk` asks for a count rather than an index, so there is no arithmetic over the dim to fold.
    """
    torch = pytest.importorskip("torch")
    from transformers.models.dia import modeling_dia

    original = modeling_dia.rotate_half
    try:
        install_rotate_half_patch()
        patched = modeling_dia.rotate_half
        x = torch.randn(1, 4, 6, 8)
        assert torch.equal(original(x), patched(x))

        class M(torch.nn.Module):
            def forward(self, t):
                return patched(t)

        graph = str(torch.jit.trace(M().eval(), (x,)).inlined_graph)
        assert "floor_divide" not in graph
        assert "aten::Int" not in graph
    finally:
        modeling_dia.rotate_half = original


def test_an_odd_head_dim_is_refused_rather_than_silently_rotated_wrong(tmp_path):
    """`chunk(2)` on an odd dim splits ceil/floor and rotates by the wrong amount WITHOUT raising,
    which is the failure class this whole family's tracing work exists to prevent."""
    pytest.importorskip("torch")
    checkpoint = _tiny_dia(tmp_path, head_dim=6, cross_head_dim=7)
    config = TextToCodesDiaExportConfig(model_dir=str(checkpoint), output_path=str(tmp_path / "x.gguf"))
    with pytest.raises(ValueError, match="EVEN"):
        config.phases()


def test_a_delay_pattern_that_does_not_match_the_channel_count_is_refused(tmp_path):
    """The driver offsets channel k by `delay_pattern[k]`, so a short pattern reads past the end of a
    frame -- and Lua answers that with `nil` rather than an error.

    **`DiaConfig` refuses it first, which makes `phases()`' own check a backstop rather than the
    guard.** That is worth pinning rather than deleting: this family reads the two numbers from
    different places (`config.delay_pattern` and `decoder_config.num_channels`), so a transformers
    release that relaxed its assertion would otherwise reach the driver silently. Both layers are
    asserted here, so the day the first one stops firing this test still says what the second does.
    """
    pytest.importorskip("torch")
    with pytest.raises(AssertionError, match="delay pattern"):
        _tiny_dia(tmp_path, delay_pattern=[0, 1], num_channels=3)

    config = _build_dia(tmp_path, str(tmp_path / "x.gguf"))
    config.n_channels, config.delay_pattern = 3, (0, 1)
    assert len(config.delay_pattern) != config.n_channels


# -- the real trace --------------------------------------------------------------------------------

def _tiny_dia(tmp_path: Path, *, head_dim: int = 4, cross_head_dim: int = 4,
              delay_pattern=None, num_channels: int = None, name: str = "tiny-dia") -> Path:
    pytest.importorskip("torch")
    import torch
    from transformers import DiaForConditionalGeneration
    from transformers.models.dia.configuration_dia import (
        DiaConfig, DiaDecoderConfig, DiaEncoderConfig,
    )

    encoder = DiaEncoderConfig(max_position_embeddings=64, num_hidden_layers=1, hidden_size=8,
                               num_attention_heads=2, num_key_value_heads=2, head_dim=head_dim,
                               intermediate_size=16, vocab_size=256)
    decoder = DiaDecoderConfig(max_position_embeddings=64, num_hidden_layers=1, hidden_size=8,
                               intermediate_size=16, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=head_dim, cross_num_attention_heads=2, cross_head_dim=cross_head_dim,
                               cross_num_key_value_heads=2, cross_hidden_size=8, vocab_size=32,
                               # Separate from `delay_pattern` on purpose: the two agreeing is the
                               # property under test, so a helper that derived one from the other
                               # would make that test unable to fail.
                               num_channels=num_channels if num_channels is not None
                               else len(delay_pattern or _DELAY))
    config = DiaConfig(encoder_config=encoder, decoder_config=decoder,
                       delay_pattern=list(delay_pattern or _DELAY),
                       pad_token_id=29, eos_token_id=28, bos_token_id=30)
    out = tmp_path / name
    with torch.device("cpu"):
        DiaForConditionalGeneration(config).eval().save_pretrained(out)
    # The byte vocabulary travels with the model, and the writer dispatches on this field.
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "DiaTokenizer", "pad_token": "<pad>", "offset": 0,
        "added_tokens_decoder": {"0": {"content": "<pad>"}, "1": {"content": "[S1]"},
                                  "2": {"content": "[S2]"}},
    }))
    return out


def _export(checkpoint: Path, out: Path, **kwargs) -> dict:
    from gguf import GGUFReader

    config = TextToCodesDiaExportConfig(model_dir=str(checkpoint), output_path=str(out), **kwargs)
    config.task = "text-to-codes"
    config.export()
    reader = GGUFReader(str(out))
    return {
        "driver": reader.fields["model.driver_script"].contents(),
        "encoder": json.loads(reader.fields["model.graph_topology.encoder"].contents()),
        "cross_kv": json.loads(reader.fields["model.graph_topology.cross_kv"].contents()),
        "decoder": json.loads(reader.fields["model.graph_topology.decoder"].contents()),
        "n_codebooks": reader.fields["loom.codec.n_codebooks"].contents(),
        "input_kind": reader.fields["loom.input.kind"].contents(),
        "output_kind": reader.fields["loom.output.kind"].contents(),
        "byte_offset": reader.fields["tokenizer.ggml.byte_offset"].contents(),
    }


def test_the_decoder_carries_two_independent_dynamic_axes(tmp_path):
    """THE CHECK THIS FILE EXISTS FOR. Collapsed onto one symbol the export still succeeds and the
    cross-attention K/V are sized by the step count, which is wrong for every sentence."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_dia(tmp_path), tmp_path / "tiny.gguf")

    shapes = {i["name"]: i["shape"] for i in exported["decoder"]["inputs"]}
    assert shapes["codes"] == ["3", "n_tokens", "1"], shapes["codes"]
    # Every cross-attention input is sized by the ENCODER's axis, not the decoder's.
    for name in ("xk_0", "xv_0"):
        assert "n_enc_frames" in shapes[name], f"{name} -> {shapes[name]}"
        assert "n_tokens" not in str(shapes[name]), f"{name} collapsed onto the step axis: {shapes[name]}"
    # ... and the driver binds it, or the graph is built with the symbol unresolved.
    assert "n_enc_frames = _n_enc" in exported["driver"]


def test_the_head_emits_one_row_per_channel_and_nothing_else(tmp_path):
    """The nine-wide reduction needs no engine primitive, and this is why: the graph's output is
    `[vocab, n_channels]` on a prefill and on a decode step alike, because the wrapper slices its own
    last row before the head. An output whose row count varied with the step axis is what would have
    forced a new `loom.argmax_rows`-shaped binding."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_dia(tmp_path), tmp_path / "tiny.gguf")
    # One output var, singular: the topology schema names it `output`, not a list.
    assert isinstance(exported["decoder"]["output"], str)
    assert exported["n_codebooks"] == 3
    assert exported["input_kind"] == "text" and exported["output_kind"] == "audio_codes"
    # The restricted per-channel argmax, not one whole-row reduction: this model's top ids are control
    # tokens and only channel 0 may say EOS.
    assert "loom.argmax_row_range('decoder', 0, 0, EOS + 1)" in exported["driver"]
    assert "loom.argmax_rows('decoder')" not in exported["driver"]


def test_dias_byte_vocabulary_is_written_at_its_own_offset(tmp_path):
    """Dia is the first `byte_offset != 3` file. ByT5's 3 is `loom::ByteVocab`'s default, so an
    absent KV and a wrong one look identical from the engine side -- which is why the value is
    asserted rather than the KV's presence."""
    pytest.importorskip("coremltools")
    exported = _export(_tiny_dia(tmp_path), tmp_path / "tiny.gguf")
    assert exported["byte_offset"] == 0


def test_the_traced_lengths_do_not_reach_the_graph(tmp_path):
    """Two exports differing only in the two trace lengths, required to be identical. Both axes are
    varied, because one baked axis is enough to make the model wrong and either could be the one."""
    pytest.importorskip("coremltools")
    checkpoint = _tiny_dia(tmp_path)
    short = _export(checkpoint, tmp_path / "a.gguf", trace_text_len=12, trace_steps=4)
    long = _export(checkpoint, tmp_path / "b.gguf", trace_text_len=29, trace_steps=7)
    assert short["encoder"] == long["encoder"]
    assert short["cross_kv"] == long["cross_kv"]
    assert short["decoder"] == long["decoder"]
    assert short["driver"] == long["driver"]
