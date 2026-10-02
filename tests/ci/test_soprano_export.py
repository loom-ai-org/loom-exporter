"""Family 9's eighth leaf (P5): Soprano TTS -- a Qwen3 LM whose final-norm hidden rows, not its drawn
ids, are what the Vocos decoder reads.

What is tested here is what the engine reads as DATA and what the trace re-spells, because a wrong
declaration there is a wrong model with nothing failing:

* the recognizer, and that the causal-LM family's Qwen3 recognizer steps aside for it;
* the decoder's re-spellings against the reference module: the linear interpolation as a transposed
  convolution, and the ISTFT head with its DC and Nyquist bins zeroed;
* the text front end's data: every inline pattern found in the reference's source, `unidecode`'s table;
* the declarations: the text-door contract, the cached LM, the reference's sampling defaults.

The numbers -- the waveform at max|d| 1.6e-05 against the reference over 143,360 samples, and the text
path at 20,000/20,000 -- are the engine gate's and loom.cpp's `test_soprano_vocab`'s.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from loom_exporter.soprano_export import (
    DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, LM_MAX_POSITIONS,
    MAX_NEW_TOKENS, SAMPLES_PER_ROW, SOPRANO_REPO, DecoderPhase, SopranoExportConfig, _is_soprano,
    linear_upsample_kernel,
)
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/soprano-1.1-80m")
HAVE_REFERENCE = Path(SOPRANO_REPO, "soprano").is_dir()
needs_reference = pytest.mark.skipif(not HAVE_REFERENCE, reason="no soprano checkout")


def _checkpoint(tmp_path: Path, model_type="qwen3", decoder=True) -> Path:
    d = tmp_path / "soprano"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type,
                                               "architectures": ["Qwen3ForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"")
    if decoder:
        (d / "decoder.pth").write_bytes(b"")
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_qwen3_checkpoint_with_a_decoder_is_claimed(tmp_path):
    assert _is_soprano(_checkpoint(tmp_path))


def test_a_plain_qwen3_is_not_claimed(tmp_path):
    assert not _is_soprano(_checkpoint(tmp_path, decoder=False))


def test_another_architecture_beside_a_decoder_is_not_claimed(tmp_path):
    assert not _is_soprano(_checkpoint(tmp_path, model_type="llama"))


def test_the_registry_routes_it_here_and_not_to_the_causal_lm_family(tmp_path):
    # `detect` raises on more than one specific match, so this pins that qwen3's recognizer stepped aside.
    rec = default_registry().detect(_checkpoint(tmp_path))
    assert (rec.task, rec.name) == ("text-to-speech", "soprano")


def test_a_plain_qwen3_still_routes_to_the_causal_lm_family(tmp_path):
    assert default_registry().detect(_checkpoint(tmp_path, decoder=False)).name == "qwen3"


# -- the decoder's re-spellings --------------------------------------------------------------------

@pytest.mark.parametrize("length", [1, 2, 3, 8, 17])
def test_the_upsample_kernel_is_linear_interpolation_with_aligned_corners(length):
    """`F.interpolate(size=4*(T-1)+1, mode='linear', align_corners=True)`, the reference's call, as an
    unpadded depthwise transposed convolution cropped by 3 at each end -- the form `DecoderPhase` traces."""
    x = torch.randn(1, 6, length, dtype=torch.float64)
    want = torch.nn.functional.interpolate(x, size=4 * (length - 1) + 1, mode="linear", align_corners=True)
    kernel = linear_upsample_kernel(4).double().view(1, 1, -1).repeat(6, 1, 1)
    got = torch.nn.functional.conv_transpose1d(x, kernel, stride=4, groups=6)[..., 3:-3]
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-14)


@needs_reference
def test_the_decoder_phase_is_the_reference_decoder():
    """`SopranoDecoder` at random weights against `DecoderPhase` at f64. The reference zeroes the DC
    and Nyquist bins in place before `torch.istft`; the phase multiplies by a 0/1 mask instead."""
    from loom_exporter.soprano_export import import_soprano

    import_soprano()
    from soprano.vocos.decoder import SopranoDecoder

    torch.manual_seed(0)
    decoder = SopranoDecoder().double().eval()
    phase = DecoderPhase(decoder).double().eval()
    hidden = torch.randn(1, 9, 512, dtype=torch.float64)
    with torch.no_grad():
        want = decoder(hidden.transpose(1, 2))[0].squeeze()
        got = phase(hidden)[0]
    # L rows -> 2048 * (L - 1) samples, which is why the reference's trim keeps everything.
    assert got.shape[0] == want.shape[0] == SAMPLES_PER_ROW * 8
    # The ISTFT basis is f32 by construction (istft.py), so f64 agrees to f32's resolution.
    assert torch.allclose(got, want, atol=1e-6 * want.abs().max().item())


# -- the text front end's data ---------------------------------------------------------------------

@needs_reference
def test_every_inline_pattern_is_found_in_the_references_source():
    from loom_exporter import soprano_tokenizer_export as t

    text_normalizer, text_splitter, tts = t.import_reference()
    rules = t.inline_rules((text_normalizer, text_splitter, tts.SopranoTTS))
    assert len(rules) == len(t.INLINE_RULES)


@needs_reference
def test_a_pattern_the_reference_no_longer_spells_is_refused(monkeypatch):
    from loom_exporter import soprano_tokenizer_export as t

    text_normalizer, text_splitter, tts = t.import_reference()
    changed = t.INLINE_RULES + (("bogus", "collapse_whitespace", "r'\\t+'", None),)
    monkeypatch.setattr(t, "INLINE_RULES", changed)
    with pytest.raises(ValueError, match="no longer spells"):
        t.inline_rules((text_normalizer, text_splitter, tts.SopranoTTS))


def test_the_unidecode_table_is_unidecode():
    pytest.importorskip("unidecode")
    from unidecode import unidecode

    from loom_exporter.soprano_tokenizer_export import unidecode_table

    cps, offsets, text = unidecode_table()
    assert len(offsets) == len(cps) + 1 and offsets[-1] == len(text)
    table = {cp: text[offsets[i]:offsets[i + 1]] for i, cp in enumerate(cps)}
    for c in "é—“£ßÆ日Ⅻ 😀":
        assert table.get(ord(c), "") == unidecode(c)


def test_a_tokenizer_of_another_shape_is_refused(tmp_path):
    from loom_exporter.soprano_tokenizer_export import read_tokenizer

    spec = {"model": {"type": "BPE", "vocab": {}, "merges": [], "byte_fallback": True},
            "normalizer": None, "pre_tokenizer": None, "post_processor": None}
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="not the shape"):
        read_tokenizer(str(tmp_path))


# -- declarations ----------------------------------------------------------------------------------

def test_the_tokenizer_is_named_rather_than_detected(tmp_path):
    kwargs = SopranoExportConfig(output_path=str(tmp_path / "x.gguf"), model_dir=str(tmp_path)).backend_kwargs()
    assert kwargs["tokenizer_family"] == "soprano"
    assert kwargs["tokenizer_dir"] == str(tmp_path)


def test_the_contract_declares_a_text_door(tmp_path):
    contract = SopranoExportConfig(output_path=str(tmp_path / "x.gguf"), model_dir=str(tmp_path)).contract()
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == 32000


def test_the_sampling_defaults_are_the_references_own():
    """`SopranoTTS.infer`: temperature 0 becomes 0.001 in the backend, top-p 0.95, penalty 1.2; and
    `generate`'s own top-k of 50, since the checkpoint's generation config sets none."""
    from transformers import GenerationConfig

    assert (DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY) == (0.001, 0.95, 1.2)
    assert DEFAULT_TOP_K == GenerationConfig().top_k
    # A 512-id prompt and 512 drawn ids fit the cache.
    assert LM_MAX_POSITIONS >= 512 + MAX_NEW_TOKENS


@pytest.mark.skipif(not (HAVE_REFERENCE and MODEL_DIR.is_dir()), reason="no soprano checkpoint")
def test_the_lm_is_cached_and_its_attention_fused(tmp_path):
    phases = {p.name: p for p in SopranoExportConfig(output_path=str(tmp_path / "x.gguf"),
                                                     model_dir=str(MODEL_DIR)).phases()}
    assert set(phases) == {"lm", "decoder"}
    assert phases["lm"].fuse_attention and phases["lm"].kv_cache_size == LM_MAX_POSITIONS
    assert not phases["decoder"].fuse_attention
