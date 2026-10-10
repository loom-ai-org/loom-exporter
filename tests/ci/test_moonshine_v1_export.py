"""Moonshine v1 (family 2) -- the hermetic half.

The wrappers against transformers' own modules on a small random checkpoint, at f64. CI's transformers
has `moonshine` (it predates only `moonshine_streaming`), so this is the real reference, not a restated
formula. The config is chosen to reach what the real ones do: a head dimension that is padded
(`pad_head_dim_to_multiple_of`), a partial RoPE, and more than one layer each side.

transformers' eager attention forces its softmax to f32 even on a `.double()` model, which is 5e-6 of
the reference's OWN rounding at f64 on moonshine-tiny; the test lifts that so the comparison can be
exact (the f64 discipline: a gap the reference's rounding explains proves nothing either way).

The real checkpoint is the gate half: loom vs transformers on 73 LibriSpeech utterances plus jfk is
identical in ids at F32, and the tensor numbers are in loom.cpp's Epic-03.
"""
import json

import pytest
import torch
import torch.nn.functional as F

import loom_exporter  # noqa: F401 -- registers the "loom" backend
from loom_exporter import moonshine_v1_export as V
from loom_exporter.moonshine_export import _MoonshineCrossKvWrapper, causal_mask

transformers = pytest.importorskip("transformers")
from transformers import MoonshineConfig, MoonshineForConditionalGeneration  # noqa: E402


def _config(**overrides):
    config = dict(vocab_size=64, hidden_size=36, intermediate_size=48, encoder_num_hidden_layers=2,
                  decoder_num_hidden_layers=2, encoder_num_attention_heads=4, decoder_num_attention_heads=4,
                  encoder_num_key_value_heads=4, decoder_num_key_value_heads=4, max_position_embeddings=32,
                  partial_rotary_factor=0.9, pad_head_dim_to_multiple_of=8, bos_token_id=1,
                  decoder_start_token_id=1, eos_token_id=2, pad_token_id=2)
    config.update(overrides)
    return MoonshineConfig(**config)


def _tiny(**overrides):
    torch.manual_seed(0)
    model = MoonshineForConditionalGeneration(_config(**overrides))
    model.config._attn_implementation = "eager"
    return model.eval().double()


@pytest.fixture
def f64_softmax(monkeypatch):
    real = F.softmax

    def softmax(x, dim=None, _stacklevel=3, dtype=None):
        return real(x, dim=dim, dtype=None if x.dtype == torch.float64 else dtype)

    monkeypatch.setattr(torch.nn.functional, "softmax", softmax)


def test_the_tiny_config_reaches_the_padded_partial_head():
    attn = _tiny().model.encoder.layers[0].self_attn
    assert attn.head_dim == 9 and attn.head_dim_padding == 7


@pytest.mark.parametrize("n_samples", [V.MIN_SAMPLES, 4000, 16000])
def test_the_encoder_is_transformers(f64_softmax, n_samples):
    model = _tiny()
    torch.manual_seed(1)
    wav = torch.randn(1, n_samples, dtype=torch.float64) * 0.1
    with torch.no_grad():
        ref = model.model.encoder(wav).last_hidden_state
        got = V._MoonshineV1EncoderWrapper(model)(wav)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=0, atol=1e-12)


def test_the_decoder_is_transformers_teacher_forced(f64_softmax):
    model = _tiny()
    torch.manual_seed(2)
    wav = torch.randn(1, 6000, dtype=torch.float64) * 0.1
    ids = torch.tensor([[1, 5, 9, 33, 7, 2]])
    with torch.no_grad():
        ref = model(input_values=wav, decoder_input_ids=ids).logits
        enc = V._MoonshineV1EncoderWrapper(model)(wav)
        cross = _MoonshineCrossKvWrapper(model.model.decoder.layers)(enc)
        n = ids.shape[1]
        got = V._MoonshineV1DecoderWrapper(model)(ids, torch.arange(n).unsqueeze(0),
                                                  causal_mask(n).double(), *cross)
    assert torch.allclose(got, ref, rtol=0, atol=1e-12)


def test_min_samples_is_the_shortest_input_with_one_encoder_row():
    model = _tiny()
    encoder = model.model.encoder
    with torch.no_grad():
        rows = V._MoonshineV1EncoderWrapper(model)(torch.zeros(1, V.MIN_SAMPLES, dtype=torch.float64))
        assert rows.shape[1] == 1
        with pytest.raises(RuntimeError):
            encoder(torch.zeros(1, V.MIN_SAMPLES - 1, dtype=torch.float64))
    assert V.STEM_STRIDE == encoder.conv1.stride[0] * encoder.conv2.stride[0] * encoder.conv3.stride[0]


def test_differing_encoder_and_decoder_heads_are_refused():
    config = _config(hidden_size=48, decoder_num_attention_heads=6, decoder_num_key_value_heads=6)
    with pytest.raises(ValueError, match="head counts differ"):
        V._check_heads(config)
    V._check_heads(_config())


def test_a_built_model_can_no_longer_tell_its_encoder_heads():
    """Why `_check_heads` reads the checkpoint's config and not `model.config`: building the model
    writes the decoder's head count over the encoder's."""
    config = _config(hidden_size=48, decoder_num_attention_heads=6, decoder_num_key_value_heads=6)
    model = MoonshineForConditionalGeneration(config)
    assert model.config.encoder_num_attention_heads == 6


def test_the_end_of_sequence_reaches_the_writer_as_the_list_it_reads():
    """The BPE writer reads `eos_token_ids` and ignores the scalar; v1 ships no `tokenizer_config.json`
    to fall back on, and the first export wrote no end of sequence at all (`</s>` in the transcript)."""
    config = V.ASRMoonshineExportConfig(checkpoint="unused")
    config.eos_token_id, config.sample_rate, config.max_positions = 2, 16000, 194
    kwargs = config.backend_kwargs()
    assert kwargs["eos_token_ids"] == [2] and "eos_token_id" not in kwargs
    assert kwargs["hparams"] == {"sample_rate": 16000, "n_ctx": 194}


def test_the_recognizer_reads_the_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moonshine"}))
    assert V._is_moonshine(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moonshine_streaming"}))
    assert not V._is_moonshine(tmp_path), "the streaming line is a different architecture"
    assert not V._is_moonshine(tmp_path / "missing")


def test_it_is_registered_for_asr():
    from loom_exporter.registry import default_registry

    registry = default_registry()
    names = {(rec.task, rec.name) for entry in registry._entries.values() for rec in entry.recognizers}
    assert ("automatic-speech-recognition", "moonshine") in names
