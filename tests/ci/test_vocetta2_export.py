"""Vocetta2 -- the hermetic half.

The export builds the checkpoint's own modules from the `microtts/models.py` that ships beside its
weights, which CI does not have, so what is tested here is everything that does not need it: the
recognizer, the dimensions read off a state_dict's shapes and their cross-check against `config.json`,
the tilt's fold into the acoustic output conv, and the declarations.

The real checkpoints are the gate half (2026-10-21, released rc18 engine, six sentences each through
upstream's own ids): every waveform the length of upstream's, max |diff| 1.3e-5 (276k), 3.1e-5 (500k) and
1.8e-4 (1m) -- the 1m's worst is upstream's own float32 spread (its f32 run is 1.5e-4 from its f64 run;
loom is 3.3e-5 from the f64 run).
"""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from loom_exporter import vocetta2_export as V


def _state(vocab=62, dur_h=12, dur_depth=3, ac_h=31, token_depth=3, frame_depth=2, dim=62, pw=186,
           blocks=4, noise=4, mels=100, k=5):
    """The state_dict shapes upstream's three modules have (`microtts/models.py`), with zeros."""
    z = torch.zeros
    duration = {"embed.weight": z(vocab, dur_h), "proj.weight": z(dur_h, dur_h + 3, 1), "out.weight": z(1, dur_h, 1)}
    for b in range(dur_depth):
        duration[f"blocks.{b}.0.weight"] = z(dur_h, dur_h, k)
        duration[f"blocks.{b}.1.weight"] = z(dur_h, dur_h, k)
        duration[f"scales.{b}"] = z(())
    acoustic = {"embed.weight": z(vocab, ac_h), "tproj.weight": z(ac_h, ac_h + 2, 1),
                "fproj.weight": z(ac_h, ac_h + 3, 1), "out.weight": z(mels, ac_h, 1)}
    for prefix, depth in (("token_blocks", token_depth), ("frame_blocks", frame_depth)):
        for b in range(depth):
            acoustic[f"{prefix}.{b}.0.weight"] = z(ac_h, ac_h, k)
            acoustic[f"{prefix}.{b}.1.weight"] = z(ac_h, ac_h, k)
    decoder = {"embed.weight": z(dim, mels, 7), "noise_adapter.weight": z(dim, noise, 7),
               "head.weight": z(1026, dim)}
    for b in range(blocks):
        decoder[f"blocks.{b}.dw.weight"] = z(dim, 1, 7)
        decoder[f"blocks.{b}.pw0.weight"] = z(pw, dim)
    return {"duration": duration, "acoustic": acoustic, "decoder": decoder}


def test_the_dimensions_are_read_off_the_weights():
    dims = V._dims(_state(ac_h=72, frame_depth=5, dim=130, pw=337))
    assert dims == {
        "vocab": 62, "duration_hidden": 12, "duration_depth": 3, "duration_kernel": 5,
        "acoustic_vocab": 62, "acoustic_hidden": 72, "acoustic_token_depth": 3, "acoustic_depth": 5,
        "acoustic_kernel": 5, "n_mels": 100, "decoder_dim": 130, "decoder_embed_kernel": 7,
        "decoder_pw": 337, "decoder_blocks": 4, "decoder_dw_kernel": 7, "noise_channels": 4, "bins": 513,
    }


def test_a_config_that_agrees_with_the_weights_passes():
    """The 1m's `architecture_dims`, verbatim; the 276k and 500k state none."""
    dims = V._dims(_state(ac_h=72, frame_depth=5, dim=130, pw=337))
    V.check_config(dims, {"model_type": "vocetta2", "sample_rate": 24000, "architecture_dims": {
        "duration_hidden": 12, "duration_depth": 3, "acoustic_hidden": 72, "acoustic_depth": 5,
        "acoustic_token_depth": 3, "decoder_dim": 130, "decoder_pw": 337, "decoder_blocks": 4,
        "noise_channels": 4, "n_mels": 100}})
    V.check_config(V._dims(_state()), {"model_type": "vocetta2", "sample_rate": 24000})


def test_a_config_that_disagrees_with_the_weights_is_refused():
    with pytest.raises(ValueError, match="decoder_pw=338, the weights 186"):
        V.check_config(V._dims(_state()), {"architecture_dims": {"decoder_pw": 338}})


def test_another_rate_is_refused():
    with pytest.raises(ValueError, match="22050 Hz"):
        V.check_config(V._dims(_state()), {"sample_rate": 22050})


def test_the_tilt_is_folded_into_the_mel_projection():
    """Upstream adds `tilt.npy` to the acoustic net's mel; the export puts it in the output conv's bias,
    and leaves the module it was handed alone."""
    torch.manual_seed(0)
    out = nn.Conv1d(8, 100, 1)
    before = out.bias.detach().clone()
    acoustic = SimpleNamespace(embed=nn.Embedding(62, 8), tproj=nn.Conv1d(10, 8, 1), token_blocks=nn.ModuleList(),
                               token_scales=nn.ParameterList(), fproj=nn.Conv1d(11, 8, 1),
                               frame_blocks=nn.ModuleList(), frame_scales=nn.ParameterList(), out=out)
    decoder = nn.Module()
    decoder.bins, decoder.n_fft, decoder.hop = 513, 1024, 256
    tilt = torch.linspace(-1.0, 1.0, 100)
    wrapper = V._SynthWrapper(acoustic, decoder, tilt)
    x = torch.randn(1, 8, 9)
    assert torch.allclose(wrapper.out(x), out(x) + tilt[None, :, None], atol=1e-6)
    assert torch.equal(out.bias.detach(), before)
    assert torch.equal(V._SynthWrapper(acoustic, decoder, None).out.bias.detach(), before)


def test_the_recognizer_reads_the_model_type(tmp_path):
    for name in V.WEIGHT_FILES:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "vocetta2"}))
    assert V.is_vocetta2(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other"}))
    assert not V.is_vocetta2(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "vocetta2"}))
    (tmp_path / "decoder.pt").unlink()
    assert not V.is_vocetta2(tmp_path)


def test_a_missing_module_file_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "vocetta2_upstream_models", raising=False)
    with pytest.raises(FileNotFoundError, match="microtts/models.py"):
        V.load_upstream_models(tmp_path)


def test_it_declares_misakis_conventions_and_24_khz(tmp_path):
    """Distilled on misaki's normalised IPA, as sanoTTS's nano voices were; the text door folds a G2P's
    output to it (loom.cpp ADR-071)."""
    config = V.TTSVocetta2ExportConfig(model_dir=str(tmp_path / "missing"))
    assert config.hparams() == {"sample_rate": 24000, "tts.phoneme_style": "misaki"}


def test_it_is_registered_for_tts():
    from loom_exporter.registry import default_registry

    assert default_registry().get("text-to-speech", "vocetta2").name == "vocetta2"
