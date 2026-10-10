"""`read_sampling_defaults`: a knob `generation_config.json` leaves out gets the value `generate()`
samples with, not "no truncation" (loom.cpp hub, Exporter: the `top_k` item; Epic-03 §2 Family 14)."""
import json

import pytest

from loom_exporter.bpe_tokenizer_export import _GENERATE_DEFAULTS, read_sampling_defaults


def _checkpoint(tmp_path, cfg):
    (tmp_path / "generation_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_a_missing_top_k_is_generates_50_not_0(tmp_path):
    # MusicGen's, csm-1b's and parler-tts' shape: do_sample with no top_k at all.
    got = read_sampling_defaults(_checkpoint(tmp_path, {"do_sample": True, "temperature": 0.9}))
    assert got == {"temperature": 0.9, "top_k": 50, "top_p": 1.0}


def test_declared_values_are_kept_including_an_explicit_0(tmp_path):
    got = read_sampling_defaults(_checkpoint(tmp_path, {"do_sample": True, "top_k": 0, "top_p": 0.9}))
    assert got == {"temperature": 1.0, "top_k": 0, "top_p": 0.9}
    got = read_sampling_defaults(_checkpoint(tmp_path, {"do_sample": True, "top_k": 20,
                                                        "top_p": 0.95, "temperature": 0.6}))
    assert got == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}


def test_no_do_sample_or_no_file_is_greedy(tmp_path):
    greedy = {"temperature": 0.0, "top_k": 0, "top_p": 1.0}
    assert read_sampling_defaults(tmp_path) == greedy
    assert read_sampling_defaults(_checkpoint(tmp_path, {"top_k": 50})) == greedy


def test_an_explicit_null_is_refused(tmp_path):
    with pytest.raises(ValueError, match="top_k"):
        read_sampling_defaults(_checkpoint(tmp_path, {"do_sample": True, "top_k": None}))


def test_sampling_defaults_match_transformers():
    """The table is what the INSTALLED transformers' `generate()` fills in. Both export venvs run
    this file: 4.x keeps the defaults on `GenerationConfig`; 5.x keeps `None` there and fills them
    from `_get_default_generation_params` inside `generate()`."""
    from transformers import GenerationConfig

    params = getattr(GenerationConfig, "_get_default_generation_params", None)
    if params is not None:
        source = params()
    else:
        config = GenerationConfig(do_sample=True)
        source = {k: getattr(config, k) for k in _GENERATE_DEFAULTS}
    assert {k: source[k] for k in _GENERATE_DEFAULTS} == _GENERATE_DEFAULTS
