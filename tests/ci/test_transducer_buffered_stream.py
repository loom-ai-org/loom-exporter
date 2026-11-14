"""NeMo's buffered transducer stream (loom.cpp ADR-110, ADR-122): one fragment, `transducer_driver/
11_buffered_stream.lua`, for Parakeet TDT/RNN-T and the unified model, beside the shared live decode
(`10_stream_decode.lua`).

What runs is checked by loom.cpp's `test_e2e_nemo_buffered_stream` against NeMo's streaming script; here, that
each leaf ships both fragments top level, binds every constant they read, declares the step call the fragment
names by a constant (with the inputs that leaf's step takes), and carries the script's defaults.
"""
import re

import pytest

from loom_exporter.driver_components import ComputedCall, ExportConstants, LuaFragment
from loom_exporter.nemotron_asr_export import ASRNemotronExportConfig
from loom_exporter.nemotron_unified_export import ASRNemotronUnifiedExportConfig
from loom_exporter.parakeet_export import ASRParakeetExportConfig


def _parakeet(durations=()):
    spec = ASRParakeetExportConfig(checkpoint="/nonexistent")
    # What `phases()` reads off a checkpoint: parakeet-tdt-0.6b-v3's duration set, or none for RNN-T.
    spec.durations = tuple(durations)
    return spec


def _live(components):
    fragments = {c.path.name: c for c in components if isinstance(c, LuaFragment)}
    (constants,) = [c for c in components if isinstance(c, ExportConstants) and c.top_level]
    return fragments, constants.values


@pytest.mark.parametrize("spec", [_parakeet(), _parakeet((0, 1, 2, 3, 4)), ASRNemotronUnifiedExportConfig(
    checkpoint="/nonexistent")], ids=["parakeet-rnnt", "parakeet-tdt", "unified"])
def test_every_constant_the_live_fragments_read_is_bound(spec):
    fragments, values = _live(spec.driver_components())
    for name in ("10_stream_decode.lua", "11_buffered_stream.lua"):
        assert fragments[name].top_level
        read = set(re.findall(r"\bS_[A-Z_]+\b", fragments[name].path.read_text()))
        assert read <= set(values), (name, read - set(values))
    assert "stream_feed" in fragments["11_buffered_stream.lua"].defines
    # The decode's durations are the checkpoint's: TDT jumps by them, RNN-T has none.
    assert values["S_DURATIONS"] == list(spec.durations) and values["S_N_DURATIONS"] == len(spec.durations)


def test_parakeet_steps_through_its_one_pass_encoder_at_the_scripts_defaults():
    fragments, values = _live(_parakeet((0, 1, 2, 3, 4)).driver_components())
    assert (values["S_STEP_PHASE"], values["S_STEP_PROMPT"], values["S_STEP_MASK"]) == ("encoder", 0, 0)
    # `speech_to_text_streaming_infer_rnnt.py`'s own: 10 s left, 2 s chunk, 2 s right.
    assert (values["S_DEFAULT_LEFT_SECONDS"], values["S_DEFAULT_CHUNK_SECONDS"],
            values["S_DEFAULT_RIGHT_SECONDS"]) == (10.0, 2.0, 2.0)
    (call,) = fragments["11_buffered_stream.lua"].drives
    assert isinstance(call, ComputedCall)
    assert call.topologies == ("encoder",) and call.inputs == ("waveform", "length")
    assert call.written == "S_STEP_PHASE"
    # A FastConformer encoder frame is 8 hops of 10 ms at 16 kHz, and the buffer is bounded by what the encoder
    # phase was traced for.
    assert (values["S_FRAME_SAMPLES"], values["S_FEATURES_PER_SEC"], values["S_SUBSAMPLING"]) == (1280, 100.0, 8)
    assert values["S_MAX_BUFFER_SAMPLES"] == 20 * 16000


def test_the_unified_model_steps_through_its_masked_step_with_its_prompt():
    fragments, values = _live(ASRNemotronUnifiedExportConfig(checkpoint="/nonexistent").driver_components())
    assert (values["S_STEP_PHASE"], values["S_STEP_PROMPT"], values["S_STEP_MASK"]) == ("encoder_step", 1, 1)
    assert (values["S_DEFAULT_LEFT_SECONDS"], values["S_DEFAULT_CHUNK_SECONDS"],
            values["S_DEFAULT_RIGHT_SECONDS"]) == (5.6, 0.56, 0.56)
    assert values["S_BLANK_ID"] == 4095 and values["S_STREAM_NAME"] == "nemotron-unified-asr"
    (call,) = fragments["11_buffered_stream.lua"].drives
    assert call.topologies == ("encoder_step",)
    assert call.inputs == ("waveform", "length", "prompt", "att_left", "att_chunk", "att_right")


def test_the_cache_aware_stream_shares_the_decode_but_not_the_buffer():
    # Nemotron 3.5 streams its cache-aware encoder (ADR-109), not a buffer: the decode only.
    fragments, values = _live(ASRNemotronExportConfig(checkpoint="/nonexistent").driver_components())
    assert "10_stream_decode.lua" in fragments and "11_buffered_stream.lua" not in fragments
    assert values["S_N_DURATIONS"] == 0


def test_a_tdt_jump_past_a_calls_rows_is_carried_to_the_next():
    text = (_live(_parakeet((0, 1, 2, 3, 4)).driver_components())[0]["10_stream_decode.lua"]).path.read_text()
    # NeMo's `time_jumps = time_indices - encoder_output_length`, carried as `prev_batched_state`; dropping it
    # restarts each chunk at its row 0 and re-reads frames a duration had skipped.
    assert "loom.set_state('time_jump', t - n_frames)" in text
    assert "local t = loom.state('time_jump')" in text
