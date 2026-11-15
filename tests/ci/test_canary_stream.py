"""Canary's live door (loom.cpp ADR-124): NeMo's streaming AED loop, wait-k and AlignAtt, as a top-level fragment.

What runs is graded in loom.cpp (`test_e2e_canary_stream`, against NeMo's own streaming loop: every step's kept
context, the first forward's logits and AlignAtt attention, the ids). Here: the decoding numbers are READ off
NeMo's defaults rather than restated -- a stand-in NeMo with other defaults must move them -- and every constant
the fragment reads is bound.
"""
import re
import sys
import types
from dataclasses import make_dataclass
from unittest import mock

import pytest

from loom_exporter.canary_export import STREAM_POLICY_DEFAULTS, STREAM_SECONDS, ASRCanaryExportConfig, stream_policy
from loom_exporter.driver_components import ExportConstants, LuaFragment


def _fake_nemo(**cfg_overrides):
    """NeMo's two dataclasses with defaults that are none of NeMo's, so a restated number cannot pass."""
    config = dict(streaming_policy="alignatt", alignatt_thr=5.0, waitk_lagging=3, exclude_sink_frames=4,
                  xatt_scores_layer=-3, max_tokens_per_alignatt_step=30, max_generation_length=300,
                  use_avgpool_for_alignatt=False, hallucinations_detector=False)
    config.update(cfg_overrides)
    decoding = types.ModuleType("nemo.collections.asr.parts.submodules.multitask_decoding")
    decoding.AEDStreamingDecodingConfig = make_dataclass(
        "AEDStreamingDecodingConfig", [(k, type(v), v) for k, v in config.items()])
    streaming = types.ModuleType("nemo.collections.asr.parts.submodules.aed_decoding.aed_batched_streaming")
    streaming.AEDStreamingState = make_dataclass(
        "AEDStreamingState", [("max_generation_length", int, 128), ("max_tokens_per_one_second", int, 7)])
    return mock.patch.dict(sys.modules, {decoding.__name__: decoding, streaming.__name__: streaming})


def test_the_policy_is_read_off_nemo():
    with _fake_nemo():
        policy = stream_policy()
    assert policy == {
        "default_policy": 1, "waitk_lagging": 3, "alignatt_thr": 5.0, "exclude_sink_frames": 4,
        "xatt_scores_layer": -3,
        # Wait-k is capped by the CONFIG's max_generation_length, AlignAtt by the STATE's.
        "waitk_max_generation": 300, "alignatt_max_generation": 128,
        "tokens_per_second": 7, "hallucinations": 0,
    }


def test_a_smoothed_alignatt_is_refused():
    with _fake_nemo(use_avgpool_for_alignatt=True), pytest.raises(ValueError, match="smooths"):
        stream_policy()


def test_every_constant_the_live_fragment_reads_is_bound():
    spec = ASRCanaryExportConfig(checkpoint="/nonexistent")
    components = spec.driver_components()
    (live,) = [c for c in components if isinstance(c, LuaFragment) and "stream_feed" in c.defines]
    assert live.top_level
    (constants,) = [c for c in components if isinstance(c, ExportConstants) and c.top_level]
    text = live.path.read_text()
    read = set(re.findall(r"\bS_[A-Z_]+\b", text))
    assert read <= set(constants.values), read - set(constants.values)
    values = constants.values
    # Without a checkpoint: the defaults NeMo 2.6.2 / 3.0 have, and the script's 10/2/2 buffer.
    assert {k: values["S_" + k.upper()] for k in ("waitk_lagging", "alignatt_thr", "exclude_sink_frames")} == {
        "waitk_lagging": 2, "alignatt_thr": 8.0, "exclude_sink_frames": 8}
    assert (values["S_DEFAULT_LEFT_SECONDS"], values["S_DEFAULT_CHUNK_SECONDS"],
            values["S_DEFAULT_RIGHT_SECONDS"]) == STREAM_SECONDS == (10.0, 2.0, 2.0)
    assert STREAM_POLICY_DEFAULTS["xatt_scores_layer"] == -2
    # AlignAtt reads the decoder's second output, and the decoder sees the rows the batch context allows.
    assert "loom.get_output('decoder', 2)" in text
    assert "{xa = {from = 'encoder', row = 0, rows = k}}" in text
