"""NeMo CTC's live path (loom.cpp ADR-123): FrameBatchASR's loop beside the unchanged one pass.

What runs is checked by loom.cpp's `test_e2e_nemo_ctc_buffered_stream` against NeMo's own FrameBatchASR; here,
that the leaf ships the fragment with every constant it reads bound, `infer` decodes the renamed one pass, the
buffer graph normalises exactly as `FeatureFrameBufferer` does, the raw-mel graph leaves the featurizer as it
found it, and the fill value follows NeMo's own test.
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from loom_exporter.driver_components import CtcGreedyEpilogue, ExportConstants, LuaFragment, SubgraphCallComponent
from loom_exporter.nemo_ctc_export import (
    MEL_MARGIN, STREAM_SECONDS, ASRNemoCtcExportConfig, CtcBuffer, RawMel, _stream_facts,
)


def _components():
    spec = ASRNemoCtcExportConfig(checkpoint="/nonexistent", architecture="conformer-ctc")
    return spec, spec.driver_components()


def test_the_driver_carries_a_live_path_beside_the_one_pass():
    _, components = _components()
    live = [c for c in components if isinstance(c, LuaFragment) and "stream_feed" in c.defines]
    assert len(live) == 1 and live[0].top_level
    (constants,) = [c for c in components if isinstance(c, ExportConstants) and c.top_level]
    read = set(re.findall(r"\bS_[A-Z_]+\b", live[0].path.read_text()))
    assert read <= set(constants.values), read - set(constants.values)
    values = constants.values
    # The script's defaults, and FrameBatchASR's numbers for a 16 kHz, 10 ms, 4x model.
    assert (values["S_DEFAULT_CHUNK_SECONDS"], values["S_DEFAULT_TOTAL_SECONDS"]) == STREAM_SECONDS == (1.6, 4.0)
    assert (values["S_HOP"], values["S_FEATURE_STRIDE"], values["S_MEL_MARGIN"]) == (160, 0.01, MEL_MARGIN)
    # infer: the one pass, retained, decoded greedily -- under its new name.
    (call,) = [c for c in components if isinstance(c, SubgraphCallComponent)]
    (epilogue,) = [c for c in components if isinstance(c, CtcGreedyEpilogue)]
    assert call.topology == "ctc" and call.retain and epilogue.retained_module == "ctc"
    assert epilogue.blank_id == values["S_BLANK_ID"]


def test_the_stream_calls_both_new_graphs_and_decodes_nemos_slice():
    _, components = _components()
    (live,) = [c for c in components if isinstance(c, LuaFragment) and "stream_feed" in c.defines]
    text = live.path.read_text()
    assert "'raw_mel'" in text and "'ctc_buffer'" in text
    # `decoded[len(decoded) - 1 - delay : ... + tokens_per_chunk]` and `greedy_merge`'s rule.
    assert "local first = #ids - 1 - loom.state('delay')" in text
    assert "(p ~= prev or prev == S_BLANK_ID) and p ~= S_BLANK_ID" in text
    # The script's pad: `int(delay * model_stride_in_secs * sample_rate)`, in that order.
    assert "math.floor(delay * stride_s * S_SAMPLE_RATE)" in text


class _Identity(nn.Module):
    """A model whose forward hands back the processed signal it was given, so the test sees the buffer."""

    def forward(self, processed_signal, processed_signal_length):
        return processed_signal, processed_signal_length, None


@pytest.mark.parametrize("frames", [400, 37])
def test_the_buffer_is_normalised_as_feature_frame_bufferer_does(frames):
    rng = np.random.default_rng(0)
    buffer = rng.normal(size=(frames, 80)).astype(np.float32) * 3 - 5
    got = CtcBuffer(_Identity())(torch.from_numpy(buffer)[None], torch.tensor([frames]))[0].numpy()
    # `get_norm_consts_per_frame` + `normalize_frame_buffers`, on the [n_mels, T] buffer.
    x = buffer.T
    want = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)
    assert np.abs(got - want).max() < 1e-5


def test_raw_mel_turns_normalisation_off_only_while_it_runs():
    seen = []

    class Featurizer:
        normalize = "per_feature"

    class Pre(nn.Module):
        featurizer = Featurizer()

        def forward(self, input_signal, length):
            seen.append(self.featurizer.normalize)
            return input_signal[:, None, :].repeat(1, 3, 1), length

    pre = Pre()
    out = RawMel(SimpleNamespace(preprocessor=pre))(torch.zeros(1, 5), torch.tensor([5]))
    assert seen == [None] and pre.featurizer.normalize == "per_feature" and out.shape == (1, 5, 3)


def _fake(log=None, n_fft=512):
    pre = SimpleNamespace(featurizer=SimpleNamespace(n_fft=n_fft))
    if log is not None:
        pre.log = log
    cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01, features=80))
    return SimpleNamespace(cfg=cfg, preprocessor=pre, encoder=SimpleNamespace(subsampling_factor=8))


def test_the_fill_value_is_nemos_own_test():
    # FeatureFrameBufferer asks the PREPROCESSOR for `log`; NeMo's has none, so the fill is 0.0 in practice.
    assert _stream_facts(_fake(), 16000)["BUFFER_FILL"] == 0.0
    assert _stream_facts(_fake(log=True), 16000)["BUFFER_FILL"] == -16.635
    assert _stream_facts(_fake(), 16000)["MODEL_STRIDE"] == 8


def test_a_margin_short_of_a_frames_reach_is_refused():
    with pytest.raises(ValueError, match="margin"):
        _stream_facts(_fake(n_fft=2048), 16000)
