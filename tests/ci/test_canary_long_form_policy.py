"""`canary_export.long_form_policy`: the numbers loom.cpp's `transcribe` reads to decode audio past
Canary's 40 s training ceiling the way NeMo does (loom.cpp asr_long_form.h).

The window bounds are READ off NeMo's own `_find_optimal_chunk_size` signature rather than restated, so
the test stands NeMo in with a module whose defaults differ from Canary's: a policy that came out as
30/40/1.0 here would be restating, not reading. CI has no NeMo; the stand-in is the same "fake module,
real check" as test_registry.py.

What the numbers DO is graded in loom.cpp (tests/ci/test_asr_long_form.cpp, NeMo-generated vectors) and
end to end against NeMo's chunked transcribe, where every window's token ids and the stitched
transcript came out identical on 79 s and 161 s of LibriSpeech.
"""
import sys
import types
from unittest import mock

import pytest

from loom_exporter.canary_export import long_form_policy


def _fake_nemo(min_sec=30, max_sec=40, overlap_sec=1.0):
    class PromptedAudioToTextLhotseDataset:
        def _find_optimal_chunk_size(self, total_len, min_sec=min_sec, max_sec=max_sec,
                                     sample_rate=16000, overlap_sec=overlap_sec):
            raise AssertionError("only the signature is read")

    module = types.ModuleType("nemo.collections.asr.data.audio_to_text_lhotse_prompted")
    module.PromptedAudioToTextLhotseDataset = PromptedAudioToTextLhotseDataset
    return mock.patch.dict(sys.modules, {module.__name__: module})


def test_canary_policy_is_nemos():
    with _fake_nemo():
        policy = long_form_policy(16000, subsampling_factor=8, window_stride=0.01)
    assert policy == {
        "window_max_samples": 640000,
        "window_min_samples": 480000,
        "window_search_step_samples": 16000,
        "window_overlap_samples": 16000,
        # `merge_parallel_chunks`: delay = int(1 / (8 / 100)) = 12 encoder frames per second.
        "merge_search_tokens": 24,
        "merge_head_tokens": 7,
    }


def test_window_bounds_follow_nemos_signature_not_a_restatement():
    with _fake_nemo(min_sec=20, max_sec=25, overlap_sec=2.0):
        policy = long_form_policy(16000, subsampling_factor=8, window_stride=0.01)
    assert policy["window_min_samples"] == 320000
    assert policy["window_max_samples"] == 400000
    assert policy["window_overlap_samples"] == 32000


def test_merge_widths_follow_the_encoders_frame_rate():
    with _fake_nemo():
        policy = long_form_policy(16000, subsampling_factor=4, window_stride=0.01)
    assert (policy["merge_search_tokens"], policy["merge_head_tokens"]) == (50, 15)


def test_a_stride_the_merge_formula_does_not_assume_is_refused():
    with _fake_nemo(), pytest.raises(ValueError, match="10 ms"):
        long_form_policy(16000, subsampling_factor=8, window_stride=0.02)
