"""A wav2vec 2.0-style CTC model streamed in the transformers pipeline's strided chunks (loom.cpp ADR-126).

What runs is checked by loom.cpp's `test_e2e_ctc_chunked_stream` against the pipeline itself; here:

1. **The driver's sizes and cut are the pipeline's**: `ctc_chunk_stream_driver/00_stream.lua`'s arithmetic,
   mirrored line for line below, against `chunk_iter` and `rescale_stride` from transformers, over clip
   lengths and sizes that land on every rounding edge (half-to-even included).
2. **The export carries the stream** beside the unchanged one pass, every constant it reads bound with the
   checkpoint's own numbers.
"""
import math
import re

import numpy as np
import pytest

from loom_exporter.ctc_chunk_stream import DEFAULT_CHUNK_SECONDS, DRIVER_DIR, stream_components
from loom_exporter.driver_components import ExportConstants, LuaFragment

from test_ctc_asr_export import _export, _tiny_checkpoint


def _py_round(x: float) -> int:
    """The fragment's `ctc_py_round`."""
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def _lua_cut(n, chunk_s, left_s, right_s, rate, align, pieces):
    """`00_stream.lua`, pushes of `pieces` samples then the finish: per chunk run, (start, samples, first kept
    frame, end of the kept frames before the clamp)."""
    size = lambda s: _py_round(s * rate / align) * align  # noqa: E731
    chunk_len, left, right = size(chunk_s), size(left_s), size(right_s)
    assert chunk_len > left + right
    step = chunk_len - left - right
    runs = []

    def run(s, e, last):
        m = e - s
        token_n = _py_round(m * (1 / align))
        l_s = 0 if s == 0 else left
        r_s = 0 if last else right
        runs.append((s, m, _py_round(l_s / m * token_n), token_n - _py_round(r_s / m * token_n)))

    s, received = 0, 0
    for p in pieces:
        received += p
        while received > s + chunk_len:
            run(s, s + chunk_len, False)
            s += step
    while s < n:
        last = s + chunk_len >= n
        if min(s + chunk_len, n) - s > (0 if s == 0 else left):
            run(s, min(s + chunk_len, n), last)
        if last:
            break
        s += step
    return runs


def _pipeline_cut(n, chunk_s, left_s, right_s, rate, align):
    """The pipeline's own preprocess sizes, `chunk_iter` and `rescale_stride`."""
    from transformers.pipelines.automatic_speech_recognition import chunk_iter, rescale_stride

    class _Extractor:
        sampling_rate = rate

        def __call__(self, chunk, **_):
            return {"input_values": chunk}

    chunk_len = int(round(chunk_s * rate / align) * align)
    left = int(round(left_s * rate / align) * align)
    right = int(round(right_s * rate / align) * align)
    runs, start = [], 0
    step = chunk_len - left - right
    for item in chunk_iter(np.zeros(n, dtype=np.float32), _Extractor(), chunk_len, left, right):
        m, l_s, r_s = item["stride"]
        token_n, l_f, r_f = rescale_stride([(m, l_s, r_s)], 1 / align)[0]
        runs.append((start, m, l_f, token_n - r_f))
        start += step
    return runs


@pytest.mark.parametrize("align", [320, 160])
@pytest.mark.parametrize("sizes", [(4.0, None, None), (10.0, None, None), (2.0, 0.5, 0.5), (6.0, 1.0, 0.5),
                                   (3.33, 0.41, 0.73), (1.0, 0.1, 0.1)])
def test_the_drivers_cut_is_the_pipelines(align, sizes):
    pytest.importorskip("transformers")
    rate = 16000
    chunk_s, left_s, right_s = sizes
    left_s = chunk_s / 6 if left_s is None else left_s
    right_s = chunk_s / 6 if right_s is None else right_s
    rng = np.random.default_rng(0)
    chunk_len = _py_round(chunk_s * rate / align) * align
    # Short clips, clips ending on and either side of a chunk's end, and random ones.
    lengths = {1601, 2000, chunk_len - 1, chunk_len, chunk_len + 1, 3 * chunk_len, 3 * chunk_len + 7,
               *rng.integers(1601, 40 * rate, 20).tolist()}
    for n in sorted(lengths):
        want = _pipeline_cut(n, chunk_s, left_s, right_s, rate, align)
        tenth = rate // 10
        for pieces in ([n], [tenth] * (n // tenth) + [n % tenth], [7919] * (n // 7919) + [n % 7919]):
            assert _lua_cut(n, chunk_s, left_s, right_s, rate, align, [p for p in pieces if p]) == want, (n, pieces)


def test_python_round_is_half_to_even():
    # The pipeline's sizes go through Python's round; 0.5 frames either way is where a floor or a half-up
    # rounding moves a chunk by a frame.
    for x in (0.5, 1.5, 2.5, 3.49999, 3.5, 32.5, 33.5, 7.0):
        assert _py_round(x) == round(x), x


def test_every_constant_the_fragment_reads_is_bound():
    constants, fragment = stream_components(rate=16000, align=320, blank_id=0, min_samples=1600,
                                            max_samples=480000)
    assert isinstance(constants, ExportConstants) and constants.top_level
    assert isinstance(fragment, LuaFragment) and fragment.top_level
    read = set(re.findall(r"\bCTC_[A-Z_]+\b", (DRIVER_DIR / "00_stream.lua").read_text()))
    assert read == set(constants.values), read ^ set(constants.values)
    assert constants.values["CTC_DEFAULT_CHUNK_SECONDS"] == DEFAULT_CHUNK_SECONDS


def test_the_export_carries_the_stream_beside_the_one_pass(tmp_path):
    pytest.importorskip("coremltools")
    exported = _export(_tiny_checkpoint(tmp_path), tmp_path / "tiny.gguf")
    script = exported["driver"]
    for fn in ("stream_open", "stream_feed", "stream_finish", "infer"):
        assert f"function {fn}(" in script, fn
    # The tiny stem's stride (5 x 2) is the alignment; the blank is the tokenizer's pad.
    for line in ("local CTC_RATE = 16000", "local CTC_ALIGN = 10", "local CTC_BLANK = 0",
                 "local CTC_MIN_SAMPLES = 1600", "local CTC_MAX_SAMPLES = 480000"):
        assert line + "\n" in script, line
    # The one pass, unchanged.
    assert "loom.argmax_rows('main_topology')" in script and "_ctc_prev = 0" in script
