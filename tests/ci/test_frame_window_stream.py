"""A frame model streamed by windows over its one-pass graph (loom.cpp ADR-120): WakeHuBERT, frame-VAD MarbleNet.

1. **The probe measures the reach a frame really has**, on a causal net and a centred one whose reach is known
   by hand, through a front end of its own.
2. **Windows holding that reach are one call**, at f64, cut the way `frame_window_stream_driver/00_stream.lua`
   cuts them: pushes of any size, a push longer than the graph takes, the stream's own end.
3. **The export carries the stream** beside an unchanged `infer`, with the toy's own numbers; a clip output
   carries none.
"""
import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import loom_exporter  # noqa: F401 -- registers the "loom" backend
from loom_exporter import audio_classification_export as A
from loom_exporter.frame_window_stream import frame_reach, window_frames


class _Centred(nn.Module):
    """hop 4: a strided convolution (kernel 8, pad 2: frame t reads samples [4t - 2, 4t + 6)), then a
    centred kernel-5 one (frames t - 2 .. t + 2). Frame t reads samples [4t - 10, 4t + 14): 10 before
    its own four, 10 after."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(1, 6, 8, stride=4, padding=2)
        self.b = nn.Conv1d(6, 3, 5, padding=2)

    def forward(self, wav):
        return self.b(torch.tanh(self.a(wav.unsqueeze(1)))).transpose(1, 2)


class _Causal(nn.Module):
    """hop 4: frame t reads samples [4t - 12, 4t + 4) -- a left-padded strided convolution (kernel 8),
    then a causal kernel-3 one at dilation 1 (two frames back)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(1, 6, 8, stride=4)
        self.b = nn.Conv1d(6, 3, 3)

    def forward(self, wav):
        x = torch.relu(self.a(F.pad(wav.unsqueeze(1), (4, 0))))
        return self.b(F.pad(x, (2, 0))).transpose(1, 2)


def test_the_probe_measures_the_reach_a_frame_has():
    torch.manual_seed(0)
    assert frame_reach(_Centred().eval(), 4, 400) == (10, 10)
    assert frame_reach(_Causal().eval(), 4, 400) == (12, 0)
    assert window_frames(10, 10, 4) == (3, 3) and window_frames(12, 0, 4) == (3, 0)


def test_a_clip_too_short_for_the_reach_is_doubled():
    torch.manual_seed(0)
    assert frame_reach(_Centred().eval(), 4, 16) == (10, 10)
    with pytest.raises(ValueError, match="not finite"):
        frame_reach(_Centred().eval(), 4, 16, max_samples=24)


def _streamed(model, wav, hop, left, right, pieces, max_piece):
    """`00_stream.lua`'s cut, in Python: each push answers the frames whose look-ahead has arrived, from
    windows of at most `max_piece` kept frames; the finish answers the rest from a window ending at the
    stream's end."""
    rows, e, received = [], 0, 0
    with torch.no_grad():
        for n in pieces:
            received += n
            ready = received // hop - right
            while ready > e:
                f = min(ready, e + max_piece)
                a = max(0, e - left)
                rows.append(model(wav[:, hop * a:hop * (f + right)])[0, e - a:f - a])
                e = f
        if right:
            a = max(0, e - left)
            rows.append(model(wav[:, hop * a:received])[0, e - a:])
    return torch.cat(rows)


@pytest.mark.parametrize("cls", [_Centred, _Causal])
@pytest.mark.parametrize("pieces", [[1] * 403, [37] * 10 + [33], [403], [100, 3, 300]])
def test_windows_holding_the_reach_are_one_call(cls, pieces):
    """403 samples, so the clip ends inside a frame; `max_piece` 7 makes a long push several windows."""
    torch.manual_seed(1)
    model = cls().eval()
    left, right = window_frames(*frame_reach(model, 4, 400), 4)
    model = model.double()
    wav = torch.randn(1, 403, dtype=torch.float64)
    with torch.no_grad():
        whole = model(wav)[0]
    got = _streamed(model, wav, 4, left, right, pieces, max_piece=7)
    assert got.shape == whole.shape
    assert (got - whole).abs().max().item() < 1e-12


def test_one_frame_less_context_is_not_one_call():
    """The margins are tight: a frame short on either side moves rows (the sabotage the engine gate runs
    by patching a constant)."""
    torch.manual_seed(1)
    model = _Centred().eval()
    left, right = window_frames(*frame_reach(model, 4, 400), 4)
    model = model.double()
    wav = torch.randn(1, 403, dtype=torch.float64)
    with torch.no_grad():
        whole = model(wav)[0]
    for l, r in [(left - 1, right), (left, right - 1)]:
        got = _streamed(model, wav, 4, l, r, [37] * 10 + [33], max_piece=7)
        assert (got - whole).abs().max().item() > 1e-3


def _toy_wakehubert(tmp_path):
    from test_audio_classification_export import _toy_wakehubert as make

    return make(tmp_path)


def test_the_export_carries_the_stream_beside_an_unchanged_infer(tmp_path):
    """The toy WakeHuBERT (hop 64: a causal 64-tap DFT at hop 32, then a causal kernel-4 stride-2 stem) reads
    96 samples before its frame and none after: two frames of context, no look-ahead."""
    from gguf import GGUFReader

    from loom_exporter.main_export import main_export

    out = tmp_path / "toy.gguf"
    main_export(str(_toy_wakehubert(tmp_path)), str(out))
    script = GGUFReader(str(out)).fields["model.driver_script"].contents()
    for fn in ("stream_open", "stream_feed", "stream_finish", "infer"):
        assert f"function {fn}(" in script, fn
    for line in ("local HOP = 64", "local LEFT_FRAMES = 2", "local RIGHT_FRAMES = 0", "local ROW_WIDTH = 16",
                 "local TAKES_LENGTH = 0", "local MIN_SAMPLES = 64"):
        assert line + "\n" in script, line
    assert "loom.run_subgraph('main_topology', {n_samples = #waveform, n_past = 0}, {waveform = waveform})" \
        in script


def test_a_clip_output_has_no_stream():
    config = A.AudioClassificationExportConfig.__new__(A.AudioClassificationExportConfig)
    config.output, config.stream = A.AudioOutput.EMBEDDING, None
    config._probe_stream(None, (), [])
    assert config.stream is None
