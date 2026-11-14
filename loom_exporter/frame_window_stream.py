"""A frame model whose frame t reads a FINITE span of samples, streamed by windows over its one-pass graph
(loom.cpp ADR-120): WakeHuBERT (strictly causal) and frame-VAD MarbleNet (centred convolutions).

The one-pass export is unchanged -- the same graph, weights and `infer` -- and the stream's three functions
sit beside `infer` at top level (`frame_window_stream_driver/00_stream.lua`). A push's frames `[e, f)` are
the one-pass graph over the samples from `hop * (e - LEFT_FRAMES)` to `hop * (f + RIGHT_FRAMES)`, their rows
kept: every sample those frames read is inside the window, so every position of every layer they read is
computed from the same inputs (a window's own zero padding and length mask stand only for positions
outside it).

The reach is MEASURED on the traced wrapper, not walked layer by layer: a NaN in one sample marks every
output frame that reads it, whatever the weights, through the front end (pre-emphasis, a centred or
left-padded STFT, the mel) and every convolution alike. A value probe would under-count: the far taps of
a receptive field move a frame by ~1e-13 on real audio.
"""
from pathlib import Path
from typing import Callable, List, Tuple

import torch

DRIVER_DIR = Path(__file__).resolve().parent / "frame_window_stream_driver"


def frame_reach(run: Callable[[torch.Tensor], torch.Tensor], hop: int, n_samples: int,
                max_samples: int = 0) -> Tuple[int, int]:
    """`(left, right)` in samples: frame t, which covers samples `[hop t, hop (t + 1))`, reads samples
    `hop t - left` to `hop (t + 1) - 1 + right` and nothing outside them. `right` is 0 for a causal model.

    `run([1, n]) -> [1, frames, width]` is the model. A frame in the middle of a clip of `n_samples` is
    probed with a NaN at one sample at a time, the two edges found by bisection (a frame reads one
    contiguous span); the clip doubles while either edge reaches the clip's own ends, up to
    `max_samples` (0: no limit)."""
    while True:
        t = (n_samples // hop) // 2
        gen = torch.Generator().manual_seed(0)
        base = torch.randn(1, n_samples, generator=gen) * 0.1

        def reads(j: int) -> bool:
            x = base.clone()
            x[0, j] = float("nan")
            with torch.no_grad():
                out = run(x)[0]
            return bool(torch.isnan(out[t]).any()) if t < out.shape[0] else False

        if not reads(hop * t):
            raise ValueError(f"frame {t} does not read its own first sample {hop * t}: the frame grid is "
                             f"not one frame per {hop} samples from 0")
        if reads(0) or reads(n_samples - 1):
            if max_samples and 2 * n_samples > max_samples:
                raise ValueError(f"a frame reads past both ends of a {n_samples}-sample clip, and the "
                                 f"graph takes at most {max_samples}: its reach is not finite here")
            n_samples *= 2
            continue
        lo, hi = 0, hop * t                    # reads(lo) is False, reads(hi) is True
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if reads(mid) else (mid, hi)
        left = hop * t - hi
        lo, hi = hop * t, n_samples - 1        # reads(lo) is True, reads(hi) is False
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if reads(mid) else (lo, mid)
        return left, max(0, lo - (hop * (t + 1) - 1))


def window_frames(left: int, right: int, hop: int) -> Tuple[int, int]:
    """The reach in whole frames: a window starts on the frame grid, `ceil(left / hop)` frames back, and
    ends `ceil(right / hop)` frames past the last frame it keeps."""
    return -(-left // hop), -(-right // hop)


def stream_components(*, hop: int, left_frames: int, right_frames: int, row_width: int, takes_length: bool,
                      min_samples: int, max_samples: int) -> List:
    """The constants and the fragment a synthesized single-graph driver gains (`CodecDecodeBuilder.stream`).
    Top level, so `stream_open` / `stream_feed` / `stream_finish` sit beside `infer` and read them."""
    from .driver_components import ExportConstants, LuaFragment

    max_piece = max_samples // hop - left_frames - right_frames
    if max_piece < 1:
        raise ValueError(f"the graph takes at most {max_samples} samples, which does not hold one frame "
                         f"between {left_frames} frames of context and {right_frames} of look-ahead")
    return [
        ExportConstants(values={
            "HOP": hop, "LEFT_FRAMES": left_frames, "RIGHT_FRAMES": right_frames, "ROW_WIDTH": row_width,
            "TAKES_LENGTH": 1 if takes_length else 0, "MIN_SAMPLES": min_samples, "MAX_PIECE": max_piece,
        }, top_level=True),
        LuaFragment(DRIVER_DIR / "00_stream.lua", top_level=True,
                    defines=("stream_open", "stream_feed", "stream_finish")),
    ]
