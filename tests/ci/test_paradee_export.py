"""Paradee's rewrites (paradee_export.py, loom.cpp ADR-135), each against upstream's own formula: the sine
bank's phase split into the driver's per-frame doubles and the graph's per-sample interpolation, and the
STFT pair as convolutions with constant kernels. None needs the checkpoint or upstream's scripts; the
formulas are copied from `training/scripts/vhead.py` (`sine_bank`) and `torch.stft`/`istft` is the
reference the head's `torch.stft` calls are."""
import math

import pytest
import torch
import torch.nn.functional as F

from loom_exporter.paradee_export import FRAME_HOP, SAMPLE_RATE, _SynthWrapper, _dft_bases, phase_ramp


def _upstream_sine_bank(f0, K=64):
    """`vhead.sine_bank`, float64 throughout (the comparison is of structure, not of upstream's rounding)."""
    B, n = f0.shape
    r = f0.double() * FRAME_HOP / SAMPLE_RATE
    c = torch.cumsum(r, 1) % 1
    d = F.pad(r[:, 1:], (0, 1))
    p = (torch.arange(n * FRAME_HOP, dtype=torch.float64) + 0.5) / FRAME_HOP - 0.5
    i = p.floor().clamp(0, n - 1).long()
    fr = (p - i).clamp(0, 1)
    cyc = c[:, i] + fr * d[:, i]
    f0s = f0.double().repeat_interleave(FRAME_HOP, 1)
    k = torch.arange(1, K + 1, dtype=torch.float64)[None, :, None]
    amp = ((f0s[:, None] * k < 11000) & (f0s[:, None] > 10)).double()
    return (torch.sin(2 * torch.pi * k * cyc[:, None]) * amp).sum(1) * 0.1, cyc[0]


def _driver_phase(f0):
    """`paradee_phase` (paradee_driver/00_paradee.lua), line for line: flat [frames + 1, 2]."""
    phase = [f0[0] * FRAME_HOP / SAMPLE_RATE % 1.0, 0.0]
    running = 0.0
    for j in range(len(f0)):
        running += f0[j] * FRAME_HOP / SAMPLE_RATE
        nxt = f0[j + 1] * FRAME_HOP / SAMPLE_RATE if j + 1 < len(f0) else 0.0
        phase += [running % 1.0, nxt]
    return torch.tensor(phase, dtype=torch.float64).reshape(-1, 2)


def test_the_split_phase_is_upstreams_sine_bank():
    # A long utterance whose running phase reaches thousands of cycles, with unvoiced stretches (f0 = 0) and
    # a top harmonic near the 11 kHz mute.
    frames = 1500
    t = torch.arange(frames, dtype=torch.float64)
    f0 = 140 + 60 * torch.sin(t / 23) + 30 * torch.sin(t / 4.1)
    f0[300:340] = 0.0
    want, want_cyc = _upstream_sine_bank(f0[None])
    phase = _driver_phase(f0.tolist())
    cyc = (phase @ phase_ramp()).reshape(-1)[FRAME_HOP // 2: -(FRAME_HOP // 2)]
    # Cycles, not radians: the graph's interpolation lands on the same phase as upstream's, frame by frame.
    assert cyc.shape == want_cyc.shape
    assert torch.allclose(cyc, want_cyc, atol=1e-12)
    k = torch.arange(1, 65, dtype=torch.float64)[:, None]
    f0s = (f0[:, None] @ torch.ones(1, FRAME_HOP, dtype=torch.float64)).reshape(-1)
    amp = ((k @ f0s[None]) < 11000).double() * (f0s > 10).double()[None]
    got = (torch.ones(1, 64, dtype=torch.float64) @ (torch.sin((2 * torch.pi * k) @ cyc[None]) * amp))[0] * 0.1
    assert (got - want[0]).abs().max() < 1e-9


def test_an_unwrapped_running_phase_in_float32_is_what_the_split_avoids():
    """Why the sum leaves the graph: the same running phase summed in float32 and never wrapped is far off
    by harmonic 64 after a few seconds."""
    frames = 1500
    f0 = 140 + 60 * torch.sin(torch.arange(frames, dtype=torch.float64) / 23)
    r = (f0 * FRAME_HOP / SAMPLE_RATE).float()
    import numpy as np
    unwrapped = torch.from_numpy(np.cumsum(r.numpy(), dtype=np.float32)).double()
    exact = torch.cumsum(f0 * FRAME_HOP / SAMPLE_RATE, 0)
    radians_at_64 = 2 * math.pi * 64 * (unwrapped - exact).abs().max()
    assert radians_at_64 > 0.1, radians_at_64


@pytest.mark.parametrize("n,hop", [(600, 150), (150, 75)])
def test_the_convolution_stft_pair_is_torchs(n, hop):
    torch.manual_seed(0)
    frames = 40
    x = torch.randn(frames * 2 * hop)
    fwd, inv, w2 = _dft_bases(n)
    re, im = _SynthWrapper._stft(x, fwd, n, hop)
    window = torch.hann_window(n)
    spec = torch.stft(x, n, hop, window=window, center=True, pad_mode="reflect", return_complex=True)
    assert re.shape == spec.real.shape
    assert (re - spec.real).abs().max() < 2e-4 and (im - spec.imag).abs().max() < 2e-4
    gain = torch.rand(spec.shape) + 0.5
    want = torch.istft(spec * gain, n, hop, window=window, center=True, length=x.shape[0])
    got = _SynthWrapper._istft(re * gain, im * gain, inv, w2, n, hop)
    assert got.shape == want.shape
    assert (got - want).abs().max() < 1e-4
