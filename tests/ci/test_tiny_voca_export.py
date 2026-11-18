"""tiny-voca's rewrites (loom.cpp ADR-133), each against upstream's own formula: the STFT filter as
convolutions with the windowed DFT basis, the interleaved rotary rotation spelled without a strided slice,
and the pulse train split into the driver's per-frame phases and the graph's float32 per-sample math.
None needs the checkpoint or the `tiny_voca` package; the formulas are copied from `tiny_voca/model.py`
(`filtered`, `rotate`, `pulse_train`)."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from loom_exporter.tiny_voca_export import (_SynthWrapper, _analysis_kernels, _rotate, _synthesis_kernel)


def _upstream_filtered(source, response, fft, hop, window):
    spectrum = torch.stft(source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, window=window, center=True,
                       length=source.shape[-1])


def _upstream_rotate(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


@pytest.mark.parametrize("fft,hop", [(64, 16), (128, 32)])
def test_the_convolution_filter_is_upstreams_stft_filter(fft, hop):
    torch.manual_seed(0)
    window = torch.hann_window(fft, dtype=torch.float64)
    frames = 12
    source = torch.randn(1, frames * hop, dtype=torch.float64)
    bins = fft // 2 + 1
    # One response per STFT frame of the padded source.
    n = 1 + (source.shape[-1] + fft // 2) // hop
    real, imaginary = torch.randn(n, bins, dtype=torch.float64), torch.randn(n, bins, dtype=torch.float64)
    # The call upstream's SourceFilter makes: half an FFT of zeros behind, cropped back to the source.
    want = _upstream_filtered(F.pad(source, (0, fft // 2)), torch.complex(real, imaginary)[None], fft, hop,
                              window)[:, :source.shape[-1]]

    a_real, a_imaginary = _analysis_kernels(window)
    host = SimpleNamespace(x_analysis_real=a_real.double(), x_analysis_imaginary=a_imaginary.double(),
                           x_synthesis=_synthesis_kernel(window).double(),
                           x_envelope=(window ** 2)[None, None])
    got = _SynthWrapper._filter(host, source, real, imaginary, "x", fft, hop)
    assert got.shape == want[0].shape
    # Both in f64, so what is left is the f32 rounding of the kernels.
    assert torch.allclose(got, want[0], atol=1e-5), (got - want[0]).abs().max()


def test_the_pair_reshape_is_the_interleaved_rotation():
    torch.manual_seed(1)
    heads, t, w = 4, 9, 16
    x = torch.randn(heads, t, w)
    angle = torch.arange(t)[:, None] * torch.rand(w // 2)[None]
    assert torch.allclose(_rotate(x, angle.cos(), angle.sin()), _upstream_rotate(x, angle.cos(), angle.sin()))


def _upstream_pulse_train(f0, sample_rate, hop, limit):
    frames = f0.shape[1]
    knots = torch.log(f0.double().clamp_min(1.0))
    knots = torch.cat((knots, knots[:, -1:]), dim=1)
    position = torch.arange(frames * hop, dtype=torch.float64) / hop
    low = position.floor().long()
    weight = position - low
    frequency = torch.exp(knots[:, low] * (1 - weight) + knots[:, low + 1] * weight)
    theta = torch.remainder(2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1), 2 * math.pi)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def _expm1(x):
    u = math.exp(x)
    if u == 1.0:
        return x
    return -1.0 if u - 1.0 == -1.0 else (u - 1.0) * x / math.log(u)


def _driver_phases(f0, hop, sample_rate):
    """`voca_phases` (tiny_voca_driver/00_voca.lua), line for line."""
    knots = [math.log(max(v, 1.0)) for v in f0] + [math.log(max(f0[-1], 1.0))]
    low, high, start, acc = [], [], [], 0.0
    for j in range(len(f0)):
        a, b = knots[j], knots[j + 1]
        low.append(a), high.append(b), start.append(acc - math.floor(acc + 0.5))
        series = hop if b == a else _expm1(b - a) / _expm1((b - a) / hop)
        acc += math.exp(a) * series / sample_rate
        acc -= math.floor(acc)
    return low, high, start


def _ggml_cumsum(x, dim):
    """ggml_vec_cumsum_f32: sequential float32 (torch's CPU cumsum accumulates in double, which would
    hide exactly the error this split exists to avoid)."""
    import numpy as np
    return torch.from_numpy(np.cumsum(x.numpy(), axis=dim, dtype=np.float32))


@pytest.mark.parametrize("centred", [True, False])
def test_the_split_pulse_train_is_upstreams(monkeypatch, centred):
    # A long, low voice: many harmonics (W ~ 200 at 55 Hz) and a phase in the thousands of cycles.
    hop, sample_rate, limit = 256, 24000, 0.47
    frames = 1200
    t = torch.arange(frames, dtype=torch.float64)
    f0 = (55 + 40 * (1 + torch.sin(t / 37)) + 25 * torch.sin(t / 5.3)).float()
    want = _upstream_pulse_train(f0[None], sample_rate, hop, limit)[0].double()

    low, high, start = _driver_phases(f0.double().tolist(), hop, sample_rate)
    ramp = torch.arange(hop, dtype=torch.float32) / hop
    host = SimpleNamespace(interpolate=torch.stack((1 - ramp, ramp)), sample_rate=float(sample_rate),
                           harmonic_limit=limit)
    monkeypatch.setattr(torch, "cumsum", _ggml_cumsum)
    if not centred:  # the [0, 2 pi) theta: the form the graph must not use
        floor = torch.floor
        monkeypatch.setattr(torch, "floor", lambda x: floor(x - 0.5) if x.shape[-1] == hop else floor(x))
    knots = torch.tensor([low, high], dtype=torch.float32).T.contiguous()
    got = _SynthWrapper._pulses(host, knots, torch.tensor(start, dtype=torch.float32))[0].double()
    snr = 10 * torch.log10((want ** 2).sum() / ((got - want) ** 2).sum())
    # Measured on this train: centred 7.1e-3 max / 84.0 dB (a pulse peaks at ~sqrt(2 W) ~ 20; through
    # upstream's own filters the waveform moves by 1e-4); the uncentred form 0.22 / 31.4 dB.
    if centred:
        assert snr > 75, snr
        assert (got - want).abs().max() < 2e-2, (got - want).abs().max()
    else:
        assert snr < 50, snr
