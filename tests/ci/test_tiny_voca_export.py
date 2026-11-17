"""tiny-voca's two rewrites (loom.cpp ADR-133), each against upstream's own formula: the STFT filter as
convolutions with the windowed DFT basis, and the interleaved rotary rotation spelled without a strided
slice. Neither needs the checkpoint or the `tiny_voca` package; the formulas are copied from
`tiny_voca/model.py` (`filtered`, `rotate`)."""
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
