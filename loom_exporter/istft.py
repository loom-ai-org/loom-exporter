"""
Pure-torch inverse-STFT, tracer-friendly by construction (EXPORT-IMPROVEMENT-BACKLOG.md item 4).

`torch.istft` has NO coremltools torch-frontend handler at all -- tracing a model that calls it directly
fails immediately with `NotImplementedError: PyTorch convert function for op 'istft' not implemented`
(confirmed directly; unlike `torch.stft`, which coremltools' own `common::lower_complex_dialect_ops` pass
decomposes automatically into ops this exporter already covers -- see this module's sibling finding for
the forward transform, and the new "pad" op handling in exporter.py it required). No MIL pass can fix this
because MIL never even sees the op: `ct.convert` fails at the torch-tracing step, before any MIL graph
exists to run a pass over.

The only viable fix is a pre-trace substitution: reimplement ISTFT using ops coremltools *can* trace, and
have a model's `forward()` call this class instead of `torch.istft` directly. `ISTFT` here does exactly
that, reducing the standard real-IDFT-then-overlap-add-then-window-sum-normalize recipe to two
`conv_transpose1d` calls (real/imag synthesis, window baked into each kernel) plus one more
`conv_transpose1d` (over an all-ones "spectrogram") to compute the overlap-added squared-window
normalization ("wsum") for an arbitrary (traced-dynamic) number of frames -- the same reduction already
proven correct in `tools/convert_kokoro/kokoro_stft_common.py`'s docstring (verified there to ~2e-8 max
diff against real `torch.istft` on random, non-self-consistent magnitude/phase), but expressed as live
torch ops here so it flows through the standard `ct.convert` pipeline instead of being hand-derived into
numpy kernels baked directly into a bespoke driver.

Unlike `kokoro_stft_common.py`'s ggml-graph version (where `wsum` is computed host-side once, since Loom's
own JSON topology format has no loop construct to compute it in-graph), this traces `wsum` as a genuine
third `conv_transpose1d` over an all-ones tensor shaped by the real (dynamic) number of frames -- the
natural, traceable way to get the same value for an arbitrary sequence length, matching this whole
pipeline's "exactly one dynamic axis, computed via ops, never hand-derived per input" convention.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

from . import dialect  # noqa: F401  registers the "loom_istft" MIL op (mb.loom_istft)


# ---------------------------------------------------------------------------------------------------
# ADR-080: the engine's ISTFT primitive, an inverse real FFT per frame, in place of the convolutions.
#
# The three `conv_transpose1d` below cost N MACs per bin per frame. On the Pi Zero that was heart's
# 2.23 s of a 4.96 s synthesis, and the FFT does the same in 0.18 s (loom.cpp `scripts/bench54.cpp`).
# A TRACE records one opaque `loom::istft` call, which becomes one `ISTFT` node; an EAGER call (the
# fixtures, every reference check) never reaches it and keeps the convolutions bit for bit.
# `LOOM_EXPORT_NO_FFT=1` traces the convolutions instead, for an A/B or an engine without ISTFT.


@torch.library.custom_op("loom::istft", mutates_args=())
def istft_op(real: Tensor, imag: Tensor, window: Tensor, n_fft: int, hop: int, center: bool) -> Tensor:
    """What the engine's ISTFT computes, in torch: the windowed inverse real FFT of every frame,
    overlap-added and divided by the overlap-added window square. Only a trace calls this, and only
    for its shape -- but it is the real thing, so a trace that ever reads the value reads the truth."""
    frames = torch.fft.irfft(torch.complex(real.double(), imag.double()).transpose(1, 2), n=n_fft)
    frames = frames * window.double()                             # (batch, n_frames, n_fft)
    n_frames = real.shape[-1]
    out_len = (n_frames - 1) * hop + n_fft
    sig = F.fold(frames.transpose(1, 2), (1, out_len), (1, n_fft), stride=(1, hop)).reshape(real.shape[0], out_len)
    wsq = (window.double() ** 2).view(1, n_fft, 1).expand(1, n_fft, n_frames)
    wsum = F.fold(wsq, (1, out_len), (1, n_fft), stride=(1, hop)).reshape(1, out_len)
    out = (sig / wsum).to(real.dtype)
    if center:
        out = out[:, n_fft // 2:out_len - n_fft // 2]
    return out


@istft_op.register_fake
def _(real, imag, window, n_fft, hop, center):
    out_len = (real.shape[-1] - 1) * hop + n_fft - (n_fft if center else 0)
    return real.new_empty((real.shape[0], out_len))


@register_torch_op(torch_alias=["loom::istft"])
def _convert_loom_istft(context, node):
    real, imag, window, n_fft, hop, center = _get_inputs(context, node, expected=6)
    context.add(mb.loom_istft(real=real, imag=imag, window=window, n_fft=n_fft, hop=hop,
                              center=mb.cast(x=center, dtype="int32"), name=node.name))


def _use_fft() -> bool:
    return os.environ.get("LOOM_EXPORT_NO_FFT", "") in ("", "0")


def _periodic_hann(n_fft: int) -> torch.Tensor:
    n = torch.arange(n_fft, dtype=torch.float64)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * n / n_fft)


class ISTFT(nn.Module):
    """Drop-in (real/imag input, not complex-dtype -- MIL/ggml have no complex tensor type) replacement
    for `torch.istft(..., return_complex_input=True)` restricted to a real (not Hann-alternative) window
    and `win_length == n_fft`, the shape every model on this roadmap needs. `forward(real, imag)` takes
    `(batch, n_freq, n_frames)` tensors (`n_freq = n_fft // 2 + 1`) and returns `(batch, out_len)`."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int = None, center: bool = True):
        super().__init__()
        if win_length is None:
            win_length = n_fft
        if win_length != n_fft:
            raise NotImplementedError(
                "ISTFT: win_length != n_fft is not supported -- pad/trim the window to n_fft yourself."
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

        n_freq = n_fft // 2 + 1
        window = _periodic_hann(n_fft)
        # Standard real-IDFT-via-Hermitian-symmetry per-bin weighting (2/n_fft interior, 1/n_fft at the
        # DC and Nyquist bins) -- confirmed numerically against real torch.istft, not assumed from theory
        # alone (kokoro_stft_common.py's own docstring records the same verification).
        w = torch.full((n_freq,), 2.0 / n_fft, dtype=torch.float64)
        w[0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            w[-1] = 1.0 / n_fft
        k = torch.arange(n_freq, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        angle = 2.0 * torch.pi * k * n / n_fft
        cos_synth = (w[:, None] * window[None, :] * torch.cos(angle)).to(torch.float32)[:, None, :]
        neg_sin_synth = (-(w[:, None] * window[None, :] * torch.sin(angle))).to(torch.float32)[:, None, :]
        # conv_transpose1d weight convention: (in_channels=n_freq, out_channels=1, kernel_size=n_fft).
        self.register_buffer("cos_synth", cos_synth)
        self.register_buffer("neg_sin_synth", neg_sin_synth)
        self.register_buffer("window_sq", (window ** 2).to(torch.float32).view(1, 1, n_fft))
        self.register_buffer("window", window.to(torch.float32), persistent=False)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_tracing() and _use_fft():
            return torch.ops.loom.istft(real, imag, self.window, self.n_fft, self.hop_length, self.center)
        synth_real = F.conv_transpose1d(real, self.cos_synth, stride=self.hop_length)
        synth_imag = F.conv_transpose1d(imag, self.neg_sin_synth, stride=self.hop_length)
        combined = synth_real + synth_imag  # (batch, 1, out_len)

        n_frames = real.shape[-1]
        ones = torch.ones((1, 1, n_frames), dtype=real.dtype, device=real.device)
        wsum = F.conv_transpose1d(ones, self.window_sq, stride=self.hop_length)  # (1, 1, out_len)

        out = (combined / wsum).squeeze(1)  # (batch, out_len)
        if self.center:
            pad = self.n_fft // 2
            out = out[:, pad:-pad]
        return out
