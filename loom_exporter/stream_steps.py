"""The two convolution steps a streaming decoder is made of, and the attention mask that goes with them.

A causal decoder streams exactly when every stateful piece carries its state from one call to the next:
a causal convolution its last input samples, a transposed convolution the overlap its next call adds
to, an attention layer its window of keys and values. Pocket-TTS's Mimi decoder (loom.cpp ADR-101) and
the codecs that stream (ADR-102) are built from the same two convolutions, so they live here once.
"""
import torch
import torch.nn as nn


# Every slice below has a fixed width counted from one END of the tensor. The frame count is a graph
# axis, and a slice whose bounds come from a shape mints a fresh symbol the axis algebra cannot follow.

def conv_step(conv: nn.Conv1d, x: torch.Tensor, previous: torch.Tensor):
    """`StreamingConv1d` on a later call: the last `(k - 1) * d + 1 - s` input samples of the previous
    call are prepended, and this call's last ones are the next call's."""
    x = torch.cat([previous, x], dim=-1)
    return conv(x), x[..., -previous.shape[-1]:]


def conv_transpose_step(convtr: nn.ConvTranspose1d, x: torch.Tensor, partial: torch.Tensor):
    """`StreamingConvTranspose1d` on a later call: the previous call's `k - s` overlapping samples are added
    to this call's first ones, and this call's own overlap (without the bias, which its first samples
    already carry) is the next call's. Spelled as head + rest, then the overlap dropped, so no slice is
    empty: the x16 upsampler's one-latent call makes exactly as many samples as it carries."""
    y = convtr(x)
    tail = partial.shape[-1]
    out = torch.cat([y[..., :tail] + partial, y[..., tail:]], dim=-1)[..., :-tail]
    rest = y[..., -tail:]
    if convtr.bias is not None:
        rest = rest - convtr.bias.view(1, -1, 1)
    return out, rest


def window_mask(positions: torch.Tensor, key_positions: torch.Tensor, window: int) -> torch.Tensor:
    """`[1, 1, q, k]`, additive: 0 where `0 <= q - k < window` and `k >= 0`, else a large negative.

    `positions` are the queries' absolute positions and `key_positions` the keys' -- a carried window
    of keys first, then the call's own. A key at a negative position is a slot no call has written yet.
    Spelled as outer products against a ones vector and `relu`s, because ggml repeats `b` into `a` and
    cannot broadcast two ways (`pocket_tts_export.MimiDecoderPhase.window_mask` says the same)."""
    q = positions.to(torch.float32)[0]
    k = key_positions.to(torch.float32)[0]
    delta = (torch.matmul(q.view(-1, 1), torch.ones_like(k).view(1, -1))
             - torch.matmul(torch.ones_like(q).view(-1, 1), k.view(1, -1)))
    outside = torch.relu(-delta) + torch.relu(delta - (window - 1)) + torch.relu(-k).view(1, -1)
    return (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, q.shape[0], k.shape[0])


class StateInit(nn.Module):
    """`(zero) -> state_1..state_N`, each its shape filled with `zero`'s one value: a fresh decoder's
    state made on the device, so a stream's first call needs no array of zeros from the host."""

    def __init__(self, shapes):
        super().__init__()
        self.shapes = [tuple(s) for s in shapes]

    def forward(self, zero):                                            # (1,)
        return tuple(zero.view(*([1] * len(s))).expand(*s).contiguous() for s in self.shapes)
