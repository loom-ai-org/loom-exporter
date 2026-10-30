"""Audio8-TTS's codec (`codec.pth`, `ArkttsCodec`): family 11's sixth member, ten codes a frame in and
44.1 kHz audio out, 2048 samples a frame.

The DECODE half only, as for every codec here (`audio_codec_export` says why); a voice's codes are
encoded once, in Python, by `audio8_tts_voices`. The decode path is a DAC-style decoder behind a
windowed transformer and a 4x upsampler:

    codes -> semantic RVQ (1 x 4096) + residual RVQ (9 x 1024) -> post_module (8 layers, window 128)
          -> 2 x (transposed conv /2, ConvNeXt) -> DAC decoder (8, 8, 4, 2) -> tanh

**The codes are CLAMPED, in the graph, and that is the reference's arithmetic rather than a guard.**
`ArkttsDownsampleQuantizer.decode` clamps codebook 0 to 4096 and the nine residual codebooks to
1024 -- and the TTS's fast stack draws all nine from ONE 4096-way head, so a residual id above 1023
is a legal draw that the reference decodes as 1023. Dropping the clamp would gather past the table.

Everything else is a rewrite of the checkpoint's own modules, each exact (checked against
`ArkttsCodec.decode` before the trace):

* **RoPE: interleaved pairs and a BF16 table**, the talker's finding again (`audio8_tts_export`): the
  q and k rows of every `wqkv` are permuted evens-then-odds, the rotation is rotate-half, and `cos`/
  `sin` are a gathered table of the reference's own bf16-rounded values, indexed by frame.
* **The window mask** is `audio_codec_export._qwen3_tts_sliding_causal_mask` -- the same
  `0 <= i - j < window` the reference builds with `column <= row & column >= row - window + 1`.
* **The causal convolutions' right pad is identically zero** at stride 1, which is every one on the
  decode path (`_patch_qwen3_tts_tokenizer`'s argument), and the residual units' trim is zero with it.
* **The transposed convolutions' crop is a STATIC negative slice**: `x[..., :x.shape[-1] - crop]`
  reads the length and kills the frame axis's algebra ([Retro-044]).
* **Snake is the plain expression**, not the `torch.jit.script` function, and LayerScale is not in place.
* **Weight norm is folded** -- both spellings, the parametrization and the legacy hook on the RVQ
  projections.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The frame count the post-module's rope table covers, and therefore the export's `max_frames`: the
# talker's own `max_seq_len`, past which it cannot have produced the codes.
MAX_FRAMES = 2048


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _permute_qk_rows(attention) -> None:
    """Reorder every q and k head's rows of `wqkv` evens-then-odds, in place (see the docstring)."""
    d = attention.head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    rows = [h * d + perm for h in range(attention.n_head + attention.n_local_heads)]
    rows.append(torch.arange((attention.n_head + attention.n_local_heads) * d,
                             (attention.n_head + 2 * attention.n_local_heads) * d))
    rows = torch.cat(rows)
    with torch.no_grad():
        attention.wqkv.weight.copy_(attention.wqkv.weight[rows].clone())
        if attention.wqkv.bias is not None:
            attention.wqkv.bias.copy_(attention.wqkv.bias[rows].clone())
    attention._loom_rotate_half = True


def install_patches(modeling) -> None:
    """The class-level rewrites the decode path needs to trace. Each keeps the reference's arithmetic;
    `check_decoder` measures that on the real checkpoint."""
    from .audio_codec_export import _qwen3_tts_sliding_causal_mask

    def attention_forward(self, x, rope, mask):
        assert getattr(self, "_loom_rotate_half", False), "the q/k rows were not permuted"
        b, t, _ = x.shape
        q_size, kv_size = self.n_head * self.head_dim, self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split((q_size, kv_size, kv_size), dim=-1)
        cos, sin = rope
        q = q.view(b, t, self.n_head, self.head_dim)
        k = k.view(b, t, self.n_local_heads, self.head_dim)
        q = (q * cos + _rotate_half(q) * sin).transpose(1, 2)
        k = (k * cos + _rotate_half(k) * sin).transpose(1, 2)
        v = v.view(b, t, self.n_local_heads, self.head_dim).transpose(1, 2)
        rep = self.n_head // self.n_local_heads
        if rep > 1:
            k = k[:, :, None].expand(b, self.n_local_heads, rep, t, self.head_dim).reshape(
                b, self.n_head, t, self.head_dim)
            v = v[:, :, None].expand(b, self.n_local_heads, rep, t, self.head_dim).reshape(
                b, self.n_head, t, self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, q_size))

    def window_forward(self, x, x_lens=None):
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        # Positions by arithmetic on the hidden state's own length, then the reference's table rows.
        positions = torch.arange(x.shape[1], device=x.device)
        cos = self._loom_cos(positions).unsqueeze(0).unsqueeze(2)       # [1, T, 1, head_dim]
        sin = self._loom_sin(positions).unsqueeze(0).unsqueeze(2)
        mask = _qwen3_tts_sliding_causal_mask(x, self.window_size)
        for layer in self.layers:
            x = layer(x, (cos, sin), mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x

    def layer_scale_forward(self, x):
        return x * self.gamma

    def causal_conv_forward(self, x):
        assert self.stride == 1, (
            "the right pad is identically zero at stride 1 only; the decode path has no strided causal "
            f"conv, but this one has stride {self.stride}")
        return self.conv(F.pad(x, (self.padding, 0)))

    def transpose_conv_forward(self, x):
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[..., :-crop] if crop else x

    def residual_forward(self, x):
        # The reference trims `x` by the length the block lost, which is zero for causal stride-1
        # convolutions -- the patched `causal_conv_forward` above.
        return x + self.block(x)

    def snake_forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

    modeling.ArkttsCodecAttention.forward = attention_forward
    modeling.ArkttsCodecWindowTransformer.forward = window_forward
    modeling.ArkttsCodecLayerScale.forward = layer_scale_forward
    modeling.ArkttsCausalConv1d.forward = causal_conv_forward
    modeling.ArkttsCausalConvTranspose1d.forward = transpose_conv_forward
    modeling.ArkttsResidualUnit.forward = residual_forward
    modeling.ArkttsSnake1d.forward = snake_forward


def _fold_weight_norm(module: nn.Module) -> None:
    from torch.nn.utils import parametrize

    for sub in module.modules():
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)
        elif hasattr(sub, "weight_g") and hasattr(sub, "weight_v"):
            torch.nn.utils.remove_weight_norm(sub)


class Audio8CodecDecoder(nn.Module):
    """`codes [1, T, 10] -> waveform [1, 1, 2048 * T]`: `ArkttsCodec.decode` with the codes taken
    frame-major, the clamp kept, and the two residual sums in the reference's order."""

    def __init__(self, codec):
        super().__init__()
        quantizer = codec.quantizer
        self.semantic = quantizer.semantic_quantizer.quantizers
        self.residual = quantizer.quantizer.quantizers
        self.semantic_size = int(quantizer.semantic_quantizer.codebook_size)
        self.residual_size = int(quantizer.quantizer.codebook_size)
        self.post_module = quantizer.post_module
        self.upsample = quantizer.upsample
        self.decoder = codec.decoder

    @staticmethod
    def _decode(quantizer, ids, n):
        rows = F.embedding(ids, quantizer.codebook.weight).reshape(1, n, quantizer.codebook.weight.shape[1])
        return quantizer.out_proj(rows.transpose(1, 2))

    def forward(self, codes):
        n = codes.shape[1]
        # Clamped as floats: exact for ids this small, and the cast back to I32 is the gather index.
        # Each column FLATTENED to one axis before the gather, `moss_tts_export._EmbedWrapper`'s rule:
        # `codes[:, :, 0]` reached ggml as a `[1, T]` index, the gather came out `[8, 1, T]`, and every
        # convolution after it ran over a length-1 axis with the frames as a batch -- the right numbers
        # in the wrong layout, until the first transposed convolution refused a rank-3 input.
        values = codes.float()
        semantic = values[:, :, :1].clamp(0, self.semantic_size - 1).to(torch.int32).reshape(n)
        hidden = 0.0 + self._decode(self.semantic[0], semantic, n)
        rest = 0.0
        for index, quantizer in enumerate(self.residual):
            ids = values[:, :, index + 1:index + 2].clamp(0, self.residual_size - 1).to(torch.int32).reshape(n)
            rest = rest + self._decode(quantizer, ids, n)
        return self.decoder(self.upsample(self.post_module(hidden + rest)))


def load(path):
    """`codec.pth` (or the directory holding it) -> the patched, folded `ArkttsCodec`, with its
    decoder module built at `_loom_decoder`."""
    from .audio8_tts_voices import load_codec

    path = Path(path)
    model_dir = path.parent if path.is_file() else path
    codec = load_codec(model_dir)
    modeling = sys.modules[type(codec).__module__]
    reference = load_codec(model_dir)            # unpatched-forward twin, for `check_decoder`
    _fold_weight_norm(codec)
    for module in (codec.quantizer.post_module,):
        head_dim, base = module.head_dim, module.rope_base
        table = modeling._rope(MAX_FRAMES, head_dim, base)       # the reference's own, bf16
        f = table.float()
        module._loom_cos = nn.Embedding.from_pretrained(torch.cat([f[..., 0], f[..., 0]], -1), freeze=True)
        module._loom_sin = nn.Embedding.from_pretrained(torch.cat([f[..., 1], f[..., 1]], -1), freeze=True)
        for layer in module.layers:
            _permute_qk_rows(layer.attention)
    reference_decode = _reference_decode(reference)
    install_patches(modeling)
    codec._loom_decoder = Audio8CodecDecoder(codec).eval()
    check_decoder(codec._loom_decoder, reference_decode)
    return codec


def _reference_decode(codec):
    """The checkpoint's own `decode`, run NOW on probe codes -- before `install_patches` rewrites the
    classes both codecs share -- and its waveforms kept for `check_decoder`."""
    generator = torch.Generator().manual_seed(0)
    probes = {}
    for frames in (3, 37):
        codes = torch.randint(0, 4096, (1, 10, frames), generator=generator)
        with torch.inference_mode():
            probes[frames] = (codes, codec.decode(codes))
    return probes


def check_decoder(decoder, probes, atol: float = 1e-4) -> None:
    """The rewritten decoder against the reference's waveforms on random codes -- residual ids up to
    4095, so the clamp is exercised -- at two lengths; raises past `atol`."""
    for frames, (codes, ref) in probes.items():
        with torch.inference_mode():
            got = decoder(codes.transpose(1, 2).contiguous())
        if got.shape != ref.shape:
            raise ValueError(f"{frames} frames: the decoder returns {tuple(got.shape)}, the reference "
                             f"{tuple(ref.shape)}")
        diff = float((got - ref).abs().max())
        print(f"  decoder vs ArkttsCodec.decode at {frames} frames: max |d| {diff:.2e}")
        if diff > atol:
            raise ValueError(f"{frames} frames: the rewritten decoder is {diff:.3e} from the reference")


def geometry(model) -> dict:
    quantizer = model.quantizer
    n = 1 + len(quantizer.quantizer.quantizers)
    return dict(n_codebooks=n, codebook_size=int(quantizer.semantic_quantizer.codebook_size),
                sample_rate=int(model.sample_rate), hop_length=int(model.frame_length),
                vq_strides=[1] * n)
