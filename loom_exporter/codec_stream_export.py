"""A codec whose decoder is causal throughout, exported with a STREAMING step beside its one-pass decode
(loom.cpp ADR-102): codes in as an LM makes them, audio out a few frames later.

Two leaves, one template: Qwen3-TTS's 12 Hz tokenizer and Audio8-TTS's codec. The first is described here. Its decoder is causal end to end -- causal convolutions,
transposed convolutions whose overlap is carried, ConvNeXt blocks, and an 8-layer transformer over a
72-frame causal window -- so carrying every piece of state from one call to the next computes what one
call over the whole sequence computes (`stream_steps` has the two convolutions). Upstream ships no
streaming decode to grade against: its `decode` is `chunked_decode(300, 25)`, which equals one whole call
through 299 frames (ADR-034) and is an approximation past it, where a 25-frame left context cuts a
receptive field the step keeps.

Three graphs:
  - `decode`:        the one-pass decode `audio_codec_export` has always traced (its `main_topology`),
                     unchanged, and the same 300 + 25 chunk loop in `infer` -- so one-pass audio does
                     not move. Renamed because `main_topology` is the flattened export's namespace.
  - `codec_step`:    `(codes [1, n, w], positions [n], key_positions [window + n], state_1..state_N)`
                     -> `(samples, state_1'..state_N')`: the next `n` frames, the decoder's state carried.
  - `codec_init`:    a fresh state, broadcast from one number on the device.

The driver's `stream_open` / `stream_feed({codes})` / `stream_finish` decode each push as it arrives,
keeping the state on the device (`loom.keep_state`), and the frame count is the stream's own position.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import math

import numpy as np
import torch
import torch.nn as nn

from .audio_codec_export import AudioCodecExportConfig, CodecFamily
from .decomposition import Decomposition, MultiPhase
from .spec_protocol import Unchecked
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .stream_steps import StateInit, conv_step, conv_transpose_step, window_mask

# A codec whose decoder is NOT causal streams by windows over its one-pass graph (`00_window_stream.lua`),
# with this many frames of context each side: DAC 44 kHz's receptive field is under 16 (16 each side is
# 4.1e-07 from one call on random codes, 8 is 5.9e-03); SNAC 24 kHz's is 3 coarse frames (7e-16 from one
# call at f64 with the noise held to its frames, 2 is 1.3e-03).
WINDOW_FAMILIES = {CodecFamily.DAC: 16, CodecFamily.SNAC: 3}

# The most frames one `codec_step` call decodes: a stream's push is a few frames, never a chunk.
MAX_STEP_FRAMES = 64
# Trace lengths, odd and distinct from every static dimension.
TRACE_STEP_FRAMES = 5


class Qwen3CodecStep(nn.Module):
    """`Qwen3TTSTokenizerV2Decoder.forward` on the next `n` frames, its state carried in and out.

    The states, in order: each transformer layer's last `window` keys (after RoPE) and values, then each
    convolution that keeps one -- the pre-conv, the ConvNeXt blocks' depthwise convs, the decoder's convs
    and its blocks' transposed convs and residual units -- in the order `forward` meets them. The x2
    upsamplers' transposed convs have `kernel == stride`, so they overlap nothing and carry nothing.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        config = self.decoder.config
        self.window = int(config.sliding_window)
        self.n_layers = int(config.num_hidden_layers)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(self.decoder.pre_transformer.layers[0].self_attn.head_dim)

    @staticmethod
    def _pad(conv) -> int:
        return conv.kernel_size - conv.stride

    def _stateful(self) -> List[Tuple[str, nn.Module]]:
        d = self.decoder
        out = [("conv", d.pre_conv)]
        for transposed, convnext in d.upsample:
            out += [("convtr", transposed), ("conv", convnext.dwconv)]
        for block in d.decoder:
            name = type(block).__name__
            if name.endswith("CausalConvNet"):
                out.append(("conv", block))
            elif name.endswith("DecoderBlock"):
                for sub in block.block:
                    kind = type(sub).__name__
                    if kind.endswith("TransConvNet"):
                        out.append(("convtr", sub))
                    elif kind.endswith("ResidualUnit"):
                        out += [("conv", sub.conv1), ("conv", sub.conv2)]
        return [(k, m) for k, m in out
                if (k == "convtr" and m.right_pad > 0) or (k == "conv" and self._pad(m) > 0)]

    def state_shapes(self) -> List[Tuple[int, ...]]:
        kv = (1, self.kv_heads, self.window, self.head_dim)
        shapes = [kv, kv] * self.n_layers
        for kind, m in self._stateful():
            if kind == "conv":
                shapes.append((1, m.conv.in_channels, self._pad(m)))
            else:
                shapes.append((1, m.conv.out_channels, m.right_pad))
        return shapes

    def forward(self, codes, positions, key_positions, *states):       # (1, n, w), (1, n), (1, window + n)
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import apply_rotary_pos_emb

        d = self.decoder
        states = list(states)
        kv = [states.pop(0) for _ in range(2 * self.n_layers)]
        carried = {id(m) for _, m in self._stateful()}
        new_kv, new_conv = [], []

        def conv(m, x):
            if id(m) not in carried:
                return m(x)
            y, s = conv_step(m.conv, x, states.pop(0))
            new_conv.append(s)
            return y

        def convtr(m, x):
            if id(m) not in carried:
                return m(x)
            y, s = conv_transpose_step(m.conv, x, states.pop(0))
            new_conv.append(s)
            return y

        hidden = d.quantizer.decode(codes.transpose(1, 2))
        hidden = conv(d.pre_conv, hidden).transpose(1, 2)
        tr = d.pre_transformer
        h = tr.input_proj(hidden)
        cos, sin = tr.rotary_emb(h, positions)
        mask = window_mask(positions, key_positions, self.window)
        for i, layer in enumerate(tr.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            b, t, _ = x.shape
            q = attn.q_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = torch.cat([kv[2 * i], k], dim=2)
            v = torch.cat([kv[2 * i + 1], v], dim=2)
            new_kv += [k[:, :, -self.window:], v[:, :, -self.window:]]
            if attn.num_key_value_groups != 1:
                raise NotImplementedError("grouped-query attention in the codec's pre-transformer")
            scores = torch.matmul(q, k.transpose(2, 3)) * attn.scaling + mask
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
            out = attn.o_proj(out.transpose(1, 2).reshape(b, t, -1))
            h = h + layer.self_attn_layer_scale(out)
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        h = tr.output_proj(tr.norm(h)).permute(0, 2, 1)
        for transposed, convnext in d.upsample:
            h = convtr(transposed, h)
            x = conv(convnext.dwconv, h).permute(0, 2, 1)
            x = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(x))))
            h = h + (convnext.gamma * x).permute(0, 2, 1)
        for block in d.decoder:
            name = type(block).__name__
            if name.endswith("CausalConvNet"):
                h = conv(block, h)
            elif name.endswith("DecoderBlock"):
                for sub in block.block:
                    kind = type(sub).__name__
                    if kind.endswith("TransConvNet"):
                        h = convtr(sub, h)
                    elif kind.endswith("ResidualUnit"):
                        x = conv(sub.conv1, sub.act1(h))
                        h = h + conv(sub.conv2, sub.act2(x))
                    else:
                        h = sub(h)
            else:
                h = block(h)
        return (h.clamp(min=-1, max=1).reshape(1, -1), *new_kv, *new_conv)


class Audio8CodecStep(nn.Module):
    """`ArkttsCodec.decode` on the next `n` frames, its state carried in and out -- Audio8-TTS's codec,
    the second leaf (loom.cpp ADR-102). The same template as Qwen3-TTS's 12 Hz codec: a windowed
    transformer (8 layers, 128 frames, 16 query heads over 8 key/value heads), two x2 upsamplers whose
    transposed convs overlap nothing, then a DAC decoder of causal convs and overlapping transposed
    convs. Its reference decodes in one call, so the stream is that call for any length.

    The states: each transformer layer's last `window` keys (after RoPE) and values, then each conv that
    keeps one, in the order `forward` meets them. RoPE is the export's: the q/k rows permuted for
    rotate-half and a gathered bf16 table (`audio8_codec_export`), indexed by `positions`.
    """

    def __init__(self, codec):
        super().__init__()
        self.codes = codec._loom_decoder
        self.post = self.codes.post_module
        attention = self.post.layers[0].attention
        self.window = int(self.post.window_size)
        self.n_layers = len(self.post.layers)
        self.kv_heads = int(attention.n_local_heads)
        self.head_dim = int(attention.head_dim)

    def _stateful(self) -> List[Tuple[str, nn.Module]]:
        out = []
        for stage in self.codes.upsample:
            transposed, convnext = stage
            out += [("convtr", transposed), ("conv", convnext.dwconv)]
        for block in self.codes.decoder.model:
            name = type(block).__name__
            if name == "ArkttsCausalConv1d":
                out.append(("conv", block))
            elif name == "ArkttsDecoderBlock":
                for sub in block.block:
                    kind = type(sub).__name__
                    if kind == "ArkttsCausalConvTranspose1d":
                        out.append(("convtr", sub))
                    elif kind == "ArkttsResidualUnit":
                        out += [("conv", m) for m in sub.block if type(m).__name__ == "ArkttsCausalConv1d"]
        return [(k, m) for k, m in out
                if (k == "convtr" and m.kernel_size > m.stride) or (k == "conv" and m.padding > 0)]

    def state_shapes(self) -> List[Tuple[int, ...]]:
        kv = (1, self.kv_heads, self.window, self.head_dim)
        shapes = [kv, kv] * self.n_layers
        for kind, m in self._stateful():
            if kind == "conv":
                shapes.append((1, m.conv.in_channels, m.padding))
            else:
                shapes.append((1, m.conv.out_channels, m.kernel_size - m.stride))
        return shapes

    def forward(self, codes, positions, key_positions, *states):       # (1, n, 10), (1, n), (1, window + n)
        from .audio8_codec_export import _rotate_half

        states = list(states)
        kv = [states.pop(0) for _ in range(2 * self.n_layers)]
        carried = {id(m) for _, m in self._stateful()}
        new_kv, new_conv = [], []

        def conv(m, x):
            if id(m) not in carried:
                return m(x)
            y, s = conv_step(m.conv, x, states.pop(0))
            new_conv.append(s)
            return y

        def convtr(m, x):
            if id(m) not in carried:
                return m(x)
            y, s = conv_transpose_step(m.conv, x, states.pop(0))
            new_conv.append(s)
            return y

        # `Audio8CodecDecoder.forward`'s quantizer half: clamped ids, the two residual sums in its order.
        d = self.codes
        n = codes.shape[1]
        values = codes.float()
        semantic = values[:, :, :1].clamp(0, d.semantic_size - 1).to(torch.int32).reshape(n)
        hidden = 0.0 + d._decode(d.semantic[0], semantic, n)
        rest = 0.0
        for index, quantizer in enumerate(d.residual):
            ids = values[:, :, index + 1:index + 2].clamp(0, d.residual_size - 1).to(torch.int32).reshape(n)
            rest = rest + d._decode(quantizer, ids, n)
        x = (hidden + rest).transpose(1, 2)
        x = self.post.look_ahead_conv(self.post.input_proj(x))

        where = positions[0]
        cos = self.post._loom_cos(where).unsqueeze(0).unsqueeze(2)
        sin = self.post._loom_sin(where).unsqueeze(0).unsqueeze(2)
        mask = window_mask(positions, key_positions, self.window)
        for i, layer in enumerate(self.post.layers):
            a = layer.attention
            h = layer.attention_norm(x)
            b, t, _ = h.shape
            q_size, kv_size = a.n_head * a.head_dim, a.n_local_heads * a.head_dim
            q, k, v = a.wqkv(h).split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(b, t, a.n_head, a.head_dim)
            k = k.view(b, t, a.n_local_heads, a.head_dim)
            q = (q * cos + _rotate_half(q) * sin).transpose(1, 2)
            k = (k * cos + _rotate_half(k) * sin).transpose(1, 2)
            v = v.view(b, t, a.n_local_heads, a.head_dim).transpose(1, 2)
            k = torch.cat([kv[2 * i], k], dim=2)
            v = torch.cat([kv[2 * i + 1], v], dim=2)
            new_kv += [k[:, :, -self.window:], v[:, :, -self.window:]]
            rep = a.n_head // a.n_local_heads
            if rep > 1:
                total = k.shape[2]
                k = k[:, :, None].expand(b, a.n_local_heads, rep, total, a.head_dim).reshape(b, a.n_head, total, a.head_dim)
                v = v[:, :, None].expand(b, a.n_local_heads, rep, total, a.head_dim).reshape(b, a.n_head, total, a.head_dim)
            scores = torch.matmul(q * (1.0 / math.sqrt(a.head_dim)), k.transpose(-1, -2)) + mask
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
            out = a.wo(out.transpose(1, 2).reshape(b, t, q_size))
            x = x + layer.attention_layer_scale(out)
            x = x + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(x)))
        x = self.post.output_proj(self.post.norm(x)).transpose(1, 2)

        for transposed, convnext in self.codes.upsample:
            x = convtr(transposed, x)
            y = conv(convnext.dwconv, x).permute(0, 2, 1)
            y = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(y))))
            x = x + (convnext.gamma * y).permute(0, 2, 1)
        for block in self.codes.decoder.model:
            name = type(block).__name__
            if name == "ArkttsCausalConv1d":
                x = conv(block, x)
            elif name == "ArkttsDecoderBlock":
                for sub in block.block:
                    kind = type(sub).__name__
                    if kind == "ArkttsCausalConvTranspose1d":
                        x = convtr(sub, x)
                    elif kind == "ArkttsResidualUnit":
                        y = x
                        for m in sub.block:
                            y = conv(m, y) if type(m).__name__ == "ArkttsCausalConv1d" else m(y)
                        x = x + y
                    else:
                        x = sub(x)
            else:
                x = block(x)
        return (x.reshape(1, -1), *new_kv, *new_conv)


def codec_step(model, family: CodecFamily) -> nn.Module:
    """The streaming step for a codec family that has one."""
    if family is CodecFamily.QWEN3_TTS_12HZ:
        return Qwen3CodecStep(model)
    if family is CodecFamily.AUDIO8:
        return Audio8CodecStep(model)
    raise NotImplementedError(f"{family.value} has no streaming step")


def check_step(model, family: CodecFamily, step: nn.Module, width: int, codebook_size: int,
               atol: float = 1e-3) -> float:
    """The step, fed uneven pieces with its state carried, against one call over the same codes -- run
    before anything is traced, so a step that stopped matching the decoder fails the export."""
    torch.manual_seed(0)
    n = 40
    codes = torch.randint(0, codebook_size, (1, n, width))
    with torch.no_grad():
        whole = family.decode(model, codes).reshape(-1)
        states = [torch.zeros(s) for s in step.state_shapes()]
        got, at = [], 0
        for g in (1, 2, 1, 5, 31):
            q = torch.arange(at, at + g, dtype=torch.int32).view(1, -1)
            k = torch.arange(at - step.window, at + g, dtype=torch.int32).view(1, -1)
            out = step(codes[:, at:at + g], q, k, *states)
            got.append(out[0].reshape(-1))
            states = list(out[1:])
            at += g
    delta = float((torch.cat(got) - whole).abs().max())
    if delta > atol:
        raise ValueError(f"the codec step disagrees with the one-call decode by {delta:.3g} (> {atol})")
    return delta


@dataclass(kw_only=True)
class StreamingCodecExportConfig(BaseMultiPhaseModelExportConfig):
    """A causal codec's decode half, one-pass and streaming -> one Loom GGUF. The one-pass half is
    `AudioCodecExportConfig`'s own graph and driver; this config adds the step beside them."""

    architecture: Optional[str] = None
    model_dir: str
    family: CodecFamily = CodecFamily.QWEN3_TTS_12HZ
    root_axis: str = "n_codes"
    chunk_frames: int = 0
    left_context_frames: int = 0
    max_frames: int = 4096
    # The one-pass decode's attention block, for a codec whose `infer` pads to whole blocks (MOSS).
    frames_per_block: int = 0
    n_frames: int = 16
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "codec_stream_driver"
    _codec: Optional[AudioCodecExportConfig] = field(default=None, init=False, repr=False)
    _step: Tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _stages: List[Tuple[int, int, int]] = field(default_factory=list, init=False, repr=False)

    __unchecked__ = {
        "architecture": Unchecked("None: the checkpoint's own `model_type`, as `AudioCodecExportConfig` "
                                  "resolves it, so the GGUF's architecture does not move"),
        "model_dir": Unchecked("path to the codec directory; the recognizer read its `model_type`"),
        "family": Unchecked("which codec this is, from the recognizer that matched it"),
        "root_axis": Unchecked("`n_codes`, the codec family's axis (`AudioCodecExportConfig`)"),
        "chunk_frames": Unchecked("the reference's own `chunked_decode` chunk, for `infer` (ADR-034)"),
        "left_context_frames": Unchecked("the reference's own left context, for `infer` (ADR-034)"),
        "max_frames": Unchecked("`chunk + context`: the one-pass graph is never asked for more"),
        "decomposition": Unchecked("MultiPhase by construction -- three graphs"),
        "driver_script_path": Unchecked("the stream fragment, parsed and checked by LuaFragment"),
        "_codec": Unchecked("the one-pass config this export reuses, built during phases()"),
        "_step": Unchecked("`codec_step`'s state count and attention window, read off the wrapper"),
        "frames_per_block": Unchecked("the one-pass decode's attention block (MOSS: 4), for `infer`"),
        "n_frames": Unchecked("the one-pass decode's trace length"),
        "_stages": Unchecked("each MOSS stage's (positions per frame, window, ring cells), read off the "
                             "stages during phases()"),
    }

    def _one_pass(self) -> AudioCodecExportConfig:
        if self._codec is None:
            self._codec = AudioCodecExportConfig(
                output_path=self.output_path, architecture=self.architecture, model_dir=self.model_dir,
                family=self.family, chunk_frames=self.chunk_frames,
                left_context_frames=self.left_context_frames, max_frames=self.max_frames,
                frames_per_block=self.frames_per_block, n_frames=self.n_frames)
        return self._codec

    def export_architecture(self) -> str:
        return self._one_pass().export_architecture()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        codec = self._one_pass()
        model = codec.load_model()
        width = codec._codes_per_frame
        # The one-pass graph `audio_codec_export` traces, a stochastic codec's noise inputs included (SNAC),
        # each declared as its multiple of the code axis.
        wrapper, dummy, mil_inputs = codec.build_trace(model)
        decode = ExportPhase(
            name="decode",
            wrapper=wrapper.eval(),
            dummy_inputs=dummy,
            mil_inputs=mil_inputs,
            root_axis="n_codes",
            declared_axes={name: {2: f"{multiple}*n_codes"} for name, multiple in codec._noise_inputs.items()},
        )
        if self.family is CodecFamily.MOSS_AUDIO_TOKENIZER:
            return [decode] + self._moss_stages(model, width)
        if self.family in WINDOW_FAMILIES:
            return [decode]
        step = codec_step(model, self.family).eval()
        shapes = step.state_shapes()
        self._step = (len(shapes), step.window)
        delta = check_step(model, self.family, step, width, codec._codebook_size)
        print(f"  codec_step: {len(shapes)} states, {sum(int(np.prod(s)) for s in shapes)} floats, "
              f"{delta:.3g} from the one-call decode")
        n = TRACE_STEP_FRAMES
        return [
            decode,
            ExportPhase(
                name="codec_init",
                wrapper=StateInit(shapes).eval(),
                dummy_inputs=(torch.zeros(1),),
                mil_inputs=[ct.TensorType(name="zero", shape=(1,), dtype=np.float32)],
            ),
            ExportPhase(
                name="codec_step",
                wrapper=step,
                dummy_inputs=(torch.randint(0, codec._codebook_size, (1, n, width)),
                              torch.arange(n, dtype=torch.int32).view(1, -1) + 3,
                              torch.arange(step.window + n, dtype=torch.int32).view(1, -1) + 3 - step.window)
                             + tuple(torch.randn(s) * 0.1 for s in shapes),
                mil_inputs=[
                    ct.TensorType(name="codes", shape=(1, ct.RangeDim(1, MAX_STEP_FRAMES), width),
                                  dtype=np.int32),
                    ct.TensorType(name="positions", shape=(1, ct.RangeDim(1, MAX_STEP_FRAMES)),
                                  dtype=np.int32),
                    ct.TensorType(name="key_positions",
                                  shape=(1, ct.RangeDim(step.window + 1, step.window + MAX_STEP_FRAMES)),
                                  dtype=np.int32),
                ] + [ct.TensorType(name=f"state_{i + 1}", shape=s, dtype=np.float32)
                     for i, s in enumerate(shapes)],
                root_axis="n_codes",
                declared_axes={"positions": {1: "n_codes"},
                               "key_positions": {1: f"n_codes + {step.window}"}},
            ),
        ]

    def _moss_stages(self, model, width: int) -> List[ExportPhase]:
        """MOSS-Audio-Tokenizer's stream: one phase per transformer stage, each fused over a ring KV cache
        of its own (`moss_codec_stream`, loom.cpp ADR-103)."""
        import coremltools as ct

        from .moss_codec_stream import MAX_PUSH_FRAMES, check_stream, stream_stages

        stages = stream_stages(model._loom_decoder)
        delta = check_stream(model._loom_decoder, stages, 24, width, [1, 3, 8, 2, 5])
        self._stages = [(s.positions_per_frame, s.W, s.ring) for s in stages]
        print(f"  {len(stages)} stream stages (positions per frame, window, ring): {self._stages}; "
              f"{delta:.3g} from the one-call decode")
        phases = []
        for k, stage in enumerate(stages):
            stage.eval()
            p = stage.positions_per_frame
            seq = ct.RangeDim(1, MAX_PUSH_FRAMES * p)
            n = TRACE_STEP_FRAMES * p
            if k == 0:
                first = (torch.randint(0, model._loom_decoder.size - 1, (1, n, width)),)
                first_input = ct.TensorType(name="codes", shape=(1, seq, width), dtype=np.int32)
            else:
                channels = stage.input_proj.in_features if hasattr(stage.input_proj, "in_features") else stage.C
                first = (torch.randn(1, n, channels),)
                first_input = ct.TensorType(name="x", shape=(1, seq, channels), dtype=np.float32)
            mask = torch.triu(torch.full((n, n), -1e30), diagonal=1).view(1, 1, n, n)
            phases.append(ExportPhase(
                name=f"stage_{k + 1}",
                wrapper=stage,
                dummy_inputs=first + (torch.arange(n, dtype=torch.int32).view(1, -1), mask),
                mil_inputs=[first_input,
                            ct.TensorType(name="position_ids", shape=(1, seq), dtype=np.int32),
                            ct.TensorType(name="attention_mask", shape=(1, 1, seq, seq), dtype=np.float32)],
                fuse_attention=True,
                kv_cache_size=stage.ring,
                own_kv_cache={"ring": True, "window": stage.W},
            ))
        return phases

    def driver_components(self) -> List:
        """`infer` is the one-pass driver `AudioCodecExportConfig` synthesizes -- the same inputs, the
        same chunk loop -- and the stream's three functions sit beside it at top level."""
        from .driver_components import (CALLER, DEFAULT_NOISE_SEED, ChunkedCodecCall, ComputedCall, DriverInputs,
                                        DriverReturn, ExportConstants, LuaFragment, MonolithicCall,
                                        PaddedCodecCall)
        from .driver_ir import BinOp, Len, Lit
        from .lua_library import LuaLibrary

        codec = self._one_pass()
        # Zero for the one caller with no checkpoint: `component_registry.usage()` builds every
        # registered config without one, and the width comes from the checkpoint's geometry.
        width = codec._codes_per_frame if codec._n_codebooks else 0
        hop = codec._hop_length or 0
        if self.chunk_frames:
            call = ChunkedCodecCall(topology="decode", inputs=("codes",), codes_var="codes",
                                    codes_per_frame=width, hop_length=hop, chunk_frames=self.chunk_frames,
                                    left_context_frames=self.left_context_frames)
            library = [LuaLibrary(uses=("array_slice",))]
        elif self.frames_per_block:
            # One call padded to whole blocks with the absent id, as the flattened export's driver makes it.
            call = PaddedCodecCall(topology="decode", inputs=("codes",), codes_var="codes",
                                   codes_per_frame=width, frames_per_block=self.frames_per_block,
                                   samples_per_frame=hop * codec._channels,
                                   pad_code=codec._absent_code if codec._absent_code is not None else 0)
            library = [LuaLibrary(uses=("array_slice",))]
        else:
            # One call over the whole sequence, as the flattened export's driver makes it.
            call = MonolithicCall(topology="decode", inputs=("codes",),
                                  n_tokens=BinOp("floordiv", Len("codes"), Lit(max(width, 1))), retained=False)
            library = []
        drives = ()
        if self.family is CodecFamily.MOSS_AUDIO_TOKENIZER:
            from .moss_codec_stream import MAX_PUSH_FRAMES

            # Each stage's positions per code frame, window and ring, in order (`_moss_stages`).
            constants = {"CODES_PER_FRAME": width, "MAX_PUSH_FRAMES": MAX_PUSH_FRAMES,
                         "STAGE_PPF": [p for p, _, _ in self._stages],
                         "STAGE_WINDOW": [w for _, w, _ in self._stages],
                         "STAGE_RING": [r for _, _, r in self._stages]}
            fragment = "00_moss_stream.lua"
            # Stages 2.. run in a Lua loop, so their names exist only at run time; declared here so the
            # call is still checked against each traced topology.
            drives = (ComputedCall(topologies=tuple(f"stage_{k + 1}" for k in range(1, len(self._stages))),
                                   inputs=("x", "position_ids", "attention_mask"), written="'stage_' .. k"),)
        elif self.family in WINDOW_FAMILIES:
            margin = WINDOW_FAMILIES[self.family]
            # Zeros for the one caller with no checkpoint (`component_registry.usage()`), which still has to
            # see the stochastic branch: the family says it is stochastic, the checkpoint says how.
            noise = list(codec._noise_multiples or ([0] if self.family is CodecFamily.SNAC else []))
            # A row is one COARSEST-codebook frame (SNAC's spans 4 hops), so that is what a frame's samples are.
            samples = hop * (codec._coarse_stride if codec._vq_strides else 1) * (codec._channels or 1)
            constants = {"CODES_PER_FRAME": width, "SAMPLES_PER_FRAME": samples,
                         "LEFT_FRAMES": margin, "LOOKAHEAD_FRAMES": margin,
                         "NOISE_PER_FRAME": noise, "NOISE_SEED": DEFAULT_NOISE_SEED}
            fragment = "00_window_stream.lua"
            if noise:
                # A stochastic codec's `infer` draws its noise frame-major, as its stream does, so the two
                # agree for a seed (ADR-104); the deterministic ones keep the one call they always made.
                return [
                    ExportConstants(values=constants, top_level=True),
                    LuaFragment(self.driver_script_path / fragment, top_level=True,
                                defines=("stream_open", "stream_feed", "stream_finish", "window_one_pass")),
                    LuaFragment(self.driver_script_path / "01_window_one_pass.lua", defines=("wave",)),
                    DriverReturn(values=("wave",)),
                ]
        else:
            constants = {"CODES_PER_FRAME": width, "CODEC_STATES": self._step[0], "CODEC_WINDOW": self._step[1],
                         "MAX_STEP_FRAMES": MAX_STEP_FRAMES}
            fragment = "00_stream.lua"
        return [
            ExportConstants(values=constants, top_level=True),
            LuaFragment(self.driver_script_path / fragment, top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish"), drives=drives),
            DriverInputs(bindings=(("codes", CALLER),), n_tokens=Len("codes")),
            *library,
            call,
            DriverReturn(values=(call.out_var,)),
        ]

    def hparams(self) -> dict:
        return self._one_pass().hparams()

    def backend_kwargs(self) -> dict:
        # Per-phase weight names: the step and the one-pass decode read the same checkpoint tensors,
        # and each phase packs its own (duplicates are aliased in the file, not stored twice).
        return dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams())
