"""Zipformer streaming (`pkufool/zipformer-*-streaming`): the chunk-causal checkpoints of `zipformer_export`'s
family, run the way upstream's `zipformer inference --streaming 1` runs them.

**One call per chunk, with the encoder's state carried by the driver.** The checkpoint is trained causal, and
upstream decodes it chunk by chunk: 64 feature frames (640 ms) plus 13 of lookahead per call, through
`StreamingEncoderWrapper`, with 74 state tensors in and 74 out (six caches per layer, the subsampler's left
pad, and the count of frames processed). The `encoder_chunk` phase is that call with every shape fixed, and
`zipformer_streaming_driver/` feeds each call's states to the next as retained outputs, so they never leave
the engine. Clip length only changes how many calls there are: cost is linear in it and memory is bounded,
which the non-streaming leaf's attention over every frame is not.

The clip still enters through the one-pass `speech2text` door: the whole waveform, one transcript. A live
door (audio pushed in, partials read out) is later work; the driver is split into init / feed / finish so
that it can call the same pieces.

**Upstream's latency settings are export constants, not checkpoint facts.** `model.pt` is shared by its
three jit files (chunk 16 / left 64, 32 / 128, 64 / 256); this export fixes 32 / 128, the middle one.

What runs as upstream wrote it: `encoder_embed.streaming_forward` and each stack's `streaming_forward`, at
fixed shapes, so every Python branch on a length is a constant. What does not:

1. **The relative shift** (`_StreamingAttentionWeights`). Under a trace the reference takes a `gather` with
   a `[T, T + L]` index, which has no engine op; NeMo's pad-and-reshape `rel_shift` gives the same
   `Q[t, k] = P[t, T - 1 - t + k]` for any score width of at least `2T - 1`.
2. **`processed_lens` is F32.** The reference's is int32 and only ever compared with `arange(L)` and
   incremented by the chunk; both are exact in F32 below 2^24 frames (93 hours).
3. **The fbank** is `zipformer_export._KaldiFbank` over the whole clip, as upstream's `_extract_features`
   computes it, followed by the `log(1e-10)` frames upstream pads with: every chunk is then a row range of
   one retained output.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked
from .zipformer_export import (
    JOINT_BLOCK, MAX_SECONDS, MIN_SECONDS, SAMPLE_RATE, TRACE_SECONDS, _Decoder, _Joiner, _KaldiFbank,
    _read_architecture, _read_tokens, import_zipformer,
)

# Upstream's middle latency: 32 frames after the 2x subsampler per chunk (64 feature frames, 640 ms), and
# 128 frames (2.56 s) of attention to the left.
CHUNK_SIZE = 32
LEFT_CONTEXT = 128
# `StreamingEncoderWrapper.pad_length`: the subsampler's 7 frames and the ConvNeXt's 3 at half rate.
PAD_LENGTH = 7 + 2 * 3
CHUNK_FEATURES = 2 * CHUNK_SIZE
TAIL = CHUNK_FEATURES + PAD_LENGTH
# `DecodeStream.LOG_EPS`, the value upstream pads features with: the clip's end, and every chunk past it.
LOG_EPS = math.log(1e-10)
# Rows of LOG_EPS after the clip's features: PAD_LENGTH, then enough that the last chunk, which starts
# before `T + PAD_LENGTH`, reads `TAIL` rows without running off the end.
FEATURE_TAIL = PAD_LENGTH + TAIL


def read_streaming_architecture(state: dict) -> dict:
    """`_read_architecture` for a chunk-causal checkpoint, whose depthwise convolutions are a causal half
    and a chunkwise full kernel; the full one is the configured `cnn_module_kernel`."""
    patched = dict(state)
    for key in state:
        if key.endswith("depthwise_conv.chunkwise_conv.weight"):
            patched[key.replace("chunkwise_conv.weight", "weight")] = state[key]
    arch = _read_architecture(patched)
    arch.update(causal=True, chunk_size=str(CHUNK_SIZE), left_context_frames=str(LEFT_CONTEXT))
    return arch


class _StreamingAttentionWeights(nn.Module):
    """`RelPositionMultiheadAttentionWeights.streaming_forward`, with the relative shift as NeMo's
    `rel_shift`. The scores are `[.., T, 2T - 1 + L]` against `T + L` keys; padding one column on the left
    and reading the result as `[2T - 1 + L + 1, T]` with its first row dropped puts `P[t, T - 1 - t + k]` at
    `[t, k]` for every `k < T + L`, the reference's `as_strided` and its traced `gather` alike."""

    def __init__(self, weights: nn.Module):
        super().__init__()
        self.in_proj = weights.in_proj
        self.linear_pos = weights.linear_pos
        self.num_heads = int(weights.num_heads)
        self.query_head_dim = int(weights.query_head_dim)
        self.pos_head_dim = int(weights.pos_head_dim)

    def streaming_forward(self, x, pos_emb, cached_key, left_context_len: int, key_padding_mask):
        x = self.in_proj(x)
        seq_len, batch_size = x.shape[0], x.shape[1]
        heads, query_dim, pos_dim = self.num_heads, self.query_head_dim, self.pos_head_dim
        width = query_dim * heads
        k = torch.cat([cached_key, x[..., width : 2 * width]], dim=0)
        cached_key = k[-left_context_len:]
        k_len = k.shape[0]
        q = x[..., :width].reshape(seq_len, batch_size, heads, query_dim).permute(2, 1, 0, 3)
        k = k.reshape(k_len, batch_size, heads, query_dim).permute(2, 1, 3, 0)
        p = x[..., 2 * width :].reshape(seq_len, batch_size, heads, pos_dim).permute(2, 1, 0, 3)
        scores = torch.matmul(q, k)

        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos = self.linear_pos(pos_emb).reshape(-1, seq_len2, heads, pos_dim).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(p, pos)                                  # (h, b, T, 2T - 1 + L)
        padded = F.pad(pos_scores, (1, 0)).view(heads, batch_size, seq_len2 + 1, seq_len)
        shifted = padded[:, :, 1:].reshape(heads, batch_size, seq_len, seq_len2)[:, :, :, :k_len]
        scores = (scores + shifted).masked_fill(key_padding_mask.unsqueeze(1), -1000)
        return scores.softmax(dim=-1), cached_key


def key_positions(ds: int) -> torch.Tensor:
    """The key padding mask of a stack at rate `ds` is `key_positions(ds) >= processed_lens`.

    Cache slot `j` has position `L - 1 - j` and holds a frame once more than that many have been processed;
    the chunk's own keys get -1 and are never masked, since every chunk is whole (upstream pads the last to
    TAIL). One comparison rather than the reference's flip and concatenation, since the engine's CONCAT
    takes no boolean operand from a comparison; and one table per stack, already at its rate, since the
    reference's `mask[..., ::ds]` is a strided slice, which has no lowering."""
    positions = torch.cat([torch.arange(LEFT_CONTEXT - 1, -1, -1), torch.full((CHUNK_SIZE,), -1)])
    return positions[::ds].to(torch.float32)


class _StreamingEncoder(nn.Module):
    """`StreamingEncoderWrapper.forward` for one chunk, then `downsample_output`'s frames through
    `encoder_proj`: `[TAIL, 80]` features and the 74 states -> `[1, CHUNK_SIZE / 2, joiner_dim]` and the 74
    new states, in `get_init_states`' order with `processed_lens` last."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder_embed = model.encoder_embed
        self.encoder = model.encoder
        for stack in self.encoder.encoders:
            inner = stack.encoder if hasattr(stack, "downsample") else stack
            for layer in inner.layers:
                layer.self_attn_weights = _StreamingAttentionWeights(layer.self_attn_weights)
        self.encoder_proj = model.joiner.encoder_proj
        self.key_positions = nn.ParameterList(
            nn.Parameter(key_positions(int(ds)), requires_grad=False) for ds in self.encoder.downsampling_factor)

    def forward(self, features: torch.Tensor, *states: torch.Tensor):
        encoder = self.encoder
        embed_left_pad, processed_lens = states[-2], states[-1]
        x, _, new_embed_left_pad = self.encoder_embed.streaming_forward(
            features.unsqueeze(0), torch.tensor([TAIL]), embed_left_pad)
        x = x.permute(1, 0, 2)

        outputs, new_states, offset = [], [], 0
        for i, module in enumerate(encoder.encoders):
            ds = encoder.downsampling_factor[i]
            n = module.num_layers
            x = _convert_channels(x, encoder.encoder_dim[i])
            x, layer_states = module.streaming_forward(
                x, states=list(states[offset * 6 : (offset + n) * 6]), left_context_len=LEFT_CONTEXT // ds,
                src_key_padding_mask=(self.key_positions[i] >= processed_lens).unsqueeze(0))
            offset += n
            outputs.append(x)
            new_states += layer_states
        x = encoder.downsample_output(encoder._get_full_dim_output(outputs))
        # `[1, T, C]`, the non-streaming encoder's layout, whose rows the joint names. `x[:, 0]` lowers to a
        # view that keeps the batch axis between the two, and the engine retained one row of 16 frames.
        out = self.encoder_proj(x.permute(1, 0, 2))
        return (out, *new_states, new_embed_left_pad, processed_lens + CHUNK_SIZE)


def _convert_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if channels <= x.shape[-1]:
        return x[..., :channels]
    return F.pad(x, (0, channels - x.shape[-1]))


class _Features(nn.Module):
    """`[1, n] -> [T + FEATURE_TAIL, 80]`: the clip's kaldi fbank and the LOG_EPS frames upstream pads it
    with, so every chunk the driver feeds is a row range of this one output."""

    def __init__(self):
        super().__init__()
        self.fbank = _KaldiFbank()
        self.register_buffer("tail", torch.full((FEATURE_TAIL, 80), LOG_EPS))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fbank(waveform)[0], self.tail], dim=0)


def state_shapes(model: nn.Module) -> List[tuple]:
    """The 74 state shapes, from the reference's own `get_init_states`, `processed_lens` as F32 `[1]`."""
    shapes = [tuple(t.shape) for t in model.encoder.get_init_states()]
    shapes.append(tuple(model.encoder_embed.get_init_states().shape))
    shapes.append((1,))
    return shapes


def state_names(model: nn.Module) -> List[str]:
    """`state_1` .. `state_N`, in `state_shapes`' order: the driver builds the names from the count, so one
    chunk loop serves every size. Output `k + 1` of `encoder_chunk` is `state_k`'s next value."""
    return [f"state_{i + 1}" for i in range(len(state_shapes(model)))]


@dataclass
class ZipformerStreamingExportConfig(BaseMultiPhaseModelExportConfig):
    """A `pkufool/zipformer-*-streaming` checkpoint directory: `model.pt` and `data/tokens.txt`."""

    model_dir: str = ""
    architecture: str = "zipformer_streaming"
    output_path: str = "zipformer-streaming.gguf"
    root_axis: str = "n_samples"
    driver_script_path: Path = Path(__file__).resolve().parent / "zipformer_streaming_driver"
    decomposition: Decomposition = field(default_factory=lambda: MultiPhase())
    # Filled during `phases()`, READ off the checkpoint.
    blank_id: int = field(default=0, init=False, repr=False)
    context_size: int = field(default=2, init=False, repr=False)
    state_sizes: List[int] = field(default_factory=list, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked("the path this leaf's loader is pointed at, established by "
                               "`_is_zipformer_streaming`; `load_model` raises on anything it cannot restore"),
        "architecture": Unchecked("the GGUF's own architecture string; there is no second authority"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the fbank ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("the hand-written chunk loop, cross-checked against the traced "
                                        "topologies by its own parsed call sites"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "blank_id": Unchecked("READ off `data/tokens.txt`, whose `<blk>` row it is, during phases()"),
        "context_size": Unchecked("READ off the decoder convolution's kernel during phases()"),
        "state_sizes": Unchecked("READ off the reference's `get_init_states` during phases()"),
    }

    def load_model(self):
        import_zipformer()
        from zipformer.modules.model import AsrModel

        state = torch.load(Path(self.model_dir) / "model.pt", map_location="cpu", weights_only=True)["model"]
        model = AsrModel(**read_streaming_architecture(state))
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False)

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        pieces = _read_tokens(Path(self.model_dir))
        if pieces[0] != "<blk>" or len(pieces) != model.joiner.output_linear.out_features:
            raise ValueError(f"data/tokens.txt has {len(pieces)} rows starting {pieces[0]!r}; the joiner "
                             f"emits {model.joiner.output_linear.out_features} and blank is row 0")
        self.blank_id = 0
        self.context_size = int(model.decoder.context_size)
        joiner_dim = int(model.joiner.output_linear.in_features)
        shapes, names = state_shapes(model), state_names(model)
        self.state_sizes = [int(np.prod(s)) for s in shapes]

        n_samples = int(TRACE_SECONDS * SAMPLE_RATE)
        seq_dim = ct.RangeDim(int(MIN_SECONDS * SAMPLE_RATE), int(MAX_SECONDS * SAMPLE_RATE))
        states = tuple(torch.randn(s) * 0.1 for s in shapes[:-1]) + (torch.tensor([float(LEFT_CONTEXT)]),)
        return [
            ExportPhase(name="fbank", wrapper=_Features().eval(),
                        dummy_inputs=(torch.randn(1, n_samples, dtype=torch.float32) * 0.1,),
                        mil_inputs=[ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32)],
                        root_axis=self.root_axis),
            ExportPhase(name="encoder_chunk", wrapper=_StreamingEncoder(model).eval(),
                        dummy_inputs=(torch.randn(TAIL, 80),) + states,
                        mil_inputs=[ct.TensorType(name="features", shape=(TAIL, 80), dtype=np.float32)]
                        + [ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in zip(names, shapes)]),
            ExportPhase(name="decoder", wrapper=_Decoder(model).eval(),
                        dummy_inputs=(torch.tensor([self.blank_id] * self.context_size, dtype=torch.int64),),
                        mil_inputs=[ct.TensorType(name="context", shape=(self.context_size,), dtype=np.int32)]),
            ExportPhase(name="joint", wrapper=_Joiner(model).eval(),
                        dummy_inputs=(torch.randn(JOINT_BLOCK, joiner_dim), torch.randn(joiner_dim)),
                        mil_inputs=[ct.TensorType(name="encoder_frame", dtype=np.float32,
                                                  shape=(ct.RangeDim(1, JOINT_BLOCK), joiner_dim)),
                                    ct.TensorType(name="decoder_out", shape=(joiner_dim,), dtype=np.float32)],
                        root_axis="n_enc_frames"),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        constants = {
            "BLANK_ID": self.blank_id, "CONTEXT_SIZE": self.context_size, "JOINT_BLOCK": JOINT_BLOCK,
            "CHUNK_FEATURES": CHUNK_FEATURES, "TAIL": TAIL, "PAD_LENGTH": PAD_LENGTH,
            "FEATURE_TAIL": FEATURE_TAIL, "STATE_SIZES": self.state_sizes,
        }
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("waveform", CALLER),), n_tokens=Len("waveform")),
            LuaFragment(self.driver_script_path / "01_transcribe.lua",
                        reads=("waveform",) + tuple(constants), defines=("tokens",)),
            DriverReturn(values=("tokens",)),
        ]

    def hparams(self) -> dict:
        return {"sample_rate": SAMPLE_RATE}

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=True, root_axis=self.root_axis, hparams=self.hparams(),
                    tokenizer_dir=self.model_dir, tokenizer_family="zipformer")


def _is_zipformer_streaming(path: Path) -> bool:
    """A `pkufool/zipformer-*-streaming` directory: the non-streaming leaf's layout, with chunk-causal
    depthwise convolutions."""
    if not (path.is_dir() and (path / "model.pt").is_file() and (path / "data" / "tokens.txt").is_file()):
        return False
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model", state) if isinstance(state, dict) else {}
    return ("encoder.encoders.0.layers.0.bypass_scale" in state and "decoder.conv.weight" in state
            and "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.chunkwise_conv.weight" in state)


def _build_zipformer_streaming(path: Path, output_path: str) -> ZipformerStreamingExportConfig:
    return ZipformerStreamingExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ZipformerStreamingExportConfig,
        recognizers=[ModelRecognizer(name="zipformer_streaming", detect=_is_zipformer_streaming,
                                     build_config=_build_zipformer_streaming)],
    ))
