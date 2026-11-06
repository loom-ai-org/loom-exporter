"""Zipformer (arXiv:2310.11230): a multi-rate transformer encoder with a stateless-prediction transducer,
from `pkufool/zipformer`, which is icefall's `egs/librispeech/ASR/zipformer` as a package without `k2`.

Family 1, but not `BaseTransducerExportConfig`: that template is built around an `nn.LSTM`, and this
checkpoint's prediction network is an embedding and one grouped convolution over the last two labels,
with no state at all. So there are three phases rather than four, and a decode loop of its own in
`zipformer_driver/`.

Non-streaming only. The checkpoint is `causal=False`, and the whole clip goes through once, which is what
loom's one-pass `speech2text` door is. The `-streaming` checkpoints are a different leaf, with six cache
tensors per layer.

**The encoder's modules run as they are, and four things around them do not.** Each is exact rather than
an approximation, and each exists for one reason: the length of the clip has to stay symbolic.

1. **`_KaldiFbank`, the front end.** `torchaudio.compliance.kaldi.fbank` with `snip_edges=False`, which
   reflects the waveform at both ends. `sanm_asr_export` already folds kaldi's per-frame work into one
   strided convolution; what is new here is the reflection, whose two widths are constants.
2. **`_Positions`, for `CompactRelPositionalEncoding`.** The reference caches a table and slices it, under
   a Python branch on the length. The values depend only on the offset, so they are computed.
3. **`_Downsample`, for `SimpleDownsample`.** The reference repeats the last frame by `-T mod ds`, which a
   trace would freeze at its own length. Padding by the most it can ever need, `ds - 1`, and taking the
   weighted sum as a strided convolution gives the same frames: the surplus is never read, because the
   convolution's output length is already `ceil(T / ds)`. This is `KaldiFbankLfrCmvn`'s argument again.
4. **`_Stack`, for `DownsampledZipformerEncoder`.** The reference repeats each frame `ds` times and slices
   the result back to `T`, which is a shape-derived slice. Reading frame `floor(t / ds)` for each `t` in
   `arange(T)` is the same tensor and its length is `T` by construction.
"""
import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .sanm_asr_export import LOG_FLOOR, _frame_matrix
from .spec_protocol import Unchecked

# `pkufool/zipformer`, a git clone on `sys.path`, the resolution the other families without an importable
# package use. Its pip package pins torch below 2.8 and pulls in the training stack. Verified at 8621a77.
ZIPFORMER_REPO = os.environ.get("LOOM_ZIPFORMER_REPO", "/home/flavio/Dev/zipformer")
# Imported by `zipformer.modules` and `zipformer.decode` for training and LM decoding. Nothing on the
# greedy path calls them.
_TRAINING_IMPORTS = ("atdataset", "kaldifst", "simple_sentencepiece")

SAMPLE_RATE = 16000
N_MELS = 80
# The ranges coremltools is told the sample axis takes. The floor is the encoder's own: the subsampling
# and the convnext consume 7 frames before anything comes out. The ceiling is a trace-time statement.
TRACE_SECONDS = 4.0
MIN_SECONDS, MAX_SECONDS = 0.25, 300.0
# Every checkpoint in pkufool's table. `linear_pos` is `heads * pos_head_dim` wide, so it has to be known
# to read the head count.
POS_HEAD_DIM = 4


def import_zipformer():
    """Put the reference checkout on `sys.path`, with its training-only imports stubbed."""
    def _stub_attribute(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

    for name in _TRAINING_IMPORTS:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.__getattr__ = _stub_attribute
            stub.__file__ = f"<stub {name}>"
            sys.modules[name] = stub
    if ZIPFORMER_REPO not in sys.path:
        sys.path.insert(0, ZIPFORMER_REPO)


class _KaldiFbank(nn.Module):
    """`torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, snip_edges=False, high_freq=-400)`
    in ops that trace: `[1, n] -> [1, T, 80]`, `T = (n + hop // 2) // hop`.

    **`snip_edges=False` is a reflection, and both its widths are constants.** Kaldi prepends the first
    `window // 2 - hop // 2` samples reversed (120) and appends the whole waveform reversed, of which the
    last frame can only reach the first 200. With those 200 the convolution's own length,
    `floor((n + 320 - 400) / 160) + 1`, is exactly `T` for every `n`, so nothing has to be counted.

    The reversal is a `gather` on a constant index, over a slice of constant width, and along rows: the
    engine's GET_ROWS reads rows, and a gather along the samples comes out `[120, 120]` wide.
    """

    def __init__(self, n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE, frame_length: float = 25.0,
                 frame_shift: float = 10.0, low_freq: float = 20.0, high_freq: float = -400.0,
                 preemphasis: float = 0.97):
        super().__init__()
        import torchaudio.compliance.kaldi as kaldi

        self.hop = int(sample_rate * frame_shift * 0.001)
        self.window_size = int(sample_rate * frame_length * 0.001)
        self.n_fft = 2 ** (self.window_size - 1).bit_length()
        self.pad_left = self.window_size // 2 - self.hop // 2
        self.pad_right = self.window_size - self.hop // 2 - self.pad_left
        self.register_buffer("reverse_left", torch.arange(self.pad_left - 1, -1, -1))
        self.register_buffer("reverse_right", torch.arange(self.pad_right - 1, -1, -1))

        window = kaldi._feature_window_function("povey", self.window_size, 0.42, torch.device("cpu"),
                                                torch.float64)
        frame_w = _frame_matrix(self.window_size, self.n_fft, window, preemphasis, True)
        self.register_buffer("frame_w", frame_w.reshape(self.n_fft, 1, self.window_size).float())

        col = torch.arange(self.n_fft, dtype=torch.float64)
        row = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64).unsqueeze(1)
        angle = 2.0 * math.pi * row * col / self.n_fft
        self.register_buffer("dft_cos", torch.cos(angle).float())
        self.register_buffer("dft_sin", torch.sin(angle).float())

        banks, _ = kaldi.get_mel_banks(n_mels, self.n_fft, sample_rate, low_freq, high_freq,
                                       100.0, -500.0, 1.0)
        # Kaldi drops the Nyquist bin, which is the trailing zero column.
        self.register_buffer("mel_banks", F.pad(banks, (0, 1), value=0.0).float())

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        left = waveform[:, : self.pad_left].t().index_select(0, self.reverse_left).t()
        right = waveform[:, -self.pad_right :].t().index_select(0, self.reverse_right).t()
        padded = torch.cat((left, waveform, right), dim=1)
        framed = F.conv1d(padded.unsqueeze(1), self.frame_w, stride=self.hop)[0]    # (n_fft, T)
        real = torch.matmul(self.dft_cos, framed)
        imag = torch.matmul(self.dft_sin, framed)
        mel = torch.matmul(self.mel_banks, real * real + imag * imag)
        return torch.clamp(mel, min=LOG_FLOOR).log().transpose(0, 1).unsqueeze(0)


class _Positions(nn.Module):
    """`CompactRelPositionalEncoding` with no cache: `[1, 2T - 1, embed_dim]` for offsets `-(T-1) .. T-1`.

    The reference reads this out of a table it keeps long enough, and rebuilds the table under a Python
    branch on the length. Every row depends only on its own offset, so a table of any length holds the
    same numbers in the same rows, and computing them is exact.
    """

    def __init__(self, embed_dim: int, length_factor: float):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"CompactRelPositionalEncoding needs an even embed_dim, got {embed_dim}")
        self.compression = embed_dim ** 0.5
        self.length_scale = length_factor * embed_dim / (2.0 * math.pi)
        self.register_buffer("freqs", (1 + torch.arange(embed_dim // 2)).to(torch.float32))
        # `pe[:, -1] = 1.0`, the reference's bias column: the last sine, overwritten.
        keep = torch.ones(embed_dim, dtype=torch.float32)
        keep[-1] = 0.0
        self.register_buffer("keep", keep)
        self.register_buffer("bias", 1.0 - keep)

    def forward(self, x: torch.Tensor, left_context_len: int = 0) -> torch.Tensor:
        n = x.shape[0]
        offsets = torch.arange(1 - n, n, device=x.device).to(torch.float32).unsqueeze(1)
        compressed = self.compression * offsets.sign() * (
            (offsets.abs() + self.compression).log() - math.log(self.compression))
        # An outer product as a matmul: the engine broadcasts one operand into the other, never both.
        angles = torch.matmul((compressed / self.length_scale).atan(), self.freqs.unsqueeze(0))
        table = torch.stack((angles.cos(), angles.sin()), dim=2).reshape(2 * n - 1, -1)
        return (table * self.keep + self.bias).unsqueeze(0)


class _Downsample(nn.Module):
    """`SimpleDownsample`: `[T, 1, C] -> [ceil(T / ds), 1, C]`, a learned weighted sum over each `ds`
    frames, as a depthwise strided convolution over the clip padded by the most it could ever need."""

    def __init__(self, downsample: nn.Module, channels: int):
        super().__init__()
        self.ds = int(downsample.downsample)
        weights = downsample.bias.detach().softmax(dim=0)
        self.register_buffer("kernel", weights.repeat(channels, 1, 1))     # (C, 1, ds)
        self.channels = channels

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src.permute(1, 2, 0)                                           # (1, C, T)
        padded = torch.cat((x, x[:, :, -1:].expand(-1, -1, self.ds - 1)), dim=2)
        return F.conv1d(padded, self.kernel, stride=self.ds, groups=self.channels).permute(2, 0, 1)


class _AttentionWeights(nn.Module):
    """`RelPositionMultiheadAttentionWeights.forward` for one clip, with its relative shift spelled as
    NeMo's `rel_shift`.

    Under a trace the reference uses `gather_along_axis` with a `[T, T]` index, which has no engine op.
    Both forms compute `Q[t, k] = P[t, T - 1 - t + k]` from the `[.., T, 2T - 1]` position scores; pad,
    reshape, drop a row, reshape and crop is the one whose lengths the exporter already follows, through
    every Conformer in the zoo. The padding masks are `None` here: one clip has no padding.
    """

    def __init__(self, weights: nn.Module):
        super().__init__()
        self.in_proj = weights.in_proj
        self.linear_pos = weights.linear_pos
        self.num_heads = int(weights.num_heads)
        self.query_head_dim = int(weights.query_head_dim)
        self.pos_head_dim = int(weights.pos_head_dim)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, key_padding_mask=None, attn_mask=None):
        x = self.in_proj(x)
        seq_len, batch_size = x.shape[0], x.shape[1]
        heads, query_dim, pos_dim = self.num_heads, self.query_head_dim, self.pos_head_dim
        width = query_dim * heads
        q = x[..., :width].reshape(seq_len, batch_size, heads, query_dim).permute(2, 1, 0, 3)
        k = x[..., width : 2 * width].reshape(seq_len, batch_size, heads, query_dim).permute(2, 1, 3, 0)
        p = x[..., 2 * width :].reshape(seq_len, batch_size, heads, pos_dim).permute(2, 1, 0, 3)
        scores = torch.matmul(q, k)

        seq_len2 = 2 * seq_len - 1
        pos = self.linear_pos(pos_emb).reshape(-1, seq_len2, heads, pos_dim).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(p, pos)                                  # (h, b, T, 2T - 1)
        padded = F.pad(pos_scores, (1, 0)).view(heads, batch_size, seq_len2 + 1, seq_len)
        shifted = padded[:, :, 1:].view(heads, batch_size, seq_len, seq_len2)[:, :, :, : scores.size(-1)]
        return (scores + shifted).softmax(dim=-1)


class _Stack(nn.Module):
    """`DownsampledZipformerEncoder`: downsample, the stack, upsample, and the bypass back to the input."""

    def __init__(self, stack: nn.Module, channels: int):
        super().__init__()
        self.ds = int(stack.downsample_factor)
        self.downsample = _Downsample(stack.downsample, channels)
        self.encoder = stack.encoder
        self.out_combiner = stack.out_combiner

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        low = self.encoder(self.downsample(src))
        # Frame `floor(t / ds)` for each of `T` frames: `SimpleUpsample` and the slice back to `T`.
        frames = torch.floor(torch.arange(src.shape[0], device=src.device).to(torch.float32) / self.ds)
        # Through a 2-D table, which is the shape GET_ROWS looks up in.
        return self.out_combiner(src, low.squeeze(1).index_select(0, frames.long()).unsqueeze(1))


def _convert_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    """`convert_num_channels`, with `F.pad` rather than a `zeros` of a shape read off the clip."""
    if channels <= x.shape[-1]:
        return x[..., :channels]
    return F.pad(x, (0, channels - x.shape[-1]))


class _ZipformerEncoder(nn.Module):
    """`[1, n] -> [1, T', joiner_dim]`: the front end, `encoder_embed`, the six stacks at their own rates,
    and `encoder_proj`, which runs here once per frame rather than once per symbol in the joiner."""

    def __init__(self, model: nn.Module):
        super().__init__()
        encoder = model.encoder
        self.fbank = _KaldiFbank()
        self.encoder_embed = model.encoder_embed
        self.encoder_dim = tuple(int(d) for d in encoder.encoder_dim)
        stacks = []
        for dim, stack in zip(self.encoder_dim, encoder.encoders):
            inner = stack.encoder if hasattr(stack, "downsample") else stack
            pos = inner.encoder_pos
            inner.encoder_pos = _Positions(pos.embed_dim, pos.length_factor)
            for layer in inner.layers:
                layer.self_attn_weights = _AttentionWeights(layer.self_attn_weights)
            stacks.append(_Stack(stack, dim) if hasattr(stack, "downsample") else stack)
        self.stacks = nn.ModuleList(stacks)
        self.downsample_output = _Downsample(encoder.downsample_output, max(self.encoder_dim))
        self.encoder_proj = model.joiner.encoder_proj

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.fbank(waveform)
        x, _ = self.encoder_embed(features, torch.tensor([features.shape[1]]))
        x = x.permute(1, 0, 2)
        outputs = []
        for dim, stack in zip(self.encoder_dim, self.stacks):
            x = stack(_convert_channels(x, dim))
            outputs.append(x)
        return self.encoder_proj(self.downsample_output(self._full_dim(outputs)).permute(1, 0, 2))

    def _full_dim(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """`Zipformer._get_full_dim_output`: each channel from the last stack that had it."""
        pieces = [outputs[-1]]
        current = self.encoder_dim[-1]
        for i in range(len(outputs) - 2, -1, -1):
            if self.encoder_dim[i] > current:
                pieces.append(outputs[i][..., current : self.encoder_dim[i]])
                current = self.encoder_dim[i]
        return torch.cat(pieces, dim=-1)


class _Decoder(nn.Module):
    """`[context_size] -> [joiner_dim]`: the stateless prediction network and its projection. The
    reference's `Decoder` runs as it is, including the zeros it gives a negative label."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.decoder = model.decoder
        self.decoder_proj = model.joiner.decoder_proj

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.decoder_proj(self.decoder(context.unsqueeze(0), need_pad=False))[0, 0]


class _Joiner(nn.Module):
    """`Joiner.forward(..., project_input=False)` on one frame: both projections already ran."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.output_linear = model.joiner.output_linear

    def forward(self, encoder_frame: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        return self.output_linear(torch.tanh(encoder_frame + decoder_out))


def _read_architecture(state: dict) -> dict:
    """`AsrModel`'s keyword arguments, read off the weights: the checkpoints ship no configuration, and
    the variants differ in exactly these. What is not here has one value across pkufool's table.

    `encoder_unmasked_dim` is not in the weights, and only `get_feature_masks` reads it, in training: it
    is set to the narrowest stack's width, which any value at or below it satisfies."""
    def stack_key(i: int, rest: str) -> str:
        wrapped = f"encoder.encoders.{i}.encoder.{rest}"
        return wrapped if wrapped in state else f"encoder.encoders.{i}.{rest}"

    n_stacks = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("encoder.encoders."))
    dims, layers, kernels, heads, feedforward, factors = [], [], [], [], [], []
    for i in range(n_stacks):
        dims.append(state[stack_key(i, "layers.0.bypass_scale")].shape[0])
        prefixes = (f"encoder.encoders.{i}.encoder.layers.", f"encoder.encoders.{i}.layers.")
        layers.append(1 + max(int(k.split("layers.")[1].split(".")[0]) for k in state if k.startswith(prefixes)))
        depthwise = state.get(stack_key(i, "layers.0.conv_module1.depthwise_conv.weight"))
        if depthwise is None:
            raise ValueError("this checkpoint's convolutions are chunk-causal: it is a `-streaming` "
                             "checkpoint, which this non-streaming export does not take")
        kernels.append(depthwise.shape[-1])
        heads.append(state[stack_key(i, "layers.0.self_attn_weights.linear_pos.weight")].shape[0] // POS_HEAD_DIM)
        feedforward.append(state[stack_key(i, "layers.0.feed_forward2.out_proj.weight")].shape[1])
        bias = state.get(f"encoder.encoders.{i}.downsample.bias")
        factors.append(1 if bias is None else bias.shape[0])
    as_list = lambda values: ",".join(str(int(v)) for v in values)
    return dict(
        encoder_dim=as_list(dims), num_encoder_layers=as_list(layers), downsampling_factor=as_list(factors),
        cnn_module_kernel=as_list(kernels), num_heads=as_list(heads), feedforward_dim=as_list(feedforward),
        encoder_unmasked_dim=as_list([min(dims)] * n_stacks),
        query_head_dim="32", value_head_dim="12", pos_head_dim=str(POS_HEAD_DIM),
        pos_dim=state[stack_key(0, "layers.0.self_attn_weights.linear_pos.weight")].shape[1],
        vocab_size=state["joiner.output_linear.weight"].shape[0],
        decoder_dim=state["decoder.embedding.weight"].shape[1],
        joiner_dim=state["joiner.output_linear.weight"].shape[1],
        context_size=state["decoder.conv.weight"].shape[-1],
        use_ctc="ctc_output.1.weight" in state, use_transducer=True, causal=False,
        chunk_size="-1", left_context_frames="-1",
    )


@dataclass
class ZipformerExportConfig(BaseMultiPhaseModelExportConfig):
    """A non-streaming `pkufool/zipformer-*` checkpoint directory: `model.pt` and `data/tokens.txt`."""

    model_dir: str = ""
    architecture: str = "zipformer"
    output_path: str = "zipformer.gguf"
    root_axis: str = "n_samples"
    driver_script_path: Path = Path(__file__).resolve().parent / "zipformer_driver"
    decomposition: Decomposition = field(default_factory=lambda: MultiPhase())
    # Filled during `phases()`, READ off the checkpoint.
    blank_id: int = field(default=0, init=False, repr=False)
    context_size: int = field(default=2, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked("the path this leaf's loader is pointed at, established by `_is_zipformer`; "
                               "`load_model` raises on anything it cannot restore"),
        "architecture": Unchecked("the GGUF's own architecture string; there is no second authority"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the encoder ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("the hand-written decode loop, cross-checked against the traced "
                                        "topologies by its own parsed call sites"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "blank_id": Unchecked("READ off `data/tokens.txt`, whose `<blk>` row it is, during phases()"),
        "context_size": Unchecked("READ off the decoder convolution's kernel during phases()"),
    }

    def load_model(self):
        import_zipformer()
        from zipformer.modules.model import AsrModel

        state = torch.load(Path(self.model_dir) / "model.pt", map_location="cpu", weights_only=True)["model"]
        model = AsrModel(**_read_architecture(state))
        model.load_state_dict(state, strict=True)
        # `ConvNeXt.forward_internal` branches on `x.requires_grad`, into a whitener that does nothing here,
        # and the trace and torch's re-trace check took the branch differently.
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

        n_samples = int(TRACE_SECONDS * SAMPLE_RATE)
        seq_dim = ct.RangeDim(int(MIN_SECONDS * SAMPLE_RATE), int(MAX_SECONDS * SAMPLE_RATE))
        return [
            ExportPhase(name="encoder", wrapper=_ZipformerEncoder(model).eval(),
                        dummy_inputs=(torch.randn(1, n_samples, dtype=torch.float32) * 0.1,),
                        mil_inputs=[ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32)],
                        root_axis=self.root_axis),
            ExportPhase(name="decoder", wrapper=_Decoder(model).eval(),
                        dummy_inputs=(torch.tensor([-1] * (self.context_size - 1) + [self.blank_id],
                                                   dtype=torch.int64),),
                        mil_inputs=[ct.TensorType(name="context", shape=(self.context_size,), dtype=np.int32)]),
            ExportPhase(name="joint", wrapper=_Joiner(model).eval(),
                        dummy_inputs=(torch.randn(joiner_dim), torch.randn(joiner_dim)),
                        mil_inputs=[ct.TensorType(name="encoder_frame", shape=(joiner_dim,), dtype=np.float32),
                                    ct.TensorType(name="decoder_out", shape=(joiner_dim,), dtype=np.float32)]),
        ]

    def driver_components(self) -> List:
        from .driver_components import ExportConstants, LuaFragment, SubgraphCallComponent
        from .driver_ir import Len, Var

        fragment = self.driver_script_path
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            ExportConstants(values={"BLANK_ID": self.blank_id, "CONTEXT_SIZE": self.context_size}),
            LuaFragment(fragment / "01_inputs.lua", defines=("_waveform",)),
            # Retained and binding nothing: the decode loop names one row per call.
            SubgraphCallComponent(topology="encoder", outputs=(), retain=True,
                                  inputs={"waveform": Var("_waveform")}, length=Len("_waveform")),
            LuaFragment(fragment / "02_decode.lua", reads=("BLANK_ID", "CONTEXT_SIZE"), defines=("tokens",)),
            LuaFragment(fragment / "03_return.lua", reads=("tokens",)),
        ]

    def hparams(self) -> dict:
        return {"sample_rate": SAMPLE_RATE}

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=True, root_axis=self.root_axis, hparams=self.hparams(),
                    tokenizer_dir=self.model_dir, tokenizer_family="zipformer")


def _read_tokens(model_dir: Path) -> List[str]:
    """`data/tokens.txt`, `piece id` per line, as an id-indexed list."""
    rows = {}
    for line in (model_dir / "data" / "tokens.txt").read_text(encoding="utf-8").splitlines():
        piece, _, index = line.rpartition(" ")
        rows[int(index)] = piece
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("data/tokens.txt does not number its rows 0..N-1")
    return [rows[i] for i in range(len(rows))]


def _is_zipformer(path: Path) -> bool:
    """A `pkufool/zipformer-*` directory: its layout, and a transducer's prediction network in `model.pt`."""
    if not (path.is_dir() and (path / "model.pt").is_file() and (path / "data" / "tokens.txt").is_file()):
        return False
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model", state) if isinstance(state, dict) else {}
    return "encoder.encoders.0.layers.0.bypass_scale" in state and "decoder.conv.weight" in state


def _build_zipformer(path: Path, output_path: str) -> ZipformerExportConfig:
    return ZipformerExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ZipformerExportConfig,
        recognizers=[ModelRecognizer(name="zipformer", detect=_is_zipformer, build_config=_build_zipformer)],
    ))
