"""Moonshine Streaming -- Useful Sensors' `moonshine_streaming` (tiny, small): a sliding-window encoder
over the raw waveform feeding a RoPE decoder that cross-attends to it. Family 2's shape -- an `encoder`
run once, a `cross_kv` phase that projects its output into every decoder layer's cross-attention K/V
once, and a KV-cached `decoder` step in a loop -- so `whisper_export` and `canary_export` are the
modules to read beside this one.

What is Moonshine's own:

* **The front end is in the graph and needs no STFT.** The waveform is cut into 5 ms frames (80 samples
  at 16 kHz), each normalised on its own (`FrameCMVN`), compressed with `asinh(k x)`, projected, and
  taken to 50 Hz by two causal stride-2 convolutions. `asinh` has no engine primitive, so it is spelled
  `sign(x) log(|x| + sqrt(x^2 + 1))` (`_asinh`); the frame is a reshape, so the driver hands the graph
  a whole number of frames (see below).
* **The encoder attends through a sliding window per layer** -- `(16, 4)` (16 frames back, 3 ahead) on
  the first two and last two layers, `(16, 0)` between -- built in the graph from the frame count, so
  one topology covers every length. transformers applies the windows ONLY when the call carries an
  `attention_mask`; the model card's own usage passes the processor's, and without one every layer
  attends to the whole clip, which is not the model the card describes. The export always windows.
* **Position enters after the encoder** ("ergodic" encoder): a learned table of 4096 rows is added to
  the encoder output and projected to the decoder's width. Both are functions of the encoder alone, so
  they end the `encoder` phase; transformers does them inside the DECODER's forward, IN PLACE on the
  tensor it was handed, which is why the decoder step below does not call that forward.
* **The decoder's RoPE is interleaved and partial** (`rotate_half` pairs dims `2i, 2i+1`; 32 of 40 or 64
  head dims rotate). transformers spells it with `repeat_interleave` and stride-2 slices; the step here
  multiplies by the same cos/sin with the frequencies already interleaved, and rotates with a constant
  +-1 permutation matrix -- exact, one nonzero per column (`_InterleavedRope`).

**The partial last frame.** The processor zero-pads a clip to a multiple of 80 samples and masks the
padded frame; a masked frame contributes exactly zero (its embedding is zeroed), and an all-zero frame
embeds to exactly zero anyway (CMVN of zeros is zero, the projection has no bias, `silu(0) = 0`). So
the driver zeroes the tail of the last partial frame and pads it to a whole one, and the graph needs no
mask: every encoder row it produces is one transformers would have let the decoder see.

**Decoding is the model card's**: greedy from `<s>`, ended by `</s>`, and capped at `6.5` tokens per
second of audio ("to avoid hallucination loops"), `max_length` counting the start token.
"""
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
from .spec_protocol import Unchecked

# The model card's token budget: `max_length = int(n_samples * 6.5 / sample_rate)`, start token included.
TOKENS_PER_SECOND = 6.5
# What a masked attention score becomes. transformers fills `finfo.min`; anything `exp` takes to exactly
# 0 after the row max is subtracted is the same softmax, and every row keeps its own diagonal.
_MASKED_SCORE = -1e30


def _asinh(x):
    """`asinh`, which the engine has no primitive for. Odd, so the magnitude goes through the log and
    the sign is put back: `log(x + sqrt(x^2 + 1))` on its own cancels catastrophically for x << 0."""
    a = torch.abs(x)
    return torch.sign(x) * torch.log(a + torch.sqrt(a * a + 1.0))


def window_mask(n: int, window, dtype=torch.float32, device=None):
    """`sliding_window_mask_function(window)` as an additive `[1, 1, n, n]` mask over a length the graph
    reads off its own input. transformers' predicate is `0 <= d < left` or `0 < -d < right` (`d = q - k`);
    for the windows these checkpoints use that is the single band `-max(right, 1) < d < left`, which
    needs no `logical_or` (the engine lowers `and` as a product and has no `or`)."""
    left, right = int(window[0]), int(window[1])
    # `q - k` for every pair as ONE matmul, `[q, 1] @ [1, -k]^T` -- the engine's elementwise ops
    # broadcast one operand into the other, never both, so `idx[:, None] - idx[None, :]` aborts at run
    # time (SUB: incompatible shapes [1, n] and [n, 1]). Integers below 2^24, so the product is exact.
    idx = torch.arange(n, device=device).to(dtype)
    ones = idx * 0.0 + 1.0
    d = torch.matmul(torch.stack([idx, ones], dim=1), torch.stack([ones, -idx], dim=0))
    allowed = (d > -max(right, 1)) & (d < left)
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), _MASKED_SCORE, dtype=dtype)
                       ).view(1, 1, n, n)


class _MoonshineEncoderWrapper(nn.Module):
    """`waveform [1, 80 * n_frames] -> [1, n_enc, d_decoder]`: the front end, the windowed layers, the
    final norm, the decoder's position table and its projection."""

    def __init__(self, model):
        super().__init__()
        encoder = model.model.encoder
        decoder = model.model.decoder
        self.embedder = encoder.embedder
        self.layers = encoder.layers
        self.final_norm = encoder.final_norm
        self.windows = [tuple(w) for w in encoder.config.sliding_windows]
        self.frame_len = int(self.embedder.frame_len)
        self.pos_emb = decoder.pos_emb
        self.proj = decoder.proj

    def forward(self, waveform):
        emb = self.embedder
        frames = waveform.reshape(1, -1, self.frame_len)
        mean = frames.mean(dim=-1, keepdim=True)
        centered = frames - mean
        hidden = centered / (centered.pow(2).mean(dim=-1, keepdim=True) + emb.cmvn.eps).sqrt()
        hidden = _asinh(torch.exp(emb.comp.log_k) * hidden)
        hidden = F.silu(emb.linear(hidden)).transpose(1, 2)
        hidden = F.silu(emb.conv1(hidden)[0])
        hidden = emb.conv2(hidden)[0].transpose(1, 2)
        n = hidden.shape[1]
        masks = {w: window_mask(n, w, hidden.dtype) for w in sorted(set(self.windows))}
        for layer, window in zip(self.layers, self.windows):
            hidden = layer(hidden, attention_mask=masks[window])
        hidden = self.final_norm(hidden)
        hidden = hidden + self.pos_emb(torch.arange(n))
        return self.proj(hidden)


class _MoonshineCrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`: every decoder layer's cross-attention K/V, once -- a function of
    the encoder alone, so the decoder step takes them as inputs rather than projecting per token."""

    def __init__(self, decoder_layers):
        super().__init__()
        self.projs = nn.ModuleList()
        for layer in decoder_layers:
            self.projs.append(layer.encoder_attn.k_proj)
            self.projs.append(layer.encoder_attn.v_proj)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


def cross_kv_input_names(n_layers: int) -> tuple:
    names = []
    for i in range(n_layers):
        names.append(f"xk_{i}")
        names.append(f"xv_{i}")
    return tuple(names)


class _InterleavedRope:
    """`apply_rotary_pos_emb` for one attention module, without `repeat_interleave` or stride-2 slices.

    transformers computes `cos(p * f_i)` and repeats each entry twice; `inv_freq` interleaved once at
    export gives the same numbers from one product. `rotate_half` sends `(x_{2i}, x_{2i+1})` to
    `(-x_{2i+1}, x_{2i})`, which is `x @ R` for a constant `R` with one +-1 per column."""

    def __init__(self, inv_freq: torch.Tensor):
        self.freq = inv_freq.repeat_interleave(2).clone()
        dim = self.freq.numel()
        rot = torch.zeros(dim, dim, dtype=self.freq.dtype)
        for i in range(dim // 2):
            rot[2 * i + 1, 2 * i] = -1.0
            rot[2 * i, 2 * i + 1] = 1.0
        self.rot = rot

    def cos_sin(self, position_ids, dtype):
        # In f32 whatever the model's dtype, as transformers forces it (`maybe_autocast(enabled=False)`
        # around a `.float()` product), then cast.
        # An outer product, as transformers spells it (`inv_freq @ position_ids`): `[t, 1] * [32]` is a
        # broadcast of BOTH operands, which the engine's MUL refuses -- invisible in the decode loop,
        # whose steps are one token each, and an abort on any longer prefill (found by the probe's
        # teacher-forced pass). One factor per product, so the matmul is exact.
        angle = torch.matmul(position_ids.to(torch.float32).unsqueeze(-1),
                             self.freq.to(torch.float32).unsqueeze(0))
        return (torch.cos(angle).to(dtype).unsqueeze(1), torch.sin(angle).to(dtype).unsqueeze(1))

    def apply(self, x, cos, sin):
        dim = self.freq.numel()
        x_rot, x_pass = x[..., :dim], x[..., dim:]
        return torch.cat([x_rot * cos + torch.matmul(x_rot, self.rot) * sin, x_pass], dim=-1)


def _self_attention(attn, rope, hidden, cos, sin, mask):
    """`MoonshineStreamingAttention.forward` for self-attention with no cache, its RoPE spelled by
    `_InterleavedRope` -- the pattern `fuse_loom_attention` matches."""
    bsz, q_len = hidden.shape[:-1]
    heads, head_dim = attn.config.num_key_value_heads, attn.head_dim
    q = attn.q_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
    # The scale on Q, not on the scores: `fuse_loom_attention` matches `add(matmul(q * s, k^T), mask)`,
    # which is how transformers' SDPA path traces and the form the engine's ATTENTION node takes.
    scores = torch.matmul(q * attn.scaling, k.transpose(2, 3)) + mask
    out = torch.matmul(F.softmax(scores, dim=-1), v)
    return attn.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


def _cross_attention(attn, hidden, xk, xv):
    bsz, q_len = hidden.shape[:-1]
    heads, head_dim = attn.config.num_key_value_heads, attn.head_dim
    q = attn.q_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    k = xk.view(bsz, -1, heads, head_dim).transpose(1, 2)
    v = xv.view(bsz, -1, heads, head_dim).transpose(1, 2)
    out = torch.matmul(F.softmax(torch.matmul(q, k.transpose(2, 3)) * attn.scaling, dim=-1), v)
    return attn.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


class _MoonshineDecoderWrapper(nn.Module):
    """`(tokens, position_ids, attention_mask, xk_0, xv_0, ...) -> logits`.

    `MoonshineStreamingDecoder.forward` is not called: it adds the encoder position table to
    `encoder_hidden_states` IN PLACE (here that tensor would be `xk_0`) and builds its masks from Python
    shapes. What runs is each layer's parts in its order -- pre-norm self-attention, pre-norm
    cross-attention, pre-norm gated MLP -- the final norm and the output projection."""

    def __init__(self, model):
        super().__init__()
        decoder = model.model.decoder
        if decoder.config.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError(f"moonshine-streaming: rope_type "
                             f"{decoder.config.rope_parameters['rope_type']!r}; only the default "
                             f"(unscaled) RoPE is reproduced.")
        self.embed_tokens = decoder.embed_tokens
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.proj_out = model.proj_out
        self._rope = _InterleavedRope(decoder.rotary_emb.inv_freq.detach())
        self.register_buffer("rope_freq", self._rope.freq)
        self.register_buffer("rope_rot", self._rope.rot)
        for layer in self.layers:
            if layer.self_attn.head_dim_padding or layer.encoder_attn.head_dim_padding:
                raise ValueError("moonshine-streaming: a padded head dimension "
                                 "(`pad_head_dim_to_multiple_of`) is not reproduced.")

    def forward(self, tokens, position_ids, attention_mask, *cross):
        self._rope.freq, self._rope.rot = self.rope_freq, self.rope_rot
        hidden = self.embed_tokens(tokens)
        cos, sin = self._rope.cos_sin(position_ids, hidden.dtype)
        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden = residual + _self_attention(layer.self_attn, self._rope, layer.input_layernorm(hidden),
                                                cos, sin, attention_mask)
            residual = hidden
            hidden = residual + _cross_attention(layer.encoder_attn, layer.post_attention_layernorm(hidden),
                                                 cross[2 * i], cross[2 * i + 1])
            residual = hidden
            hidden = residual + layer.mlp(layer.final_layernorm(hidden))
        return self.proj_out(self.norm(hidden))


def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def _check_fused_attention(topo: dict, n_layers: int) -> int:
    """Exactly one cached ATTENTION node per decoder layer -- the self-attention blocks. Cross-attention
    carries no mask add, which is what keeps it an ordinary softmax; one that fused would claim a
    KV-cache slot the self-attention blocks address by occurrence order (`canary_export`'s check)."""
    n_fused = sum(1 for node in topo.get("nodes", []) if node.get("op") == "ATTENTION")
    if n_fused != n_layers:
        raise ValueError(f"moonshine-streaming decoder: expected {n_layers} fused ATTENTION nodes (one "
                         f"self-attention block per layer), found {n_fused}.")
    return 0


@dataclass(kw_only=True)
class ASRMoonshineStreamingExportConfig(BaseMultiPhaseModelExportConfig):
    """Moonshine Streaming as three traced phases -- `encoder`, `cross_kv`, `decoder` -- and a driver
    that runs the first two once and loops the third."""

    checkpoint: str = ""
    architecture: str = "moonshine-streaming"
    output_path: str = "moonshine-streaming.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "moonshine_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    # The encoder's trace length in seconds, and the decoder's trace lengths -- free, not 1, and
    # different from each other so the two symbols cannot be confused by the trace.
    trace_seconds: float = 2.0
    trace_tokens: int = 4
    trace_enc: int = 6

    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    frame_len: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
    max_enc_frames: Optional[int] = field(default=None, init=False, repr=False)
    max_positions: Optional[int] = field(default=None, init=False, repr=False)
    decoder_start_id: Optional[int] = field(default=None, init=False, repr=False)
    eos_token_id: Optional[int] = field(default=None, init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)

    __unchecked__ = {
        "checkpoint": Unchecked("the HF directory; the recognizer read its config.json's model_type and "
                                "from_pretrained raises on anything it cannot load"),
        "architecture": Unchecked("the GGUF's own architecture string"),
        "output_path": Unchecked("where to write"),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("the hand-written fragments are parsed and cross-checked by "
                                        "LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "trace_seconds": Unchecked("a property of the TRACE, not of the model"),
        "trace_tokens": Unchecked("same"),
        "trace_enc": Unchecked("same"),
        "sample_rate": Unchecked("READ off the encoder config (`sample_rate`)"),
        "frame_len": Unchecked("READ off the embedder (`sample_rate * frame_ms / 1000`)"),
        "d_model": Unchecked("READ off the decoder config"),
        "n_layers": Unchecked("READ off the decoder's own layer count"),
        "max_enc_frames": Unchecked("READ off the decoder's encoder-position table (`pos_emb`)"),
        "max_positions": Unchecked("READ off the decoder config's max_position_embeddings: the KV-cache "
                                   "capacity"),
        "decoder_start_id": Unchecked("READ off the checkpoint's generation config"),
        "eos_token_id": Unchecked("READ off the checkpoint's generation config"),
        "cross_kv_names": Unchecked("derived by `cross_kv_input_names`, which also orders the "
                                    "cross_kv phase's outputs"),
        "decoder_bindings": Unchecked("derived from the same mil_inputs the trace is declared with"),
    }

    def load_model(self):
        from transformers import GenerationConfig, MoonshineStreamingForConditionalGeneration

        print(f"Loading Moonshine Streaming from {self.checkpoint}...")
        # Eager attention: the encoder layers run transformers' own attention with the masks built
        # here, and only the eager path is a plain matmul/softmax the trace can lower.
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            self.checkpoint, attn_implementation="eager", dtype=torch.float32).eval()
        generation = GenerationConfig.from_pretrained(self.checkpoint)
        self.decoder_start_id = int(generation.decoder_start_token_id)
        self.eos_token_id = int(generation.eos_token_id)
        return model

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        encoder_wrapper = _MoonshineEncoderWrapper(model).eval()
        decoder = model.model.decoder
        self.sample_rate = int(model.config.encoder_config.sample_rate)
        self.frame_len = encoder_wrapper.frame_len
        self.n_layers = len(decoder.layers)
        self.d_model = int(model.config.hidden_size)
        self.max_enc_frames = int(decoder.pos_emb.num_embeddings)
        self.max_positions = int(model.config.max_position_embeddings)
        self.cross_kv_names = cross_kv_input_names(self.n_layers)

        n_samples = int(self.trace_seconds * self.sample_rate) // self.frame_len * self.frame_len
        token_axis = ct.RangeDim(1, self.max_positions)
        enc_axis = ct.RangeDim(1, self.max_enc_frames)
        decoder_inputs = [
            ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
        ] + [
            ct.TensorType(name=name, shape=(1, enc_axis, self.d_model), dtype=np.float32)
            for name in self.cross_kv_names
        ]
        self.decoder_bindings = tuple((t.name, _binding_kind(t.name)) for t in decoder_inputs)
        trace_tokens, trace_enc = int(self.trace_tokens), int(self.trace_enc)

        return [
            ExportPhase(
                name="encoder", wrapper=encoder_wrapper, dummy_inputs=(torch.randn(1, n_samples) * 0.1,),
                # Any length here; the DRIVER makes it a whole number of frames (01_frames.lua).
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, ct.RangeDim(
                    self.frame_len, self.max_samples())), dtype=np.float32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="cross_kv", wrapper=_MoonshineCrossKvWrapper(decoder.layers).eval(),
                dummy_inputs=(torch.zeros(1, trace_enc, self.d_model),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, enc_axis, self.d_model), dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder", wrapper=_MoonshineDecoderWrapper(model).eval(),
                dummy_inputs=(
                    torch.zeros((1, trace_tokens), dtype=torch.long),
                    torch.arange(trace_tokens).unsqueeze(0),
                    causal_mask(trace_tokens),
                ) + tuple(torch.zeros(1, trace_enc, self.d_model) for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                topology_rewrite=lambda topo: _check_fused_attention(topo, self.n_layers),
                fuse_attention=True,
                kv_cache_size=self.max_positions,
            ),
        ]

    def max_samples(self) -> int:
        """The longest clip the encoder-position table covers: `max_enc_frames` rows at the two stride-2
        convolutions' 4 frames per row."""
        return int(self.max_enc_frames) * 4 * int(self.frame_len)

    def hparams(self) -> dict:
        """`samples_per_chunk` is the frame: the engine pads a clip to whole frames and passes the real
        count as `audio_samples` (loom.cpp `transcribe`, family 3's contract), which is the processor's
        own padding. `n_ctx` is the KV-cache capacity."""
        if not self.sample_rate:
            return {}
        return {"sample_rate": self.sample_rate, "samples_per_chunk": self.frame_len,
                "n_ctx": self.max_positions}

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        # Named, not detected: the tokenizer is a SentencePiece BPE converted to `tokenizer.json`, whose
        # llama.cpp chkhsh is in no table, so detection would fall back to a byte-level shape and decode
        # every U+2581 as nothing ("Andso,myfellowAmericans" -- the first export's transcript).
        kwargs = dict(tokenizer_dir=self.checkpoint, tokenizer_pre="spm-byte-fallback",
                      hparams=self.hparams())
        if self.eos_token_id is not None:
            kwargs["eos_token_id"] = self.eos_token_id
        return kwargs

    def driver_components(self) -> List:
        """The frames, the encoder once, cross-attention K/V once, then the decode loop."""
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Len, Lit, OutputRef, Var

        waveform = FieldAccess("inputs", "waveform")
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "FRAME": self.frame_len or 0,
                "SAMPLE_RATE": self.sample_rate or 0,
                "TOKENS_PER_SECOND": TOKENS_PER_SECOND,
                "MAX_ENC_FRAMES": self.max_enc_frames or 0,
                "MAX_POSITIONS": self.max_positions or 0,
                "DECODER_START": self.decoder_start_id if self.decoder_start_id is not None else 1,
            }),
            LuaFragment(
                self.driver_script_path / "01_frames.lua",
                reads=("FRAME", "SAMPLE_RATE", "TOKENS_PER_SECOND", "MAX_ENC_FRAMES", "MAX_POSITIONS"),
                defines=("_max_new",),
            ),
            SubgraphCallComponent(
                topology="encoder", outputs=(), retain=True,
                inputs={"waveform": waveform},
                axes={"n_samples": Len(waveform), "n_past": Lit(0)},
                note="Encoder: frames, sliding-window layers, the position table and the projection.",
            ),
            LuaFragment(
                self.driver_script_path / "02_prompt.lua",
                reads=("DECODER_START",),
                defines=("_n_enc", "_prompt"),
            ),
            SubgraphCallComponent(
                topology="cross_kv", outputs=(), retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": Var("_n_enc"), "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, computed once.",
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                bound={name: OutputRef("cross_kv", index=i + 1)
                       for i, name in enumerate(self.cross_kv_names)},
                extra_axes={"n_enc_frames": Var("_n_enc")},
                prompt=Var("_prompt"),
                default_max_new_tokens=self.max_positions or 16,
                default_eos_token=self.eos_token_id if self.eos_token_id is not None else -1,
            ),
        ]


def _hf_model_type(path: Path) -> Optional[str]:
    import json

    config = path / "config.json"
    if not path.is_dir() or not config.is_file():
        return None
    try:
        return json.loads(config.read_text()).get("model_type")
    except (OSError, ValueError):
        return None


def _is_moonshine_streaming(path: Path) -> bool:
    return _hf_model_type(path) == "moonshine_streaming"


def _build_moonshine_streaming(path: Path, output_path: str) -> ASRMoonshineStreamingExportConfig:
    return ASRMoonshineStreamingExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRMoonshineStreamingExportConfig,
        recognizers=[ModelRecognizer(name="moonshine-streaming", detect=_is_moonshine_streaming,
                                     build_config=_build_moonshine_streaming)],
    ))
