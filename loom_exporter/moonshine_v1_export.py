"""Moonshine (v1) -- Useful Sensors' original `moonshine` (tiny, base): a convolution stem over the raw
waveform and a full-attention RoPE encoder, feeding a RoPE decoder that cross-attends to it. Family 2's
shape -- `encoder` once, `cross_kv` once, a KV-cached `decoder` step in a loop -- and the decoder is the
streaming line's in everything the trace sees, so `moonshine_export` is the module to read beside this
one; its RoPE (`_InterleavedRope`), cross-attention and cached self-attention are reused as they are.

What differs from Moonshine Streaming:

* **The front end is three plain convolutions**, no frames and no CMVN: `tanh(conv k=127 s=64)`, a
  one-group GroupNorm, then `gelu(conv k=7 s=3)` and `gelu(conv k=3 s=2)` -- 384 samples per encoder
  row (41.7 Hz). The stem takes any length of at least 895 samples (one row out of the last conv), so
  the driver pads nothing and the export declares no `samples_per_chunk`: the engine hands the whole
  waveform over with its `length`, the NeMo shape of a one-pass call.
* **The encoder attends to the whole clip and carries its own RoPE** (interleaved and partial, as the
  decoder's): there is no sliding window and no learned position table, so nothing caps the encoder.
  What caps a clip is the DECODER: `max_position_embeddings` (194) is the KV cache, and the card's 6.5
  tokens per second reaches it at 29.8 s. Past that the driver errors rather than truncating the
  transcript -- a cut transcript would be a silent divergence from the reference, which decodes on.
* **`pad_head_dim_to_multiple_of`** (36 -> 40 on tiny) zero-pads Q, K and V before attention and slices
  the output back. The scale is taken on the UNPADDED head (`head_dim ** -0.5`) and zero dims add
  nothing to a dot product or to the slice that is kept, so the padding is an identity and is not
  reproduced.

**Heads are read off the projections**, not off `attn.config`: transformers' `MoonshineAttention`
`update`s ONE shared config with its own `num_attention_heads`, which the config's attribute map writes
through to `encoder_num_attention_heads` -- so after construction every attention module, and the
config itself, reports the decoder's. The two are equal on every released checkpoint, which is the only
reason transformers' own forward is right; `_check_heads` checks it on a config read BEFORE the model
is built, since the built one can no longer tell.

**Decoding is the model card's**: greedy from `<s>`, ended by `</s>`, `max_length` = 6.5 tokens per
second of audio, the start token included.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .moonshine_export import (
    TOKENS_PER_SECOND, _cross_attention, _hf_model_type, _InterleavedRope, _MoonshineCrossKvWrapper,
    _self_attention, causal_mask, cross_kv_input_names,
)
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked

# The stem's total stride, and the shortest input that leaves one row after the last convolution:
# conv3 needs 3 rows of conv2, conv2 needs 7 + 2 * 3 = 13 rows of conv1, conv1 needs 127 + 12 * 64.
STEM_STRIDE = 64 * 3 * 2
MIN_SAMPLES = 127 + 12 * 64


def _check_heads(config) -> None:
    """On a config read fresh from the checkpoint: building the model rewrites it (see the docstring)."""
    if (config.encoder_num_attention_heads != config.decoder_num_attention_heads
            or config.encoder_num_key_value_heads != config.decoder_num_key_value_heads
            or config.decoder_num_attention_heads != config.decoder_num_key_value_heads):
        raise ValueError("moonshine: encoder and decoder head counts differ, or attention is grouped. "
                         "transformers' MoonshineAttention reads ONE shared config for both, so the "
                         "reference itself would be wrong; not reproduced.")
    rope = getattr(config, "rope_scaling", None)
    if isinstance(rope, dict) and rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError(f"moonshine: rope_scaling {rope!r}; only the default (unscaled) RoPE is "
                         f"reproduced.")


def _encoder_attention(attn, rope, hidden, cos, sin):
    """`MoonshineAttention.forward` for the encoder: every row sees every row, so no mask is added."""
    bsz, q_len = hidden.shape[:-1]
    head_dim = attn.head_dim
    heads = attn.q_proj.out_features // head_dim
    q = attn.q_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(bsz, q_len, heads, head_dim).transpose(1, 2)
    q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
    scores = torch.matmul(q * attn.scaling, k.transpose(2, 3))
    out = torch.matmul(F.softmax(scores, dim=-1), v)
    return attn.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


class _MoonshineV1EncoderWrapper(nn.Module):
    """`waveform [1, n_samples] -> [1, n_enc, d_model]`: the stem, the layers, the final norm."""

    def __init__(self, model):
        super().__init__()
        encoder = model.model.encoder
        self.conv1, self.conv2, self.conv3 = encoder.conv1, encoder.conv2, encoder.conv3
        self.groupnorm = encoder.groupnorm
        self.layers = encoder.layers
        self.layer_norm = encoder.layer_norm
        self._rope = _InterleavedRope(encoder.rotary_emb.inv_freq.detach())
        self.register_buffer("rope_freq", self._rope.freq)
        self.register_buffer("rope_rot", self._rope.rot)

    def forward(self, waveform):
        self._rope.freq, self._rope.rot = self.rope_freq, self.rope_rot
        hidden = torch.tanh(self.conv1(waveform.unsqueeze(1)))
        hidden = self.groupnorm(hidden)
        hidden = F.gelu(self.conv2(hidden))
        hidden = F.gelu(self.conv3(hidden)).transpose(1, 2)
        cos, sin = self._rope.cos_sin(torch.arange(hidden.shape[1]).unsqueeze(0), hidden.dtype)
        for layer in self.layers:
            hidden = hidden + _encoder_attention(layer.self_attn, self._rope, layer.input_layernorm(hidden),
                                                 cos, sin)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.layer_norm(hidden)


class _MoonshineV1DecoderWrapper(nn.Module):
    """`(tokens, position_ids, attention_mask, xk_0, xv_0, ...) -> logits`, each layer's parts in its
    order: pre-norm self-attention, pre-norm cross-attention, pre-norm gated MLP."""

    def __init__(self, model):
        super().__init__()
        decoder = model.model.decoder
        self.embed_tokens = decoder.embed_tokens
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.proj_out = model.proj_out
        self._rope = _InterleavedRope(decoder.rotary_emb.inv_freq.detach())
        self.register_buffer("rope_freq", self._rope.freq)
        self.register_buffer("rope_rot", self._rope.rot)

    def forward(self, tokens, position_ids, attention_mask, *cross):
        self._rope.freq, self._rope.rot = self.rope_freq, self.rope_rot
        hidden = self.embed_tokens(tokens)
        cos, sin = self._rope.cos_sin(position_ids, hidden.dtype)
        for i, layer in enumerate(self.layers):
            hidden = hidden + _self_attention(layer.self_attn, self._rope, layer.input_layernorm(hidden),
                                              cos, sin, attention_mask)
            hidden = hidden + _cross_attention(layer.encoder_attn, layer.post_attention_layernorm(hidden),
                                               cross[2 * i], cross[2 * i + 1])
            hidden = hidden + layer.mlp(layer.final_layernorm(hidden))
        return self.proj_out(self.norm(hidden))


def _check_fused_attention(topo: dict, n_layers: int) -> int:
    """One cached ATTENTION node per decoder layer -- see `moonshine_export._check_fused_attention`."""
    n_fused = sum(1 for node in topo.get("nodes", []) if node.get("op") == "ATTENTION")
    if n_fused != n_layers:
        raise ValueError(f"moonshine decoder: expected {n_layers} fused ATTENTION nodes (one "
                         f"self-attention block per layer), found {n_fused}.")
    return 0


@dataclass(kw_only=True)
class ASRMoonshineExportConfig(BaseMultiPhaseModelExportConfig):
    """Moonshine v1 as three traced phases -- `encoder`, `cross_kv`, `decoder` -- and a driver that runs
    the first two once and loops the third."""

    checkpoint: str = ""
    architecture: str = "moonshine"
    output_path: str = "moonshine.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "moonshine_v1_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    trace_seconds: float = 2.0
    trace_tokens: int = 4
    trace_enc: int = 6
    # The longest clip the encoder phase is declared for. The model has no table that caps it -- the
    # decoder's KV cache does, at ~30 s -- so this only has to be past that.
    max_seconds: float = 60.0

    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
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
        "max_seconds": Unchecked("the encoder axis's upper bound; the driver's KV-cache check is what "
                                 "refuses a clip, long before this"),
        "sample_rate": Unchecked("READ off the checkpoint's preprocessor config"),
        "d_model": Unchecked("READ off the config"),
        "n_layers": Unchecked("READ off the decoder's own layer count"),
        "max_positions": Unchecked("READ off the config's max_position_embeddings: the KV-cache capacity"),
        "decoder_start_id": Unchecked("READ off the checkpoint's generation config"),
        "eos_token_id": Unchecked("READ off the checkpoint's generation config"),
        "cross_kv_names": Unchecked("derived by `cross_kv_input_names`, which also orders the "
                                    "cross_kv phase's outputs"),
        "decoder_bindings": Unchecked("derived from the same mil_inputs the trace is declared with"),
    }

    def load_model(self):
        import json

        from transformers import GenerationConfig, MoonshineConfig, MoonshineForConditionalGeneration

        print(f"Loading Moonshine from {self.checkpoint}...")
        _check_heads(MoonshineConfig.from_pretrained(self.checkpoint))
        model = MoonshineForConditionalGeneration.from_pretrained(
            self.checkpoint, attn_implementation="eager", dtype=torch.float32).eval()
        generation = GenerationConfig.from_pretrained(self.checkpoint)
        self.decoder_start_id = int(generation.decoder_start_token_id)
        self.eos_token_id = int(generation.eos_token_id)
        preprocessor = Path(self.checkpoint) / "preprocessor_config.json"
        self.sample_rate = int(json.loads(preprocessor.read_text())["sampling_rate"])
        return model

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        decoder = model.model.decoder
        self.n_layers = len(decoder.layers)
        self.d_model = int(model.config.hidden_size)
        self.max_positions = int(model.config.max_position_embeddings)
        self.cross_kv_names = cross_kv_input_names(self.n_layers)
        max_samples = int(self.max_seconds * self.sample_rate)
        max_enc = (max_samples - MIN_SAMPLES) // STEM_STRIDE + 1

        token_axis = ct.RangeDim(1, self.max_positions)
        enc_axis = ct.RangeDim(1, max_enc)
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
                name="encoder", wrapper=_MoonshineV1EncoderWrapper(model).eval(),
                dummy_inputs=(torch.randn(1, int(self.trace_seconds * self.sample_rate)) * 0.1,),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, ct.RangeDim(MIN_SAMPLES, max_samples)),
                                          dtype=np.float32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="cross_kv", wrapper=_MoonshineCrossKvWrapper(decoder.layers).eval(),
                dummy_inputs=(torch.zeros(1, trace_enc, self.d_model),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, enc_axis, self.d_model), dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder", wrapper=_MoonshineV1DecoderWrapper(model).eval(),
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

    def hparams(self) -> dict:
        """No `samples_per_chunk`: the stem takes any length, so the engine passes the waveform as it is
        with its `length` (loom.cpp `transcribe`, the one-pass branch). `n_ctx` is the KV cache."""
        if not self.sample_rate:
            return {}
        return {"sample_rate": self.sample_rate, "n_ctx": self.max_positions}

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        # The same SentencePiece-converted `tokenizer.json` as Moonshine Streaming; see its note.
        kwargs = dict(tokenizer_dir=self.checkpoint, tokenizer_pre="spm-byte-fallback",
                      hparams=self.hparams())
        # `eos_token_ids`, the list: the BPE writer reads that and ignores the scalar, and falls back
        # to `tokenizer_config.json`'s `eos_token` -- which the streaming checkpoints ship and v1 does
        # not. Without it the file declared no end of sequence and `</s>` reached the transcript.
        if self.eos_token_id is not None:
            kwargs["eos_token_ids"] = [self.eos_token_id]
        return kwargs

    def driver_components(self) -> List:
        """The budget check, the encoder once, cross-attention K/V once, then the decode loop."""
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Len, Lit, OutputRef, Var

        waveform = FieldAccess("inputs", "waveform")
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "SAMPLE_RATE": self.sample_rate or 0,
                "MIN_SAMPLES": MIN_SAMPLES,
                "TOKENS_PER_SECOND": TOKENS_PER_SECOND,
                "MAX_POSITIONS": self.max_positions or 0,
                "DECODER_START": self.decoder_start_id if self.decoder_start_id is not None else 1,
            }),
            LuaFragment(
                self.driver_script_path / "01_budget.lua",
                reads=("SAMPLE_RATE", "MIN_SAMPLES", "TOKENS_PER_SECOND", "MAX_POSITIONS"),
                defines=("_max_new",),
            ),
            SubgraphCallComponent(
                topology="encoder", outputs=(), retain=True,
                inputs={"waveform": waveform},
                axes={"n_samples": Len(waveform), "n_past": Lit(0)},
                note="Encoder: the convolution stem, the RoPE layers and the final norm.",
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


def _is_moonshine(path: Path) -> bool:
    return _hf_model_type(path) == "moonshine"


def _build_moonshine(path: Path, output_path: str) -> ASRMoonshineExportConfig:
    return ASRMoonshineExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRMoonshineExportConfig,
        recognizers=[ModelRecognizer(name="moonshine", detect=_is_moonshine,
                                     build_config=_build_moonshine)],
    ))
