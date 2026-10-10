"""Family 14 (music, P5), on MusicGen: a text prompt in, four delayed EnCodec code streams out.

    text -> T5 encoder -> MusicGen decoder -> 4 delayed code streams -> realign -> EnCodec 32 kHz -> audio

**It is two families already in the tree, composed, and that is why it costs this little.** The text
side is family 6's T5 encoder, run the way `t5_export` runs it: a block loop over a caller-supplied
relative-position bias, with `t5_position_bias` from `t5_driver/00_header.lua` building that bias in the
driver. The audio side is family 10's shape, run the way `dia_export` runs Dia: `encoder` /
`cross_kv` / `decoder`, a KV-cached step emitting one logit row per codebook, and classifier-free
guidance as second STREAMS of the last two phases. The codec it feeds is `facebook/encodec_32khz`,
which family 11 exported and published (`encodec-32khz-loom`); its config is MusicGen's
`audio_encoder` key for key.

Three things are MusicGen's own, and each is the reason for a piece of code here:

* **The positions are an INPUT.** `MusicgenSinusoidalPositionalEmbedding.forward` indexes its table at
  `arange(seq_len) + past_key_values_length`, and in a cache-free trace that length is always 0, so
  traced as-is every decode step would be embedded at position 0 and no error would say so. See
  `_PositionSlot`.

* **The unconditional stream is ZERO cross-attention, not a blank prompt.** `generate()` concatenates
  `zeros_like(last_hidden_state)` and a zero attention mask, and `forward` multiplies the projected
  states by that mask -- so the unconditional K/V are exactly zero whatever `enc_to_dec_proj`'s bias
  is. Dia's unconditional half re-ran its encoder over zero BYTES; this one must not, since a T5 pass
  over id 0 is not zero. The driver hands `cross_kv_uncond` a zero tensor instead.

* **The delay pattern is `build_delay_pattern_mask`'s, which has no EOS.** Codebook k is offset by k
  steps, there is no stop token, and the generation length is the caller's (`max_new_tokens`, HF's own
  meaning). The scaffold is index arithmetic in the driver, like Dia's.

The modality pair is `text -> audio_codes`, the same as Dia's: the SentencePiece vocabulary travels
with the model, and what comes out is what `audio-codec` decodes (ADR-020).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .bpe_tokenizer_export import read_sampling_defaults
from .decomposition import Decomposition, MultiPhase
from .dia_export import _CrossKvSlot, causal_mask, cross_kv_input_names
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked
from .t5_export import _check_fused_attention, relative_attention_bias_table


def install_scaled_query_attention() -> None:
    """Scale Q before `Q @ K^T` in MusicGen's eager attention, so `fuse_loom_attention` can match it.

    HF's `eager_attention_forward` here computes `matmul(q, k^T) * scaling` and THEN adds the mask. The
    fusion pass anchors on `add(matmul(q_scaled, k^T), mask)`, the form Llama and Dia trace to, so a
    `mul` between the matmul and the add matches nothing: every block stays an unfused SOFTMAX, the
    decoder has no KV cache, and each step attends to itself alone with no error raised.

    Patched here, in this family, rather than taught to the shared pass. Loosening the pass would also
    match any MASKED cross-attention in another fused family, which would then be given a cache slot it
    must never have (`t5_export._check_fused_attention` exists for that).

    **Bit-identical, not merely close, for this checkpoint:** `scaling` is `head_dim ** -0.5`, and a
    head_dim of 64 makes it 1/8, a power of two, so `(q / 8) @ k` and `(q @ k) / 8` round identically.
    `phases()` checks the head_dim is a power of four so that this stays true.
    """
    from transformers.models.musicgen import modeling_musicgen

    def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                                head_mask=None, **kwargs):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        attn_weights = torch.matmul(query * scaling, key.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_musicgen.eager_attention_forward = eager_attention_forward


class _MusicgenEncoderWrapper(nn.Module):
    """`(input_ids, position_bias) -> projected encoder states`, T5 plus `enc_to_dec_proj`.

    The T5 half is `t5_export._T5EncoderWrapper`'s block loop for its reason: `T5Stack.forward` derives
    the relative bias from a Python-level length, which tracing bakes, so the bias is passed in.

    `enc_to_dec_proj` is here rather than in `cross_kv` because of the unconditional stream: what HF
    zeroes is the PROJECTED state (`forward` multiplies after projecting), so `cross_kv`'s input has to
    be that state for a zero input to mean what HF's zero means. Its `* attention_mask` factor is 1 for
    the caller's own unpadded prompt and is left out.
    """

    def __init__(self, model):
        super().__init__()
        self.shared = model.text_encoder.shared
        self.enc = model.text_encoder.encoder
        self.proj = model.enc_to_dec_proj

    def forward(self, input_ids, position_bias):
        hidden = self.shared(input_ids)
        for block in self.enc.block:
            hidden = block(hidden, attention_mask=None, position_bias=position_bias)[0]
        return self.proj(self.enc.final_layer_norm(hidden))


class _MusicgenCrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`, every decoder layer's cross-attention K/V, once.

    `dia_export._DiaCrossKvWrapper`'s job and its ordering rule: build this BEFORE the decoder wrapper,
    which then swaps the projections for slots."""

    def __init__(self, decoder):
        super().__init__()
        self.projs = nn.ModuleList()
        for layer in decoder.layers:
            self.projs.append(layer.encoder_attn.k_proj)
            self.projs.append(layer.encoder_attn.v_proj)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


class _PositionSlot(nn.Module):
    """Stands in for `MusicgenSinusoidalPositionalEmbedding` and reads the table at the caller's
    `position_ids`.

    The original computes `arange(seq_len) + past_key_values_length`. A cache-free trace has no past,
    so that is `arange(seq_len)` -- correct for the first step and wrong for every later one, where the
    single row the engine runs sits at position `n_past`. Indexing the SAME table at a declared input
    makes the position the driver's, as `position_ids` is for every other cached decoder here. The
    table is the checkpoint's own buffer, not recomputed, so the values are HF's bit for bit.
    """

    def __init__(self, table: torch.Tensor, holder: list):
        super().__init__()
        self.register_buffer("table", table.detach().clone(), persistent=False)
        self._holder = holder

    def forward(self, input_ids, past_key_values_length=0):
        return self.table.index_select(0, self._holder[0].reshape(-1))


class _MusicgenDecoderWrapper(nn.Module):
    """`(codes, position_ids, attention_mask, xk_0, xv_0, ...) -> the LAST step's logit rows, one per
    codebook`.

    `codes` is `[1, steps, n_codebooks]`, frame-major like Dia's, and transposed here into HF's
    `(n_codebooks, steps)` input layout. The head runs on the last row only, Dia's reason: the output
    is `[1, n_codebooks, vocab]` at every length, which `loom.sample_row` reads one row per codebook.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder.model.decoder
        self.lm_heads = model.decoder.lm_heads
        self.n_codebooks = int(self.decoder.num_codebooks)
        self._cross = [None] * (2 * len(self.decoder.layers))
        for i, layer in enumerate(self.decoder.layers):
            layer.encoder_attn.k_proj = _CrossKvSlot(self._cross, 2 * i)
            layer.encoder_attn.v_proj = _CrossKvSlot(self._cross, 2 * i + 1)
        self._positions = [None]
        self.decoder.embed_positions = _PositionSlot(self.decoder.embed_positions.weights,
                                                     self._positions)
        # **The mask must reach attention as handed in, and MusicGen's builder does not pass a 4-D one
        # through.** Dia's does; this one's eager path calls `_prepare_4d_causal_attention_mask`, which
        # reads a 4-D mask as 0/1 and inverts it (`1 - mask`, then `masked_fill(min)`). The additive
        # mask the driver builds would come out inverted, and the computed mask is not a graph input.
        # With `install_scaled_query_attention` in place the exporter refuses that graph (a cached
        # ATTENTION whose mask is no declared input); without it, the first export matched NOTHING:
        # 48 bare SOFTMAX nodes, no cached ATTENTION, a decoder attending to the current step alone.
        self.decoder._update_causal_mask = lambda attention_mask, *args, **kwargs: attention_mask

    def forward(self, codes, position_ids, attention_mask, *cross):
        for i, tensor in enumerate(cross):
            self._cross[i] = tensor
        self._positions[0] = position_ids
        hidden = self.decoder(
            input_ids=codes[0].transpose(0, 1), attention_mask=attention_mask,
            # `cross[0]` only keeps `is_cross_attention` true; the projections that read it are slots.
            encoder_hidden_states=cross[0], encoder_attention_mask=None, use_cache=False,
        ).last_hidden_state
        last = hidden[:, -1:, :]
        return torch.cat([head(last) for head in self.lm_heads], dim=1)


@dataclass
class TextToCodesMusicgenExportConfig(BaseMultiPhaseModelExportConfig):
    """MusicGen as `encoder` (T5 + projection), `cross_kv` (per-layer K/V, once) and `decoder` (one
    cached step, one logit row per codebook), with `cross_kv_uncond` / `decoder_uncond` streams for
    classifier-free guidance. Dia's arrangement; see `dia_export.TextToCodesDiaExportConfig`."""

    model_dir: str = ""
    architecture: str = "musicgen"
    output_path: str = "musicgen.gguf"
    # Counts decoder STEPS, spelled `n_tokens` because `GraphBuilder` sizes the KV cache's cell index
    # off that name (Dia's comment on the same field).
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "musicgen_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    trace_text_len: int = 12
    trace_steps: int = 8

    n_codebooks: Optional[int] = field(default=None, init=False, repr=False)
    codebook_size: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
    hidden: Optional[int] = field(default=None, init=False, repr=False)
    text_hidden: Optional[int] = field(default=None, init=False, repr=False)
    n_text_head: Optional[int] = field(default=None, init=False, repr=False)
    num_buckets: Optional[int] = field(default=None, init=False, repr=False)
    max_distance: Optional[int] = field(default=None, init=False, repr=False)
    max_text_len: Optional[int] = field(default=None, init=False, repr=False)
    max_positions: Optional[int] = field(default=None, init=False, repr=False)
    pad_token_id: Optional[int] = field(default=None, init=False, repr=False)
    eos_token_id: Optional[int] = field(default=None, init=False, repr=False)
    max_length: Optional[int] = field(default=None, init=False, repr=False)
    encoder_bias: tuple = field(default=(), init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    guidance_scale: float = field(default=1.0, init=False, repr=False)
    sampling_defaults: dict = field(default_factory=dict, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked("path to the HF directory, established by the recognizer's detect(); "
                               "MusicgenForConditionalGeneration.from_pretrained raises on anything else"),
        "architecture": Unchecked("the GGUF's own architecture string; nothing to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the decoder ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("the fragments are parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "trace_text_len": Unchecked("the concrete trace length; the dynamic range is declared through "
                                    "ct.convert's inputs=, so it constrains nothing the checkpoint "
                                    "could disagree with"),
        "trace_steps": Unchecked("same, for the decoder's step axis"),
        "n_codebooks": Unchecked("READ off the decoder config in phases() (`num_codebooks`)"),
        "codebook_size": Unchecked("same -- `decoder.vocab_size`, the width of one logit row"),
        "n_layers": Unchecked("same -- a COUNT of decoder layers, two cross_kv outputs each"),
        "hidden": Unchecked("same -- `decoder.hidden_size`, the cross-attention K/V width"),
        "text_hidden": Unchecked("same -- `text_encoder.d_model`"),
        "n_text_head": Unchecked("same -- `text_encoder.num_heads`, the bias tensor's head axis"),
        "num_buckets": Unchecked("same -- `relative_attention_num_buckets`; the table length is "
                                 "cross-checked against it in phases()"),
        "max_distance": Unchecked("same -- `relative_attention_max_distance`"),
        "max_text_len": Unchecked("same -- the T5 encoder's `n_positions`, the RangeDim bound"),
        "max_positions": Unchecked("same -- `decoder.max_position_embeddings`, the KV cache capacity "
                                   "and the positional table's row count"),
        "pad_token_id": Unchecked("same -- the decoder's `pad_token_id`, which is also its BOS and the "
                                  "only id the delay scaffold writes. phases() checks they agree."),
        "eos_token_id": Unchecked("the T5 tokenizer's `</s>`, read off the text encoder config"),
        "max_length": Unchecked("READ off the resolved GenerationConfig: the default generation length"),
        "encoder_bias": Unchecked("the checkpoint's own relative-attention table, flattened by "
                                  "`t5_export.relative_attention_bias_table`; its length is checked"),
        "cross_kv_names": Unchecked("derived by `dia_export.cross_kv_input_names`, the one function "
                                    "that orders both the outputs and the decoder's inputs"),
        "guidance_scale": Unchecked("READ off the resolved GenerationConfig, verbatim: MusicGen's "
                                    "processor is the standard uncond-centred form loom.sample_row "
                                    "implements, so no conversion"),
        "sampling_defaults": Unchecked("READ off generation_config.json by "
                                       "`bpe_tokenizer_export.read_sampling_defaults`, which fills "
                                       "a missing knob with what generate() uses (top_k 50)"),
    }

    def prepare_environment(self) -> None:
        install_scaled_query_attention()

    def load_model(self):
        from transformers import MusicgenForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        return MusicgenForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float32,
            # Eager, so the 4-D mask the decoder is handed passes straight through
            # `_prepare_4d_causal_attention_mask` rather than being rebuilt from a traced length.
            attn_implementation="eager",
        ).eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        cfg = model.config
        dec_cfg, txt_cfg = cfg.decoder, cfg.text_encoder
        decoder = model.decoder.model.decoder

        if int(getattr(dec_cfg, "audio_channels", 1)) != 1:
            raise ValueError(
                "this is a stereo MusicGen checkpoint (audio_channels = 2). Its codebooks interleave "
                "left and right and its delay pattern is per channel PAIR; the driver implements the "
                "mono pattern only, and would realign a stereo stream wrong without raising."
            )
        if int(dec_cfg.pad_token_id) != int(dec_cfg.bos_token_id):
            raise ValueError(
                f"the decoder's pad ({dec_cfg.pad_token_id}) and bos ({dec_cfg.bos_token_id}) differ. "
                f"`build_delay_pattern_mask` writes ONE id into both the start column and the delay "
                f"padding, which the driver relies on."
            )
        if (int(txt_cfg.d_model) == int(dec_cfg.hidden_size)
                or getattr(dec_cfg, "cross_attention_hidden_size", None) is not None):
            raise ValueError(
                "this checkpoint has no `enc_to_dec_proj` step (equal widths, or a declared "
                "cross_attention_hidden_size). The encoder phase applies it unconditionally."
            )

        head_dim = int(dec_cfg.hidden_size) // int(dec_cfg.num_attention_heads)
        # `install_scaled_query_attention` moves the 1/sqrt(head_dim) factor onto Q. That reorders a
        # rounding unless the factor is a power of two, i.e. unless head_dim is a power of four.
        if head_dim <= 0 or head_dim & (head_dim - 1) or (head_dim.bit_length() - 1) % 2:
            raise ValueError(
                f"decoder head_dim is {head_dim}; the scaled-query patch is exact only when "
                f"head_dim ** -0.5 is a power of two (head_dim a power of four). Check the decoder "
                f"against transformers at f32 before relaxing this."
            )

        self.n_codebooks = int(dec_cfg.num_codebooks)
        self.codebook_size = int(dec_cfg.vocab_size)
        self.n_layers = len(decoder.layers)
        self.hidden = int(dec_cfg.hidden_size)
        self.text_hidden = int(txt_cfg.d_model)
        self.n_text_head = int(txt_cfg.num_heads)
        self.num_buckets = int(txt_cfg.relative_attention_num_buckets)
        self.max_distance = int(txt_cfg.relative_attention_max_distance)
        self.max_text_len = int(getattr(txt_cfg, "n_positions", 512))
        self.max_positions = int(dec_cfg.max_position_embeddings)
        self.pad_token_id = int(dec_cfg.pad_token_id)
        self.eos_token_id = int(txt_cfg.eos_token_id)
        self.max_length = int(model.generation_config.max_length)
        self.guidance_scale = float(model.generation_config.guidance_scale or 1.0)
        self.sampling_defaults = read_sampling_defaults(self.model_dir)
        self.encoder_bias = tuple(relative_attention_bias_table(model.text_encoder.encoder))
        if len(self.encoder_bias) != self.num_buckets * self.n_text_head:
            raise ValueError(
                f"the T5 relative-attention table holds {len(self.encoder_bias)} values, but the config "
                f"declares {self.num_buckets} buckets x {self.n_text_head} heads. The driver indexes it "
                f"as `bucket * n_head + head`, which a mismatched table satisfies silently."
            )
        self.cross_kv_names = cross_kv_input_names(self.n_layers)

        step_axis = ct.RangeDim(1, self.max_positions)
        # A separate instance, so a separate symbol: prompt length and generation length are unrelated.
        src_axis = ct.RangeDim(1, self.max_text_len)

        decoder_inputs = [
            ct.TensorType(name="codes", shape=(1, step_axis, self.n_codebooks), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, step_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, step_axis, step_axis), dtype=np.float32),
        ] + [
            ct.TensorType(name=name, shape=(1, src_axis, self.hidden), dtype=np.float32)
            for name in self.cross_kv_names
        ]

        # ORDER IS LOAD-BEARING (Dia's rule): the cross_kv wrapper captures the real projections before
        # the decoder wrapper replaces them with slots.
        cross_kv_wrapper = _MusicgenCrossKvWrapper(decoder).eval()
        decoder_wrapper = _MusicgenDecoderWrapper(model).eval()
        trace_src, trace_steps = int(self.trace_text_len), int(self.trace_steps)

        return [
            ExportPhase(
                name="encoder",
                wrapper=_MusicgenEncoderWrapper(model).eval(),
                dummy_inputs=(torch.zeros((1, trace_src), dtype=torch.long),
                              torch.zeros(1, self.n_text_head, trace_src, trace_src)),
                mil_inputs=[
                    ct.TensorType(name="input_ids", shape=(1, src_axis), dtype=np.int32),
                    ct.TensorType(name="position_bias",
                                  shape=(1, self.n_text_head, src_axis, src_axis), dtype=np.float32),
                ],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv_wrapper,
                dummy_inputs=(torch.zeros(1, trace_src, self.hidden),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, src_axis, self.hidden),
                                          dtype=np.float32)],
                root_axis="n_enc_frames",
                # The unconditional K/V must survive the whole generation beside the conditional ones.
                extra_streams=("cross_kv_uncond",),
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder_wrapper,
                dummy_inputs=(
                    torch.full((1, trace_steps, self.n_codebooks), self.pad_token_id, dtype=torch.long),
                    torch.arange(trace_steps).unsqueeze(0),
                    causal_mask(trace_steps),
                ) + tuple(torch.zeros(1, trace_src, self.hidden) for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                # Exactly one cached ATTENTION per layer and no fused cross-attention, or the export
                # fails. `fuse_attention` is only a request, and this decoder's first export fused
                # nothing and wrote a file anyway (see `install_scaled_query_attention`).
                topology_rewrite=lambda topo: _check_fused_attention(topo, self.n_layers),
                fuse_attention=True,
                kv_cache_size=self.max_positions,
                # Its own KV cache: the two streams are fed the same codes, and a shared cache would
                # make the second run overwrite the cell the first just wrote.
                extra_streams=("decoder_uncond",),
            ),
        ]

    def hparams(self) -> dict:
        """`codec.n_codebooks` / `codec.codebook_size` in the codec family's own spelling, so a host
        piping this into the EnCodec GGUF can compare the two ends; plus the decoding defaults."""
        if self.n_codebooks is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        hparams = {
            "codec.n_codebooks": self.n_codebooks,
            "codec.codebook_size": self.codebook_size,
            "n_text_ctx": self.max_text_len,
            "n_codes_ctx": self.max_positions,
        }
        hparams.update({f"sampling.{k}": v for k, v in self.sampling_defaults.items()})
        hparams["sampling.guidance_scale"] = self.guidance_scale
        return hparams

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        """The T5 SentencePiece vocabulary travels with the model, with `</s>` appended to every
        prompt: `T5Tokenizer` does, and `MusicgenProcessor` is that tokenizer."""
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams(),
                    add_eos_token=True, eos_token_id=self.eos_token_id)

    def driver_components(self) -> List:
        from .driver_components import (
            DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import Call, FieldAccess, Len, Lit, OutputRef, Var

        src_len = Len(FieldAccess("inputs", "tokens"))
        t5_header = Path(__file__).resolve().parent / "t5_driver" / "00_header.lua"
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            # `t5_position_bias`, shared with family 6 rather than copied: it is
            # `_relative_position_bucket`, and two copies of it could drift.
            LuaFragment(t5_header, top_level=True),
            ExportConstants(values={
                "N_CODEBOOKS": self.n_codebooks or 0,
                "N_LAYERS": self.n_layers or 0,
                "HIDDEN": self.hidden or 0,
                "PAD": self.pad_token_id or 0,
                "MAX_CODES": self.max_positions or 0,
                "MAX_LENGTH": self.max_length or 0,
                "N_TEXT_HEAD": self.n_text_head or 0,
                "REL_BUCKETS": self.num_buckets or 0,
                "REL_MAX_DISTANCE": self.max_distance or 0,
                "ENC_REL_BIAS": list(self.encoder_bias),
                "TEMPERATURE": self.sampling_defaults.get("temperature", 0.0),
                "TOP_K": self.sampling_defaults.get("top_k", 0),
                "TOP_P": self.sampling_defaults.get("top_p", 1.0),
                "GUIDANCE_SCALE": self.guidance_scale,
            }),
            SubgraphCallComponent(
                topology="encoder",
                outputs=(),
                retain=True,
                inputs={
                    "input_ids": FieldAccess("inputs", "tokens"),
                    "position_bias": Call("t5_position_bias",
                                          [Var("ENC_REL_BIAS"), Var("N_TEXT_HEAD"), Var("REL_BUCKETS"),
                                           Var("REL_MAX_DISTANCE"), src_len, Lit(0), Lit(1)]),
                },
                axes={"n_tokens": src_len, "n_past": Lit(0)},
                note="T5 encoder + enc_to_dec_proj: one bidirectional pass over the prompt.",
            ),
            SubgraphCallComponent(
                topology="cross_kv",
                outputs=(),
                retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": src_len, "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, once per prompt.",
            ),
            LuaFragment(
                self.driver_script_path / "01_decode.lua",
                reads=("N_CODEBOOKS", "N_LAYERS", "HIDDEN", "PAD", "MAX_CODES", "MAX_LENGTH",
                       "TEMPERATURE", "TOP_K", "TOP_P", "GUIDANCE_SCALE"),
                defines=("_codes",),
            ),
            DriverReturn(values=("_codes",)),
        ]


def _is_musicgen(path: Path) -> bool:
    """An HF directory declaring `model_type == "musicgen"`. Not `musicgen_melody`, whose conditioning
    is a chromagram concatenated to the decoder input rather than cross-attended text."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "musicgen"


def _build_musicgen(path: Path, output_path: str) -> TextToCodesMusicgenExportConfig:
    return TextToCodesMusicgenExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-codes",
        config_class=TextToCodesMusicgenExportConfig,
        recognizers=[ModelRecognizer(name="musicgen", detect=_is_musicgen,
                                     build_config=_build_musicgen)],
    ))
