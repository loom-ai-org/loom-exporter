"""The AR codec-token LM family (`EXPORT-ROADMAP.md` R5's family 10, P5), on Dia-1.6B.

**This is the composition target, and that is why it is this checkpoint.** Family 11 exported a codec
DECODER and verified it on the waveform; a codec decoder with nothing to feed it is half a pipeline.
Dia's own `audio_tokenizer_config.json` names `descript/dac_44khz` -- the codec already exported and
verified -- so family 10 costs the LM half only and the pair composes end to end:

    text -> Dia -> 9 delayed code streams -> realign -> DAC -> waveform

MusicGen was the earlier pick and was dropped because it would have dragged EnCodec in, whose two
blockers are named in `audio_codec_export.ENCODEC_BLOCKERS`.

Structurally this is family 2's shape -- an encoder run once, then a KV-cached decoder cross-attending
to its output -- so `multi_phase_export` + a three-phase `encoder`/`cross_kv`/`decoder` split is the
precedent, and `whisper_export` is the module to read beside this one. Four things differ, and each is
the reason for a piece of code here that has no Whisper counterpart:

* **The text axis is genuinely dynamic, so the cross-attention K/V carry a SECOND symbol.** Whisper's
  encoder always emits 1500 frames, which is what lets its decoder declare fixed-shape `xk_i`/`xv_i`
  and have exactly one dynamic axis. Dia's encoder emits one frame per input BYTE. The decoder
  therefore has two independent dynamic axes -- its own step count and the encoder's frame count --
  which is the case `declared_axes` exists for and which `_reject_shared_symbol_overrides` requires be
  declared for *every* input carrying the symbol, not just one. Measured, not assumed: the traced
  program comes back with `codes (1, is0, 9)` and `xk_0 (1, is1, 2048)`, two distinct symbols, and the
  emitted topology resolves them as `n_tokens` and `n_enc_frames`.

  **So Dia needs no padded or bucketed text axis**, which is the outcome Supertonic did not get
  ([Retro-005](../../loom.cpp/docs/retros/retro-005-supertonic-fixed-text-length.md)). That model's
  fixed width has a second, independent cause -- a length-derived pad coremltools refuses -- and Dia
  has nothing like it.

* **Nine output heads.** `logits_dense` is one `Linear(2048, 9 * 1028)` whose result HF reshapes to
  `[batch * 9, seq, 1028]`. Every decode loop in this tree reduces ONE row to ONE token; this one must
  emit nine per step. It needs no new engine primitive, and `_DiaDecoderWrapper` is why -- see its
  docstring.

* **The delay pattern.** Channel `k` is offset by `delay_pattern[k]` steps, so generation happens in
  delayed space and the codes must be realigned before they reach DAC. It is declared in `config.json`,
  so it is read rather than derived, and it is index arithmetic over a nine-element array -- Lua, not
  C++, by [ADR-013] §2. The engine never learns what a delay pattern is.

* **`rotate_half` has to be replaced, and the diagnosis is the transferable part** -- see
  `install_rotate_half_patch`.

The modality pair is `text -> audio_codes`: the byte vocabulary travels with the model, so a host hands
this a sentence, and what comes out is what `audio-codec` decodes. That is the composition ADR-020
argues for, stated from the producing side.
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
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


def install_rotate_half_patch() -> None:
    """Replace `modeling_dia.rotate_half` with a form that needs no arithmetic over the last dim.

    **Dia does not convert without this, and the failure is not the dynamic-shape class every other
    family on this roadmap hit.** HF's version slices at `x.shape[-1] // 2`:

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

    Under `torch.jit.trace` a `.shape[i]` read is a 0-d **Tensor**, so `// 2` traces as
    `aten::floor_divide` and the slice bound that consumes it as `aten::Int` -- 48 of them in the
    12-layer encoder alone. coremltools' `_int` handler does `int(x.val)` on an array that is not 0-d
    and dies with `TypeError: only 0-dimensional arrays can be converted to Python scalars` at
    `encoder/0/self_attention/128`. It fails at a STATIC text length too, and under both `sdpa` and
    `eager`, so there is nothing to be gained by bucketing the text axis over it.

    **`torch.chunk` asks for a COUNT rather than an index**, so it needs no arithmetic over the last
    dim at all and lowers to a single `split`. That is the whole fix, and it is what makes the encoder
    convert with a fully symbolic axis: `tokens (1, is0)` in, `(1, is0, 1024)` out, using only ops
    already in the dialect.

    **A form that reads the midpoint at all cannot work here, which is worth stating because the
    obvious patch does exactly that.** Deriving `half` and guarding it with `isinstance(half, int)`
    looks like it distinguishes a static last dim from a dynamic one; it does not. Under tracing
    *every* dim comes back as a 0-d Tensor -- the static ones included, confirmed directly -- so such a
    guard raises unconditionally the moment it is traced, and a version without the guard is the
    original bug. `chunk` sidesteps the question rather than answering it.

    The one property `chunk` needs is an even last dim, which is checked in `phases()` against the
    checkpoint's own three head dims rather than assumed: on an odd dim `chunk(2)` splits `ceil`/`floor`
    and would rotate by the wrong amount **silently**, which is the failure mode this whole path exists
    to prevent.

    Verified bit-identical to the original: `max|patched - original| = 0` on the full 12-layer encoder
    at text lengths 7, 32 and 128.
    """
    from transformers.models.dia import modeling_dia

    def rotate_half_chunk(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_dia.rotate_half = rotate_half_chunk


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `create_causal_mask` passes straight through.

    The same tensor -- and the same reason -- as `whisper_export.causal_mask` and
    `causal_lm_export._causal_mask`: an already-prepared 4-D mask short-circuits the internal mask
    builder entirely, so it never derives a key length from a Python-level shape that tracing would
    bake in.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class _DiaEncoderWrapper(nn.Module):
    """`byte ids -> encoder hidden states`, the one tensor the encoder phase exports.

    **`attention_mask=None`, and that is a real decision rather than a default.** Dia's encoder builds
    no mask of its own when handed none (`_update_full_mask` returns `None`, and `eager_attention_
    forward` has the `if attention_mask is not None` guard that DistilBERT's did not, which is what
    made [ADR-019] settle on an all-ones mask instead). Passing nothing is therefore the cheapest
    correct answer here, not a weaker one -- verified: `max|None - all-ones| = 0` at lengths 7, 32 and
    128.

    It is correct because this door takes exactly the bytes the caller wrote and there is no padding to
    hide. HF's own `DiaProcessor` pads only to the longest item in a batch, so a single utterance is
    unpadded there too -- the reference this is graded against is computing the same thing.

    **`position_ids` is NOT an input here, and Dia is the first family where it does not have to be.**
    The encoder computes `torch.arange(input_ids.shape[-1])` internally, which is exactly the pattern
    [ADR-019] rules out -- except that here it traces to `shape` + `range_1d` and stays symbolic rather
    than baking, confirmed on the emitted MIL. The rule is unchanged; this checkpoint satisfies it
    without help, and `phases()` proves that by exporting at two lengths and requiring one topology.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, tokens):
        return self.encoder(input_ids=tokens, attention_mask=None).last_hidden_state


class _CrossKvSlot(nn.Module):
    """Stands in for a cross-attention `k_proj`/`v_proj` and returns a tensor handed in from outside.

    Identical in purpose to `whisper_export._CrossKvSlot`: the projection it replaces is a function of
    the encoder output alone, so its result is the same at every decode step, and traced as-is it is
    recomputed per token. For Dia that is 36 matmuls of `[n_bytes, 1024] x [1024, 2048]` on every one
    of ~900 steps.

    A plain list rather than a buffer or submodule, deliberately: it carries per-CALL tensors, and
    registering them would make them state of the traced module.
    """

    def __init__(self, holder: list, key: int):
        super().__init__()
        self._holder = holder
        self._key = key

    def forward(self, x):
        return self._holder[self._key]


class _DiaCrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`, the cross-attention K/V for every decoder layer, once.

    **Construct this BEFORE `_DiaDecoderWrapper`.** It holds the projection modules directly, so that
    the decoder wrapper may then replace the `layer.cross_attention.k_proj` attributes that used to
    reach them without this phase losing the weights it exports. Built the other way round it would
    trace `_CrossKvSlot`s and export nothing -- the same ordering constraint, for the same reason, as
    `whisper_export`'s pair.

    Interleaved k,v per layer rather than all-k-then-all-v, so the driver's `index` arithmetic is
    `2 * layer + 1` and reads as the pair it is.

    **Unlike Whisper's, this emits K and V in the same layout, and the V transpose is NOT hoisted
    here.** Whisper's `xv_i` leaves already head-split and transposed because rebuilding that layout
    per token measured at 47% of its decode loop -- over a 1500-frame encoder output. Dia's encoder
    output is one frame per input byte, so the same transform is over tens of frames rather than
    fifteen hundred, and the hoist would buy roughly three orders of magnitude less while costing the
    `topology_rewrite` that has to delete the traced chain and raise when it cannot find it. It is left
    on the table deliberately, and it is the first thing to reach for if the decode loop is ever
    profiled: the mechanism is `whisper_export.hoist_cross_v_transpose`, unchanged.
    """

    def __init__(self, model):
        super().__init__()
        self.projs = nn.ModuleList()
        for layer in model.model.decoder.layers:
            self.projs.append(layer.cross_attention.k_proj)
            self.projs.append(layer.cross_attention.v_proj)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


def cross_kv_input_names(n_layers: int) -> tuple:
    """`("xk_0", "xv_0", "xk_1", ...)` -- the decoder's per-layer cross-attention inputs, in the order
    `_DiaCrossKvWrapper` returns them, which is the order their `index` binding assumes."""
    names = []
    for i in range(n_layers):
        names.append(f"xk_{i}")
        names.append(f"xv_{i}")
    return tuple(names)


class _DiaDecoderWrapper(nn.Module):
    """`(codes, position_ids, attention_mask, xk_0, xv_0, ...) -> the LAST step's nine logit rows`.

    **The last-row slice is what keeps the nine-wide head from needing an engine primitive, and it is
    the one idea in this module worth carrying to the next family-10 leaf.** HF's head projects every
    row and reshapes to `[batch * 9, seq, 1028]`. An autoregressive step only ever reads the final row,
    so slicing `hidden[:, -1:, :]` before `logits_dense` leaves an output of `[1, 9, 1028]` -- nine rows
    of 1028 -- which is precisely the `[n_classes, n_rows]` tensor `loom.argmax_rows` was built for in
    P4.0.17 and which family 12 reused unchanged. Nine ids come back from one call, and no tensor
    crosses the Lua boundary.

    Two things fall out of it that are worth stating separately, because only the first is obvious:

    * the head runs on one row instead of `n_tokens` of them, which for a prefill is the same saving
      `PrefillDecodeLoop.head_topology` was created for on Qwen3-ASR -- 9252 floats per prompt row that
      nothing reads;
    * the graph's output shape stops depending on the token axis at all, so the reduction is the same
      shape on a prefill and on a decode step. A full-width head would have made the driver slice
      `9 * n_tokens` rows down to the last nine, which `loom.argmax_rows` cannot express and which
      would have been the new engine primitive.

    `encoder_attention_mask=None` for the same reason the encoder takes no mask: the encoder output is
    unpadded, so every frame is real and cross-attention attends over all of them.

    `use_cache=False` is what makes the trace cache-free, which is the shape `fuse_loom_attention`
    matches; the cache appears at run time, in the engine, not in the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.logits_dense = model.logits_dense
        self.n_channels = int(model.num_channels)
        self.vocab_size = int(model.vocab_size)
        self._cross = [None] * (2 * len(self.decoder.layers))
        for i, layer in enumerate(self.decoder.layers):
            layer.cross_attention.k_proj = _CrossKvSlot(self._cross, 2 * i)
            layer.cross_attention.v_proj = _CrossKvSlot(self._cross, 2 * i + 1)

    def forward(self, codes, position_ids, attention_mask, *cross):
        for i, tensor in enumerate(cross):
            self._cross[i] = tensor
        hidden = self.decoder(
            input_ids=codes, position_ids=position_ids, attention_mask=attention_mask,
            # `encoder_hidden_states` is what makes these blocks CROSS-attention
            # (`is_cross_attention = key_value_states is not None`), and nothing downstream of that
            # test reads it any more -- the projections that did are slots now. Passing `cross[0]`
            # keeps the flag true without declaring an input the graph would not otherwise use, the
            # same trick `_WhisperDecoderWrapper` plays.
            encoder_hidden_states=cross[0], encoder_attention_mask=None, use_cache=False,
        ).last_hidden_state
        return self.logits_dense(hidden[:, -1:, :]).view(1, self.n_channels, self.vocab_size)


@dataclass
class TextToCodesDiaExportConfig(BaseMultiPhaseModelExportConfig):
    """Dia as three traced phases -- `encoder` (bytes -> hidden states), `cross_kv` (those states ->
    per-layer cross-attention K/V, once), `decoder` (one cached, cross-attending step emitting nine
    logit rows) -- plus a driver that runs the first two once and loops the third.

    A `MultiPhase` config for the reason `ASRWhisperExportConfig` records at length and this family
    only confirms: the orchestration is two phases run once and a third looped, which is what
    `MultiPhase` + `MultiPhaseDriverBuilder` already are. What differs from Whisper is carried by
    fields on the pieces that own them -- `ExportPhase.declared_axes` for the second dynamic symbol,
    and the nine-wide reduction, which is a property of the traced output shape rather than of any
    component.
    """

    model_dir: str = ""
    architecture: str = "dia"
    output_path: str = "dia_mil.gguf"
    # The DECODER's own axis, and it counts codec frames rather than subword tokens. It is spelled
    # `n_tokens` regardless, because a KV-cached topology's axis is not a free choice: `GraphBuilder`
    # reads `n_tokens` directly out of `SymbolEnv` to size the cache's cell index
    # (`graph_builder.cpp`), so a cached phase that named its axis `n_codes` would build a graph whose
    # cache addresses an axis nothing bound. The honest name for the quantity is in `contract()`, where
    # a host reads it.
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "dia_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # The concrete lengths `torch.jit.trace` runs at. Free, and deliberately not 1: the graph must
    # contain a real axis for the RangeDim to make dynamic, and a length-1 trace gives coremltools a
    # size-1 axis it is entitled to fold away.
    trace_text_len: int = 24
    trace_steps: int = 8

    # Read off the checkpoint in `phases()`, which is the only moment the model is in hand. Fields
    # rather than recomputed because the driver components and `hparams()` need them after the trace.
    n_channels: Optional[int] = field(default=None, init=False, repr=False)
    n_codebook_vocab: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
    cross_kv_width: Optional[int] = field(default=None, init=False, repr=False)
    enc_hidden: Optional[int] = field(default=None, init=False, repr=False)
    max_position_embeddings: Optional[int] = field(default=None, init=False, repr=False)
    max_text_len: Optional[int] = field(default=None, init=False, repr=False)
    delay_pattern: tuple = field(default=(), init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    audio_tokens: dict = field(default_factory=dict, init=False, repr=False)
    # `generation_config.json`'s own `guidance_scale`, under Dia's own centring -- see `hparams()`.
    # 1.0 is "no guidance", which is what a checkpoint that does not name one means.
    guidance_scale: float = field(default=1.0, init=False, repr=False)
    sampling_defaults: dict = field(default_factory=dict, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory, already established by the recognizer's own detect(), which "
            "reads its config.json `model_type`. DiaForConditionalGeneration.from_pretrained raises "
            "on anything it cannot load."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the decoder ExportPhase's own Axis link, which is where the "
                               "value is actually used"),
        "driver_script_path": Unchecked("the hand-written fragments here are still parsed and "
                                        "cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction -- see the class docstring"),
        "trace_text_len": Unchecked(
            "the concrete text length torch.jit.trace runs at. The dynamic range is declared "
            "separately through ct.convert's own inputs=, so this constrains nothing the checkpoint "
            "could disagree with -- and `phases()` proves it by requiring an identical topology at a "
            "second length."
        ),
        "trace_steps": Unchecked("same, for the decoder's step axis"),
        "n_channels": Unchecked("READ off the checkpoint's own decoder config in phases() "
                                "(`num_channels`), not declared"),
        "n_codebook_vocab": Unchecked("same -- `decoder_config.vocab_size`, the width of one channel's "
                                      "logit row"),
        "n_layers": Unchecked("same -- `len(model.model.decoder.layers)`. It is a COUNT of exported "
                              "outputs rather than a claim about them: `cross_kv` emits two per layer."),
        "cross_kv_width": Unchecked(
            "same -- `cross_num_key_value_heads * cross_head_dim`, which is what `k_proj`/`v_proj` "
            "project to. Derived from the two config fields rather than assumed equal to "
            "`hidden_size`, because for this checkpoint it is NOT: 2048 against a 1024-wide encoder."
        ),
        "enc_hidden": Unchecked("same -- `encoder_config.hidden_size`"),
        "max_position_embeddings": Unchecked(
            "same -- `decoder_config.max_position_embeddings`, which is the KV cache capacity a decode "
            "loop can address"
        ),
        "max_text_len": Unchecked("same -- `encoder_config.max_position_embeddings`, the ct.RangeDim "
                                  "upper bound on the text axis"),
        "delay_pattern": Unchecked(
            "READ off the checkpoint's own config.json (`delay_pattern`), never derived. It is a "
            "per-checkpoint fact that the file states, which is exactly what ADR-020 says belongs in "
            "the driver rather than in the engine -- and `phases()` DOES cross-check the one property "
            "the driver's arithmetic depends on: that it has one entry per channel."
        ),
        "decoder_bindings": Unchecked(
            "(name, kind) per decoder input, derived in phases() from the SAME mil_inputs list the "
            "trace is declared with, through `exporter._binding_kind` -- so the driver cannot disagree "
            "with the trace about the order or the names."
        ),
        "cross_kv_names": Unchecked(
            "derived in phases() by `cross_kv_input_names(n_layers)`, which is also what orders "
            "`_DiaCrossKvWrapper`'s return tuple -- one function, so the decoder's input names, the "
            "phase's output order and the driver's `index` arithmetic cannot disagree."
        ),
        "audio_tokens": Unchecked(
            "READ off the checkpoint's own config in phases() (`bos_token_id`, `eos_token_id`, "
            "`pad_token_id`), never declared -- they are the scaffold the delay pattern is built out "
            "of, and the checkpoint is the only authority on them."
        ),
        "sampling_defaults": Unchecked(
            "READ off generation_config.json in phases(), through the same `read_sampling_defaults` "
            "every sampling family uses. There is no second authority on what a checkpoint asked to "
            "be decoded with, and a declaration here could only restate the file or contradict it."
        ),
        "guidance_scale": Unchecked(
            "same: the checkpoint's own generation config, verbatim and under its own centring. The "
            "conversion to the engine's form is one `+ 1` in the driver, where the convention it "
            "belongs to is."
        ),
    }

    def prepare_environment(self) -> None:
        install_rotate_half_patch()

    def load_model(self):
        from transformers import DiaForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        return DiaForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float32,
            # Load-bearing for the same reason [ADR-019] gives: the sdpa path expands masks to a
            # Python-level length. Dia reaches it through `_update_full_mask` and
            # `_update_cross_attn_mask`, both of which this family avoids by passing no mask at all --
            # but the flag is what makes that avoidance robust rather than incidental.
            attn_implementation="eager",
        ).eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        cfg = model.config
        dec_cfg = cfg.decoder_config
        enc_cfg = cfg.encoder_config

        self.n_channels = int(dec_cfg.num_channels)
        self.n_codebook_vocab = int(dec_cfg.vocab_size)
        self.n_layers = len(model.model.decoder.layers)
        self.cross_kv_width = int(dec_cfg.cross_num_key_value_heads) * int(dec_cfg.cross_head_dim)
        self.enc_hidden = int(enc_cfg.hidden_size)
        self.max_position_embeddings = int(dec_cfg.max_position_embeddings)
        self.max_text_len = int(enc_cfg.max_position_embeddings)
        self.delay_pattern = tuple(int(d) for d in cfg.delay_pattern)
        self.audio_tokens = {
            "BOS": int(cfg.bos_token_id),
            "EOS": int(cfg.eos_token_id),
            "PAD": int(cfg.pad_token_id),
        }
        self.sampling_defaults = read_sampling_defaults(self.model_dir)
        # `generation_config.json` rather than the model config: guidance is a DECODING knob, and the
        # generation config is where this checkpoint states the ones it was published with. Read
        # through `GenerationConfig` so a checkpoint that spells it only in the model config (as some
        # do) still reaches this, and defaulted to 1.0 -- the value that means "no guidance" both here
        # and in `transformers`, where the processor is not installed at all below it.
        self.guidance_scale = float(getattr(model.generation_config, "guidance_scale", None) or 1.0)

        # The delay pattern indexes channels, and the driver walks it as one array per frame. A
        # checkpoint whose pattern and channel count disagreed would produce a realignment that reads
        # off the end of a frame -- silently, since Lua returns nil rather than raising.
        #
        # **`DiaConfig` asserts this too, so today this is a backstop and not the guard.** It is here
        # anyway because the two numbers are read from DIFFERENT places -- `config.delay_pattern` and
        # `decoder_config.num_channels` -- and a transformers release that relaxed its own assertion
        # would otherwise reach the driver with nothing between. Cheap, and the failure it covers is
        # silent.
        if len(self.delay_pattern) != self.n_channels:
            raise ValueError(
                f"this checkpoint declares {len(self.delay_pattern)} delay-pattern entries "
                f"{self.delay_pattern} for {self.n_channels} channels. The driver offsets channel k by "
                f"delay_pattern[k], so the two must be the same length."
            )
        # `install_rotate_half_patch` rotates by `chunk(2, dim=-1)`, which on an ODD last dim splits
        # ceil/floor and rotates by the wrong amount without raising. Every dim it will ever see is a
        # head dim, and this checkpoint states three of them; all three are checked rather than the one
        # that happens to be shared. (They are all 128 here, which is exactly why checking only one
        # would look correct on this model and fail silently on the next.)
        head_dims = {
            "encoder head_dim": int(enc_cfg.head_dim),
            "decoder head_dim": int(dec_cfg.head_dim),
            "decoder cross_head_dim": int(dec_cfg.cross_head_dim),
        }
        odd = {name: dim for name, dim in head_dims.items() if dim % 2}
        if odd:
            raise ValueError(
                f"rotate_half splits the last dim in two with torch.chunk, which needs it EVEN; this "
                f"checkpoint declares {odd}. On an odd dim chunk(2) yields unequal halves and rotates "
                f"by the wrong amount without raising."
            )

        self.cross_kv_names = cross_kv_input_names(self.n_layers)

        step_axis = ct.RangeDim(1, self.max_position_embeddings)
        # A SEPARATE RangeDim instance from `step_axis`, which is the whole point: coremltools gives
        # each instance its own symbol, and the decoder's two axes are genuinely independent (a
        # sentence's byte count has nothing to do with how many codec frames it becomes). Shared, they
        # would collapse onto one name and the emitted shapes would be wrong rather than malformed.
        frame_axis = ct.RangeDim(1, self.max_text_len)

        decoder_inputs = [
            ct.TensorType(name="codes", shape=(1, step_axis, self.n_channels), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, step_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, step_axis, step_axis),
                          dtype=np.float32),
        ] + [
            ct.TensorType(name=name, shape=(1, frame_axis, self.cross_kv_width), dtype=np.float32)
            for name in self.cross_kv_names
        ]
        self.decoder_bindings = tuple((t.name, _binding_kind(t.name)) for t in decoder_inputs)

        # ORDER IS LOAD-BEARING: `_DiaCrossKvWrapper` captures the real projection modules, and
        # `_DiaDecoderWrapper.__init__` then replaces the attributes that reached them. Built the other
        # way round the cross_kv phase would trace `_CrossKvSlot`s and export no weights at all.
        cross_kv_wrapper = _DiaCrossKvWrapper(model).eval()
        decoder_wrapper = _DiaDecoderWrapper(model).eval()

        return [
            ExportPhase(
                name="encoder",
                wrapper=_DiaEncoderWrapper(model).eval(),
                dummy_inputs=(torch.zeros((1, self.trace_text_len), dtype=torch.long),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, ct.RangeDim(1, self.max_text_len)),
                                          dtype=np.int32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv_wrapper,
                dummy_inputs=(torch.zeros(1, self.trace_text_len, self.enc_hidden),),
                mil_inputs=[ct.TensorType(name="xa",
                                          shape=(1, ct.RangeDim(1, self.max_text_len), self.enc_hidden),
                                          dtype=np.float32)],
                # This phase never sees a decoder step, so its one axis is the encoder's frame count --
                # which for a byte-level text encoder is the input byte count, one frame each. The name
                # comes from `axes.py`, where Kokoro declared it for the structurally identical job:
                # the encoder-output length a downstream phase consumes.
                root_axis="n_enc_frames",
                # The unconditional stream's cross-attention K/V. Classifier-free guidance runs the
                # encoder twice -- over the caller's bytes and over an all-zero prompt of the same
                # length, which is `_prepare_model_inputs`' `torch.zeros_like(inputs)` -- and BOTH
                # results have to survive the whole generation, because the decode loop reads them at
                # every step. One module cannot hold two: the second run's retained output overwrites
                # the first's before the next step reads it.
                extra_streams=("cross_kv_uncond",),
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder_wrapper,
                dummy_inputs=(
                    torch.zeros((1, self.trace_steps, self.n_channels), dtype=torch.long),
                    torch.arange(self.trace_steps).unsqueeze(0),
                    causal_mask(self.trace_steps),
                ) + tuple(torch.zeros(1, self.trace_text_len, self.cross_kv_width)
                          for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                # **The second dynamic symbol.** Every one of the 36 cross-attention inputs shares the
                # `frame_axis` RangeDim instance, so they share one MIL symbol -- and substitution is
                # per SYMBOL, not per input, which is why `_reject_shared_symbol_overrides` requires
                # that every input carrying it be declared here rather than just one of them.
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                # The self-attention blocks become cached ATTENTION nodes; the cross-attention blocks
                # do not, and that is correct rather than a miss, for the reason `whisper_export`
                # states: `fuse_loom_attention` anchors on the `add(scores, mask)` that only a masked
                # block has, and this cross-attention has no mask at all.
                fuse_attention=True,
                kv_cache_size=self.max_position_embeddings,
                # The unconditional decode stream, with its OWN KV cache. It is fed the same codes and
                # the same positions as the conditional one -- `transformers` batches them, and the
                # decoder input ids are shared -- and differs only in which cross-attention K/V it
                # reads. Sharing one cache would make each step's second run overwrite the cell the
                # first just wrote and then attend to a mixture of the two histories, which produces
                # plausible tokens and raises nothing.
                extra_streams=("decoder_uncond",),
            ),
        ]

    def hparams(self) -> dict:
        """What a HOST must know to call this driver, or to interpret what comes back.

        `codec.n_codebooks` is the load-bearing one: the driver returns a FLAT array, frame-major, and
        a caller cannot cut it into frames without knowing the width. It is deliberately spelled the
        same as `audio_codec_export`'s own key, because it is the same number seen from the two ends of
        one pipeline -- a host that pipes this into a DAC GGUF compares them, and a mismatch is the
        whole class of composition error worth catching.
        """
        if self.n_channels is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        hparams = {
            "codec.n_codebooks": self.n_channels,
            "codec.codebook_size": self.n_codebook_vocab,
            "n_text_ctx": self.max_text_len,
            "n_codes_ctx": self.max_position_embeddings,
        }
        # **The checkpoint's own decoding defaults, declared rather than chosen.** This one asks to be
        # sampled -- `do_sample: true, temperature 1.8, top_k 50, top_p 0.9` -- and to be run with
        # classifier-free guidance at 3.0, and an export that dropped those would ship a model whose
        # audio is not what its authors published. `read_sampling_defaults` is the same reader
        # `causal_lm_export` uses, so a checkpoint declaring `do_sample: false` gets `temperature 0.0`,
        # which is the engine's spelling of greedy, here as there.
        hparams.update({f"sampling.{k}": v for k, v in read_sampling_defaults(self.model_dir).items()})
        # Guidance is Dia's own knob and not one of the three above, so it is read here. **The number
        # is the CHECKPOINT's, under the checkpoint's own convention** -- Dia centres its combination
        # on the conditional logits (`cond + g * (cond - uncond)`) where the general form centres on
        # the unconditional one, so the driver, not this, is where the `+1` happens. Declaring a
        # converted number would make `model.hparam("sampling.guidance_scale")` disagree with
        # `generation_config.json` for no one's benefit.
        hparams["sampling.guidance_scale"] = self.guidance_scale
        return hparams

    def contract(self) -> dict:
        """The task's pair, plus what a host needs to hand the codes on to a codec.

        The delay pattern is NOT here, and that is the ADR-020 line: it is a fact about how this LM
        emits, the driver undoes it before returning, and a host that received it would have nothing to
        do with it. What a host does need is the geometry of what comes back, which is `hparams()`.
        """
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams())

    def driver_components(self) -> List:
        """Encoder once, cross-attention K/V once, then the nine-channel decode loop.

        The first two are IR, so each is checked against its real traced topology by
        `SubgraphCallComponent`. The loop is a `LuaFragment` rather than `PrefillDecodeLoop`, and that
        is a statement about the loop rather than a shortcut: `PrefillDecodeLoop` reduces one row to
        one token and feeds that token back, and every part of this loop's body differs from it -- nine
        ids per step, a delay scaffold forcing some of them, a stop condition on channel 0 alone, and a
        realignment after the loop that no other family has. Its own `run_subgraph_and_retain('decoder',
        ...)` call is parsed out of the text and declared against the traced topology regardless.
        """
        from .driver_components import (
            DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Len, Lit, OutputRef

        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "N_CHANNELS": self.n_channels,
                "N_LAYERS": self.n_layers,
                "DELAY_PATTERN": list(self.delay_pattern),
                # Guarded for the same reason `hparams()` returns {} without a checkpoint:
                # `component_registry.usage()` builds every registered config with no model in hand, to
                # attribute driver components, and `phases()` is what fills these in. Whisper's empty
                # `prompt_constants` is the same accommodation.
                "MAX_DELAY": max(self.delay_pattern) if self.delay_pattern else 0,
                # `.get`, for the reason MAX_DELAY is guarded above: `usage()` introspects this list
                # without a checkpoint, and `phases()` is what reads these off the config.
                "BOS": self.audio_tokens.get("BOS", 0),
                "EOS": self.audio_tokens.get("EOS", 0),
                "PAD": self.audio_tokens.get("PAD", 0),
                "MAX_CODES": self.max_position_embeddings,
                # The checkpoint's own decoding defaults, handed to the driver as its `or`-fallbacks --
                # the same numbers `hparams()` writes for the HOST, rendered twice from one attribute
                # set, which is the arrangement `causal_lm_export` already uses for exactly this.
                "TEMPERATURE": self.sampling_defaults.get("temperature", 0.0),
                "TOP_K": self.sampling_defaults.get("top_k", 0),
                "TOP_P": self.sampling_defaults.get("top_p", 1.0),
                "GUIDANCE_SCALE": self.guidance_scale,
            }),
            SubgraphCallComponent(
                topology="encoder",
                # Retained, not bound to a local: `cross_kv` is the only reader and it runs
                # backend-side, so a Lua table here would marshal `n_bytes * 1024` floats for nothing.
                outputs=(),
                retain=True,
                inputs={"tokens": FieldAccess("inputs", "tokens")},
                axes={"n_tokens": Len(FieldAccess("inputs", "tokens")), "n_past": Lit(0)},
                note="Byte encoder: one pass over the caller's text, one frame per byte.",
            ),
            SubgraphCallComponent(
                topology="cross_kv",
                # Retained for a sharper version of the same reason: these are the tensors the decode
                # loop reads at every step, and the whole point of the phase is that they are produced
                # ONCE per utterance rather than re-projected per generated frame.
                outputs=(),
                retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": Len(FieldAccess("inputs", "tokens")), "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, computed once from the encoder "
                     "output instead of re-projected at every generated frame.",
            ),
            LuaFragment(
                self.driver_script_path / "01_decode.lua",
                reads=("N_CHANNELS", "N_LAYERS", "DELAY_PATTERN", "MAX_DELAY", "BOS", "EOS", "PAD",
                       "MAX_CODES", "TEMPERATURE", "TOP_K", "TOP_P", "GUIDANCE_SCALE"),
                defines=("_codes",),
            ),
            DriverReturn(values=("_codes",)),
        ]


def _hf_model_type(path: Path) -> Optional[str]:
    """An HF-style directory's own `config.json`'s `model_type`, or None if `path` isn't one. Never
    raises: `detect()` runs against unidentified paths by construction."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return config.get("model_type") if isinstance(config, dict) else None


def _is_dia(path: Path) -> bool:
    """An HF directory declaring `model_type == "dia"`.

    Specific rather than generic, like the codec family's and unlike family 12's: there is no
    `AutoModelForTextToCodes`, and the family-10 checkpoints this recognizer's siblings will cover
    (CSM, Orpheus, Parler) are structurally unrelated to each other -- different codecs, different
    channel counts, and in Parler's case a codec this tree cannot yet export at all. A generic
    recognizer here would claim checkpoints this wrapper cannot drive.
    """
    return _hf_model_type(path) == "dia"


def _build_dia(path: Path, output_path: str) -> TextToCodesDiaExportConfig:
    return TextToCodesDiaExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-codes",
        config_class=TextToCodesDiaExportConfig,
        recognizers=[ModelRecognizer(name="dia", detect=_is_dia, build_config=_build_dia)],
    ))
