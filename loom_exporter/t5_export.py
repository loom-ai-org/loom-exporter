"""The text encoder-decoder family (`EXPORT-ROADMAP.md` R5's family 6, P5), on `google/flan-t5-small`.

**Why this checkpoint.** The zoo had no encoder-decoder TEXT model and no Unigram/SentencePiece LM at
all; this one is 60M params / 308 MB, against the 2.24 GB of the other candidate. Its tokenizer ships
`spiece.model`, so it takes the existing `sentencepiece_proto` path unchanged -- it is not what
[ADR-027]'s `sentencepiece_json` family was for.

Structurally this is family 2/10's shape -- an encoder run once, then a KV-cached decoder
cross-attending to its output -- so `whisper_export` and `dia_export` are the modules to read beside
this one, and the `encoder`/`cross_kv`/`decoder` split is theirs. Dia is the closer of the two: its
encoder length is genuinely dynamic, which is what makes the decoder carry a SECOND symbol
(`n_enc_frames`) and need `declared_axes`.

**The one thing that is new here is T5's learned relative attention bias, and the whole design turns
on where it is computed.** T5 has no positional embedding. Instead `_relative_position_bucket` maps
(key - query) into 32 log-spaced buckets, indexes a `[32, n_head]` table and ADDS the result to the
attention scores -- recomputed for every (query, key) pair, i.e. per sequence length. The engine has
no such primitive, and `src/core/relative_position.cpp` is NOT it despite the name: that is VITS's
windowed `pad_crop_relative_embeddings`, a different mechanism.

It needs no primitive, and the reason is a property of the graph rather than a trick. HF computes
`position_bias = compute_bias(...) + causal_mask` ONCE per stack and hands the same tensor to every
layer, which then does `scores += position_bias` -- so the bias and the mask reach attention as ONE
additive `[1, n_head, q, k]` tensor, in exactly the place `fuse_loom_attention` already anchors on and
`ggml_soft_max_ext` already accepts (its mask may carry a head axis: `a->ne[2] % mask->ne[2] == 0`).
So this family hands that whole tensor in as the traced graph's mask input and lets the DRIVER build
it, which makes the bias a host-side reduction of a 32x6 table rather than anything the engine or the
graph has to learn:

* the two bias tables (encoder, decoder) are read off the checkpoint and become `ExportConstants` --
  192 floats each, the same order as Whisper's prompt table, and a number only the checkpoint knows is
  what that component is for;
* `t5_position_bias` in `t5_driver/00_header.lua` is `_relative_position_bucket` in Lua, and folds the
  causal mask into the same array for the decoder -- so the decoder's mask input is a mask in the
  engine's sense and `_retype_fused_mask_input` retypes it to `n_kv` with nothing new to check;
* every stack is run by walking `T5Stack.block` directly rather than by calling the stack, which is
  what lets `position_bias` be PASSED rather than computed. Calling the stack would trace
  `compute_bias`'s `torch.arange(query_length)` at the dummy length and bake it (`query_length` is a
  Python int off `hidden_states.shape`), and would add HF's own all-ones attention mask on top.

**Cross-attention is not fused and not cached**, which is `whisper_export`'s outcome for the same
reason stated one step further along: T5 hands cross-attention a zero bias (its blocks have no
relative-attention table), and MIL's `noop_elimination` removes `scores + 0` -- so there is no
`add(scores, mask)` for the pass to anchor on, and the block converts as an ordinary expanded softmax.
`_check_fused_attention` below asserts that outcome rather than trusting it: a release that kept the
zero add would silently give every cross-attention block a KV cache addressing the self-attention
blocks' slots.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


class _CrossKvSlot(nn.Module):
    """Stands in for a cross-attention `k`/`v` projection and returns a tensor handed in from outside.

    Identical in purpose to `whisper_export._CrossKvSlot` and `dia_export`'s: the projection it
    replaces is a function of the encoder output alone, so its result is the same at every decode step,
    and traced as-is it is recomputed per token -- for flan-t5-small, 16 matmuls of
    `[n_src, 512] x [512, 512]` on every step.

    A plain list rather than a buffer or submodule, deliberately: it carries per-CALL tensors, and
    registering them would make them state of the traced module.
    """

    def __init__(self, holder: list, key: int):
        super().__init__()
        self._holder = holder
        self._key = key

    def forward(self, x):
        return self._holder[self._key]


class _T5EncoderWrapper(nn.Module):
    """`(input_ids, position_bias) -> encoder hidden states`.

    **The block loop is the point, not a convenience.** `T5Stack.forward` computes the relative bias
    itself the first time a layer needs one, from `torch.arange(query_length)` where `query_length` is
    a Python int read off `hidden_states.shape` -- so tracing it bakes the dummy length into the graph.
    It also manufactures an all-ones `attention_mask` when handed none and adds `(1 - mask) * finfo.min`
    to the bias. Walking `self.enc.block` lets both be skipped: `T5Attention` takes the
    `if position_bias is None` branch only when it is None, and with `mask=None` there is nothing to
    add. What reaches every layer is then this call's own `position_bias` input, unmodified.

    `self.shared` rather than `self.enc.embed_tokens` -- they are the same module, and calling it under
    the name the DECODER wrapper also uses is what makes the 65 MB embedding matrix dedup in
    `merge_phase_weights` instead of being written twice. Nothing else may share a name between the two
    wrappers, which is why the stacks are `enc`/`dec` and not both `stack`.
    """

    def __init__(self, model):
        super().__init__()
        self.shared = model.shared
        self.enc = model.encoder

    def forward(self, input_ids, position_bias):
        hidden = self.shared(input_ids)
        for block in self.enc.block:
            hidden = block(hidden, attention_mask=None, position_bias=position_bias)[0]
        return self.enc.final_layer_norm(hidden)


class _T5CrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`, the cross-attention K/V for every decoder layer, once.

    **Construct this BEFORE `_T5DecoderWrapper`.** It holds the projection modules directly, so the
    decoder wrapper may then replace the `EncDecAttention.k`/`.v` attributes that used to reach them
    without this phase losing the weights it exports. Built the other way round it would trace
    `_CrossKvSlot`s and export nothing -- the same ordering constraint, for the same reason, as
    `whisper_export`'s and `dia_export`'s pairs.

    Interleaved k,v per layer rather than all-k-then-all-v, so the driver's `index` arithmetic is
    `2 * layer + 1` and reads as the pair it is.

    K and V leave in the same natural `[1, n_src, d_model]` layout, and V is deliberately NOT
    pre-transposed the way Whisper's is. That optimization is worth what the transpose costs, and here
    the transposed tensor is `n_src * 512` floats against Whisper's `1500 * 768` -- a prompt-length
    tensor for a text model, not a fixed 30 s of audio. It is a size/speed item for a later measurement,
    not a correctness one.
    """

    def __init__(self, model):
        super().__init__()
        self.projs = nn.ModuleList()
        for block in model.decoder.block:
            self.projs.append(block.layer[1].EncDecAttention.k)
            self.projs.append(block.layer[1].EncDecAttention.v)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


def cross_kv_input_names(n_layers: int) -> tuple:
    """`("xk_0", "xv_0", "xk_1", ...)` -- the decoder's per-layer cross-attention inputs, in the order
    `_T5CrossKvWrapper` returns them, which is the order their `index` binding assumes."""
    names = []
    for i in range(n_layers):
        names.append(f"xk_{i}")
        names.append(f"xv_{i}")
    return tuple(names)


class _T5DecoderWrapper(nn.Module):
    """`(tokens, position_bias, xk_0, xv_0, ...) -> logits`.

    `position_bias` is this family's mask input: the relative bias and the causal mask summed, which is
    the one tensor T5 adds to its scores (see the module docstring). It is therefore what
    `fuse_loom_attention` matches and what `_retype_fused_mask_input` gives the `n_kv` axis -- no
    separate `attention_mask`, and no `position_ids` either, since T5 has no absolute positions to
    index.

    `use_cache=False` is what makes the trace cache-free, which is the shape `fuse_loom_attention`
    matches; the cache appears at run time, in the engine, not in the graph (KV-CACHE.md §2).

    `encoder_decoder_position_bias` is a `[1, 1, 1, 1]` zero, and passing it is what keeps the
    cross-attention blocks from computing one: HF would otherwise build `torch.zeros((1, n_heads,
    seq_length, key_length))` from Python ints off the traced shapes. MIL's `noop_elimination` then
    removes the `scores + 0` entirely, which is what leaves cross-attention as an unfused, uncached
    softmax -- checked, not assumed, by `_check_fused_attention`.
    """

    def __init__(self, model):
        super().__init__()
        self.shared = model.shared
        self.dec = model.decoder
        self.lm_head = model.lm_head
        self._cross = [None] * (2 * len(model.decoder.block))
        for i, block in enumerate(self.dec.block):
            block.layer[1].EncDecAttention.k = _CrossKvSlot(self._cross, 2 * i)
            block.layer[1].EncDecAttention.v = _CrossKvSlot(self._cross, 2 * i + 1)
        # `T5Block.forward` evaluates `query_length=cache_position[-1] + 1` as an ARGUMENT to the
        # cross-attention call, before anything decides whether it is needed -- and it is not, because
        # this trace supplies `encoder_decoder_position_bias`. A one-element buffer keeps that
        # subscript legal; what it computes is dead and DCE removes it.
        self.register_buffer("_cache_position", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_no_cross_bias", torch.zeros(1, 1, 1, 1))

    def forward(self, tokens, position_bias, *cross):
        for i, tensor in enumerate(cross):
            self._cross[i] = tensor
        hidden = self.shared(tokens)
        for block in self.dec.block:
            hidden = block(
                hidden, attention_mask=None, position_bias=position_bias,
                # `encoder_hidden_states` is what makes these blocks CROSS-attention
                # (`is_cross_attention = key_value_states is not None`), and nothing downstream of that
                # test reads it any more -- the projections that did are slots now. Passing `cross[0]`
                # keeps the flag true without declaring an input the graph would not otherwise use.
                encoder_hidden_states=cross[0], encoder_attention_mask=None,
                encoder_decoder_position_bias=self._no_cross_bias,
                use_cache=False, cache_position=self._cache_position,
            )[0]
        hidden = self.dec.final_layer_norm(hidden)
        return self.lm_head(hidden)


def relative_attention_bias_table(stack) -> List[float]:
    """One T5 stack's `[num_buckets, n_head]` bias table, flattened row-major, as plain floats.

    Row-major so the driver's index is `bucket * n_head + head`, which is what `t5_position_bias`
    assumes -- the two spellings of that layout are here and in the Lua, and this is the one that
    reads it off the checkpoint.

    Only block 0 has the table; every later block reuses the tensor it returns, which is the same fact
    that makes one `position_bias` input serve the whole stack.
    """
    attention = stack.block[0].layer[0].SelfAttention
    if not attention.has_relative_attention_bias:
        raise ValueError(
            "this checkpoint's first block carries no `relative_attention_bias`, so there is no table "
            "to hand the driver. Every T5 stack puts one on block 0 and reuses its output for the rest; "
            "a checkpoint that does not is not the architecture this family exports."
        )
    return [float(v) for v in attention.relative_attention_bias.weight.detach().numpy().ravel()]


def _check_fused_attention(topo: dict, n_layers: int) -> int:
    """Exactly `n_layers` cached `ATTENTION` nodes in the decoder -- the self-attention blocks and no
    others.

    An `ExportPhase.topology_rewrite` that rewrites nothing, which is a use of the hook worth naming:
    the thing that must not drift here is an ABSENCE, and an absence has no output to diff. Whisper
    and Dia get cross-attention left alone because their cross blocks have no mask at all; T5's have a
    zero one, and what removes it is a coremltools optimization pass rather than anything in this tree.
    If a release stops folding `scores + 0`, the fusion claims 16 blocks instead of 8, the extra eight
    are handed `kv_cache=true` and `layer` indices 8..15, and the model decodes plausible tokens out of
    a cache half of which is the encoder's -- the failure Retro-006 is the standing warning about, on a
    file that matched its reference and still shipped unusable.

    Raises rather than repairing: this is a statement about what converted, and the fix is upstream of
    the topology.
    """
    cached = [node for node in topo["nodes"]
              if node["op"] == "ATTENTION" and node.get("attrs", {}).get("kv_cache", True)]
    total = [node for node in topo["nodes"] if node["op"] == "ATTENTION"]
    if len(cached) != n_layers or len(total) != n_layers:
        raise ValueError(
            f"t5 decoder: fused {len(total)} ATTENTION node(s) ({len(cached)} cached) for a stack with "
            f"{n_layers} self-attention blocks. Cross-attention must NOT fuse -- it has no mask once "
            f"MIL folds its zero bias away, and a cached cross-attention block would take a KV cache "
            f"slot the self-attention blocks address."
        )
    return n_layers


@dataclass
class Text2TextT5ExportConfig(BaseMultiPhaseModelExportConfig):
    """T5 as three traced phases -- `encoder`, `cross_kv`, `decoder` -- plus a driver that runs the
    first two once and loops the third.

    `MultiPhase` for the reason `whisper_export`'s docstring argues at length: the orchestration is two
    phases run once and a cached step in a loop, which `MultiPhaseDriverBuilder` already is. What is
    this family's own is where the relative bias comes from, and that is a driver component and a Lua
    helper rather than a decomposition.
    """

    model_dir: str = ""
    architecture: str = "t5"
    output_path: str = "t5_mil.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "t5_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # The lengths the two axes are TRACED at. Free, and deliberately not 1: the graph must contain a
    # real axis for each `RangeDim` to make dynamic, and a length-1 trace gives coremltools a size-1
    # axis it is entitled to fold away. Fields rather than locals so a test can export the same
    # checkpoint at two different pairs and require the topologies to be identical -- which is the only
    # way a baked length is visible at all (family 11's lesson, and Dia's own test).
    trace_tokens: int = 4
    trace_src: int = 6

    # Read off the checkpoint in `phases()`, which is the only moment the model is in hand. Fields
    # rather than recomputed values because the driver components and `hparams()` need them after the
    # trace; defaulted so `component_registry.usage()` can introspect this config with no checkpoint.
    n_head: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    inner_dim: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
    max_positions: Optional[int] = field(default=None, init=False, repr=False)
    num_buckets: Optional[int] = field(default=None, init=False, repr=False)
    max_distance: Optional[int] = field(default=None, init=False, repr=False)
    decoder_start_token_id: int = field(default=0, init=False, repr=False)
    eos_token_id: int = field(default=1, init=False, repr=False)
    encoder_bias: tuple = field(default=(), init=False, repr=False)
    decoder_bias: tuple = field(default=(), init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory, already established by the recognizer's own detect(), which "
            "reads its config.json `model_type`. T5ForConditionalGeneration.from_pretrained raises on "
            "anything it cannot load."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link, which is where the value "
                               "is actually used"),
        "driver_script_path": Unchecked("the one hand-written fragment here is the bias helper; its "
                                        "contents are still parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction -- see the class docstring"),
        "trace_tokens": Unchecked(
            "the length torch.jit.trace runs the decoder at. A property of the TRACE, not of the "
            "model -- and the claim that it does not reach the graph is not a spec claim either: it "
            "is checked by exporting twice at different lengths and diffing the topologies, which is "
            "the only check that can fail."
        ),
        "trace_src": Unchecked("same, for the source axis"),
        "n_head": Unchecked("READ off the checkpoint in phases() (`config.num_heads`), not declared"),
        "d_model": Unchecked("same -- `config.d_model`, the RESIDUAL width, which is what the encoder "
                             "emits and the cross-attention projections read"),
        "inner_dim": Unchecked(
            "same -- `config.num_heads * config.d_kv`, which is what those projections WRITE and is "
            "not the same number: flan-t5-small is 6 x 64 = 384 against a `d_model` of 512. T5 sizes "
            "its heads independently of its residual stream, and this family is the first in the tree "
            "where the two differ."
        ),
        "n_layers": Unchecked("same -- `len(model.decoder.block)`, a COUNT of exported outputs rather "
                              "than a claim about them: `cross_kv` emits two per decoder layer"),
        "max_positions": Unchecked("same -- `config.n_positions`, the length this checkpoint was "
                                   "trained at and the KV cache capacity a decode loop can address"),
        "num_buckets": Unchecked("same -- `config.relative_attention_num_buckets`. Cross-checked "
                                 "against the bias table's own row count in phases(), which is the "
                                 "second authority that exists."),
        "max_distance": Unchecked("same -- `config.relative_attention_max_distance`. No second "
                                  "authority: it is a scalar in the bucketing formula, not a shape."),
        "decoder_start_token_id": Unchecked("READ off the checkpoint's own generation config, which is "
                                            "the only authority on the token a decode starts from"),
        "eos_token_id": Unchecked("same -- and the vocabulary's own eos is written independently by "
                                  "the tokenizer export, which is what a host reads"),
        "encoder_bias": Unchecked(
            "the encoder stack's own `relative_attention_bias.weight`, flattened by "
            "`relative_attention_bias_table` -- weights read off the checkpoint, with no second "
            "authority to compare them against. What IS checked is the shape they imply: phases() "
            "cross-checks the row count against `relative_attention_num_buckets`."
        ),
        "decoder_bias": Unchecked("same, for the decoder stack"),
        "cross_kv_names": Unchecked(
            "derived in phases() by `cross_kv_input_names(n_layers)`, which is also what orders "
            "`_T5CrossKvWrapper`'s return tuple -- one function, so the decoder's input names, the "
            "phase's output order and the driver's `index` arithmetic cannot disagree. "
            "PrefillDecodeLoop's `inputs` link re-checks the names against the emitted topology."
        ),
        "decoder_bindings": Unchecked(
            "(name, kind) per decoder input, derived in phases() from the SAME mil_inputs list the "
            "trace is declared with, through `exporter._binding_kind` -- so the driver cannot disagree "
            "with the trace about the order or the names."
        ),
    }

    def load_model(self):
        from transformers import T5ForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        return T5ForConditionalGeneration.from_pretrained(self.model_dir, dtype=torch.float32).eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        cfg = model.config
        self.n_head = int(cfg.num_heads)
        self.d_model = int(cfg.d_model)
        # NOT `d_model`, and T5 is the first model here where that matters: `inner_dim` is
        # `num_heads * d_kv` (384 on flan-t5-small) while the residual stream is 512, so the
        # cross-attention K/V the decoder declares are narrower than the encoder output they come
        # from. Declaring them at `d_model` exports without complaint and fails at the first decode
        # step, on a shape the engine checks.
        self.inner_dim = int(cfg.num_heads) * int(cfg.d_kv)
        self.n_layers = len(model.decoder.block)
        self.max_positions = int(cfg.n_positions)
        self.num_buckets = int(cfg.relative_attention_num_buckets)
        self.max_distance = int(cfg.relative_attention_max_distance)
        gen_cfg = model.generation_config
        self.decoder_start_token_id = int(gen_cfg.decoder_start_token_id)
        self.eos_token_id = int(gen_cfg.eos_token_id)
        self.encoder_bias = tuple(relative_attention_bias_table(model.encoder))
        self.decoder_bias = tuple(relative_attention_bias_table(model.decoder))
        # The driver indexes the flattened table as `bucket * n_head + head`, so its length is the one
        # place the two declared numbers and the real tensor can be caught disagreeing. A table with
        # the wrong row count would not raise anywhere: the bucketing would return an index inside the
        # array and the driver would read some other bucket's bias for every pair.
        for name, table in (("encoder", self.encoder_bias), ("decoder", self.decoder_bias)):
            if len(table) != self.num_buckets * self.n_head:
                raise ValueError(
                    f"this checkpoint's {name} relative-attention table holds {len(table)} values, but "
                    f"its config declares {self.num_buckets} buckets x {self.n_head} heads = "
                    f"{self.num_buckets * self.n_head}. The driver indexes it as "
                    f"`bucket * n_head + head`, which a mismatched table satisfies silently."
                )
        self.cross_kv_names = cross_kv_input_names(self.n_layers)

        trace_tokens, trace_src = int(self.trace_tokens), int(self.trace_src)
        token_axis = ct.RangeDim(1, self.max_positions)
        # A SEPARATE RangeDim instance, which is the point: a source sentence's length and the number
        # of tokens generated from it are independent, and one shared instance would collapse them onto
        # one symbol and emit shapes that are wrong rather than malformed. Dia is the precedent.
        src_axis = ct.RangeDim(1, self.max_positions)

        decoder_inputs = [
            ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
            # NOT `attention_mask`, and the name is load-bearing twice over: it is what this family
            # hands the driver's own bias builder rather than `loom.causal_mask`, and `_binding_kind`
            # therefore reads it as a CALLER input that `PrefillDecodeLoop.bound` supplies.
            ct.TensorType(name="position_bias",
                          shape=(1, self.n_head, token_axis, token_axis), dtype=np.float32),
        ] + [
            ct.TensorType(name=name, shape=(1, src_axis, self.inner_dim), dtype=np.float32)
            for name in self.cross_kv_names
        ]
        self.decoder_bindings = tuple((t.name, _binding_kind(t.name)) for t in decoder_inputs)

        # ORDER IS LOAD-BEARING: `_T5CrossKvWrapper` captures the real projection modules, and
        # `_T5DecoderWrapper.__init__` then replaces the attributes that reached them.
        cross_kv_wrapper = _T5CrossKvWrapper(model).eval()
        decoder_wrapper = _T5DecoderWrapper(model).eval()

        return [
            ExportPhase(
                name="encoder",
                wrapper=_T5EncoderWrapper(model).eval(),
                dummy_inputs=(torch.zeros((1, trace_src), dtype=torch.long),
                              torch.zeros(1, self.n_head, trace_src, trace_src)),
                mil_inputs=[
                    ct.TensorType(name="input_ids", shape=(1, src_axis), dtype=np.int32),
                    ct.TensorType(name="position_bias",
                                  shape=(1, self.n_head, src_axis, src_axis), dtype=np.float32),
                ],
                root_axis="n_tokens",
                # Not fused, and that is the same per-phase decision Whisper's encoder makes: this is a
                # single bidirectional pass over the whole source, so a KV cache would be wrong. It is
                # also what keeps this phase's `position_bias` out of `_retype_fused_mask_input`'s way,
                # since an unfused mask is an ordinary input over the root axis.
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv_wrapper,
                dummy_inputs=(torch.zeros(1, trace_src, self.d_model),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, src_axis, self.d_model),
                                          dtype=np.float32)],
                # This phase never sees a decoder step, so its one axis is the encoder's frame count --
                # which for a text encoder is the source token count, one frame each. The name comes
                # from `axes.py`, where Kokoro declared it for the structurally identical job.
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder_wrapper,
                dummy_inputs=(
                    torch.zeros((1, trace_tokens), dtype=torch.long),
                    torch.zeros(1, self.n_head, trace_tokens, trace_tokens),
                ) + tuple(torch.zeros(1, trace_src, self.inner_dim) for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                # **The second dynamic symbol.** Every cross-attention input shares the `src_axis`
                # RangeDim instance, so they share one MIL symbol -- and substitution is per SYMBOL,
                # which is why every input carrying it must be declared here rather than just one.
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                # See `_check_fused_attention`: a rewrite that rewrites nothing, because what must not
                # drift here is the ABSENCE of a fused cross-attention block.
                topology_rewrite=lambda topo: _check_fused_attention(topo, self.n_layers),
                fuse_attention=True,
                kv_cache_size=self.max_positions,
            ),
        ]

    def hparams(self) -> dict:
        """What a HOST must know to call this driver at all.

        `n_ctx` is the one that is load-bearing: T5 has no absolute positions, so nothing in the graph
        stops a caller handing it a longer source than the checkpoint was trained on -- but the KV
        cache is built at this capacity and the relative buckets saturate past
        `relative_attention_max_distance`, so this is the length the file is honest about.

        Empty without a checkpoint, which is the same accommodation every family whose hparams are
        READ makes: `component_registry.usage()` builds every registered config with no model in hand
        to attribute driver components, and `phases()` is what fills this in.
        """
        return {"n_ctx": self.max_positions} if self.max_positions else {}

    def contract(self) -> dict:
        """The task's default pair, plus the two ids a decode loop needs from the checkpoint.

        `text.frontend = vocab` for the reason every text family declares it: this file carries the
        SentencePiece vocabulary that encodes the source sentence, so a host can offer a text door with
        nothing happening outside the engine.
        """
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        """The tokenizer travels with the model.

        `add_eos_token` is not a default and not a guess: `T5Tokenizer.build_inputs_with_special_tokens`
        appends `</s>` to every sequence it encodes, and a source sentence encoded without it is not
        the input this checkpoint was trained on. The proto carries no such flag, so the export states
        it -- the same split `spm_tokenizer_export`'s own docstring draws between what the protobuf is
        the authority on and what it is not.
        """
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams(),
                    add_eos_token=True, eos_token_id=self.eos_token_id)

    def driver_components(self) -> List:
        """Encoder once, cross-attention K/V once, then the decode loop.

        All three are IR, so each is checked against its real traced topology. The only hand-written
        Lua is `t5_position_bias`, which is `_relative_position_bucket` and nothing else -- arithmetic
        over the checkpoint's own table, with no `run_subgraph` call in it.
        """
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import ArrayLit, Call, FieldAccess, Len, Lit, OutputRef, Var

        src_len = Len(FieldAccess("inputs", "tokens"))
        bias_shape = [Var("N_HEAD"), Var("REL_BUCKETS"), Var("REL_MAX_DISTANCE")]
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "N_HEAD": self.n_head or 0,
                "REL_BUCKETS": self.num_buckets or 0,
                "REL_MAX_DISTANCE": self.max_distance or 0,
                # The two 32x6 tables, flattened `bucket * n_head + head`. A driver reads no GGUF
                # metadata, and these are the only model VALUES it needs -- 192 floats each, which is
                # the same order as Whisper's prompt table and three orders below any real weight.
                "ENC_REL_BIAS": list(self.encoder_bias),
                "DEC_REL_BIAS": list(self.decoder_bias),
            }),
            SubgraphCallComponent(
                topology="encoder",
                # Retained, not bound to a local: `cross_kv` is the only reader and it runs
                # backend-side, so a Lua table here would marshal `n_src * 512` floats for nothing.
                outputs=(),
                retain=True,
                inputs={
                    "input_ids": FieldAccess("inputs", "tokens"),
                    # Bidirectional (1) and no past: an encoder attends over the whole source, so
                    # there is no causal half to fold in and the array is the bias alone.
                    "position_bias": Call("t5_position_bias",
                                          [Var("ENC_REL_BIAS")] + bias_shape
                                          + [src_len, Lit(0), Lit(1)]),
                },
                axes={"n_tokens": src_len, "n_past": Lit(0)},
                note="Encoder: one bidirectional pass over the caller's source tokens.",
            ),
            SubgraphCallComponent(
                topology="cross_kv",
                # Retained for a sharper version of the same reason: these are the tensors the decode
                # loop reads at every step, and the whole point of the phase is that they are produced
                # ONCE per call rather than re-projected per generated token.
                outputs=(),
                retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": src_len, "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, computed once from the encoder "
                     "output instead of re-projected at every token.",
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                bound={
                    # `cross_kv` output `2 * layer + 1` is that layer's K and `+ 2` its V, which is the
                    # interleaving `_T5CrossKvWrapper` returns and `cross_kv_input_names` names.
                    **{name: OutputRef("cross_kv", index=i + 1)
                       for i, name in enumerate(self.cross_kv_names)},
                    # **Bound, but not constant across iterations** -- the one entry in this field that
                    # is not. `bound` is what `_call_inputs` consults FIRST and it is re-evaluated
                    # inside the loop body, so an expression over the loop's own `_n_tokens`/`_n_past`
                    # locals is rebuilt per step, which is exactly what a mask has to be. It is here
                    # rather than as a new binding kind because the kind would have to name a helper
                    # only this family has.
                    #
                    # Unidirectional (0): the array carries the causal mask as well as the bias, since
                    # T5 adds one tensor to its scores and the engine takes one mask.
                    "position_bias": Call("t5_position_bias",
                                          [Var("DEC_REL_BIAS")] + bias_shape
                                          + [Var("_n_tokens"), Var("_n_past"), Lit(0)]),
                },
                # The decode starts from the checkpoint's own start token, not from the caller's
                # tokens: those are the SOURCE, and they went to the encoder. This is the difference
                # between an encoder-decoder loop and a causal one, in one field.
                # The decoder's SECOND dynamic axis. Its cross-attention K/V are declared over the
                # source length, which the loop cannot derive from `n_tokens`/`n_past` -- and the
                # engine says so rather than guessing: an unbound symbol raises at graph build.
                extra_axes={"n_enc_frames": src_len},
                prompt=ArrayLit([Lit(self.decoder_start_token_id)]),
                default_eos_token=self.eos_token_id,
            ),
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


def _is_t5(path: Path) -> bool:
    """An HF directory declaring `model_type == "t5"`.

    Claims every T5 checkpoint and every fine-tune of one -- flan-t5, t5-v1.1, the translation
    checkpoints -- because they are the same architecture with different weights, and everything that
    varies (layer count, head count, bucket count, the bias tables themselves) is read off the
    checkpoint rather than assumed. It does NOT claim `mt5`/`umt5`/`longt5`, which spell their own
    `model_type` and differ in ways this wrapper does not cover: `longt5`'s local/transient-global
    attention is a different block entirely.
    """
    return _hf_model_type(path) == "t5"


def _build_t5(path: Path, output_path: str) -> Text2TextT5ExportConfig:
    return Text2TextT5ExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text2text-generation",
        config_class=Text2TextT5ExportConfig,
        recognizers=[ModelRecognizer(name="t5", detect=_is_t5, build_config=_build_t5)],
    ))
