"""
Loom's own MIL->MIL graph-rewrite passes (EXPORT-IMPROVEMENT-BACKLOG.md item 3).

Coremltools' own backend never mixes graph rewriting with serialization: rewrites run as real
`PassPipeline` stages over the pymil graph *before* backend translation
(`coremltools/converters/mil/backend/mil/load.py`'s `MILProtoExporter.translate_generic_op` is purely
mechanical/schema-driven). `exporter.py`'s `generate_graph_topology` used to interleave both -- detecting
and fusing the GQA `repeat_kv()` idiom inline, from inside the same walk that emits Loom JSON nodes. Pulling
that fusion out to run here, as a real MIL->MIL pass over the pymil graph before `generate_graph_topology`
ever sees it, makes the fusion testable directly against pymil graph structure (pattern match + replace)
instead of against Loom's derived JSON node list, and lets plain `common::dead_code_elimination` clean up
the original tile/reshape idiom's now-orphaned dependency chain (the "reps" computation subgraph --
gather/concat/equal/select/div -- HF traces alongside `repeat_kv()`) instead of a hand-rolled
backward-reachability walk over Loom's own node list.

Run via `apply_loom_mil_passes(prog)`, called once by `LoomGGUFExporter.export()` right after
`ct.convert(...)` has produced the `Program` it was constructed with, before any topology generation.
"""

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY, register_pass
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource

from . import dialect  # noqa: F401  registers "loom_broadcast_to" etc. (mb.loom_broadcast_to, ...)
from .value_facts import static_ints, static_value


def _is_int(d):
    return isinstance(d, (int, np.integer))


def _scope_ctx_like(op):
    """A `mb.scope(...)` context copying `op`'s own TORCHSCRIPT_MODULE_NAME (if any) onto every op
    built within it, so a rewrite pass's replacement ops keep attributing to the right decoder
    layer/submodule instead of relying on positional adjacency for any future scope-based tooling (see
    EXPORT-IMPROVEMENT-BACKLOG.md item 2's two real mis-attribution bugs)."""
    scope = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME) if op.scopes else None
    if scope:
        return mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=list(scope)))
    return mb.scope()


@register_pass(namespace="loom")
class fuse_gqa_repeat_kv(AbstractGraphPass):
    """
    Detects HF's standard `repeat_kv()` idiom -- a tile of a size-1 axis immediately merged back into an
    adjacent axis by a reshape, i.e. `unsqueeze -> tile -> reshape` -- and replaces the `tile`/`reshape`
    pair with an equivalent `reshape -> tile -> reshape` sequence built from provably-reliable shape
    information, entirely bypassing the pattern's own poisoned intermediate shape inference.

    Ported from `exporter.py`'s former `_try_fuse_gqa_repeat_kv` (see EXPORT-BACKLOG.md item 3 and
    EXPORT-IMPROVEMENT-BACKLOG.md item 3), with the derivation reworked to operate directly on real MIL
    `Var.shape` tuples in their natural (forward) axis order instead of Loom's ne-order/string-shape
    representation -- that representation only existed to serialize into Loom's JSON topology format, and
    isn't needed to *identify* the pattern.

    Why this pattern needs special handling at all: coremltools reports `tile`'s own `reps` input as
    non-constant here (`reps.val is None`), because `n_rep` -- an architecturally FIXED hyperparameter --
    gets computed via a runtime shape query during tracing anyway. That poisons shape inference for the
    tile's output and (empirically) for every axis of the following reshape's output too, not just the
    tiled one -- so the fusion derives every non-changed axis from `tile`'s own pre-expand *input* shape
    (unaffected by any of this, reliable by construction) rather than trusting the reshape's declared
    output shape anywhere except the one axis whose change we can positively confirm via a concrete-int
    comparison.

    Why a plain single `REPEAT`/`tile` of the original (pre-unsqueeze) tensor is NOT equivalent: a single
    `ggml_repeat`/MIL `tile` block-tiles an axis (`dst[i] = src[i % ne_src]`, i.e. concatenating whole
    copies: kv0,kv1,...,kv7,kv0,kv1,...,kv7), while `repeat_kv()`'s unsqueeze->expand->reshape-merge idiom
    produces an *interleaved* repeat (`dst[i] = src[i // n_rep]`, i.e. kv0,kv0,kv1,kv1,...,kv7,kv7) -- the
    standard GQA head-group convention. These only agree when n_rep==1. The replacement composes three
    ops that reproduce the interleaved semantics exactly: (1) RESHAPE the pre-tile tensor to insert a
    genuine size-1 axis in the position the real (un-collapsed) unsqueeze put it -- a pure relabeling,
    moves no data since the axis being vacated (batch) is already size 1; (2) TILE that size-1 axis up to
    `n_rep` -- always safe regardless of block-tile-vs-interleave semantics, since tiling a *single* source
    element by any tiling scheme yields the same output; (3) RESHAPE again to merge the now-`n_rep`-sized
    axis into the adjacent kv-heads axis, with `n_rep` as the faster-varying component of the pair --
    exactly reproducing `dst[i] = src[i // n_rep]` via a plain contiguous axis-merge.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._fuse_block(f)

    @block_context_manager
    def _fuse_block(self, block):
        for op in list(block.operations):
            # `getattr(..., block)` rather than a bare `op.enclosing_block`: this exporter's own "bespoke"
            # workflow accepts hand-built Programs containing synthetic, duck-typed ops standing in for
            # ops MIL itself doesn't have (see test_compiler.py's MockOperation), which don't carry a real
            # Operation's full attribute set. Defaulting to `block` (truthy) treats "attribute missing" the
            # same as "still present" -- correct, since a mock op was never removed by this pass.
            if getattr(op, "enclosing_block", block) is None:
                # Already removed by an earlier fusion in this same walk.
                continue
            for b in op.blocks:
                self._fuse_block(b)
            if op.op_type != "tile":
                continue
            child_ops = list(op.outputs[0].child_ops) if op.outputs else []
            if len(child_ops) != 1 or child_ops[0].op_type != "reshape":
                continue
            self._try_to_transform(op, child_ops[0], block)

    @staticmethod
    def _try_to_transform(tile_op, reshape_op, block) -> bool:
        pre_tile_x = tile_op.inputs.get("x")
        if pre_tile_x is None:
            return False
        # `tile_op`'s own "x" is itself the output of the preceding unsqueeze (traced as MIL
        # "expand_dims"), NOT the genuine pre-expand tensor -- walk back through it to find the tensor
        # whose rank actually matches the reshape's own (merged) output rank.
        producer = pre_tile_x.op
        if producer is not None and producer.op_type == "expand_dims":
            inner_x = producer.inputs.get("x") or producer.inputs.get("data")
            if inner_x is not None:
                pre_tile_x = inner_x
        if pre_tile_x.shape is None:
            return False
        pre_shape = tuple(pre_tile_x.shape)
        pre_rank = len(pre_shape)

        out_var = reshape_op.outputs[0]
        if out_var.shape is None or len(out_var.shape) != pre_rank:
            return False
        out_shape = tuple(out_var.shape)

        # Find the one axis whose OUTPUT dim is a reliable (concrete-int) value that differs from the
        # pre-expand input's own dim at the same position. Every other axis is derived from `pre_shape`
        # alone below, never from `out_shape` -- see the class docstring for why `out_shape`'s other axes
        # can't be trusted here.
        changed_axis = None
        for i in range(pre_rank):
            d = out_shape[i]
            if _is_int(d) and (not _is_int(pre_shape[i]) or int(d) != int(pre_shape[i])):
                changed_axis = i

        # The changed axis can never be the fastest-varying (last, MIL-order) axis -- repeat_kv() only
        # ever grows the heads axis, never head_dim.
        if changed_axis is None or changed_axis == pre_rank - 1:
            return False
        if not _is_int(pre_shape[changed_axis]):
            return False

        kv_count = int(pre_shape[changed_axis])
        out_count = int(out_shape[changed_axis])
        if kv_count <= 0 or out_count % kv_count != 0:
            return False
        ratio = out_count // kv_count
        if ratio == 1:
            return False

        # ggml caps tensors at 4 dims -- making room for a genuine new axis at `changed_axis` (by
        # dropping the leading axes before it) only works if that dropped prefix's product is 1, i.e. it's
        # just the (always size-1, batch=1 on this roadmap) leading axis. Verify rather than assume.
        if pre_rank != 4 or not _is_int(pre_shape[0]) or int(pre_shape[0]) != 1:
            return False

        def entry(d):
            # Any non-concrete (symbolic/dynamic) dim collapses to -1, delegating to MIL reshape's own
            # numpy-style single-inferred-axis inference -- valid here because this whole exporter only
            # ever targets models with exactly one true dynamic quantity (sequence length; see
            # exporter.py's `get_var_info` for the full invariant), so at most one entry is ever -1.
            return int(d) if _is_int(d) else -1

        tail = [entry(d) for d in pre_shape[changed_axis + 1:]]
        # (1) insert a genuine size-1 axis right after the (unchanged) kv-heads dim, pushing the
        # always-size-1 batch axis out of the shape entirely -- a pure relabeling of the same flat data.
        reshape1_shape = [entry(pre_shape[changed_axis]), 1] + tail
        # (2) grow that new size-1 axis to `ratio`.
        repeat_reps = [1, ratio] + [1] * len(tail)
        # (3) merge (ratio, kv_count) back into one axis of size `ratio*kv_count`, restoring the original
        # leading axes (batch) that were dropped from (1)/(2).
        final_shape = [entry(d) for d in pre_shape[:changed_axis]] + [out_count] + tail

        out_name = out_var.name

        # Preserve the torch-module scope of the op being replaced (if any) on all three new ops --
        # `try_replace_uses_of_var_after_op` below only auto-copies scope onto the LAST new op (the one
        # whose var directly replaces `out_var`), and downstream scope-based tooling (debugging, any
        # future scope-partitioned discovery aid) needs every op it walks to carry the correct
        # TORCHSCRIPT_MODULE_NAME to attribute it to the right decoder layer -- relying on the two
        # intermediate ops merely landing in the right slice by positional adjacency would be exactly the
        # class of fragile mis-attribution EXPORT-IMPROVEMENT-BACKLOG.md item 2 already documents two real
        # bugs from.
        with _scope_ctx_like(tile_op):
            r1 = mb.reshape(x=pre_tile_x, shape=reshape1_shape, name=out_name + "_gqa_unsqueeze", before_op=tile_op)
            rep = mb.tile(x=r1, reps=repeat_reps, name=out_name + "_gqa_repeat", before_op=tile_op)
            r2 = mb.reshape(x=rep, shape=final_shape, name=out_name, before_op=tile_op)

        if not reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=out_var, new_var=r2,
        ):
            return False
        block.remove_ops([tile_op, reshape_op])
        return True


@register_pass(namespace="loom")
class normalize_matmul(AbstractGraphPass):
    """
    Rewrites `matmul(x, y, transpose_x=True, transpose_y=ty)` into the equivalent
    `matmul(transpose(x), y, transpose_x=False, transpose_y=ty)` -- EXPORT-ROADMAP.md R2a.

    `topology_ops.py`'s matmul rule table only ever composed a correct ggml lowering for
    `transpose_x=False` (see its own comment on `ggml_mul_mat`'s fixed contraction convention): every
    other combination fell through to `_op_matmul_unsupported`, documented there as "only
    transpose_x=False has been needed so far" rather than a real ceiling. Running this pass before
    `generate_graph_topology` ever walks the graph means every matmul it sees already has
    transpose_x=False, so the table's two existing composed rules -- (False, True) and (False, False)
    -- cover every matmul in the program, closing that gap without adding a third composition.

    A pure rewrite, not a new op: `transpose` and `matmul` (with transpose_x=False) are both already
    fully composed by `topology_ops.py`, so this only ever removes a guard, never adds one.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                # Already removed by an earlier rewrite in this same walk.
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "matmul":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        if not bool(static_value(op.inputs.get("transpose_x"), False)):
            return False
        x = op.inputs.get("x")
        y = op.inputs.get("y")
        if x is None or y is None or x.shape is None:
            return False
        rank = len(x.shape)
        if rank < 2:
            # matmul's own "promote 1-D x to a matrix" rule (see its docstring) never sets
            # transpose_x=True for a 1-D operand in practice -- guard rather than assume.
            return False
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        transpose_y = bool(static_value(op.inputs.get("transpose_y"), False))
        out_name = op.outputs[0].name

        with _scope_ctx_like(op):
            xt = mb.transpose(x=x, perm=perm, name=f"{out_name}_normalize_matmul_xt", before_op=op)
            new_out = mb.matmul(x=xt, y=y, transpose_x=False, transpose_y=transpose_y,
                                 name=out_name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_out,
        ):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class insert_explicit_broadcasts(AbstractGraphPass):
    """
    Rewrites an `add`/`mul` whose two operands need MUTUAL (different-axis) broadcasting -- each
    operand is size-1 on a DIFFERENT axis than the other, so neither is simply "the other's shape with
    some 1s" (`ggml_add`/`ggml_mul` only ever let ONE operand broadcast into the other's already-correct
    shape) -- into two explicit `loom_broadcast_to` ops feeding a plain `add`/`mul` whose operands are
    already at matching shape. EXPORT-ROADMAP.md R2a.

    This used to be a shape-string comparison the EMITTER itself performed (`exporter.py`'s add/mul
    case in `transpile_operation`), deciding whether to splice `REPEAT` nodes into the JSON node list
    by rendering both operands' shapes and checking for "1" vs. not. Running this as a real graph
    rewrite before `generate_graph_topology` ever walks the program means the emitter never has to look
    at either operand's shape at all: by the time it sees this op, both operands are already
    broadcast-compatible.

    First confirmed needed on SupertonicTTS's fractional-RoPE angle computation (`theta[d] *
    frac_pos[pos]`, ne=[32,1,1] * ne=[1,L,1] -> ne=[32,L,1], L dynamic) -- see `loom_broadcast_to`'s own
    docstring in `dialect.py` for why lowering to a real graph op (rather than conjuring a `REPEAT` JSON
    node ad hoc at emission time) is what lets `topology_ops.py`'s already-dynamic-shape-aware `REPEAT`
    lowering handle it unmodified.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type not in ("add", "mul"):
                continue
            self._try_transform(op, block)

    @staticmethod
    def _needs_broadcast(shape, out_shape):
        """True iff some axis of `shape` is a literal 1 while the SAME axis of `out_shape` isn't --
        the same "1 vs. not-1" test the old string-comparison code ran on rendered shape expressions,
        but directly on MIL's own shape tuples: a concrete-int axis is unambiguous either way, and a
        genuinely dynamic (symbolic) axis never renders as the literal string "1", so raw-shape ints
        and rendered-shape strings agree on every case this ever needs to distinguish."""
        return any(_is_int(s) and int(s) == 1 and not (_is_int(t) and int(t) == 1)
                    for s, t in zip(shape, out_shape))

    @classmethod
    def _try_transform(cls, op, block) -> bool:
        x = op.inputs.get("x")
        y = op.inputs.get("y")
        if x is None or y is None or x.shape is None or y.shape is None:
            return False
        out_var = op.outputs[0]
        if out_var.shape is None:
            return False
        out_shape = tuple(out_var.shape)
        if len(x.shape) != len(out_shape) or len(y.shape) != len(out_shape):
            return False
        if not (cls._needs_broadcast(x.shape, out_shape) and cls._needs_broadcast(y.shape, out_shape)):
            return False

        # `like=` the ORIGINAL other operand, not `out_var` itself -- `out_var` is produced by `op`,
        # which both new ops are inserted BEFORE, so using it here would be a data-dependency cycle.
        # `infer_type_with_broadcast(x, y)` gives the identical shape `op`'s own type inference already
        # computed for `out_var`, so this loses no information.
        node_tag = op.name
        with _scope_ctx_like(op):
            bx = mb.loom_broadcast_to(x=x, like=y, name=f"{node_tag}_bcast_x", before_op=op)
            by = mb.loom_broadcast_to(x=y, like=x, name=f"{node_tag}_bcast_y", before_op=op)
            builder_fn = mb.add if op.op_type == "add" else mb.mul
            new_out = builder_fn(x=bx, y=by, name=out_var.name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=out_var, new_var=new_out,
        ):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class canonicalize_replicate_pad(AbstractGraphPass):
    """
    Rewrites a `pad(mode="replicate")` op into a `loom_replicate_pad` op -- EXPORT-ROADMAP.md R2.

    First (and so far only) needed by SupertonicTTS's `ConvNextBlock` (used by every encoder/decoder in
    that model), which pads via `nn.functional.pad(x, pad, mode="replicate")` before every depthwise
    conv. `topology_ops.py`'s `pad` rule used to decide, at emission time, whether `mode` was
    "replicate" and if so compose VIEW/REPEAT/CONCAT inline; this pass makes that decision once, before
    emission, so the emitter just dispatches on op type like any other.

    Validates the same invariants `topology_ops.py`'s old inline code did -- pad values must be
    compile-time constants, and only the fastest-varying (last, MIL-order) axis may have a non-zero
    replicate pad (the only shape ggml's composed-from-primitives approach here can express) -- raising
    the same errors that code raised, just earlier (right after `ct.convert()`, not mid-emission).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "pad":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        mode = static_value(op.inputs.get("mode"), "constant")
        if mode != "replicate":
            return False
        x = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
        if x is None or x.shape is None:
            raise NotImplementedError(
                f"pad op '{op.name}' has mode='replicate' but an input with no known rank."
            )
        pad_vals = static_ints(op.inputs.get("pad"))
        if pad_vals is None or len(pad_vals) % 2 != 0:
            raise NotImplementedError(
                f"pad op '{op.name}' has mode='replicate' with a non-constant or odd-length 'pad' "
                "input, which this exporter doesn't support."
            )
        n_padded = len(pad_vals) // 2
        rank = len(x.shape)
        lp0 = rp0 = 0
        for i in range(n_padded):
            mil_axis = rank - n_padded + i
            lp, rp = pad_vals[2 * i], pad_vals[2 * i + 1]
            if lp == 0 and rp == 0:
                continue
            if mil_axis != rank - 1:
                raise NotImplementedError(
                    f"pad op '{op.name}' pads MIL axis {mil_axis} (non-zero {lp}/{rp}) with "
                    "mode='replicate', but this exporter only supports replicate-padding the "
                    "fastest-varying axis (ne[0]/MIL's last axis) -- padding any other axis needs a "
                    "new C++ primitive first."
                )
            lp0, rp0 = lp, rp

        out_var = op.outputs[0]
        if lp0 == 0 and rp0 == 0:
            # A genuine identity pad (every entry zero) -- just alias the op away entirely.
            if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=x):
                return False
            block.remove_ops([op])
            return True

        with _scope_ctx_like(op):
            new_out = mb.loom_replicate_pad(x=x, lp=lp0, rp=rp0, name=out_var.name, before_op=op)
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=new_out):
            return False
        block.remove_ops([op])
        return True


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]


@register_pass(namespace="loom")
class canonicalize_conv_transpose_dw(AbstractGraphPass):
    """
    Rewrites a depthwise (`groups == in_channels == out_channels`) `conv_transpose` into a
    `loom_conv_transpose_dw` op -- EXPORT-ROADMAP.md R2.

    First (and so far only) needed by Kokoro's `AdainResBlk1d` upsample "pool" (`ConvTranspose1d
    (kernel=3, stride=2, groups=dim_in, padding=1, output_padding=1)`), also reused by StyleTTS2's
    driver. `topology_ops.py`'s `conv_transpose` rule used to decide, at emission time, whether `groups`
    made this depthwise and if so compose the zero-stuff-then-depthwise-conv identity inline; this pass
    makes that decision once, before emission.

    Validates the same invariants `topology_ops.py`'s old inline code did -- only a true 1D depthwise
    case (`is_2d=False`, `groups == in_channels`, one output channel per group), zero dilation, zero pad
    (every depthwise conv_transpose this exporter has seen traces with `pad=[0,0]`, deferring any real
    crop to a separate downstream `slice_by_index`), and a compile-time-constant weight (needed to flip
    the kernel) -- raising the same errors that code raised for anything else, just earlier (right after
    `ct.convert()`, not mid-emission). A non-depthwise (`groups == 1`) `conv_transpose` is untouched,
    left for `topology_ops.py`'s own `conv_transpose` rule exactly as before.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "conv_transpose":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        groups = static_value(op.inputs.get("groups"), 1)
        g_val = int(_as_list(groups)[0])
        if g_val == 1:
            return False

        pad_type = static_value(op.inputs.get("pad_type"), "valid")
        if pad_type not in ("valid", "custom"):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has pad_type='{pad_type}', which this exporter "
                "doesn't support (only 'valid' and a 'custom' symmetric-crop composition exist)."
            )
        strides_list = _as_list(static_value(op.inputs.get("strides"), [1]))
        is_2d = len(strides_list) == 2
        if is_2d and pad_type == "custom":
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is 2D with pad_type='custom' -- only the 1D "
                "crop composition has been needed/written so far."
            )
        dilations = _as_list(static_value(op.inputs.get("dilations"), [1]))
        if any(int(d) != 1 for d in dilations):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has non-unit 'dilations' {dilations!r}, which "
                "this exporter doesn't support."
            )

        x = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
        weight = op.inputs.get("weight")
        if x is None or x.shape is None or weight is None or weight.shape is None:
            return False
        in_channels = int(x.shape[1])
        out_per_group = int(weight.shape[1])
        if is_2d or g_val != in_channels or out_per_group != 1:
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has groups={g_val} (in_channels={in_channels}, "
                f"out_channels/group={out_per_group}) -- only a true 1D depthwise case "
                "(groups == in_channels == out_channels) is composed; anything else has no "
                "ggml-side implementation yet."
            )
        pad_list = _as_list(static_value(op.inputs.get("pad"), [0]))
        if any(int(p) != 0 for p in pad_list):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is depthwise with non-zero pad={pad_list!r} -- "
                "every depthwise conv_transpose this exporter has seen traces with pad=[0,0] "
                "(deferring any real crop to a separate downstream slice_by_index op); this "
                "composition doesn't know how to fold a non-zero pad in directly."
            )
        if static_value(weight) is None:
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is depthwise but its weight isn't a resolved "
                "constant -- this composition needs to flip the kernel at export time."
            )

        s0 = int(strides_list[0])
        bias = op.inputs.get("bias")
        out_name = op.outputs[0].name
        with _scope_ctx_like(op):
            new_out = mb.loom_conv_transpose_dw(x=x, weight=weight, bias=bias, stride=s0,
                                                 name=out_name, before_op=op)
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class lower_stack(AbstractGraphPass):
    """
    Rewrites `stack(values, axis)` into `concat([expand_dims(v, axes=[axis]) for v in values], axis)`
    -- EXPORT-ROADMAP.md R2. Unlike every other pass in this module, this introduces no new dialect op:
    `expand_dims` and `concat` are both already-real MIL ops with their own full, general
    `topology_ops.py` rules (`reshape`/`expand_dims` and `concat`), so this is a pure lowering -- once it
    runs, `topology_ops.py` no longer needs a dedicated `stack` composition at all, and gets the exact
    same N-ary CONCAT-chaining logic `concat` already has, instead of a second, parallel copy of it.

    First (and so far only) needed by a hand-rolled conv-based STFT's real/imag parts
    (`torch.stack([real, imag], dim=-1)`, seen when a model computes its DFT via CONV_1D kernels
    directly rather than `torch.stft`, which decomposes differently via coremltools' own
    `lower_complex_dialect_ops`).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "stack":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        values = op.inputs.get("values")
        if not values:
            return False
        real_values = [v for v in values if isinstance(v, Var)]
        if not real_values:
            return False
        out_var = op.outputs[0]
        if out_var.shape is None:
            return False
        out_rank = len(out_var.shape)
        axis_val = int(static_value(op.inputs.get("axis"), 0))
        axis = axis_val + out_rank if axis_val < 0 else axis_val

        with _scope_ctx_like(op):
            if len(real_values) == 1:
                # A single real operand -- still a genuine rank-increasing op, not an identity, so
                # this is the op's own real output name directly rather than an intermediate one.
                new_out = mb.expand_dims(x=real_values[0], axes=[axis], name=out_var.name, before_op=op)
            else:
                expanded = [
                    mb.expand_dims(x=v, axes=[axis], name=f"{out_var.name}_stack_unsq_{i}", before_op=op)
                    for i, v in enumerate(real_values)
                ]
                new_out = mb.concat(values=expanded, axis=axis, name=out_var.name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class lower_reduce_mean(AbstractGraphPass):
    """
    Rewrites a single-axis `reduce_mean` into whichever of two real ops its reduced axis's countability
    allows -- EXPORT-ROADMAP.md R2:

    * a statically-known reduction count -> `reduce_sum` (already a real, general MIL op with its own
      `topology_ops.py` rule) followed by `loom_scale(n)` (dividing by `n`; see that op's own docstring
      for why it carries `n` rather than the pre-divided `1/n`);
    * a run-time-only count, but on the fastest-varying (last, MIL-order) axis -> `loom_mean`, which
      `ggml_mean` can reduce natively (it supplies its own count at run time);
    * anything else (a run-time-only count on any other axis, or a genuine multi-axis reduction) is
      unrepresentable and raises -- the same errors `topology_ops.py`'s old two guards raised, just
      earlier (right after `ct.convert()`, not mid-emission).

    This is what makes all three outcomes explicit and pass-driven, rather than two of them being
    `topology_ops.py` guards and the third an unstated fall-through to the generic OP_MAP path (which
    happened to already do the right thing for the ne[0] case, but only because nothing else claimed
    "reduce_mean" first).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "reduce_mean":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        x = op.inputs.get("x")
        axes_val = static_ints(op.inputs.get("axes"))
        if x is None or x.shape is None or axes_val is None or len(axes_val) != 1:
            raise NotImplementedError(
                f"reduce_mean op '{op.name}': only a single reduction axis is supported "
                f"(got axes={static_value(op.inputs.get('axes'))!r}); a genuine multi-axis case "
                "(e.g. GroupNorm, see group_norm_op.py) needs its own composition."
            )
        rank = len(x.shape)
        axis = axes_val[0]
        torch_axis = axis + rank if axis < 0 else axis
        if not (0 <= torch_axis < rank):
            return False
        ne_axis = rank - 1 - torch_axis
        n_raw = x.shape[torch_axis]
        keep_dims = bool(static_value(op.inputs.get("keep_dims"), False))
        out_name = op.outputs[0].name

        if _is_int(n_raw):
            n = int(n_raw)
            with _scope_ctx_like(op):
                summed = mb.reduce_sum(x=x, axes=[axis], keep_dims=keep_dims,
                                        name=f"{out_name}_rmean_sum", before_op=op)
                new_out = mb.loom_scale(x=summed, n=n, name=out_name, before_op=op)
        elif ne_axis == 0:
            with _scope_ctx_like(op):
                new_out = mb.loom_mean(x=x, keep_dims=keep_dims, name=out_name, before_op=op)
        else:
            raise NotImplementedError(
                f"reduce_mean op '{op.name}': reducing ne axis {ne_axis} over a count that is only "
                "known at run time has no composition here. REDUCE_SUM + SCALE needs the count at "
                "export time, and ggml_mean supplies its own count only for ne[0]. A dynamic count on "
                "another axis needs its own composition (see loom_group_norm's custom-op bridge for "
                "that case)."
            )

        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class fuse_rms_norm(AbstractGraphPass):
    """
    Replaces the five-op chain PyTorch's RMSNorm traces to with one `loom_rms_norm`, which
    `topology_ops.py` lowers to the engine's `RMS_NORM` primitive.

        pow(x, 2) -> reduce_mean(axes=[-1], keep_dims) -> add(eps) -> rsqrt -> mul(x, .)

    **Unconditional, unlike `fuse_loom_attention`.** That pass is opt-in because fusing changes what a
    model MEANS -- an ATTENTION node can reach a KV cache, which is wrong for a non-autoregressive
    model. This one changes nothing but the node count: same arithmetic, same axis, same epsilon. There
    is no model for which emitting `RMS_NORM` here would be the wrong answer, so there is no flag.

    **What it is worth.** `pow` and `rsqrt` are `ggml_map_custom` host callbacks -- C function pointers,
    which no backend but the CPU can dispatch -- so on a device build each one cuts the graph and costs a
    device→host→device round trip. Qwen3-0.6B traced 113 of each, which is what put its 3050-node graph
    into 453 scheduler splits and cost it its entire GPU speedup (BACKLOG.md P4.7).

    **Anchored on the `mul`, because that is where the pattern closes.** Matching from `pow` forward
    would find the chain but could not confirm the multiply feeds back the SAME `x` the square was taken
    of, which is the whole difference between RMS normalization and an unrelated rsqrt.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type == "mul":
                self._try_transform(op, block)

    @staticmethod
    def _producer(var, op_type):
        """`var`'s producing op when it is of `op_type` AND nothing else consumes `var`.

        The second half is what makes the rewrite safe rather than merely correct-looking: every
        intermediate in this chain is about to become unreachable, and a `variance` that some other node
        also reads is one this pass must leave alone.
        """
        producer = getattr(var, "op", None)
        if producer is None or producer.op_type != op_type:
            return None
        if len(getattr(var, "child_ops", []) or []) != 1:
            return None
        return producer

    @classmethod
    def _try_transform(cls, mul_op, block) -> bool:
        # mul(x, rsqrt(...)) in either operand order -- MIL does not normalize commutative operands, and
        # `x * torch.rsqrt(v)` and `torch.rsqrt(v) * x` are both written in the wild.
        x_in, y_in = mul_op.inputs.get("x"), mul_op.inputs.get("y")
        for x, maybe_rsqrt in ((x_in, y_in), (y_in, x_in)):
            if x is None or maybe_rsqrt is None:
                continue
            rsqrt_op = cls._producer(maybe_rsqrt, "rsqrt")
            if rsqrt_op is None:
                continue
            add_op = cls._producer(rsqrt_op.inputs.get("x"), "add")
            if add_op is None:
                continue
            # add(variance, eps) in either order, again.
            for variance, eps_var in ((add_op.inputs.get("x"), add_op.inputs.get("y")),
                                       (add_op.inputs.get("y"), add_op.inputs.get("x"))):
                eps = static_value(eps_var)
                if eps is None or np.asarray(eps).size != 1:
                    continue
                mean_op = cls._producer(variance, "reduce_mean")
                if mean_op is None:
                    continue
                pow_op = cls._producer(mean_op.inputs.get("x"), "pow")
                if pow_op is None:
                    continue
                # The multiply must feed back the very var that was squared. Identity, not equality:
                # two structurally identical tensors are still two tensors, and normalizing by the wrong
                # one is exactly the bug this check exists to refuse.
                if pow_op.inputs.get("x") is not x:
                    continue
                exponent = static_value(pow_op.inputs.get("y"))
                if exponent is None or np.asarray(exponent).size != 1 or float(np.asarray(exponent).reshape(-1)[0]) != 2.0:
                    continue
                if not cls._reduces_last_axis(mean_op, x):
                    continue
                # keep_dims=False would leave the multiply broadcasting against a rank it no longer has;
                # torch's RMSNorm always keeps it, and a trace that did not is not this pattern.
                if not bool(static_value(mean_op.inputs.get("keep_dims"), False)):
                    continue

                # The real epsilon is the sum of the two: MIL's rsqrt adds its own (default 1e-12) on top
                # of the traced `variance + self.eps`. Summed in Python double precision and stored once,
                # because MIL casts every float const to fp32 anyway.
                rsqrt_eps = static_value(rsqrt_op.inputs.get("epsilon"), 0.0)
                total_eps = float(np.asarray(eps).reshape(-1)[0]) + float(np.asarray(rsqrt_eps).reshape(-1)[0])

                with _scope_ctx_like(mul_op):
                    new_out = mb.loom_rms_norm(
                        x=x,
                        epsilon=np.float32(total_eps),
                        name=mul_op.outputs[0].name,
                        before_op=mul_op,
                    )
                if not block.try_replace_uses_of_var_after_op(
                        anchor_op=mul_op, old_var=mul_op.outputs[0], new_var=new_out):
                    return False
                # Only the anchor is removed here; `pow`/`reduce_mean`/`add`/`rsqrt` are now unreachable
                # and `common::dead_code_elimination` -- which apply_loom_mil_passes runs after every
                # rewrite, for exactly this -- collects them along with the consts they read.
                block.remove_ops([mul_op])
                return True
        return False

    @staticmethod
    def _reduces_last_axis(mean_op, x) -> bool:
        """`ggml_rms_norm` normalizes ne[0] and nothing else, so a mean over any other axis is not this
        primitive however much the surrounding chain looks like it."""
        axes = static_ints(mean_op.inputs.get("axes"))
        if axes is None or len(axes) != 1 or x.shape is None:
            return False
        rank = len(x.shape)
        axis = axes[0]
        return (axis + rank if axis < 0 else axis) == rank - 1


def _squares(var, x) -> bool:
    """Whether `var` is `x` squared, in either spelling MIL uses for it.

    `pow(x, 2)` is what a trace produces and `square` is what `lower_pow` rewrites it to, so a matcher
    that knew only one would silently depend on pass ordering -- and the ordering that breaks it is the
    one where somebody moves `lower_pow` earlier for an unrelated reason.
    """
    producer = getattr(var, "op", None)
    if producer is None:
        return False
    if producer.op_type == "square":
        return producer.inputs.get("x") is x
    if producer.op_type != "pow" or producer.inputs.get("x") is not x:
        return False
    exponent = static_value(producer.inputs.get("y"))
    if exponent is None or np.asarray(exponent).size != 1:
        return False
    return float(np.asarray(exponent).reshape(-1)[0]) == 2.0


@register_pass(namespace="loom")
class lower_pow(AbstractGraphPass):
    """
    Rewrites `pow(x, 2)` into MIL's own `square`, which `exporter.py` already maps to the engine's `SQR`
    primitive -- replacing a `ggml_map_custom` host callback with a real ggml op.

    **Every `POW` this exporter has ever emitted is a square.** Counted across the thirteen fixture
    models: 149 `pow` ops, exponent 2.0 in every single one (Kokoro 50, StyleTTS2 50, Matcha 38, the
    NeMo encoders 3 each, GigaAM and Whisper 1 each). So this is not a special case carved out of a
    general op -- it is the only case there has ever been.

    **Only 2.** `pow(x, 0.5)` is `sqrt` and would be one more line, but no traced model has produced one,
    and this repo adds a primitive path when a model needs it rather than speculatively (see the
    attention-variant note in BACKLOG.md's scope limitations). The general `pow` path stays exactly where
    it was for anything else.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type == "pow":
                self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        exponent = static_value(op.inputs.get("y"))
        if exponent is None or np.asarray(exponent).size != 1:
            return False
        if float(np.asarray(exponent).reshape(-1)[0]) != 2.0:
            return False
        with _scope_ctx_like(op):
            new_out = mb.square(x=op.inputs["x"], name=op.outputs[0].name, before_op=op)
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class fuse_layer_norm(AbstractGraphPass):
    """
    Recognises a HAND-ROLLED layer norm -- the four-op statistic a model writes when it normalizes a
    channel axis itself instead of calling `torch.nn.LayerNorm` -- and replaces it with MIL's own
    `layer_norm`, transposed into place when the axis it normalizes is not the trailing one:

        mean = reduce_mean(x, axes=[a], keep_dims=True)
        out  = mul(sub(x, mean), rsqrt(add(reduce_mean(square(sub(x, mean)), axes=[a]), eps)))

    **`ggml_norm` only ever normalizes ne[0]**, so an axis that is already trailing lowers to MIL's
    `layer_norm` directly and any other axis is transposed into place first -- `transpose → layer_norm →
    transpose`, built here in MIL so the existing `transpose` rule lowers each half rather than
    `topology_ops.py`'s `layer_norm` rule growing ne-order arithmetic of its own. The engine's
    `LAYER_NORM` calls `ensure_packed`, so the non-contiguous view a permute produces is handled there.

    **Those two copies per norm are not the cost they look like, and this was measured properly because
    the first measurement of it was wrong.** On Matcha's `encoder_mu`, best of six interleaved rounds of
    twenty runs each (a single best-of-three said the opposite, twice, in both directions -- this module
    swings 33-85 ms between runs on the same binary):

        unfused chain               cpu 50.5 ms   gpu 45.7 ms
        transpose + layer_norm      cpu 44.2 ms   gpu 12.1 ms
        div(centered, sqrt(...))    cpu 54.5 ms   gpu 12.8 ms

    Transposing is the fastest of the three on the CPU as well as on the device: `ggml_norm` is one fused
    pass over the data where the chain it replaces is eight, and that buys more than two copies cost. The
    third row is the alternative rewrite that avoids moving anything -- turning the reciprocal back into
    a division, which removes the `rsqrt` host callback just as well -- and it is the slowest on the CPU
    and no better on the device, so it is not what this pass does.

    **What it is worth.** Matcha-TTS writes this in its text encoder over the channel axis of a (B, C, T)
    tensor: 32 `rsqrt` between its three topologies, every one a `ggml_map_custom` host callback that cut
    a device graph. Eight ops become three, 61 scheduler splits become one, and nothing falls back.

    Deliberately NOT folded into `fuse_rms_norm`: the two differ by exactly the mean-centring, and a
    matcher that treated `sub(x, mean)` as optional would emit `RMS_NORM` for a layer norm the moment a
    `sub` failed to match for some unrelated reason. They are separate patterns and separate passes.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type == "mul":
                self._try_transform(op, block)

    @classmethod
    def _try_transform(cls, mul_op, block) -> bool:
        producer = fuse_rms_norm._producer
        for centered, maybe_rsqrt in ((mul_op.inputs.get("x"), mul_op.inputs.get("y")),
                                       (mul_op.inputs.get("y"), mul_op.inputs.get("x"))):
            if centered is None or maybe_rsqrt is None:
                continue
            # The centred tensor feeds BOTH the variance and this multiply, so unlike every other
            # intermediate here it legitimately has two consumers -- asked for directly rather than
            # through the single-consumer `_producer`.
            sub_op = getattr(centered, "op", None)
            if sub_op is None or sub_op.op_type != "sub":
                continue
            x = sub_op.inputs.get("x")
            mean_op = producer(sub_op.inputs.get("y"), "reduce_mean")
            if x is None or mean_op is None or mean_op.inputs.get("x") is not x:
                continue

            rsqrt_op = producer(maybe_rsqrt, "rsqrt")
            if rsqrt_op is None:
                continue
            add_op = producer(rsqrt_op.inputs.get("x"), "add")
            if add_op is None:
                continue
            for variance, eps_var in ((add_op.inputs.get("x"), add_op.inputs.get("y")),
                                       (add_op.inputs.get("y"), add_op.inputs.get("x"))):
                eps = static_value(eps_var)
                if eps is None or np.asarray(eps).size != 1:
                    continue
                var_mean_op = producer(variance, "reduce_mean")
                if var_mean_op is None or not _squares(var_mean_op.inputs.get("x"), centered):
                    continue
                axis = cls._shared_axis(mean_op, var_mean_op, x)
                if axis is None:
                    continue
                if not (bool(static_value(mean_op.inputs.get("keep_dims"), False))
                        and bool(static_value(var_mean_op.inputs.get("keep_dims"), False))):
                    continue

                rsqrt_eps = static_value(rsqrt_op.inputs.get("epsilon"), 0.0)
                total_eps = float(np.asarray(eps).reshape(-1)[0]) + float(np.asarray(rsqrt_eps).reshape(-1)[0])
                rank = len(x.shape)
                out_name = mul_op.outputs[0].name

                with _scope_ctx_like(mul_op):
                    if axis == rank - 1:
                        new_out = mb.layer_norm(x=x, axes=[-1], epsilon=np.float32(total_eps),
                                                 name=out_name, before_op=mul_op)
                    else:
                        # An involution: the same permutation undoes itself, so one list serves both ends.
                        perm = list(range(rank))
                        perm[axis], perm[rank - 1] = perm[rank - 1], perm[axis]
                        moved = mb.transpose(x=x, perm=perm, name=f"{out_name}_ln_perm", before_op=mul_op)
                        normed = mb.layer_norm(x=moved, axes=[-1], epsilon=np.float32(total_eps),
                                                name=f"{out_name}_ln", before_op=mul_op)
                        new_out = mb.transpose(x=normed, perm=perm, name=out_name, before_op=mul_op)
                if not block.try_replace_uses_of_var_after_op(
                        anchor_op=mul_op, old_var=mul_op.outputs[0], new_var=new_out):
                    return False
                block.remove_ops([mul_op])
                return True
        return False

    @staticmethod
    def _shared_axis(mean_op, var_mean_op, x):
        """The one axis both reductions agree on, normalized to a non-negative index -- or None.

        Both means must reduce the SAME axis: a graph where they differ is not a layer norm, whatever
        else it is."""
        if x.shape is None:
            return None
        rank = len(x.shape)
        axes = []
        for op in (mean_op, var_mean_op):
            got = static_ints(op.inputs.get("axes"))
            if got is None or len(got) != 1:
                return None
            axes.append(got[0] + rank if got[0] < 0 else got[0])
        if axes[0] != axes[1] or not (0 <= axes[0] < rank):
            return None
        return axes[0]


@register_pass(namespace="loom")
class fuse_loom_attention(AbstractGraphPass):
    """
    Replaces each traced scaled-dot-product-attention block with one `loom_fused_attention` op, which
    `topology_ops.py` lowers to the engine's `ATTENTION` primitive -- the only node type that can reach
    a KV cache (KV-CACHE.md stage 2).

    **Opt-in, and that is a correctness requirement rather than caution.** The pattern below is generic
    SDPA, so it matches VITS's/Kokoro's/StyleTTS2's self-attention just as well as a causal LM's -- and
    those are non-autoregressive, so giving them an ATTENTION node (whose `kv_cache` attr defaults to
    TRUE) would hand them a persistent cache they must never have. Only the causal-LM family sets
    `fuse_attention=True`; every other model's topology is untouched, which is also what keeps their
    byte-identity gates meaningful.

    The window, anchored on `softmax` and confirmed against a real trace (a randomly-initialised
    2-layer Llama and, at full size, Qwen3-0.6B -- both produce it identically):

        mul       (q, 1/sqrt(head_dim))            -- scale folded onto Q by HF, not onto the scores
        matmul    (q_scaled, k, transpose_y=True)  -- Q @ K^T
        add       (scores, mask)                   -- mask is a slice_by_index of the graph input
        softmax   (axis=-1)
        matmul    (probs, v)
        transpose (perm=[0, 2, 1, 3])              -- [b, h, s, d] -> [b, s, h, d]
        reshape                                    -- -> [b, s, h*d]

    Both trailing ops are absorbed, because `op_attention` already returns the flattened
    `[n_embd, n_tokens]` context; stopping at the second matmul would leave the op's declared MIL type
    disagreeing with what the engine actually computes.

    **`layer` is assigned in attention-block occurrence order, NOT by torch module index**, and the
    distinction is load-bearing. The index addresses a cache slot, and the cache has one slot per
    ATTENTION block -- so for an architecture that interleaves non-attention layers (LFM2's conv
    blocks), the dense occurrence index is the correct one and the module index would address past the
    end of the cache. It also means `loom.n_layer` for cache sizing is the count of attention blocks,
    which for a uniform decoder like Qwen3 is the same number and for LFM2 is not.

    Anything that does not match is left exactly as it was: an unfused block still exports and still
    runs, just without a cache. A partial match must never half-rewrite.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._next_layer = 0
            self._fuse_block(f)

    @block_context_manager
    def _fuse_block(self, block):
        for op in list(block.operations):
            # Same guard as fuse_gqa_repeat_kv: `getattr(..., block)` because the bespoke workflow's
            # duck-typed MockOperations carry no `enclosing_block`, and "attribute missing" must read as
            # "still present" rather than as "already removed".
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._fuse_block(b)
            if op.op_type != "softmax":
                continue
            if self._try_to_transform(op, block):
                self._next_layer += 1

    @staticmethod
    def _binary_operands(op, want_op_type):
        """`op`'s two operands as (the one produced by `want_op_type`, the other), or None. Written
        order-agnostically because `add`'s operands are commutative and their traced order is not a
        promise -- keying on position would make this pass architecture-sensitive for no reason."""
        x, y = op.inputs.get("x"), op.inputs.get("y")
        if x is None or y is None:
            return None
        if x.op is not None and x.op.op_type == want_op_type:
            return x, y
        if y.op is not None and y.op.op_type == want_op_type:
            return y, x
        return None

    @staticmethod
    def _pre_gqa_repeat(var):
        """The un-repeated tensor behind `fuse_gqa_repeat_kv`'s `reshape -> tile -> reshape` triple, or
        None if `var` is not the output of one.

        Matched structurally rather than by the `_gqa_unsqueeze`/`_gqa_repeat` names that pass gives its
        ops: a name is a debugging aid, and keying on one would make this silently stop working the day
        those strings change.
        """
        reshape_out = var.op
        if reshape_out is None or reshape_out.op_type != "reshape":
            return None
        tile_var = reshape_out.inputs.get("x")
        if tile_var is None or tile_var.op is None or tile_var.op.op_type != "tile":
            return None
        reps = tile_var.op.inputs.get("reps")
        if reps is None or reps.val is None:
            return None
        # repeat_kv() only ever grows ONE axis (the KV-head one); anything else is a different tile.
        if sum(1 for r in np.array(reps.val).ravel() if int(r) != 1) != 1:
            return None
        inner = tile_var.op.inputs.get("x")
        if inner is None or inner.op is None or inner.op.op_type != "reshape":
            return None
        src = inner.op.inputs.get("x")
        if src is None or src.shape is None or len(src.shape) != 4:
            return None
        return src

    @staticmethod
    def _mask_kv_slice_source(mask_var):
        """The tensor behind HF's `mask[..., :kv_len]` slice, or `mask_var` unchanged.

        The traced mask does not reach the attention block directly: transformers slices it to the
        current KV length on the way in, which comes out of the converter as
        `slice_by_index(attention_mask, begin=[0,0,0,0], end=[...], end_mask=[T,T,T,False])` -- full
        extent on every axis but the last, and the last cut to a computed `kv_len`. With no cache in the
        trace, `kv_len == seq_len`, so it is an identity slice that exists only because the traced model
        expected to be given a mask wider than it needed.

        A cached step is the case that slice was written for, and the driver now builds the mask at
        exactly `[n_tokens, n_kv]` (`loom.causal_mask(n_tokens, n_past)`) -- so the slice is not merely
        redundant, it is *wrong*: its extents were baked at trace time and would cut a decode step's
        mask back to the prefill width. Bypassing it is what lets the mask input be declared `["n_kv",
        "n_tokens"]` at all (KV-CACHE.md 3.2), because the retyping is only sound while the input's own
        consumers are all fused-attention nodes -- and a surviving slice is a consumer that is not.

        Every guard bails to "leave it alone", the same rule the rest of this pass follows: an unmatched
        shape leaves a graph that still exports, prefill-only, rather than one rewritten halfway.
        """
        op = getattr(mask_var, "op", None)
        if op is None or op.op_type != "slice_by_index":
            return mask_var
        src = op.inputs.get("x")
        if src is None or src.shape is None or mask_var.shape is None:
            return mask_var
        rank = len(src.shape)
        if rank != len(mask_var.shape) or rank < 2:
            return mask_var

        def mask_bits(name, default):
            var = op.inputs.get(name)
            if var is None or var.val is None:
                return [default] * rank
            bits = list(np.array(var.val).ravel())
            return bits if len(bits) == rank else None

        # Nothing may be squeezed away, nothing strided, and every axis but the last must be taken
        # whole: `begin` at 0 (or ignored via begin_mask) and `end` ignored via end_mask.
        squeeze = mask_bits("squeeze_mask", False)
        stride = mask_bits("stride", 1)
        begin_mask = mask_bits("begin_mask", False)
        end_mask = mask_bits("end_mask", False)
        if squeeze is None or stride is None or begin_mask is None or end_mask is None:
            return mask_var
        if any(bool(b) for b in squeeze) or any(int(st) != 1 for st in stride):
            return mask_var
        begin_var = op.inputs.get("begin")
        begin = list(np.array(begin_var.val).ravel()) if begin_var is not None and begin_var.val is not None else None
        for axis in range(rank):
            if not bool(begin_mask[axis]) and (begin is None or int(begin[axis]) != 0):
                return mask_var
            if axis < rank - 1 and not bool(end_mask[axis]):
                return mask_var
        # The last axis IS sliced (that is the whole point); if it were not, this is some other slice.
        if bool(end_mask[rank - 1]):
            return mask_var
        return src

    def _strip_gqa_repeat(self, k_var, v_var, q_var):
        """`(k, v)` with HF's `repeat_kv()` expansion undone when it is safe to do so, else unchanged.

        `op_attention` reads `n_head_kv` straight off K's own shape and lets `ggml_mul_mat`'s broadcast
        map query head `i` to KV head `i // ratio` -- integer division, i.e. exactly the interleaved
        correspondence `repeat_kv()` materializes (see `fuse_gqa_repeat_kv`'s docstring on why that is
        interleaved and not block-tiled). So attending against the un-repeated K/V is the same
        arithmetic, and it HALVES Qwen3-0.6B's cache: 16 stored heads become the 8 the checkpoint
        actually has.

        Correctness never depends on this. Keeping the repeat is numerically identical, merely wasteful,
        which is why every guard below bails to "leave it alone" rather than raising -- and why K and V
        are stripped only TOGETHER and only to the same head count. Stripping one and not the other
        would leave the cache's K and V widths disagreeing, which no later check would catch.
        """
        k_src, v_src = self._pre_gqa_repeat(k_var), self._pre_gqa_repeat(v_var)
        if k_src is None or v_src is None:
            return k_var, v_var
        n_head, n_head_kv = q_var.shape[1], k_src.shape[1]
        if not isinstance(n_head, int) or not isinstance(n_head_kv, int):
            return k_var, v_var
        if n_head_kv != v_src.shape[1] or n_head_kv <= 0 or n_head % n_head_kv != 0:
            return k_var, v_var
        return k_src, v_src

    @staticmethod
    def _insertion_anchor(block, chain, operands):
        """The op in `chain` to insert the fused node before: the earliest one that already follows
        every operand's own definition, or None if no position in the chain does.

        **Not simply `chain[0]`, and Whisper is what proved it.** The QK matmul was the anchor from this
        pass's first version, on the reasonable-looking grounds that it is the first op being subsumed.
        That silently assumes V is projected *before* Q@K^T, which is true of the traces this pass was
        written against (Qwen3, LFM2, a 2-layer Llama) and false of HF's Whisper decoder, where
        `value_states` is traced four ops *after* the matmul. Anchoring there builds an op that reads a
        var defined later -- an SSA violation `mb` does not reject and `try_replace_uses_of_var_after_op`
        does not notice. It surfaces one pass later, as `common::dead_code_elimination` walking the block
        in reverse, reaching the V transpose before it has seen the consumer that sits above it, judging
        it dead, and raising `Cannot delete op 'transpose_17' with active output`.

        Choosing the earliest *valid* position rather than always the last is what keeps the existing
        exports byte-identical: wherever the old anchor was already sound, this returns it, so node order
        in the emitted topology does not move for any model that fused before.
        """
        index = {id(op): i for i, op in enumerate(block.operations)}
        # A `mask` that is a graph input (or otherwise producer-less) constrains nothing.
        defs = [index.get(id(v.op)) for v in operands if getattr(v, "op", None) is not None]
        last_def = max([i for i in defs if i is not None], default=-1)
        for op in chain:
            position = index.get(id(op))
            if position is not None and position > last_def:
                return op
        return None

    def _try_to_transform(self, softmax_op, block) -> bool:
        axis = softmax_op.inputs.get("axis")
        if axis is None or axis.val is None or int(axis.val) not in (-1, 3):
            return False

        scores = softmax_op.inputs.get("x")
        if scores is None or scores.op is None or scores.op.op_type != "add":
            return False
        add_op = scores.op
        operands = self._binary_operands(add_op, "matmul")
        if operands is None:
            return False
        qk_var, mask_var = operands
        qk_op = qk_var.op

        # Q @ K^T, and K must NOT be pre-transposed by a separate op -- `transpose_y` is how the traced
        # graph spells it, and a False here means this is some other matmul that happens to feed a
        # softmax.
        transpose_y = qk_op.inputs.get("transpose_y")
        transpose_x = qk_op.inputs.get("transpose_x")
        if transpose_y is None or transpose_y.val is None or not bool(transpose_y.val):
            return False
        if transpose_x is not None and transpose_x.val is not None and bool(transpose_x.val):
            return False

        q_var = qk_op.inputs.get("x")
        k_var = qk_op.inputs.get("y")
        if q_var is None or k_var is None:
            return False

        # The scale HF folds onto Q. Recovered rather than recomputed from head_dim: a model with a
        # non-default scale (or none) is then still correct, and `scale=1.0` with the `mul` left in
        # place is a valid outcome rather than a silent 1/sqrt(d) that was never in the graph.
        scale = 1.0
        if q_var.op is not None and q_var.op.op_type == "mul":
            factors = (q_var.op.inputs.get("x"), q_var.op.inputs.get("y"))
            const_side = [f for f in factors if f is not None and f.val is not None and f.shape in ((), (1,))]
            other_side = [f for f in factors if f is not None and f.val is None]
            if len(const_side) == 1 and len(other_side) == 1:
                scale = float(np.array(const_side[0].val).ravel()[0])
                q_var = other_side[0]

        # Down the graph: probs @ V, then the transpose+reshape back to [b, seq, n_embd].
        probs_children = list(softmax_op.outputs[0].child_ops)
        if len(probs_children) != 1 or probs_children[0].op_type != "matmul":
            return False
        av_op = probs_children[0]
        if av_op.inputs.get("x") is not softmax_op.outputs[0]:
            return False
        v_var = av_op.inputs.get("y")
        if v_var is None:
            return False
        for flag in ("transpose_x", "transpose_y"):
            f = av_op.inputs.get(flag)
            if f is not None and f.val is not None and bool(f.val):
                return False

        av_children = list(av_op.outputs[0].child_ops)
        if len(av_children) != 1 or av_children[0].op_type != "transpose":
            return False
        transpose_op = av_children[0]
        perm = transpose_op.inputs.get("perm")
        if perm is None or perm.val is None or list(np.array(perm.val).ravel()) != [0, 2, 1, 3]:
            return False

        transpose_children = list(transpose_op.outputs[0].child_ops)
        if len(transpose_children) != 1 or transpose_children[0].op_type != "reshape":
            return False
        reshape_op = transpose_children[0]
        out_var = reshape_op.outputs[0]
        if out_var.shape is None or len(out_var.shape) != 3:
            return False

        # Undo HF's repeat_kv() where it is safe, so the cache stores the checkpoint's real KV heads
        # rather than the expanded ones (KV-CACHE.md 2.3). Purely a size win; see _strip_gqa_repeat.
        k_var, v_var = self._strip_gqa_repeat(k_var, v_var, q_var)

        # Attend against the mask the driver actually builds, not the trace-width slice of it
        # (KV-CACHE.md 3.2). Unlike the GQA strip above this one is a correctness requirement for a
        # cached step, not a size win -- see _mask_kv_slice_source.
        mask_var = self._mask_kv_slice_source(mask_var)

        # Every rank check the op's own type_inference would make, made here first -- a pass that raises
        # from inside mb.loom_fused_attention leaves the block half-rewritten, whereas bailing here
        # leaves a graph that still exports.
        for var in (q_var, k_var, v_var):
            if var.shape is None or len(var.shape) != 4:
                return False
        if not isinstance(q_var.shape[1], int) or not isinstance(v_var.shape[3], int):
            return False

        # The subsumed chain, in block order. The fused op replaces all six, so it may be inserted at
        # any of their positions that is still after every operand it reads -- see `_insertion_anchor`.
        chain = [qk_op, add_op, softmax_op, av_op, transpose_op, reshape_op]
        anchor = self._insertion_anchor(block, chain, (q_var, k_var, v_var, mask_var))
        if anchor is None:
            return False

        with _scope_ctx_like(softmax_op):
            fused = mb.loom_fused_attention(
                q=q_var, k=k_var, v=v_var, mask=mask_var,
                scale=np.float32(scale), layer=np.int32(self._next_layer),
                name=out_var.name, before_op=anchor,
            )

        if not reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=out_var, new_var=fused,
        ):
            return False
        # Only the ops this fusion definitively subsumed. Everything upstream (the q `mul`, the mask's
        # own slice chain) is left to dead_code_elimination, which is the pass that knows whether some
        # other consumer still needs it -- this one does not.
        block.remove_ops([reshape_op, transpose_op, av_op, softmax_op, add_op, qk_op])
        return True


@register_pass(namespace="loom")
class fuse_loom_short_conv(AbstractGraphPass):
    """
    Replaces each traced causal depthwise convolution with one `loom_short_conv` op, which
    `topology_ops.py` lowers to the engine's `SHORT_CONV` primitive -- the only node type that can reach
    a `ConvStateCache`, and therefore the thing that lets a hybrid architecture decode incrementally
    (BACKLOG.md P4.0.10).

    **Opt-in, for the same correctness reason `fuse_loom_attention` is.** The pattern is a depthwise
    conv with symmetric `kernel - 1` padding whose output is sliced back to the input length, and that
    is not unique to a causal LM -- a non-autoregressive model could produce it too, and giving one a
    `conv_state` attr (which defaults to TRUE in op_short_conv) would hand it persistent state it must
    never have. Only the causal-LM family sets `fuse_conv=True`.

    The window, measured on a real LFM2-350M trace rather than assumed:

        conv           (x=[b, C, s], weight=[C, 1, K], groups=C, pad=[K-1, K-1], pad_type='custom')
        slice_by_index (begin=[0,0,0], end_mask=[True, True, False])   -- back to the first `s` columns

    That pair is how transformers writes a causal conv with no cache: pad both sides, then discard the
    trailing K-1 outputs. It is correct for a prefill and unusable for a decode step, since the K-1
    columns a length-1 window needs live in the PREVIOUS call. `op_short_conv` keeps them in a slot
    instead, which is why the fusion absorbs the slice rather than leaving it to DCE -- a surviving
    slice would cut a decode step's output using extents chosen at trace time, the same trap
    `_mask_kv_slice_source` exists for on the attention side (KV-CACHE.md 3.2, second bullet).

    `layer` is assigned in conv-block OCCURRENCE order, not by torch module index, for the reason
    `fuse_loom_attention`'s is: LFM2-350M declares 16 hidden layers and has 10 conv blocks, so a module
    index would address past the end of a 10-slot store.

    Anything that does not match is left exactly as it was -- an unfused conv still exports and still
    runs, just without state. A partial match must never half-rewrite.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._next_layer = 0
            self._fuse_block(f)

    @block_context_manager
    def _fuse_block(self, block):
        for op in list(block.operations):
            # Same guard as fuse_loom_attention: "attribute missing" must read as "still present".
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._fuse_block(b)
            if op.op_type != "conv":
                continue
            if self._try_to_transform(op, block):
                self._next_layer += 1

    @staticmethod
    def _causal_slice(conv_var):
        """The `slice_by_index` that cuts a symmetrically-padded conv back to its input length, or None.

        Requires the slice to take the FULL extent on every axis but the last (begin 0, end ignored via
        end_mask) and to start at 0 on the last -- i.e. it keeps a leading prefix and nothing else. A
        slice doing anything more than that is not the causal-trim idiom and is left alone.
        """
        children = list(conv_var.child_ops)
        if len(children) != 1 or children[0].op_type != "slice_by_index":
            return None
        slice_op = children[0]
        rank = len(conv_var.shape) if conv_var.shape is not None else 0
        if rank != 3:
            return None

        def bits(name, default):
            var = slice_op.inputs.get(name)
            if var is None or var.val is None:
                return [default] * rank
            vals = list(np.array(var.val).ravel())
            return vals if len(vals) == rank else None

        begin = bits("begin", 0)
        stride = bits("stride", 1)
        squeeze = bits("squeeze_mask", False)
        begin_mask = bits("begin_mask", False)
        end_mask = bits("end_mask", False)
        if any(b is None for b in (begin, stride, squeeze, begin_mask, end_mask)):
            return None
        if any(bool(sq) for sq in squeeze) or any(int(st) != 1 for st in stride):
            return None
        # Every axis starts at 0, whether stated or masked away.
        if any(not bool(bm) and int(bg) != 0 for bg, bm in zip(begin, begin_mask)):
            return None
        # Full extent on all but the last axis; the last is the one being trimmed.
        if not all(bool(em) for em in end_mask[:-1]) or bool(end_mask[-1]):
            return None
        return slice_op

    def _try_to_transform(self, conv_op, block):
        x_var = conv_op.inputs.get("x")
        weight_var = conv_op.inputs.get("weight")
        if x_var is None or weight_var is None or weight_var.shape is None:
            return False
        if conv_op.inputs.get("bias") is not None:
            return False  # op_short_conv takes no bias; LFM2's conv has none.
        if x_var.shape is None or len(x_var.shape) != 3:
            return False

        channels = x_var.shape[1]
        if not isinstance(channels, int):
            return False
        groups = conv_op.inputs.get("groups")
        if groups is None or groups.val is None or int(groups.val) != channels:
            return False  # depthwise only: one filter per channel, which is what the state layout assumes

        kernel = int(weight_var.shape[-1])
        if kernel < 2:
            return False  # width-1 is position-wise and carries no history; op_short_conv rejects it too

        pad = conv_op.inputs.get("pad")
        strides = conv_op.inputs.get("strides")
        dilations = conv_op.inputs.get("dilations")
        for var, want in ((strides, 1), (dilations, 1)):
            if var is None or var.val is None or any(int(v) != want for v in np.array(var.val).ravel()):
                return False
        if pad is None or pad.val is None:
            return False
        pad_vals = [int(v) for v in np.array(pad.val).ravel()]
        # Symmetric kernel-1 padding is what makes the trailing trim a CAUSAL one. Any other padding is
        # a different convolution and must not silently acquire state.
        if pad_vals != [kernel - 1, kernel - 1]:
            return False

        slice_op = self._causal_slice(conv_op.outputs[0])
        if slice_op is None:
            return False
        out_var = slice_op.outputs[0]

        with _scope_ctx_like(conv_op):
            fused = mb.loom_short_conv(
                x=x_var, weight=weight_var, layer=np.int32(self._next_layer),
                name=out_var.name, before_op=conv_op,
            )

        if not slice_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=slice_op, old_var=out_var, new_var=fused,
        ):
            return False
        block.remove_ops([slice_op, conv_op])
        return True


_LOOM_PASS_NAMES = [
    "loom::fuse_gqa_repeat_kv",
    "loom::normalize_matmul",
    "loom::insert_explicit_broadcasts",
    "loom::canonicalize_replicate_pad",
    "loom::canonicalize_conv_transpose_dw",
    "loom::lower_stack",
    # Before lower_reduce_mean, and that ordering is the whole reason this is a list rather than a set:
    # the RMS-norm chain contains a `reduce_mean`, and lowering it first would leave the pattern spelled
    # `reduce_sum` + `loom_scale` -- still fusable, but only by a matcher that knows about a rewrite that
    # has nothing to do with it.
    "loom::fuse_rms_norm",
    # After fuse_rms_norm (an RMS norm has no mean-centring, so it cannot match this one, but matching
    # the cheaper pattern first keeps the layer-norm matcher off graphs that are already gone) and
    # before lower_pow, so this still sees the `pow` spelling it was written against -- `_squares`
    # accepts both, so the order is a preference rather than a dependency.
    "loom::fuse_layer_norm",
    # Last of the three, deliberately: it mops up every square the two fusions did NOT claim, and
    # running it earlier would only mean the fusions had to match `square` instead.
    "loom::lower_pow",
    "loom::lower_reduce_mean",
    "common::dead_code_elimination",
]

# Runs only when the caller asks for it (KV-CACHE.md decision 4). Placed after the GQA fusion, which
# normalizes `repeat_kv()` into a reshape/tile/reshape triple the attention fusion can see past, and
# before dead_code_elimination, which is what removes the subgraph the fusion orphans.
_LOOM_ATTENTION_PASS_NAME = "loom::fuse_loom_attention"

# Same placement and the same reason: after the rewrites it must see past, before the DCE that clears
# what it orphans. Independent of the attention flag -- a model can genuinely want one and not the other
# (a pure causal LM has no convs to fuse; a Mamba-style model would have no attention to fuse).
_LOOM_SHORT_CONV_PASS_NAME = "loom::fuse_loom_short_conv"


def apply_loom_mil_passes(prog, fuse_attention: bool = False, fuse_conv: bool = False) -> None:
    """
    Runs Loom's own MIL->MIL rewrite passes -- GQA `repeat_kv()` fusion, matmul transpose_x
    normalization (R2a), mutual-broadcast insertion (R2a), replicate-pad and depthwise-conv_transpose
    canonicalization, `stack` and `reduce_mean` lowering (R2) -- plus `common::dead_code_elimination`
    over `prog` in place. Must run before any topology/driver generation sees `prog` --
    `common::dead_code_elimination` is what actually removes each rewrite's now-orphaned dependency
    chain (the original tile/reshape idiom, a stale `transpose_x` bool operand, etc).

    Invokes each registered pass callable directly (`PASS_REGISTRY[name](prog)`) rather than going through
    `PassPipelineManager.apply_pipeline` -- that manager additionally calls `prog.validate()` before/after
    every pass, which is real MIL API surface (`Operation.get_flattened_inputs()` etc.) that this
    exporter's own "bespoke" workflow doesn't need to satisfy: it deliberately accepts hand-built
    `Program`s with synthetic, duck-typed submodule-dispatch ops standing in for ops MIL itself doesn't
    have (see `test_compiler.py`'s `MockOperation`), which are never meant to pass a real MIL validate().
    """
    for pass_name in _LOOM_PASS_NAMES:
        if pass_name == "common::dead_code_elimination":
            # Must land between the rewrites and the DCE that cleans up after them, which is why these
            # are spliced here rather than appended to the list.
            if fuse_attention:
                PASS_REGISTRY[_LOOM_ATTENTION_PASS_NAME](prog)
            if fuse_conv:
                PASS_REGISTRY[_LOOM_SHORT_CONV_PASS_NAME](prog)
        PASS_REGISTRY[pass_name](prog)
