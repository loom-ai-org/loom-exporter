"""Zero-pad weight axes to what the kernels need (ADR-081, P4.34).

**Why.** A kernel's fast path has an alignment predicate and an unaligned weight silently takes the slow
one. Measured on Vocetta2 (Pi 4 + Pi Zero, rc18, 5 paired rounds): aarch64's tinyBLAS F32 GEMM wants the
contraction K % 4 (a ragged k runs a scalar tail); ARMv6's F32 GEMM (`ggml_armv6_mm_ok`) and the direct
1-D conv on every ISA (`ggml_conv_1d_direct_ok`) want the output count N % 4 (else a generic kernel or
im2col); and a Q8_0 / Q8_0_PC weight needs K % 32 or `pack_weights` leaves it F32 -- which made Vocetta2's
276k and 1m Q8_0_PC exports byte-identical to F32. Padding F32 K and N to 4 was 1.10-1.34x on the Pi Zero
and 1.09-1.15x on the Pi 4 at one thread, with outputs equal to unpadded up to summation order.

**Targets, per weight read as the FIRST input of exactly one `MUL_MAT` or `CONV_1D` (one group) node:**
a weight that will be block-quantized pads K to the block size and leaves N alone (`CPU_ARMV6_I8` pads
its own rows); every other weight pads K and N to 4. A pad that would grow either axis by more than 25%
is skipped: padding the 1m's acoustic stream 72 -> 96 (+33%) on top of its decoder ran 7-10% slower than
the decoder alone, while its decoder's 130 -> 160 (+23%) with 337 -> 352 was the fastest 1m measured --
which a cap on the product of the two (1.285) would have refused.

**Mechanisms, all exact in real arithmetic:**

* N side: zero rows in the weight and zeros in the bias it is summed with (`ADD` of a vector, or the
  conv form's `RESHAPE` + `ADD`), then a `VIEW` (offset 0) back to N right after the bias -- unless the
  next rule consumes the padded rows. The view's shape is the MIL var's own, which is why only
  topologies produced by `generate_graph_topology` are touched: nothing else recorded one.
* K side carried by the producer: if the consumer's input comes from an N-padded producer of the same
  kind through nodes that each have ONE consumer and map 0 to 0 (`ZERO_PRESERVING`), the padded channels
  arrive as zeros and need no view and no pad. `pw0 -> GELU -> pw1` is this.
* K side supplied by the consumer, `MUL_MAT` only: `PAD_1D` on the input, which pads ne[0]. This is what
  a LayerNorm-fed matmul gets: LayerNorm normalises over the axis, so its width cannot be padded with it.
* A conv's channel axis is ne[1] and no engine op pads it, so a conv with no producer route keeps its K.
  Padding a residual stream as a whole is ADR-081's v2.

Weights shared by several nodes, read by anything else, or already packed are left alone, as are nodes
with a third input (a fused bias would not have been padded with its weight).
"""
from dataclasses import dataclass, field
from math import gcd
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

#: Unary ops with f(0) = 0 per element, so zero channels stay zero through them.
ZERO_PRESERVING = ("GELU", "SILU", "RELU", "TANH", "LEAKY_RELU")
#: What the F32 kernels want on both axes (see the module docstring for which kernel wants which).
F32_MULTIPLE = 4
#: Kp / K or Np / N above which that weight is left as it is.
GROWTH_CAP = 1.25
PAD_OPS = ("MUL_MAT", "CONV_1D")


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


@dataclass
class _Site:
    node: dict
    weight: str
    kind: str                    # "MUL_MAT" or "CONV_1D"
    n: int                       # output rows (out_features / OC)
    k: int                       # input channels (in_features / IC)
    width: int                   # conv kernel width; 1 for a matmul
    quantize: bool
    slice_point: str = ""        # the tensor the padded rows are sliced back from (after the bias)
    bias: Optional[tuple] = None # ("vec", name) or ("reshape", node, name)
    np_: int = 0
    kp: int = 0
    carried_out: bool = False    # N padding consumed by a carried K below it: no VIEW
    carried_in: bool = False     # K padding arrives from the producer: no PAD_1D

    def k_multiple(self, block_size: int) -> int:
        """The IC multiple that makes the contraction (IC x width for a conv) a multiple of the target."""
        target = block_size if self.quantize else F32_MULTIPLE
        return target // gcd(target, self.width)

    def n_multiple(self) -> int:
        return 1 if self.quantize else F32_MULTIPLE


@dataclass
class PadReport:
    #: {tensor name: (true out_features / OC / bias length, true in_features / IC, 0 for a bias)}.
    padded: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    n_views: int = 0
    n_pads: int = 0
    n_carried: int = 0


def pad_weight_axes(topologies: dict, weights: dict, pending: set,
                    shapes: Dict[str, Callable[[str], Optional[list]]],
                    will_quantize: Callable[[str], bool], block_size: int) -> PadReport:
    """Pad every eligible weight in `weights` (mutated) and rewrite `topologies`' nodes (mutated in
    place: an `extra_streams` alias shares its primary's node list, so the list object is kept).

    `shapes[topology](var)` is the ggml-order shape `generate_graph_topology` recorded for a var, or
    None; a topology with no entry is skipped whole. `will_quantize(name)` says whether
    `pack_weights` will block-quantize the name if its K comes out aligned to `block_size`.
    """
    report = PadReport()
    lists = []
    seen = set()
    for topo_name, topo in topologies.items():
        nodes = topo.get("nodes")
        if nodes is None or id(nodes) in seen:
            continue
        seen.add(id(nodes))
        lists.append((topo_name, topo, nodes))

    # Every (op, position) a weight is read at and by how many nodes, over EVERY topology: a weight is
    # padded only when its whole use is one node's first input.
    usage: Dict[str, set] = {}
    readers: Dict[str, int] = {}
    for _, _, nodes in lists:
        for node in nodes:
            for position, name in enumerate(node.get("inputs") or []):
                if name in weights:
                    usage.setdefault(name, set()).add((node.get("op"), position))
                    readers[name] = readers.get(name, 0) + 1

    def sole_use(name, op, position) -> bool:
        return usage.get(name) == {(op, position)} and readers.get(name) == 1

    for topo_name, topo, nodes in lists:
        if topo_name in shapes:
            _pad_topology(topo, nodes, weights, pending, shapes[topo_name], will_quantize, block_size,
                          sole_use, report)
    return report


def _pad_topology(topo, nodes, weights, pending, shape_of, will_quantize, block_size,
                  sole_use, report: PadReport) -> None:
    producer = {}
    consumers: Dict[str, list] = {}
    for node in nodes:
        for out in node.get("outputs") or []:
            producer[out] = node
        for position, name in enumerate(node.get("inputs") or []):
            consumers.setdefault(name, []).append((node, position))
    outputs = set(topo.get("outputs") or [topo.get("output")])

    def single_consumer(name):
        found = consumers.get(name, [])
        if len(found) != 1 or name in outputs:
            return None
        return found[0]

    sites: List[_Site] = []
    for node in nodes:
        op = node.get("op")
        inputs = node.get("inputs") or []
        if op not in PAD_OPS or len(inputs) != 2 or len(node.get("outputs") or []) != 1:
            continue
        name = inputs[0]
        array = weights.get(name)
        if (name not in pending or array is None or not np.issubdtype(array.dtype, np.floating)
                or not sole_use(name, op, 0)):
            continue
        if op == "MUL_MAT":
            if array.ndim != 2:
                continue
            n, k, width = int(array.shape[0]), int(array.shape[1]), 1
        else:
            if array.ndim != 3 or node.get("attrs", {}).get("groups", 1) != 1:
                continue
            n, k, width = (int(d) for d in array.shape)
        sites.append(_Site(node=node, weight=name, kind=op, n=n, k=k, width=width,
                           quantize=block_size > 1 and will_quantize(name)))
    if not sites:
        return

    # Each site's bias and the tensor its rows would be sliced back from.
    for site in sites:
        out = site.node["outputs"][0]
        site.slice_point = out
        use = single_consumer(out)
        if use is None:
            continue
        add, position = use
        if add.get("op") != "ADD" or len(add.get("inputs") or []) != 2:
            continue
        other = add["inputs"][1 - position]
        if site.kind == "MUL_MAT":
            vec = weights.get(other)
            if (other in pending and vec is not None and vec.shape == (site.n,)
                    and sole_use(other, "ADD", 1 - position)):
                site.bias = ("vec", other)
                site.slice_point = add["outputs"][0]
        else:
            reshape = producer.get(other)
            if reshape is None or reshape.get("op") != "RESHAPE" or single_consumer(other) is None:
                continue
            vec_name = (reshape.get("inputs") or [None])[0]
            vec = weights.get(vec_name)
            shape = reshape.get("attrs", {}).get("shape")
            if (vec_name in pending and vec is not None and vec.shape == (site.n,)
                    and sole_use(vec_name, "RESHAPE", 0)
                    and isinstance(shape, list) and len(shape) in (2, 3)
                    and shape[0] == 1 and shape[1] == site.n and all(d == 1 for d in shape[2:])):
                site.bias = ("reshape", reshape, vec_name)
                site.slice_point = add["outputs"][0]

    # Producer routes: Y's input reaches an X's slice point through single-consumer zero-preserving
    # nodes, X and Y of the same kind so the padded axis is the same one (ne[0] for a matmul, ne[1]
    # for a conv). Each X has at most one Y, since its slice point has one consumer.
    by_slice = {site.slice_point: site for site in sites}
    routes: Dict[int, _Site] = {}           # id(Y) -> X
    for y in sites:
        current = y.node["inputs"][1]
        while True:
            if single_consumer(current) is None:
                break
            x = by_slice.get(current)
            if x is not None:
                if x is not y and x.kind == y.kind and x.n == y.k:
                    routes[id(y)] = x
                break
            step = producer.get(current)
            if step is None or step.get("op") not in ZERO_PRESERVING or len(step.get("inputs") or []) != 1:
                break
            current = step["inputs"][0]

    # A quantize target only stands if its K can actually be aligned: already aligned, or reachable
    # (a route, or PAD_1D for a matmul) within the cap. Otherwise the weight stays F32 and gets the
    # F32 rule instead.
    for site in sites:
        if not site.quantize:
            continue
        multiple = site.k_multiple(block_size)
        reachable = site.kind == "MUL_MAT" or id(site) in routes
        if site.k % multiple and (not reachable or _round_up(site.k, multiple) / site.k > GROWTH_CAP):
            site.quantize = False

    def view_shape(site):
        # Never at a topology output: those are materialized contiguous when the topology is generated
        # (`_materialize_view_outputs`), and a view added after that would hand the host strided rows.
        if site.slice_point in outputs:
            return None
        shape = shape_of(site.slice_point)
        axis = 0 if site.kind == "MUL_MAT" else 1
        if not shape or len(shape) <= axis or str(shape[axis]) != str(site.n):
            return None
        return shape

    # Sizes, then the cap on each axis; a weight over it is frozen on both axes and the sizes are
    # recomputed without it, which can only shrink every pad, so this terminates.
    frozen = set()
    while True:
        for site in sites:
            site.np_, site.kp = site.n, site.k
            site.carried_out = site.carried_in = False
        for y in sites:
            x = routes.get(id(y))
            if x is None or id(x) in frozen or id(y) in frozen:
                continue
            width = _round_up(x.n, _lcm(x.n_multiple(), y.k_multiple(block_size)))
            if width != x.n:
                x.np_, y.kp = width, width
                x.carried_out = y.carried_in = True
        for site in sites:
            if id(site) in frozen:
                continue
            if not site.carried_out and site.n % site.n_multiple() and view_shape(site) is not None:
                site.np_ = _round_up(site.n, site.n_multiple())
            if not site.carried_in and site.kind == "MUL_MAT" and site.k % site.k_multiple(block_size):
                site.kp = _round_up(site.k, site.k_multiple(block_size))
        over = [s for s in sites if id(s) not in frozen
                and (s.np_ / s.n > GROWTH_CAP or s.kp / s.k > GROWTH_CAP)]
        if not over:
            break
        frozen.update(id(s) for s in over)

    # Apply: weights, biases, then the graph.
    before: Dict[int, list] = {}   # id(node) -> nodes to insert before it
    after: Dict[int, list] = {}    # id(node) -> nodes to insert after it
    pad_inputs: Dict[Tuple[str, int], str] = {}
    for site in sites:
        if site.np_ == site.n and site.kp == site.k:
            continue
        array = weights[site.weight]
        widths = [(0, site.np_ - site.n), (0, site.kp - site.k)] + [(0, 0)] * (array.ndim - 2)
        weights[site.weight] = np.pad(array, widths)
        report.padded[site.weight] = (site.n, site.k)
        if site.np_ != site.n:
            if site.bias is not None:
                vec_name = site.bias[-1]
                weights[vec_name] = np.pad(weights[vec_name], (0, site.np_ - site.n))
                report.padded[vec_name] = (site.n, 0)
                if site.bias[0] == "reshape":
                    site.bias[1]["attrs"]["shape"][1] = site.np_
            if site.carried_out:
                report.n_carried += 1
            else:
                last = producer[site.slice_point]
                padded_name = f"{site.slice_point}_npad"
                last["outputs"] = [padded_name if o == site.slice_point else o for o in last["outputs"]]
                after.setdefault(id(last), []).append({
                    "op": "VIEW", "inputs": [padded_name], "outputs": [site.slice_point],
                    "attrs": {"shape": list(view_shape(site)), "offset": "0"},
                })
                report.n_views += 1
        if site.kp != site.k and not site.carried_in:
            source = site.node["inputs"][1]
            key = (source, site.kp)
            if key not in pad_inputs:
                pad_inputs[key] = f"{source}_kpad{site.kp}"
                before.setdefault(id(site.node), []).append({
                    "op": "PAD_1D", "inputs": [source], "outputs": [pad_inputs[key]],
                    "attrs": {"lp0": 0, "rp0": site.kp - site.k},
                })
                report.n_pads += 1
            site.node["inputs"][1] = pad_inputs[key]

    if before or after:
        rewritten = []
        for node in nodes:
            rewritten.extend(before.get(id(node), ()))
            rewritten.append(node)
            rewritten.extend(after.get(id(node), ()))
        nodes[:] = rewritten
