"""Common-subexpression elimination over a finished graph topology (loom.cpp ADR-079 P4.32f).

A traced model repeats work the trace cannot see is repeated. NeMo's encoders rebuild the time mask
`arange(T) < length` before EVERY use: Citrinet-256-ls's topology carries 239 `RANGE_1D -> RESHAPE ->
CAST -> LESS -> RESHAPE` chains of which 4 are distinct, one per resolution. In MIL they are not
duplicates -- each `range_1d` ends at a different tensor's `shape` -- and only become textually equal
once the exporter has rewritten every dimension as an expression over the root axis
(`floor(floor(n_samples/160)/2) + 1`). So this runs on the topology, last, where they are.

Beyond the ~1000 tiny nodes, the duplicates cost what ggml's graph ORDER makes of them: ggml expands a
node's sources depth first, so a mask built right before its MUL lands between the RELU it multiplies
and the MUL, and the CPU backend's convolution epilogue (ggml-0036) only fuses a MUL that is the next
node computed. With one mask per resolution, every use after the first follows its producer directly.

Only ops in `PURE_OPS` merge: stateless and deterministic, so two nodes with the same op, the same
inputs (after earlier merges are substituted) and the same attributes compute the same tensor. A node
whose output the topology declares is never removed, since a driver reads those by name.
"""
import json
from typing import Dict, List, Tuple

import numpy as np

#: The ops two copies of which, on the same inputs and attributes, are the same tensor. A whitelist on
#: purpose: anything that reads or writes state (ATTENTION's KV cache, SHORT_CONV, SSM_*, RWKV_*), or
#: whose cost makes a duplicate worth a look before merging it (the convolutions, MUL_MAT), stays out.
PURE_OPS = frozenset({
    # elementwise arithmetic and math
    "ADD", "SUB", "MUL", "DIV", "FLOOR_DIV", "SCALE", "SQR", "SQRT", "RSQRT", "ABS", "NEG", "SIGN",
    "EXP", "LOG", "SIN", "COS", "ATAN", "ATAN2", "POW", "ROUND", "FLOOR", "CLAMP", "MAXIMUM", "MINIMUM",
    # activations
    "RELU", "SIGMOID", "TANH", "SILU", "GELU", "ELU", "LEAKY_RELU", "SOFTPLUS", "STEP",
    # comparisons and selection
    "EQUAL", "NOT_EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL", "NOT", "SELECT",
    # casts, layout and index generation
    "CAST", "IDENTITY", "RESHAPE", "PERMUTE", "VIEW", "CONT", "CONCAT", "REPEAT", "RANGE_1D", "FILL",
    "SHAPE",
})


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        return super().default(obj)


def _outputs(node: dict) -> List[str]:
    if "outputs" in node:
        return list(node["outputs"])
    if "output" in node:
        return [node["output"]]
    return []


def _declared_outputs(topo: dict) -> set:
    if "outputs" in topo:
        return set(topo["outputs"])
    if "output" in topo:
        return {topo["output"]}
    return set()


def _rebound_names(topo: dict) -> set:
    """The names the topology defines more than once (a graph input counts as a definition).

    A topology is not single-assignment: a name may be written again further down, and the engine
    reads whichever definition came last. Chatterbox's estimator writes `transpose_0 = PERMUTE(input_7)`,
    pads it, and writes `transpose_0` again; merging a later `transpose_1 = PERMUTE(input_7)` into the
    first one redirected the residual conv to the PADDED tensor (ADD: 366 vs 364 frames)."""
    count: Dict[str, int] = {}
    names = [i["name"] if isinstance(i, dict) else i for i in topo.get("inputs", [])]
    names += [o for node in topo.get("nodes", []) for o in _outputs(node)]
    for n in names:
        count[n] = count.get(n, 0) + 1
    return {n for n, c in count.items() if c > 1}


def eliminate_common_subexpressions(topo: dict) -> Tuple[dict, int]:
    """`topo` with every repeated pure node merged into its first occurrence, and how many were removed.

    One forward pass suffices: nodes are in dependency order, and each node's inputs are rewritten to
    the surviving names before its key is taken, so a whole duplicated chain collapses link by link.
    A node that writes a name the topology defines twice neither merges nor survives a merge, since its
    name would not mean the same tensor at every later use; and an input is keyed by which definition
    of it a node reads, so two nodes on the same name either side of a rewrite stay apart.
    The input dict is not modified."""
    keep_names = _declared_outputs(topo)
    rebound = _rebound_names(topo)
    version: Dict[str, int] = {}
    rename: Dict[str, str] = {}
    seen: Dict[tuple, List[str]] = {}
    nodes = []
    removed = 0
    for node in topo.get("nodes", []):
        node = dict(node)
        if "inputs" in node:
            node["inputs"] = [rename.get(i, i) if isinstance(i, str) else i for i in node["inputs"]]
        outs = _outputs(node)
        if node.get("op") in PURE_OPS and outs and not (set(outs) & rebound):
            rest = {k: v for k, v in node.items() if k not in ("op", "inputs", "outputs", "output")}
            reads = [[i, version.get(i, 0)] if isinstance(i, str) else i for i in node.get("inputs", [])]
            key = (node["op"], json.dumps(reads, cls=_Encoder),
                   json.dumps(rest, sort_keys=True, cls=_Encoder), len(outs))
            first = seen.get(key)
            if first is not None and not (set(outs) & keep_names):
                for o, f in zip(outs, first):
                    rename[o] = f
                removed += 1
                continue
            if first is None:
                seen[key] = outs
        for o in outs:
            version[o] = version.get(o, 0) + 1
        nodes.append(node)
    out = dict(topo)
    out["nodes"] = nodes
    return out, removed
