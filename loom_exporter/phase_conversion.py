"""Converting ONE phase of a multi-phase model, and packing its weights while it is still the only
phase in hand (BACKLOG.md P5.0).

Peak RSS during *conversion* -- not the family template, not the driver -- is what decides which
checkpoints this exporter can produce a GGUF for at all. `MultiPhase.export` made that peak a *sum*
over phases where it should be a *max*, and P5.0 is three changes that take it apart. The first
shipped with family 3's second leaf: the traced module, the wrapper and the converted MIL program are
the *same arrays*, so they have to be released together -- dropping the torch half alone frees 0.2 GB,
measured. That took Granite-Speech from 30.4 GB to 22.9 GB, OOM to clean export.

**This module is the second change.** What change 1 did not touch is a phase's *weights*: they were
carried as F32 numpy arrays from the moment that phase converted until the last one had, and only then
dtype-cast and quantized, in one pass inside `write_gguf`. `convert_phase` calls
`LoomGGUFExporter.pack_weights()` as soon as the phase's topology exists, so what crosses a phase
boundary is the tensor's **on-disk payload** -- a quarter of its F32 size at Q8_0 -- rather than its
F32 array.

**What that is and is not worth, measured rather than asserted.** Peak RSS, `--quantize Q8_0`, on a
4-core/33 GB box with no swap: granite-speech-4.0.1b **20.99 -> 21.11 GiB**, kokoro **2.67 -> 2.74**.
It did not move either peak, and the reason is structural rather than a defect: the peak is
`max over phases (what was carried, plus that phase's own conversion)`, and on both models the phase
that sets it is one where little had been carried yet. Granite's order is encoder -> embed ->
**decoder** -> lm_head and the decoder is the 40-layer LM; 16.1 GiB was resident when it began
converting, of which the encoder's packed weights had saved 0.3. This is the right shape for a model
whose large phases come first, or whose phases are many and alike, and it is not what saves Granite.
What does is P5.0's third change, which draws a process boundary around `convert_phase`.

Separating the function from `MultiPhase.export`'s loop is what makes that boundary drawable at all:
everything a converted phase is worth is in the `PhaseResult` it returns, and none of it holds the
conversion's memory.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class PhaseResult:
    """Everything the rest of a multi-phase export wants from one converted phase, and nothing that
    would pin the conversion's memory.

    One phase can produce more than one topology -- a `RecurrentPhase` emits a cell per direction per
    layer, and an `ExportPhase` with `extra_streams` emits an alias per stream -- so `topologies` is a
    dict rather than a single entry keyed by the phase's name.

    `weights` are already PACKED: dtype-normalised, conv kernels folded, eligible tensors quantized.
    `packing` is what `write_gguf` reads each one's `raw_dtype` back out of, and what the export's
    final coverage line is summed from.
    """

    topologies: Dict[str, dict]
    weights: Dict[str, np.ndarray]
    packing: "object"  # exporter.WeightPacking; imported lazily to keep this module import-cheap
    geometry: Dict[str, list] = field(default_factory=dict)


def convert_phase(phase, quantize: Optional[str] = None) -> PhaseResult:
    """Trace, convert, lower and pack one phase, then release everything the conversion needed.

    The release is P5.0's first change and the comment below it is the whole finding: the traced
    module, the wrapper and the converted MIL program are the SAME arrays, so they have to die
    together or none of them frees anything.
    """
    import gc

    import coremltools as ct
    import torch

    from .decomposition import _rss_note
    from .exporter import LoomGGUFExporter
    from .multi_phase_export import RecurrentPhase

    # A RecurrentPhase produces its topologies without a static trace at all: ggml has no LSTM op, so
    # an nn.LSTM becomes per-timestep CELL topologies plus a host-side loop. See
    # multi_phase_export.RecurrentPhase, and LoomGGUFExporter.generate_graph_topology's own raise,
    # which named this as the missing wiring.
    if isinstance(phase, RecurrentPhase):
        cells, weights = phase.topologies()
        print(f"  {phase.name}: {len(cells)} recurrent cell topolog(ies), {len(weights)} weights")
        # Packed through an exporter like every other phase's, and for a reason that is easy to miss:
        # a cell's `MUL_MAT` nodes make its gate weights quantization-eligible, and they were eligible
        # before this split because the merged topologies included the cells. Packing them here is
        # what keeps that true now that the merge sees bytes rather than arrays.
        packer = LoomGGUFExporter(None, quantize=quantize)
        packer.topologies = cells
        packer.weights = weights
        packing = packer.pack_weights()
        # Released for the same reason an `ExportPhase`'s wrapper is, and it was not being: the old
        # loop returned early for this branch and never reached the release below. `build_lstm_cell_
        # topologies` has copied every gate matrix out by now, so the module is holding a second copy
        # of them -- six times over on Kokoro and StyleTTS2, which have six BiLSTMs each.
        phase.module = None
        gc.collect()
        return PhaseResult(topologies=cells, weights=packer.weights, packing=packing)

    traced = torch.jit.trace(phase.wrapper, phase.dummy_inputs)
    mil_prog = ct.convert(
        traced, inputs=phase.mil_inputs, convert_to="milinternal",
        compute_precision=ct.precision.FLOAT32,
    )
    exporter = LoomGGUFExporter(
        mil_prog, root_axis=phase.root_axis, declared_axes=phase.declared_axes,
        # Per-phase, not per-export: an encoder-decoder model caches its decoder's attention and must
        # not cache its encoder's. See ExportPhase.fuse_attention.
        fuse_attention=phase.fuse_attention, kv_cache_size=phase.kv_cache_size,
        # Reaches a PHASE exporter as of P5.0's second change, where it used to reach only the output
        # one. Quantization is what makes packing early worth doing at all -- an F32 pack is a dtype
        # cast and frees nothing.
        quantize=quantize,
    )
    main_func = mil_prog.functions["main"]
    topo = exporter.generate_graph_topology(main_func, phase.name)
    if phase.topology_rewrite is not None:
        # See ExportPhase.topology_rewrite: the one thing a wrapper cannot express is a transform of a
        # graph input that is invariant across the DRIVER's calls. A rewrite raises if its pattern is
        # absent; nothing here checks it, because there is nothing here that knows what it was looking
        # for.
        before = len(topo["nodes"])
        phase.topology_rewrite(topo)
        print(f"  {phase.name}: topology rewrite removed {before - len(topo['nodes'])} node(s)")
    print(f"  {phase.name}: {len(topo['nodes'])} nodes, {len(exporter.weights)} weights{_rss_note()}")

    topologies = {phase.name: topo}
    # See ExportPhase.extra_streams. A second stream of one topology, declared so the engine gives it
    # its own KV cache and its own retained output rather than sharing this one's. A shallow copy on
    # purpose: it shares the node list, so the fold `pack_weights` runs below rewrites both at once.
    for alias in phase.extra_streams:
        topologies[alias] = dict(topo, kv_cache_scope="private")

    exporter.topologies = topologies
    packing = exporter.pack_weights()
    weights = exporter.weights
    geometry = exporter.fused_geometry()

    # **Everything this phase needed is now extracted, and holding any of it is what decides which
    # models can be exported at all** (BACKLOG.md P5.0). Three things die here together, and they have
    # to die together because they are the same memory: `phase.wrapper` holds the submodule's torch
    # parameters, `torch.jit.trace` holds a module beside it, and the converted MIL program's constants
    # are those same arrays again. Releasing any one of them alone frees nothing -- measured, not
    # assumed: dropping the torch half while the program lived moved the peak by 0.2 GB.
    #
    # What the code after this wants from `phase` is `name`, `root_axis` and `kv_cache_size`; what it
    # wants from this conversion is the three values above, all of which are ordinary data by now.
    del traced, main_func, mil_prog, exporter
    phase.wrapper = None
    gc.collect()
    print(f"    released {phase.name}'s torch and MIL halves:{_rss_note() or ' (rss unknown)'}")
    return PhaseResult(topologies=topologies, weights=weights, packing=packing, geometry=geometry)
