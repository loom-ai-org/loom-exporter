"""Converting a phase in a process of its own (BACKLOG.md P5.0's third change).

Peak RSS during conversion is what decides which checkpoints this exporter can produce a GGUF for at
all, and `MultiPhase.export` made that peak a *sum* over phases where it should be a *max*. The first
two changes work inside one process -- release a phase's torch and MIL halves together (30.4 -> 22.9 GB
on Granite-Speech), then pack its weights as it converts (`phase_conversion.convert_phase`, which moved
no peak at all; its docstring says why). What neither can reach is everything a conversion touched that
Python cannot see it holding: coremltools' and torch's own caches, and the arenas glibc keeps rather
than returning to the OS. A process that exits returns all of it, unconditionally.

**What it is worth.** Peak RSS, `--quantize Q8_0`, 4 cores / 33 GB / no swap. *self* is the largest a
single process had to hold; *tree* sums the whole descendant tree, which is what the machine must
satisfy:

| | without | with |
|---|---|---|
| granite-speech-4.0.1b, 3.2 GB artifact | 21.11 | **14.56** self / **15.85** tree |
| kokoro, 143 MB artifact | 2.74 | 2.56 self / **4.34** tree |

Granite is the model P5.0 was scoped against and the largest single allocation fell by 31%. Kokoro is
the counter-case and it is not an aberration: a 143 MB artifact's peak is the framework floor, and
isolation pays that floor twice. That is the measured form of the argument for leaving this off by
default -- it is a property of the *machine*, not of the model.

**Isolation is a RESPAWN, not a fork, and that is measured rather than chosen.** A fork would have been
free -- the child inherits the config object, the loaded checkpoint and every wrapper by copy-on-write,
so nothing would need to be serialized or reloaded. It deadlocks. Probed on this machine (torch
2.8.0+cpu, coremltools 9.0), forking and then calling `torch.jit.trace` in the child:

| parent before the fork | child |
|---|---|
| no forward pass at all | converts, exits 0 |
| `torch.set_num_threads(1)`, then 20 forwards | converts, exits 0 |
| 20 forwards at the default thread count | **hangs indefinitely** |

which is the classic OpenMP-after-fork hazard: the child inherits the thread pool's locks without its
threads. It is not avoidable here by ordering, because `phases()` itself runs real forward passes --
the speech-LM family probes its encoder's row geometry and its padding invariance before it returns a
single phase, and that is exactly the check that makes `audio_geometry` safe to declare. Serializing
the config and paying for a second checkpoint load is the cost of not having that hang.

**What crosses the boundary.** Into the child: the config, pickled *before* `phases()` has run, which
for every multi-phase family in the registry is 288-710 bytes of paths and scalars. Out of it: the
phase's topologies as JSON and its packed weights as one spill file the parent memory-maps. The parent
therefore never holds a phase's weights in anonymous memory at all -- `GGUFWriter.add_tensor` keeps
the array and `write_tensors_to_file` streams it with `tofile`, so a memmap goes from the spill
straight into the artifact, page by page.

**The child calls `phases()` too**, and loads the checkpoint to do it. That is the honest cost of this
shape: an N-phase model is loaded N+1 times, and Granite's four phases took roughly twice the baseline's
wall time. The alternative the backlog names -- load only that phase's submodule -- needs a per-family
partial-loading hook that does not exist, and would not change the peak, only the time.
"""
import contextlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from .phase_conversion import PhaseResult

#: Spill payloads start on a 64-byte boundary. `np.memmap` accepts any offset (it rounds down to the
#: allocation granularity itself and slices back), so this is not a correctness requirement -- it is so
#: a mapped weight is cache-line aligned for the `tofile` that reads it straight into the GGUF.
_SPILL_ALIGN = 64

#: The env var `--isolate-phases` sets, and the one a caller who is not going through the CLI reaches
#: for. Same shape as `$LOOM_QUANTIZE`: the flag wins, the variable is the fallback.
ISOLATION_ENV = "LOOM_PHASE_ISOLATION"


@contextlib.contextmanager
def spill_root_for(output_path: str):
    """A temporary directory for the phase spills, BESIDE the artifact -- not under `$TMPDIR`.

    **/tmp is a tmpfs on a normal Linux box**, sized at a fraction of RAM, and this machine's is small
    enough that the model sweep already has a standing rule against writing exports there. Spilling a
    phase's weights to a tmpfs would put them straight back into memory under a different name, which
    is precisely the one thing this mechanism must not do -- and it would not even fail cleanly: the
    export would run, the peak would be the sum again, and the only symptom would be an OOM at a
    different line.

    Beside the output is the right default because the spill is *bounded by the artifact*: it holds
    exactly the tensors the GGUF will hold, in exactly the form it will hold them, so a directory with
    room for the output has room for the spill. `$LOOM_PHASE_SPILL_DIR` overrides it for the case that
    is not true -- a network-mounted output directory, say.
    """
    root = os.environ.get("LOOM_PHASE_SPILL_DIR") or str(Path(output_path).resolve().parent)
    Path(root).mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix=".loom-phases-", dir=root)
    try:
        yield Path(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def release_heap_to_os() -> bool:
    """`malloc_trim(0)` — hand glibc's free arenas back to the kernel. True if it ran and freed.

    **Measured, and it is the difference between isolation paying and not paying.** A parent that has
    answered `phases()` and then dropped every wrapper is holding nothing a Python profiler can see,
    and it still sat at **4.97 GiB** on Granite-Speech: 1.21 GiB of that is the interpreter with torch
    and coremltools imported, and the rest is arenas glibc kept rather than returning, because
    `free()` only releases the top of the heap. `gc.collect()` cannot touch it -- the objects are
    already gone. One `malloc_trim(0)` took the same process to **1.31 GiB**.

    That matters here and nowhere else in the export. In one process the freed pages are simply reused
    by the next phase, so trimming changes the current RSS and not the peak. Under isolation the
    parent's residency is charged against *every child*, in parallel, for the whole conversion — which
    is the one place a number that looks like bookkeeping is the whole tree's peak.

    glibc-only (`musl` and macOS have no `malloc_trim`), so a failure to find it is a normal answer and
    not an error: the export is correct either way, it just holds more.
    """
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except (OSError, AttributeError):
        return False


def isolation_requested(explicit: Optional[bool] = None) -> bool:
    """Whether to convert phases in child processes: the caller's answer if it has one, else
    `$LOOM_PHASE_ISOLATION`.

    A machine property rather than a model one, which is why it is not a field on the family's config
    the way `phases()` is. The same checkpoint isolates on a 16 GB laptop and does not need to on a
    128 GB workstation, and neither answer is a fact about the checkpoint.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(ISOLATION_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------------------------------
# The spill: one phase's result, on disk, in the form the parent can memory-map straight into a GGUF.
# --------------------------------------------------------------------------------------------------

def write_spill(result: PhaseResult, directory: Path) -> None:
    """`result` into `directory` as `manifest.json` + `weights.bin`.

    One concatenated payload file rather than an `.npy` per tensor: a large phase has thousands of
    weights, and one mapping the parent slices is one file descriptor and one `mmap` rather than
    thousands of each. The manifest carries what `np.memmap` needs to reconstitute each view -- dtype,
    shape, offset -- plus the GGML type the tensor was packed to, which is not recoverable from the
    array (a Q8_0 payload is uint8 bytes and says so about nothing else).
    """
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    with open(directory / "weights.bin", "wb") as blob:
        for name, array in result.weights.items():
            pad = (-blob.tell()) % _SPILL_ALIGN
            if pad:
                blob.write(b"\0" * pad)
            offset = blob.tell()
            contiguous = np.ascontiguousarray(array)
            contiguous.tofile(blob)
            raw_dtype = result.packing.raw_dtypes[name]
            entries.append({
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": [int(d) for d in contiguous.shape],
                "offset": offset,
                # The GGML type NAME, not its value: the enum is the gguf package's and a number would
                # silently mean something else the day that package renumbers it.
                "raw_dtype": None if raw_dtype is None else raw_dtype.name,
            })

    packing = result.packing
    manifest = {
        "topologies": result.topologies,
        # `fused_geometry()`'s records are `(op name, geometry tuple)`; JSON has no tuple, so they
        # come back as lists and `read_spill` puts the tuples back. `_kv_cache_geometry` unpacks them.
        "geometry": {kind: [[name, list(values)] for name, values in records]
                     for kind, records in result.geometry.items()},
        "packing": {
            "n_folded": packing.n_folded,
            "n_declined_shape": packing.n_declined_shape,
            "quantized_bytes_before": packing.quantized_bytes_before,
            "quantized_bytes_after": packing.quantized_bytes_after,
            "float_bytes_total": packing.float_bytes_total,
        },
        "weights": entries,
    }
    from .exporter import NumpyEncoder

    (directory / "manifest.json").write_text(json.dumps(manifest, cls=NumpyEncoder))


def read_spill(directory: Path) -> PhaseResult:
    """A spill back as a `PhaseResult` whose weights are `np.memmap`s over `weights.bin`.

    Mapped rather than read, which is the half that makes isolation pay on the PARENT's side too: the
    parent accumulates N phases' worth of tensors and never faults more of any one of them in than the
    writer's `tofile` walks through. The mapping has to outlive the arrays, so each memmap holds its
    own reference to the file -- which `np.memmap` does.
    """
    from gguf import GGMLQuantizationType

    from .exporter import WeightPacking

    manifest = json.loads((directory / "manifest.json").read_text())
    blob = directory / "weights.bin"
    weights, raw_dtypes = {}, {}
    for entry in manifest["weights"]:
        shape = tuple(entry["shape"])
        if int(np.prod(shape)) == 0:
            # `np.memmap` refuses a zero-length mapping, and a zero-element tensor has no payload to
            # map anyway. Rebuilt empty rather than special-cased downstream.
            weights[entry["name"]] = np.empty(shape, dtype=np.dtype(entry["dtype"]))
        else:
            weights[entry["name"]] = np.memmap(
                blob, dtype=np.dtype(entry["dtype"]), mode="r", offset=entry["offset"], shape=shape,
            )
        raw_dtypes[entry["name"]] = (None if entry["raw_dtype"] is None
                                     else GGMLQuantizationType[entry["raw_dtype"]])

    packing = WeightPacking(raw_dtypes=raw_dtypes, **manifest["packing"])
    geometry = {kind: [(name, tuple(values)) for name, values in records]
                for kind, records in manifest["geometry"].items()}
    return PhaseResult(topologies=manifest["topologies"], weights=weights, packing=packing,
                       geometry=geometry)


# --------------------------------------------------------------------------------------------------
# Parent and child.
# --------------------------------------------------------------------------------------------------

def convert_phase_isolated(config_blob: Path, index: int, directory: Path,
                           quantize: Optional[str] = None) -> PhaseResult:
    """Convert phase `index` of the pickled config in a child process and read its spill back.

    Raises on a non-zero exit rather than falling back to converting in-process: an export that
    silently stopped isolating is an export whose peak is the sum again, which is the one thing a
    caller asked for by turning this on. The child's stdout and stderr are inherited, so its own
    per-phase node/weight/rss lines land in the same log as every other phase's.
    """
    directory.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "loom_exporter.phase_isolation",
            str(config_blob), str(index), str(directory)]
    if quantize:
        argv += ["--quantize", quantize]
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"the phase-{index} worker exited {completed.returncode}. Its traceback is above, in this "
            f"process's own stderr. Re-run without --isolate-phases to convert every phase in one "
            f"process, which is the same conversion with a larger peak and a shorter traceback."
        )
    return read_spill(directory)


def _phase_worker(argv: List[str]) -> int:
    """`python -m loom_exporter.phase_isolation <config.pkl> <index> <spill dir> [--quantize TYPE]`.

    Rebuilds the phase list from the config rather than receiving the phase, because an `ExportPhase`
    holds a live `nn.Module` and is not serializable in any useful sense -- the whole point of the
    boundary is that the module never crosses it. `phases()` is called exactly as the parent calls it,
    so anything it checks (the speech-LM family's row-geometry and padding-invariance probes) is
    checked here too, on this checkpoint, in this process.
    """
    import argparse
    import gc

    parser = argparse.ArgumentParser(prog="loom_exporter.phase_isolation", description=__doc__)
    parser.add_argument("config_blob")
    parser.add_argument("index", type=int)
    parser.add_argument("directory")
    parser.add_argument("--quantize", default=None)
    args = parser.parse_args(argv)

    with open(args.config_blob, "rb") as handle:
        config = pickle.load(handle)
    config.prepare_environment()
    phases = config.phases()
    if not 0 <= args.index < len(phases):
        raise IndexError(
            f"phase index {args.index} is out of range: this config declares {len(phases)} phase(s). "
            f"The parent and the child disagree about the phase list, which means `phases()` is not a "
            f"function of the config alone for this family."
        )
    phase = phases[args.index]
    # Drop every phase this worker is NOT converting before it converts anything. They were built by
    # the `phases()` call above and each holds its own submodule's parameters, which is the peak this
    # whole mechanism exists to bound -- isolating a phase and then keeping its three siblings
    # resident would isolate the conversion and not the memory.
    for other in phases:
        if other is not phase:
            for attribute in ("wrapper", "module"):
                if getattr(other, attribute, None) is not None:
                    setattr(other, attribute, None)
    del phases
    gc.collect()

    from .phase_conversion import convert_phase

    write_spill(convert_phase(phase, quantize=args.quantize), Path(args.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(_phase_worker(sys.argv[1:]))
