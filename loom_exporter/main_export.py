"""Single entry point for exporting any model this project knows how to export -- BACKLOG.md P3.2's
`main_export()` + `loom-export` CLI, this project's `optimum-cli export onnx --model <id>` equivalent.

    loom-export <model-path> -o <out.gguf>                              # fully automatic
    loom-export <model-path> -o <out.gguf> \\
        --task automatic-speech-recognition --model parakeet-tdt        # explicit

See `registry.py` for how a model is recognized (task, then model within it) and BACKLOG.md's P3.2
entry for why detection is two-axis rather than one flat per-model key; `tasks.py` for the canonical
task vocabulary `--task` accepts.
"""
import argparse
from pathlib import Path

import numpy as np

#: One block-aligned F32 row, used only to ask `gguf.quants` which types it can actually write.
_PROBE = np.zeros((1, 256), dtype=np.float32)

from .registry import default_registry
from .tasks import known_tasks


def quantize_choices() -> list:
    """The GGML type names `--quantize` accepts, from the `gguf` package rather than a hardcoded list.

    Restricted to the types this exporter can actually WRITE: `gguf.quants.quantize` implements a
    subset of the enum, and offering a name it raises on turns a typo into a traceback halfway through
    an export that has already spent minutes tracing.
    """
    from gguf import GGMLQuantizationType, quants

    names = []
    for qtype in GGMLQuantizationType:
        try:
            quants.quantize(_PROBE, qtype)
        except Exception:
            continue
        names.append(qtype.name)
    return names


def validate_quantize(name: str) -> str:
    """`name` uppercased and checked against `quantize_choices()`, or a ValueError that lists them."""
    resolved = name.upper()
    choices = quantize_choices()
    if resolved not in choices:
        raise ValueError(
            f"unknown quantization {name!r}. This exporter can write: {', '.join(choices)}."
        )
    return resolved


def companion_output(output_path: str, companion) -> str:
    """Where a companion's GGUF goes: beside the primary one, named after the companion.

    `-o out/talker.gguf` puts the codec at `out/qwen3-tts-tokenizer-12hz.gguf` -- the companion's own
    catalogue slug, so the two files on disk are named the way the two repos on the Hub are, and so a
    second talker exported into the same directory writes the identical codec path rather than a
    second copy under a talker-derived name.
    """
    return str(Path(output_path).parent / f"{companion.name}.gguf")


def main_export(model_path: str, output_path: str, task: str = None, model: str = None,
                quantize: str = None, companions: bool = False, isolate_phases: bool = None) -> str:
    """Exports whatever `model_path` names to `output_path`. `task`/`model` are optional overrides --
    with neither, both axes are auto-detected; with `task` alone, detection is restricted to that task's
    recognizers; `model` requires `task` (it names one specific recognizer within it). `quantize` names
    a GGML type for the weights that are eligible for one; unset falls back to $LOOM_QUANTIZE.
    `isolate_phases` converts each phase of a multi-phase model in its own process; unset falls back to
    $LOOM_PHASE_ISOLATION. Returns `output_path`."""
    if model is not None and task is None:
        raise ValueError("--model requires --task (which family's recognizer to look up)")
    if quantize:
        quantize = validate_quantize(quantize)

    registry = default_registry()
    path = Path(model_path)
    recognizer = registry.get(task, model) if model is not None else registry.detect(path, task)
    config = recognizer.build_config(path, output_path)
    # The task, onto the config, because this is the last point at which anything knows it: `detect()`
    # returns a recognizer and `build_config` is handed a path and an output path. It used to stop here
    # -- `tasks.py` argued a task name "never reaches build_config, let alone a KV" -- which was right
    # while no host offered a task-shaped door and stopped being right the moment one did. It is written
    # into the file as `loom.task` now (loom.cpp docs/HIGH-LEVEL-API.md §3).
    config.task = recognizer.task
    # Same instance-attribute route as `task`, and for the same reason: this is a property of the
    # REQUESTED export, so it arrives from the caller rather than from the recognizer or the checkpoint.
    # `resolved_backend_kwargs` is what carries it to the backend, so every family gets it -- including
    # the ones whose own `backend_kwargs` never learned about quantization.
    if quantize:
        config.quantize = quantize
    # Onto the DECOMPOSITION rather than the config, because that is what reads it -- and only one of
    # the three has phases to isolate. A request for it against a single-graph model is accepted and
    # does nothing rather than raising: detection picks the decomposition, so a caller exporting a
    # directory of checkpoints cannot know in advance which of them are multi-phase.
    if isolate_phases is not None:
        from .decomposition import MultiPhase

        if isinstance(getattr(config, "decomposition", None), MultiPhase):
            config.decomposition.isolate_phases = isolate_phases
    primary = config.export()

    # **Off by default, and the default is the load-bearing half.** `tools/build_model_cards.py` calls
    # this once per Hub repo with that repo's own output directory; a second GGUF appearing beside the
    # first would put a stray file in a repo whose README lists exactly one. The CLI opts in, because
    # the CLI is where a person -- rather than a script that already knows -- points at a checkpoint.
    if companions:
        for companion in config.companions():
            destination = companion_output(output_path, companion)
            print(f"\nAlso in this checkpoint: {companion.name} -- {companion.why}.")
            print(f"  exporting it to {destination}")
            # `isolate_phases` travels with it: a companion is its own multi-phase export (family
            # 10's talker ships a codec that is one), so a caller who needed the flag for the primary
            # needs it for what the primary drags along.
            main_export(str(companion.checkpoint), destination, quantize=quantize, companions=True,
                        isolate_phases=isolate_phases)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Path to a checkpoint directory or file (HF dir, .nemo archive, ...)")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF path")
    parser.add_argument(
        "--task", default=None, choices=known_tasks(),
        help="Restrict/override task detection (the canonical vocabulary, see tasks.py)",
    )
    parser.add_argument("--model", default=None, help="Explicit model override within --task, e.g. 'qwen3'")
    parser.add_argument(
        "--quantize", default=None, type=str.upper, choices=quantize_choices(), metavar="TYPE",
        help="Quantize eligible weights to this GGML type (e.g. Q8_0, F16). Default: $LOOM_QUANTIZE, "
             "else none. Only weights ggml can read in that form are converted -- the export reports "
             "the coverage it achieved, which for convolutional models is a fraction of the file.",
    )
    parser.add_argument(
        "--isolate-phases", action="store_true",
        help="Convert each phase of a multi-phase model in its own process, so the peak memory of an "
             "export is one phase's conversion rather than the sum of all of them (BACKLOG.md P5.0). "
             "Costs a checkpoint load per phase, so it is off unless asked for -- turn it on for a "
             "model that OOMs. Default: $LOOM_PHASE_ISOLATION.",
    )
    parser.add_argument(
        "--no-companions", action="store_true",
        help="Do not export the other models this checkpoint contains. A family-10 talker ships its "
             "own codec (Qwen3-TTS does; Dia's is a separate repo), and by default that second GGUF "
             "is written beside this one -- they stay two files, which is what a codec being shared "
             "and the codes being the useful intermediate buy you.",
    )
    args = parser.parse_args()

    output_path = main_export(args.model_path, args.output, task=args.task, model=args.model,
                              quantize=args.quantize, companions=not args.no_companions,
                              isolate_phases=args.isolate_phases or None)
    print(f"SUCCESS! Exported to: {output_path}")


if __name__ == "__main__":
    main()
