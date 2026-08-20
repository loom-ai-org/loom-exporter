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


def main_export(model_path: str, output_path: str, task: str = None, model: str = None,
                quantize: str = None) -> str:
    """Exports whatever `model_path` names to `output_path`. `task`/`model` are optional overrides --
    with neither, both axes are auto-detected; with `task` alone, detection is restricted to that task's
    recognizers; `model` requires `task` (it names one specific recognizer within it). `quantize` names
    a GGML type for the weights that are eligible for one; unset falls back to $LOOM_QUANTIZE. Returns
    `output_path`."""
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
    return config.export()


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
    args = parser.parse_args()

    output_path = main_export(args.model_path, args.output, task=args.task, model=args.model,
                              quantize=args.quantize)
    print(f"SUCCESS! Exported to: {output_path}")


if __name__ == "__main__":
    main()
