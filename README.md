<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom-exporter
</h1>

The export pipeline for [loom.cpp](https://github.com/loom-ai-org/loom.cpp). It traces a PyTorch model
through Core ML Tools' MIL, lowers that program to the engine's own primitives, and writes a single
GGUF carrying the weights, the **graph topologies** as JSON and the **driver script** as embedded Lua.
The engine then runs that file without knowing anything about the architecture.

All the per-model complexity lives here on purpose: the engine targets edge devices, so a new
architecture should cost a Python change rather than a specialized C++ driver.

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | the runtime that executes what this produces |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | this one — tracing, lowering, driver synthesis |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | Python bindings to the engine |

## Exporting

```sh
./loom-export /path/to/checkpoint -o model.gguf
```

The task and architecture are detected from the checkpoint; `--task` and `--model` override that, and
an unrecognised checkpoint gets a candidate list rather than a guess. The registry of what can be
exported is in [`loom_exporter/registry.py`](loom_exporter/registry.py).

## Testing

Same two classes as the engine, and a test's directory is which class it is in.

```sh
pytest tests/ci      # hermetic: traces toy modules through the real compiler. What CI runs.
pytest tests/gate    # exports real checkpoints and diffs the artifacts. Needs models and hours.
```

`tests/ci/` needs torch and coremltools but **no checkpoint** — every model it compiles is built in
the test. `tests/gate/` is the byte-identity sweep: export a real checkpoint, snapshot the GGUF's
structure, and diff it against a baseline, which is the regression gate any change to this repo is
held to. It skips cleanly when the checkpoints are absent.

## Documentation

| | |
|---|---|
| [`docs/EXPORT-ROADMAP.md`](docs/EXPORT-ROADMAP.md) | the model-family map: what is covered, what is next, and at what cost |
| [`docs/EXPORT-PREPARATION.md`](docs/EXPORT-PREPARATION.md) | the decisions the pipeline is built on |
| [`docs/DRIVER-COMPONENTS.md`](docs/DRIVER-COMPONENTS.md) | every driver component, what it checks, and which models use it (generated) |
| [`docs/LOOM_MIL_CONVERSION.md`](docs/LOOM_MIL_CONVERSION.md) | how a MIL program becomes engine primitives |
| [`docs/BACKEND.md`](docs/BACKEND.md) | the working log of exporter findings |

The project ledger lives with the engine, in [loom.cpp's `BACKLOG.md`](https://github.com/loom-ai-org/loom.cpp/blob/main/BACKLOG.md).

## Licence

MIT — see [`LICENSE`](LICENSE).
