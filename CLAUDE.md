# loom-exporter — orientation

The export pipeline. One of **three repos**, all under `github.com/loom-ai-org` and, on a dev machine,
side by side under one parent directory:

| | |
|---|---|
| `loom.cpp` | the ggml engine that runs what this produces; holds `BACKLOG.md`, the shared ledger |
| `loom-exporter` | this repo — tracing, lowering, driver synthesis |
| `loom-py` | Python bindings to the engine |

## The one idea

A PyTorch model is traced through Core ML Tools' **MIL**, that program is lowered to the engine's own
primitives, and the result is written as a single GGUF carrying the weights, the *graph topologies* as
JSON and the *driver script* as embedded Lua. The engine then runs the file without knowing anything
about the architecture.

**All the per-model complexity belongs here on purpose.** The engine targets edge devices, so a new
architecture should cost a Python change rather than a specialized C++ driver.

## Layout

```
loom_exporter/     the package: tracing, passes, driver components, per-family configs
tools/             the pre-MIL per-model converters, plus codegen and quantize. Several are still
                   live: they own checkpoint loaders and the hand-written *_driver/ Lua fragments
fixture_gen/       reference-forward generators for REAL checkpoints (the gate suite's oracles)
docs/              the exporter's markdown
```

**Never locate a sibling with `Path(__file__).parent.parent`.** That idiom meant `tools/` before the
repo split and something else after, and it broke twelve places at once plus every test that had
copied it. `loom_exporter/paths.py` holds the answer: `REPO_ROOT`, `CONVERTERS`, `driver_dir()`,
`engine_root()`.

## Exporting and testing

```sh
./loom-export /path/to/checkpoint -o model.gguf     # task/architecture auto-detected
pytest tests/ci      # 511 hermetic tests, no checkpoint. What CI runs.
pytest tests/gate    # the byte-identity sweep: needs real models and hours
```

**A test's directory is which class it is in.** `tests/ci/` traces toy modules through the real
compiler; `tests/gate/` exports real checkpoints, snapshots each GGUF's structure and diffs it against
a recorded baseline. That diff is the gate this repo is actually held to, because almost every change
here is meant to be output-preserving and almost none of that is covered by asserting the exporter
still runs.

## Conventions worth knowing before changing anything

* **`BACKLOG.md` lives in loom.cpp** and is the ledger for all three repos. Code here references it by
  item (`BACKLOG.md P4.3e`). Read the relevant entry first; add to it when you finish something.
* **The catalogue is generated.** `docs/DRIVER-COMPONENTS.md` comes from the component declarations —
  run `python -m loom_exporter.component_registry` after adding one, or CI fails.
* **A gate that cannot fail proves nothing.** The sweep has reported a false pass before, by measuring
  a baseline against itself. Name the model you expect to differ, and check that it did.
* **Two environments, and neither exports everything.** NeMo pins `transformers~=4.53` while Qwen3-ASR
  needs `>=5.13`, so there is no single resolve; `docs/EXPORT-PREPARATION.md` records which is which.
* **Comments say why, not what** — where a line is the way it is because an alternative failed, the
  comment records the failure and the measurement. Match that density.
