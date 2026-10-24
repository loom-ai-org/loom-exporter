<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-inline-dark.svg">
    <img src="assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom-exporter
</h1>

<p align="center">
  <b>Turn a PyTorch checkpoint into one self-describing GGUF that <a href="https://github.com/loom-ai-org/loom.cpp">loom.cpp</a> runs anywhere.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/loom-ai-org"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20exported%20models-64-ffcc4d"></a>
  <a href="LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-1a7f37"></a>
</p>

---

```sh
loom-export ~/models/whisper-small -o whisper-small.gguf
```

```python
import loom                                    # pip install loom-py-rt
model = loom.Model.from_file("whisper-small.gguf")
print(model.speech2text.infer(audio, language="en").text)
```

The exporter traces the model, lowers it to loom's primitives and writes **one file** holding the
weights, the compute graphs, the script that drives them, and the tokenizer, voices and chat template
where the model has them. The engine then runs that file without knowing anything about the
architecture, on a GPU, a laptop or a Raspberry Pi Zero.

Every model on [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org) was made with it: speech
recognition (Whisper, Parakeet, Canary, Moonshine, ...), text to speech (Kokoro, Piper VITS, Pocket TTS,
CosyVoice3, Qwen3-TTS, Voxtral, ...), VAD, speaker and language id, codecs, music and small language
models. The [loom.cpp README](https://github.com/loom-ai-org/loom.cpp#supported-models) has the full list.

## Install

```sh
git clone https://github.com/loom-ai-org/loom-exporter && cd loom-exporter
pip install -e .
```

It needs PyTorch, transformers and Core ML Tools. Some checkpoints' own libraries pin transformers in
opposite directions (NeMo wants `~=4.53`, Qwen3-ASR needs `>=5.13`), so no single environment exports
everything. [`docs/EXPORT-PREPARATION.md`](docs/EXPORT-PREPARATION.md) lists which environment exports
which model.

## Export

```sh
# the task and architecture are detected from the checkpoint
loom-export /path/to/checkpoint -o model.gguf

# or named explicitly
loom-export /path/to/parakeet-tdt-0.6b-v3.nemo -o parakeet.gguf \
    --task automatic-speech-recognition --model parakeet-tdt

# per-channel int8 for small boards (fast on a Raspberry Pi 4 and Zero)
loom-export /path/to/checkpoint -o model-q8_0_pc.gguf --quantize Q8_0_PC
```

| Option | |
|---|---|
| `--task`, `--model` | override detection; an unrecognised checkpoint gets a list of candidates instead of a guess |
| `--quantize TYPE` | `Q8_0_PC`, `Q8_0`, `F16`, ... for the weights ggml can read in that form; the export reports how much it covered |
| `--isolate-phases` | convert each part of a multi-part model in its own process, for a model whose export runs out of memory |
| `--no-companions` | skip the second file some checkpoints contain (Qwen3-TTS's codec is written beside the talker by default) |

`loom-export --help` lists the tasks. What can be exported is the registry in
[`loom_exporter/registry.py`](loom_exporter/registry.py).

## Contributing

```sh
pytest tests/ci      # hermetic: traces toy modules through the real compiler, no checkpoint (what CI runs)
pytest tests/gate    # exports real checkpoints and diffs each GGUF against a recorded baseline
```

The gate is the byte-identity sweep that every change here is held to; it skips cleanly when the
checkpoints are absent.

| | |
|---|---|
| [`docs/EXPORT-ROADMAP.md`](docs/EXPORT-ROADMAP.md) | the model-family map: what is covered, what is next, and at what cost |
| [`docs/EXPORT-PREPARATION.md`](docs/EXPORT-PREPARATION.md) | the decisions the pipeline is built on, and the two export environments |
| [`docs/DRIVER-COMPONENTS.md`](docs/DRIVER-COMPONENTS.md) | every driver component, what it checks, and which models use it (generated) |
| [`docs/LOOM_MIL_CONVERSION.md`](docs/LOOM_MIL_CONVERSION.md) | how a MIL program becomes engine primitives |
| [`docs/BACKEND.md`](docs/BACKEND.md) | the working log of exporter findings |

The project's knowledge base lives with the engine, in
[loom.cpp's `docs/`](https://github.com/loom-ai-org/loom.cpp/tree/main/docs): open work in the
[hub](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/backlog/active-index.md), decisions in
[ADRs](https://github.com/loom-ai-org/loom.cpp/tree/main/docs/adrs), lessons in
[retros](https://github.com/loom-ai-org/loom.cpp/tree/main/docs/retros), and this repo's domain in
[Epic-02](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/epics/epic-02-mil-exporter-and-compiler.md).

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | the engine that runs what this produces |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | Python bindings, on PyPI as `loom-py-rt` |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | this one: tracing, lowering, driver synthesis |

## Licence

MIT, see [`LICENSE`](LICENSE). An exported model keeps the licence of the checkpoint it came from.
