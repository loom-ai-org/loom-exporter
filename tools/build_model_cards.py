#!/usr/bin/env python3
"""Exports models from `CATALOG` below and writes each one, GGUF + README.md, into its own directory
under `--output-dir` -- one directory per HuggingFace repo, ready to `huggingface-cli upload` straight
to https://huggingface.co/loom-ai-org.

    ~/.venvs/piper/bin/python3 tools/build_model_cards.py --list
    ~/.venvs/piper/bin/python3 tools/build_model_cards.py qwen3-0.6b-base whisper-small
    ~/.venvs/piper/bin/python3 tools/build_model_cards.py --all         # every model this venv can export
    ~/.venvs/ovos/bin/python3  tools/build_model_cards.py qwen3-asr-0.6b

Two venvs, same as the rest of the exporter (see BACKLOG.md / [[env-python-venvs-export]]): `piper` does
everything except `qwen3-asr-0.6b`, which needs `ovos`. `--all` only exports what the *running*
interpreter's venv covers, and says so for the rest, rather than crashing on the first mismatched import.

`--readme-only` regenerates just the README.md for models whose GGUF already exists in `--output-dir`
-- useful for fixing card wording without re-running a 7-minute export.

Each catalog entry records, by hand, the one thing this script cannot derive from the checkpoint on
disk: which upstream HF repo (or, for the couple of models with no HF repo, which upstream source) it
was exported from, and that repo's license and language tags, read directly off its model card. See
each entry's `base_repo`/`source_url`, `license_*` and `language` fields.
"""
import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# REPO_ROOT computed locally, not imported from loom_exporter.paths -- this script is invoked directly
# (`python tools/build_model_cards.py`), which puts `tools/` on sys.path[0], not the repo root, so
# `import loom_exporter` would fail before paths.py could even be reached.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODELS_ROOT = Path("/home/flavio/Dev/models")
DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "hf-models"

LOOM_PY_URL = "https://github.com/loom-ai-org/loom-py"
LOOM_CPP_URL = "https://github.com/loom-ai-org/loom.cpp"
EXPORTER_URL = "https://github.com/loom-ai-org/loom-exporter"


@dataclass(frozen=True)
class ModelCard:
    # Directory name under --output-dir, and the suffix-free half of the suggested HF repo id
    # (`loom-ai-org/<slug>-loom`, matching loom-py's own README example).
    slug: str
    # Checkpoint path. Relative paths resolve against --models-root; a couple of models (vits, the
    # supertonic fork) live outside it and are given absolute paths directly -- see
    # [[loom-engine-model-sweep-recipe]] for why.
    checkpoint: Path
    # A canonical task name from `loom_exporter.tasks` -- it picks the usage snippet AND becomes the
    # card's `pipeline_tag`. Deliberately the SAME vocabulary the export declares rather than a
    # card-local list: a card whose pipeline tag disagreed with the file's own `loom.task` would be
    # documenting a different model from the one beside it. `snippet_key` is where a task with more
    # than one shape resolves that.
    task_type: str
    # Whether this model's GGUF carries a vocabulary, for the TTS models where that is not implied by
    # the task: the phoneme-input families (Kokoro, Matcha, VITS, StyleTTS2) take ids a phonemiser
    # produces outside the engine, while a grapheme model (Supertonic) encodes text itself. Only read
    # for task_type == "text-to-speech"; the LM and ASR families all carry one.
    takes_text: bool = False
    # Whether `speech2text.infer(..., language=...)` does anything on this model. Only read for
    # task_type == "automatic-speech-recognition", and a per-model fact for the same reason
    # `takes_text` is: it is not implied by the task, and it is not implied by how many languages the
    # checkpoint was TRAINED on either. Whisper is windowed, so each window gets a prompt and a
    # language token can go in it; every other ASR export here is dynamic-length and calls the driver
    # with the waveform and its length alone, so no language argument can reach the model however
    # multilingual it is. Putting `language=` in those cards taught a copy-paste that the engine now
    # (correctly) warns about.
    selects_language: bool = False
    # Whether this checkpoint is instruction-tuned AND its chat template survived export (P4.23), i.e.
    # whether the GGUF carries `tokenizer.chat_template.*` and `model.text2text.chat(...)` works.
    # Only read for task_type == "text-generation", and a per-model fact for the same reason
    # `takes_text` is: it is not implied by the task. A base model has no template at all, and an
    # instruction-tuned one whose template does not reduce to role tags exports without it.
    #
    # **This is why the card snippet changed.** Every text-generation card used to show
    # `text2text.infer("The capital of France is", max_new_tokens=14)`, which on an IT model returns
    # ' Paris.\n\nThe capital of France is Paris.\n\nThe capital of' -- the model continuing in the
    # prompt's own format because nothing put it inside a turn. The card demonstrated the defect to
    # anyone who pasted it.
    chat: bool = False
    # Whether this token classifier's classes are punctuation MARKS rather than entity TAGS. Only read
    # for task_type == "token-classification", and a per-model fact for the same reason `takes_text` is:
    # both models return one label per token through one door, and what a reader does with the labels is
    # not the same job. A NER card reconstructs SPANS (`B-PER I-PER` is one person); a punctuation card
    # reconstructs a SENTENCE, inserting each word's mark after its last piece. Showing the NER snippet
    # on a punctuation model would print `hello/, o/,` and explain nothing.
    restores_punctuation: bool = False
    # `--task`/`--model` for loom-export; empty means auto-detection resolves both.
    export_task: Optional[str] = None
    export_model: Optional[str] = None
    # Which venv's interpreter can import this model's loader. "piper" covers everything except
    # qwen3-asr, which needs "ovos" (transformers >= 5.13). See [[env-python-venvs-export]].
    venv: str = "piper"
    # The upstream checkpoint this was exported from: an HF repo id (most models), or -- for the two
    # models with no HF repo for the checkpoint itself -- a source_url instead. Exactly one is set.
    base_repo: Optional[str] = None
    source_url: Optional[str] = None
    source_name: Optional[str] = None  # display name when base_repo is None
    # License, read off the upstream repo's own model card (or its LICENSE file when the repo publishes
    # no `license:` tag). `license_id` is what HF's `license:` YAML key accepts (an SPDX id, or "other");
    # `license_name`/`license_url` are only used when `license_id == "other"`.
    license_id: str = "other"
    license_name: Optional[str] = None
    license_url: Optional[str] = None
    # ISO-639 codes exactly as the upstream repo declares them (empty list if upstream declares none).
    language: List[str] = field(default_factory=list)
    # Free-text appended after the language tags, for cases the tag list alone misrepresents (e.g. a
    # repo whose `language:` key is a placeholder while its card claims many more).
    language_note: Optional[str] = None
    # The waveform rate this model's audio comes out at, written into the card's `infer(...)` call as
    # `sample_rate=`. TTS only.
    #
    # Hand-recorded here for the same reason as `base_repo` and `license_id`: THE CHECKPOINT DOES NOT
    # CARRY IT. Kokoro's `config.json` has `istftnet`'s whole upsampling ladder and never says what rate
    # it lands on; Piper's voice JSON does say (`audio.sample_rate`), and Matcha's is in the training
    # config rather than the `.ckpt`. So this is a value a user is expected to KNOW FROM THE MODEL'S
    # DOCUMENTATION and pass, and putting it in the snippet is what saves the next reader from
    # discovering it by ear -- a rate cannot be recovered from a list of floats afterwards, and getting
    # it wrong does not fail: 24 kHz played at 16 kHz is a slow voice, not an error.
    #
    # `loom-py` prefers what the GGUF itself declares and falls back to this argument, so passing it is
    # correct whether or not the export got round to declaring one. Each value below carries its source.
    sample_rate: Optional[int] = None
    # A short human title and one-line description for the card.
    title: str = ""
    summary: str = ""
    # Markdown rendered as a "Known limitations" section. For anything a user would otherwise discover
    # by getting a wrong answer -- a constraint the export carries that the upstream model does not.
    limitations: Optional[str] = None
    # Markdown appended after the usage code block, for a model whose API needs more than the shared
    # per-task snippet can say. Supertonic is the case it exists for: its GGUF embeds one voice and the
    # repo ships nine more as loose files, so "how do I use a different voice" is a real question that
    # only this model has. Bring your own fenced code block -- this lands as raw markdown, after the
    # snippet's fence has closed.
    usage_extra: Optional[str] = None
    # Extra bullets for the "Files" section, for a repo that ships more than the GGUF. Each string is
    # one bullet, markdown, without the leading "- ".
    extra_files: List[str] = field(default_factory=list)


# The 19 models the exporter can produce today (BACKLOG.md's implementation-sequence table, P4/P5).
# Per-model invocations are [[loom-engine-model-sweep-recipe]]; license/language were read off each
# checkpoint's own upstream README.md (see this repo's `/home/flavio/Dev/models/*/README.md` where one
# was downloaded alongside the weights) or, where the checkpoint itself carries no README, off the
# upstream HF repo directly.
CATALOG = [
    ModelCard(
        slug="qwen3-0.6b-base", checkpoint=Path("qwen3-0.6b-base"),
        task_type="text-generation",
        base_repo="Qwen/Qwen3-0.6B-Base", license_id="apache-2.0",
        language=[], language_note="pre-trained on 119 languages and dialects; upstream publishes no per-language tag list",
        title="Qwen3-0.6B-Base", summary="Qwen3 0.6B base causal LM, exported for loom.cpp.",
    ),
    ModelCard(
        slug="lfm2-350m-monolithic", checkpoint=Path("lfm2-350m"),
        export_task="text-generation", export_model="lfm2-monolithic", task_type="text-generation",
        chat=True,
        base_repo="LiquidAI/LFM2-350M", license_id="other",
        license_name="LFM Open License v1.0", license_url="https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE",
        language=["en", "ar", "zh", "fr", "de", "ja", "ko", "es"],
        title="LFM2-350M (monolithic export)",
        summary="Liquid AI's LFM2-350M hybrid conv/attention LM, exported as a single fused graph.",
    ),
    ModelCard(
        slug="lfm2-350m-modular", checkpoint=Path("lfm2-350m"),
        export_task="text-generation", export_model="lfm2-modular", task_type="text-generation",
        chat=True,
        base_repo="LiquidAI/LFM2-350M", license_id="other",
        license_name="LFM Open License v1.0", license_url="https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE",
        language=["en", "ar", "zh", "fr", "de", "ja", "ko", "es"],
        title="LFM2-350M (modular export)",
        summary="Liquid AI's LFM2-350M hybrid conv/attention LM, exported as per-layer topologies.",
    ),
    ModelCard(
        slug="smollm2-360m-instruct", checkpoint=Path("smollm2-360m-it"),
        task_type="text-generation",
        chat=True,
        base_repo="HuggingFaceTB/SmolLM2-360M-Instruct", license_id="apache-2.0", language=["en"],
        title="SmolLM2-360M-Instruct", summary="HuggingFaceTB's SmolLM2 360M instruct-tuned LM, exported for loom.cpp.",
    ),
    ModelCard(
        slug="gemma-3-270m-it", checkpoint=Path("gemma-3-270m-it"),
        task_type="text-generation",
        chat=True,
        base_repo="google/gemma-3-270m-it", license_id="other",
        license_name="Gemma license", license_url="https://ai.google.dev/gemma/terms",
        language=[], language_note="trained on 140+ languages per the upstream model card; no ISO tag list published",
        title="Gemma 3 270M IT", summary="Google's Gemma 3 270M instruction-tuned LM, exported for loom.cpp.",
    ),
    ModelCard(
        slug="whisper-small", checkpoint=Path("whisper-small"),
        task_type="automatic-speech-recognition",
        # The only windowed ASR export here (`loom.n_samples` = 480000), so the only one whose decode
        # has a prompt for a language token to go in. Verified against the file, not assumed from the
        # 99 language tags below -- parakeet-tdt is multilingual too and takes no language argument.
        selects_language=True,
        base_repo="openai/whisper-small", license_id="apache-2.0",
        language=["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
                  "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
                  "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
                  "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
                  "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                  "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
                  "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"],
        title="Whisper Small", summary="OpenAI's Whisper small encoder-decoder ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="conformer-ctc-small", checkpoint=Path("conformer-ctc-small/stt_en_conformer_ctc_small.nemo"),
        export_task="automatic-speech-recognition", export_model="conformer-ctc",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/stt_en_conformer_ctc_small", license_id="cc-by-4.0", language=["en"],
        title="Conformer-CTC Small (en)", summary="NVIDIA NeMo's small Conformer-CTC English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="parakeet-tdt-0.6b", checkpoint=Path("parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo"),
        export_task="automatic-speech-recognition", export_model="parakeet-tdt",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/parakeet-tdt-0.6b-v3", license_id="cc-by-4.0",
        language=["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
                  "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"],
        title="Parakeet-TDT 0.6B v3", summary="NVIDIA NeMo's multilingual Parakeet-TDT 0.6B ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="parakeet-rnnt-0.6b", checkpoint=Path("parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo"),
        export_task="automatic-speech-recognition", export_model="parakeet-rnnt",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/parakeet-rnnt-0.6b", license_id="cc-by-4.0", language=["en"],
        title="Parakeet-RNNT 0.6B", summary="NVIDIA NeMo's Parakeet-RNNT 0.6B English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="gigaam-v3-rnnt", checkpoint=Path("gigaam-v3"),
        export_task="automatic-speech-recognition", export_model="gigaam-rnnt",
        task_type="automatic-speech-recognition",
        base_repo="ai-sage/GigaAM-v3", license_id="mit", language=["ru", "en"],
        title="GigaAM-v3 RNNT", summary="Sber's GigaAM-v3 Conformer RNNT Russian/English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="qwen3-asr-0.6b", checkpoint=Path("qwen3-asr-0.6b-hf"), venv="ovos",
        task_type="automatic-speech-recognition",
        base_repo="Qwen/Qwen3-ASR-0.6B", license_id="apache-2.0",
        language=["zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi",
                  "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu",
                  "mk", "ro"],
        language_note="exported from the `-hf` mirror of this repo (Transformers-native layout); see "
                       "[[reference-qwen3-asr-hf-checkpoint]]",
        title="Qwen3-ASR-0.6B", summary="Alibaba's Qwen3-ASR 0.6B multilingual ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="granite-speech-4.0-1b", checkpoint=Path("granite-speech-4.0.1b"),
        task_type="automatic-speech-recognition",
        base_repo="ibm-granite/granite-4.0-1b-speech", license_id="apache-2.0",
        language=["en", "fr", "de", "es", "pt", "ja"], language_note="upstream also tags the model \"multilingual\"",
        title="Granite-4.0-1b-speech", summary="IBM's Granite 4.0 1B speech-language model (ASR + AST), exported for loom.cpp.",
    ),
    ModelCard(
        slug="kokoro-82m", checkpoint=Path("kokoro_model"),
        export_task="text-to-speech", export_model="kokoro", task_type="text-to-speech",
        base_repo="hexgrad/Kokoro-82M", license_id="apache-2.0", language=["en"],
        language_note="upstream's own `language:` tag is `en`; the model card additionally documents 8 languages / 54 voices",
        # Documentation only: `config.json` declares `istftnet`'s upsample_rates/gen_istft_hop_size and
        # never the rate they land on. 24 kHz is what upstream `KPipeline` yields and what the model
        # card states, and is the exact value that fixed this model's audio (BACKLOG.md P4.12).
        sample_rate=24000,
        title="Kokoro-82M", summary="hexgrad's Kokoro-82M TTS model, exported for loom.cpp. Takes phoneme ids, not text.",
        usage_extra="""### Choosing a voice

This file embeds one voice (`af_heart`) and uses it whenever no style is passed. All 54 upstream voice
packs ship in this repo under `voices/`, copied unmodified:

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("loom-ai-org/kokoro-82m-loom", filename="voices/af_bella.pt")
pack = torch.load(path, weights_only=True)          # (510, 1, 256) float32

# A Kokoro voice is a PACK, not a vector: one 256-float style per phoneme count, and upstream picks the
# row by the length of the phoneme string (`pack[len(ps)-1]`). That is why the phonemes are produced
# here rather than inside `infer` -- the row and the ids come from the same string.
phonemes = loom.phonemizers.phonemize("hello world")
ref_s = pack[len(phonemes) - 1].flatten().tolist()  # 256 floats

# A specific voice is a knob the high-level door does not name, so it rides through as a driver input.
audio = model.text2speech.infer(tokens=model.tokenize(phonemes), ref_s=ref_s)
audio.save("out.wav")
```

`ref_s` is the two 128-float halves back to back -- decoder style first, predictor style second, which is
the order `KModel.forward` splits it in -- so a row of an upstream pack is already in the right layout
and needs no rearranging. A different voice predicts different durations, so the waveform generally
changes length as well as timbre.

`torch` appears here only to read upstream's `.pt` format; loom-py itself has no runtime dependencies
and takes any sequence of 256 floats, so a voice you have stored some other way works the same way.

Which voice is which is upstream's own `VOICES.md`, copied into this repo: 8 languages, with a quality
and training-duration grade per voice. Note that the phonemes have to match the voice's language, and
the G2P above is the one you installed -- a Japanese voice needs Japanese phonemes to sound like
anything.""",
        extra_files=[
            "`voices/*.pt` -- all 54 upstream voice packs, copied unmodified from\n"
            "  [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M). `af_heart` is also\n"
            "  embedded in the GGUF as the default, so these are only needed to select a different\n"
            "  voice. See the usage example above.",
            "`VOICES.md` -- upstream's own voice table (language, traits, quality grades), copied\n"
            "  unmodified.",
        ],
    ),
    ModelCard(
        slug="matcha-tts-ljspeech", checkpoint=Path("matcha_model/ckpt"),
        export_task="text-to-speech", export_model="matcha", task_type="text-to-speech",
        source_url="https://github.com/shivammehta25/Matcha-TTS", source_name="Matcha-TTS (LJSpeech checkpoint)",
        license_id="mit", language=["en"],
        # Not in `matcha_ljspeech.ckpt` -- its `hyper_parameters` hold the model geometry and the mel
        # statistics, no rate. From the training config the checkpoint was produced by,
        # `Matcha-TTS/configs/data/ljspeech.yaml`: `sample_rate: 22050`.
        sample_rate=22050,
        title="Matcha-TTS (LJSpeech)", summary="Matcha-TTS's LJSpeech flow-matching TTS checkpoint, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="supertonic-2", checkpoint=Path("/home/flavio/Dev/supertonic-tts/assets/pt"),
        export_task="text-to-speech", export_model="supertonic", task_type="text-to-speech",
        takes_text=True,
        base_repo="Supertone/supertonic-2", license_id="other",
        license_name="OpenRAIL-M", license_url="https://huggingface.co/Supertone/supertonic-2/blob/main/LICENSE",
        language=["en", "ko", "es", "pt", "fr"],
        # The one TTS model here whose GGUF already declares its own rate, so this only restates it:
        # `supertonic_export.DEC_SAMPLE_RATE`, the rate its `decoder` emits at.
        sample_rate=44100,
        title="Supertonic 2",
        summary="Supertone's Supertonic 2 on-device TTS model, exported for loom.cpp. Encodes text "
                "itself -- no external phonemiser needed.",
        limitations=
            "**One synthesis call carries at most `model.hparam(\"txt_len\")` ids** -- 512 in this "
            "export, roughly 490 characters once the `<lang>...</lang>` wrap and the inserted final "
            "period are counted, so a short paragraph. Anything shorter is padded and masked by the "
            "driver, so any count up to the ceiling synthesizes correctly; anything longer has to be "
            "split by the caller, and this export deliberately does not do that for you (where a "
            "sentence may be broken is a text-domain decision, not a model contract). The text length "
            "is *fixed* rather than dynamic for two independent reasons -- a single dynamic-length "
            "symbol per graph, and a relative-position windowing step that cannot be traced "
            "dynamically -- so the graphs are traced at several widths and the driver runs the "
            "smallest that fits your text. Short text therefore does not pay for the ceiling.\n\n"
            "**One voice is built in.** `infer` uses it when you pass no style, and takes any other "
            "voice as a `style_ttl`/`style_dp` pair. What this export does *not* carry is the two "
            "style encoders, so it cannot derive a style from your own audio -- cloning a new voice "
            "needs the upstream checkpoint. Selecting among existing voices does not.",
        usage_extra="""### Choosing a voice

This file embeds one voice (`F1`) and uses it whenever no style is passed. Nine more ship in this repo
under `voice_styles/`:

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("loom-ai-org/supertonic-2-loom", filename="voice_styles/M1.json")
style = json.load(open(path))

# Each file holds two embeddings, stored with a leading batch axis: style_ttl is (1, 50, 256) and
# style_dp is (1, 8, 16). `infer` takes them flat, so drop the batch axis and concatenate the rows.
flatten = lambda entry: [v for row in entry["data"][0] for v in row]
style_ttl = flatten(style["style_ttl"])   # 50 * 256 = 12800 floats
style_dp = flatten(style["style_dp"])     #  8 *  16 =   128 floats

# A specific voice is a knob the high-level door does not name, so this goes through `infer`.
txt_ids = model.tokenize("hello world")
audio = model.infer(txt_ids=txt_ids, style_ttl=style_ttl, style_dp=style_dp, n_steps=4, seed=1234)
```

The two arguments travel together: pass neither for the built-in voice, or both to select another. A
different voice predicts a different duration, so the waveform generally changes length as well as
timbre.

Plain lists are fine -- this package has no runtime dependencies and accepts any sequence of floats, so
`numpy.asarray(...).ravel()` works equally well if numpy is already around.""",
        extra_files=[
            "`voice_styles/*.json` -- ten precomputed voices (`F1`-`F5`, `M1`-`M5`), copied unmodified\n"
            "  from the upstream checkpoint. `F1` is also embedded in the GGUF as the default, so these\n"
            "  are only needed to select a different voice. See the usage example above.",
        ],
    ),
    ModelCard(
        slug="vits-piper-en-gb-miro",
        checkpoint=Path("/home/flavio/Dev/piper/pipertts_en-GB_miro/epoch=9772-step=1494014.ckpt"),
        export_task="text-to-speech", export_model="vits", task_type="text-to-speech",
        base_repo="OpenVoiceOS/pipertts_en-GB_miro", license_id="other",
        license_name=None, license_url=None,
        language=["en"],
        language_note="upstream declares no `license:` tag; check https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro "
                       "directly before redistributing",
        # The only one of the five that IS on disk, though not in the `.ckpt` this exports: the Piper
        # voice's `miro_en-GB.onnx.json` declares `audio.sample_rate: 22050`. Per-VOICE rather than
        # per-architecture -- another Piper voice may differ, so this belongs to this catalogue entry.
        sample_rate=22050,
        title="Piper VITS en-GB (miro)", summary="OpenVoiceOS's Piper-compatible VITS en-GB \"miro\" voice, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="styletts2-ljspeech",
        checkpoint=Path("styletts2_model/ckpt/Models/LJSpeech/epoch_2nd_00100.pth"),
        export_task="text-to-speech", export_model="styletts2", task_type="text-to-speech",
        base_repo="yl4579/StyleTTS2-LJSpeech", license_id="mit",
        language_note="the HF repo carries no `license:`/`language:` tags; MIT per the upstream "
                       "GitHub repo's LICENSE (github.com/yl4579/StyleTTS2)",
        language=["en"],
        # From the checkpoint's own `config.yml`: `preprocess_params.sr: 24000`. NOT the `slm.sr: 16000`
        # a few lines above it in the same file -- that is the speech language model discriminator's
        # input rate, used in training only, and it is the wrong number to reach for here.
        sample_rate=24000,
        title="StyleTTS2 (LJSpeech)", summary="yl4579's StyleTTS2 LJSpeech checkpoint, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="dac-44khz", checkpoint=Path("dac-44khz"),
        task_type="text-to-audio",
        base_repo="descript/dac_44khz", license_id="mit", language=[],
        # The HF repo publishes NO `license:` tag and its README is an unfilled template, so the tag
        # here comes from the upstream project -- github.com/descriptinc/descript-audio-codec, whose
        # LICENSE is MIT and whose README says so of the WEIGHTS specifically ("Weights are released
        # as part of this repo under MIT license"), which is the claim that matters for a re-upload.
        # Same shape of gap, and same resolution, as StyleTTS2's entry above.
        language_note="a codec, not a language model: it carries no vocabulary and no language. "
                       "The upstream HF repo carries no `license:` tag; MIT is from the project's own "
                       "LICENSE and its README's explicit statement about the weights.",
        title="DAC 44.1 kHz (decoder)",
        summary="Descript Audio Codec at 44.1 kHz, decode half, exported for loom.cpp. Family 11: "
                "codec tokens in, a waveform out.",
        limitations=(
            "**This is the DECODE half only.** `encode` is audio-in/codes-out -- a different contract "
            "with a different modality pair -- and no model that decodes through this codec ever calls "
            "it, so exporting it would be weight in the file for a door nothing opens. To go the other "
            "way, use the upstream checkpoint.\n\n"
            "It takes **9 code streams per frame at 86.13 frames per second**, and one frame decodes "
            "to 512 samples. Codes from a different codec -- or from DAC at a different sample rate -- "
            "are integers in the right range and produce noise rather than an error.\n\n"
            "**It does not undo a delay pattern.** An AR model that emits these codes typically offsets "
            "stream *k* by *k* steps; realigning them is a property of that model, not of the codec, so "
            "feed it aligned codes."
        ),
    ),
    ModelCard(
        slug="dia-1.6b", checkpoint=Path("dia-1.6b"),
        task_type="text-to-speech",
        base_repo="nari-labs/Dia-1.6B", license_id="apache-2.0", language=["en"],
        title="Dia-1.6B",
        summary="Nari Labs' Dia-1.6B dialogue TTS model, exported for loom.cpp. Family 10: text in, "
                "neural-codec tokens out -- pair it with `dac-44khz-loom` for audio.",
        limitations=(
            "**This model does not produce audio.** It emits nine streams of DAC codec tokens, and a "
            "codec turns those into a waveform -- [`dac-44khz-loom`](https://huggingface.co/loom-ai-org/dac-44khz-loom), "
            "which is the codec this checkpoint was trained against. They stay separate because one "
            "codec serves many models like this one, and because the codes are the useful "
            "intermediate. The usage snippet above is the whole of the joining.\n\n"
            "**It samples, and it is high-variance.** The export declares this checkpoint's own "
            "decoding -- `temperature 1.8`, `top_k 50`, `top_p 0.9`, and classifier-free guidance at "
            "`3.0` -- so two runs of the same sentence give two different takes. Some of them are not "
            "the sentence: on the snippet's own text, one seed in four gave it back verbatim, one "
            "gave laughter and two gave near-silence. That is the model rather than the export "
            "(`transformers` behaves identically), which is why the snippet names a seed. **Expect "
            "to try several.**\n\n"
            "Guidance costs a second decoder pass at every step, so generation is about twice the "
            "work of a comparable LM. Pass `guidance_scale=1.0` to turn it off -- faster, and worse.\n\n"
            "`max_new_tokens` counts **audio frames** at 86.13 per second, not decoder steps: the two "
            "differ by this family's delay pattern, which the driver applies and undoes for you. "
            "Reaching the cap forces a clean ending rather than truncating, so a budget that is too "
            "small gives you a complete, shorter utterance.\n\n"
            "**Speaker tags are part of the text.** `[S1]` and `[S2]` are tokens this checkpoint was "
            "trained on and are what make it a dialogue model; text without one is out of "
            "distribution. Non-verbal cues like `(laughs)` work the same way. Voices are not "
            "selectable -- without an audio prompt the model picks one, and the seed is what decides "
            "it.\n\n"
            "**It is a big download**: 6.4 GB, F32, like every other model in this collection. These "
            "are reference exports as much as they are downloads, and one lossy artifact among "
            "seventeen faithful ones is a difference nothing in the file would tell you about. "
            "`loom-export --quantize Q8_0` on the upstream checkpoint packs the eligible weights to "
            "about 1.8 GB if you would rather have that -- it moves the logits slightly, which for a "
            "sampler means a different take rather than a worse one."
        ),
    ),
    ModelCard(
        slug="distilbert-ner", checkpoint=Path("distilbert-ner"),
        task_type="token-classification",
        base_repo="dslim/distilbert-NER", license_id="apache-2.0", language=["en"],
        title="DistilBERT-NER",
        summary="dslim's DistilBERT fine-tuned on CoNLL-2003 for named-entity recognition, exported "
                "for loom.cpp. Family 12: text in, one class per token out.",
        limitations=(
            "Trained on CoNLL-2003, which is **newswire from 1996-1997** -- entity names that did not "
            "exist then, and text that does not read like a news wire, are outside what it saw. It "
            "recognises four types (`PER`, `ORG`, `LOC`, `MISC`) and nothing else, and it is **cased**: "
            "lowercasing your input before passing it costs real accuracy, because capitalisation is "
            "most of what a NER model has to go on.\n\n"
            "The labels line up with the tokenizer's PIECES, not with your words. A word the "
            "vocabulary splits gets one label per piece, and they can disagree -- deciding which one "
            "wins is the caller's rule, which is why the export hands back the pieces alongside the "
            "labels rather than a list of words.\n\n"
            "The export takes one sequence at a time and no padding, so there is no batch dimension "
            "to fill and no attention mask to pass. Sequences are capped at 512 tokens by the "
            "checkpoint's own learned position table."
        ),
    ),
    ModelCard(
        slug="fullstop-punc", checkpoint=Path("fullstop-punc"),
        task_type="token-classification", restores_punctuation=True,
        base_repo="oliverguhr/fullstop-punctuation-multilang-large", license_id="mit",
        language=["en", "de", "fr", "it"],
        title="FullStop Punctuation (multilingual)",
        summary="oliverguhr's XLM-RoBERTa-large fine-tuned on Europarl for punctuation restoration, "
                "exported for loom.cpp. Family 12: text in, one class per token out -- here the class "
                "is the mark that follows the token.",
        limitations=(
            "Trained on **Europarl**, which is parliamentary proceedings: formal, complete sentences "
            "in a register that is not chat, not code and not casual speech. It restores six classes "
            "and nothing else (`.`, `,`, `?`, `-`, `:`, and `0` for no mark), so it will not give you "
            "semicolons, quotation marks or apostrophes, and it does not capitalise -- truecasing is a "
            "different head. The upstream card evaluates **en, de, fr, it**; the underlying encoder is "
            "trained on 100 languages and the model will answer for all of them, at an accuracy nobody "
            "has measured.\n\n"
            "Feed it text with the punctuation already **removed**. Given punctuated input it still "
            "labels every token and you get marks on top of marks.\n\n"
            "The labels line up with the tokenizer's PIECES, not with your words -- a SentencePiece "
            "vocabulary splits `wolfgang` into three -- and the mark you want is the one on a word's "
            "LAST piece. The usage snippet above does that walk; the export hands back the pieces "
            "alongside the labels rather than guessing at the rule for you.\n\n"
            "The export takes one sequence at a time and no padding, so there is no batch dimension to "
            "fill and no attention mask to pass. Sequences are capped at **512** tokens: this "
            "checkpoint's position table has 514 rows and its first two are reserved, which is a "
            "property of the RoBERTa family rather than an off-by-two."
        ),
    ),
]

CATALOG_BY_SLUG = {m.slug: m for m in CATALOG}


# Appears under the install block of every phoneme-input TTS card. Written to be read by someone who
# has just installed the extra and is about to be disappointed by English: the bundled G2P is
# rule-based, and English is one of the deep-orthography languages that cannot be reached that way.
PHONEMIZER_NOTE = """
**NOTE:** This is a work in progress. For now, in order to avoid license conflicts and keep
dependencies at a minimum, we opted for orthography2ipa as our "swiss-knife" phonemizer. For
deep-orthography languages like English, to get stressing rules and context-based phonemization, the
phonemizer must register a reference lexicon (e.g., ipa-dict) to get highly accurate phonemization.
Full quality can be achieved by phonemizing the text yourself using your engine of choice and feeding
it to the model via the argument `phonemes` as in the example below.
"""

USAGE_SNIPPETS = {
    # The HIGH-LEVEL door for each task, because that is what a reader arriving from a model page
    # wants: one call, end to end, with the windowing/sampling/assembly a model needs already applied.
    # The low-level API is one line at the bottom of every card pointing at the repos, since which
    # inputs a driver takes is a property of the model rather than of a card template.
    "text-generation": """import loom

model = loom.Model.from_pretrained("{repo_id}")
print(model.text2text.infer("The capital of France is", max_new_tokens=14))
""",
    # The instruction-tuned door. `chat` applies the checkpoint's OWN chat template -- carried in the
    # GGUF as role tags and assembled by the engine -- so the model is asked a question inside a turn
    # rather than handed a prefix to continue. Decoding follows the checkpoint's own
    # `generation_config.json`; `temperature=0.0` is how you ask for greedy instead.
    "text-generation-chat": """import loom

model = loom.Model.from_pretrained("{repo_id}")
print(model.text2text.chat("Who discovered Brazil?", max_new_tokens=256))

# The same model, one turn at a time, and with the decode rule named rather than inherited:
print(model.chat([("user", "Name one river in Brazil.")], temperature=0.0))
""",
    # Two ASR snippets, for the same reason there are two TTS ones: `language=` is not a property of
    # the task. `selects_language` below is what picks between them.
    "automatic-speech-recognition": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# Audio is a mono float list at 16 kHz. This model decodes in the one language it was trained for and
# takes no `language=` argument -- passing one warns and is ignored, because nothing in its decode
# could act on it.
result = model.speech2text.infer(audio, timestamps=True)
print(result.text)

# It emits no timestamp tokens, so `segments` is one span covering the whole clip and
# `result.timestamped` is False. Check that before treating a start/end as a boundary the model chose.
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
""",
    "automatic-speech-recognition-multilingual": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# Audio is a mono float list at 16 kHz. Long files are windowed for you, and a model that emits
# timestamps is seeked to where it closed its last segment rather than cut at a fixed stride.
# Omit `language=` to let the model detect it; `task="translate"` renders the speech in English.
result = model.speech2text.infer(audio, language="en", timestamps=True)
print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
""",
    # Two TTS snippets, because "TTS" is not one answer. Which one a model gets is `takes_text` below,
    # a per-model fact read off the export rather than assumed from the task -- the single phoneme-ids
    # snippet this used to have was simply wrong for Supertonic, whose GGUF carries its own vocabulary.
    "text-to-speech": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# {slug} is trained on phonemes. Its symbol table ships in the GGUF, so the only piece that is not in
# the file is grapheme-to-phoneme -- a property of the language rather than of this checkpoint, which
# is why it is the `phonemes` extra above rather than part of the model.

# THE FULL-QUALITY PATH: phonemes you produced yourself, with whatever G2P you trust. The symbol table
# in the GGUF is what encodes them, so anything that emits IPA works.
audio = model.text2speech.infer(phonemes="h\u0259\u02c8lo\u028a w\u02c8\u025c\u02d0ld", sample_rate={sample_rate})
audio.save("out.wav")

# THE BUILT-IN PATH: text straight in, phonemized by the bundled rule-based G2P. Good enough for
# shallow orthographies; for English see the note above, and give it a lexicon so it has stress and
# real vowels to work with -- "time" is /t\u026am/ without one.
#
# open-dict-data/ipa-dict (MIT) publishes ~65k-entry wordlists WITH stress for en_UK and en_US, in
# almost the right shape: its IPA is wrapped in slashes and a rare entry carries two comma-separated
# variants, both of which the loader rejects. Take the RAW file: a github.com/.../blob/... URL serves
# an HTML PAGE, and the sed below will turn that into a .tsv that parses to zero entries -- which is
# indistinguishable from no lexicon at all except for the warning `set_lexicon` raises. Two lines:
#     curl -LO https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt
#     sed 's:/::g; s/\\t\\([^,]*\\),.*/\\t\\1/' en_UK.txt > en_UK.tsv
loom.phonemizers.set_lexicon("en_UK.tsv")    # a path, an http(s):// URL, or hf://<repo>/<path>

# sample_rate={sample_rate}: this checkpoint does not carry its own rate, so it is a value you have to
# know from the model's documentation and pass. It is used only if the GGUF declares none; a wrong rate
# does not fail, it plays the voice at the wrong speed.
audio = model.text2speech.infer("hello world", sample_rate={sample_rate})
audio.save("out.wav")
""",
    # The fourth door, and the first non-audio one. Fixed text rather than a placeholder, for the same
    # reason every TTS card says "hello world": the release gate reads the card's OWN output back and
    # grades it, so the sentence has to be one an expectation can be written against.
    "token-classification": """import loom

model = loom.Model.from_pretrained("{repo_id}")

result = model.text2class.infer("My name is Wolfgang and I live in Berlin")
print(result.text)
# My/O name/O is/O Wolfgang/B-PER and/O I/O live/O in/O Berlin/B-LOC

# The labels come back beside the PIECES the tokenizer produced, not the words you wrote -- a
# WordPiece encode splits unknown and long words, so "Wolfgang" may be one piece here and three in
# another sentence. Joining them back into words is a rule only you can make, which is why both
# halves are handed over rather than one.
for token in result:
    print(token.piece, token.label)

# Every class this checkpoint can choose between, in the order its ids run:
print(result.labels)

# The framing tokens the encode adds ([CLS] and [SEP]) are dropped for you, on the ids the file
# declares rather than on their spelling. Ask for the raw alignment if you want them:
raw = model.text2class.infer("My name is Wolfgang and I live in Berlin", strip_special=False)
print(len(raw), "rows including [CLS] and [SEP], against", len(result), "without")
""",
    # The same door on a checkpoint whose classes are punctuation MARKS rather than entity TAGS, and
    # the first card in this set whose model is not English-only or WordPiece. Its sentence is fixed for
    # the reason every other one is -- the release gate reads the card's own output back and grades it --
    # and it is deliberately the same sentence the NER card labels, so the two cards demonstrate two
    # readings of one identical call.
    "token-classification-punctuation": """import loom

model = loom.Model.from_pretrained("{repo_id}")

result = model.text2class.infer("hello my name is wolfgang and i live in berlin do you know it")
print(result.text)
# hell/, o/, my/0 name/0 is/0 ... berlin/. do/0 you/0 know/0 it/?

# Every class this checkpoint can choose between, in the order its ids run. "0" is "no mark here",
# which is most tokens in most sentences:
print(result.labels)
# ['0', '.', ',', '?', '-', ':']

# Restoring the text is the point of this model, and the rule is one line long: a mark belongs to the
# WORD, so it is the label on the word's LAST piece. A piece starts a new word when decoding it
# together with the piece before puts a space between them -- which is what the vocabulary knows and
# the piece text alone does not.
restored = ""
for i, token in enumerate(result):
    following = result[i + 1] if i + 1 < len(result) else None
    restored += token.piece
    if following is None or " " in model.detokenize([token.token, following.token]):
        if token.label != "0":
            restored += token.label
        restored += " "
print(restored.strip())
# hello, my name is wolfgang and i live in berlin. do you know it?

# The framing tokens the encode adds (<s> and </s>) are dropped for you, on the ids the file declares
# rather than on their spelling. Ask for the raw alignment if you want them:
raw = model.text2class.infer("hello my name is wolfgang and i live in berlin do you know it",
                             strip_special=False)
print(len(raw), "rows including <s> and </s>, against", len(result), "without")
""",
    # A codec DECODER, which is the one card here whose input a reader cannot type. Codes come from
    # an encoder or from an AR codec-token LM, so the snippet demonstrates the GEOMETRY -- how many
    # streams, how wide, how many frames to a second -- and decodes a run of them. The numbers it
    # prints are what the release gate grades, and they are the exact thing that was silently wrong in
    # the first working export of this family: a decoder that returns one frame's worth of audio for
    # any input produces a plausible file and the wrong length.
    "audio-codec": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# The geometry a caller needs, declared by the file rather than looked up in a paper:
n_codebooks = model.hparam("codec.n_codebooks")       # code streams per frame
codebook_size = model.hparam("codec.codebook_size")   # valid id range per stream
frame_rate = model.hparam("codec.frame_rate", "f32")  # codes per second
print(n_codebooks, codebook_size, frame_rate, model.contract["sample_rate"])

# Codes are FRAME-MAJOR: all `n_codebooks` codes for frame 0, then frame 1, and so on. This file
# is the DECODE half -- real codes come from the matching encoder, or an AR model that emits them.
frames = round(frame_rate)                            # one second of audio
codes = [[0] * n_codebooks for _ in range(frames)]
audio = model.codes2speech.infer(codes)
print(len(audio), "samples at", audio.sample_rate, "Hz =", round(audio.duration, 3), "s")
audio.save("out.wav")

# A flat list works too, and is what a driver that emitted the codes hands over. One that is not a
# whole number of frames is refused rather than reinterpreted at a different width.
audio = model.codes2speech.infer([0] * (frames * n_codebooks))
""",
    # An AR codec-token LM, which is the one card here whose model does not produce its own output
    # kind: it emits codec TOKENS and a second file turns them into audio (loom.cpp ADR-022). So the
    # snippet is the only one that loads two models, and the chaining -- two calls and the array
    # between them -- is the thing it exists to show.
    #
    # **The seed is in the snippet, and it is not decoration.** This checkpoint declares
    # `do_sample: true` at temperature 1.8 with classifier-free guidance, which is genuinely
    # high-variance: of four seeds tried on this sentence, one gave it back verbatim, one gave
    # laughter and two gave near-silence, and `transformers` behaves the same way (loom.cpp
    # Retro-032). A card that drew an unseeded sample would look like a broken model to whoever ran
    # it next.
    "text-to-codes": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# What comes back is codec TOKENS, not audio -- frame-major, one row per frame, `n_codebooks` wide.
# `max_new_tokens` counts AUDIO FRAMES rather than decoder steps.
codes = model.text2codes.infer(
    "[S1] Hey, can you shut down the computer, my friend?",
    max_new_tokens=260, seed=1234,
)
print(len(codes), "frames x", len(codes[0]), "codebooks")

# The second half of the pair, in a repo of its own: one codec serves many models like this one, and
# the codes are worth having on their own -- cache them, edit them, decode them somewhere else.
codec = loom.Model.from_pretrained("loom-ai-org/dac-44khz-loom")
audio = codec.codes2speech.infer(codes)
print(len(audio), "samples at", audio.sample_rate, "Hz =", round(audio.duration, 2), "s")
audio.save("out.wav")

# Nothing goes between those two calls. Both files declare the width of a frame, so a pair that does
# not fit says so instead of producing audio of the wrong duration:
print(model.hparam("codec.n_codebooks"), "==", codec.hparam("codec.n_codebooks"))

# This model SAMPLES by default, at the settings its own generation config declares. `seed=` above is
# what makes a result reproducible; drop it for a different take, or decode greedily for the same
# answer every time -- greedy is much flatter, and not what this checkpoint was tuned for.
print(model.hparam("sampling.temperature", "f32"), model.hparam("sampling.guidance_scale", "f32"))
""",
    "text-to-speech-with-vocab": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# This model encodes text itself -- no phonemiser needed at all.
print(model.tokenizer)                       # kind, vocabulary size, default language

# sample_rate={sample_rate}: a rate is not something a checkpoint necessarily carries, so it is a value
# you have to know from the model's documentation and pass. It is used only if the GGUF declares none;
# a wrong rate does not fail, it plays the voice at the wrong speed.
audio = model.text2speech.infer("hello world", sample_rate={sample_rate})
audio.save("out.wav")

# That uses whatever voice the file itself defaults to. See below for choosing another.
""",
}


#: The only placeholders a usage snippet may carry. Substituted by name rather than through
#: `str.format`, which was the previous mechanism and is wrong for this content: a snippet is PYTHON,
#: and Python has braces. The first card to write `{n_codebooks}` inside an explanatory comment
#: crashed the build with `KeyError: 'n_codebooks'`, and the first one to show a dict or a set literal
#: would have done the same. Targeted replacement cannot: an unknown brace is just text.
SNIPPET_PLACEHOLDERS = ("repo_id", "slug", "sample_rate")


def render_snippet(text: str, **values) -> str:
    """One usage snippet with its placeholders filled in.

    Raises on a placeholder this table does not know rather than leaving it in the published card:
    a card that shipped a literal `{voice}` would be telling a reader to type it.
    """
    unknown = set(values) - set(SNIPPET_PLACEHOLDERS)
    if unknown:
        raise ValueError(f"unknown snippet placeholder(s) {sorted(unknown)}; "
                         f"add them to SNIPPET_PLACEHOLDERS if they are real")
    for key in SNIPPET_PLACEHOLDERS:
        text = text.replace("{" + key + "}", str(values.get(key, "")))
    return text


def repo_id(card: ModelCard) -> str:
    return f"loom-ai-org/{card.slug}-loom"


def snippet_key(card: ModelCard) -> str:
    """Which `USAGE_SNIPPETS` entry this model's card gets. The task decides it for every family except
    TTS, where whether the GGUF carries a vocabulary is a per-model fact -- see `takes_text`."""
    if card.task_type == "text-generation" and card.chat:
        return "text-generation-chat"
    if card.task_type == "text-to-speech" and card.takes_text:
        return "text-to-speech-with-vocab"
    if card.task_type == "automatic-speech-recognition" and card.selects_language:
        return "automatic-speech-recognition-multilingual"
    if card.task_type == "token-classification" and card.restores_punctuation:
        return "token-classification-punctuation"
    return card.task_type


def resolve_checkpoint(card: ModelCard, models_root: Path) -> Path:
    return card.checkpoint if card.checkpoint.is_absolute() else models_root / card.checkpoint


def export_args(card: ModelCard) -> List[str]:
    args = []
    if card.export_task:
        args += ["--task", card.export_task]
    if card.export_model:
        args += ["--model", card.export_model]
    return args


def render_readme(card: ModelCard, gguf_name: str) -> str:
    # A TTS snippet interpolates `sample_rate`, so an entry that never set one would render the literal
    # `sample_rate=None` into a card telling readers to pass it -- a wrong example is worse than no
    # example, and this is the only field where the value cannot be checked against anything on disk.
    if card.task_type == "text-to-speech" and not card.sample_rate:
        raise ValueError(
            f"{card.slug}: no sample_rate. Every TTS card's usage snippet states one, because a rate is "
            f"not something these checkpoints carry -- look it up in the model's documentation and "
            f"record it on the catalogue entry, with where you got it."
        )

    lang_lines = "".join(f"- {code}\n" for code in card.language)
    frontmatter = ["---", f"license: {card.license_id}"]
    if card.language:
        frontmatter += ["language:", lang_lines.rstrip("\n")]
    if card.base_repo:
        frontmatter += [f"base_model:", f"- {card.base_repo}"]
    frontmatter += [f"pipeline_tag: {card.task_type}", "library_name: loom-py-rt", "---", ""]

    if card.base_repo:
        source_line = f"[`{card.base_repo}`](https://huggingface.co/{card.base_repo})"
    else:
        source_line = f"[{card.source_name}]({card.source_url})"

    if card.license_id == "other":
        if card.license_name and card.license_url:
            license_line = f"[{card.license_name}]({card.license_url}) -- inherited from the base model above."
        elif card.license_name:
            license_line = f"{card.license_name} -- inherited from the base model above."
        else:
            license_line = "Inherited from the base model above; see its repo for terms."
    else:
        license_line = f"`{card.license_id}`, inherited from the base model above."

    if card.language:
        lang_line = ", ".join(f"`{c}`" for c in card.language)
    else:
        lang_line = "(none tagged upstream)"
    if card.language_note:
        lang_line += f"\n\n{card.language_note}"

    limitations_section = f"\n## Known limitations\n\n{card.limitations}\n" if card.limitations else ""
    # Both land verbatim: `usage_extra` right after the snippet's closing fence, `extra_files` as
    # further bullets under the GGUF's own.
    usage_extra_section = f"\n{card.usage_extra}\n" if card.usage_extra else ""

    # Phoneme-input TTS needs one more extra than everything else, and needs saying WHY. The phonemizer
    # is optional (`loom-py-rt` declares no runtime dependencies on purpose), so a card that installs
    # only `[hub]` and then calls the text door sends the reader to a LookupError; and a card that
    # installs it without qualification implies the result is production-quality English, which
    # rule-based transduction is not. `takes_text` is the discriminator, same as for the snippet:
    # Supertonic encodes graphemes itself and needs none of this.
    needs_phonemes = card.task_type == "text-to-speech" and not card.takes_text
    install_extras = "hub,phonemes" if needs_phonemes else "hub"
    phonemizer_note = PHONEMIZER_NOTE if needs_phonemes else ""
    extra_files_section = "".join(f"- {bullet}\n" for bullet in card.extra_files)
    # Named in the Files list rather than left to be discovered from the byte count: a quantized
    # export is not the same artifact as the F32 one, and a reader comparing this against their own
    # export needs to know which they are looking at.

    body = f"""# {card.title}

{card.summary}

This is a [loom.cpp](https://github.com/loom-ai-org/loom.cpp) export: a single self-describing GGUF
that carries its own graph topologies, tokenizer (if any) and driver script, produced by
[loom-exporter]({EXPORTER_URL}).

## Original model

Exported from {source_line}. Weights are unmodified; this repo packages the same parameters into
loom.cpp's GGUF format.

## License

{license_line}

## Language(s)

{lang_line}

## Usage

Run it with [loom-py]({LOOM_PY_URL}) -- `loom-py-rt` on PyPI:

```sh
pip install -U "loom-py-rt[{install_extras}]"
```
{phonemizer_note}
```python
{render_snippet(USAGE_SNIPPETS[snippet_key(card)], repo_id=repo_id(card), slug=card.slug, sample_rate=card.sample_rate)}```
{usage_extra_section}
### The layer underneath

The call above is the high-level door: one per task, named for the modality pair it maps between, with
the windowing, sampling and assembly this model needs already applied. Under it, `model.infer(...)`
passes your arguments straight to the driver this GGUF embeds -- which is where you go for a knob the
door does not name.

`model.driver_source` prints that driver, including a header comment documenting every argument it
accepts for this model, and is the authority on it. See [loom-py]({LOOM_PY_URL}) for the API and
[loom.cpp]({LOOM_CPP_URL}) for what the engine does between the two.
{limitations_section}
## Files

- `{gguf_name}` -- the model, exported with loom-exporter.
{extra_files_section}"""
    return "\n".join(frontmatter) + body


def do_export(card: ModelCard, checkpoint: Path, out_gguf: Path) -> None:
    from loom_exporter.main_export import main_export

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    main_export(str(checkpoint), str(out_gguf), task=card.export_task, model=card.export_model)


def build_one(card: ModelCard, models_root: Path, output_dir: Path, readme_only: bool) -> None:
    model_dir = output_dir / card.slug
    gguf_name = f"{card.slug}.gguf"
    gguf_path = model_dir / gguf_name

    if readme_only:
        if not gguf_path.exists():
            print(f"  [skip] {card.slug}: --readme-only but {gguf_path} does not exist")
            return
    else:
        checkpoint = resolve_checkpoint(card, models_root)
        if not checkpoint.exists():
            print(f"  [skip] {card.slug}: checkpoint not found at {checkpoint}")
            return
        print(f"  [export] {card.slug}  ({checkpoint} -> {gguf_path})")
        do_export(card, checkpoint, gguf_path)

    (model_dir / "README.md").write_text(render_readme(card, gguf_name))
    print(f"  [ok] {card.slug}: {model_dir}")


def running_venv() -> str:
    """Best-effort label for which of the two export venvs is running this interpreter -- just the
    trailing path component of sys.prefix, which is `piper` or `ovos` for the venvs this repo uses."""
    return Path(sys.prefix).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slugs", nargs="*", help="Model slugs to build (see --list). Default: none.")
    parser.add_argument("--all", action="store_true", help="Build every catalog entry this venv can export")
    parser.add_argument("--list", action="store_true", help="Print the catalog and exit")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT,
                         help=f"Where checkpoints live (default {DEFAULT_MODELS_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help=f"Where to write <slug>/ dirs (default {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--readme-only", action="store_true",
                         help="Regenerate README.md only, for slugs whose GGUF already exists")
    parser.add_argument("--force-venv", action="store_true",
                         help="Build models tagged for the other venv anyway (will ImportError if it can't load)")
    args = parser.parse_args()

    if args.list:
        for card in CATALOG:
            print(f"{card.slug:28s} [{card.task_type:28s} venv={card.venv:5s}] {repo_id(card)}")
        return

    venv = running_venv()
    if args.all:
        selected = [c for c in CATALOG if args.force_venv or c.venv == venv]
        skipped = [c.slug for c in CATALOG if c not in selected]
        if skipped:
            print(f"[build_model_cards] this interpreter is '{venv}'; skipping (need a different venv): {skipped}")
    else:
        unknown = [s for s in args.slugs if s not in CATALOG_BY_SLUG]
        if unknown:
            parser.error(f"unknown slug(s): {unknown}; see --list")
        selected = [CATALOG_BY_SLUG[s] for s in args.slugs]
        if not selected:
            parser.error("give one or more slugs, or --all / --list")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for card in selected:
        if not args.force_venv and card.venv != venv and not args.readme_only:
            print(f"  [skip] {card.slug}: needs the '{card.venv}' venv, this interpreter is '{venv}'")
            continue
        build_one(card, args.models_root, args.output_dir, args.readme_only)


if __name__ == "__main__":
    main()
