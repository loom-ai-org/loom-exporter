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
    # "text-generation" | "automatic-speech-recognition" | "text-to-speech" -- picks the usage snippet.
    task_type: str
    # Whether this model's GGUF carries a vocabulary, for the TTS models where that is not implied by
    # the task: the phoneme-input families (Kokoro, Matcha, VITS, StyleTTS2) take ids a phonemiser
    # produces outside the engine, while a grapheme model (Supertonic) encodes text itself. Only read
    # for task_type == "text-to-speech"; the LM and ASR families all carry one.
    takes_text: bool = False
    # Output sample rate in Hz, required for task_type == "text-to-speech" and unused elsewhere. It is a
    # per-card constant rather than something read off the GGUF because no TTS export declares it as an
    # hparam -- the number reaches the engine as a driver constant at most (Supertonic's SAMPLE_RATE),
    # and a waveform whose rate the caller has to guess is a waveform played at the wrong speed. Each
    # card's value is sourced in a comment beside it.
    sample_rate: Optional[int] = None
    # INPUT sample rate in Hz for task_type == "automatic-speech-recognition", and unused elsewhere --
    # the mirror of `sample_rate` above, which is an OUTPUT rate for TTS. Same reasoning applies in
    # reverse: audio resampled to the wrong rate is not rejected, it is transcribed badly, so a caller
    # who has to guess this gets a worse transcript and no error.
    #
    # A per-card constant rather than something read off the GGUF, because only ONE export declares it:
    # `whisper_mil.gguf` carries `loom.sample_rate = 16000`, and the other five ASR GGUFs carry no
    # sample-rate key at all. Rather than have the cards be sometimes-derived and sometimes-hardcoded,
    # every value here is sourced from the upstream checkpoint in a comment, exactly as the TTS rates
    # above are. If the exporter ever emits this for every ASR family, this field is what to delete.
    input_sample_rate: Optional[int] = None
    # Audio length the export FIXES, in samples, or None when the model accepts any length. Only
    # Whisper has one: its topology declares `waveform` as `["480000", "1"]` -- a literal, not the
    # symbolic `["n_samples", "1"]` every other ASR export uses -- so 480000 samples at 16 kHz is
    # exactly 30 seconds and nothing else will load. Shorter audio must be padded and longer audio
    # chunked, which is a thing a caller discovers by crashing unless the card says so.
    fixed_audio_samples: Optional[int] = None
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


# The 17 models the exporter can produce today (BACKLOG.md's implementation-sequence table, P4/P5).
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
        base_repo="LiquidAI/LFM2-350M", license_id="other",
        license_name="LFM Open License v1.0", license_url="https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE",
        language=["en", "ar", "zh", "fr", "de", "ja", "ko", "es"],
        title="LFM2-350M (monolithic export)",
        summary="Liquid AI's LFM2-350M hybrid conv/attention LM, exported as a single fused graph.",
    ),
    ModelCard(
        slug="lfm2-350m-modular", checkpoint=Path("lfm2-350m"),
        export_task="text-generation", export_model="lfm2-modular", task_type="text-generation",
        base_repo="LiquidAI/LFM2-350M", license_id="other",
        license_name="LFM Open License v1.0", license_url="https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE",
        language=["en", "ar", "zh", "fr", "de", "ja", "ko", "es"],
        title="LFM2-350M (modular export)",
        summary="Liquid AI's LFM2-350M hybrid conv/attention LM, exported as per-layer topologies.",
    ),
    ModelCard(
        slug="smollm2-360m-instruct", checkpoint=Path("smollm2-360m-it"),
        task_type="text-generation",
        base_repo="HuggingFaceTB/SmolLM2-360M-Instruct", license_id="apache-2.0", language=["en"],
        title="SmolLM2-360M-Instruct", summary="HuggingFaceTB's SmolLM2 360M instruct-tuned LM, exported for loom.cpp.",
    ),
    ModelCard(
        slug="gemma-3-270m-it", checkpoint=Path("gemma-3-270m-it"),
        task_type="text-generation",
        base_repo="google/gemma-3-270m-it", license_id="other",
        license_name="Gemma license", license_url="https://ai.google.dev/gemma/terms",
        language=[], language_note="trained on 140+ languages per the upstream model card; no ISO tag list published",
        title="Gemma 3 270M IT", summary="Google's Gemma 3 270M instruction-tuned LM, exported for loom.cpp.",
    ),
    ModelCard(
        slug="whisper-small", checkpoint=Path("whisper-small"),
        task_type="automatic-speech-recognition",
        # whisper_mil.gguf's own `loom.sample_rate`, the one ASR export that declares it, and its
        # `loom.n_samples` -- which the topology repeats as a literal `waveform` dim of 480000.
        input_sample_rate=16000, fixed_audio_samples=480000,
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
        slug="conformer-ctc-small",
        # stt_en_conformer_ctc_small.nemo's model_config.yaml `sample_rate`.
        input_sample_rate=16000, checkpoint=Path("conformer-ctc-small/stt_en_conformer_ctc_small.nemo"),
        export_task="automatic-speech-recognition", export_model="conformer-ctc",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/stt_en_conformer_ctc_small", license_id="cc-by-4.0", language=["en"],
        title="Conformer-CTC Small (en)", summary="NVIDIA NeMo's small Conformer-CTC English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="parakeet-tdt-0.6b",
        # parakeet-tdt-0.6b-v3.nemo's model_config.yaml `sample_rate`.
        input_sample_rate=16000, checkpoint=Path("parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo"),
        export_task="automatic-speech-recognition", export_model="parakeet-tdt",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/parakeet-tdt-0.6b-v3", license_id="cc-by-4.0",
        language=["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
                  "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"],
        title="Parakeet-TDT 0.6B v3", summary="NVIDIA NeMo's multilingual Parakeet-TDT 0.6B ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="parakeet-rnnt-0.6b",
        # parakeet-rnnt-0.6b.nemo's model_config.yaml `sample_rate`.
        input_sample_rate=16000, checkpoint=Path("parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo"),
        export_task="automatic-speech-recognition", export_model="parakeet-rnnt",
        task_type="automatic-speech-recognition",
        base_repo="nvidia/parakeet-rnnt-0.6b", license_id="cc-by-4.0", language=["en"],
        title="Parakeet-RNNT 0.6B", summary="NVIDIA NeMo's Parakeet-RNNT 0.6B English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="gigaam-v3-rnnt",
        # gigaam-v3/config.json `cfg.model.cfg.preprocessor.sample_rate`.
        input_sample_rate=16000, checkpoint=Path("gigaam-v3"),
        export_task="automatic-speech-recognition", export_model="gigaam-rnnt",
        task_type="automatic-speech-recognition",
        base_repo="ai-sage/GigaAM-v3", license_id="mit", language=["ru", "en"],
        title="GigaAM-v3 RNNT", summary="Sber's GigaAM-v3 Conformer RNNT Russian/English ASR model, exported for loom.cpp.",
    ),
    ModelCard(
        slug="qwen3-asr-0.6b",
        # qwen3-asr-0.6b-hf/processor_config.json `feature_extractor.sampling_rate`.
        input_sample_rate=16000, checkpoint=Path("qwen3-asr-0.6b-hf"), venv="ovos",
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
        slug="granite-speech-4.0-1b",
        # granite-speech-4.0.1b/preprocessor_config.json `sampling_rate`.
        input_sample_rate=16000, checkpoint=Path("granite-speech-4.0.1b"),
        task_type="automatic-speech-recognition",
        base_repo="ibm-granite/granite-4.0-1b-speech", license_id="apache-2.0",
        language=["en", "fr", "de", "es", "pt", "ja"], language_note="upstream also tags the model \"multilingual\"",
        title="Granite-4.0-1b-speech", summary="IBM's Granite 4.0 1B speech-language model (ASR + AST), exported for loom.cpp.",
    ),
    ModelCard(
        slug="kokoro-82m", checkpoint=Path("kokoro_model"),
        export_task="text-to-speech", export_model="kokoro", task_type="text-to-speech",
        # Not in the checkpoint's config.json: upstream hardcodes it in `Generator.__init__`, which is
        # where tools/convert_kokoro/convert_kokoro_sinegen.py reads its own copy from.
        sample_rate=24000,
        base_repo="hexgrad/Kokoro-82M", license_id="apache-2.0", language=["en"],
        language_note="upstream's own `language:` tag is `en`; the model card additionally documents 8 languages / 54 voices",
        title="Kokoro-82M", summary="hexgrad's Kokoro-82M TTS model, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="matcha-tts-ljspeech", checkpoint=Path("matcha_model/ckpt"),
        export_task="text-to-speech", export_model="matcha", task_type="text-to-speech",
        sample_rate=22050,  # matcha_ljspeech.ckpt's own `datamodule_hyper_parameters["sample_rate"]`
        source_url="https://github.com/shivammehta25/Matcha-TTS", source_name="Matcha-TTS (LJSpeech checkpoint)",
        license_id="mit", language=["en"],
        title="Matcha-TTS (LJSpeech)", summary="Matcha-TTS's LJSpeech flow-matching TTS checkpoint, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="supertonic-2", checkpoint=Path("/home/flavio/Dev/supertonic-tts/assets/pt"),
        export_task="text-to-speech", export_model="supertonic", task_type="text-to-speech",
        takes_text=True,
        sample_rate=44100,  # loom_exporter.supertonic_export.DEC_SAMPLE_RATE, and see its comment
        base_repo="Supertone/supertonic-2", license_id="other",
        license_name="OpenRAIL-M", license_url="https://huggingface.co/Supertone/supertonic-2/blob/main/LICENSE",
        language=["en", "ko", "es", "pt", "fr"],
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
        sample_rate=22050,  # the voice's own miro_en-GB.onnx.json `audio.sample_rate`
        base_repo="OpenVoiceOS/pipertts_en-GB_miro", license_id="other",
        license_name=None, license_url=None,
        language=["en"],
        language_note="upstream declares no `license:` tag; check https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro "
                       "directly before redistributing",
        title="Piper VITS en-GB (miro)", summary="OpenVoiceOS's Piper-compatible VITS en-GB \"miro\" voice, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
    ModelCard(
        slug="styletts2-ljspeech",
        checkpoint=Path("styletts2_model/ckpt/Models/LJSpeech/epoch_2nd_00100.pth"),
        export_task="text-to-speech", export_model="styletts2", task_type="text-to-speech",
        # config.yml's `preprocess_params.sr`. NOT the `sr: 16000` further up, which belongs to the
        # SLM (WavLM) discriminator used in training and says nothing about the decoder's output.
        sample_rate=24000,
        base_repo="yl4579/StyleTTS2-LJSpeech", license_id="mit",
        language_note="the HF repo carries no `license:`/`language:` tags; MIT per the upstream "
                       "GitHub repo's LICENSE (github.com/yl4579/StyleTTS2)",
        language=["en"],
        title="StyleTTS2 (LJSpeech)", summary="yl4579's StyleTTS2 LJSpeech checkpoint, exported for loom.cpp. Takes phoneme ids, not text.",
    ),
]

CATALOG_BY_SLUG = {m.slug: m for m in CATALOG}

USAGE_SNIPPETS = {
    "text-generation": """import loom

model = loom.Model.from_pretrained("{repo_id}")
print(model.generate("The capital of France is", max_new_tokens=14))
""",
    "automatic-speech-recognition": """import loom

model = loom.Model.from_pretrained("{repo_id}")
transcript = model.detokenize(model.infer(waveform=audio, audio_samples=len(audio)))
print(transcript)
""",
    # Two TTS snippets, because "TTS" is not one answer. Which one a model gets is `takes_text` below,
    # a per-model fact read off the export rather than assumed from the task -- the single phoneme-ids
    # snippet this used to have was simply wrong for Supertonic, whose GGUF carries its own vocabulary.
    "text-to-speech": """import loom

model = loom.Model.from_pretrained("{repo_id}")
# {slug} takes phoneme ids, not text -- see model.driver_source for the exact driver inputs.
audio = model.infer(tokens=[16, 40, 22, 30, 12, 3], n_steps=4, seed=1234)
""",
    "text-to-speech-with-vocab": """import loom

model = loom.Model.from_pretrained("{repo_id}")

# This model encodes text itself -- no external phonemiser needed.
print(model.tokenizer)                       # kind, vocabulary size, default language
txt_ids = model.tokenize("hello world")      # model.tokenize(..., lang="ko") to pick a language

# Any length up to `model.hparam("txt_len")` -- the driver pads and masks the rest.
audio = model.infer(txt_ids=txt_ids, n_steps=4, seed=1234)

# That uses whatever voice the file itself defaults to. See below for choosing another.
""",
}


def repo_id(card: ModelCard) -> str:
    return f"loom-ai-org/{card.slug}-loom"


def snippet_key(card: ModelCard) -> str:
    """Which `USAGE_SNIPPETS` entry this model's card gets. The task decides it for every family except
    TTS, where whether the GGUF carries a vocabulary is a per-model fact -- see `takes_text`."""
    if card.task_type == "text-to-speech" and card.takes_text:
        return "text-to-speech-with-vocab"
    return card.task_type


def wav_section(card: ModelCard) -> str:
    """The "Saving the audio" section every TTS card gets, or "" for a non-TTS one.

    Shared rather than per-card because the step after `infer` is the same for all five TTS families --
    scale the floats to int16 and write -- and the only thing that differs is the rate, which is exactly
    the part a caller cannot infer and the part that is silently wrong if they guess. A waveform written
    at the wrong rate still plays; it just plays at the wrong pitch and speed, which is why this is in
    the card at all rather than left as an exercise."""
    if card.task_type != "text-to-speech":
        return ""
    if card.sample_rate is None:
        raise ValueError(f"{card.slug}: a text-to-speech card must declare sample_rate")
    khz = f"{card.sample_rate / 1000:g}"
    return f"""### Saving the audio

`infer` returns the waveform as a plain list of floats in `[-1, 1]` at **{khz} kHz**. Writing it to a
`.wav` is a scale and a write -- numpy and scipy are not loom dependencies, they are just the shortest
way to say it:

```python
import numpy as np
import scipy.io.wavfile as wavfile

sample_rate = {card.sample_rate}   # {card.title}'s output rate -- playing it at any other rate shifts pitch and speed

# 16-bit PCM is the usual container format, so scale the floats to its integer range.
audio_int16 = (np.asarray(audio, dtype=np.float32) * 32767).astype(np.int16)
wavfile.write("output.wav", sample_rate, audio_int16)
```
"""


def audio_section(card: ModelCard) -> str:
    """The "Loading the audio" section every ASR card gets, or "" for a non-ASR one.

    The exact mirror of wav_section, and it exists for the same reason: the usage snippet says
    `waveform=audio` without ever saying where `audio` comes from, and every way of getting it wrong
    is silent. Audio at the wrong rate transcribes badly rather than failing; a stereo file transcribes
    one interleaved channel as if it were twice the length; int16 samples read as floats are 32767x
    too loud. None of those raise -- they just produce a worse transcript, which the caller has no way
    to attribute."""
    if card.task_type != "automatic-speech-recognition":
        return ""
    if card.input_sample_rate is None:
        raise ValueError(f"{card.slug}: an ASR card must declare input_sample_rate")

    khz = f"{card.input_sample_rate / 1000:g}"
    body = f"""### Loading the audio

`infer` takes a plain list of floats in `[-1, 1]` at **{khz} kHz**, mono. Getting there from a `.wav`
is a read, a rate check and a scale -- numpy and scipy are not loom dependencies, they are just the
shortest way to say it:

```python
import numpy as np
import scipy.io.wavfile as wavfile

SAMPLE_RATE = {card.input_sample_rate}   # what {card.title} was trained on; other rates transcribe badly rather than failing

rate, data = wavfile.read("speech.wav")
assert rate == SAMPLE_RATE, f"expected {{SAMPLE_RATE}} Hz, got {{rate}} -- resample first (e.g. `ffmpeg -i in.wav -ar {card.input_sample_rate} -ac 1 speech.wav`)"

# Mono. A stereo file read as-is is two interleaved channels, which transcribes as noise.
if data.ndim > 1:
    data = data.mean(axis=1)

# int16 PCM is the usual container; the model wants floats in [-1, 1].
audio = (data.astype(np.float32) / 32768.0) if data.dtype == np.int16 else data.astype(np.float32)
```
"""

    if card.fixed_audio_samples is not None:
        seconds = card.fixed_audio_samples / card.input_sample_rate
        body += f"""
### This model takes exactly {seconds:g} seconds, and `infer` will not pad for you

{card.title}'s export declares its `waveform` input as a literal **{card.fixed_audio_samples} samples**
-- {seconds:g} s at {khz} kHz -- where every other ASR export here declares a symbolic length. The GGUF says so
in `loom.n_samples`, and that is the only ASR export that carries the key.

**Which half of loom you use decides whether you notice.** `loom_cli` reads `loom.n_samples` and does
the windowing itself -- zero-padding a short clip and splitting a long one, following Whisper's own
`pad_or_trim` convention -- so a file of any length transcribes with no special handling. `model.infer()`
does not: it passes the waveform you give it straight through. That difference is why a clip that works
on the command line can surprise you from Python.

```python
WINDOW = {card.fixed_audio_samples}   # {seconds:g} s at {khz} kHz, fixed by the export

if len(audio) < WINDOW:
    audio = np.pad(audio, (0, WINDOW - len(audio)))   # silence, as Whisper itself pads
else:
    audio = audio[:WINDOW]                            # one window per call -- see below for longer audio
```

Padding with silence is what the upstream model does, so a short clip transcribes normally. Audio
longer than {seconds:g} s needs splitting into consecutive windows and transcribing each, concatenating the
transcripts -- this export gives you one window per `infer` call.
"""
    return body


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
    extra_files_section = "".join(f"- {bullet}\n" for bullet in card.extra_files)
    # Before `usage_extra`, so a TTS card reads snippet -> what to do with the waveform -> whatever else
    # that one model needs said (Supertonic's voice selection).
    wav = wav_section(card)
    wav_section_md = f"\n{wav}" if wav else ""
    # BEFORE the snippet, not after, and the asymmetry with wav_section is deliberate: the ASR snippet
    # passes `waveform=audio` without `audio` existing yet, so a reader meets an undefined name unless
    # the loading comes first. TTS is the other way round -- `infer` produces the waveform, and saving
    # it is what happens next.
    audio = audio_section(card)
    audio_section_md = f"{audio}\n" if audio else ""

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
pip install -U "loom-py-rt[hub]"
```

{audio_section_md}```python
{USAGE_SNIPPETS[snippet_key(card)].format(repo_id=repo_id(card), slug=card.slug)}```
{wav_section_md}{usage_extra_section}
`model.driver_source` prints the exact driver script this GGUF embeds, including a header comment
documenting every argument `model.infer()`/`model.generate()` accepts for this model.
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
