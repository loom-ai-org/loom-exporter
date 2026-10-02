"""LFM2.5-Audio's voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

An LFM2.5-Audio voice is not a recording or a state: it is the SYSTEM PROMPT the model was trained to
read, one of the README's four ("Perform TTS. Use the US male voice." ...). So a voice file (loom.cpp
ADR-045) holds one driver input, `voice_prompt` -- the prompt's ids, tokenized as `ChatState.add_text`
tokenizes it -- and the model GGUF carries its default (`us_male`) as a constant. loom-py's
`text2speech.infer(text, voice="uk_female")` then fetches the file from the model's repo by name.

**A voice fits the tokenizer it was made with**, because it is ids: every file is stamped with
`loom.voice.compat`, a fingerprint of the checkpoint's `tokenizer.json`, which the model GGUF declares
too, and `loom::load_voice` refuses a mismatch.

    python -m loom_exporter.lfm25_audio_voices ~/Dev/models/lfm2.5-audio-1.5b -o voices/   # all four
"""
import argparse
import hashlib
from pathlib import Path
from typing import Dict

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "lfm2.5-audio-tts"
LICENSE = "LFM Open License v1.0 (the voice is one of the model's own system prompts)"


def tokenizer_fingerprint(model_dir) -> str:
    """sha256 of `tokenizer.json`: what a voice's ids mean."""
    return hashlib.sha256((Path(model_dir) / "tokenizer.json").read_bytes()).hexdigest()[:32]


def write_voice(ids, out: Path, *, name: str, compat: str, origin: str) -> None:
    from gguf import GGUFWriter

    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", LICENSE)
    w.add_string("loom.voice.origin", origin)
    w.add_tensor("voice_prompt", np.asarray(ids, dtype=np.float32).reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def convert(model_dir, out_dir) -> Dict[str, list]:
    """Every README voice, one file each."""
    from .lfm25_audio_export import TTS_VOICES, tts_prompt_ids

    ids = tts_prompt_ids(str(model_dir))["voices"]
    compat = tokenizer_fingerprint(model_dir)
    for name, prompt in TTS_VOICES.items():
        write_voice(ids[name], Path(out_dir) / f"{name}.gguf", name=name, compat=compat,
                    origin=f"LiquidAI/LFM2.5-Audio-1.5B README: {prompt!r}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("-o", "--out-dir", required=True)
    args = ap.parse_args()
    for name, ids in convert(args.model_dir, args.out_dir).items():
        print(f"{name}: {len(ids)} ids -> {Path(args.out_dir) / (name + '.gguf')}")


if __name__ == "__main__":
    main()
