"""Voxtral-4B-TTS voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

A Voxtral preset voice (`voice_embedding/<name>.pt`) is `[n, 3072]` rows the LM reads as INPUT
EMBEDDINGS: vllm-omni's `tts_preprocess` writes them over the prompt's n [AUDIO] slots, and
`mistral_common` sizes those slots from `tekken.json`'s `voice_num_audio_tokens`. This converts each into
a **voice file** -- a tiny GGUF whose tensors are DRIVER INPUTS by name, here one, `voice`, which the
driver writes over the slots in place of the file's built-in voice (loom.cpp ADR-045). `loom_cli
--voice` and loom-py's `text2speech.infer(text, voice="fr_female")` read it with no per-model code.

    python -m loom_exporter.voxtral_tts_voices ~/Dev/models/voxtral-4b-tts-2603 -o voices  # all twenty

**A voice only fits the LM it was made for**, so every file is stamped with `loom.voice.compat`, a
fingerprint of the checkpoint's LM tensors (and the audio-code embeddings the LM also reads), and the
model GGUF declares the same key; `loom::load_voice` refuses a mismatch by name.

**Every voice is CC BY-NC 4.0.** The model card: the voice references come from EARS, CML-TTS,
IndicVoices-R and the Arabic Natural Audio dataset, all non-commercial, "and this model inherits the
same license". New voices cannot be made here: the open checkpoint ships no codec ENCODER, which is
what turns a clip into these rows (upstream's own words: "the open-source variant only supports preset
voices").
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "voxtral_tts"
LICENSE = "CC-BY-NC-4.0"
ORIGIN = "mistralai/Voxtral-4B-TTS-2603, voice_embedding/{name}.pt (from EARS, CML-TTS, IndicVoices-R or ANAD)"
# The tensors a voice depends on: the LM that reads its rows, and the code embeddings that feed the same
# LM every frame after them. The flow head and the codec are excluded -- they never see a voice row.
_FINGERPRINTED = ("layers.", "norm.", "mm_audio_embeddings.")


def weights_fingerprint(consolidated: Path) -> str:
    """sha256 over every LM tensor's name, dtype, shape and bytes, in name order, read raw from the
    safetensors file (pocket_tts_voices' recipe)."""
    with open(consolidated, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        base = 8 + header_len
        digest = hashlib.sha256()
        for name in sorted(k for k in header if k.startswith(_FINGERPRINTED)):
            entry = header[name]
            start, end = entry["data_offsets"]
            digest.update(f"{name}|{entry['dtype']}|{entry['shape']}|".encode())
            f.seek(base + start)
            remaining = end - start
            while remaining:
                chunk = f.read(min(remaining, 1 << 24))
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()[:32]


def expected_rows(model_dir: Path) -> Dict[str, int]:
    """`tekken.json`'s `voice_num_audio_tokens`: how many [AUDIO] slots `mistral_common` gives each voice."""
    return json.loads((Path(model_dir) / "tekken.json").read_text())["audio"]["voice_num_audio_tokens"]


def write_voice(rows: np.ndarray, out: Path, *, name: str, compat: str, license: str, origin: str) -> int:
    from gguf import GGUFWriter

    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    w.add_uint32("loom.voice.n_rows", int(rows.shape[0]))
    # The tensor's NAME is the driver input it becomes: `inputs.voice`, flat `[n * 3072]`.
    w.add_tensor("voice", np.ascontiguousarray(rows.reshape(-1), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return int(rows.shape[0])


def convert(model_dir: Path, out_dir: Path, *, only=None) -> Dict[str, dict]:
    """Every preset voice under `model_dir/voice_embedding` (or `only` of them). Each is checked against
    the slot count the tokenizer gives it, since a voice is exactly that many prompt rows."""
    from .voxtral_tts_export import read_voice

    model_dir = Path(model_dir)
    compat = weights_fingerprint(model_dir / "consolidated.safetensors")
    slots = expected_rows(model_dir)
    names = sorted(p.stem for p in (model_dir / "voice_embedding").glob("*.pt"))
    if only:
        missing = set(only) - set(names)
        if missing:
            raise FileNotFoundError(f"no voice_embedding/<name>.pt for {sorted(missing)} in {model_dir}")
        names = [n for n in names if n in only]
    written = {}
    for name in names:
        rows = read_voice(str(model_dir), name)
        if slots.get(name) != rows.shape[0]:
            raise ValueError(f"voice {name!r} has {rows.shape[0]} rows and tekken.json gives it "
                             f"{slots.get(name)} [AUDIO] slots")
        write_voice(rows, Path(out_dir) / f"{name}.gguf", name=name, compat=compat, license=LICENSE,
                    origin=ORIGIN.format(name=name))
        written[name] = {"n_rows": int(rows.shape[0]), "license": LICENSE}
    return written


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the mistralai/Voxtral-4B-TTS-2603 directory")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--only", nargs="+", help="convert only these preset voices")
    args = ap.parse_args(argv)
    written = convert(args.model_dir, args.out, only=args.only)
    for voice, info in written.items():
        print(f"{voice:16s} {info['n_rows']:4d} rows  {info['license']}")
    print(f"wrote {len(written)} voice file(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
