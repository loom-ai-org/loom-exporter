"""Pocket-TTS voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

A Pocket-TTS voice is the flow LM's KV cache after it has heard the speaker (loom.cpp ADR-043), and the
reference ships its predefined voices, and writes a user's own with `pocket-tts export-voice`, as
safetensors of that cache. This converts either into a **voice file**: a tiny GGUF whose tensors are
DRIVER INPUTS by name -- here one, `voice_kv`, in the layout `loom.seed_kv` reads -- so no host needs to
know this model's key names or layout (loom.cpp ADR-045). The engine's `loom::load_voice` reads it
for both hosts: `loom_cli --voice`, and loom-py's `text2speech.infer(text, voice="marius")`.

**A voice only fits the weights it was made with.** The reference says so itself: a predefined voice
fed to other weights "typically never emits EOS". So every file is stamped with
`loom.voice.compat`, a fingerprint of the checkpoint's `flow_lm.*` tensors, and the model GGUF declares
the same key; `loom::load_voice` refuses a mismatch by name rather than letting a voice for the
2026-04 release run on 2026-09.

    python -m loom_exporter.pocket_tts_voices ~/Dev/models/pocket-tts/languages/english_2026-09 \\
        -o hf-models/pocket-tts-loom/voices                      # all 26 predefined voices
    python -m loom_exporter.pocket_tts_voices <model_dir> -o voices \\
        --from my_voice.safetensors --name me --license "CC0-1.0"   # a voice you exported yourself

**Licences travel with each file** (`loom.voice.license`, `loom.voice.origin`), because they are not
one licence. They are the recordings', per `kyutai/tts-voices`' README, and two of the predefined
voices are NON-COMMERCIAL (`cosette`, Expresso; `jean`, EARS).
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "pocket-tts"

# The recordings behind the reference's predefined voices (`utils._ORIGINS_OF_PREDEFINED_VOICES`), by
# the folder they come from, and each folder's licence as `kyutai/tts-voices`' README states it
# (read 2026-09-24). Matched on the origin's path, so a voice the reference adds later gets a licence
# only if its folder is one of these, and "unstated" otherwise -- never a guess.
_LICENSE_BY_FOLDER = {
    "kyutai/tts-voices/vctk/": "CC-BY-4.0",
    "kyutai/tts-voices/alba-mackenna/": "CC-BY-4.0",
    "kyutai/tts-voices/voice-donations/": "CC0-1.0",
    "kyutai/tts-voices/voice-zero/": "CC0-1.0",
    # "The others are our own recordings and you may use them as CC0" -- Unmute's section.
    "kyutai/tts-voices/unmute-prod-website/developpeuse": "CC0-1.0",
    "kyutai/tts-voices/expresso/": "CC-BY-NC-4.0",
    "kyutai/tts-voices/ears/": "CC-BY-NC-4.0",
    # Mozilla Common Voice, which is CC0.
    "kyutai/pocket-tts/common_voice_": "CC0-1.0",
}
UNSTATED = ("unstated: the source clip ships in kyutai/pocket-tts (CC-BY-4.0) with no licence of its "
            "own; check before redistributing")


def weights_fingerprint(model_safetensors: Path) -> str:
    """sha256 over every `flow_lm.*` tensor's name, dtype, shape and bytes, in name order: what a voice
    depends on. Mimi is excluded on purpose -- the release without voice cloning zeroes its encoder,
    and a voice made with either release is the same voice."""
    with open(model_safetensors, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        digest = hashlib.sha256()
        for name in sorted(k for k in header if k.startswith("flow_lm.")):
            entry = header[name]
            start, end = entry["data_offsets"]
            digest.update(f"{name}|{entry['dtype']}|{entry['shape']}|".encode())
            f.seek(base + start)
            digest.update(f.read(end - start))
    return digest.hexdigest()[:32]


def origin_and_license(name: str) -> Tuple[str, str]:
    from .pocket_tts_export import import_pocket_tts

    import_pocket_tts()
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

    origin = _ORIGINS_OF_PREDEFINED_VOICES.get(name, "")
    for prefix, licence in _LICENSE_BY_FOLDER.items():
        if origin.startswith("hf://" + prefix):
            return origin, licence
    return origin, UNSTATED


def write_voice(src: Path, out: Path, *, name: str, compat: str, n_layers: int, license: str,
                origin: str) -> int:
    """One saved voice state -> one voice file. Returns its length in positions."""
    from gguf import GGUFWriter

    from .pocket_tts_export import read_voice

    flat, n_rows = read_voice(src, n_layers)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    w.add_uint32("loom.voice.n_rows", n_rows)
    # The tensor's NAME is the driver input it becomes: `inputs.voice_kv`, which the driver seeds into
    # `lm`'s cache in place of the built-in voice.
    w.add_tensor("voice_kv", flat.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n_rows


def convert(model_dir: Path, out_dir: Path, *, only=None, source: Optional[Path] = None,
            name: Optional[str] = None, license: Optional[str] = None) -> Dict[str, dict]:
    """Every predefined voice under `model_dir/embeddings` (or `only` of them), or the single saved
    state `source` under `name`. Returns what was written, by name."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    compat = weights_fingerprint(model_dir / "model.safetensors")
    with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
        n_layers = len({k.split(".")[3] for k in f.keys() if k.startswith("flow_lm.transformer.layers.")})
    if source is not None:
        if not name or not license:
            raise ValueError("a voice from your own file needs --name and --license: the licence of the "
                             "recording it was made from, which only you know")
        jobs = {name: (Path(source), license, str(source))}
    else:
        jobs = {}
        for path in sorted((model_dir / "embeddings").glob("*.safetensors")):
            if only and path.stem not in only:
                continue
            origin, licence = origin_and_license(path.stem)
            jobs[path.stem] = (path, licence, origin)
        missing = set(only or ()) - set(jobs)
        if missing:
            raise FileNotFoundError(f"no embeddings/<name>.safetensors for {sorted(missing)} in {model_dir}")
    written = {}
    for voice, (path, licence, origin) in jobs.items():
        n_rows = write_voice(path, Path(out_dir) / f"{voice}.gguf", name=voice, compat=compat,
                             n_layers=n_layers, license=licence, origin=origin)
        written[voice] = {"n_rows": n_rows, "license": licence, "origin": origin}
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the pocket-tts languages/<name> directory")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--only", nargs="+", help="convert only these predefined voices")
    ap.add_argument("--from", dest="source", type=Path, help="a state saved by `pocket-tts export-voice`")
    ap.add_argument("--name", help="the voice's name, with --from")
    ap.add_argument("--license", help="the recording's licence, with --from")
    args = ap.parse_args(argv)
    written = convert(args.model_dir, args.out, only=args.only, source=args.source, name=args.name,
                      license=args.license)
    for voice, info in written.items():
        flag = "  NON-COMMERCIAL" if "NC" in info["license"] else ""
        print(f"{voice:16s} {info['n_rows']:4d} rows  {info['license']}{flag}")
    print(f"wrote {len(written)} voice file(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
