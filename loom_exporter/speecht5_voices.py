"""SpeechT5 voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

A SpeechT5 voice is a speaker x-vector: 512 floats from SpeechBrain's `spkrec-xvect-voxceleb`
extractor, which the decoder prenet L2-normalises and projects. This writes one as a **voice file**
(loom.cpp ADR-045): a tiny GGUF whose one tensor is the driver input it becomes, `speaker`. The
engine's `loom::load_voice` reads it for `loom_cli --voice` and for loom-py's
`text2speech.infer(text, voice="bdl")`.

**The stamp is the embedding space, not a weights hash** (loom.cpp ADR-058). A Pocket-TTS voice is
the model's own attention state, so it fits only the weights that produced it. An x-vector is the
EXTRACTOR's output, and it fits any SpeechT5 trained on that extractor's embeddings, fine-tunes
included. `loom.voice.compat` is therefore `xvector:<extractor>:<dim>`, which the model declares too.

    python -m loom_exporter.speecht5_voices ~/Dev/models/speecht5-tts -o hf-models/speecht5-tts/voices
    python -m loom_exporter.speecht5_voices <model_dir> -o voices \\
        --from my_xvector.npy --name me --license "CC0-1.0"        # an x-vector you extracted yourself

The seven CMU ARCTIC speakers come from `<model_dir>/xvectors/spkrec-xvect.zip`
(`Matthijs/cmu-arctic-xvectors`, MIT). Each voice is its speaker's `arctic_a0508` utterance, the same
sentence as the `slt` x-vector every published SpeechT5 example uses. The recordings are CMU's, "free
for use for any purpose (commercial or otherwise)" provided the copyright notice is kept, and each
file records that.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "speecht5"
EXTRACTOR = "speechbrain/spkrec-xvect-voxceleb"
XVECTOR_ZIP = Path("xvectors") / "spkrec-xvect.zip"
UTTERANCE = "arctic_a0508"

# The seven CMU ARCTIC speakers, as the dataset's README describes them.
SPEAKERS = {
    "awb": "Scottish male",
    "bdl": "US male",
    "clb": "US female",
    "jmk": "Canadian male",
    "ksp": "Indian male",
    "rms": "US male",
    "slt": "US female",
}
CMU_ARCTIC_LICENSE = ("CMU ARCTIC: free for use for any purpose (commercial or otherwise); keep the "
                      "Carnegie Mellon University copyright notice (c) 2003 and mark modifications. "
                      "x-vector: MIT (Matthijs/cmu-arctic-xvectors)")


def compat(dim: int = 512) -> str:
    """What a SpeechT5 voice file must match: the embedding space its x-vector lives in."""
    return f"xvector:{EXTRACTOR}:{dim}"


def utterance(speaker: str) -> str:
    """The `spkrec-xvect.zip` member stem of a speaker's voice: `cmu_us_<speaker>_arctic-wav-arctic_a0508`."""
    if speaker not in SPEAKERS:
        raise KeyError(f"no CMU ARCTIC speaker {speaker!r}; the seven are {', '.join(SPEAKERS)}")
    return f"cmu_us_{speaker}_arctic-wav-{UTTERANCE}"


def read_xvector(model_dir: Path, speaker: str, dim: int = 512) -> np.ndarray:
    """A speaker's x-vector out of `<model_dir>/xvectors/spkrec-xvect.zip`."""
    path = Path(model_dir) / XVECTOR_ZIP
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. SpeechT5 needs a speaker x-vector and its checkpoint ships none; "
            f"download `spkrec-xvect.zip` from the `Matthijs/cmu-arctic-xvectors` dataset into "
            f"{path.parent}/ (do not unzip it: 7931 files).")
    member = f"spkrec-xvect/{utterance(speaker)}.npy"
    with zipfile.ZipFile(path) as z:
        if member not in z.namelist():
            raise KeyError(f"{path.name} has no {member}")
        return check_xvector(np.load(io.BytesIO(z.read(member))), member, dim)


def check_xvector(x, what: str, dim: int = 512) -> np.ndarray:
    """`x` as `[dim]` float32, accepting the `[1, dim]` a batched extractor returns."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    if x.shape != (dim,) or not np.isfinite(x).all() or not np.any(x):
        raise ValueError(f"{what}: expected {dim} finite, not-all-zero floats (the checkpoint's "
                         f"`speaker_embedding_dim`), got shape {x.shape}")
    return x


def write_voice(xvector: np.ndarray, out: Path, *, name: str, license: str, origin: str) -> None:
    from gguf import GGUFWriter

    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat(xvector.shape[0]))
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    # The tensor's NAME is the driver input it becomes: `inputs.speaker`, un-normalised, as the
    # reference takes it (the graph normalises).
    w.add_tensor("speaker", xvector.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def convert(model_dir: Path, out_dir: Path, *, only=None, source: Optional[Path] = None,
            name: Optional[str] = None, license: Optional[str] = None, dim: int = 512) -> Dict[str, dict]:
    """The CMU ARCTIC speakers (or `only` of them), or the single x-vector `source` (`.npy`) under
    `name`. Returns what was written, by name."""
    if source is not None:
        if not name or not license:
            raise ValueError("a voice from your own x-vector needs --name and --license: the licence of "
                             "the recording it was extracted from, which only you know")
        jobs = {name: (check_xvector(np.load(source), str(source), dim), license, str(source))}
    else:
        unknown = set(only or ()) - set(SPEAKERS)
        if unknown:
            raise KeyError(f"no CMU ARCTIC speaker(s) {sorted(unknown)}; the seven are {', '.join(SPEAKERS)}")
        jobs = {}
        for speaker in SPEAKERS:
            if only and speaker not in only:
                continue
            origin = (f"Matthijs/cmu-arctic-xvectors spkrec-xvect/{utterance(speaker)}.npy "
                      f"(CMU ARCTIC {speaker}, {SPEAKERS[speaker]})")
            jobs[speaker] = (read_xvector(model_dir, speaker, dim), CMU_ARCTIC_LICENSE, origin)
    written = {}
    for voice, (xvector, licence, origin) in jobs.items():
        write_voice(xvector, Path(out_dir) / f"{voice}.gguf", name=voice, license=licence, origin=origin)
        written[voice] = {"license": licence, "origin": origin}
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the speecht5_tts directory (with xvectors/)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--only", nargs="+", help=f"convert only these speakers ({', '.join(SPEAKERS)})")
    ap.add_argument("--from", dest="source", type=Path, help="a 512-float x-vector saved with numpy")
    ap.add_argument("--name", help="the voice's name, with --from")
    ap.add_argument("--license", help="the recording's licence, with --from")
    args = ap.parse_args(argv)
    written = convert(args.model_dir, args.out, only=args.only, source=args.source, name=args.name,
                      license=args.license)
    for voice, info in written.items():
        print(f"{voice:8s} {info['origin']}")
    print(f"wrote {len(written)} voice file(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
