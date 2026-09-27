"""MOSS-TTS voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

MOSS-TTS-Local clones a voice from one or more reference clips, and what it reads of a clip is its
CODES: MOSS-Audio-Tokenizer-v2's encoder turns the clip into 12 codebooks per frame, and the prompt
carries them between `<audio_start>` and `<audio_end>` rows under "- Reference(s):"
(`processing_moss_tts._build_generation_or_voice_clone_codes`). This tool runs the reference's own
encode path ONCE, in Python -- its channel handling, 48 kHz resample and loudness normalisation, then
the codec's `batch_encode` at F32 -- and writes the codes as a **voice file**: a tiny GGUF whose tensors
are DRIVER INPUTS by name, which `loom::load_voice` reads for both hosts (`loom_cli --voice`, loom-py's
`text2codes.infer(text, voice="me")`). So cloning costs Python and the 8.5 GB codec once per voice, and
the codec's encoder never has to reach the engine (loom.cpp ADR-045).

    python -m loom_exporter.moss_tts_voices ~/Dev/models/moss-tts-local-transformer-v1.5 -o voices \\
        --wav me.wav --name me --license "CC0-1.0"
    python -m loom_exporter.moss_tts_voices <model_dir> -o voices --wav s1.wav --wav s2.wav \\
        --name dialogue --license "CC0-1.0"          # two references, for "[S1] ... [S2] ..." text

**Several clips are one voice.** The reference takes a LIST of references, one per speaker of a
dialogue, and a file holds them all: `reference_codes` (every reference's frames, frame-major, 12 per
frame) and `reference_frames` (one frame count per reference, in order). A host passes a voice's
inputs through without knowing what they are, so a multi-speaker cast has to be one file.

**A voice fits the CODEC, not the TTS weights.** Codes mean what the codec's quantizer says they mean,
so every file is stamped with `loom.voice.compat`, a fingerprint of the codec's `quantizer.*` tensors,
and the TTS export declares the fingerprint of the codec its config names. Any MOSS-TTS-Local checkpoint
trained on this codec reads the same voice; one trained on another codec refuses it by name.

**The licence is the recording's**, which only whoever made the clip knows, so `--license` is required.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "moss_tts_local"
# The driver inputs a voice sets. `reference_frames` is what splits `reference_codes` into references.
VOICE_INPUTS = ("reference_codes", "reference_frames")


def find_codec(model_dir, codec_dir=None) -> Path:
    """The codec a MOSS-TTS checkpoint speaks through: `codec_dir` when given, else the directory beside
    the checkpoint named after its config's `audio_tokenizer_name_or_path` (a Hub id, so its last
    component: `MOSS-Audio-Tokenizer-v2` -> `moss-audio-tokenizer-v2`, compared without case)."""
    if codec_dir is not None:
        path = Path(codec_dir).expanduser()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"codec_dir {path} has no config.json")
        return path
    config = json.loads((Path(model_dir) / "config.json").read_text())
    named = str(config.get("audio_tokenizer_name_or_path") or "")
    if not named:
        raise ValueError(f"{model_dir}/config.json names no audio_tokenizer_name_or_path; pass the codec "
                         f"directory explicitly")
    if Path(named).expanduser().is_dir():
        return Path(named).expanduser()
    want = named.rstrip("/").split("/")[-1].lower()
    parent = Path(model_dir).expanduser().resolve().parent
    for sibling in sorted(parent.iterdir()):
        if sibling.is_dir() and sibling.name.lower() == want and (sibling / "config.json").is_file():
            return sibling
    raise FileNotFoundError(
        f"the codec this checkpoint names ({named}) is not beside it in {parent}: pass its directory "
        f"(`codec_dir`, or `--codec`). A voice's fingerprint is the codec's, so it cannot be skipped.")


def codec_fingerprint(codec_dir) -> str:
    """sha256 over the codec's `quantizer.*` tensors -- name, dtype, shape and bytes, in name order --
    which is what decides what a code MEANS. The encoder and decoder are left out: a code is the
    quantizer's index, and a codec that re-trained either side around the same codebooks still means
    the same thing by it."""
    import torch
    from safetensors import safe_open

    codec_dir = Path(codec_dir)
    index = codec_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(codec_dir / "model.safetensors"), framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    names = sorted(k for k in weight_map if k.startswith("quantizer."))
    if not names:
        raise ValueError(f"{codec_dir} holds no quantizer.* tensors; is it a MOSS audio tokenizer?")
    digest = hashlib.sha256()
    by_file: Dict[str, List[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for shard, shard_names in by_file.items():
        with safe_open(str(codec_dir / shard), framework="pt") as f:
            for name in shard_names:
                tensors[name] = f.get_tensor(name)
    for name in names:
        t = tensors[name].contiguous()
        digest.update(f"{name}|{t.dtype}|{list(t.shape)}|".encode())
        digest.update(t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()[:32]


def load_processor(model_dir, codec_dir):
    """The reference's processor with an F32 codec. `MossTTSLocalProcessor.from_pretrained` would fetch
    the codec by Hub id at bf16; this builds the same object from local files at the dtype the export
    runs at. The codec's decoder is dropped: encoding never reaches it, and it is half the memory."""
    import torch
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    codec = transformers.AutoModel.from_pretrained(str(codec_dir), trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    codec.set_attention_implementation("sdpa")
    codec.set_compute_dtype("fp32")     # the checkpoint's bf16 would autocast on CPU
    codec.decoder = torch.nn.ModuleList()
    Proc = get_class_from_dynamic_module("processing_moss_tts.MossTTSLocalProcessor", str(model_dir))
    return Proc(tokenizer=tok, audio_tokenizer=codec, model_config=config)


def encode(processor, wavs: Sequence[Path]) -> List[np.ndarray]:
    """The reference's own encode of each clip -> one `[frames, n_vq]` int array per clip, batched the
    way the processor batches a message's references."""
    import torch

    with torch.no_grad():
        codes = processor.encode_audios_from_path([str(w) for w in wavs])
    return [c.numpy().astype(np.int64) for c in codes]


def voice_arrays(codes: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-reference `[frames, n_vq]` codes -> the two driver inputs."""
    if not codes:
        raise ValueError("a voice needs at least one reference")
    widths = {int(c.shape[1]) for c in codes}
    if len(widths) != 1 or any(c.ndim != 2 for c in codes):
        raise ValueError(f"every reference must be [frames, n_vq] with one n_vq; got "
                         f"{[tuple(c.shape) for c in codes]}")
    return {
        "reference_codes": np.concatenate([c.reshape(-1) for c in codes]).astype(np.float32),
        "reference_frames": np.array([c.shape[0] for c in codes], dtype=np.float32),
    }


def write_voice(arrays: Dict[str, np.ndarray], out: Path, *, name: str, compat: str, license: str,
                origin: str) -> int:
    """The two arrays -> one voice file. Returns the total reference length in frames."""
    from gguf import GGUFWriter

    missing = set(VOICE_INPUTS) - set(arrays)
    if missing:
        raise ValueError(f"a MOSS-TTS voice needs {sorted(VOICE_INPUTS)}; missing {sorted(missing)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    n_frames = int(arrays["reference_frames"].sum())
    w.add_uint32("loom.voice.n_references", int(arrays["reference_frames"].size))
    w.add_uint32("loom.voice.n_frames", n_frames)
    for key in VOICE_INPUTS:
        w.add_tensor(key, np.ascontiguousarray(arrays[key], dtype=np.float32).reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n_frames


def convert(model_dir, out_dir, *, wavs: Sequence[Path], name: str, license: str,
            codec_dir: Optional[Path] = None) -> Dict[str, dict]:
    """One or more clips -> `<out_dir>/<name>.gguf`."""
    if not (wavs and name and license):
        raise ValueError("a voice needs --wav (one per reference), --name and --license: the licence of "
                         "the recording, which only you know")
    codec = find_codec(model_dir, codec_dir)
    compat = codec_fingerprint(codec)
    codes = encode(load_processor(model_dir, codec), wavs)
    arrays = voice_arrays(codes)
    origin = ", ".join(str(w) for w in wavs)
    n = write_voice(arrays, Path(out_dir) / f"{name}.gguf", name=name, compat=compat, license=license,
                    origin=origin)
    return {name: {"n_frames": n, "frames": [int(c.shape[0]) for c in codes], "license": license}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the MOSS-TTS-Local-Transformer-v1.5 checkpoint directory")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--wav", type=Path, action="append", required=True,
                    help="a reference clip; repeat for several (one per speaker, in [S1], [S2] order)")
    ap.add_argument("--name", required=True, help="the voice's name")
    ap.add_argument("--license", required=True, help="the recordings' licence")
    ap.add_argument("--codec", type=Path, help="the MOSS-Audio-Tokenizer-v2 directory (default: the one "
                                               "beside the checkpoint, as its config names it)")
    args = ap.parse_args(argv)
    written = convert(args.model_dir, args.out, wavs=args.wav, name=args.name, license=args.license,
                      codec_dir=args.codec)
    for voice, info in written.items():
        print(f"{voice:16s} {info['n_frames']:4d} frames {info['frames']}  {info['license']}")
    print(f"wrote {len(written)} voice file(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
