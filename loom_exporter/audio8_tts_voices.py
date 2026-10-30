"""Audio8-TTS voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

Audio8 clones a voice from one reference clip AND its exact transcript: the clone prompt's system turn
carries the transcript as text and the clip as codec frames (`processing_arktts._prompt_segments`,
`modeling_arktts._prepare_prompt`). This tool runs the reference's own encode path ONCE, in Python --
mono mix, resample to 44.1 kHz, the codec's `encode` at F32 -- tokenizes the transcript the way the
processor formats it, and writes both as a **voice file** whose tensors are DRIVER INPUTS by name
(loom.cpp ADR-045), so the codec's encoder never has to reach the engine:

    python -m loom_exporter.audio8_tts_voices ~/Dev/models/audio8-tts-realistic-0.6b -o voices \\
        --wav me.wav --text "The exact words of the clip." --name me --license "CC0-1.0"

**The transcript is tokenized HERE, with the speaker prefix, rather than by the host.** The processor
writes `<|speaker:0|>` in front of the transcript and encodes the two as ONE string; the prefix is not
a special token in this vocabulary, so its closing `|>` can merge with a transcript that opens with
punctuation. Tokenized here, the ids are the processor's exactly, whatever the transcript starts with.

**A voice fits the CODEC, not the TTS weights.** Codes mean what the codec's quantizers say they mean,
so every file is stamped with `loom.voice.compat`, a fingerprint of the two residual quantizers in
`codec.pth`, and the TTS export declares the fingerprint of the codec it ships beside.

**The licence is the recording's**, which only whoever made the clip knows, so `--license` is required.
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "audio8_tts"
# The driver inputs a voice sets: the formatted transcript's ids, and the clip's codes frame-major.
VOICE_INPUTS = ("reference_tokens", "reference_codes")
# What `codec_fingerprint` hashes: the residual quantizers, which are what decide what a code means.
QUANTIZER_PREFIXES = ("quantizer.semantic_quantizer.", "quantizer.quantizer.")

_FINGERPRINTS: Dict[str, str] = {}


def codec_state(model_dir) -> dict:
    """`codec.pth`'s state dict as `ArkttsModel.load_codec` reads it: unwrapped from `state_dict`, the
    `generator.` prefix stripped, the derived `freqs_cis`/`causal_mask` buffers dropped. `weights_only`,
    so the pickle can hold tensors and nothing else."""
    import torch

    state = torch.load(str(Path(model_dir) / "codec.pth"), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {key.replace("generator.", ""): value for key, value in state.items() if "generator." in key}
    return {k: v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}


def load_codec(model_dir):
    """The checkpoint's own `ArkttsCodec` at F32, from its remote code and `codec.pth`, strict."""
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    Codec = get_class_from_dynamic_module("modeling_arktts_codec.ArkttsCodec", str(model_dir))
    codec = Codec(config)
    codec.load_state_dict(codec_state(model_dir), strict=True)
    return codec.float().eval()


def codec_fingerprint(model_dir) -> str:
    """sha256 over the codec's quantizer tensors -- name, dtype, shape and bytes, in name order. The
    encoder, decoder and the transformers around the quantizers are left out: a code is a quantizer's
    index, and a codec re-trained around the same codebooks still means the same thing by it."""
    import torch

    key = str(Path(model_dir).resolve())
    if key in _FINGERPRINTS:
        return _FINGERPRINTS[key]
    state = codec_state(model_dir)
    names = sorted(k for k in state if k.startswith(QUANTIZER_PREFIXES))
    if not names:
        raise ValueError(f"{model_dir}/codec.pth holds no {QUANTIZER_PREFIXES} tensors; is it an Audio8 codec?")
    digest = hashlib.sha256()
    for name in names:
        t = state[name].contiguous()
        digest.update(f"{name}|{t.dtype}|{list(t.shape)}|".encode())
        digest.update(t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    _FINGERPRINTS[key] = digest.hexdigest()[:32]
    return _FINGERPRINTS[key]


def reference_tokens(model_dir, transcript: str) -> np.ndarray:
    """The transcript as the processor formats and encodes it (`_format_reference_text`)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    Proc = get_class_from_dynamic_module("processing_arktts.ArkttsProcessor", str(model_dir))
    formatted = Proc._format_reference_text(transcript)
    if not formatted.strip():
        raise ValueError("the reference transcript is empty; Audio8 needs the exact words of the clip")
    return np.array(tokenizer.encode(formatted, add_special_tokens=False), dtype=np.int64)


def encode(model_dir, wav: Path) -> np.ndarray:
    """The reference's own encode of one clip -> `[frames, n_codebooks]` int codes: the processor's
    `_load_audio` (mono mix, resample to the codec's rate) then `ArkttsCodec.encode`."""
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    Proc = get_class_from_dynamic_module("processing_arktts.ArkttsProcessor", str(model_dir))
    codec = load_codec(model_dir)
    audio = Proc(tokenizer=tokenizer, audio_sampling_rate=int(codec.sample_rate))._load_audio(str(wav), None)
    with torch.inference_mode():
        codes, lengths = codec.encode(audio[None, None], torch.tensor([audio.numel()]))
    return codes[0, :, :int(lengths[0])].T.numpy().astype(np.int64)


def write_voice(arrays: Dict[str, np.ndarray], out: Path, *, name: str, compat: str, license: str,
                origin: str) -> int:
    """The two arrays -> one voice file. Returns the reference length in frames."""
    from gguf import GGUFWriter

    missing = set(VOICE_INPUTS) - set(arrays)
    if missing:
        raise ValueError(f"an Audio8 voice needs {sorted(VOICE_INPUTS)}; missing {sorted(missing)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    codes = arrays["reference_codes"]
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    w.add_uint32("loom.voice.n_references", 1)
    w.add_uint32("loom.voice.n_frames", int(codes.shape[0]))
    for key in VOICE_INPUTS:
        w.add_tensor(key, np.ascontiguousarray(arrays[key], dtype=np.float32).reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return int(codes.shape[0])


def convert(model_dir, out_dir, *, wav: Path, text: str, name: str, license: str) -> dict:
    """One clip and its transcript -> `<out_dir>/<name>.gguf`."""
    if not (wav and text and name and license):
        raise ValueError("a voice needs --wav, --text (the clip's exact transcript), --name and "
                         "--license: the licence of the recording, which only you know")
    arrays = {"reference_tokens": reference_tokens(model_dir, text), "reference_codes": encode(model_dir, wav)}
    n = write_voice(arrays, Path(out_dir) / f"{name}.gguf", name=name, compat=codec_fingerprint(model_dir),
                    license=license, origin=str(wav))
    return {"n_frames": n, "n_tokens": int(arrays["reference_tokens"].size), "license": license}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the Audio8-TTS checkpoint directory (with codec.pth)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--wav", type=Path, required=True, help="the reference clip")
    ap.add_argument("--text", required=True, help="its exact transcript")
    ap.add_argument("--name", required=True, help="the voice's name")
    ap.add_argument("--license", required=True, help="the recording's licence")
    args = ap.parse_args(argv)
    info = convert(args.model_dir, args.out, wav=args.wav, text=args.text, name=args.name, license=args.license)
    print(f"{args.name:16s} {info['n_frames']:4d} frames {info['n_tokens']:4d} tokens  {info['license']}")
    print(f"wrote {args.out / (args.name + '.gguf')}", file=sys.stderr)


if __name__ == "__main__":
    main()
