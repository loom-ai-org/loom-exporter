"""CosyVoice3 voices as loom voice files: `voices/<name>.gguf`, loadable by name or path.

CosyVoice3 clones a voice zero-shot from a clip and its transcript. The reference turns the pair into
four arrays (`CosyVoiceFrontEnd.frontend_zero_shot`) -- the transcript's ids, the clip's S3 speech
tokens, its 80-bin mel, and a CAMPPlus speaker embedding -- and two of those need models the release
ships ONLY as ONNX (`speech_tokenizer_v3.onnx`, `campplus.onnx`). This tool runs that front end ONCE,
in Python, and writes the four arrays as a **voice file**: a tiny GGUF whose tensors are DRIVER INPUTS
by name (`prompt_text`, `prompt_speech_tokens`, `prompt_feat`, `embedding`), which `loom::load_voice`
reads for both hosts -- `loom_cli --voice`, and loom-py's `text2speech.infer(text, voice="me")`. So
cloning costs Python and onnxruntime once per voice, and neither ONNX model reaches the engine
(loom.cpp ADR-045). The export ships the same arrays for its default voice as driver weights.

**A voice only fits the weights it was made with.** Its speech tokens and mel are read by the LM and
the flow, so every file is stamped with `loom.voice.compat`, a fingerprint of `llm.pt` and `flow.pt`,
and the model GGUF declares the same key; `loom::load_voice` refuses a mismatch by name.

    python -m loom_exporter.cosyvoice3_voices ~/Dev/models/fun-cosyvoice3-0.5b-2512 -o voices \\
        --wav me.wav --text "What the clip says." --name me --license "CC0-1.0"
    python -m loom_exporter.cosyvoice3_voices <model_dir> -o voices --default   # the built-in voice

`--text` is the clip's transcript. The reference's own prompts prefix it with the system line
`You are a helpful assistant.<|endofprompt|>`, and so does this tool unless the text already holds
`<|endofprompt|>` (then it is used as given: the LM needs that token somewhere in the prompt).

**The licence is the recording's**, which only whoever made the clip knows, so `--license` is required
for your own clips. The clip must be at most 30 s (the reference's speech tokenizer refuses longer) and
at least 16 kHz.
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np

VOICE_FILE_ARCH = "loom-voice"
MODEL_ARCH = "cosyvoice3"
SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_VOICE_NAME = "zero_shot_prompt"
DEFAULT_VOICE_LICENSE = ("unstated: ships in FunAudioLLM/CosyVoice (Apache-2.0) as asset/zero_shot_prompt.wav, "
                         "whose README says some examples are sourced from the internet; check before "
                         "redistributing")
# The export's driver weights are `voice.<input>`; a voice file names the INPUT itself.
VOICE_INPUTS = ("prompt_text", "prompt_speech_tokens", "prompt_feat", "embedding")


def weights_fingerprint(model_dir) -> str:
    """sha256 over every tensor of `llm.pt` and `flow.pt` -- name, dtype, shape and bytes, in name order
    -- which is what a voice depends on: the LM reads its text and speech tokens, the flow its tokens,
    mel and embedding. HiFT never sees a voice and is left out."""
    import torch

    digest = hashlib.sha256()
    for part in ("llm", "flow"):
        state = torch.load(Path(model_dir) / f"{part}.pt", map_location="cpu", weights_only=True, mmap=True)
        for name in sorted(state):
            t = state[name].contiguous()
            digest.update(f"{part}.{name}|{t.dtype}|{list(t.shape)}|".encode())
            digest.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        del state
    return digest.hexdigest()[:32]


def prompt_text(transcript: str) -> str:
    return transcript if "<|endofprompt|>" in transcript else SYSTEM_PROMPT + transcript


def compute(model_dir, wav: Path, transcript: str) -> Dict[str, np.ndarray]:
    """The reference's own front end on one clip -> the four driver inputs, by input name."""
    from .cosyvoice3_export import build_frontend, compute_voice, import_cosyvoice

    import_cosyvoice()
    arrays = compute_voice(build_frontend(str(model_dir)), str(wav), prompt_text(transcript))
    return {name.removeprefix("voice."): values for name, values in arrays.items()}


def write_voice(arrays: Dict[str, np.ndarray], out: Path, *, name: str, compat: str, license: str,
                origin: str) -> int:
    """Four arrays -> one voice file. Returns the prompt's length in speech tokens."""
    from gguf import GGUFWriter

    missing = set(VOICE_INPUTS) - set(arrays)
    if missing:
        raise ValueError(f"a CosyVoice3 voice needs {sorted(VOICE_INPUTS)}; missing {sorted(missing)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), VOICE_FILE_ARCH)
    w.add_string("loom.voice.architecture", MODEL_ARCH)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", license)
    w.add_string("loom.voice.origin", origin)
    n_tokens = int(arrays["prompt_speech_tokens"].size)
    w.add_uint32("loom.voice.n_prompt_tokens", n_tokens)
    for key in VOICE_INPUTS:
        w.add_tensor(key, np.ascontiguousarray(arrays[key], dtype=np.float32).reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n_tokens


def convert(model_dir, out_dir, *, wav=None, text=None, name=None, license=None, default=False) -> Dict[str, dict]:
    """The export's default voice (`default=True`) or one clip + transcript, written under `out_dir`."""
    from .cosyvoice3_export import COSYVOICE_REPO, DEFAULT_VOICE_TEXT, DEFAULT_VOICE_WAV

    if default:
        jobs = {DEFAULT_VOICE_NAME: (Path(COSYVOICE_REPO) / DEFAULT_VOICE_WAV, DEFAULT_VOICE_TEXT,
                                     DEFAULT_VOICE_LICENSE, f"FunAudioLLM/CosyVoice:{DEFAULT_VOICE_WAV}")}
    else:
        if not (wav and text and name and license):
            raise ValueError("a voice from your own clip needs --wav, --text (what the clip says), --name "
                             "and --license: the licence of the recording, which only you know")
        jobs = {name: (Path(wav), text, license, str(wav))}
    compat = weights_fingerprint(model_dir)
    written = {}
    for voice, (clip, transcript, licence, origin) in jobs.items():
        arrays = compute(model_dir, clip, transcript)
        n = write_voice(arrays, Path(out_dir) / f"{voice}.gguf", name=voice, compat=compat, license=licence,
                        origin=origin)
        written[voice] = {"n_prompt_tokens": n, "license": licence, "origin": origin}
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="the Fun-CosyVoice3-0.5B-2512 checkpoint directory")
    ap.add_argument("-o", "--out", type=Path, required=True, help="directory for <name>.gguf")
    ap.add_argument("--default", action="store_true", help=f"write the export's default voice "
                                                            f"({DEFAULT_VOICE_NAME})")
    ap.add_argument("--wav", type=Path, help="the clip to clone (<= 30 s, >= 16 kHz)")
    ap.add_argument("--text", help="what the clip says")
    ap.add_argument("--name", help="the voice's name")
    ap.add_argument("--license", help="the recording's licence")
    args = ap.parse_args(argv)
    written = convert(args.model_dir, args.out, wav=args.wav, text=args.text, name=args.name,
                      license=args.license, default=args.default)
    for voice, info in written.items():
        print(f"{voice:16s} {info['n_prompt_tokens']:4d} prompt tokens  {info['license']}")
    print(f"wrote {len(written)} voice file(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
