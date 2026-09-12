#!/usr/bin/env python3
"""Ground truth for a MIL-exported CNN+transformer+CTC model (family 4, P5).

Runs HF's own `AutoModelForCTC` -- the library, not this repo's exporter -- over a real waveform and
writes what an end-to-end check needs to localize a failure to a STAGE rather than only report one:

    ref_ctc_input.npy      (n_samples,)        f32  -- the waveform AS THE GRAPH RECEIVES IT (raw; the
                                                       normalization is inside the exported graph)
    ref_ctc_features.npy   (n_frames, d_conv)  f32  -- the convolutional feature encoder's output
    ref_ctc_hidden.npy     (n_frames, d_model) f32  -- the feature projection's output, which is the
                                                       last tensor before the positional convolution
    ref_ctc_logits.npy     (n_frames, V)       f32  -- the CTC head's logits, THE oracle
    ref_ctc_ids.npy        (n_kept,)           i32  -- the greedy CTC decode of those logits
    ref_ctc_text.txt                                -- what those ids detokenize to

**The logits are the oracle and the ids are the weaker check beside them**, which is the standing rule
this family has its own reason to follow: a CTC decode collapses repeats and drops the blank, so a
frame whose argmax is wrong is invisible whenever its neighbour already carried that label. Family 3
measured that directly -- a wrong encoder decoded 71 of 80 tokens correctly.

**The three intermediates exist because this family's stages fail differently.** The feature encoder is
seven dense convolutions over raw audio; the projection is one linear; the positional convolution is
GROUPED, which is the one thing here that no other family in the zoo has. A mismatch that starts at
`features` is the convolution stack, one that starts at `hidden` is the projection, and one that starts
only at `logits` is the encoder or the head -- and the grouped convolution sits exactly at that last
boundary, which is where this family's first real bug was (loom.cpp Retro-046).

    python3 fixture_gen/reference_forward_ctc_asr_mil.py <model_dir> <wav> <out_dir> [max_seconds]
"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer


def read_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> float32 in [-1, 1], mono. No soundfile/torchaudio dependency: every fixture
    this repo ships is 16-bit mono PCM, and the reference must not depend on a resampler the exported
    graph does not have."""
    with wave.open(path) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {wav.getsampwidth() * 8}-bit")
        frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if wav.getnchannels() > 1:
            samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1)
        return samples, wav.getframerate()


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model_dir, wav_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    # `eager`, matching the export. The numbers are the same either way; what differs is that the sdpa
    # path builds a mask of the traced size, which is the whole reason the export asks for eager.
    model = AutoModelForCTC.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)

    samples, rate = read_wav_mono_f32(wav_path)
    if rate != int(extractor.sampling_rate):
        raise ValueError(f"{wav_path} is {rate} Hz; this checkpoint declares "
                         f"{extractor.sampling_rate} Hz and the graph does not resample")
    if max_seconds:
        samples = samples[: int(max_seconds * rate)]

    waveform = torch.from_numpy(samples.copy())[None]
    # The SAME normalization the exported graph performs, applied here rather than through the
    # processor -- see `ctc_asr_export`. Skipped for a checkpoint that declares `do_normalize: false`,
    # for which the graph also skips it.
    if bool(getattr(extractor, "do_normalize", True)):
        waveform = (waveform - waveform.mean(dim=-1, keepdim=True)) / torch.sqrt(
            waveform.var(dim=-1, keepdim=True, unbiased=False) + 1e-7)

    base = model.base_model
    with torch.no_grad():
        features = base.feature_extractor(waveform).transpose(1, 2)
        hidden = base.feature_projection(features)
        # `feature_projection` returns (hidden, extracted) for wav2vec2/data2vec and a bare tensor for
        # HuBERT -- read structurally rather than by model_type, exactly as the export's wrapper reads
        # its arguments off a signature.
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        logits = model(input_values=waveform).logits

    np.save(out_dir / "ref_ctc_input.npy", samples.astype(np.float32))
    np.save(out_dir / "ref_ctc_features.npy", features[0].numpy().astype(np.float32))
    np.save(out_dir / "ref_ctc_hidden.npy", hidden[0].numpy().astype(np.float32))
    np.save(out_dir / "ref_ctc_logits.npy", logits[0].numpy().astype(np.float32))

    # The driver's own epilogue, in numpy: per-frame argmax, drop consecutive duplicates, drop the
    # blank. The blank is the tokenizer's pad id, NOT the last class -- see `ctc_asr_export`.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    blank = int(tokenizer.pad_token_id)
    frames = logits[0].argmax(dim=-1).tolist()
    kept, previous = [], blank
    for token_id in frames:
        if token_id != previous and token_id != blank:
            kept.append(token_id)
        previous = token_id
    np.save(out_dir / "ref_ctc_ids.npy", np.asarray(kept, dtype=np.int32))
    # `group_tokens=False` because the collapse ALREADY HAPPENED above. HF's CTC decode does both jobs
    # in one call and defaults to grouping, so letting it group a second time silently deletes every
    # genuine double letter -- "FELLOW" comes back as "FELOW", which reads as a plausible ASR error
    # rather than as a bug in the oracle. Caught by the engine disagreeing with the reference in the
    # one direction that made the ENGINE look wrong.
    text = tokenizer.decode(kept, group_tokens=False)
    (out_dir / "ref_ctc_text.txt").write_text(text + "\n")

    print(f"{len(samples)} samples -> {logits.shape[1]} frames x {logits.shape[2]} classes, "
          f"blank={blank}")
    print(f"text: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
