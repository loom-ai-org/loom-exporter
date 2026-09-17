#!/usr/bin/env python3
"""Ground truth for a MIL-exported SANM/FunASR model (family 5, P5).

Runs FunASR's own modules -- the library, not this repo's exporter -- over a real waveform and writes
what an end-to-end check needs to localize a failure to a STAGE rather than only report one:

    ref_sanm_input.npy     (n_samples,)          f32  -- the waveform AS THE GRAPH RECEIVES IT (raw;
                                                         the fbank, the LFR stacking and the CMVN are
                                                         all inside the exported graph)
    ref_sanm_features.npy  (n_rows, n_mels*lfr_m) f32 -- `WavFrontend`'s output, which is the last
                                                         tensor before the prompt is prepended
    ref_sanm_encoder.npy   (n_rows + 4, d_model)  f32 -- the encoder's output, the tensor the CTC head
                                                         is the only thing standing after
    ref_sanm_logits.npy    (n_rows + 4, V)        f32 -- the CTC head's logits, THE oracle
    ref_sanm_ids.npy       (n_kept,)              i32 -- the greedy CTC decode of those logits
    ref_sanm_text.txt                                 -- what those ids detokenize to

**`dither` is forced to 0, and that is the one thing about this reference that cannot be skipped.**
`WavFrontend`'s own default is kaldi's, `1.0`, so FunASR's pipeline adds Gaussian noise to the waveform
before every fbank and does not produce the same features -- or necessarily the same emotion tag --
twice. A graph has no dither. Comparing against a dithered reference grades noise, and grades it at a
magnitude (~1e-3 on the logits) that is the same order as the accumulated f32 difference the comparison
is trying to measure.

**The logits are the oracle and the ids are the weaker check beside them**, the standing rule. A CTC
decode collapses repeats and drops the blank, so a frame whose argmax is wrong is invisible whenever its
neighbour already carried that label -- family 3 measured that directly, 71 of 80 tokens correct from a
wrong encoder.

**The two intermediates exist because this family's stages fail differently, and the first one is where
the work is.** `features` is the rebuilt front end: kaldi framing, a folded DC/pre-emphasis/window
matrix, a DFT written as two matmuls, kaldi's mel banks, and the LFR stacking. `encoder` is 70 SANM
blocks. A mismatch that starts at `features` is the front end, one that starts only at `encoder` is the
SANM stack or the position encoding, and one that starts only at `logits` is the head.

    python3 fixture_gen/reference_forward_sanm_asr_mil.py <model_dir> <wav> <out_dir> [max_seconds]
"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def read_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> float32 in [-1, 1], mono. No soundfile/torchaudio dependency: every fixture
    this repo ships is 16-bit mono PCM, and the reference must not depend on a resampler the exported
    graph does not have."""
    with wave.open(path) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {wav.getsampwidth() * 8}-bit")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        samples /= 32768.0
        if wav.getnchannels() > 1:
            samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1)
        return samples, wav.getframerate()


@torch.no_grad()
def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model_dir, wav_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

    from funasr import AutoModel
    from funasr.frontends.wav_frontend import apply_cmvn, apply_lfr
    from funasr.utils import fbank as kaldi

    auto = AutoModel(model=model_dir, device="cpu", disable_update=True)
    model, frontend = auto.model.eval(), auto.kwargs["frontend"]
    tokenizer = auto.kwargs["tokenizer"]

    samples, rate = read_wav_mono_f32(wav_path)
    if rate != int(frontend.fs):
        raise ValueError(
            f"{wav_path} is {rate} Hz and the checkpoint's frontend declares {int(frontend.fs)} Hz. "
            f"Resampling here would put a step in front of the reference that is not in front of the "
            f"graph.")
    samples = samples[: int(max_seconds * rate)]

    waveform = torch.from_numpy(samples)[None, :]
    scale = float(1 << 15) if frontend.upsacle_samples else 1.0
    mat = kaldi.fbank(waveform * scale, num_mel_bins=frontend.n_mels,
                      frame_length=frontend.frame_length, frame_shift=frontend.frame_shift,
                      dither=0.0, energy_floor=0.0, window_type=frontend.window,
                      sample_frequency=int(frontend.fs), snip_edges=True)
    features = apply_cmvn(apply_lfr(mat, frontend.lfr_m, frontend.lfr_n), frontend.cmvn)

    # The prompt `SenseVoiceSmall.inference` builds for `language="auto", use_itn=False`, in its own
    # order: the language query, the two fixed event/emotion rows, then the text-normalization query.
    prompt = torch.LongTensor([[model.lid_dict["auto"], 1, 2, model.textnorm_dict["woitn"]]])
    speech = torch.cat((model.embed(prompt), features[None, :, :]), dim=1)
    encoder_out, _ = model.encoder(speech, torch.LongTensor([speech.shape[1]]))
    logits = model.ctc.ctc_lo(encoder_out)[0]

    frames = logits.argmax(dim=-1)
    collapsed = torch.unique_consecutive(frames, dim=-1)
    ids = collapsed[collapsed != model.blank_id]
    text = tokenizer.decode(ids.tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ref_sanm_input.npy", samples.astype(np.float32))
    np.save(out_dir / "ref_sanm_features.npy", features.numpy().astype(np.float32))
    np.save(out_dir / "ref_sanm_encoder.npy", encoder_out[0].numpy().astype(np.float32))
    np.save(out_dir / "ref_sanm_logits.npy", logits.numpy().astype(np.float32))
    np.save(out_dir / "ref_sanm_ids.npy", ids.numpy().astype(np.int32))
    (out_dir / "ref_sanm_text.txt").write_text(text + "\n")
    print(f"{len(samples)} samples -> {tuple(features.shape)} features -> {tuple(logits.shape)} logits")
    print(f"{len(ids)} ids: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
