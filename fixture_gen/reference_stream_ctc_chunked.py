"""Ground truth for a wav2vec 2.0-style CTC model streamed in the transformers pipeline's strided chunks
(`test_e2e_ctc_chunked_stream.cpp`, loom.cpp ADR-126): HuBERT, data2vec-audio, wav2vec 2.0.

The reference is `transformers.pipeline("automatic-speech-recognition", model=..., chunk_length_s=C,
stride_length_s=S)` itself, run here, not re-derived. Two hooks record what it does without changing it: the
model's forward (each chunk's input length and logits) and the tokenizer's `decode` (the concatenated kept
ids the pipeline merges). The pipeline's own text is the check that the hooks changed nothing.

Per clip and per `chunk/stride` configuration, in `<out_dir>/<clip>.<chunk>-<stride>/` (seconds, `%g`):

    ids.npy        (k,)       f32 -- the merged ids (the kept ids grouped, the pad/blank dropped)
    text.txt                      -- the pipeline's text
    steps.npy      (S, 4)     f32 -- per chunk: its first sample, its sample count, the first kept logit frame,
                                     the end of the kept frames (both after `rescale_stride`, the end clamped
                                     to the chunk's frames)
    logits_<s>.npy (k, V)     f32 -- chunk s's kept logit rows
    facts.txt                     -- chunk_len, stride_left, stride_right (samples), align, blank
    offline_ids.npy (k',)     f32 -- the one pass's greedy ids, for the record

    python fixture_gen/reference_stream_ctc_chunked.py <model_dir> <out_dir> \\
        <chunk_s/stride_s>[,...] <clip.npy|clip.wav> [...]
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).reshape(-1).astype(np.float32)
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32")
    if rate != 16000:
        raise SystemExit(f"{path} is {rate} Hz, not 16 kHz")
    return audio.reshape(-1)


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    from transformers import AutoModelForCTC, AutoProcessor, pipeline
    import transformers.pipelines.automatic_speech_recognition as asr_pipeline

    # The pipeline imports torchcodec whenever it is installed, only to recognise a decoder object as input;
    # a torchcodec whose CUDA library is missing then fails the import. Our input is an array either way.
    asr_pipeline.is_torchcodec_available = lambda: False

    model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    configs = [tuple(float(v) for v in c.split("/")) for c in sys.argv[3].split(",")]
    clips = [Path(p) for p in sys.argv[4:]]

    # Eager attention, as the export traces it; the pipeline at f32 on the CPU, batch 1.
    model = AutoModelForCTC.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, device="cpu")
    align = int(model.config.inputs_to_logits_ratio)
    blank = int(processor.tokenizer.pad_token_id)

    record = {"chunks": [], "items": None}
    forward = model.forward

    def hooked_forward(*args, **kwargs):
        out = forward(*args, **kwargs)
        x = kwargs.get("input_values", args[0] if args else None)
        record["chunks"].append((int(x.shape[-1]), out.logits[0].detach().float().numpy().copy()))
        return out

    decode = processor.tokenizer.decode

    def hooked_decode(items, *args, **kwargs):
        record["items"] = np.asarray(items).reshape(-1).copy()
        return decode(items, *args, **kwargs)

    model.forward = hooked_forward
    processor.tokenizer.decode = hooked_decode

    for clip in clips:
        audio = _load(clip)
        with torch.no_grad():
            offline = model(**processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")).logits
        offline_ids = [k for k, _ in itertools.groupby(offline[0].argmax(-1).tolist()) if k != blank]
        for chunk_s, stride_s in configs:
            record["chunks"], record["items"] = [], None
            text = pipe(audio.copy(), chunk_length_s=chunk_s, stride_length_s=stride_s)["text"]
            sr = pipe.feature_extractor.sampling_rate
            chunk_len = int(round(chunk_s * sr / align) * align)
            stride = int(round(stride_s * sr / align) * align)
            step = chunk_len - 2 * stride
            steps, kept_rows = [], []
            for s, (n, logits) in enumerate(record["chunks"]):
                start = s * step
                left = 0 if start == 0 else stride
                right = 0 if start + chunk_len >= len(audio) else stride
                # `rescale_stride` and the postprocess slice, as the pipeline computes them.
                token_n = int(round(n * (1 / align)))
                l_f = int(round(left / n * token_n))
                r_f = int(round(right / n * token_n))
                end = min(token_n - r_f, logits.shape[0])
                steps.append((start, n, l_f, end))
                kept_rows.append(logits[l_f:end])
            kept = np.concatenate([r.argmax(-1) for r in kept_rows]) if kept_rows else np.zeros(0)
            if record["items"] is None or not np.array_equal(kept, record["items"]):
                raise SystemExit(f"{clip.stem} {chunk_s}/{stride_s}: the recorded chunks do not rebuild the "
                                 f"ids the pipeline merged")
            ids = [k for k, _ in itertools.groupby(record["items"].tolist()) if k != blank]
            d = out_dir / f"{clip.stem}.{chunk_s:g}-{stride_s:g}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "ids.npy", np.asarray(ids, dtype=np.float32))
            np.save(d / "offline_ids.npy", np.asarray(offline_ids, dtype=np.float32))
            np.save(d / "steps.npy", np.asarray(steps, dtype=np.float32))
            for s, rows in enumerate(kept_rows):
                np.save(d / f"logits_{s}.npy", rows.astype(np.float32))
            (d / "text.txt").write_text(text)
            (d / "facts.txt").write_text(f"chunk_len {chunk_len}\nstride_left {stride}\nstride_right {stride}\n"
                                         f"align {align}\nblank {blank}\n")
            same = "== one pass" if ids == offline_ids else f"one pass differs ({len(offline_ids)} ids)"
            print(f"{d.name}: {len(steps)} chunks, {len(ids)} ids, {same}\n  {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
