"""Ground truth for NeMo's buffered CTC stream (`test_e2e_nemo_ctc_buffered_stream.cpp`, loom.cpp ADR-123):
Conformer-CTC and Citrinet the way NeMo's `FrameBatchASR` transcribes them.

The reference is `examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py` (r2.6.0; the
CTC script left NeMo's examples in 3.0, and NeMo 3.0 dropped `get_buffered_pred_feat` with it), which drives
NeMo's own `FrameBatchASR` -- still in NeMo 2.6.2's `streaming_utils`, run here, not re-derived:

  * the clip is padded with `delay * model_stride` seconds of zeros and its raw log-mel computed ONCE
    (`AudioFeatureIterator`, normalisation off), then cut into `chunk` seconds of mel frames, the last
    partial chunk zero-filled;
  * each chunk slides into a `total`-second mel buffer (filled at the start with NeMo's
    `ZERO_LEVEL_SPEC_DB_VAL`), which is normalised per feature over itself and run through the model;
  * from each buffer's argmax ids it keeps `tokens_per_chunk` frames ending `delay` before its end, and
    one greedy CTC merge runs over the concatenation.

`model_stride` is the encoder's own `subsampling_factor`. The script read it from `asr_model.subsampling_factor`
from r2.4 on, which no CTC model has (an AttributeError); up to r2.3 it was a parameter the docstring sets to 4
for Conformer and 8 for Citrinet -- the same numbers. Batch 1 (the script batches buffers of one length, so
batching changes nothing but rounding).

Per clip and per `chunk/total` configuration, in `<out_dir>/<clip>.<chunk>-<total>/` (seconds, `%g`):

    ids.npy        (k,)       f32 -- the merged greedy ids
    text.txt                      -- and their text
    mel.npy        (F, n_mel) f32 -- the raw log-mel of the padded clip, frames NeMo counts
    steps.npy      (S, 2)     f32 -- per buffer: its encoder frame count, the first kept frame
    logits_<s>.npy (k, V)     f32 -- buffer s's kept log-prob rows
    facts.txt                     -- tokens_per_chunk, delay, model_stride, the buffer's fill value
    offline_ids.npy (k',)     f32 -- the one pass's greedy ids, for the record

Needs NeMo 2.6 (`~/.venvs/piper`).

    python fixture_gen/reference_stream_nemo_ctc_buffered.py <model.nemo> <out_dir> \\
        <chunk_s/total_s>[,...] <clip.npy|clip.wav> [...]
"""
import math
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.nemo_asr_export import prepare_nemo_environment

    prepare_nemo_environment()
    import copy

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
    from omegaconf import OmegaConf

    checkpoint, out_dir = sys.argv[1], Path(sys.argv[2])
    configs = [tuple(float(v) for v in c.split("/")) for c in sys.argv[3].split(",")]
    clips = [Path(p) for p in sys.argv[4:]]

    torch.set_grad_enabled(False)
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu").eval()
    # The script's `model_cfg` edits (no dither, no pad) -- FrameBatchASR makes the same ones on its own copy.
    model_cfg = copy.deepcopy(model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0
    if model_cfg.preprocessor.normalize != "per_feature":
        raise SystemExit("the script supports per_feature normalisation only")
    feature_stride = float(model_cfg.preprocessor["window_stride"])
    model_stride = int(model.encoder.subsampling_factor)
    model_stride_in_secs = feature_stride * model_stride

    for chunk_len, total_buffer in configs:
        tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
        mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
        tag = f"{chunk_len:g}-{total_buffer:g}"
        for clip in clips:
            where = out_dir / f"{clip.stem}.{tag}"
            where.mkdir(parents=True, exist_ok=True)
            asr = FrameBatchASR(asr_model=model, frame_len=chunk_len, total_buffer=total_buffer, batch_size=1)
            asr.reset()
            # `read_audio_file`, with the clip in hand rather than a path.
            samples = _load(clip)
            samples = np.pad(samples, (0, int(mid_delay * model_stride_in_secs * model._cfg.sample_rate)))
            reader = AudioFeatureIterator(samples, asr.frame_len, asr.raw_preprocessor, model.device)
            asr.set_frame_reader(reader)
            hypothesis, logits = asr.transcribe(tokens_per_chunk, mid_delay, keep_logits=True)

            # The merged ids, by `greedy_merge`'s own rule, before they become text.
            ids, previous = [], asr.blank_id
            for p in asr.unmerged:
                if (p != previous or previous == asr.blank_id) and p != asr.blank_id:
                    ids.append(int(p))
                previous = p
            assert model.tokenizer.ids_to_text(ids) == hypothesis
            np.save(where / "ids.npy", np.asarray(ids, dtype=np.float32))
            (where / "text.txt").write_text(hypothesis)
            n = int(reader._features_len[0])
            np.save(where / "mel.npy", np.ascontiguousarray(reader._features[:, :n].T.float().cpu().numpy()))
            steps = []
            for s, log_prob in enumerate(asr.all_logits):
                t = int(log_prob.shape[0])
                first = t - 1 - mid_delay
                steps.append((t, first))
                np.save(where / f"logits_{s}.npy",
                        np.ascontiguousarray(log_prob[first:first + tokens_per_chunk].float().numpy()))
            np.save(where / "steps.npy", np.asarray(steps, dtype=np.float32))
            (where / "facts.txt").write_text(
                f"tokens_per_chunk {tokens_per_chunk}\ndelay {mid_delay}\nmodel_stride {model_stride}\n"
                f"fill {asr.frame_bufferer.ZERO_LEVEL_SPEC_DB_VAL}\nblank {asr.blank_id}\n")
            # The one pass, for the record (buffered streaming is not it).
            audio = torch.from_numpy(_load(clip))[None]
            log_probs, _, _ = model(input_signal=audio, input_signal_length=torch.tensor([audio.shape[1]]))
            frames, offline, previous = log_probs[0].argmax(-1).tolist(), [], asr.blank_id
            for p in frames:
                if p != previous and p != asr.blank_id:
                    offline.append(p)
                previous = p
            np.save(where / "offline_ids.npy", np.asarray(offline, dtype=np.float32))
            print(f"{clip.stem:24s} {tag:10s} {len(steps):3d} buffers {len(ids):4d} ids "
                  f"(one pass {len(offline)}; fill {asr.frame_bufferer.ZERO_LEVEL_SPEC_DB_VAL}): {hypothesis!r}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
