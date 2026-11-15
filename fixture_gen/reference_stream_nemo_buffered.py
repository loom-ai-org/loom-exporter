"""Ground truth for NeMo's buffered transducer stream (`test_e2e_nemo_buffered_stream.cpp`): Parakeet TDT and
RNN-T (loom.cpp ADR-122) and Nemotron-Labs-Unified-ASR (ADR-110).

NeMo streams an offline transducer BUFFERED (`examples/asr/asr_chunked_inference/rnnt/
speech_to_text_streaming_infer_rnnt.py`, and the unified model's card gives the same recipe): every step runs
the whole model -- preprocessor, encoder, the prompt where the model takes one -- over a buffer of up to
`left + chunk + right` seconds and decodes the buffer's chunk frames with the stateful label-looping greedy
decoder. An encoder whose attention style is `chunked_limited_with_rc` (the unified model) runs under the mask
`set_default_att_context_size([left, chunk, right])` selects; any other runs its own full attention over the
buffer. This is that loop at batch 1, transcribed from the script at NeMo 3.0 / main (the two agree on
everything it uses), with NeMo's own `StreamingBatchedAudioBuffer` and decoding computer -- not a
re-derivation. The decoding config is the script's own default (`RNNTDecodingConfig()`, greedy_batch with
label looping, `max_symbols` 10), not the checkpoint's.

Per clip and per `left/chunk/right` configuration it writes, in
`<out_dir>/<clip>.<left>-<chunk>-<right>.<lang>/` (the sizes in seconds, as `%g`; `<lang>` is `none` for a
model without a language prompt):

    ids.npy       (k,)    f32 -- the streamed greedy ids (exact far below 2^24)
    text.txt                  -- and their text
    steps.npy     (S, 3)  f32 -- per step: buffer samples, left frames, frames decoded
    jumps.npy     (S,)    f32 -- per step: the TDT jump carried into the next (0 for RNN-T)
    rows_<s>.npy  (n, d)  f32 -- step s's decoded encoder rows (the prompt projection included)
    offline_ids.npy (k',) f32 -- the one pass's ids, for the record: buffered streaming is NOT the
                                 one pass, so these differ from `ids.npy` in general

Needs NeMo 3.0 (`~/.venvs/nemo3`).

    python fixture_gen/reference_stream_nemo_buffered.py <model.nemo> <out_dir> <lang|none> \\
        <left_s/chunk_s/right_s>[,...] <clip.npy|clip.wav> [...]
"""
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
    if len(sys.argv) < 6:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.nemo_asr_export import prepare_nemo_environment

    prepare_nemo_environment()
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer
    from omegaconf import OmegaConf, open_dict

    checkpoint, out_dir, lang = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    configs = [tuple(float(v) for v in c.split("/")) for c in sys.argv[4].split(",")]
    clips = [Path(p) for p in sys.argv[5:]]

    torch.set_grad_enabled(False)
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu").eval()
    # The script's own streaming settings: no dither, no padding to a multiple.
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    decoding = OmegaConf.structured(RNNTDecodingConfig())
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"
        decoding.greedy.loop_labels = True
        decoding.greedy.preserve_alignments = False
        decoding.fused_batch_size = -1
        decoding.beam.return_best_hypothesis = True
    model.change_decoding_strategy(decoding)
    computer = model.decoding.decoding.decoding_computer
    table = (model.cfg.get("model_defaults") or {}).get("prompt_dictionary")
    if lang == "none":
        if table:
            raise SystemExit("this model takes a language prompt; name one")
        prompt_kwargs = {}
    else:
        prompt_kwargs = {"prompt_indices": torch.tensor([int(dict(table)[lang])])}
    # The script's condition (`att_context_size_as_chunk`, on by default): only a chunked-with-right-context
    # encoder gets the mask; any other keeps its own attention.
    masked = model.cfg.encoder.get("att_context_style") == "chunked_limited_with_rc"
    print(f"decoding computer: {type(computer).__name__}; chunked mask: {masked}", flush=True)

    rate = int(model.cfg.preprocessor.sample_rate)
    stride = float(model.cfg.preprocessor.window_stride)
    factor = int(model.encoder.subsampling_factor)
    per_sec = 1.0 / stride
    feature_samples = int(rate * stride)
    if feature_samples % factor:
        raise SystemExit("the script rounds the hop to a multiple of the subsampling; this one is not")
    frame_samples = feature_samples * factor

    for left_s, chunk_s, right_s in configs:
        frames = ContextSize(left=int(left_s * per_sec / factor), chunk=int(chunk_s * per_sec / factor),
                             right=int(right_s * per_sec / factor))
        samples = ContextSize(left=frames.left * frame_samples, chunk=frames.chunk * frame_samples,
                              right=frames.right * frame_samples)
        if masked:
            model.encoder.set_default_att_context_size(
                att_context_size=[frames.left, frames.chunk, frames.right])
        tag = f"{left_s:g}-{chunk_s:g}-{right_s:g}"
        for clip in clips:
            audio = torch.from_numpy(_load(clip))[None]
            n = audio.shape[1]
            where = out_dir / f"{clip.stem}.{tag}.{lang}"
            where.mkdir(parents=True, exist_ok=True)

            hyps, state = None, None
            left_sample, right_sample = 0, min(samples.chunk + samples.right, n)
            buffer = StreamingBatchedAudioBuffer(batch_size=1, context_samples=samples, dtype=audio.dtype,
                                                 device="cpu")
            rest = torch.tensor([n])
            steps, jumps = [], []
            while left_sample < n:
                chunk_length = min(right_sample, n) - left_sample
                last_batch = chunk_length >= rest
                last = right_sample >= n
                lengths = torch.where(last_batch, rest, torch.full_like(rest, chunk_length))
                buffer.add_audio_batch_(audio[:, left_sample:right_sample], audio_lengths=lengths,
                                        is_last_chunk=last, is_last_chunk_batch=last_batch)
                out, out_len = model(input_signal=buffer.samples,
                                     input_signal_length=buffer.context_size_batch.total(), **prompt_kwargs)
                out = out.transpose(1, 2)
                context = buffer.context_size.subsample(factor=frame_samples)
                context_batch = buffer.context_size_batch.subsample(factor=frame_samples)
                out = out[:, context.left:]
                decode_len = torch.where(last_batch, out_len - context_batch.left, context_batch.chunk)
                chunk_hyps, state = computer(x=out, out_len=decode_len, prev_batched_state=state)
                if hyps is None:
                    hyps = chunk_hyps
                else:
                    hyps.merge_(chunk_hyps)
                k = int(decode_len[0])
                np.save(where / f"rows_{len(steps)}.npy", np.ascontiguousarray(out[0, :k].float().numpy()))
                steps.append((buffer.samples.shape[1], context.left, k))
                # TDT carries how far its last duration reached past the chunk (`time_jumps`); RNN-T none.
                jump = getattr(state, "time_jumps", None)
                jumps.append(int(jump[0]) if jump is not None else 0)
                rest -= lengths
                left_sample = right_sample
                right_sample = min(right_sample + samples.chunk, n)

            hyp = batched_hyps_to_hypotheses(hyps, batch_size=1)[0]
            ids = [int(t) for t in hyp.y_sequence]
            text = model.tokenizer.ids_to_text(ids)
            np.save(where / "ids.npy", np.asarray(ids, dtype=np.float32))
            np.save(where / "steps.npy", np.asarray(steps, dtype=np.float32))
            np.save(where / "jumps.npy", np.asarray(jumps, dtype=np.float32))
            (where / "text.txt").write_text(text)
            print(f"{clip.stem:24s} {tag:12s} {len(steps):3d} steps {len(ids):4d} ids "
                  f"{sum(j > 0 for j in jumps):3d} jumps carried: {text!r}", flush=True)
        # The one pass at full context (the encoder's restored default), for the record.
        if masked:
            model.encoder.set_default_att_context_size(att_context_size=[-1, -1, -1])
        for clip in clips:
            audio = torch.from_numpy(_load(clip))[None]
            enc, enc_len = model(input_signal=audio, input_signal_length=torch.tensor([audio.shape[1]]),
                                 **prompt_kwargs)
            hyp = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True)[0]
            np.save(out_dir / f"{clip.stem}.{tag}.{lang}" / "offline_ids.npy",
                    np.asarray([int(t) for t in hyp.y_sequence], dtype=np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
