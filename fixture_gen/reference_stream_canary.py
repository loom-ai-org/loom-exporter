"""Ground truth for Canary's live stream (`test_e2e_canary_stream.cpp`, loom.cpp ADR-124): wait-k and AlignAtt, the
way NeMo's `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py` (NVIDIA-NeMo/Speech main)
runs them.

That script runs the whole model over a buffer of up to `left + chunk + right` seconds (`StreamingBatchedAudioBuffer`,
10 + 2 + 2 by default), masks the decoder's cross-attention to the left context and the chunk (the right context
too on the last chunk), and decodes with `GreedyBatchedStreamingAEDComputer`, its decoder cache carried across
chunks. This is that loop at batch 1, transcribed from the script, with NeMo's own buffer, state and computer -- not a
re-derivation. The decoding config is the script's default `AEDStreamingDecodingConfig` with only the policy set.
The prompt is the script's: `return_decoder_input_ids` with the default dialog slots (both turns, 13 ids), the two
language slots set.

Per clip, policy, `left/chunk/right` and languages, in `<out_dir>/<clip>.<policy>.<L>-<C>-<R>.<src>-<tgt>/`:

    ids.npy        (k,)       f32 -- the transcript ids (`pred_tokens_ids[P:ccl]`)
    text.txt                      -- and their text
    prompt.npy     (P,)       f32 -- the decoder prompt
    steps.npy      (S, 6)     f32 -- per step: buffer samples, buffer length (the batch bookkeeping's), encoder rows
                                     the decoder sees, prev_encoder_shift, last chunk, context length after it
    rows_<s>.npy   (n, d)     f32 -- step s's encoder rows the decoder sees
    first.npy      (2,)       f32 -- the first decoding step's index and its first token
    xatt.npy       (P, n)     f32 -- that step's first forward: layer -2 cross-attention, mean over heads
    logits.npy     (V,)       f32 -- and its last row's logits

Needs NeMo with `aed_decoding` (2.6.2 in `~/.venvs/piper` has it, identical to 3.0's but for LAAL).

    python fixture_gen/reference_stream_canary.py <model.nemo> <out_dir> <waitk|alignatt> \\
        <left_s/chunk_s/right_s>[,...] <src> <tgt> <clip.npy|clip.wav> [...]
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
    if len(sys.argv) < 8:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.nemo_asr_export import prepare_nemo_environment

    prepare_nemo_environment()
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models.aed_multitask_models import lens_to_mask
    from nemo.collections.asr.parts.submodules.aed_decoding import GreedyBatchedStreamingAEDComputer
    from nemo.collections.asr.parts.submodules.multitask_decoding import (
        AEDStreamingDecodingConfig, MultiTaskDecodingConfig,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

    checkpoint, out_dir, policy = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    configs = [tuple(float(v) for v in c.split("/")) for c in sys.argv[4].split(",")]
    src, tgt = sys.argv[5], sys.argv[6]
    clips = [Path(p) for p in sys.argv[7:]]

    torch.set_grad_enabled(False)
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu").eval()
    model.freeze()
    multitask = MultiTaskDecodingConfig()
    multitask.strategy = "greedy"
    model.change_decoding_strategy(multitask)
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # `return_decoder_input_ids` with no prompt override: the default slots of every turn, languages set.
    turns = model.prompt.get_default_dialog_slots()
    for turn in turns:
        if "source_lang" in turn["slots"]:
            turn["slots"]["source_lang"] = f"<|{src}|>"
            turn["slots"]["target_lang"] = f"<|{tgt}|>"
    prompt = model.prompt.encode_dialog(turns=turns)["context_ids"].unsqueeze(0)
    n_prompt = prompt.shape[1]

    decoding = AEDStreamingDecodingConfig()
    decoding.streaming_policy = policy
    rate = int(model.cfg.preprocessor.sample_rate)
    stride = float(model.cfg.preprocessor.window_stride)
    factor = int(model.encoder.subsampling_factor)
    feature_samples = int(rate * stride)
    if feature_samples % factor:
        raise SystemExit("the script rounds the hop to a multiple of the subsampling; this one is not")
    frame = feature_samples * factor
    per_sec = 1.0 / stride
    greedy = model.decoding.decoding.greedy_search

    for left_s, chunk_s, right_s in configs:
        frames = ContextSize(left=int(left_s * per_sec / factor), chunk=int(chunk_s * per_sec / factor),
                             right=int(right_s * per_sec / factor))
        samples = ContextSize(left=frames.left * frame, chunk=frames.chunk * frame, right=frames.right * frame)
        tag = f"{policy}.{left_s:g}-{chunk_s:g}-{right_s:g}.{src}-{tgt}"
        for clip in clips:
            audio = torch.from_numpy(_load(clip))[None]
            n = audio.shape[1]
            where = out_dir / f"{clip.stem}.{tag}"
            where.mkdir(parents=True, exist_ok=True)
            computer = GreedyBatchedStreamingAEDComputer(model, frame_chunk_size=frames.chunk, decoding_cfg=decoding)
            state = GreedyBatchedStreamingAEDComputer.initialize_aed_model_state(
                asr_model=model, decoder_input_ids=prompt, batch_size=1, context_encoder_frames=frames,
                chunk_secs=chunk_s, right_context_secs=right_s)
            left_sample = 0
            end_of_window = min(samples.chunk, n)
            right_sample = min(samples.chunk + samples.right, n)
            buffer = StreamingBatchedAudioBuffer(batch_size=1, context_samples=samples, dtype=audio.dtype,
                                                 device="cpu")
            rest = torch.tensor([n])
            steps, first = [], None
            while left_sample < n:
                chunk_length = min(right_sample, n) - left_sample
                last_batch = chunk_length >= rest
                last = right_sample >= n
                lengths = torch.where(last_batch, rest, torch.full_like(rest, chunk_length))
                buffer.add_audio_batch_(audio[:, left_sample:right_sample], audio_lengths=lengths,
                                        is_last_chunk=last, is_last_chunk_batch=torch.zeros_like(last_batch))
                state.is_last_chunk_batch = last_batch
                _, encoded_len, enc_states, _ = model(input_signal=buffer.samples,
                                                      input_signal_length=buffer.context_size_batch.total())
                context_batch = buffer.context_size_batch.subsample(factor=frame)
                no_rc = context_batch.left + context_batch.chunk
                corrected = torch.where(last_batch, encoded_len, no_rc)
                mask = lens_to_mask(corrected, enc_states.shape[1]).to(enc_states.dtype)
                state.prev_encoder_shift = max(end_of_window // frame - frames.left - frames.chunk, 0)
                k = int(corrected[0])
                s = len(steps)
                started = state.decoding_step >= 0
                state = computer(encoder_output=enc_states, encoder_output_len=corrected, encoder_input_mask=mask,
                                 prev_batched_state=state)
                if first is None and state.decoding_step >= 0 and not started:
                    # The step that decoded first: its first forward again, the prompt over this step's rows.
                    logits, _, xatt = greedy._one_step_forward(prompt, enc_states, mask, None,
                                                               torch.zeros(1, dtype=torch.long), return_scores=False)
                    np.save(where / "xatt.npy", np.ascontiguousarray(xatt[-2].mean(1)[0, :, :k].float().numpy()))
                    np.save(where / "logits.npy", np.ascontiguousarray(logits[0, -1].float().numpy()))
                    first = (s, int(state.pred_tokens_ids[0, n_prompt]))
                np.save(where / f"rows_{s}.npy", np.ascontiguousarray(enc_states[0, :k].float().numpy()))
                steps.append((buffer.samples.shape[1], int(buffer.context_size_batch.total()[0]), k,
                              state.prev_encoder_shift, int(last), int(state.current_context_lengths[0])))
                rest -= lengths
                left_sample = right_sample
                right_sample = min(right_sample + samples.chunk, n)
                end_of_window += samples.chunk

            ccl = int(state.current_context_lengths[0])
            ids = [int(t) for t in state.pred_tokens_ids[0, n_prompt:ccl]]
            text = model.tokenizer.ids_to_text(ids).strip()
            np.save(where / "ids.npy", np.asarray(ids, dtype=np.float32))
            np.save(where / "prompt.npy", np.asarray(prompt[0].tolist(), dtype=np.float32))
            np.save(where / "steps.npy", np.asarray(steps, dtype=np.float32))
            np.save(where / "first.npy", np.asarray(first if first else (-1, -1), dtype=np.float32))
            (where / "text.txt").write_text(text)
            print(f"{clip.stem:24s} {tag:28s} {len(steps):3d} steps {len(ids):4d} ids: {text!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
