"""Ground truth for the exported Nemotron-Labs-Unified-ASR (`test_e2e_nemotron_unified_asr_export.cpp`).

Runs NeMo's OWN `EncDecRNNTBPEModelWithPrompt` -- `forward(..., prompt_indices=...)` and the model's
own greedy RNN-T decode -- over one clip, and writes:

    ref_waveform.npy          (n,)        f32 -- the clip
    ref_prompts.npy           (P,)        f32 -- the prompt ids the arrays below are for, likewise
    ref_prompt_names.txt                      -- their names, one per line, in the same order
    ref_encoder_<i>.npy       (T, d)      f32 -- the encoder phase's output under prompt i: the
                                                 conformer AND the prompt projection, so a dropped or
                                                 misread prompt shows up as a tensor, not only as a
                                                 transcript (on jfk every prompt decodes the same)
    ref_ids_<i>.npy           (k,)        f32 -- the greedy token ids under prompt i, as floats (exact
                                                 far below 2^24) so the gate reads every array with the
                                                 one `npy_fixture.h` reader
    ref_text_<i>.txt                          -- and their text

**Not `transcribe()`.** NeMo 3.0's `transcribe(target_lang=...)` loses the language on its way to the
dataset, whose default prompt mode is the training one (`unified`: the `auto` prompt or the real
language, chosen at random per clip). `forward` with an explicit `prompt_indices` is deterministic and
is what an explicit language means.

Needs NeMo 3.0 (`~/.venvs/nemo3`); see `loom_exporter/nemotron_unified_export.py`.

    python fixture_gen/reference_forward_nemotron_unified.py <model.nemo> <out_dir> [wav] [lang ...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_WAV = Path(__file__).resolve().parents[1] / "samples" / "jfk.wav"
DEFAULT_LANGS = ("en", "de", "auto")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.nemo_asr_export import prepare_nemo_environment

    prepare_nemo_environment()
    import nemo.collections.asr as nemo_asr

    checkpoint, out_dir = sys.argv[1], Path(sys.argv[2])
    wav = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_WAV
    langs = tuple(sys.argv[4:]) or DEFAULT_LANGS
    out_dir.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu").eval()
    table = dict(model.cfg.model_defaults.prompt_dictionary)
    waveform, rate = sf.read(str(wav), dtype="float32")
    if rate != int(model.cfg.preprocessor.sample_rate):
        raise SystemExit(f"{wav} is {rate} Hz; the checkpoint takes {model.cfg.preprocessor.sample_rate}")
    x = torch.from_numpy(waveform)[None]
    n = torch.tensor([waveform.shape[0]])

    prompts = []
    for i, lang in enumerate(langs):
        prompt = int(table[lang])
        prompts.append(prompt)
        with torch.inference_mode():
            encoded, encoded_len = model.forward(input_signal=x, input_signal_length=n,
                                                 prompt_indices=torch.tensor([prompt]))
            hyp = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True)[0]
        ids = [int(t) for t in hyp.y_sequence]
        # `ascontiguousarray`: the transpose is a view, which `np.save` writes Fortran-ordered, and the
        # gate's reader refuses that rather than reading it transposed.
        np.save(out_dir / f"ref_encoder_{i}.npy",
                np.ascontiguousarray(encoded[0].transpose(0, 1).float().numpy()))
        np.save(out_dir / f"ref_ids_{i}.npy", np.asarray(ids, dtype=np.float32))
        (out_dir / f"ref_text_{i}.txt").write_text(hyp.text)
        print(f"{lang} (prompt {prompt}): {len(ids)} tokens: {hyp.text!r}")

    np.save(out_dir / "ref_waveform.npy", waveform.astype(np.float32))
    np.save(out_dir / "ref_prompts.npy", np.asarray(prompts, dtype=np.float32))
    (out_dir / "ref_prompt_names.txt").write_text("\n".join(langs) + "\n")
    print(f"wrote fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
