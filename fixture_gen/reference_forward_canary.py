#!/usr/bin/env python3
"""Ground truth for a MIL-exported Canary (`canary_export.py`): NeMo's own decode and its tensors.

For each case, written to <out_dir>:

    <case>.json            the prompt NeMo built, the ids its beam_size-1 search produced, the text
    <case>.f32 / _len.f32  the waveform and its sample count, as the encoder phase takes them
    <case>_enc32/64.npy    encoder states cut to `encoded_len` -- what cross_kv reads
    <case>_lp32/64.npy     the decoder's LOGITS over prompt + generated, teacher-forced through NeMo's
                           own uncached forward -- THE oracle; the export emits logits too

**f64 is the arbiter, not decoration.** Loom and NeMo-f32 are two f32 implementations of one model; the
question a delta has to answer is whether loom is further from the true value than NeMo's own rounding
puts NeMo. Measured 2026-10-01 on JFK: logits loom 6.0e-5 / NeMo-f32 1.7e-4 from f64, encoder 1.2e-5 /
3.3e-5, and NeMo's ids reproduced exactly in all three cases.

**Logits, not log-probs.** coremltools lowers `log_softmax` as `log(softmax(x))`, which underflows to
-inf below about -103; the export's head stops before it, and so does this.

    python3 fixture_gen/reference_forward_canary.py <canary.nemo> <wav> <out_dir>
"""
import sys, json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loom_exporter.canary_export import ASRCanaryExportConfig  # noqa: E402
from loom_exporter.nemo_asr_export import prepare_nemo_environment  # noqa: E402

prepare_nemo_environment()
CK, WAV, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
cfg = ASRCanaryExportConfig(checkpoint=CK)
m = cfg.load_model()
cfg._read_prompt(m)
print("prompt", cfg.prompt_ids, "eos", cfg.eos_token_id, "pad", cfg.pad_token_id)

wav, sr = sf.read(WAV, dtype="float32")
# The whole clip; a cut whose encoder emits one frame past `encoded_len` (48777 samples: 39 vs 38);
# and a translation, which is the target-language slot the driver fills.
cases = {"jfk_en": (wav, "en", "en"), "jfk_48777": (wav[:48777], "en", "en"), "jfk_en_fr": (wav, "en", "fr")}


def prompt_for(src, tgt):
    p = list(cfg.prompt_ids)
    p[cfg.source_slot] = cfg.lang_to_id[src]; p[cfg.target_slot] = cfg.lang_to_id[tgt]
    return p


def run(model, x, prompt, dtype):
    with torch.no_grad():
        sig = torch.from_numpy(x.astype(dtype))[None]
        _, enc_len, enc, enc_mask = model(input_signal=sig, input_signal_length=torch.tensor([len(x)]))
        n = int(enc_len[0])
        gen = model.decoding.decoding.beam_search(
            encoder_hidden_states=enc, encoder_input_mask=enc_mask,
            decoder_input_ids=torch.tensor([prompt]))
        ids = [int(i) for i in (gen[0] if isinstance(gen, (list, tuple)) else gen)[0]]
        # Teacher-forced log-probs over the WHOLE sequence, NeMo's own uncached forward.
        seq = torch.tensor([ids])
        dec_mask = torch.ones_like(seq)
        dec = model.transf_decoder(input_ids=seq, decoder_mask=dec_mask, encoder_embeddings=enc,
                                   encoder_mask=enc_mask)
        with model.log_softmax.with_log_softmax_enabled(False):
            lp = model.log_softmax(hidden_states=dec)
    return ids, enc[0, :n].double().numpy(), enc.shape[1], n, lp[0].double().numpy()


for name, (x, src, tgt) in cases.items():
    prompt = prompt_for(src, tgt)
    ids, enc, t_full, n, lp = run(m, x, prompt, np.float32)
    text = m.tokenizer.ids_to_text([i for i in ids[len(prompt):] if i not in (cfg.eos_token_id, cfg.pad_token_id)])
    print(f"{name}: enc frames {t_full} (encoded_len {n}); {len(ids) - len(prompt)} generated; {text!r}")
    np.save(OUT / f"{name}_enc32.npy", enc); np.save(OUT / f"{name}_lp32.npy", lp)
    json.dump({"prompt": prompt, "ids": ids, "text": text, "n_enc": n}, open(OUT / f"{name}.json", "w"))
    x.astype(np.float32).tofile(OUT / f"{name}.f32")
    np.array([len(x)], np.float32).tofile(OUT / f"{name}_len.f32")

m64 = m.double()
for name, (x, src, tgt) in cases.items():
    ref = json.load(open(OUT / f"{name}.json"))
    with torch.no_grad():
        sig = torch.from_numpy(x.astype(np.float64))[None]
        _, enc_len, enc, enc_mask = m64(input_signal=sig, input_signal_length=torch.tensor([len(x)]))
        seq = torch.tensor([ref["ids"]])
        dec = m64.transf_decoder(input_ids=seq, decoder_mask=torch.ones_like(seq),
                                 encoder_embeddings=enc, encoder_mask=enc_mask)
        with m64.log_softmax.with_log_softmax_enabled(False):
            lp = m64.log_softmax(hidden_states=dec)
    n = int(enc_len[0])
    np.save(OUT / f"{name}_enc64.npy", enc[0, :n].numpy()); np.save(OUT / f"{name}_lp64.npy", lp[0].numpy())
    e32 = np.load(OUT / f"{name}_enc32.npy"); l32 = np.load(OUT / f"{name}_lp32.npy")
    print(f"{name}: |enc32-enc64| {np.abs(e32 - enc[0, :n].numpy()).max():.2e}  "
          f"|lp32-lp64| {np.abs(l32 - lp[0].numpy()).max():.2e}")
