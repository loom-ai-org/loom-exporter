#!/usr/bin/env python3
"""Ground truth for the MIL-exported T5 (family 6, P5).

Runs HF's own `T5ForConditionalGeneration` -- the library, not this repo's exporter -- over a fixed
prompt, and writes what an end-to-end check needs to localize a failure to a HALF rather than only
report one:

    ref_t5_source.npy     (n_src,)        i32  -- the encoded source, `</s>` included
    ref_t5_encoder.npy    (n_src, d)      f32  -- the encoder's own hidden states
    ref_t5_bias_enc.npy   (h, n, n)       f32  -- the encoder's relative attention bias at n = n_src
    ref_t5_bias_dec.npy   (h, n, n)       f32  -- the decoder's, causal mask NOT applied
    ref_t5_decoder.npy    (n_out, V)      f32  -- teacher-forced decoder logits over the generated
                                                  prefix, so the decoder half is checkable in isolation
    ref_t5_generated.npy  (n_new,)        i32  -- the ids HF greedily generates
    ref_t5_text.txt                            -- what those ids detokenize to

**The two bias arrays are the point of this file**, and [Retro-006] is why: a wrong bias still decodes
fluent text, because the argmax of a slightly-wrong distribution is usually the same token. The bias is
the one thing in this export that no other family exercises, so it is dumped as a TENSOR and compared
as one, and the token sequence is the weaker check that runs beside it.

They are written WITHOUT the causal mask, deliberately: what the driver builds is bias-plus-mask, so
comparing the sum would let a wrong bias and a wrong mask cancel on the cells where the mask is -inf --
which is half of them.

    python3 fixture_gen/reference_forward_t5_mil.py <model_dir> <out_dir> [n_new_tokens]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# An instruction-shaped prompt, because flan-t5 is instruction-tuned and a bare sentence is out of
# distribution for it. Fixed rather than random: unlike the audio families, whose reference scripts
# argue for noise, a text model's input has to be tokenizable -- and the tensor comparison below is
# what carries the strictness that randomness would otherwise have to.
PROMPT = "translate English to German: The weather today is cold and rainy in the north."


def relative_position_bias(stack, n: int) -> np.ndarray:
    """`compute_bias` at length `n`, with no mask applied -- `[n_head, n, n]`."""
    attention = stack.block[0].layer[0].SelfAttention
    with torch.no_grad():
        bias = attention.compute_bias(n, n)
    return bias[0].numpy().astype(np.float32)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = T5ForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch.float32).eval()

    source = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        encoder_out = model.encoder(input_ids=source).last_hidden_state
        # `num_beams=1, do_sample=False` -- greedy, so the comparison is an exact integer-sequence
        # equality with no tolerance question. flan-t5's own `generation_config` names beams for the
        # translation task presets, which a driver does not implement and must not be compared against.
        generated = model.generate(input_ids=source, max_new_tokens=n_new,
                                   num_beams=1, do_sample=False)[0]
        # `generate` returns `decoder_start_token_id` in front of what it generated; the driver's own
        # contract is the generated ids alone, so the prefix is dropped here rather than in whatever
        # reads this.
        decoder_input = generated[:-1].unsqueeze(0)
        logits = model(input_ids=source, decoder_input_ids=decoder_input).logits[0]

    np.save(out_dir / "ref_t5_source.npy", source[0].numpy().astype(np.int32))
    np.save(out_dir / "ref_t5_encoder.npy", encoder_out[0].numpy().astype(np.float32))
    np.save(out_dir / "ref_t5_bias_enc.npy", relative_position_bias(model.encoder, source.shape[1]))
    np.save(out_dir / "ref_t5_bias_dec.npy", relative_position_bias(model.decoder, int(generated.numel())))
    np.save(out_dir / "ref_t5_decoder.npy", logits.numpy().astype(np.float32))
    np.save(out_dir / "ref_t5_generated.npy", generated[1:].numpy().astype(np.int32))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    (out_dir / "ref_t5_text.txt").write_text(text + "\n")

    print(f"source     {tuple(source.shape)} -> {source[0].tolist()}")
    print(f"encoder    {tuple(encoder_out.shape)}")
    print(f"generated  {generated[1:].tolist()}")
    print(f"text       {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
