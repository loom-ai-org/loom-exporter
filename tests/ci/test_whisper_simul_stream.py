"""Whisper's live door (loom.cpp ADR-128): SimulStreaming's AlignAtt in `whisper_driver/10_stream.lua`.

What runs is checked by loom.cpp's `test_e2e_whisper_simul_stream` against SimulStreaming itself; here:

1. **The decoder's second output is the alignment heads' cross-attention**, the same probabilities HF's eager
   attention returns with `output_attentions`, and the logits are unchanged by it.
2. **The driver's most-attended frame is SimulStreaming's**: running sums in double, the reflect-padded
   median, the heads' mean, mirrored from the Lua, against SimulStreaming's `std_mean` + `median_filter` in
   torch.
3. **Every constant the fragment reads is bound.**
"""
import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from loom_exporter import whisper_export as W


def _tiny_whisper():
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, encoder_layers=2, decoder_layers=3, d_model=16,
                        encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=20, max_target_positions=16, pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, decoder_start_token_id=1)
    # Eager, so the reference can return its attention (`output_attentions`); the wrapper never asks for it.
    cfg._attn_implementation = "eager"
    return WhisperForConditionalGeneration(cfg).eval()


def test_the_second_output_is_the_alignment_heads_attention():
    pytest.importorskip("transformers")
    model = _tiny_whisper()
    heads = ((2, 1), (0, 3), (2, 0))
    torch.manual_seed(1)
    xa = torch.randn(1, 20, 16)
    tokens = torch.tensor([[1, 5, 9, 4]])
    with torch.no_grad():
        ref = model.model.decoder(input_ids=tokens, encoder_hidden_states=xa, output_attentions=True,
                                  use_cache=False)
        ref_logits = model.proj_out(ref.last_hidden_state)
        # K and V in their natural layout: `cross_kv` emits V head-split and transposed for the engine's
        # rewritten graph (`hoist_cross_v_transpose`), which an untraced decoder would misread.
        cross = []
        for layer in model.model.decoder.layers:
            cross += [layer.encoder_attn.k_proj(xa), layer.encoder_attn.v_proj(xa)]
        wrapper = W._WhisperDecoderWrapper(model, heads).eval()
        logits, align = wrapper(tokens, torch.arange(4).unsqueeze(0), W.causal_mask(4), *cross)
    assert torch.allclose(logits, ref_logits, atol=1e-5)
    want = torch.stack([ref.cross_attentions[l][0, h] for l, h in sorted(heads)])
    assert align.shape == (1, 3, 4, 20)
    assert torch.allclose(align[0], want, atol=1e-6)


def _median7_reflect(z):
    """The Lua's `ws_median7`."""
    n = len(z)
    out = []
    for f in range(n):
        w = []
        for j in range(-3, 4):
            i = f + j
            i = -i if i < 0 else (2 * (n - 1) - i if i > n - 1 else i)
            w.append(z[i])
        out.append(sorted(w)[3])
    return out


def _lua_frame(rows, content):
    """`ws_infer`'s frame for the newest row of `rows` ([H, T, S]): running sums, then the newest row."""
    h_n, t_n, s_n = rows.shape
    mean = np.zeros(s_n)
    for h in range(h_n):
        m = rows[h].sum(0) / t_n
        var = np.maximum((rows[h] ** 2).sum(0) / t_n - m * m, 0)
        z = (rows[h, -1] - m) / np.sqrt(var)
        mean += np.asarray(_median7_reflect(list(z)))
    mean /= h_n
    return int(np.argmax(mean[:content]))


def _simul_frame(rows, content):
    """SimulStreaming's: `std_mean` over the tokens, `whisper.timing.median_filter`, the heads' mean."""
    a = torch.tensor(rows, dtype=torch.float32)[None]
    std, mean = torch.std_mean(a, dim=-2, keepdim=True, unbiased=False)
    a = (a - mean) / std
    pad = F.pad(a, (3, 3, 0, 0), mode="reflect")
    a = pad.unfold(-1, 7, 1).sort()[0][..., 3]
    a = a.mean(dim=1)[:, :, :content]
    return int(torch.argmax(a[:, -1, :], dim=-1)[0])


@pytest.mark.parametrize("seed", range(20))
def test_the_most_attended_frame_is_simulstreamings(seed):
    rng = np.random.default_rng(seed)
    h, t, s = 3, int(rng.integers(5, 30)), 60
    logits = rng.normal(size=(h, t, s)) * 3
    rows = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    content = int(rng.integers(10, s))
    assert _lua_frame(rows, content) == _simul_frame(rows, content)


def test_every_constant_the_fragment_reads_is_bound():
    text = (W.Path(W.__file__).resolve().parent / "whisper_driver" / "10_stream.lua").read_text()
    read = set(re.findall(r"\bW_[A-Z_]+\b", text))
    bound = {"W_SOT", "W_EOT", "W_SOT_PREV", "W_NO_TIMESTAMPS", "W_TRANSCRIBE", "W_LANG_LO", "W_LANG_HI",
             "W_LANG_DEFAULT", "W_SUPPRESS", "W_SPACE", "W_N_ALIGN", "W_N_AUDIO_CTX", "W_N_TEXT_CTX", "W_N_SAMPLES",
             "W_RATE", "W_HOP", "W_N_TEXT_LAYERS", "W_DEFAULT_CHUNK_SECONDS", "W_MAX_SECONDS", "W_FRAME_THRESHOLD",
             "W_REWIND", "W_DEC_PAD"}
    assert read <= bound, read - bound
    # And `stream_constants` binds exactly that set (keys read off its source, no checkpoint needed).
    src = W.Path(W.__file__).read_text()
    body = src[src.index("def stream_constants"):src.index("def load_feature_extractor")]
    assert set(re.findall(r'"(W_[A-Z_]+)"', body)) == bound
