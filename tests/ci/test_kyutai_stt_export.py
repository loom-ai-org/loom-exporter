"""Kyutai STT (P5): a decoder-only streaming ASR over Mimi codes, run the way Kyutai's `moshi` runs it.

What is tested here is what the trace re-spells and what the engine reads as DATA:

* the recognizer;
* the encoder transformer's mask against a SIMULATION of moshi's ring KV cache filled two positions per
  call -- the reason the first of each pair keeps one key fewer;
* the causal convolutions' two padding modes, against moshi's own streaming convolution;
* the RMSNorm re-spelling, against moshi's, and its one-sided broadcasts;
* the declarations: the ring KV cache, the named SentencePiece model, the 24 kHz contract.

The numbers -- codes 156/156 and 2474/2474 (196 s, chunked) against moshi's streamed Mimi, LM logits
within 6.3e-05 teacher-forced, the transcript id for id -- are the engine gate's and the module
docstring's.
"""
import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from loom_exporter.kyutai_stt_export import (
    CHUNK_FRAMES, CONTEXT_FRAMES, MOSHI_REPO, KyutaiSttExportConfig, _is_kyutai_stt, ring_pair_mask,
)
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/kyutai-stt-1b-en-fr")
HAVE_MOSHI = Path(MOSHI_REPO, "moshi", "moshi").is_dir()
needs_moshi = pytest.mark.skipif(not HAVE_MOSHI, reason="no moshi checkout")


def _release(tmp_path: Path, model_type="stt") -> Path:
    d = tmp_path / "stt"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type, "mimi_name": "m.safetensors",
                                               "tokenizer_name": "t.model", "delays": [0] * 33}))
    (d / "model.safetensors").write_bytes(b"")
    return d


def test_a_moshi_format_stt_release_is_claimed(tmp_path):
    assert _is_kyutai_stt(_release(tmp_path))
    assert default_registry().detect(_release(tmp_path / "again")).name == "kyutai-stt"


def test_a_moshi_speech_to_speech_release_is_not(tmp_path):
    assert not _is_kyutai_stt(_release(tmp_path, model_type="moshi"))


def _ring_simulation(n_positions: int, capacity: int, per_call: int) -> np.ndarray:
    """Which keys each query sees when moshi's `RingKVCache` of `capacity` is written `per_call`
    positions at a time, every write before any read: allowed iff the key's position is still in its
    slot and `0 <= q - k < capacity`."""
    allowed = np.zeros((n_positions, n_positions), dtype=bool)
    slot = {}
    for start in range(0, n_positions, per_call):
        call = range(start, min(start + per_call, n_positions))
        for p in call:
            slot[p % capacity] = p
        held = set(slot.values())
        for q in call:
            for k in held:
                if 0 <= q - k < capacity:
                    allowed[q, k] = True
    return allowed


@pytest.mark.parametrize("context", [6, 250])
def test_the_encoder_mask_is_moshis_ring_filled_two_at_a_time(context):
    # Whole pairs only: the encoder's positions come two per frame, so a half-written pair never occurs.
    n = 2 * context + 10
    want = _ring_simulation(n, context, per_call=2)
    mask = ring_pair_mask(torch.arange(n, dtype=torch.int32).view(1, -1), context)[0, 0].numpy()
    assert np.array_equal(mask == 0, want)


def test_the_encoder_mask_holds_at_absolute_positions():
    """A chunk starts at an even position, so the parity -- and with it each row's window -- is the
    same as in the whole-clip mask."""
    full = ring_pair_mask(torch.arange(40, dtype=torch.int32).view(1, -1), 6)[0, 0]
    part = ring_pair_mask(torch.arange(20, 40, dtype=torch.int32).view(1, -1), 6)[0, 0]
    assert torch.equal(full[20:, 20:], part)


def test_the_context_covers_the_receptive_field():
    # 8 layers of at most 249 positions back, two positions per frame, plus the convolutions.
    assert CONTEXT_FRAMES * 2 >= 8 * 249 + 8
    assert CHUNK_FRAMES > 0


@needs_moshi
@pytest.mark.parametrize("pad_mode", ["constant", "replicate"])
def test_the_causal_conv_is_moshis_streaming_conv(pad_mode):
    from loom_exporter.kyutai_stt_export import _causal_conv, import_moshi

    import_moshi()
    from moshi.modules.conv import StreamingConv1d

    torch.manual_seed(0)
    conv = StreamingConv1d(3, 4, kernel_size=4, stride=2, causal=True, pad_mode=pad_mode).eval()
    x = torch.randn(1, 3, 16)
    # moshi's own streaming, four samples per call, against one whole-signal call.
    with torch.no_grad(), conv.streaming(1):
        want = torch.cat([conv(x[..., i:i + 4]) for i in range(0, 16, 4)], dim=-1)
        got = _causal_conv(conv, x)
    assert torch.allclose(got, want, atol=1e-6)


@needs_moshi
def test_the_rms_norm_is_moshis():
    from loom_exporter.kyutai_stt_export import _RMSNorm, import_moshi

    import_moshi()
    from moshi.modules.transformer import RMSNorm

    torch.manual_seed(0)
    norm = RMSNorm(8, eps=1e-8, dtype=torch.float32)
    norm.alpha.data.uniform_(0.5, 1.5)
    x = torch.randn(1, 5, 8)
    assert torch.allclose(_RMSNorm(norm)(x), norm(x), atol=1e-6)


def test_the_cache_is_a_ring_and_the_tokenizer_is_named(tmp_path):
    d = _release(tmp_path)
    kwargs = KyutaiSttExportConfig(output_path=str(tmp_path / "x.gguf"), model_dir=str(d)).backend_kwargs()
    assert kwargs["kv_cache_ring"] is True
    assert kwargs["tokenizer_family"] == "sentencepiece_proto"
    assert kwargs["tokenizer_proto_name"] == "t.model"


def test_the_contract_is_24_khz_text(tmp_path):
    contract = KyutaiSttExportConfig(output_path=str(tmp_path / "x.gguf"),
                                     model_dir=str(_release(tmp_path))).contract()
    assert contract["sample_rate"] == 24000
    assert contract["text.frontend"] == "vocab"


# ---- the live door (loom.cpp ADR-108): the encoder a few frames per call with its state carried ----

def test_the_carried_key_mask_is_a_block_of_the_whole_clips_mask():
    """Queries [p0, p0 + 2n) against keys [p0 - ctx, p0 + 2n): the same block of the one-call mask, with the
    keys before position 0 -- slots no frame has written -- masked."""
    from loom_exporter.kyutai_stt_export import ring_pair_key_mask

    ctx, total = 6, 40
    full = ring_pair_mask(torch.arange(total, dtype=torch.int32).view(1, -1), ctx)[0, 0]
    for p0, n in ((0, 4), (2, 6), (10, 2), (30, 10)):
        q = torch.arange(p0, p0 + n, dtype=torch.int32).view(1, -1)
        k = torch.arange(p0 - ctx, p0 + n, dtype=torch.int32).view(1, -1)
        got = ring_pair_key_mask(q, k, ctx)[0, 0]
        for j, kp in enumerate(range(p0 - ctx, p0 + n)):
            want = full[p0:p0 + n, kp] if kp >= 0 else torch.full((n,), -1e30)
            assert torch.equal(got[:, j], want), (p0, n, kp)


def _toy_mimi():
    """moshi's own Mimi encoder pieces at toy sizes: a causal SEANet (x4, so 8 samples a frame), a 2-layer
    RoPE transformer with a 6-position context, and the learnt x2 downsample, which pads by replicate."""
    from loom_exporter.kyutai_stt_export import import_moshi

    import_moshi()
    from moshi.modules.resample import ConvDownsample1d
    from moshi.modules.seanet import SEANetEncoder
    from moshi.modules.transformer import ProjectedTransformer

    torch.manual_seed(0)
    mimi = types.SimpleNamespace(
        encoder=SEANetEncoder(channels=1, dimension=16, n_filters=4, n_residual_layers=1, ratios=[2, 2],
                              causal=True, pad_mode="constant"),
        encoder_transformer=ProjectedTransformer(
            input_dimension=16, output_dimensions=(16,), d_model=16, num_heads=2, num_layers=2, causal=True,
            context=6, conv_layout=True, positional_embedding="rope", layer_scale=0.01, dim_feedforward=32,
            gating="none", norm="layer_norm", max_period=10000),
        downsample=ConvDownsample1d(2, dimension=16, learnt=True, causal=True))
    for m in (mimi.encoder, mimi.encoder_transformer, mimi.downsample):
        m.eval()
    return mimi


def _stepped(step, wave, n, pieces, first_flag=True):
    from loom_exporter.kyutai_stt_export import MimiEncodeStepPhase  # noqa: F401  (the type under test)

    frame = wave.shape[1] // n
    states = [torch.zeros(s, dtype=wave.dtype) for s in step.state_shapes()]
    rows, f, i = [], 0, 0
    while f < n:
        k = min(pieces[i % len(pieces)], n - f)
        i += 1
        out = step(wave[:, f * frame:(f + k) * frame],
                   torch.arange(2 * f, 2 * (f + k), dtype=torch.int32).view(1, -1),
                   torch.arange(2 * f - step.context, 2 * (f + k), dtype=torch.int32).view(1, -1),
                   torch.tensor([1.0 if (f == 0 and first_flag) else 0.0], dtype=wave.dtype), *states)
        rows.append(out[0][0])
        states = list(out[1:])
        f += k
    return torch.cat(rows)


@needs_moshi
def test_the_encoder_step_is_moshis_streamed_encoder():
    from loom_exporter.kyutai_stt_export import MimiEncodePhase, MimiEncodeStepPhase

    mimi = _toy_mimi()
    n, frame = 20, 8                       # 40 transformer positions: the 6-position window wraps
    torch.manual_seed(1)
    wave = torch.randn(1, n * frame) * 0.3
    with torch.no_grad():
        # moshi, a frame per call, every module streaming.
        with mimi.encoder.streaming(1), mimi.encoder_transformer.streaming(1), \
                mimi.downsample.conv.streaming(1):
            want = torch.cat([mimi.downsample(mimi.encoder_transformer(mimi.encoder(
                wave[:, f * frame:(f + 1) * frame].view(1, 1, -1)))[0]) for f in range(n)], dim=-1)[0].t()
        enc = MimiEncodePhase(mimi).eval()
        step = MimiEncodeStepPhase(enc).eval()
        one = enc(wave, torch.arange(2 * n, dtype=torch.int32).view(1, -1))[0]
        assert torch.allclose(one, want, atol=1e-5)
        for pieces in ([1], [1, 3, 7, 2, 5], [n]):
            assert torch.allclose(_stepped(step, wave, n, pieces), want, atol=1e-5), pieces
        # At f64 the step is the one call to rounding.
        enc64 = MimiEncodePhase(copy.deepcopy(mimi)).eval().double()
        step64 = MimiEncodeStepPhase(enc64).eval()
        one64 = enc64(wave.double(), torch.arange(2 * n, dtype=torch.int32).view(1, -1))[0]
        assert float((_stepped(step64, wave.double(), n, [1, 3, 7, 2, 5]) - one64).abs().max()) < 1e-12
        # Without `first`, the downsample pads its first call with zeros, not the first column: wrong.
        assert not torch.allclose(_stepped(step, wave, n, [2], first_flag=False), want, atol=1e-3)


def test_the_driver_carries_a_live_path():
    text = (Path(__file__).resolve().parents[2] / "loom_exporter/kyutai_stt_driver/00_stream.lua").read_text()
    for fn in ("function stream_open", "function stream_feed", "function stream_finish"):
        assert fn in text
    assert "mimi_encode_step" in text and "keep_state('mimi'" in text


@needs_moshi
def test_the_rvq_in_one_graph_draws_moshis_codes():
    """`rvq_encode` -- the split RVQ with every stage's argmax and gather in the graph -- against moshi's own
    `SplitResidualVectorQuantizer.encode`, at toy sizes with random codebooks."""
    from loom_exporter.kyutai_stt_export import RvqEncodePhase, import_moshi

    import_moshi()
    from moshi.quantization.vq import SplitResidualVectorQuantizer

    torch.manual_seed(0)
    q = SplitResidualVectorQuantizer(dimension=8, input_dimension=16, output_dimension=16, n_q=5, bins=32).eval()
    for layer in list(q.rvq_first.vq.layers) + list(q.rvq_rest.vq.layers):
        cb = layer._codebook
        cb.embedding_sum.data.normal_()
        cb.cluster_usage.data.fill_(1.0)
        cb._initialized.data.fill_(1.0)
        cb.register_buffer("_embedding", None, persistent=False)
    rows = torch.randn(1, 12, 16)
    with torch.no_grad():
        want = q.encode(rows.transpose(1, 2))[0]                             # [n_q, n]
        got = RvqEncodePhase(q).eval()(rows)
    assert got.shape == (1, 12, 5) and got.dtype == torch.float32
    assert torch.equal(got[0].long().t(), want)
    assert len(set(want.flatten().tolist())) > 5                             # not one code everywhere
