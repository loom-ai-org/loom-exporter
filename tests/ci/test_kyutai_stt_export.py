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
import json
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
