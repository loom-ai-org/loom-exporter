"""Family 9's eighth leaf (P5): Voxtral-4B-TTS -- an autoregressive LM over frames of 37 codes, each
frame's acoustic half integrated by a flow-matching head, then a causal ALiBi codec.

What is tested here is what the Lua is generated from and what the engine reads as DATA, because a wrong
declaration there is a wrong model with nothing failing:

* the recognizer;
* the Tekken writer -- the ids (markers, then ranks, truncated as `Tekkenizer` truncates), the GPT-2 byte
  spelling of each rank, and the refusal of any regex but the one the engine's `tekken` shape scans;
* the LM's Q/K permutation, on a TINY random Mistral: the traced (rotate-half) stack against the
  checkpoint's native interleaved-pair RoPE -- and the check going red without the permutation;
* every re-spelling the trace needs (the flow head, the frame embedding, the codec), against vllm-omni's
  own modules on a TINY random model built from their classes, at f64 -- so the check needs the
  vllm-omni checkout but not the 8 GB checkpoint;
* the codec's causal paddings and ALiBi-plus-window bias against upstream's formulas;
* the voice fingerprint: it covers the LM, not the flow head or the codec.

The numbers -- codes identical to the reference free-running and teacher-forced over 142 frames, the
waveform at rmse 2e-8, the tokenizer at 12000/12000 against `mistral_common` -- are the engine gate's
(`tests/gate/test_e2e_voxtral_tts_lua_driver.cpp`) and the export's docstring's.
"""
import base64
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from loom_exporter import voxtral_tts_export as V
from loom_exporter.registry import default_registry
from loom_exporter.tekken_tokenizer_export import TEKKEN_PATTERN, read_tekken, tekken_ids
from loom_exporter.voxtral_tts_voices import expected_rows, weights_fingerprint

HAVE_UPSTREAM = Path(V.VLLM_OMNI_SRC, "voxtral_tts_audio_generation.py").is_file()
needs_upstream = pytest.mark.skipif(not HAVE_UPSTREAM, reason="no vllm-omni checkout (VLLM_OMNI_VOXTRAL)")


def _checkpoint_dir(tmp_path: Path, model_type="voxtral_tts", missing=()) -> Path:
    d = tmp_path / "voxtral"
    d.mkdir()
    if "params.json" not in missing:
        (d / "params.json").write_text(json.dumps({"model_type": model_type, "dim": 3072}))
    for name in ("consolidated.safetensors", "tekken.json"):
        if name not in missing:
            (d / name).write_bytes(b"")
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_voxtral_tts_directory_is_claimed(tmp_path):
    assert V._is_voxtral_tts(_checkpoint_dir(tmp_path))


@pytest.mark.parametrize("missing", ["params.json", "consolidated.safetensors", "tekken.json"])
def test_a_directory_missing_a_release_file_is_not_claimed(tmp_path, missing):
    assert not V._is_voxtral_tts(_checkpoint_dir(tmp_path, missing=(missing,)))


def test_another_mistral_checkpoint_is_not_claimed(tmp_path):
    """Every Mistral release ships params.json + consolidated.safetensors + tekken.json; only
    `model_type: voxtral_tts` is this export's."""
    assert not V._is_voxtral_tts(_checkpoint_dir(tmp_path, model_type="voxtral"))


def test_the_registry_routes_a_voxtral_tts_directory_to_this_recognizer(tmp_path):
    assert default_registry().detect(_checkpoint_dir(tmp_path)).name == "voxtral-tts"


# -- the Tekken writer ----------------------------------------------------------------------------

def _tekken(tmp_path: Path, extra_ranks=(b"\n\n", b"ab", b"abc"), n_special=5, vocab_size=None,
            pattern=TEKKEN_PATTERN) -> Path:
    ranks = [bytes([b]) for b in range(256)] + list(extra_ranks)
    spec = {
        "config": {"pattern": pattern, "num_vocab_tokens": len(ranks), "version": "v7",
                   "default_num_special_tokens": n_special,
                   "default_vocab_size": vocab_size or n_special + len(ranks)},
        "vocab": [{"rank": i, "token_bytes": base64.b64encode(r).decode(), "token_str": None}
                  for i, r in enumerate(ranks)],
        "special_tokens": [{"rank": i, "token_str": s, "is_control": True}
                           for i, s in enumerate(["<unk>", "<s>", "</s>", "[AUDIO]"])],
    }
    (tmp_path / "tekken.json").write_text(json.dumps(spec))
    return tmp_path


def test_markers_come_first_and_unlisted_slots_are_fillers(tmp_path):
    pieces, types_, n_special = tekken_ids(read_tekken(str(_tekken(tmp_path))))
    assert n_special == 5
    assert pieces[:5] == ["<unk>", "<s>", "</s>", "[AUDIO]", "<SPECIAL_4>"]
    assert types_[:5] == [3] * 5 and set(types_[5:]) == {1}


def test_a_rank_is_its_bytes_through_the_gpt2_byte_map(tmp_path):
    pieces, _, n = tekken_ids(read_tekken(str(_tekken(tmp_path))))
    assert pieces[n + ord("a")] == "a"
    assert pieces[n + 0x20] == "Ġ"          # space, as every "gpt2" vocabulary spells it
    assert pieces[n + 256] == "ĊĊ"          # the first multi-byte rank: two newlines


def test_the_ranks_are_cut_to_the_default_vocab_size(tmp_path):
    """`Tekkenizer` keeps the first `default_vocab_size - n_special` ranks and drops the rest."""
    pieces, _, n = tekken_ids(read_tekken(str(_tekken(tmp_path, vocab_size=5 + 257))))
    assert len(pieces) == 5 + 257 and pieces[-1] == "ĊĊ"


def test_another_regex_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not the one"):
        read_tekken(str(_tekken(tmp_path, pattern=r"\s+|\S+")))


def test_two_ranks_with_the_same_bytes_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same bytes"):
        tekken_ids(read_tekken(str(_tekken(tmp_path, extra_ranks=(b"ab", b"ab")))))


# -- the LM ----------------------------------------------------------------------------------------

def _tiny_mistral(n_layers=2, dim=32, heads=4, kv_heads=2, head_dim=8, hidden=48, vocab=20001):
    torch.manual_seed(0)
    params = {"n_layers": n_layers, "dim": dim, "n_heads": heads, "n_kv_heads": kv_heads,
              "head_dim": head_dim, "hidden_dim": hidden, "rope_theta": 1e6, "norm_eps": 1e-5}
    w = {"mm_audio_embeddings.tok_embeddings.weight": torch.randn(vocab, dim),
         "norm.weight": 1 + 0.1 * torch.randn(dim)}
    for i in range(n_layers):
        L = f"layers.{i}."
        w.update({L + "attention.wq.weight": torch.randn(heads * head_dim, dim) * 0.2,
                  L + "attention.wk.weight": torch.randn(kv_heads * head_dim, dim) * 0.2,
                  L + "attention.wv.weight": torch.randn(kv_heads * head_dim, dim) * 0.2,
                  L + "attention.wo.weight": torch.randn(dim, heads * head_dim) * 0.2,
                  L + "feed_forward.w1.weight": torch.randn(hidden, dim) * 0.2,
                  L + "feed_forward.w2.weight": torch.randn(dim, hidden) * 0.2,
                  L + "feed_forward.w3.weight": torch.randn(hidden, dim) * 0.2,
                  L + "attention_norm.weight": 1 + 0.1 * torch.randn(dim),
                  L + "ffn_norm.weight": 1 + 0.1 * torch.randn(dim)})
    return params, w


def test_the_permuted_lm_is_the_native_one():
    params, w = _tiny_mistral()
    assert V.check_lm_permutation(params, w, n_layers=2) < 1e-5


def test_the_permutation_check_fails_without_the_permutation(monkeypatch):
    """The check above would pass vacuously if both spellings shared a bug; drop the permutation and it
    must go red."""
    params, w = _tiny_mistral()
    monkeypatch.setattr(V, "permute_for_rotate_half", lambda t, n: t)
    with pytest.raises(AssertionError, match="permutation"):
        V.check_lm_permutation(params, w, n_layers=2)


def test_the_lm_returns_the_last_row_only():
    params, w = _tiny_mistral(n_layers=1)
    lm = V.LMPhase(params, w)
    out = lm(torch.randn(1, 5, 32), V.positions(5), V.causal_mask(5))
    assert out.shape == (1, 1, 32)


# -- the frame and the codec, against vllm-omni's modules -----------------------------------------------

def _tiny_upstream(dtype=torch.float64):
    gen_mod, tok_mod = V.import_upstream()
    torch.manual_seed(1)
    args = {"semantic_codebook_size": 40, "acoustic_codebook_size": 21, "n_acoustic_codebook": 36,
            "acoustic_transformer_args": {"input_dim": 32, "dim": 32, "n_layers": 2, "head_dim": 8,
                                          "hidden_dim": 64, "n_heads": 4, "n_kv_heads": 2,
                                          "n_decoding_steps": 7}}
    flow = gen_mod.FlowMatchingAudioTransformer(json.loads(json.dumps(args)))
    for p in flow.parameters():
        torch.nn.init.normal_(p, std=0.2)
    codec_args = {"semantic_codebook_size": 40, "semantic_dim": 8, "acoustic_codebook_size": 21,
                  "acoustic_dim": 36, "dim": 32, "hidden_dim": 64, "head_dim": 8, "n_heads": 4,
                  "n_kv_heads": 4, "layer_scale_init": 0.5, "decoder_transformer_lengths_str": "1,1,1,1",
                  "encoder_transformer_lengths_str": "1,1,1,1"}
    hf = types.SimpleNamespace(audio_config={"codec_args": codec_args, "audio_model_args": args},
                               text_config=types.SimpleNamespace(hidden_size=32))
    cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=hf))
    codecs = []
    for _ in range(2):
        torch.manual_seed(2)
        c = tok_mod.VoxtralTTSAudioTokenizer(vllm_config=cfg)
        for p in c.parameters():
            torch.nn.init.normal_(p, std=0.3)
        sc = c.quantizer.semantic_codebook
        sc.embedding_sum = torch.randn(40, 8)
        sc.cluster_usage = torch.rand(40) + 0.5
        codecs.append(c)
    flow = flow.to(dtype).eval()
    for c in codecs:
        c.to(dtype).eval()
        sc = c.quantizer.semantic_codebook
        sc.embedding_sum, sc.cluster_usage, sc._embedding = sc.embedding_sum.to(dtype), sc.cluster_usage.to(dtype), None
    return flow, codecs[0], codecs[1]


@needs_upstream
def test_every_wrapper_is_its_upstream_module_at_f64():
    flow, codec, codec_ref = _tiny_upstream()
    out = V.compare_wrappers(flow, codec, codec_ref, n_frames=12)
    assert set(out) == {"semantic_logits", "embed_frame", "codec"}


@needs_upstream
def test_the_wrapper_check_catches_a_scaled_rescale(monkeypatch):
    """The one real defect the f64 check found was the acoustic rescale spelled `c * 0.1`; a rescale off
    by a real amount must be caught by name."""
    flow, codec, codec_ref = _tiny_upstream()
    original = V.CodecPhase.forward

    def skewed(self, codes, *pos):
        return original(self, codes + 0 * codes, *pos) * 1.001

    monkeypatch.setattr(V.CodecPhase, "forward", skewed)
    with pytest.raises(AssertionError, match="codec wrapper"):
        V.compare_wrappers(flow, codec, codec_ref, n_frames=12)


@needs_upstream
@pytest.mark.parametrize("mode", ["replicate", "reflect"])
def test_the_causal_paddings_are_upstreams(mode):
    _, tok_mod = V.import_upstream()
    x = torch.randn(1, 3, 10, dtype=torch.float64)
    want = tok_mod.pad1d(x, (6, 0), mode=mode)
    assert torch.equal(V.CodecPhase._pad_left(x, mode, 6), want)


def test_the_bias_is_alibi_inside_a_causal_window():
    phase = V.CodecPhase.__new__(V.CodecPhase)
    torch.nn.Module.__init__(phase)
    phase.slopes = [1.0, 0.5]
    bias = phase._bias(V.positions(6), 2, True)[0]                  # [2, 6, 6]
    for h, slope in enumerate(phase.slopes):
        for i in range(6):
            for j in range(6):
                inside = 0 <= i - j <= 2
                if inside:
                    assert float(bias[h, i, j]) == slope * (j - i)
                else:
                    assert float(bias[h, i, j]) < -1e29


# -- voices ----------------------------------------------------------------------------------------

def _safetensors(path: Path, tensors: dict) -> Path:
    from safetensors.numpy import save_file

    save_file(tensors, str(path))
    return path


def test_the_fingerprint_covers_the_lm_and_not_the_flow_head_or_codec(tmp_path):
    base = {"layers.0.attention.wq.weight": np.ones((2, 2), np.float32),
            "acoustic_transformer.norm.weight": np.ones(2, np.float32),
            "audio_tokenizer.output_proj.weight": np.ones(2, np.float32)}
    a = weights_fingerprint(_safetensors(tmp_path / "a.safetensors", base))
    head = dict(base, **{"acoustic_transformer.norm.weight": np.zeros(2, np.float32),
                         "audio_tokenizer.output_proj.weight": np.zeros(2, np.float32)})
    assert weights_fingerprint(_safetensors(tmp_path / "b.safetensors", head)) == a
    lm = dict(base, **{"layers.0.attention.wq.weight": np.zeros((2, 2), np.float32)})
    assert weights_fingerprint(_safetensors(tmp_path / "c.safetensors", lm)) != a


def test_a_voice_is_as_many_rows_as_its_slots(tmp_path):
    (tmp_path / "tekken.json").write_text(json.dumps({"audio": {"voice_num_audio_tokens": {"casual_male": 147}}}))
    assert expected_rows(tmp_path) == {"casual_male": 147}


# -- the declarations ------------------------------------------------------------------------------

def test_the_defaults_are_the_deploy_configs():
    """vllm-omni's `deploy/voxtral_tts.yaml` stage 0: cfg_alpha 1.2, max_tokens 2048, max_model_len
    4096; its parser's `n_decoding_steps` default 7; the model card's example voice."""
    assert (V.DEFAULT_CFG, V.MAX_FRAMES, V.LM_MAX_POSITIONS, V.N_DECODING_STEPS) == (1.2, 2048, 4096, 7)
    assert V.DEFAULT_VOICE == "casual_male" and V.SAMPLE_RATE == 24000 and V.SAMPLES_PER_FRAME == 1920
    assert V.CODEC_CONTEXT >= 20       # the receptive field measured on real speech is ~20 frames
