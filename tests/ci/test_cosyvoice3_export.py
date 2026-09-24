"""Family 9's seventh leaf (P5): Fun-CosyVoice3-0.5B-2512 -- a Qwen2 LM sampled under `ras_sampling`,
a flow-matching DiT and a causal HiFT vocoder in one GGUF.

What is tested here is what the Lua is GENERATED from and the substitutions a trace depends on, because
a wrong declaration or a wrong rewrite is a wrong model with nothing failing:

* the recognizer, which must not claim CosyVoice 1 or 2;
* the sampler's declaration and the text-door contract, and the RAS/stop constants the driver reads;
* the DiT's rope substitution, against x_transformers' own function on the shape CosyVoice3 calls it
  with -- the projected `(b, n, 1024)` query, so only head 0 rotates;
* the causal-convolution padding patch, against the reference's own zero-concatenation;
* the staged tokenizer: `<|endofprompt|>` at 151646 and every added token in the written file.

The numbers -- 76/76 sampled tokens identical, the mel at the reference's f32/f64 floor -- are the
engine gate's (`tests/gate/test_e2e_cosyvoice3_lua_driver.cpp`).
"""
import json
from pathlib import Path

import pytest
import torch

from loom_exporter.cosyvoice3_export import (
    COSYVOICE_REPO, END_OF_PROMPT, FLOW_CFG_RATE, FLOW_STEPS, LM_HEAD, MAX_SILENT_RUN,
    MAX_TOKEN_TEXT_RATIO, MIN_TOKEN_TEXT_RATIO, RAS_TAU, RAS_TOP_K, RAS_TOP_P, RAS_WIN, SAMPLES_PER_FRAME,
    SILENT_TOKENS, SOS, TASK_ID, CosyVoice3ExportConfig, _apply_rope_first_head, _is_cosyvoice3,
)
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/fun-cosyvoice3-0.5b-2512")
RELEASE_FILES = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx",
                 "speech_tokenizer_v3.onnx")


def _release(tmp_path: Path, missing=()) -> Path:
    d = tmp_path / "cosyvoice3"
    d.mkdir()
    for name in RELEASE_FILES:
        if name not in missing:
            (d / name).write_bytes(b"")
    if "CosyVoice-BlankEN" not in missing:
        (d / "CosyVoice-BlankEN").mkdir()
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_release_directory_is_claimed(tmp_path):
    assert _is_cosyvoice3(_release(tmp_path))


@pytest.mark.parametrize("missing", RELEASE_FILES + ("CosyVoice-BlankEN",))
def test_a_directory_missing_any_release_file_is_not_claimed(tmp_path, missing):
    """The two ONNX models are required too: the default voice is computed from them at export."""
    assert not _is_cosyvoice3(_release(tmp_path, missing=(missing,)))


def test_cosyvoice2_is_not_claimed(tmp_path):
    """CosyVoice 2 ships the same file names beside `cosyvoice2.yaml`; its flow is a U-Net, not a DiT."""
    d = _release(tmp_path, missing=("cosyvoice3.yaml",))
    (d / "cosyvoice2.yaml").write_text("")
    assert not _is_cosyvoice3(d)


def test_the_registry_routes_it_to_text_to_speech():
    assert default_registry().get("text-to-speech", "cosyvoice3").name == "cosyvoice3"


# -- declarations the driver is generated from ------------------------------------------------------

def _config(tmp_path):
    return CosyVoice3ExportConfig(output_path=str(tmp_path / "o.gguf"), model_dir=str(_release(tmp_path)))


def test_the_sampler_declares_guidance_a_caller_schedule_and_caller_noise(tmp_path):
    """Chatterbox's S3Gen declaration exactly: `CausalConditionalCFM` is the same solver."""
    spec, = _config(tmp_path).samplers()
    assert spec.guidance is True
    assert spec.schedule == "caller"
    assert spec.caller_noise is True
    assert spec.method == "euler"
    assert spec.carried_input == "x"
    assert spec.fixed_inputs == ["mu", "spks", "cond"]


def test_the_contract_declares_a_text_door(tmp_path):
    config = _config(tmp_path)
    config.task = "text-to-speech"
    contract = config.contract()
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == 24000
    assert contract["tts.default_steps"] == FLOW_STEPS


def test_the_sampling_constants_are_the_references_own():
    """`cosyvoice3.yaml`'s `ras_sampling` partial, `Qwen2LM.inference`'s ratios and
    `CosyVoice3Model`'s silence cap. `RAS_WIN * RAS_TAU` must be exactly 1.0: the reference compares
    an integer count against it, so ANY repeat in the window triggers the redraw."""
    assert (RAS_TOP_K, RAS_TOP_P, RAS_WIN, RAS_TAU) == (25, 0.8, 10, 0.1)
    assert RAS_WIN * RAS_TAU == 1.0
    assert (MIN_TOKEN_TEXT_RATIO, MAX_TOKEN_TEXT_RATIO) == (2, 20)
    assert (FLOW_STEPS, FLOW_CFG_RATE) == (10, 0.7)
    assert MAX_SILENT_RUN == 5 and SILENT_TOKENS == (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)


def test_the_speech_rows_and_the_stop_range():
    """`sos` is 6561 and `task_id` 6563; every id from 6561 up to the 6761-wide head's end stops the
    decode, and `sampling_ids`' `ignore_eos` bans 6561 -- which is `sos`, not `eos_token` (6562)."""
    assert (SOS, TASK_ID, LM_HEAD) == (6561, 6563, 6761)
    assert END_OF_PROMPT == 151646
    lm = (Path(__file__).resolve().parents[2] / "loom_exporter" / "cosyvoice3_driver" / "01_lm.lua").read_text()
    assert "if _token >= SOS then break end" in lm
    assert "if _step < _min_len then _banned[1] = SOS end" in lm
    assert "top_p_mass = 'row'" in lm


def test_the_nsf_noise_is_one_symbol_with_the_mel(tmp_path):
    """480 output samples per mel frame, declared as that multiple -- read off the phase list without
    tracing, since `declared_axes` is a literal in `phases()`."""
    src = (Path(__file__).resolve().parents[2] / "loom_exporter" / "cosyvoice3_export.py").read_text()
    assert 'declared_axes={"nsf_noise": {2: f"{SAMPLES_PER_FRAME} * n_enc_frames"}}' in src
    assert SAMPLES_PER_FRAME == 480


# -- the substitutions a trace depends on -----------------------------------------------------------

def test_the_rope_substitution_is_x_transformers_partial_rotation():
    """CosyVoice3's `AttnProcessor` applies rope to the PROJECTED query, `(b, n, heads * 64)`, with a
    64-wide table -- so x_transformers rotates channels [0, 64) and passes the rest through. The
    substitution must be that function, bit for bit, not a per-head rotation."""
    xt = pytest.importorskip("x_transformers.x_transformers")
    from loom_exporter.f5_tts_export import _rope_pair_swap

    torch.manual_seed(0)
    rot = xt.RotaryEmbedding(64)
    freqs, scale = rot.forward_from_seq_len(19)
    t = torch.randn(1, 19, 1024)
    want = xt.apply_rotary_pos_emb(t, freqs, 1.0)
    got = _apply_rope_first_head(t, (freqs.cos(), freqs.sin(), _rope_pair_swap(64)), 1.0)
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.equal(got[..., 64:], t[..., 64:])         # heads 1-15 untouched
    assert not torch.allclose(got[..., :64], t[..., :64])  # head 0 really rotated


def _cosyvoice_checkout():
    if not Path(COSYVOICE_REPO, "cosyvoice").is_dir():
        pytest.skip("no FunAudioLLM/CosyVoice checkout")
    from loom_exporter.cosyvoice3_export import import_cosyvoice, install_patches

    import_cosyvoice()
    install_patches()


@pytest.mark.parametrize("causal_type,kernel,dilation", [("left", 3, 1), ("left", 3, 5), ("left", 7, 1),
                                                          ("right", 4, 1), ("right", 5, 1)])
def test_the_padded_causal_conv_is_the_references(causal_type, kernel, dilation):
    """`F.pad` on the same side by the same width as the reference's `torch.zeros` + concat. The
    reference's own path is still reachable -- a non-empty cache -- and is the oracle here."""
    _cosyvoice_checkout()
    from cosyvoice.transformer.convolution import CausalConv1d

    torch.manual_seed(1)
    conv = CausalConv1d(6, 5, kernel, dilation=dilation, causal_type=causal_type).eval()
    x = torch.randn(1, 6, 23)
    zeros = torch.zeros(1, 6, conv.causal_padding)
    with torch.no_grad():
        assert torch.equal(conv(x), conv(x, zeros))


def test_the_staged_tokenizer_carries_every_added_token(tmp_path):
    """`add_special_tokens` assigns ids in list order after Qwen2's own three; the staged file must hold
    them all, with `<|endofprompt|>` where the LM checks for it."""
    if not (MODEL_DIR / "CosyVoice-BlankEN").is_dir():
        pytest.skip("no Fun-CosyVoice3 checkpoint")
    _cosyvoice_checkout()
    from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
    from loom_exporter.cosyvoice3_export import stage_tokenizer

    stage_tokenizer(str(MODEL_DIR), str(tmp_path))
    written = json.loads((tmp_path / "tokenizer.json").read_text())
    added = {t["content"]: t["id"] for t in written["added_tokens"]}
    reference = CosyVoice3Tokenizer(str(MODEL_DIR / "CosyVoice-BlankEN"))
    for token in reference.special_tokens["additional_special_tokens"]:
        assert added[token] == reference.tokenizer.convert_tokens_to_ids(token)
    assert added["<|endofprompt|>"] == END_OF_PROMPT
