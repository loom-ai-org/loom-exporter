"""Family 9's sixth leaf (P5): VoxCPM2 -- a diffusion-autoregressive TTS whose loop carries patches of
continuous latents, each integrated by a guided local DiT inside the driver's loop.

What is tested here is what the Lua is generated from and what the engine reads as DATA, because a wrong
declaration there is a wrong model with nothing failing:

* the recognizer;
* the text front end the file carries -- its tag, the normalizer strings, the added tokens (and the two
  the config names that the table does not), and the Chinese split table against the reference's rule;
* the Euler schedule and the zero-init count, against `solve_euler` itself;
* every re-spelling the trace needs, against the reference module it replaces, on a TINY random VoxCPM2
  built from the reference's own classes -- so the check needs the checkout but not the 9 GB
  checkpoint. That includes the AudioVAE's weight norm folded BEFORE any forward: its hook refreshes
  `weight` per call, so a check run after the reference's own forward cannot see a stale fold;
* the declarations: the text-door contract and the defaults the driver is generated from.

The numbers -- the waveform against the reference, teacher-forced and free-running, and the text path
at 4992/4992 -- are the engine gate's (`tests/gate/test_e2e_voxcpm2_lua_driver.cpp`) and the export's
docstring's.
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from loom_exporter.voxcpm2_export import (
    AUDIO_START_TOKEN, BADCASE_RATIO, DEFAULT_CFG, DEFAULT_TIMESTEPS, MAX_LEN, MIN_LEN, SAMPLE_RATE,
    SWAY_COEF, VOXCPM_REPO, BaseLMPhase, DiTStepPhase, FeatEncodePhase, ResidualLMPhase, VAEDecodePhase,
    VoxCPM2ExportConfig, _is_voxcpm2, causal_mask, euler_schedule, fold_weight_norm, import_voxcpm,
    zero_init_steps,
)
from loom_exporter.voxcpm2_tokenizer_export import added_tokens, normalizer_strings, split_table
from loom_exporter.registry import default_registry

HAVE_REFERENCE = Path(VOXCPM_REPO, "voxcpm").is_dir()
needs_reference = pytest.mark.skipif(not HAVE_REFERENCE, reason="no OpenBMB/VoxCPM checkout")


def _checkpoint_dir(tmp_path: Path, architecture="voxcpm2", missing=()) -> Path:
    d = tmp_path / "voxcpm2"
    d.mkdir()
    if "config.json" not in missing:
        (d / "config.json").write_text(json.dumps({"architecture": architecture}))
    for name in ("model.safetensors", "audiovae.pth"):
        if name not in missing:
            (d / name).write_bytes(b"")
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_voxcpm2_directory_is_claimed(tmp_path):
    assert _is_voxcpm2(_checkpoint_dir(tmp_path))


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors", "audiovae.pth"])
def test_a_directory_missing_a_release_file_is_not_claimed(tmp_path, missing):
    assert not _is_voxcpm2(_checkpoint_dir(tmp_path, missing=(missing,)))


def test_the_first_voxcpm_is_not_claimed(tmp_path):
    """VoxCPM 1.x ships the same files with `architecture: voxcpm` -- a different DiT and no residual
    split into two projections -- and this export is VoxCPM2's."""
    assert not _is_voxcpm2(_checkpoint_dir(tmp_path, architecture="voxcpm"))


def test_a_malformed_config_is_a_no_not_a_traceback(tmp_path):
    d = _checkpoint_dir(tmp_path)
    (d / "config.json").write_text("{not json")
    assert not _is_voxcpm2(d)


def test_the_registry_routes_a_voxcpm2_directory_to_this_recognizer(tmp_path):
    rec = default_registry().detect(_checkpoint_dir(tmp_path))
    assert rec.name == "voxcpm2"


# -- the text front end ----------------------------------------------------------------------------

def _tokenizer_json(tmp_path: Path, vocab: dict, added=(), normalizer=None) -> Path:
    spec = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>", "byte_fallback": True},
        "normalizer": normalizer or {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
        "pre_tokenizer": None,
        "added_tokens": list(added),
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    return tmp_path


def test_the_normalizer_is_read_and_anything_else_is_refused(tmp_path):
    spec = json.loads((_tokenizer_json(tmp_path, {"<unk>": 0}) / "tokenizer.json").read_text())
    assert normalizer_strings(spec) == {"prepend": "▁", "space": "▁"}
    spec["normalizer"]["normalizers"].insert(0, {"type": "NFKC"})
    with pytest.raises(ValueError, match="not Prepend then Replace"):
        normalizer_strings(spec)


def test_config_added_tokens_join_unless_the_table_disagrees(tmp_path):
    """`tokenizer_config.json` registers `<|audio_start|>` (101) and 14 more; two of its entries name ids
    whose rows are something else, and transformers gives those spellings ids past the embedding."""
    vocab = {"<unk>": 0, "<s>": 1, "<|audio_start|>": 101, "<|ref_audio_start|>": 103}
    d = _tokenizer_json(tmp_path, vocab, added=[{"id": 0, "content": "<unk>", "special": True}])
    (d / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {
        "1": {"content": "<s>", "special": True},
        "101": {"content": "<|audio_start|>", "special": True},
        "103": {"content": "<|audio_prompt_start|>", "special": True}}}))
    spec = json.loads((d / "tokenizer.json").read_text())
    assert [a["content"] for a in added_tokens(str(d), spec)] == ["<unk>", "<s>", "<|audio_start|>"]


def test_a_stripping_added_token_is_refused(tmp_path):
    d = _tokenizer_json(tmp_path, {"<unk>": 0}, added=[{"id": 0, "content": "<unk>", "lstrip": True}])
    with pytest.raises(ValueError, match="not a plain literal"):
        added_tokens(str(d), json.loads((d / "tokenizer.json").read_text()))


@needs_reference
def test_the_split_table_is_the_reference_wrappers_rule():
    """`mask_multichar_chinese_tokens` over a small table: a piece whose `▁`-stripped text is itself a
    multi-character Chinese piece becomes its characters -- with or without the `▁`, and an absent
    character becomes `<unk>` as `convert_tokens_to_ids` makes it."""
    import_voxcpm()
    from voxcpm.model.utils import mask_multichar_chinese_tokens

    tokens = ["<unk>", "▁", "你", "好", "你好", "▁你好", "世界", "▁a", "ab", "中国人"]

    class Table:
        vocab = {t: i for i, t in enumerate(tokens)}
        unk = 0

        def tokenize(self, text, **_):
            return text.split("|")

        def convert_tokens_to_ids(self, pieces):
            return [self.vocab.get(p, self.unk) for p in pieces]

    wrapper = mask_multichar_chinese_tokens(Table())
    table = split_table(tokens, 0)
    for i, piece in enumerate(tokens):
        assert table.get(i, [i]) == wrapper(piece), piece
    assert table[tokens.index("中国人")] == [0, 0, 0]


# -- the Euler schedule ----------------------------------------------------------------------------

@needs_reference
def test_the_schedule_is_solve_eulers_own():
    """`(t, dt)` per step as `solve_euler` forms them: `t` read off the estimator's own calls (exact
    equality), and the `dt`s through the update itself -- a unit velocity makes `x` the running sum."""
    import_voxcpm()
    from voxcpm.modules.locdit.unified_cfm import CfmConfig, UnifiedCFM

    seen = []

    class Unit(torch.nn.Module):
        def forward(self, x, mu, t, cond, dt):
            seen.append(float(t[0]))
            return torch.ones_like(x)

    n = DEFAULT_TIMESTEPS
    cfm = UnifiedCFM(in_channels=1, cfm_params=CfmConfig(), estimator=Unit())
    t_span = torch.linspace(1, 0, n + 1)
    t_span = t_span + SWAY_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    x = cfm.solve_euler(x=torch.zeros(1, 1, 1), t_span=t_span, mu=torch.zeros(1, 1), cond=torch.zeros(1, 1, 1),
                        cfg_value=DEFAULT_CFG, use_cfg_zero_star=True)
    rows = euler_schedule(n).reshape(-1, 2)
    skip = zero_init_steps(n)
    assert skip == 1
    assert seen == [float(t) for t, _ in rows[skip:]]
    mine = torch.zeros(1)
    for _, dt in rows[skip:]:
        mine = mine - torch.tensor([dt])
    assert float(x.reshape(-1)[0]) == float(mine[0])


# -- the re-spellings, against a tiny random VoxCPM2 -----------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    """A VoxCPM2 from the reference's own classes at toy sizes, with EVERY parameter randomised (the
    norms' ones, the sample-rate scales and the weight-norm magnitudes included), so no re-spelling can
    pass by multiplying by a one."""
    if not HAVE_REFERENCE:
        pytest.skip("no OpenBMB/VoxCPM checkout")
    import_voxcpm()
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

    torch.manual_seed(0)
    kv = 16
    rope = {"type": "longrope", "long_factor": [1.0 + i / 8 for i in range(kv // 2)],
            "short_factor": [1.0 + i / 16 for i in range(kv // 2)], "original_max_position_embeddings": 64}
    config = VoxCPMConfig.model_validate({
        "lm_config": {"bos_token_id": 1, "eos_token_id": 2, "hidden_size": 64, "intermediate_size": 96,
                      "max_position_embeddings": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
                      "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "rope_scaling": rope, "vocab_size": 50,
                      "use_mup": False, "scale_emb": 12, "dim_model_base": 256, "scale_depth": 1.4,
                      "rope_theta": 10000, "kv_channels": kv},
        "patch_size": 4, "feat_dim": 8, "scalar_quantization_latent_dim": 16, "scalar_quantization_scale": 9,
        "residual_lm_num_layers": 2, "residual_lm_no_rope": True,
        "encoder_config": {"hidden_dim": 32, "ffn_dim": 48, "num_heads": 4, "num_layers": 2, "kv_channels": kv},
        "dit_config": {"hidden_dim": 32, "ffn_dim": 48, "num_heads": 4, "num_layers": 2, "kv_channels": kv,
                       "cfm_config": {"sigma_min": 1e-6, "solver": "euler", "t_scheduler": "log-norm",
                                      "inference_cfg_rate": 2.0}},
        "device": "cpu", "dtype": "float32"})
    vae = AudioVAEV2(config=AudioVAEConfigV2(encoder_dim=4, encoder_rates=[2, 2], latent_dim=8, decoder_dim=16,
                                             decoder_rates=[2, 3], sample_rate=16000, out_sample_rate=48000))

    class Table:
        vocab = {}

    model = VoxCPM2Model(config, Table(), vae, device="cpu").float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3 + (0.5 if p.dim() == 1 else 0.0))
    return model


def test_the_feature_encoder_is_voxcpm_loc_enc(tiny):
    feats = torch.randn(1, 5, 4, 8)
    with torch.no_grad():
        ref = tiny.enc_to_lm_proj(tiny.feat_encoder(feats))
        assert torch.allclose(FeatEncodePhase(tiny)(feats), ref, atol=1e-5)


def test_the_two_lms_are_the_references_prefill_then_steps(tiny):
    """A prefill whose last rows are audio (so the FSQ path is exercised), then steps, each against the
    reference's `forward_step` over its own static cache. The phases' `pasts` stand in for the
    engine's cache; the stop logits are checked where the reference computes them."""
    n = 7
    ids = torch.randint(3, 50, (1, n))
    tm = torch.ones(1, n, 1)
    tm[:, 5:] = 0
    am = 1 - tm
    with torch.no_grad():
        fe = tiny.enc_to_lm_proj(tiny.feat_encoder(torch.randn(1, n, 4, 8)))
        h, kv = tiny.base_lm(inputs_embeds=tm * tiny.base_lm.embed_tokens(ids) + am * fe, is_causal=True)
        tiny.base_lm.kv_cache.fill_caches(kv)
        enc_ref = tiny.fsq_layer(h) * am + h * tm
        base, res = BaseLMPhase(tiny), ResidualLMPhase(tiny)
        pasts = [[None, None] for _ in base.stack.layers]
        enc, last, stop = base(ids.int(), fe, tm, am, torch.arange(n).view(1, -1), causal_mask(n), pasts)
        assert torch.allclose(enc, enc_ref, atol=1e-5)
        assert torch.allclose(stop, tiny.stop_head(tiny.stop_actn(tiny.stop_proj(enc_ref[:, -1:]))), atol=1e-5)
        r_ref, rkv = tiny.residual_lm(inputs_embeds=tiny.fusion_concat_proj(torch.cat((enc_ref, am * fe), -1)),
                                      is_causal=True)
        tiny.residual_lm.kv_cache.fill_caches(rkv)
        rpasts = [[None, None] for _ in res.stack.layers]
        assert torch.allclose(res(enc, fe, am, causal_mask(n), rpasts), r_ref[:, -1:], atol=1e-5)
        for _ in range(3):
            e = tiny.enc_to_lm_proj(tiny.feat_encoder(torch.randn(1, 1, 4, 8)))
            pos = tiny.base_lm.kv_cache.step()
            want = tiny.fsq_layer(tiny.base_lm.forward_step(e[:, 0], torch.tensor([pos])))
            got, _, _ = base(torch.zeros(1, 1, dtype=torch.int32), e, torch.zeros(1, 1, 1), torch.ones(1, 1, 1),
                             torch.tensor([[pos]]), torch.zeros(1, 1, 1, pos + 1), pasts)
            assert torch.allclose(got[:, 0], want, atol=1e-5)
            rpos = tiny.residual_lm.kv_cache.step()
            rr = tiny.residual_lm.forward_step(tiny.fusion_concat_proj(torch.cat((want, e[:, 0]), -1)),
                                               torch.tensor([rpos]))
            assert torch.allclose(res(got, e, torch.ones(1, 1, 1), torch.zeros(1, 1, 1, rpos + 1), rpasts)[:, 0],
                                  rr, atol=1e-5)


def test_the_dit_steps_are_the_guided_euler_solve(tiny):
    """Every step of `dit_step`, fed the export's schedule and skipping the zero-init steps, is
    `solve_euler` with CFG-Zero* on the same draw. The patch is PATCH-major here and channel-major in
    the reference -- a transposed layout would still be a plausible patch, so this is the check."""
    dit = DiTStepPhase(tiny)
    lm_h, res_h = torch.randn(1, 1, 64), torch.randn(1, 1, 64)
    cond, z = torch.randn(1, 4, 8), torch.randn(1, 8, 4)
    with torch.no_grad():
        mu = torch.cat((tiny.lm_to_dit_proj(lm_h[:, 0]), tiny.res_to_dit_proj(res_h[:, 0])), dim=-1)
        t_span = torch.linspace(1, 0, DEFAULT_TIMESTEPS + 1)
        t_span = t_span + SWAY_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        ref = tiny.feat_decoder.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond.transpose(1, 2).contiguous(),
                                            cfg_value=DEFAULT_CFG, use_cfg_zero_star=True)
        x = z.transpose(1, 2).contiguous()
        for t, dt in euler_schedule(DEFAULT_TIMESTEPS).reshape(-1, 2)[zero_init_steps(DEFAULT_TIMESTEPS):]:
            x = dit(lm_h, res_h, cond, x, torch.tensor([[t]]), torch.tensor([[dt]]), torch.tensor([[DEFAULT_CFG]]))
    assert torch.allclose(x, ref.transpose(1, 2), atol=1e-5)


def test_the_vae_decode_folds_weight_norm_before_any_forward(tiny):
    """The fold runs on a module that has NEVER run forward: the old-style weight-norm hook leaves
    `weight` at its initialisation until the first call, and a fold that read it would still pass a
    check made after the reference had decoded once."""
    import copy

    import_voxcpm()
    from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

    fresh = AudioVAEV2(config=AudioVAEConfigV2(encoder_dim=4, encoder_rates=[2, 2], latent_dim=8, decoder_dim=16,
                                               decoder_rates=[2, 3], sample_rate=16000, out_sample_rate=48000))
    fresh.load_state_dict(tiny.audio_vae.state_dict())
    fold_weight_norm(fresh)
    lat = torch.randn(1, 8, 6)
    with torch.no_grad():
        want = tiny.audio_vae.decode(lat)[:, 0]
        got = VAEDecodePhase(fresh.eval())(lat.transpose(1, 2))
    assert got.shape == want.shape == (1, 6 * 2 * 3)
    assert torch.allclose(got, want, atol=1e-5)
    del copy


# -- the declarations ------------------------------------------------------------------------------

def test_the_tokenizer_is_named_rather_than_detected(tmp_path):
    kwargs = VoxCPM2ExportConfig(output_path=str(tmp_path / "x.gguf"), model_dir=str(tmp_path)).backend_kwargs()
    assert kwargs["tokenizer_family"] == "voxcpm2"


def test_the_contract_declares_a_text_door_at_48k(tmp_path):
    contract = VoxCPM2ExportConfig(output_path=str(tmp_path / "x.gguf"), model_dir=str(tmp_path)).contract()
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == SAMPLE_RATE == 48000


@needs_reference
def test_the_defaults_are_the_references_own():
    """`VoxCPM.generate`'s defaults (cfg, steps, lengths, the badcase ratio), `UnifiedCFM.forward`'s sway
    coefficient and `VoxCPM2Model`'s audio-start id -- read off the reference rather than restated."""
    import_voxcpm()
    from voxcpm.core import VoxCPM
    from voxcpm.modules.locdit.unified_cfm import UnifiedCFM

    gen = inspect.signature(VoxCPM._generate).parameters
    assert gen["cfg_value"].default == DEFAULT_CFG
    assert gen["inference_timesteps"].default == DEFAULT_TIMESTEPS
    assert gen["min_len"].default == MIN_LEN
    assert gen["max_len"].default == MAX_LEN
    assert gen["retry_badcase_ratio_threshold"].default == BADCASE_RATIO
    assert inspect.signature(UnifiedCFM.forward).parameters["sway_sampling_coef"].default == SWAY_COEF
    source = inspect.getsource(__import__("voxcpm.model.voxcpm2", fromlist=["x"]).VoxCPM2Model.__init__)
    assert f"self.audio_start_token = {AUDIO_START_TOKEN}" in source
