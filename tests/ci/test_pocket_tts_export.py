"""Family 9's fifth leaf (P5): Pocket-TTS -- a flow LM whose autoregressive loop carries continuous
latents, with its voices shipped as KV caches.

What is tested here is what the Lua is generated from and what the engine reads as DATA, because a wrong
declaration there is a wrong model with nothing failing:

* the recognizer, and the voice file's layout as `loom.seed_kv` will read it (loom.cpp ADR-043);
* the text front end the file carries -- its tag, and the case and word-start tables it derives;
* the re-spellings the trace needs, each against the reference module it replaces: the interleaved
  RoPE, the windowed mask, the flow head's norms, and the one-shot Mimi decode;
* the declarations: the text-door contract, the cached LM, Mimi's one-symbol position axis.

The numbers -- a waveform at rmse 1.8e-06 teacher-forced, and the text path at 7000/7000 -- are the
engine gate's (`tests/gate/test_e2e_pocket_tts_lua_driver.cpp`) and the export's docstring's.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from loom_exporter.pocket_tts_export import (
    DEFAULT_DECODE_STEPS, DEFAULT_EOS_THRESHOLD, GEN_SECONDS_PADDING, LM_MAX_POSITIONS,
    MAX_TOKEN_PER_CHUNK, MIMI_UPSAMPLE, MIN_FRAMES_BEFORE_EOS, POCKET_TTS_REPO,
    TOKENS_PER_SECOND_ESTIMATE, PocketTTSExportConfig, _is_pocket_tts, _rope_tables, _rotate_pairs,
    read_voice,
)
from loom_exporter.pocket_tts_tokenizer_export import _case_table
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/pocket-tts/languages/english_2026-09")
HAVE_REFERENCE = Path(POCKET_TTS_REPO, "pocket_tts").is_dir()
needs_reference = pytest.mark.skipif(not HAVE_REFERENCE, reason="no pocket-tts checkout")
needs_checkpoint = pytest.mark.skipif(not (HAVE_REFERENCE and MODEL_DIR.is_dir()),
                                      reason="no pocket-tts checkpoint and checkout")


def _language_dir(tmp_path: Path, keys=("flow_lm.bos_emb", "mimi.quantizer.output_proj.weight"),
                  missing=()) -> Path:
    from safetensors.numpy import save_file

    d = tmp_path / "languages" / "english_2026-09"
    (d / "embeddings").mkdir(parents=True)
    if "model.safetensors" not in missing:
        save_file({k: np.zeros(2, dtype=np.float32) for k in keys}, str(d / "model.safetensors"))
    if "tokenizer.model" not in missing:
        (d / "tokenizer.model").write_bytes(b"")
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_language_directory_is_claimed(tmp_path):
    assert _is_pocket_tts(_language_dir(tmp_path))


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.model"])
def test_a_directory_missing_a_release_file_is_not_claimed(tmp_path, missing):
    assert not _is_pocket_tts(_language_dir(tmp_path, missing=(missing,)))


def test_safetensors_without_both_halves_is_not_claimed(tmp_path):
    """The bundled weights are a `flow_lm.` and a `mimi.` half; any other safetensors beside a
    SentencePiece model is somebody else's checkpoint."""
    assert not _is_pocket_tts(_language_dir(tmp_path, keys=("flow_lm.bos_emb",)))


def test_a_malformed_safetensors_is_a_no_not_a_traceback(tmp_path):
    d = _language_dir(tmp_path)
    (d / "model.safetensors").write_bytes(b"not a safetensors file")
    assert not _is_pocket_tts(d)


def test_the_registry_routes_a_language_directory_to_this_recognizer(tmp_path):
    # `detect` raises on more than one match, so this also pins that no other family claims it.
    assert default_registry().detect(_language_dir(tmp_path)).name == "pocket-tts"


# -- the voice, as the engine will seed it ---------------------------------------------------------

def _voice(tmp_path: Path, n_layers=2, n_rows=3, heads=2, dim=2, offset=None, nan=False) -> Path:
    from safetensors.torch import save_file

    tensors = {}
    for i in range(n_layers):
        cache = torch.arange(2 * n_rows * heads * dim, dtype=torch.float32).view(2, 1, n_rows, heads, dim)
        cache = cache + 1000 * i
        if nan:
            cache[0, 0, 0, 0, 0] = float("nan")
        tensors[f"transformer.layers.{i}.self_attn/cache"] = cache
        tensors[f"transformer.layers.{i}.self_attn/offset"] = torch.tensor([offset or n_rows])
    path = tmp_path / "voice.safetensors"
    save_file(tensors, str(path))
    return path


def test_a_voice_flattens_per_layer_k_then_v_with_heads_inside_a_row(tmp_path):
    """`loom.seed_kv`'s layout: per layer, K `[n, heads * dim]` then V, each row the head axes
    flattened in the order an ATTENTION node writes one."""
    flat, n = read_voice(_voice(tmp_path), n_layers=2)
    assert n == 3
    per_layer = 2 * 3 * 4
    for layer in range(2):
        block = flat[layer * per_layer:(layer + 1) * per_layer]
        np.testing.assert_array_equal(block, np.arange(per_layer, dtype=np.float32) + 1000 * layer)


def test_a_padded_voice_is_refused(tmp_path):
    """A cache longer than its offset would seed rows the voice never wrote."""
    with pytest.raises(ValueError, match="offset"):
        read_voice(_voice(tmp_path, offset=2), n_layers=2)


def test_a_voice_holding_nan_is_refused(tmp_path):
    with pytest.raises(ValueError, match="NaN"):
        read_voice(_voice(tmp_path, nan=True), n_layers=2)


# -- the text front end ----------------------------------------------------------------------------

def test_the_case_table_is_what_prepare_text_prompt_applies():
    """`text[0].upper()` unless `text[0].isupper()`: full mappings, titlecase letters included, and
    nothing that is already upper case."""
    table = dict(zip(*_case_table()))
    assert table["a"] == "A" and table["ß"] == "SS" and table["ﬁ"] == "FI"
    assert table["ǅ"] == "Ǆ"          # titlecase Dz: not isupper(), so it IS upper-cased
    assert "A" not in table and "1" not in table and " " not in table


def test_the_tokenizer_is_named_rather_than_detected(tmp_path):
    """Detected, the `.model` is plain "sentencepiece_proto": it tokenizes, but does not prepare or
    chunk the text the way the reference does before any id reaches the model."""
    config = PocketTTSExportConfig(output_path=str(tmp_path / "o.gguf"),
                                   model_dir=str(_language_dir(tmp_path)))
    assert config.backend_kwargs()["tokenizer_family"] == "pocket_tts"


def test_the_contract_declares_a_text_door(tmp_path):
    config = PocketTTSExportConfig(output_path=str(tmp_path / "o.gguf"),
                                   model_dir=str(_language_dir(tmp_path)))
    config.task = "text-to-speech"
    contract = config.contract()
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == 24000


@needs_reference
def test_the_defaults_are_the_references_own():
    sys.path.insert(0, POCKET_TTS_REPO)
    from pocket_tts import default_parameters as d
    from pocket_tts.models.tts_model import TTSModel

    assert DEFAULT_EOS_THRESHOLD == d.DEFAULT_EOS_THRESHOLD
    assert DEFAULT_DECODE_STEPS == d.DEFAULT_SAMPLER_DECODE_STEPS
    assert MAX_TOKEN_PER_CHUNK == d.MAX_TOKEN_PER_CHUNK
    assert d.DEFAULT_NOISE_CLAMP is None               # the driver never clamps its draw
    assert TOKENS_PER_SECOND_ESTIMATE == TTSModel._TOKENS_PER_SECOND_ESTIMATE
    assert GEN_SECONDS_PADDING == TTSModel._GEN_SECONDS_PADDING
    assert MIN_FRAMES_BEFORE_EOS == TTSModel._MIN_FRAMES_BEFORE_EOS


@needs_checkpoint
def test_word_start_flags_count_what_split_counts():
    """The driver's `len(text.split())` for `frames_after_eos`, read off ids. Exact for ASCII-space
    text; a tab or NBSP between words is the documented miss (47 of 9452 generated chunks)."""
    import sentencepiece as spm

    from loom_exporter.pocket_tts_export import word_start_flags

    sp = spm.SentencePieceProcessor(str(MODEL_DIR / "tokenizer.model"))
    flags = word_start_flags(sp)

    def count(ids):
        n = 0
        for i, t in enumerate(ids):
            if flags[t] == 1 or (flags[t] == 2 and (i + 1 == len(ids) or flags[ids[i + 1]] == 0)):
                n += 1
        return n

    for text in ["Hello world.", "One two three four.", "Hi, (there) - you!", "A  b.", "x"]:
        assert count(sp.encode(text)) == len(text.split()), text


# -- the re-spellings, each against the module it replaces -----------------------------------------

@needs_reference
def test_rotate_pairs_rope_is_apply_rope():
    sys.path.insert(0, POCKET_TTS_REPO)
    from pocket_tts.modules.rope import apply_rope

    torch.manual_seed(0)
    q, k = torch.randn(1, 5, 3, 8), torch.randn(1, 5, 3, 8)
    want_q, want_k = apply_rope(q, k, offset=17, max_period=10000.0)
    cos, sin = _rope_tables(torch.arange(17, 22).view(1, -1), 8, 10000.0)
    torch.testing.assert_close(q * cos + _rotate_pairs(q) * sin, want_q, rtol=0, atol=1e-6)
    torch.testing.assert_close(k * cos + _rotate_pairs(k) * sin, want_k, rtol=0, atol=1e-6)


@needs_checkpoint
def test_the_wrappers_reproduce_the_reference_modules():
    """Each phase against the reference module it re-spells, on random inputs: the LM's hidden row
    and EOS logit (no cache), one flow-head step, and Mimi over enough frames that the attention
    window (250 decoder positions) is exceeded."""
    from loom_exporter.pocket_tts_export import (
        FlowHeadPhase, LMPhase, MimiDecoderPhase, causal_mask, load_reference,
    )
    from pocket_tts.modules.stateful_module import init_states

    tts, config = load_reference(str(MODEL_DIR))
    fl = tts.flow_lm
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.randn(1, 9, fl.dim)
        lm = LMPhase(fl, float(config["flow_lm"]["transformer"]["max_period"]))
        hidden, eos = lm(x, torch.arange(9).view(1, -1), causal_mask(9))
        want = fl.out_norm(fl.transformer(x, None))[:, -1:]
        torch.testing.assert_close(hidden, want, rtol=0, atol=1e-4)
        torch.testing.assert_close(eos, fl.out_eos(want), rtol=0, atol=1e-4)

        c, z = torch.randn(1, fl.dim), torch.randn(1, fl.ldim)
        std = math.sqrt(0.3)
        got = FlowHeadPhase(fl.flow_net)(c, torch.zeros(1, 1), torch.ones(1, 1), z,
                                         torch.tensor([[std]]), torch.ones(1, 1))
        x0 = z * std
        want = x0 + fl.flow_net(c, torch.zeros(1, 1), torch.ones(1, 1), x0)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5)

        n = 20                                  # 320 decoder positions > the 250-position window
        latents = torch.randn(1, n, fl.ldim)
        got = MimiDecoderPhase(fl, tts.mimi)(latents, torch.arange(n * MIMI_UPSAMPLE).view(1, -1))
        state = init_states(tts.mimi, batch_size=1, sequence_length=n * MIMI_UPSAMPLE)
        want = tts.mimi.decode_from_latent(latents * fl.emb_std + fl.emb_mean, state)[:, 0]
        torch.testing.assert_close(got, want, rtol=0, atol=1e-4)


# -- declarations ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phases():
    if not (HAVE_REFERENCE and MODEL_DIR.is_dir()):
        pytest.skip("no pocket-tts checkpoint and checkout to trace")
    return {p.name: p for p in PocketTTSExportConfig(
        output_path="/dev/null", model_dir=str(MODEL_DIR)).phases()}


def test_the_lm_is_cached_and_its_attention_fused(phases):
    lm = phases["lm"]
    assert lm.fuse_attention is True and lm.kv_cache_size == LM_MAX_POSITIONS
    assert not lm.extra_streams                  # no guidance: one stream


def test_mimis_positions_are_one_symbol_with_its_frames(phases):
    """16 decoder positions per latent, declared as that multiple, so the window mask the graph builds
    from them and the latents it decodes cannot disagree about the length."""
    assert phases["mimi_decoder"].declared_axes == {"positions": {1: f"{MIMI_UPSAMPLE} * n_codes"}}


# -- voice files (loom.cpp ADR-045) and the chunk headers ------------------------------------------

def test_the_fingerprint_follows_the_flow_lm_and_ignores_mimi(tmp_path):
    """Mimi is excluded because the release without voice cloning zeroes its encoder, and a voice made
    with either release is the same voice; any `flow_lm` change must change it."""
    from safetensors.numpy import save_file

    from loom_exporter.pocket_tts_voices import weights_fingerprint

    def fp(flow, mimi):
        path = tmp_path / f"{flow}_{mimi}.safetensors"
        save_file({"flow_lm.bos_emb": np.full(4, flow, np.float32),
                   "mimi.encoder.w": np.full(4, mimi, np.float32)}, str(path))
        return weights_fingerprint(path)

    assert fp(1, 1) == fp(1, 2)
    assert fp(1, 1) != fp(2, 1)


def test_a_written_voice_file_is_the_seed_layout_under_its_input_name(tmp_path):
    import gguf

    from loom_exporter.pocket_tts_voices import write_voice

    src = _voice(tmp_path)
    out = tmp_path / "voices" / "v.gguf"
    n = write_voice(src, out, name="v", compat="ab" * 16, n_layers=2, license="CC0-1.0", origin="o")
    assert n == 3
    reader = gguf.GGUFReader(str(out))
    fields = {f.name: f.contents() for f in reader.fields.values() if f.name.startswith("loom.voice.")}
    assert fields == {"loom.voice.architecture": "pocket-tts", "loom.voice.compat": "ab" * 16,
                      "loom.voice.name": "v", "loom.voice.license": "CC0-1.0",
                      "loom.voice.origin": "o", "loom.voice.n_rows": 3}
    tensor, = reader.tensors
    assert tensor.name == "voice_kv"
    np.testing.assert_array_equal(np.asarray(tensor.data), read_voice(src, 2)[0])


def test_your_own_voice_needs_a_name_and_the_recordings_licence(tmp_path):
    from loom_exporter.pocket_tts_voices import convert

    with pytest.raises(ValueError, match="--license"):
        convert(_language_dir(tmp_path), tmp_path / "out", source=_voice(tmp_path), name="me")


@needs_reference
def test_each_predefined_voice_carries_its_recordings_licence():
    """Per `kyutai/tts-voices`' README: two of them are NON-COMMERCIAL, and two state none."""
    from loom_exporter.pocket_tts_voices import UNSTATED, origin_and_license

    assert origin_and_license("alba")[1] == "CC-BY-4.0"
    assert origin_and_license("anna")[1] == "CC-BY-4.0"
    assert origin_and_license("marius")[1] == "CC0-1.0"
    assert origin_and_license("estelle")[1] == "CC0-1.0"
    assert origin_and_license("giovanni")[1] == "CC0-1.0"
    assert origin_and_license("cosette")[1] == "CC-BY-NC-4.0"
    assert origin_and_license("jean")[1] == "CC-BY-NC-4.0"
    assert origin_and_license("juergen")[1] == UNSTATED
    assert origin_and_license("not_a_voice")[1] == UNSTATED


def test_the_contract_declares_the_voice_fingerprint_and_the_builtin_voice(tmp_path):
    from loom_exporter.pocket_tts_voices import weights_fingerprint

    d = _language_dir(tmp_path)
    config = PocketTTSExportConfig(output_path=str(tmp_path / "o.gguf"), model_dir=str(d))
    config.task = "text-to-speech"
    contract = config.contract()
    assert contract["voice.compat"] == weights_fingerprint(d / "model.safetensors")
    assert contract["tts.voices"] == ["alba"]


@needs_reference
def test_the_chunk_header_constants_are_prepare_text_prompts():
    """The tail the headers stand for, and the word threshold between them, against the function."""
    sys.path.insert(0, POCKET_TTS_REPO)
    from pocket_tts.models.text_chunking import prepare_text_prompt

    from loom_exporter.pocket_tts_tokenizer_export import (
        LONG_CHUNK_FRAMES_AFTER_EOS, SHORT_CHUNK_FRAMES_AFTER_EOS, SHORT_CHUNK_MAX_WORDS,
    )

    words = " ".join(["word"] * SHORT_CHUNK_MAX_WORDS)
    assert prepare_text_prompt(words, False, False)[1] == SHORT_CHUNK_FRAMES_AFTER_EOS
    assert prepare_text_prompt(words + " more", False, False)[1] == LONG_CHUNK_FRAMES_AFTER_EOS
    # Counted as `str.split()` counts: a tab separates words.
    assert prepare_text_prompt("\t".join(["w"] * 5), False, False)[1] == LONG_CHUNK_FRAMES_AFTER_EOS
