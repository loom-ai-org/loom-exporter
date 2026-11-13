"""Family 9's fourth leaf (P5): Chatterbox -- an AR token LM (T3) and a flow-matching decoder (S3Gen) in
one GGUF.

What is tested here is what the Lua is GENERATED from and what the engine reads as DATA, because a wrong
declaration there is a wrong model with nothing failing:

* the recognizer, which must not claim the multilingual or Turbo checkpoints;
* the text front end the file carries -- its tag, its refusal of a `tokenizer.json` that is not this
  scheme, the pre-tokenizer's word set, and `punc_norm`'s rules, which are checked against the
  reference's own function when the checkout is importable;
* the sampler's declaration (guided, caller-scheduled, caller noise) and the text-door contract;
* the LAYOUT every phase agrees on, which is where F5-TTS's cost was (Retro-052).

The numbers -- a waveform at 2.5e-05 from the reference, and the tokenizer at 3000/3000 -- are the engine
gate's (`tests/gate/test_e2e_chatterbox_lua_driver.cpp`) and the export's docstring's.
"""
import json
import sys
from pathlib import Path

import pytest

from loom_exporter.chatterbox_export import (
    DEFAULT_CFG_WEIGHT, DEFAULT_EXAGGERATION, DEFAULT_MIN_P, DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, FLOW_CFG_RATE, FLOW_STEPS, SAMPLES_PER_FRAME,
    ChatterboxExportConfig, _is_chatterbox,
)
from loom_exporter.chatterbox_tokenizer_export import (
    EMPTY_TEXT, PUNC_REPLACEMENTS, SENTENCE_ENDERS, TERMINAL, read_tokenizer, upper_case_table,
    word_chars, write_chatterbox_vocab,
)
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/chatterbox")
RELEASE_FILES = ("t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt")


def _release(tmp_path: Path, missing=()) -> Path:
    d = tmp_path / "chatterbox"
    d.mkdir()
    for name in RELEASE_FILES:
        if name not in missing:
            (d / name).write_bytes(b"")
    return d


def _tokenizer_json(tmp_path: Path, **overrides) -> Path:
    """A miniature of the real file: the three leading control rows, an event tag, characters and
    two merges -- with every field `read_tokenizer` checks set the way Chatterbox's is."""
    vocab = {"[STOP]": 0, "[UNK]": 1, "[SPACE]": 2, "[laughter]": 3, "a": 4, "b": 5, ".": 6,
             " ": 7, "ab": 8, "_": 9}
    spec = {
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "added_tokens": [{"id": i, "content": t, "special": True}
                         for t, i in vocab.items() if t.startswith("[")],
        "model": {"type": "BPE", "unk_token": "[UNK]", "dropout": None, "fuse_unk": False,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "vocab": vocab, "merges": ["a b"]},
    }
    for key, value in overrides.items():
        if key.startswith("model."):
            spec["model"][key[6:]] = value
        else:
            spec[key] = value
    d = tmp_path / "tok"
    d.mkdir(exist_ok=True)
    (d / "tokenizer.json").write_text(json.dumps(spec))
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_release_directory_is_claimed(tmp_path):
    assert _is_chatterbox(_release(tmp_path))


@pytest.mark.parametrize("missing", RELEASE_FILES)
def test_a_directory_missing_any_release_file_is_not_claimed(tmp_path, missing):
    """`t3_cfg` is the discriminator (the multilingual T3 is `t3_mtl23ls_*`, Turbo's is GPT-2), and
    `conds.pt` is the built-in voice, without which `infer(text)` has no voice to speak in."""
    assert not _is_chatterbox(_release(tmp_path, missing=(missing,)))


def test_the_registry_routes_a_release_to_this_recognizer(tmp_path):
    # `detect` raises on more than one match, so this also pins that no other family claims it.
    assert default_registry().detect(_release(tmp_path)).name == "chatterbox"


# -- the text front end ----------------------------------------------------------------------------

def test_a_tokenizer_that_is_not_this_scheme_is_refused_by_name(tmp_path):
    """Each field is one `ChatterboxVocab` assumes. A byte-level BPE or a normalizer would load and
    tokenize -- wrongly -- so they are refused at export rather than discovered in audio."""
    for i, (field, value) in enumerate((("normalizer", {"type": "NFC"}),
                                        ("pre_tokenizer", {"type": "ByteLevel"}),
                                        ("model.fuse_unk", True),
                                        ("model.byte_fallback", True),
                                        ("model.continuing_subword_prefix", "##"))):
        case = tmp_path / str(i)
        case.mkdir()
        d = _tokenizer_json(case, **{field: value})
        with pytest.raises(ValueError, match=field.split(".")[-1]):
            read_tokenizer(str(d))


def test_word_chars_are_the_reference_pre_tokenizers_own_answer():
    """`\\w` is decided by `tokenizers`' own `Whitespace`, per character: a letter and `_` join a
    word, punctuation does not, and whitespace is not a class at all. `\\u00a0` is the case that makes
    asking worth it -- HF treats it as PUNCTUATION, which no reading of "whitespace" would predict."""
    got = word_chars(["a", "_", ".", " ", " ", "ab"])
    assert got == ["a", "_"]


def test_the_case_table_is_pythons_full_mapping():
    """`punc_norm` upper-cases the first character with `str.upper()`, which is not single-codepoint."""
    table = dict(upper_case_table())
    assert table["a"] == "A"
    assert table["ß"] == "SS"
    assert table["ﬁ"] == "FI"
    assert "A" not in table          # only codepoints whose `islower()` holds
    assert len(table) == 1494


def test_the_table_is_written_under_its_own_tag(tmp_path):
    import numpy as np
    from gguf import GGUFReader, GGUFWriter

    out = tmp_path / "t.gguf"
    w = GGUFWriter(str(out), "chatterbox-test")
    write_chatterbox_vocab(w, str(_tokenizer_json(tmp_path)))
    w.add_tensor("x", np.zeros(1, dtype="float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    r = GGUFReader(str(out))
    fields = {f.name: f for f in r.fields.values()}

    def strings(name):
        f = fields[name]
        return [bytes(f.parts[i]).decode() for i in f.data]

    assert bytes(fields["tokenizer.ggml.model"].parts[-1]).decode() == "chatterbox"
    assert strings("tokenizer.ggml.merges") == ["a b"]
    assert int(fields["tokenizer.ggml.unknown_token_id"].parts[-1][0]) == 1
    p = "tokenizer.ggml.chatterbox."
    assert strings(p + "added_tokens") == ["[STOP]", "[UNK]", "[SPACE]", "[laughter]"]
    assert strings(p + "word_chars") == ["a", "b", "_"]
    # The order of the replacement pairs is part of the function: `...` must be seen before `.` rules.
    assert strings(p + "replace_from") == [f for f, _ in PUNC_REPLACEMENTS]
    assert strings(p + "sentence_enders") == list(SENTENCE_ENDERS)
    assert strings(p + "decode_drop") == ["[STOP]", "[UNK]"]
    assert bytes(fields[p + "space_token"].parts[-1]).decode() == "[SPACE]"


CHATTERBOX_SRC = Path("/home/flavio/Dev/chatterbox/src")


def _punc_norm():
    if not CHATTERBOX_SRC.is_dir():
        pytest.skip("no Chatterbox checkout to compare against")
    from loom_exporter.chatterbox_export import import_chatterbox

    import_chatterbox()
    try:
        from chatterbox.tts import punc_norm
    except ImportError as exc:
        pytest.skip(f"Chatterbox's tts module is not importable: {exc}")
    return punc_norm


def _normalize_from_the_exported_rules(text: str) -> str:
    """`ChatterboxVocab::normalize`'s SHAPE, driven only by what the export writes. If the constants
    in `chatterbox_tokenizer_export` drift from the reference's, this and `punc_norm` disagree."""
    if not text:
        return EMPTY_TEXT
    upper = dict(upper_case_table())
    text = upper.get(text[0], text[0]) + text[1:]
    text = " ".join(text.split())
    for old, new in PUNC_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(e) for e in SENTENCE_ENDERS):
        text += TERMINAL
    return text


@pytest.mark.parametrize("text", [
    "", "   ", "hello world", "Wait... what?", "one: two; three", "a - b", "em—dash and en–dash",
    "“quoted” and ‘single’", "trailing space   ", "ßtart", "ends with a comma,", "x ,y", "done -",
    "tab\tand\nnewline", "ellipsis… here",
])
def test_the_exported_rules_ARE_punc_norm(text):
    assert _normalize_from_the_exported_rules(text) == _punc_norm()(text)


# -- declarations the driver is generated from ------------------------------------------------------

def _config(tmp_path):
    return ChatterboxExportConfig(output_path=str(tmp_path / "o.gguf"),
                                  model_dir=str(_release(tmp_path)))


DRIVER = Path(__file__).resolve().parents[2] / "loom_exporter" / "chatterbox_driver"


def test_the_flow_runs_guided_on_the_callers_schedule_and_noise():
    """Unguided, S3Gen integrates a velocity field it was not trained to; on a uniform schedule it
    spends its ten steps in the wrong places; and without the driver's own noise nothing can be compared
    or re-run. The ODE call is hand-written in the stream (ADR-117), so its declaration is read there."""
    lua = (DRIVER / "00_stream.lua").read_text()
    assert "loom.run_ode_and_retain('estimator'" in lua
    assert "{carried = 'x', time = 't', method = 'euler', times = chatterbox_cosine_times(" in lua
    assert "n_elems = n_frames * N_MEL, state = noise," in lua
    assert "guidance = {scale = loom.state('flow_cfg'),\n" \
           "                     inputs = {mu = zeros, spks = chatterbox_zeros(N_MEL), cond = zeros}}" in lua


def test_one_pass_is_the_stream_run_in_one_hop():
    """`infer` opens the stream with `piece_tokens = 0` and feeds it to its end, so the one-pass gate
    grades the live path's code (ADR-100)."""
    lua = (DRIVER / "01_generate.lua").read_text()
    assert "if inputs.piece_tokens == nil then inputs.piece_tokens = 0 end" in lua
    assert "stream_open(inputs)" in lua and "stream_feed({})" in lua
    assert sorted(p.name for p in DRIVER.glob("*.lua")) == ["00_header.lua", "00_stream.lua", "01_generate.lua"]


def test_the_stream_cut_is_cosyvoice2_models_own():
    """S3Gen kept CosyVoice2's flow and its `finalize` hook but no loop around it; the stream's hop, its
    growth and ceiling are `CosyVoice2Model`'s, read off its source (without importing it)."""
    from loom_exporter.chatterbox_export import STREAM_HOP_SCALE, STREAM_HOP_TOKENS, STREAM_MAX_HOP_TOKENS

    model_py = Path("/home/flavio/Dev/CosyVoice/cosyvoice/cli/model.py")
    if not model_py.is_file():
        pytest.skip("no CosyVoice checkout to read CosyVoice2Model from")
    src = model_py.read_text()
    src = src[src.index("class CosyVoice2Model"):]
    src = src[:src.index("\nclass ")]
    assert f"self.token_hop_len = {STREAM_HOP_TOKENS}" in src
    assert "self.token_max_hop_len = 4 * self.token_hop_len" in src
    assert STREAM_MAX_HOP_TOKENS == 4 * STREAM_HOP_TOKENS
    assert f"self.stream_scale_factor = {STREAM_HOP_SCALE}" in src


def _random_vocoder():
    """A `HiftVocoderPhase` over S3Gen's HiFT layout with random weights, at f64."""
    if not CHATTERBOX_SRC.is_dir():
        pytest.skip("no Chatterbox checkout to build HiFT from")
    import torch
    from loom_exporter.chatterbox_export import HiftVocoderPhase, import_chatterbox

    import_chatterbox()
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
    from chatterbox.models.s3gen.hifigan import HiFTGenerator

    torch.manual_seed(0)
    hift = HiFTGenerator(sampling_rate=24000, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                         source_resblock_kernel_sizes=[7, 7, 11],
                         source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                         f0_predictor=ConvRNNF0Predictor())
    n_trim = 480
    fade = torch.zeros(2 * n_trim)
    fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
    return HiftVocoderPhase(hift, fade).eval().double()


def test_the_vocoder_holds_back_exactly_its_reach():
    """The stream re-runs HiFT over every frame so far and hands out all but the last
    `vocoder_right_frames` frames' samples. At f64 on a random HiFT, those samples over a mel cut at
    `m` frames are the samples over the whole mel bit for bit; one frame fewer held back and they are not.

    The F0 predictor's bias is raised so the source is VOICED: with random weights F0 sits under the 10 Hz
    threshold, the sine source ignores it, and the F0 path's reach (its convolutions, then the source's
    STFT) never shows -- the reach then measures ~13 frames, not 19.2."""
    import torch
    from loom_exporter.chatterbox_export import vocoder_right_frames

    voc = _random_vocoder()
    with torch.no_grad():
        voc.h.f0_predictor.classifier.bias.fill_(150.0)
    right = vocoder_right_frames(voc)
    assert right == 20      # 9,240 samples, 19.25 frames
    torch.manual_seed(1)
    n, m = 72, 44
    mel = torch.randn(1, n, 80, dtype=torch.float64) - 4.0
    phase = torch.rand(1, voc.harmonics, 1, dtype=torch.float64)
    noise = torch.randn(1, voc.harmonics, n * 480, dtype=torch.float64)
    with torch.no_grad():
        whole = voc(mel, phase, noise)[0]
        cut = voc(mel[:, :m], phase, noise[:, :, :m * 480])[0]
    assert float(whole[:m * 480].abs().max()) > 1e-3
    settled = (m - right) * 480
    assert bool((cut[:settled] == whole[:settled]).all())
    # 9,232 samples back is the first that moves (by ~1e-16): inside the 20th frame, past the 19th.
    assert not bool((cut[:settled + 480] == whole[:settled + 480]).all())


def test_the_contract_declares_a_text_door(tmp_path):
    config = _config(tmp_path)
    config.task = "text-to-speech"
    contract = config.contract()
    assert contract["input.kind"] == "text"
    assert contract["text.frontend"] == "vocab"
    assert contract["sample_rate"] == 24000
    assert contract["tts.default_steps"] == FLOW_STEPS


def test_the_defaults_are_the_references_own():
    """`ChatterboxTTS.generate`'s signature and `CFM_PARAMS` -- the numbers every published sample
    used, and the ones the driver falls back to."""
    assert (DEFAULT_TEMPERATURE, DEFAULT_MIN_P, DEFAULT_TOP_P) == (0.8, 0.05, 1.0)
    assert (DEFAULT_CFG_WEIGHT, DEFAULT_REPETITION_PENALTY, DEFAULT_EXAGGERATION) == (0.5, 1.2, 0.5)
    assert (FLOW_STEPS, FLOW_CFG_RATE) == (10, 0.7)


def test_the_tokenizer_is_named_rather_than_detected(tmp_path):
    """A `tokenizer.json` BPE is what byte-level "gpt2" looks like on disk, and this one is not."""
    kwargs = _config(tmp_path).backend_kwargs()
    assert kwargs["tokenizer_family"] == "chatterbox"


# -- the layout every phase agrees on --------------------------------------------------------------

@pytest.fixture(scope="module")
def phases():
    if not _is_chatterbox(MODEL_DIR) or not CHATTERBOX_SRC.is_dir():
        pytest.skip("no Chatterbox checkpoint and checkout to trace")
    return {p.name: p for p in ChatterboxExportConfig(
        output_path="/dev/null", model_dir=str(MODEL_DIR)).phases()}


def _shapes(phase):
    return {i.name: list(i.shape.symbolic_shape) for i in phase.mil_inputs}


def test_the_ode_state_and_its_conditioning_share_one_frame_major_layout(phases):
    """`x` is what the ODE carries, `mu` is the flow encoder's output and `cond` is the voice's mel;
    the sampler hands all three to one graph, so they must agree -- and frame-major, so that the
    driver's slice of the generated frames is a contiguous suffix."""
    est = _shapes(phases["estimator"])
    assert est["x"] == est["mu"] == est["cond"]
    assert est["x"][2] == 80 and not isinstance(est["x"][1], int)


def test_the_vocoder_takes_the_layout_the_estimator_produces(phases):
    """Retro-052's join: every graph can grade clean and the audio still be wrong if two of them
    disagree about which axis is which."""
    voc = _shapes(phases["vocoder"])
    assert voc["mel"][2] == 80 and not isinstance(voc["mel"][1], int)


def test_the_nsf_noise_is_one_symbol_with_the_mel(phases):
    """480 output samples per mel frame; declared as that multiple so the graph and the driver
    agree about the draw's length without a second free axis."""
    assert phases["vocoder"].declared_axes == {"nsf_noise": {2: f"{SAMPLES_PER_FRAME} * n_enc_frames"}}


def test_t3_runs_guidance_on_a_private_second_stream(phases):
    """The unconditional prefill differs, so its KV cache differs for the whole decode: one cache
    cannot serve both (loom.cpp ADR-023)."""
    lm = phases["t3_lm"]
    assert lm.extra_streams == ("t3_lm_uncond",)
    assert lm.fuse_attention is True and lm.kv_cache_size
