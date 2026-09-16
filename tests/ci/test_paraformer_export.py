"""Family 5's second leaf (P5): Paraformer -- SANM encoder, CIF predictor, NAR decoder.

**The CIF arithmetic is what is tested here, and it is tested against numbers rather than against a
shape.** Everything else in this export is either inherited (the front end and the position encoding
are the first leaf's, covered by `test_sanm_asr_export.py`) or ordinary (two traced phases). What is
this family's own is a boundary decision that happens in the DRIVER, in float32 emulated on doubles,
and whose failure mode is not an exception: get the arithmetic order wrong and the acoustic embeddings
move by ~2.6e-01, the logits by ~3.8e-01, and the transcript stays *almost* right.

So `cif_fire`'s own recipe is pinned twice over -- once as the fire positions and once as the
resampling weights -- against values taken from FunASR's `cif_wo_hidden_v1` on a real utterance.
"""
import json
import math
from pathlib import Path

import pytest

from loom_exporter.paraformer_export import (
    MAX_TOKENS, MIN_TOKENS, ParaformerExportConfig, _build_funasr_paraformer, _is_funasr_paraformer,
)
from loom_exporter.registry import default_registry

PARAFORMER_YAML = "model: Paraformer\nencoder: SANMEncoder\npredictor: CifPredictorV2\n"
SENSEVOICE_YAML = "model: SenseVoiceSmall\nencoder: SenseVoiceEncoderSmall\n"


def _funasr_dir(tmp_path: Path, name: str, config: str, *, with_weights: bool = True) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.yaml").write_text(config)
    if with_weights:
        (d / "model.pt").write_bytes(b"")
    return d


# -- detection: the two leaves must not claim each other -------------------------------------------

def test_a_paraformer_checkpoint_is_claimed(tmp_path):
    assert _is_funasr_paraformer(_funasr_dir(tmp_path, "pf", PARAFORMER_YAML))


def test_a_sensevoice_checkpoint_is_not_claimed(tmp_path):
    """The two share a directory layout AND an `encoder: SANMEncoder`-shaped encoder, so both
    recognizers name the model class. A structural check on the encoder would claim whichever
    checkpoint it met first for whichever template asked."""
    assert not _is_funasr_paraformer(_funasr_dir(tmp_path, "sv", SENSEVOICE_YAML))


@pytest.mark.parametrize("model_name", ["ContextualParaformer", "ParaformerStreaming", "BiCifParaformer"])
def test_the_paraformer_VARIANTS_are_not_claimed(tmp_path, model_name):
    """Deliberate: each adds machinery this template does not export (a hotword encoder, a chunked
    predictor, a second predictor pass). Claiming them on a prefix match would produce a GGUF missing
    part of the model, which is the failure a named recognizer exists to prevent."""
    assert not _is_funasr_paraformer(
        _funasr_dir(tmp_path, model_name, f"model: {model_name}\nencoder: SANMEncoder\n"))


def test_the_registry_routes_each_leaf_to_its_own_recognizer(tmp_path):
    registry = default_registry()
    assert registry.detect(_funasr_dir(tmp_path, "pf", PARAFORMER_YAML)).name == "funasr-paraformer"
    assert registry.detect(_funasr_dir(tmp_path, "sv", SENSEVOICE_YAML)).name == "funasr-sensevoice"


# -- what the config declares ----------------------------------------------------------------------

def test_the_token_bounds_are_a_real_range():
    assert 1 <= MIN_TOKENS < MAX_TOKENS


# -- the CIF arithmetic, against FunASR's own numbers -----------------------------------------------

def round_half_to_even(x):
    floor = math.floor(x)
    diff = x - floor
    if diff < 0.5:
        return floor
    if diff > 0.5:
        return floor + 1
    return floor if floor % 2 == 0 else floor + 1


def to_f32(x):
    """`to_f32.lua`, transliterated. The Lua is what ships; this is what pins it."""
    import numpy as np
    if x == 0 or x != x or x in (math.inf, -math.inf):
        return x
    significand, exponent = math.frexp(x)
    return math.ldexp(round_half_to_even(significand * 16777216.0) / 16777216.0, exponent)


def cif_fire(alphas, threshold, n_frames):
    """`cif_fire.lua`, transliterated, including both float32 traps."""
    idx, remain = [], []
    prefix, prev_floor = 0.0, 0.0
    for i, alpha in enumerate(alphas):
        prefix += alpha
        prefix32 = to_f32(prefix)
        cur_floor = math.floor(prefix32)
        if cur_floor > prev_floor:
            fired = to_f32(to_f32(threshold + prefix32) - cur_floor)
            idx.append(i)
            remain.append(fired - math.floor(fired))
        prev_floor = cur_floor

    weights = [0.0] * (len(idx) * n_frames)
    prev_f = -1
    for i, f in enumerate(idx):
        row = i * n_frames
        for t in range(prev_f + 1, min(f, n_frames - 1) + 1):
            weights[row + t] += alphas[t]
        if i > 0 and prev_f < n_frames:
            weights[row + prev_f] += remain[i - 1]
        if f < n_frames:
            weights[row + f] -= remain[i]
        prev_f = f
    return idx, remain, weights


def test_to_f32_matches_a_real_float32_cast():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    values = list(rng.random(4000) * 100.0) + list(np.cumsum(rng.random(400) * 0.4))
    # The knife-edge case from the real model: a prefix sum 1.9e-06 below an integer.
    values += [31.999998092651367, 32.99999809265137, 1.9999980926513672, 0.9999980926513672]
    for v in values:
        assert to_f32(float(v)) == float(np.float32(v)), v


def test_the_remainder_depends_on_the_MAGNITUDE_of_the_running_total():
    """The trap, pinned as its own case because it is invisible in the algebra.

    At a prefix sum of 31.999998 the float32 spacing is 3.8e-06, so `1 + prefix` lands on a
    representable midpoint and rounds to 33.0 -- making the remainder 0. Computing `1 + (prefix -
    floor(prefix))` instead is the same number in exact arithmetic and gives ~1. FunASR does the
    former, so loom must too.
    """
    prefix32 = to_f32(31.999998092651367)
    fired = to_f32(to_f32(1.0 + prefix32) - math.floor(prefix32))
    assert fired == 2.0 and fired - math.floor(fired) == 0.0
    naive = to_f32(1.0 + to_f32(prefix32 - math.floor(prefix32)))
    assert naive - math.floor(naive) == pytest.approx(0.9999980926513672)


def test_cif_fire_reproduces_funasr_on_a_real_utterance():
    """Fire positions and resampling weights, against values taken from `cif_wo_hidden_v1` on 13 s of
    real speech. The fixture is alphas in, indices/weights out -- no model needed."""
    np = pytest.importorskip("numpy")
    case = json.loads((Path(__file__).parent / "data" / "paraformer_cif_case.json").read_text())
    idx, remain, weights = cif_fire(case["alphas"], 1.0, case["n_frames"])
    assert idx == case["idx"]
    assert len(idx) == case["n_tokens"]
    assert np.allclose(remain, case["remain"], atol=0, rtol=0), "the remainders are bit-exact or wrong"
    assert np.abs(np.array(weights) - np.array(case["weights"])).max() == 0.0


def test_a_pure_float64_host_would_put_a_token_on_the_WRONG_FRAME():
    """The sabotage arm for the precision recipe, and the reason `to_f32` exists at all.

    Accumulating in double and never rounding to float32 is strictly more accurate and gives a
    different answer -- which is the whole finding. If this ever stops failing, the fixture's
    boundaries have stopped being knife-edge and this test has stopped testing anything.
    """
    case = json.loads((Path(__file__).parent / "data" / "paraformer_cif_case.json").read_text())
    idx, prefix, prev_floor = [], 0.0, 0.0
    for i, alpha in enumerate(case["alphas"]):
        prefix += alpha
        if math.floor(prefix) > prev_floor:
            idx.append(i)
        prev_floor = math.floor(prefix)
    assert idx != case["idx"], "the f64 and f32 boundaries agree here; pick a harder fixture"


def test_the_architecture_is_a_field_not_only_a_method():
    """`MultiPhase.export` reads the FIELD -- a config that answered only through
    `export_architecture()` would export as the fallback `mil_model` with nothing raising, which is
    what this file did on its first successful export."""
    assert ParaformerExportConfig(output_path="x.gguf", model_dir="d").architecture == "paraformer"


# -- the vocabulary: a flat table whose pieces compose differently -----------------------------------

def test_the_vocabulary_is_written_under_its_own_tag():
    """Not `ctc`, and not a rewrite into SentencePiece. ADR-033's rule is that a tag answers "which
    scheme is this", and `@@` marks "I continue into the next piece" where `U+2581` and `##` mark "a
    word starts here" -- duals, with the same piece string appearing in both roles, so no per-piece
    rewrite turns one into the other."""
    kwargs = ParaformerExportConfig(output_path="x.gguf", model_dir="d").backend_kwargs()
    assert kwargs["tokenizer_family"] == "funasr"
    assert kwargs["tokenizer_dir"] == "d"


@pytest.mark.parametrize("piece,expected", [
    ("and", 2), ("can't", 2), ("low", 2),        # Latin: alphabetic-or-apostrophe, per character
    ("你", 1), ("9", 1), ("@", 1), ("9@@", 1),    # CJK: the block, ASCII DIGITS, and '@' all count
    ("f@@", 0), ("<s>", 0), ("<blank>", 0),      # neither -- and `f@@` is still a continuation
])
def test_the_piece_script_is_per_character(piece, expected):
    """The two predicates are per-CHARACTER, which decides real cases: `can't` is a word because the
    apostrophe is allowed per character, and `9@@` is CJK because digits and '@' both are -- so it is
    never treated as a subword continuation however much it looks like one."""
    from loom_exporter.funasr_tokenizer_export import piece_script
    assert piece_script(piece) == expected


def test_the_continuation_marker_is_orthogonal_to_the_script():
    """`f@@` classifies as OTHER (an '@' is neither CJK nor alphabetic) and is still a continuation,
    while `9@@` is CJK and is NOT one. A reader that tested `@@` only inside its Latin branch would
    emit `f@@` literally -- which is what the first version of `FunasrVocab::decode` did."""
    from loom_exporter.funasr_tokenizer_export import SCRIPT_CJK, SCRIPT_OTHER, piece_script
    assert piece_script("f@@") == SCRIPT_OTHER
    assert piece_script("9@@") == SCRIPT_CJK


def test_a_tokens_json_that_is_not_a_piece_array_is_refused(tmp_path):
    from loom_exporter.funasr_tokenizer_export import read_funasr_tokens
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tokens.json").write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="JSON array"):
        read_funasr_tokens(str(d))
    (d / "tokens.json").write_text(json.dumps(["ok", 3]))
    with pytest.raises(ValueError, match="non-string"):
        read_funasr_tokens(str(d))


def test_the_blank_is_not_treated_as_a_control_piece():
    """It is row 0 and it looks like one, but `sentence_postprocess` drops exactly `<s>/</s>/<unk>/<OOV>`
    and prints anything else literally. Marking `<blank>` control made the engine disagree with the
    reference on one sequence in 5,011 -- unobservable in practice, because a non-autoregressive decoder
    cannot emit it, which is precisely why it should not have been left in."""
    from loom_exporter.funasr_tokenizer_export import CONTROL_PIECES
    assert "<blank>" not in CONTROL_PIECES
    assert set(CONTROL_PIECES) == {"<s>", "</s>", "<unk>", "<OOV>"}
