"""Qwen3-ASR's live path (loom.cpp ADR-121): upstream's streaming mode as a top-level fragment beside `infer`.

What runs is checked by loom.cpp's `test_e2e_qwen3_asr_stream` against upstream's loop; here, that the leaf
ships the fragment, binds what it reads top level, and hands it the SAME audio-row expression `infer` inlines.
"""
from loom_exporter.driver_components import ExportConstants, ExprFunction, LuaFragment
from loom_exporter.driver_ir import Var, evaluate
from loom_exporter.qwen3_asr_export import ASRQwen3SpeechLMExportConfig


def _components():
    spec = ASRQwen3SpeechLMExportConfig(model_dir="x")
    return spec, spec.driver_components()


def test_the_driver_carries_a_live_path():
    _, components = _components()
    live = [c for c in components if isinstance(c, LuaFragment) and "stream_feed" in c.defines]
    assert len(live) == 1 and live[0].top_level and live[0].path.is_file()
    text = live[0].path.read_text()
    constants = [c for c in components if isinstance(c, ExportConstants) and c.top_level]
    assert len(constants) == 1
    # Every S_ name the fragment reads is bound, and upstream's three cadence defaults are the ones bound.
    import re
    read = set(re.findall(r"\bS_[A-Z_]+\b", text))
    assert read <= set(constants[0].values), read - set(constants[0].values)
    values = constants[0].values
    assert (values["S_UPDATE_SECONDS"], values["S_UNFIXED_CHUNKS"], values["S_UNFIXED_TOKENS"]) == (2.0, 2, 5)
    # The prefix is cut from the answer's re-encoded TEXT, as upstream cuts it, not from the model's own ids
    # (oruk-loop's non-canonical " Voxtrol" changed the next word when the ids were cut; ADR-121).
    assert "loom.tokenize(loom.detokenize(raw))" in text
    # One encoder attention window (8 s, 104 rows) and the decoder's cache, which is what bounds a stream now
    # that only changed windows are re-encoded.
    assert (values["S_WINDOW_SAMPLES"], values["S_WINDOW_ROWS"]) == (128000, 104)
    assert values["S_N_CTX"] == ASRQwen3SpeechLMExportConfig(model_dir="x").max_seq_len
    # Window-by-window: the rows come from `encoder_window`, the floor from `window_mel_range`.
    assert "'encoder_window'" in text and "'window_mel_range'" in text
    # Without a checkpoint there is no marker; -1 matches no id, so the whole answer is the transcript.
    assert values["S_TRANSCRIPT_AFTER"] == -1


def test_the_stream_reads_infers_own_row_expression():
    spec, components = _components()
    rows = [c for c in components if isinstance(c, ExprFunction)]
    assert len(rows) == 1 and rows[0].name == "S_audio_rows" and rows[0].params == ("_audio_samples",)
    infer_expr = spec.audio_rows(Var("_audio_samples"))
    assert rows[0].expr.render() == infer_expr.render()
    # And it is HF's `_get_audio_token_length` at this checkpoint's numbers (hop 160, 100 frames, 13 rows).
    for samples in (1, 159, 160, 15999, 16000, 16001, 23456, 176000 - 8000, 480000):
        mel = samples // 160
        rest = mel % 100
        for _ in range(3):
            rest = (rest - 1) // 2 + 1
        assert evaluate(rows[0].expr, {"_audio_samples": samples}) == rest + (mel // 100) * 13, samples


def test_the_rows_helper_renders_as_a_top_level_function():
    _, components = _components()
    (rows,) = [c for c in components if isinstance(c, ExprFunction)]
    lines = rows.prelude(None)
    assert lines[0] == "local function S_audio_rows(_audio_samples)"
    assert lines[1].startswith("    return ") and lines[2] == "end"
    assert rows.emit(None) == []
