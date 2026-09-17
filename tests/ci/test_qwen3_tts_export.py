"""Family 10's second leaf: Qwen3-TTS, whose prompt has two shapes and whose ICL half is arithmetic.

The talker shipped without a file here, which is why this one starts at the recognizer. What it exists
for, though, is the **ICL prompt's row layout**, because that is the part of this export that can be
wrong without anything failing:

* The reference builds the ICL block by branching on `text_lens > codec_lens` and slicing one stream
  by the other's length. A graph can carry neither the branch nor the slice, so the DRIVER does that
  arithmetic and the graph does elementwise work on two already-sized arrays. If the layout the
  wrapper assumes and the layout the driver builds ever drift apart, the export still succeeds, the
  model still speaks, and it speaks in the wrong voice -- there is no shape to catch it.
* `bos_mask` is the other half of the same idea. The codec stream is `codec_bos` followed by one
  summed embedding per reference frame, which is one row longer than `ref_code` -- so rather than
  concatenate (a second symbol for one row), the driver writes a zero row and a mask selects
  `codec_bos` over it. The test that matters is that the mask actually selects: a wrapper that ignored
  it would differ only in row 0, out of a hundred and forty.

The stub talker is deliberate. What is under test is this file's own algebra -- which table each
group's code is looked up in, which rows the two streams contribute, what the schedule carries -- and
a real checkpoint would hide all of it behind 914 M parameters and a fifteen-minute load.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from loom_exporter.qwen3_tts_export import (  # noqa: E402
    PREFILL_LEN,
    REF_HEAD,
    REF_TAIL,
    TEXT_HEAD,
    TEXT_TAIL,
    _IclPrefillEmbedWrapper,
    _is_qwen3_tts,
)

HIDDEN = 8
N_GROUPS = 4
CODEBOOK = 5
TEXT_VOCAB = 100
CODEC_VOCAB = 30


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_qwen3_tts_directory_is_claimed(tmp_path):
    assert _is_qwen3_tts(_hf_dir(tmp_path, "talker", {"model_type": "qwen3_tts"}))


def test_the_codec_subfolder_is_not_claimed_by_the_talkers_recognizer(tmp_path):
    """`speech_tokenizer/` is the SAME checkpoint's other half and exports through
    `audio_codec_export`. Claiming it here would trace a talker's phases over a codec's weights."""
    assert not _is_qwen3_tts(
        _hf_dir(tmp_path, "codec", {"model_type": "qwen3_tts_tokenizer_12hz"}))


def test_a_directory_without_a_config_is_not_claimed(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert not _is_qwen3_tts(d)


# -- the ICL prompt's algebra ----------------------------------------------------------------------

class _StubTalker(torch.nn.Module):
    """Enough of the talker for the prompt: two embedding tables and the text projection.

    The two tables are given disjoint, recognisable values -- text rows count up from 1, codec rows
    count down from -1 -- so an assertion can say WHICH table a row came from rather than only that
    the arithmetic closed.
    """

    def __init__(self):
        super().__init__()
        self.text = torch.nn.Embedding(TEXT_VOCAB, HIDDEN)
        self.codec = torch.nn.Embedding(CODEC_VOCAB, HIDDEN)
        self.text_projection = torch.nn.Identity()
        with torch.no_grad():
            self.text.weight.copy_(
                torch.arange(1, TEXT_VOCAB + 1, dtype=torch.float32).view(-1, 1).repeat(1, HIDDEN))
            self.codec.weight.copy_(
                -torch.arange(1, CODEC_VOCAB + 1, dtype=torch.float32).view(-1, 1).repeat(1, HIDDEN))

    def get_text_embeddings(self):
        return self.text

    def get_input_embeddings(self):
        return self.codec


class _StubConfig:
    tts_bos_token_id, tts_eos_token_id, tts_pad_token_id = 1, 2, 3

    class talker_config:
        codec_think_id, codec_think_bos_id, codec_think_eos_id = 4, 5, 6
        codec_pad_id, codec_bos_id = 7, 8


def _wrapper():
    torch.manual_seed(0)
    embeddings = torch.arange(
        1, (N_GROUPS - 1) * CODEBOOK * HIDDEN + 1, dtype=torch.float32
    ).view((N_GROUPS - 1) * CODEBOOK, HIDDEN) * 0.5
    return _IclPrefillEmbedWrapper(_StubTalker(), _StubConfig, embeddings).eval(), embeddings


def _call(wrapper, n_frames, n_schedule=2, language_id=9):
    role = torch.tensor([[10, 11, 12]])
    replay = n_frames + 1
    icl_text = torch.arange(13, 13 + replay).view(1, -1)
    codes = torch.zeros((1, replay, N_GROUPS), dtype=torch.long)
    codes[0, 1:, 0] = torch.arange(1, n_frames + 1) % CODEC_VOCAB
    for g in range(1, N_GROUPS):
        codes[0, 1:, g] = (g - 1) * CODEBOOK + (torch.arange(n_frames) % CODEBOOK)
    mask = torch.zeros(1, replay, 1)
    mask[0, 0, 0] = 1.0
    schedule_ids = torch.arange(30, 30 + n_schedule).view(1, -1)
    with torch.no_grad():
        return wrapper(role, icl_text, codes, mask, schedule_ids,
                       torch.tensor([language_id]), torch.zeros(1, HIDDEN))


def test_the_icl_prompt_is_nine_rows_plus_one_per_reference_frame_plus_codec_bos():
    """The x-vector prompt's tenth row is the first text token plus `codec_bos`; ICL does not build
    it, because `codec_bos` opens the replay instead. Nine, not ten, and the driver's `_prefill_len`
    says the same thing -- a disagreement here is an off-by-one in every position id after it."""
    wrapper, _ = _wrapper()
    for n_frames in (1, 7, 40):
        prefill, _ = _call(wrapper, n_frames)
        assert prefill.shape == (1, (PREFILL_LEN - 1) + n_frames + 1, HIDDEN)


def test_the_first_replay_row_is_codec_bos_and_not_the_codes_underneath_it():
    """`bos_mask`'s whole job. The driver writes a zero row into `ref_code` where `codec_bos` goes;
    without the mask the prompt would open on whatever row 0 of each table happens to hold, which is a
    valid embedding, a correct shape and the wrong voice."""
    wrapper, _ = _wrapper()
    prefill, _ = _call(wrapper, 5)
    head = PREFILL_LEN - 1
    codec_bos = _StubConfig.talker_config.codec_bos_id
    expected = wrapper.codec_embedding(torch.tensor([codec_bos]))[0]
    text_row = wrapper._text(torch.tensor([[13]]))[0, 0]
    assert torch.allclose(prefill[0, head], expected + text_row, atol=1e-6)


def test_group_zero_reads_the_talkers_table_and_the_rest_read_the_merged_one():
    """The convention `_FrameEmbedWrapper` already uses, in reverse: the driver offsets groups 1..15
    into one concatenated table on the way in, so a row is a sum of one talker lookup and fifteen
    merged ones. Reading group 0 from the merged table would be a silent 2048-row shift."""
    wrapper, embeddings = _wrapper()
    n_frames = 3
    prefill, _ = _call(wrapper, n_frames)
    head = PREFILL_LEN - 1
    for f in range(n_frames):
        first = (f + 1) % CODEC_VOCAB
        rows = [(g - 1) * CODEBOOK + (f % CODEBOOK) for g in range(1, N_GROUPS)]
        expected = (wrapper.codec_embedding(torch.tensor([first]))[0]
                    + sum(embeddings[r] for r in rows)
                    + wrapper._text(torch.tensor([[13 + 1 + f]]))[0, 0])
        assert torch.allclose(prefill[0, head + 1 + f], expected, atol=1e-5)


def test_the_schedule_is_the_trailing_text_with_one_pad_appended():
    """The pad is what turns the reference's "trailing[step] while there is one, else pad" into an
    index the driver clamps. Without it the last frames would read a row that is not there."""
    wrapper, _ = _wrapper()
    _, schedule = _call(wrapper, 4, n_schedule=3)
    assert schedule.shape == (1, 4, HIDDEN)
    pad = wrapper._text(torch.tensor([[_StubConfig.tts_pad_token_id]]))[0, 0]
    assert torch.allclose(schedule[0, -1], pad, atol=1e-6)
    assert torch.allclose(schedule[0, 0], wrapper._text(torch.tensor([[30]]))[0, 0], atol=1e-6)


def test_the_language_id_reaches_the_control_run_rather_than_being_baked():
    """One GGUF serves all ten languages, which is only true if this input moves the prompt."""
    wrapper, _ = _wrapper()
    a, _ = _call(wrapper, 3, language_id=9)
    b, _ = _call(wrapper, 3, language_id=14)
    assert not torch.allclose(a, b)
    # and it moves exactly one row: the language sits inside the codec control run
    differing = (a - b).abs().amax(dim=-1)[0] > 1e-6
    assert int(differing.sum()) == 1


def test_the_template_counts_match_the_reference_implementations_slices():
    """`input_id[:, 3:-5]` and `ref_ids[:, 3:-2]` in `Qwen3TTSForConditionalGeneration.generate`. The
    driver slices with these four numbers, so they are the template's shape stated once."""
    assert (TEXT_HEAD, TEXT_TAIL) == (3, 5)
    assert (REF_HEAD, REF_TAIL) == (3, 2)
