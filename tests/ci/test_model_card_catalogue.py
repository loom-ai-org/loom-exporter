"""The card catalogue's own consistency, checked without exporting anything.

**Why this file exists.** `ModelCard.task_type` drives three unrelated things: which usage snippet a
card gets, which of `render_readme`'s per-family checks run, and (until `pipeline_tag` was split out)
what was published as the HuggingFace tag. Editing it to satisfy one of those silently re-points the
others, and the failure lands minutes into an export -- after the GGUF has been written -- because
that is when the card is rendered.

That is not hypothetical: a commit retagging `dac-44khz` to `text-to-audio` and `dia-1.6b` to
`text-to-speech` for the Hub's sake left the first pointing at a snippet key that does not exist and
the second at the text-to-speech snippet, for a model whose only door is `text2codes`. A `--all` run
wrote dac's 208 MB GGUF, raised `KeyError` on its card, and took the five entries after it down with
it -- including both models a release was waiting on.

Every check here is answerable from the catalogue alone, in milliseconds, which is the point: this is
the class of defect that must not be discovered by running the thing.
"""
import importlib.util
from pathlib import Path

import pytest

from loom_exporter.paths import REPO_ROOT
from loom_exporter.tasks import known_tasks


def _catalogue():
    path = REPO_ROOT / "tools" / "build_model_cards.py"
    spec = importlib.util.spec_from_file_location("build_model_cards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cards():
    return _catalogue()


def test_every_entry_resolves_to_a_snippet_that_exists(cards):
    """THE CHECK THIS FILE EXISTS FOR. `render_readme` indexes `USAGE_SNIPPETS[snippet_key(card)]`
    directly, so a key that is not there is a KeyError *after* the export has already run."""
    missing = {c.slug: cards.snippet_key(c) for c in cards.CATALOG
               if cards.snippet_key(c) not in cards.USAGE_SNIPPETS}
    assert not missing, f"catalogue entries whose snippet key does not exist: {missing}"


def test_every_task_type_is_a_canonical_task(cards):
    """`task_type` is documented as a name from `loom_exporter.tasks`, and that is what makes a card's
    claim about a model comparable with the export's own `loom.task`. The Hub's tag is a separate
    field precisely so this one can stay canonical."""
    unknown = {c.slug: c.task_type for c in cards.CATALOG if c.task_type not in known_tasks()}
    assert not unknown, f"entries whose task_type is not a canonical task: {unknown}"


def test_every_tts_entry_states_a_sample_rate(cards):
    """`render_readme` raises for a text-to-speech card with no rate, because its snippet interpolates
    one. Same defect class as the snippet key: raised at render time, minutes into an export."""
    missing = [c.slug for c in cards.CATALOG
               if c.task_type == "text-to-speech" and not c.sample_rate]
    assert not missing, f"text-to-speech entries with no sample_rate: {missing}"


def test_slugs_are_unique(cards):
    """Two entries with one slug write to the same directory, and the second silently wins."""
    slugs = [c.slug for c in cards.CATALOG]
    assert len(slugs) == len(set(slugs)), f"duplicate slugs: {sorted({s for s in slugs if slugs.count(s) > 1})}"


# HuggingFace's recognized `pipeline_tag` values, transcribed from
# `https://huggingface.co/api/models-tags-by-type` (52 tags, read 2026-09-11). Re-derive with:
#
#     python -c "import json,urllib.request; d=json.load(urllib.request.urlopen(
#         'https://huggingface.co/api/models-tags-by-type')); print(sorted(
#         t['id'] for t in d['pipeline_tag']))"
#
# Transcribed rather than fetched because tests/ci is hermetic: a check that reaches the network
# fails for reasons that are not about this repo, and this list moves about once a year.
HF_PIPELINE_TAGS = frozenset("""
any-to-any audio-classification audio-text-to-text audio-to-audio automatic-speech-recognition
depth-estimation document-question-answering feature-extraction fill-mask graph-ml
image-classification image-feature-extraction image-segmentation image-text-to-image
image-text-to-text image-text-to-video image-to-3d image-to-image image-to-text image-to-video
keypoint-detection mask-generation object-detection question-answering reinforcement-learning
robotics sentence-similarity summarization table-question-answering tabular-classification
tabular-regression text-classification text-generation text-ranking text-to-3d text-to-audio
text-to-image text-to-speech text-to-video time-series-forecasting token-classification
translation unconditional-image-generation video-classification video-text-to-text video-to-video
visual-document-retrieval visual-question-answering voice-activity-detection
zero-shot-classification zero-shot-image-classification zero-shot-object-detection
""".split())


def test_every_card_publishes_a_tag_huggingface_recognizes(cards):
    """The Hub only auto-identifies tags from its own closed list, and a card carrying anything else
    renders with a visible inconsistency.

    Three of this project's canonical task names are not on that list -- `audio-codec`,
    `text-to-codes` and `text2text-generation`, the last of which HF retired -- which is exactly why
    `pipeline_tag` exists as a field separate from `task_type`. Editing `task_type` to satisfy the Hub
    instead is what broke the export sweep, so this check is the one that keeps the two concerns apart
    without anyone having to remember which is which.
    """
    unrecognized = {c.slug: (c.pipeline_tag or c.task_type) for c in cards.CATALOG
                    if (c.pipeline_tag or c.task_type) not in HF_PIPELINE_TAGS}
    assert not unrecognized, (
        f"cards whose pipeline_tag HuggingFace does not recognize: {unrecognized}. Set an explicit "
        f"`pipeline_tag=` from HF's list; do NOT change `task_type`, which drives the usage snippet "
        f"and render_readme's per-family checks."
    )
