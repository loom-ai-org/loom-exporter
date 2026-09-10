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
