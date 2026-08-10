"""Where this repo keeps things, in one place.

Every family whose driver is assembled from hand-written Lua fragments needs to find them, and each
used to spell that as `Path(__file__).resolve().parent.parent / "convert_kokoro" / "kokoro_driver"` --
"my package's sibling". That worked while the package lived at `loom_exporter/` and the
converters at `tools/convert_*/`, and broke the moment the package moved to the repo root, in twelve
places at once, with the tests computing the same relationship a second time from *their* location.

So the relationship is written down once here instead of being re-derived from whoever is asking.
A future move edits this file.
"""
import os
from pathlib import Path
from typing import Optional

# The repo root: this file is `<root>/loom_exporter/paths.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]

# The pre-MIL per-model converters. They are still here because several of them own something the MIL
# path uses -- a checkpoint loader, a phonemizer, and the hand-written `*_driver/` Lua fragments the
# TTS families' drivers are assembled from.
CONVERTERS = REPO_ROOT / "tools"

# Reference-forward generators for real checkpoints (the gate suite's oracles).
FIXTURE_GEN = REPO_ROOT / "fixture_gen"

# Audio the fixture generators and the engine's tests both read.
SAMPLES = REPO_ROOT / "samples"


def engine_root() -> Optional[Path]:
    """The loom.cpp checkout, or None.

    A handful of checks here are genuinely about the *engine*: the pre-tokenizer names this exporter
    can emit have to be names `src/core/bpe_vocab.cpp` implements, and a mismatch is a model that
    exports cleanly and tokenizes wrongly. That check was free when both halves were one repo and is
    now a question about somebody else's working copy, so it is answered the same way the gate tests
    answer theirs: look, and skip cleanly rather than fail when the answer is no.

    `LOOM_CPP_ROOT` first, then the sibling checkout that `Dev/loom/{loom.cpp,loom-exporter}` puts
    there anyway.
    """
    explicit = os.environ.get("LOOM_CPP_ROOT")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(REPO_ROOT.parent / "loom.cpp")
    for candidate in candidates:
        if (candidate / "src" / "core").is_dir():
            return candidate
    return None


def driver_dir(converter: str, driver: str) -> Path:
    """`tools/convert_kokoro/kokoro_driver` -- a family's hand-written Lua fragments.

    Named rather than concatenated at each call site so that "which converter owns which driver" is a
    question with one answer, and so `LuaFragment`'s own error names a path that exists.
    """
    return CONVERTERS / converter / driver
