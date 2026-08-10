"""The byte-identity sweep: export a real checkpoint and prove the artifact did not move.

**Why this is the gate for this repo.** Almost every change here is meant to be output-preserving --
a refactor, a new component, a shared pass -- and almost none of it is covered by asserting that the
exporter still runs. What matters is whether the GGUF changed, and for which model, and where. The
answer has to be produced by exporting real checkpoints, which needs gigabytes on disk and minutes to
hours of CPU, so it lives here rather than in `tests/ci/`.

**What "did not move" means, concretely.** `snapshot_gguf` writes a GGUF's *structure* to diffable
text: every metadata KV, one pretty-printed JSON per graph topology, the embedded driver script
verbatim, and one `name / shape / dtype / sha256` line per tensor. Two snapshots that diff clean are
the same model in every respect a host can observe. That is a stronger statement than "the tests still
pass" and a much cheaper one to read than a binary diff.

**A gate that cannot fail proves nothing** -- the standing rule for this sweep, learned the expensive
way when a baseline was accidentally measured against itself and reported eleven models identical
including one that had to differ. So `test_a_changed_export_is_detected` deliberately corrupts a
snapshot and requires the comparison to notice.

Running it:

    export LOOM_MODELS=~/Dev/models              # where the checkpoints live
    export LOOM_EXPORT_BASELINE=~/loom-baseline  # snapshots from the tree you are comparing against
    pytest tests/gate -q

Every model whose checkpoint is absent skips; with no baseline recorded, the sweep still exports and
checks the artifact is loadable and self-describing, which is the weaker half of the same question.
Record a baseline by checking out the reference commit and running with `LOOM_EXPORT_RECORD=1`.

**Export from the tree you are measuring.** `loom-export` runs `python -m`, which puts the caller's
cwd on `sys.path` ahead of everything, so driving one checkout's script from another's directory
silently measures the wrong exporter. This module invokes the exporter in-process from *this* repo,
which sidesteps that entirely.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loom_exporter.paths import REPO_ROOT

# The sweep, as (name, checkpoint subpath, extra loom-export arguments). Kept here rather than in a
# config file because it is a statement about which architectures this repo claims to export, and the
# thing that should fail review when a family is added and not swept.
MODELS = [
    ("conformer-ctc", "conformer-ctc-small/stt_en_conformer_ctc_small.nemo",
     ["--task", "automatic-speech-recognition", "--model", "conformer-ctc"]),
    ("gigaam-rnnt", "gigaam-v3", ["--task", "automatic-speech-recognition", "--model", "gigaam-rnnt"]),
    ("kokoro", "kokoro_model", ["--task", "text-to-speech", "--model", "kokoro"]),
    ("matcha", "matcha_model/ckpt", ["--task", "text-to-speech", "--model", "matcha"]),
    ("lfm2-monolithic", "lfm2-350m", ["--task", "text-generation", "--model", "lfm2-monolithic"]),
    ("lfm2-modular", "lfm2-350m", ["--task", "text-generation", "--model", "lfm2-modular"]),
    ("qwen3", "qwen3-0.6b-base", []),
    ("whisper", "whisper-small", []),
    ("granite-speech", "granite-speech-4.0.1b", []),
    # Qwen3-ASR needs transformers >= 5.13 and the rest of the sweep needs <= 4.57, so it cannot run
    # in the same interpreter as its neighbours here. It is swept from the other environment; see
    # docs/EXPORT-PREPARATION.md.
    pytest.param("qwen3-asr", "qwen3-asr-0.6b-hf", [],
                 marks=pytest.mark.skip(reason="needs the transformers>=5.13 environment")),
]


def models_root() -> Path:
    root = os.environ.get("LOOM_MODELS")
    if not root:
        pytest.skip("LOOM_MODELS is not set; it names the directory holding the real checkpoints")
    return Path(root)


def snapshot(gguf: Path, into: Path) -> Path:
    """`snapshot_gguf` over one artifact, returning the directory it wrote."""
    from loom_exporter import snapshot_gguf

    snapshot_gguf.snapshot(gguf, into)
    return into / gguf.stem


def export(name: str, checkpoint: Path, extra: list, out_dir: Path) -> Path:
    """Run this repo's exporter in a subprocess, so one model's memory peak is one process's."""
    out = out_dir / f"{name}.gguf"
    result = subprocess.run(
        [sys.executable, "-m", "loom_exporter.main_export", str(checkpoint), "-o", str(out), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if result.returncode != 0:
        pytest.fail(f"exporting {name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
    return out


def diff_snapshots(baseline: Path, current: Path) -> str:
    """The text of `diff -r`, or "" when the two are identical."""
    result = subprocess.run(["diff", "-r", str(baseline), str(current)], capture_output=True, text=True)
    return result.stdout


@pytest.mark.gate
@pytest.mark.parametrize("name,subpath,extra", MODELS)
def test_export_matches_baseline(name, subpath, extra, tmp_path):
    checkpoint = models_root() / subpath
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint} is not present")

    gguf = export(name, checkpoint, extra, tmp_path)
    assert gguf.exists() and gguf.stat().st_size > 0

    current = snapshot(gguf, tmp_path / "snap")
    # Self-describing, whatever the baseline says: an artifact with no topologies and no driver is
    # broken in a way a matching baseline would not catch, because the baseline would be broken too.
    assert (current / "kv.txt").exists()
    assert list(current.glob("model_graph_topology_*.json")), f"{name} declared no topologies"

    if os.environ.get("LOOM_EXPORT_RECORD"):
        destination = Path(os.environ["LOOM_EXPORT_BASELINE"]) / name
        shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(current, destination)
        pytest.skip(f"recorded baseline for {name}")

    baseline_root = os.environ.get("LOOM_EXPORT_BASELINE")
    if not baseline_root or not (Path(baseline_root) / name).exists():
        pytest.skip(f"no recorded baseline for {name}; exported and validated only")

    difference = diff_snapshots(Path(baseline_root) / name, current)
    assert not difference, f"{name}'s export moved:\n{difference[:8000]}"


@pytest.mark.gate
def test_a_changed_export_is_detected(tmp_path):
    """The sweep's own gate. A byte-identity check that cannot fail is worth nothing, and this one has
    reported a false pass before -- eleven models "identical", including one that had to differ,
    because both sides were measured from the same tree. Corrupting a snapshot and requiring the
    comparison to notice is cheap insurance against that shape of mistake."""
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    for directory in (baseline, current):
        directory.mkdir()
        (directory / "kv.txt").write_text("loom.architecture = qwen3\n")
        (directory / "tensors.txt").write_text("blk.0.attn_q.weight [1024,1024] F32 abc123\n")

    assert diff_snapshots(baseline, current) == ""
    (current / "tensors.txt").write_text("blk.0.attn_q.weight [1024,1024] F32 deadbeef\n")
    assert "deadbeef" in diff_snapshots(baseline, current)
