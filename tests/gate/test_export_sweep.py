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

# The release the `qwen3_asr` module first ships in -- see [[env-python-venvs-export]] and
# docs/EXPORT-PREPARATION.md for why that forces a second environment.
QWEN3_ASR_MIN_TRANSFORMERS = (5, 13)


def _transformers_version() -> tuple:
    """`(major, minor)` of the installed transformers, or `()` when it is not installed.

    Read from package metadata rather than by importing transformers: this runs at COLLECTION time,
    for every invocation of this file, and importing transformers to decide whether to skip one
    parameter would cost every other model's run several seconds."""
    try:
        from importlib.metadata import version

        return tuple(int(part) for part in version("transformers").split(".")[:2])
    except Exception:
        return ()


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
    ("smollm2", "smollm2-360m-it", []),
    ("gemma-3-270m-it", "gemma-3-270m-it", []),
    ("whisper", "whisper-small", []),
    ("granite-speech", "granite-speech-4.0.1b", []),
    ("parakeet-tdt", "parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo",
     ["--task", "automatic-speech-recognition", "--model", "parakeet-tdt"]),
    ("parakeet-rnnt", "parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo",
     ["--task", "automatic-speech-recognition", "--model", "parakeet-rnnt"]),
    ("styletts2", "styletts2_model/ckpt/Models/LJSpeech/epoch_2nd_00100.pth",
     ["--task", "text-to-speech", "--model", "styletts2"]),
    # These two checkpoints are whole repos rather than directories under one models root, so they
    # name their own variable (see `resolve_checkpoint`) instead of pinning one machine's layout here.
    ("supertonic", "$LOOM_SUPERTONIC_ROOT/assets/pt",
     ["--task", "text-to-speech", "--model", "supertonic"]),
    ("vits", "$LOOM_VITS_CHECKPOINT", ["--task", "text-to-speech", "--model", "vits"]),
    # Family 12 (P5): the first non-audio task in the sweep, and the smallest export in it. Reached
    # through the generic `hf-token-classifier` fallback, so it needs no `--model` -- which is itself
    # part of what is being swept, since a second family claiming a `*ForTokenClassification` directory
    # would show up here as an ambiguity rather than as a wrong export.
    ("bert-ner", "bert-base-ner", []),
    # The same family through a structurally different encoder: no token-type embeddings, no
    # `position_ids` argument, `.transformer` where BERT has `.encoder`. It is in the sweep as the
    # second checkpoint the template is held to, not as a second name -- an artifact difference
    # between these two rows is how a change that quietly re-specialises the family shows up.
    ("distilbert-ner", "distilbert-ner", []),
    # The third, and the first whose TOKENIZER is the new thing rather than the encoder: XLM-R, a
    # SentencePiece Unigram vocabulary whose ids are NOT the protobuf's piece order (P5). The two rows
    # above are both WordPiece with a CoNLL-03 head, so this is the row that fails if
    # `spm_tokenizer_export`'s id authority stops being read -- and the only row in the sweep whose
    # position table numbers from 2, which the artifact records as an `add` the other two fold away.
    ("fullstop-punc", "fullstop-punc", []),
    # Family 11 (P5): the first codec decoder, and the first export whose ROOT AXIS is a codec-frame
    # count rather than tokens or samples. Swept because its failure mode is invisible in a snapshot
    # that only checks the export ran -- see tests/ci/test_audio_codec_export.py.
    ("dac-44khz", "dac-44khz", []),
    # Family 10 (P5): the other half of that pair, and the sweep's largest export by an order of
    # magnitude -- 6.4 GB and ~12 minutes. It is here rather than left out for its cost because it is
    # the only row whose artifact contains an ALIASED topology: `cross_kv_uncond` and `decoder_uncond`
    # are the same graphs under second names, and the one with a KV cache declares
    # `kv_cache_scope: "private"` (loom.cpp ADR-023). A change that stopped emitting either, or
    # emitted them without that key, is a silent wrong answer under classifier-free guidance and is
    # invisible everywhere else in this file.
    ("dia-1.6b", "dia-1.6b", []),
    # Qwen3-ASR needs transformers >= 5.13 and the rest of the sweep needs <= 4.57, so it cannot run
    # in the same interpreter as its neighbours here. It is swept from the other environment; see
    # docs/EXPORT-PREPARATION.md.
    #
    # Conditional on the INTERPRETER rather than marked skip outright, which is what it was: an
    # unconditional skip meant this entry ran in no environment at all, so the one model that most
    # needs saying "sweep me from over there" was the one nothing ever swept. Now `-k qwen3-asr` from
    # the transformers>=5.13 venv runs it and the piper venv still skips it, with a reason that names
    # the version it actually found.
    pytest.param("qwen3-asr", "qwen3-asr-0.6b-hf", [],
                 marks=pytest.mark.skipif(
                     _transformers_version() < QWEN3_ASR_MIN_TRANSFORMERS,
                     reason=f"needs transformers >= "
                            f"{'.'.join(map(str, QWEN3_ASR_MIN_TRANSFORMERS))}; this interpreter has "
                            f"{'.'.join(map(str, _transformers_version())) or 'none'}")),
]


def models_root() -> Path:
    root = os.environ.get("LOOM_MODELS")
    if not root:
        pytest.skip("LOOM_MODELS is not set; it names the directory holding the real checkpoints")
    return Path(root)


def resolve_checkpoint(subpath: str) -> Path:
    """`LOOM_MODELS`-relative, unless the entry names its own variable.

    Most checkpoints are directories under one root. Two are not -- the Supertonic fork and the Piper
    VITS voice are separate repos -- so those entries are written `$VAR/rest` and skip when `VAR` is
    unset, rather than pinning one machine's paths into a list that is meant to be a statement about
    which architectures this repo exports."""
    if subpath.startswith("$"):
        var, _, rest = subpath[1:].partition("/")
        root = os.environ.get(var)
        if not root:
            pytest.skip(f"{var} is not set; it names the checkpoint this entry needs")
        return Path(root) / rest if rest else Path(root)
    return models_root() / subpath


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
    checkpoint = resolve_checkpoint(subpath)
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint} is not present")

    gguf = export(name, checkpoint, extra, tmp_path)
    assert gguf.exists() and gguf.stat().st_size > 0

    current = snapshot(gguf, tmp_path / "snap")
    # The artifact's whole purpose here is the snapshot, and pytest keeps `tmp_path` for the session
    # (and for two sessions after it), so holding these would make the sweep's disk cost the SUM over
    # models instead of the largest one. Granite-Speech alone is 8.75 GB and the full list is ~30 GB;
    # this machine has run as low as 19 GB free. Nothing below reads the file again.
    gguf.unlink()
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
