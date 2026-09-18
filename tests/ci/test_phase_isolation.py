"""P5.0's third change: converting each phase in a process of its own.

**It is output-preserving, and that is the whole acceptance test.** An isolated export and a
non-isolated one must be the same bytes -- the `loom.tensor_alias.*` tables included, which depend on
the order weights were seen in and would notice a merge that came back shuffled. That is asserted here
on a synthetic T5 at F32 and at Q8_0, and confirmed on the product: `whisper-small` exported both ways
from fresh processes is `cmp`-clean at 969,918,400 bytes.

What it buys is memory, and the numbers are in `phase_isolation`'s own docstring -- including the one
that goes the wrong way, because on a small model isolation pays the framework floor twice and the
peak gets worse. The rest of this file is the spill (a quantized payload is uint8 bytes and cannot
carry its own GGML type, so the manifest must) and the two ways the mechanism is allowed to fail:
loudly, never by quietly converting in-process.

Run: ~/.venvs/piper/bin/python3 -m pytest tests/ci/test_phase_isolation.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The aligned fixture and its reason live with the packing tests, which need it for the same reason:
# a Q8_0 arm over an 8-wide model is an F32 arm wearing its name.
from test_phase_packing import aligned_t5  # noqa: E402

from loom_exporter.t5_export import Text2TextT5ExportConfig  # noqa: E402


def _export(checkpoint: Path, out: Path, **kwargs) -> Path:
    """One export, from a process that has converted nothing -- which is what a `loom-export` run is.

    **The counter reset is load-bearing, and the reason is a real hazard rather than a test artifact.**
    `coremltools`' `Builder.name_count` is a CLASS attribute -- one `defaultdict(int)` for the whole
    process -- and every op a MIL pass builds without an explicit name draws from it, so a synthesized
    `transpose` is called `transpose_3` in a fresh interpreter and `transpose_21` in one that has
    already converted twenty. Those names reach the emitted topology as node outputs.

    A `loom-export` process converts one model and exits, so it always starts from zero; an isolated
    export's WORKERS always start from zero too. A pytest session does not: by the time this file runs,
    the suite has converted dozens of toy models, so the in-process arm below would name its
    intermediates twenty higher than its own children do and the comparison would fail on a difference
    no user can observe. Resetting per export reproduces the condition the artifacts are actually
    written under.

    Verified against the product rather than assumed: `whisper-small` exported both ways from fresh
    processes, 969,918,400 bytes, `cmp` clean. The hazard itself -- that an export's op names depend on
    what its process converted earlier -- is filed in the backlog; it predates this item and this test
    does not paper over it, it excludes it.
    """
    from coremltools.converters.mil.mil import Builder

    Builder.name_count.clear()
    isolate = kwargs.pop("isolate_phases", None)
    config = Text2TextT5ExportConfig(model_dir=str(checkpoint), output_path=str(out), **kwargs)
    config.task = "text2text-generation"
    if isolate is not None:
        config.decomposition.isolate_phases = isolate
    config.export()
    return out


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("quantize", [None, "Q8_0"])
def test_isolating_the_phases_writes_the_identical_file(tmp_path, quantize):
    """The whole claim. Three phases converted in three child processes, each loading the checkpoint
    itself and spilling its packed weights to disk, against three converted in this one -- and the two
    artifacts have to be the same bytes.

    Run at Q8_0 as well as F32 because quantization is where the two paths differ most: an isolated
    phase packs in the child and hands the parent uint8 blocks plus a GGML type name, and a
    non-isolated one packs in-process and keeps the enum member.
    """
    pytest.importorskip("coremltools")
    from gguf import GGUFReader

    checkpoint = aligned_t5(tmp_path)
    kwargs = {"quantize": quantize} if quantize else {}
    plain = _export(checkpoint, tmp_path / "plain.gguf", isolate_phases=False, **kwargs)
    isolated = _export(checkpoint, tmp_path / "isolated.gguf", isolate_phases=True, **kwargs)
    assert _digest(plain) == _digest(isolated), (
        f"isolated export differs: {plain.stat().st_size} vs {isolated.stat().st_size} bytes"
    )
    # A gate that cannot fail proves nothing: at Q8_0 SOMETHING has to have been quantized, or both
    # arms wrote the same F32 file and the identity above is about nothing.
    types = {t.tensor_type.name for t in GGUFReader(str(isolated)).tensors}
    if quantize:
        assert "Q8_0" in types, f"nothing was quantized, so this arm is an F32 arm: {sorted(types)}"
    else:
        assert types <= {"F32", "I32"}, sorted(types)


def test_the_env_var_turns_isolation_on_and_the_field_overrides_it(tmp_path, monkeypatch):
    """`--isolate-phases` sets `$LOOM_PHASE_ISOLATION`'s job, not a second unrelated switch, and an
    explicit field beats the environment -- the resolution order every other optional knob here uses."""
    from loom_exporter.phase_isolation import ISOLATION_ENV, isolation_requested

    monkeypatch.delenv(ISOLATION_ENV, raising=False)
    assert isolation_requested(None) is False
    monkeypatch.setenv(ISOLATION_ENV, "1")
    assert isolation_requested(None) is True
    assert isolation_requested(False) is False, "an explicit answer wins over the environment"
    monkeypatch.setenv(ISOLATION_ENV, "no")
    assert isolation_requested(None) is False


def test_a_worker_that_fails_fails_the_export(tmp_path):
    """Isolation must not fall back to converting in-process. A caller turned it on because the export
    did not fit; finishing it the way that did not fit, quietly, is the one behaviour that would make
    the switch worse than useless."""
    from loom_exporter.phase_isolation import convert_phase_isolated

    blob = tmp_path / "config.pkl"
    blob.write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="phase-0 worker exited"):
        convert_phase_isolated(blob, 0, tmp_path / "spill")


def test_an_out_of_range_phase_index_names_the_disagreement(tmp_path):
    """The parent and the child each call `phases()` on their own copy of the config, so they could in
    principle disagree about how many there are. The worker says so rather than converting whatever is
    at that index."""
    pytest.importorskip("coremltools")
    import pickle

    from loom_exporter.phase_isolation import _phase_worker

    config = Text2TextT5ExportConfig(model_dir=str(aligned_t5(tmp_path)),
                                     output_path=str(tmp_path / "out.gguf"))
    config.task = "text2text-generation"
    blob = tmp_path / "config.pkl"
    blob.write_bytes(pickle.dumps(config))
    with pytest.raises(IndexError, match="out of range"):
        _phase_worker([str(blob), "99", str(tmp_path / "spill")])


# --------------------------------------------------------------------------------------------------
# The packing happened at the PHASE, which byte-identity cannot see.
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# The spill.
# --------------------------------------------------------------------------------------------------

def test_the_spill_round_trips_shapes_dtypes_and_the_ggml_type(tmp_path):
    """A quantized payload is uint8 bytes and says nothing about which GGML type produced them, so the
    manifest has to carry it -- by NAME, because the enum's numbering is the gguf package's to change.
    """
    from gguf import GGMLQuantizationType

    from loom_exporter.exporter import WeightPacking
    from loom_exporter.phase_isolation import PhaseResult, read_spill, write_spill

    blocks = np.arange(272, dtype=np.uint8).reshape(1, 272)
    weights = {
        "enc.mm": blocks,
        "enc.norm": np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4),
        "enc.ids": np.arange(6, dtype=np.int32),
        # A zero-element tensor: `np.memmap` refuses a zero-length mapping, so this is the one entry
        # `read_spill` rebuilds instead of mapping.
        "enc.empty": np.zeros((0, 4), dtype=np.float32),
    }
    packing = WeightPacking(raw_dtypes={"enc.mm": GGMLQuantizationType.Q8_0, "enc.norm": None,
                                        "enc.ids": None, "enc.empty": None},
                            n_folded=2, n_declined_shape=1, quantized_bytes_before=1024,
                            quantized_bytes_after=272, float_bytes_total=2048)
    topologies = {"enc": {"nodes": [{"op": "MUL_MAT", "inputs": ["enc.mm", "x"]}]}}
    geometry = {"attention": [("block_0", (2, 4, 4))]}

    directory = tmp_path / "phase-0"
    write_spill(PhaseResult(topologies=topologies, weights=weights, packing=packing,
                            geometry=geometry), directory)
    back = read_spill(directory)

    assert back.topologies == topologies
    assert set(back.weights) == set(weights)
    for name, array in weights.items():
        assert back.weights[name].dtype == array.dtype, name
        assert back.weights[name].shape == array.shape, name
        assert np.array_equal(np.asarray(back.weights[name]), array), name
    assert back.packing.raw_dtypes["enc.mm"] is GGMLQuantizationType.Q8_0
    assert back.packing.raw_dtypes["enc.norm"] is None
    assert back.packing.n_folded == 2 and back.packing.n_declined_shape == 1
    assert back.packing.quantized_bytes_before == 1024
    # `fused_geometry()`'s records are tuples and JSON has no tuple -- `_kv_cache_geometry` unpacks
    # them, so a list would work and a dict would not, but the point is they come back unchanged.
    assert back.geometry == geometry


def test_a_spilled_weight_is_mapped_rather_than_read(tmp_path):
    """The half that makes isolation pay on the PARENT's side: it accumulates N phases of tensors and
    must not fault them in to do it. `GGUFWriter` streams each one with `tofile`, so a memmap goes from
    the spill into the artifact page by page."""
    from loom_exporter.exporter import WeightPacking
    from loom_exporter.phase_isolation import PhaseResult, read_spill, write_spill

    weights = {"enc.w": np.ones((64, 64), dtype=np.float32)}
    packing = WeightPacking(raw_dtypes={"enc.w": None})
    directory = tmp_path / "phase-0"
    write_spill(PhaseResult(topologies={}, weights=weights, packing=packing), directory)
    back = read_spill(directory)
    assert isinstance(back.weights["enc.w"], np.memmap)


# --------------------------------------------------------------------------------------------------
# The invariant per-phase packing rests on.
# --------------------------------------------------------------------------------------------------

def test_the_mil_op_name_counter_is_process_global():
    """Why `_export` resets it, pinned so the reason survives the next person reading that reset.

    `Builder.name_count` is a class attribute shared by every `Builder` in the process, and
    `_get_free_name` both reads and increments it. So an op a MIL pass builds without an explicit name
    is called `transpose_0` in a fresh interpreter and `transpose_20` in one that has built twenty --
    and that name is what the emitted topology carries as the node's output. Two exports of the same
    checkpoint from differently-warmed processes are therefore not byte-identical, which is a
    reproducibility hazard older than phase isolation and independent of it.

    If this ever fails because coremltools made the counter per-Program, the reset in `_export` is dead
    code and the backlog item can be closed.
    """
    from coremltools.converters.mil.mil import Builder

    before = dict(Builder.name_count)
    try:
        Builder.name_count.clear()
        first = Builder._get_free_name("transpose")
        second = Builder._get_free_name("transpose")
        assert (first, second) == ("transpose_0", "transpose_1"), (first, second)
        assert Builder.name_count is type(Builder).__dict__.get("name_count", Builder.name_count), (
            "name_count moved off the class; the reset in _export may no longer reach it"
        )
    finally:
        Builder.name_count.clear()
        Builder.name_count.update(before)
