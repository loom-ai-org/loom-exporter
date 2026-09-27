"""P5.0's second change: a phase packs its own weights as it converts, not the writer at the end.

**It is output-preserving, and that is the first thing worth asserting.** Moving the dtype cast, the
conv-kernel fold and the quantization from `write_gguf` to `convert_phase` must not change one byte of
any artifact, and almost nothing about "the exporter still runs" would notice if it had. The gate
sweep is what holds that for the real zoo; what is here is the half a byte-identity check *cannot*
see, which is that the move actually happened -- an exporter that had quietly kept packing everything
at write time would produce the identical file and still carry every phase's F32 arrays to the end,
which is the whole quantity this change exists to reduce.

The rest of the file is the invariant the move rests on: a phase may answer a question about *all*
topologies from its own only because multi-phase weights carry their own phase's prefix.

Run: ~/.venvs/piper/bin/python3 -m pytest tests/ci/test_phase_packing.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_t5_export import _tiny_t5  # noqa: E402  (the one real multi-phase config CI can build)

from loom_exporter.t5_export import Text2TextT5ExportConfig  # noqa: E402


def aligned_t5(tmp_path: Path, name: str = "tiny-t5") -> Path:
    """`test_t5_export`'s synthetic T5, widened so that `--quantize Q8_0` can actually reach something.

    **Without this a Q8_0 arm is an F32 arm wearing its name.** That fixture is `d_model=8, d_ff=16`,
    and ggml lays quantization blocks along the tensor's fastest axis: a weight whose last dimension is
    8 is declined for shape, every time, so a "quantized" export of it comes out byte-identical to its
    F32 one -- which is exactly the false pass P4.13 was chasing when it found that every one of VITS's
    conv kernels was being declined. 64 is a multiple of Q8_0's block size, so the attention and
    feed-forward weights are eligible and a coverage assertion can fail.
    """
    return _tiny_t5(tmp_path, name=name, d_model=64, d_ff=64)


def export_t5(checkpoint: Path, out: Path, **kwargs) -> Path:
    config = Text2TextT5ExportConfig(model_dir=str(checkpoint), output_path=str(out), **kwargs)
    config.task = "text2text-generation"
    config.export()
    return out


# --------------------------------------------------------------------------------------------------
# The packing happened at the PHASE, which byte-identity cannot see.
# --------------------------------------------------------------------------------------------------

def test_each_phase_hands_over_packed_bytes_rather_than_f32_arrays(tmp_path):
    """The item itself (BACKLOG.md P5.0): what a converted phase carries until the last phase converts
    is now its on-disk payload, not its F32 weights.

    Asserted by watching what `merge_phase_weights` is handed -- at Q8_0 a phase's matmul weights must
    already be uint8 blocks. An exporter that had moved nothing and simply packed everything in
    `write_gguf` would write the identical file and fail here, which is precisely the difference the
    gate sweep cannot draw.
    """
    pytest.importorskip("coremltools")
    import loom_exporter.multi_phase_export as mpe

    # `_write` imports the merge by name at call time, so rebinding it on the module is enough -- and
    # the merge is the LAST thing every phase's weights pass through before the writer, which is
    # exactly the point at which "what is still being carried" is the question.
    seen = {}
    real_merge = mpe.merge_phase_weights

    def spy(named_weights):
        for name, weights in named_weights:
            seen[name] = {k: v.dtype for k, v in weights.items()}
        return real_merge(named_weights)

    mpe.merge_phase_weights = spy
    try:
        export_t5(aligned_t5(tmp_path), tmp_path / "q.gguf", quantize="Q8_0")
    finally:
        mpe.merge_phase_weights = real_merge

    assert seen, "no phase reached the merge"
    quantized = [(phase, weight) for phase, dtypes in seen.items()
                 for weight, dtype in dtypes.items() if dtype == np.uint8]
    assert quantized, (
        f"no phase handed over a quantized payload; the merge saw "
        f"{ {phase: sorted({str(d) for d in dtypes.values()}) for phase, dtypes in seen.items()} }"
    )


def test_a_second_pack_leaves_an_already_packed_weight_alone(tmp_path):
    """`pack_weights` is what the output exporter calls over a dict the phases already packed, so it
    has to be idempotent per name. Quantizing a Q8_0 payload a second time would read its blocks as
    F32 samples -- which does not raise, it just writes noise."""
    from loom_exporter.exporter import LoomGGUFExporter

    exporter = LoomGGUFExporter(None, output_path=str(tmp_path / "x.gguf"), architecture="test",
                               quantize="Q8_0")
    exporter.topologies = {"t": {"nodes": [{"op": "MUL_MAT", "inputs": ["w", "act"]}]}}
    exporter.weights = {"w": np.full((4, 256), 0.25, dtype=np.float32)}
    first = exporter.pack_weights()
    packed = exporter.weights["w"]
    assert packed.dtype == np.uint8, "the first pack should have quantized it"
    second = exporter.pack_weights()
    assert second is first, "the packing record is accumulated, not replaced"
    assert exporter.weights["w"] is packed
    assert second.n_quantized == 1, "a second pass must not count it again"


def test_a_driver_weight_added_after_the_phases_is_still_packed(tmp_path):
    """`driver_weights` are the one thing that reaches the output exporter unpacked -- they are added
    inside `write_gguf`, after pruning, and no phase ever saw them. Kokoro's and Supertonic's default
    voice styles are these."""
    from gguf import GGUFReader

    from loom_exporter.exporter import LoomGGUFExporter, WeightPacking

    out = tmp_path / "d.gguf"
    exporter = LoomGGUFExporter(None, output_path=str(out), architecture="test",
                               driver_weights={"style": np.zeros((2, 8), dtype=np.float64)})
    exporter.topologies = {"t": {"nodes": [{"op": "MUL_MAT", "inputs": ["w", "act"]}]}}
    exporter.weights = {"w": np.full((4, 256), 0.5, dtype=np.float32),
                        "act": np.full((4, 256), 0.5, dtype=np.float32)}
    # As a multi-phase export arrives: the graph weights already packed, the driver's not yet.
    exporter.packing = WeightPacking(raw_dtypes={"w": None, "act": None})
    exporter.write_gguf("-- driver")
    types = {t.name: t.tensor_type.name for t in GGUFReader(str(out)).tensors}
    assert types["style"] == "F32", "an F64 driver weight is cast on the way in, like every other one"


# --------------------------------------------------------------------------------------------------
# The invariant per-phase packing rests on.
# --------------------------------------------------------------------------------------------------

def test_a_weight_read_by_another_phases_topology_is_refused():
    """Why per-phase packing is allowed to answer a question about ALL topologies from ONE.

    Both quantization gates -- eligibility by op, and the conv fold's "this name's entire use is this
    convolution's first input" -- are read off every topology. A phase exporter sees only its own, and
    that is the same answer only because a multi-phase export prefixes every weight with its own
    phase's name. Violated, the export does not fail: it writes a kernel folded for a consumer that
    wanted the declared shape. So the invariant is checked.
    """
    from loom_exporter.decomposition import _check_phase_weight_namespaces

    outputs = [
        ("encoder", {"encoder": {"nodes": [{"op": "MUL_MAT", "inputs": ["encoder.w", "x"]}]}},
         {"encoder.w": np.zeros(4)}),
        # The violation: the decoder reads a tensor the encoder produced and packed.
        ("decoder", {"decoder": {"nodes": [{"op": "CONV_1D", "inputs": ["encoder.w", "x"]}]}},
         {"decoder.w": np.zeros(4)}),
    ]
    with pytest.raises(ValueError, match="reads weight"):
        _check_phase_weight_namespaces(outputs)


def test_a_phases_own_extra_stream_and_recurrent_cells_are_not_violations():
    """Ownership is by PHASE, not by topology name, and this is what that distinction buys.

    One phase routinely owns topologies named something other than itself: an `extra_streams` alias
    (family 10's classifier-free guidance runs one decoder as two streams) and a `RecurrentPhase`'s
    cells (`text_encoder_lstm` emits `_fwd` and `_bwd`). Keyed on the topology's NAME, all three rows
    below read as a phase reading another phase's weights.
    """
    from loom_exporter.decomposition import _check_phase_weight_namespaces

    decoder_nodes = [{"op": "MUL_MAT", "inputs": ["decoder.w", "x"]}]
    outputs = [
        ("decoder", {"decoder": {"nodes": decoder_nodes},
                     "decoder_uncond": {"nodes": decoder_nodes}},
         {"decoder.w": np.zeros(4)}),
        ("text_encoder_lstm",
         {"text_encoder_lstm_fwd": {"nodes": [{"op": "MUL_MAT",
                                               "inputs": ["text_encoder_lstm.weight_ih", "x"]}]},
          "text_encoder_lstm_bwd": {"nodes": [{"op": "MUL_MAT",
                                               "inputs": ["text_encoder_lstm.weight_hh", "h"]}]}},
         {"text_encoder_lstm.weight_ih": np.zeros(4),
          "text_encoder_lstm.weight_hh": np.zeros(4)}),
    ]
    _check_phase_weight_namespaces(outputs)


def test_a_graph_input_is_not_mistaken_for_another_phases_weight():
    """Only a name some phase actually produced can be read by the wrong one. A topology naturally
    reads inputs and intermediate values no phase ever wrote."""
    from loom_exporter.decomposition import _check_phase_weight_namespaces

    outputs = [("encoder",
                {"encoder": {"nodes": [{"op": "MUL_MAT", "inputs": ["encoder.w", "tokens"]},
                                       {"op": "ADD", "inputs": ["var_3", "var_7"]}]}},
                {"encoder.w": np.zeros(4)})]
    _check_phase_weight_namespaces(outputs)
