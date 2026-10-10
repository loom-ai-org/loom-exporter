"""sanoTTS -- the hermetic half.

The export builds upstream's own modules from a GPL-3.0 clone that CI does not have, so what is tested
here is everything that does not need it: the package reader, the phoneme framing it declares, the
recognizer, and the two exporter changes the model needed, through the real compiler --

* a GET_ROWS gathering rows out of a PERMUTE gets a CONT in between (ggml's `get_rows` ignores the
  table's element stride; the frame expansion's audio came out at correlation 0.002 without it);
* `blank_after_bos` reaches the file only when declared, so every earlier phoneme vocabulary is
  byte-identical.

The real packages are the gate half: amy-en-1p46m's waveform matches upstream's torch path end to end at
7.2e-6 (correlation 1.0), and a Whisper transcript of it through the text door is word for word.
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import loom_exporter  # noqa: F401 -- registers the "loom" backend
from loom_exporter import sanotts_export as S


def _export_two_inputs(module, tmp_path, name="toy"):
    """Trace `module(x [1, C, T], index [F])` with both lengths dynamic; return its nodes."""
    import coremltools as ct
    from gguf import GGUFReader

    x, index = torch.randn(1, 4, 7), torch.tensor([0, 0, 3, 6, 6])
    traced = torch.jit.trace(module.eval(), (x, index))
    program = ct.convert(
        traced, inputs=[ct.TensorType(name="x", shape=(1, 4, ct.RangeDim(1, 512))),
                        ct.TensorType(name="index", shape=(ct.RangeDim(1, 512),), dtype=np.int32)],
        convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    out = tmp_path / f"{name}.gguf"
    loom_exporter.LoomGGUFBackend()(program, output_path=str(out), architecture=name,
                                    root_axis="n_tokens", flat_namespace=True,
                                    declared_axes={"index": {0: "n_enc_frames"}})
    reader = GGUFReader(str(out))
    key = next(k for k in reader.fields if k.startswith("model.graph_topology"))
    return json.loads(reader.fields[key].contents())["nodes"]


class _GatherFromConvOutput(nn.Module):
    """The synth phase's expansion in miniature: a conv stack's `[C, T]` read back as rows of `[T, C]`."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 3, padding=1)

    def forward(self, x, index):
        y = self.conv(x)
        return torch.index_select(y.squeeze(0).transpose(0, 1), 0, index)


class _EmbeddingLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(9, 4)

    def forward(self, x, index):
        return self.table(index) + x.sum(dim=2)


def test_a_gather_out_of_a_view_is_packed_first(tmp_path):
    nodes = _export_two_inputs(_GatherFromConvOutput(), tmp_path)
    producer = {out: n for n in nodes for out in n.get("outputs", [])}
    (gather,) = [n for n in nodes if n["op"] == "GET_ROWS"]
    cont = producer[gather["inputs"][0]]
    assert cont["op"] == "CONT", "the table of a gather must be a packed buffer, not a view"
    assert producer[cont["inputs"][0]]["op"] in ("PERMUTE", "TRANSPOSE")


def test_a_lookup_into_a_weight_table_is_unchanged(tmp_path):
    nodes = _export_two_inputs(_EmbeddingLookup(), tmp_path)
    assert not any(n["op"] == "CONT" for n in nodes)


class _Double(nn.Module):
    def forward(self, x):
        return x * 2.0


def _write_phonemes(tmp_path, table):
    """A GGUF carrying only a phoneme vocabulary, through the real backend."""
    import coremltools as ct
    from gguf import GGUFReader

    traced = torch.jit.trace(_Double(), (torch.randn(1, 3),))
    program = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 8)))],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    out = tmp_path / "phonemes.gguf"
    loom_exporter.LoomGGUFBackend()(program, output_path=str(out), architecture="toy",
                                    root_axis="n_tokens", flat_namespace=True, phoneme_table=table)
    return GGUFReader(str(out)).fields


_TABLE = {"symbols": ["_", "^", "$", "a"], "ids": [0, 1, 2, 3], "bos": 1, "eos": 2, "blank": 0,
          "interleave_blank": True}


def test_blank_after_bos_is_written_only_when_declared(tmp_path):
    key = "tokenizer.ggml.phoneme.blank_after_bos"
    assert key not in _write_phonemes(tmp_path, _TABLE)
    fields = _write_phonemes(tmp_path, dict(_TABLE, blank_after_bos=True))
    assert key in fields and bool(fields[key].contents())


def _package(tmp_path, id_map, *, fmt="roota.raw-fp16.v1"):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": fmt}))
    (tmp_path / "piper-phoneme-config.json").write_text(json.dumps({"phoneme_id_map": id_map}))
    return tmp_path


def test_the_framing_is_piper_phonemizes(tmp_path):
    pkg = _package(tmp_path, {"_": [0], "^": [1], "$": [2], "a": [5], "b": [4]})
    table = S.TTSSanoTTSExportConfig(package_dir=str(pkg)).phoneme_table()
    assert table["symbols"] == ["_", "^", "$", "b", "a"] and table["ids"] == [0, 1, 2, 4, 5]
    assert (table["bos"], table["eos"], table["blank"]) == (1, 2, 0)
    assert table["interleave_blank"] and table["blank_after_bos"]


def test_moved_framing_ids_are_refused(tmp_path):
    pkg = _package(tmp_path, {"_": [0], "^": [3], "$": [2], "a": [5]})
    with pytest.raises(ValueError, match="framing ids"):
        S.TTSSanoTTSExportConfig(package_dir=str(pkg)).phoneme_table()


def test_a_multi_id_symbol_is_refused(tmp_path):
    pkg = _package(tmp_path, {"_": [0], "^": [1], "$": [2], "a": [5, 6]})
    with pytest.raises(ValueError, match="several ids"):
        S.TTSSanoTTSExportConfig(package_dir=str(pkg)).phoneme_table()


def test_the_package_reader_slices_the_blob_by_the_manifest(tmp_path):
    a = np.arange(6, dtype=np.float16).reshape(2, 3)
    b = np.array([7.5], dtype=np.float16)
    blob = a.tobytes() + b.tobytes()
    (tmp_path / "w.bin").write_bytes(blob)
    tensors = {"duration": [{"name": "x", "shape": [2, 3], "dtype": "float16", "offset_bytes": 0}],
               "acoustic": [{"name": "y", "shape": [1], "dtype": "float16", "offset_bytes": a.nbytes}],
               "decoder": []}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "roota.raw-fp16.v1", "weights_file": "w.bin", "weights_size_bytes": len(blob),
        "components": {c: {"tensors": t} for c, t in tensors.items()}}))
    _, states = S.read_package(tmp_path)
    assert torch.equal(states["duration"]["x"], torch.from_numpy(a.astype(np.float32)))
    assert states["acoustic"]["y"].item() == 7.5 and states["decoder"] == {}


def test_a_truncated_blob_is_refused(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\0" * 6)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "roota.raw-fp16.v1", "weights_file": "w.bin", "weights_size_bytes": 8, "components": {}}))
    with pytest.raises(ValueError, match="6 bytes"):
        S.read_package(tmp_path)


def test_the_recognizer_reads_the_manifest_format(tmp_path):
    assert S._is_sanotts(_package(tmp_path, {}))
    other = tmp_path / "nano"
    other.mkdir()
    assert not S._is_sanotts(_package(other, {}, fmt="nano.v1"))
    assert not S._is_sanotts(tmp_path / "missing")


def test_a_missing_clone_says_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="git clone https://github.com/Ampixa/sanoTTS"):
        S.load_sanotts_modules(str(tmp_path))


def test_it_is_registered_for_tts():
    from loom_exporter.registry import default_registry

    registry = default_registry()
    names = {(rec.task, rec.name) for entry in registry._entries.values() for rec in entry.recognizers}
    assert ("text-to-speech", "sanotts") in names


# -- the nano line: the C runtime's blobs ---------------------------------------------------------------

def _nano_package(tmp_path, weight_format):
    """One 2x3 layer, rows padded to 16, plus one f32 region -- the header/blob discipline upstream's
    `export_e12_nano_q8.py` writes."""
    q = np.array([[1, -2, 3], [127, 0, -127]], dtype=np.int8)
    scale = np.array([0.5, 0.25], dtype=np.float32)
    bias = np.array([1.0, -1.0], dtype=np.float32)
    if weight_format == 1:
        rows = np.zeros((2, 16), np.float32)
        rows[:, :3] = q * scale[:, None]
        w_bytes, scale = rows.tobytes(), np.ones(2, np.float32)
    else:
        rows = np.zeros((2, 16), np.int8)
        rows[:, :3] = q
        w_bytes = rows.tobytes()
    f32 = np.array([3.5, -4.25], dtype=np.float32)
    dec = w_bytes + scale.tobytes() + bias.tobytes() + f32.tobytes()
    w_off, s_off = 0, len(w_bytes)
    header = {"NANO_FRONT_BYTES": 0, "NANO_DEC_BYTES": len(dec), "NANO_L_N16": 16,
              "DOFF_L_W8": w_off, "DOFF_L_SCALE": s_off, "DOFF_L_BIAS": s_off + 8,
              "DOFF_G_F32": s_off + 16}
    if weight_format == 1:
        header["NANO_WEIGHT_FORMAT"] = 1
    (tmp_path / "nano_q8_meta.h").write_text(
        "/* generated */\n" + "".join(f"#define {k} {v}\n" for k, v in header.items()))
    (tmp_path / "front.bin").write_bytes(b"")
    (tmp_path / "dec.bin").write_bytes(dec)
    (tmp_path / "meta.json").write_text(json.dumps({"front": "front.bin", "dec": "dec.bin"}))
    return tmp_path, q.astype(np.float32) * np.array([[0.5], [0.25]], np.float32), bias, f32


@pytest.mark.parametrize("weight_format", [0, 1])
def test_a_nano_layer_is_dequantised_and_its_padding_dropped(tmp_path, weight_format):
    pkg, want_w, want_b, want_f32 = _nano_package(tmp_path, weight_format)
    blobs = S._NanoBlobs(pkg)
    w, b = blobs.rows("dec", "DOFF", "L", 2, 3)
    assert w.shape == (2, 3)
    assert torch.equal(w, torch.from_numpy(want_w)) and torch.equal(b, torch.from_numpy(want_b))
    assert torch.equal(blobs.f32("dec", "DOFF", "G", 2), torch.from_numpy(want_f32))


def test_a_nano_blob_of_the_wrong_size_is_refused(tmp_path):
    pkg, *_ = _nano_package(tmp_path, 0)
    (pkg / "dec.bin").write_bytes((pkg / "dec.bin").read_bytes()[:-4])
    with pytest.raises(ValueError, match="the header says"):
        S._NanoBlobs(pkg)


def test_a_dyt_decoder_is_refused(tmp_path):
    pkg, *_ = _nano_package(tmp_path, 0)
    with open(pkg / "nano_q8_meta.h", "a") as f:
        f.write("#define NANO_NORM_TYPE 1\n")
    with pytest.raises(ValueError, match="DyT"):
        S._NanoBlobs(pkg)


def test_the_nano_recognizer_wants_the_header(tmp_path):
    pkg, *_ = _nano_package(tmp_path, 0)
    assert S._is_sanotts_nano(pkg) and not S._is_sanotts(pkg)
    (pkg / "nano_q8_meta.h").unlink()
    assert not S._is_sanotts_nano(pkg)


def test_the_default_seed_is_upstreams_low_32_bits():
    assert S.NANO_DEFAULT_SEED == 2236265385529901705 % 2 ** 32


def test_each_line_declares_the_phoneme_style_it_was_trained_in(tmp_path):
    """piperlite's teachers were Piper voices (espeak IPA); nano was distilled on misaki's. The text
    door folds a G2P's output to it (loom.cpp ADR-071)."""
    piperlite = S.TTSSanoTTSExportConfig(package_dir=str(tmp_path / "missing"))
    assert piperlite.hparams() == {"sample_rate": 22050, "tts.phoneme_style": "espeak"}
    nano = S.TTSSanoNanoExportConfig(package_dir=str(tmp_path / "missing"))
    assert nano.hparams() == {"sample_rate": 24000, "tts.phoneme_style": "misaki"}
