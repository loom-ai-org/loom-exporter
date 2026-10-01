"""Family 13 -- small audio classifiers and embedders (P5; loom.cpp ADR-062).

The hermetic half: what the contract declares, the attentive-pool rewrite both loaders share, the
recognizers, and the four exporter changes the family needed -- each traced through the real compiler on
a toy module, because every one of them was found as a SILENT wrong answer or a wrong length, never as an
error:

* `loom_mean` with `keep_dims=False` left ggml's unit `ne[0]` in place, so a CONCAT after it ran along the
  wrong axis and interleaved its operands (ECAPA's pool: rel 0.95 at the next matmul);
* `batch_norm` was missing from the shape walk, so every frame axis after an unfolded norm read as the
  root axis (speechbrain's TDNN blocks are conv -> ReLU -> norm, which coremltools cannot fold);
* `max_pool` had no lowering and no walk case (pyannote's SincNet);
* `run_bi_recurrent_and_retain` was unknown to the retained-read check.

The real checkpoints are the gate half: `tests/gate/` would need four downloads, and the oracle numbers
are in loom.cpp's Epic-03.
"""
import io
import json
import tarfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import loom_exporter  # noqa: F401 -- registers the "loom" backend
from loom_exporter import audio_classification_export as A


# -- the contract -----------------------------------------------------------------------------------

def test_the_contract_names_the_granularity_for_every_leaf():
    """`output.granularity` is written even where a host could guess it from a one-row answer:
    guessing from the SHAPE is what made `class` mean "per token" by accident."""
    clip = A.audio_contract(A.AudioOutput.CLIP_CLASSES, 16000, ["en", "th"], None)
    assert clip == {"task": "audio-classification", "input.kind": "audio", "output.kind": "class",
                    "output.granularity": "clip", "sample_rate": 16000, "labels": ["en", "th"]}
    emb = A.audio_contract(A.AudioOutput.EMBEDDING, 16000, [], None)
    assert emb["task"] == "audio-embedding" and emb["output.kind"] == "embeddings"
    assert emb["output.granularity"] == "clip" and "labels" not in emb


def test_a_frame_output_declares_its_rate_and_only_a_nonzero_offset():
    vad = A.audio_contract(A.AudioOutput.FRAME_CLASSES, 16000, ["non_speech", "speech"], 50.0)
    assert vad["output.frame_rate"] == 50.0 and "output.frame_offset" not in vad
    seg = A.audio_contract(A.AudioOutput.FRAME_CLASSES, 16000, ["a"], 59.26, 0.0225)
    assert seg["output.frame_offset"] == 0.0225
    # Floats, not ints: the GGUF writer picks the KV type off the Python type.
    assert isinstance(seg["output.frame_rate"], float)


def test_a_frame_rate_on_a_clip_output_is_not_declared():
    clip = A.audio_contract(A.AudioOutput.CLIP_CLASSES, 16000, ["x"], 50.0, 0.5)
    assert "output.frame_rate" not in clip and "output.frame_offset" not in clip


def test_the_vad_training_targets_become_names():
    assert A._frame_vad_labels(["0", "1"]) == ["non_speech", "speech"]
    assert A._frame_vad_labels(["music", "speech", "noise"]) == ["music", "speech", "noise"]


# -- the attentive statistics pool --------------------------------------------------------------------

class _Tdnn(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 1)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(cout)


class _ReferencePool(nn.Module):
    """The NeMo/speechbrain pool as both write it: statistics TILED over time with `.repeat`, masked
    frames filled with `-inf`. What `_install_decomposed_pool` must reproduce."""

    def __init__(self, channels=6, attention=4):
        super().__init__()
        self.tdnn = _Tdnn(3 * channels, attention)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention, channels, 1)
        self.eps = 1e-12

    def forward(self, x, length=None):
        L = x.shape[2]
        lengths = length if length is not None else torch.full((x.shape[0],), L)
        mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).float()

        def stats(w):
            mean = (w * x).sum(2)
            return mean, torch.sqrt((w * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))

        mean, std = stats(mask / mask.sum(2, keepdim=True))
        attn = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, L), std.unsqueeze(2).repeat(1, 1, L)], 1)
        attn = self.conv(self.tanh(self.tdnn.norm(self.tdnn.activation(self.tdnn.conv(attn)))))
        attn = attn.masked_fill(mask == 0, float("-inf"))
        mu, sg = stats(F.softmax(attn, dim=2))
        return torch.cat((mu, sg), 1).unsqueeze(2)


def _pool(whole_clip):
    torch.manual_seed(0)
    pool = _ReferencePool().eval()
    with torch.no_grad():
        for p in pool.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
        pool.tdnn.norm.running_mean.normal_()
        pool.tdnn.norm.running_var.uniform_(0.5, 2.0)
    reference = _ReferencePool().eval()
    reference.load_state_dict(pool.state_dict())
    A._install_decomposed_pool(pool, conv_in=pool.tdnn.conv,
                               after_conv_in=lambda h: pool.tdnn.norm(pool.tdnn.activation(h)),
                               tanh=pool.tanh, conv_out=pool.conv, eps=pool.eps, whole_clip=whole_clip)
    return pool, reference


def test_the_decomposed_pool_is_the_tiled_one_with_a_live_mask():
    """A 1x1 convolution over `[x; mean; std]` is a frame term plus a time-constant one -- no repeat, a
    third of the work -- and the masked frames still drop out of the softmax."""
    pool, reference = _pool(whole_clip=False)
    x = torch.randn(1, 6, 23)
    for valid in (23, 17, 1):
        with torch.no_grad():
            got, want = pool(x, torch.tensor([valid])), reference(x, torch.tensor([valid]))
        assert torch.allclose(got, want, atol=1e-5), valid
    # The mask is live: padding with garbage past the valid length changes nothing.
    padded = torch.cat([x[:, :, :17], torch.randn(1, 6, 9) * 100], 2)
    with torch.no_grad():
        assert torch.allclose(pool(padded, torch.tensor([17])), reference(x, torch.tensor([17])),
                              atol=1e-4)


def test_the_whole_clip_pool_takes_no_length():
    """speechbrain's whole-clip path passes `lengths=None` and counts every frame."""
    pool, reference = _pool(whole_clip=True)
    x = torch.randn(1, 6, 31)
    with torch.no_grad():
        assert torch.allclose(pool(x, lengths=None), reference(x), atol=1e-5)


def test_a_pool_with_wider_attention_kernels_is_refused():
    pool = _ReferencePool()
    pool.tdnn.conv = nn.Conv1d(18, 4, 3, padding=1)
    with pytest.raises(ValueError, match="1x1"):
        A._install_decomposed_pool(pool, conv_in=pool.tdnn.conv, after_conv_in=lambda h: h,
                                   tanh=pool.tanh, conv_out=pool.conv, eps=1e-12, whole_clip=False)


# -- the four exporter changes, through the real compiler --------------------------------------------

def _export(module, n, tmp_path, name="toy"):
    """Trace `module(x [1, 4, n])` with a dynamic last axis and return the main topology's nodes."""
    import coremltools as ct
    from gguf import GGUFReader

    x = torch.randn(1, 4, n)
    traced = torch.jit.trace(module.eval(), (x,))
    program = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=(1, 4, ct.RangeDim(8, 4096)))],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    out = tmp_path / f"{name}.gguf"
    loom_exporter.LoomGGUFBackend()(program, output_path=str(out), architecture=name,
                                    root_axis="n_samples", flat_namespace=True)
    reader = GGUFReader(str(out))
    key = next(k for k in reader.fields if k.startswith("model.graph_topology"))
    return json.loads(reader.fields[key].contents())["nodes"]


def _expressions(nodes):
    found = set()
    for node in nodes:
        for value in (node.get("attrs") or {}).values():
            for item in (value if isinstance(value, list) else [value]):
                if isinstance(item, str) and "n_samples" in item:
                    found.add(item)
    return found


class _MeanThenConcat(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean(dim=2), (x * x).mean(dim=2)], dim=1)


def test_a_mean_that_drops_its_axis_is_reshaped_before_its_consumer(tmp_path):
    """ggml's MEAN keeps the reduced `ne[0]` at size 1. Without the RESHAPE the CONCAT below runs along
    that leftover axis and INTERLEAVES its operands -- the same bytes, so a comparison of either mean
    alone passes, and a wrong answer at the first consumer that cares about layout."""
    nodes = _export(_MeanThenConcat(), 40, tmp_path)
    means = [n for n in nodes if n["op"] == "MEAN"]
    assert len(means) == 2
    for mean in means:
        consumers = [n for n in nodes if mean["outputs"][0] in n["inputs"]]
        assert [c["op"] for c in consumers] == ["RESHAPE"], consumers
        assert consumers[0]["attrs"]["shape"] == ["4", "1"]


class _MeanKept(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=2, keepdim=True)


def test_a_mean_that_keeps_its_axis_is_untouched(tmp_path):
    """The shape every published MEAN has but one (StyleTTS2's, re-exported and bit-identical)."""
    nodes = _export(_MeanKept(), 40, tmp_path)
    mean = next(n for n in nodes if n["op"] == "MEAN")
    assert not any(n["op"] == "RESHAPE" and mean["outputs"][0] in n["inputs"] for n in nodes)


class _UnfoldedNormThenChunk(nn.Module):
    """speechbrain's TDNN order -- conv, ReLU, THEN batch norm -- which coremltools cannot fold into the
    convolution, followed by a Res2Net-style channel split that reads the time axis back out."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 5, stride=2)
        self.norm = nn.BatchNorm1d(8)

    def forward(self, x):
        y = self.norm(torch.relu(self.conv(x)))
        a, b = torch.chunk(y, 2, dim=1)
        return a * b


def test_a_frame_axis_survives_an_unfolded_batch_norm(tmp_path):
    nodes = _export(_UnfoldedNormThenChunk(), 64, tmp_path)
    assert any(n["op"] == "VIEW" for n in nodes), "the chunk should lower to VIEWs"
    exprs = _expressions(nodes)
    # Every expression is the convolution's frame count; a bare `n_samples` is the walk falling back.
    assert exprs and "n_samples" not in exprs, exprs


class _MaxPools(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 3)

    def forward(self, x):
        y = F.max_pool1d(torch.abs(x), 3, 3)
        y = self.conv(y)
        y = F.max_pool1d(y, 2, 2)
        return y.transpose(1, 2)


def test_a_max_pool_lowers_to_pool_1d_with_its_frame_count(tmp_path):
    nodes = _export(_MaxPools(), 99, tmp_path)
    pools = [n for n in nodes if n["op"] == "POOL_1D"]
    assert [(p["attrs"]["op"], p["attrs"]["k0"], p["attrs"]["s0"], p["attrs"]["p0"]) for p in pools] == [
        ("max", 3, 3, 0), ("max", 2, 2, 0)]
    assert "n_samples" not in _expressions(nodes)


class _PaddedPool(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, 3, 3, padding=1)


def test_a_padded_max_pool_is_refused_rather_than_padded_with_zeros(tmp_path):
    """torch pads a max-pool with -inf, ggml's pool with zeros: an all-negative edge window would
    answer 0 instead of its maximum."""
    with pytest.raises(NotImplementedError, match="padding"):
        _export(_PaddedPool(), 30, tmp_path)


# -- the bidirectional sweep -------------------------------------------------------------------------

def test_a_bidirectional_layer_retains_into_its_forward_cell():
    from loom_exporter.driver_components import BiRecurrentCall
    from loom_exporter.driver_ir import (
        DriverIRError, Function, Lit, OutputRef, SubgraphCall, check_subgraph_calls,
    )

    call = BiRecurrentCall(forward_topology="lstm_l0_fwd", backward_topology="lstm_l0_bwd",
                           out_var="gen", sequence=OutputRef("pre"), seq_len=Lit(10), input_dim=4,
                           hidden_dim=3)
    stmts = call.emit(None)
    rendered = stmts[-1].expr.render()
    assert "loom.run_bi_recurrent_and_retain('lstm_l0_fwd', 'lstm_l0_bwd', {from = 'pre'}, 10, 4, 3, " \
           "'rows')" in rendered

    pre = SubgraphCall(outputs=[], module="pre", axes={"n_samples": Lit(10)}, inputs={}, retain=True)

    def head(reads):
        return SubgraphCall(outputs=["probs"], module="head", axes={"n_enc_frames": Lit(10)},
                            inputs={"lstm_out": OutputRef(reads)})

    # Both directions land in the FORWARD cell's store, so that is the module a reader names...
    check_subgraph_calls(Function("infer", ["inputs"], [pre, *stmts, head("lstm_l0_fwd")]), {})
    # ... and naming the backward one is reading a store nothing retained into.
    with pytest.raises(DriverIRError):
        check_subgraph_calls(Function("infer", ["inputs"], [pre, *stmts, head("lstm_l0_bwd")]), {})


# -- recognizers --------------------------------------------------------------------------------------

def _nemo_archive(path, cfg, dot_slash):
    raw = json.dumps(cfg).encode()  # JSON is YAML
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(("./" if dot_slash else "") + "model_config.yaml")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    return path


@pytest.mark.parametrize("dot_slash", [True, False])
def test_the_nemo_recognizers_read_either_member_spelling(tmp_path, dot_slash):
    """The ASR archives name their members `./model_config.yaml`; TitaNet's names it without the
    `./`, which is what made the reader's first export fail."""
    titanet = _nemo_archive(tmp_path / "t.nemo", {
        "target": "nemo.collections.asr.models.label_models.EncDecSpeakerLabelModel",
        "decoder": {"pool_mode": "attention"}}, dot_slash)
    xvector = _nemo_archive(tmp_path / "x.nemo", {
        "target": "nemo.collections.asr.models.label_models.EncDecSpeakerLabelModel",
        "decoder": {"pool_mode": "xvector"}}, dot_slash)
    vad = _nemo_archive(tmp_path / "v.nemo", {
        "target": "nemo.collections.asr.models.classification_models.EncDecFrameClassificationModel"},
        dot_slash)
    assert A._is_titanet(titanet) and not A._is_frame_vad(titanet)
    assert not A._is_titanet(xvector), "a different pool is a different model"
    assert A._is_frame_vad(vad) and not A._is_titanet(vad)


def test_the_speechbrain_and_pyannote_recognizers(tmp_path):
    sb = tmp_path / "sb"
    sb.mkdir()
    (sb / "hyperparams.yaml").write_text(
        "embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN\nclassifier: x\n")
    assert not A._is_speechbrain_ecapa_classifier(sb), "no label encoder, no labels"
    (sb / "label_encoder.txt").write_text("'th: Thai' => 0\n")
    assert A._is_speechbrain_ecapa_classifier(sb)

    pa = tmp_path / "pa"
    pa.mkdir()
    (pa / "pytorch_model.bin").write_bytes(b"")
    (pa / "config.yaml").write_text("model:\n  _target_: pyannote.audio.models.segmentation.PyanNet\n")
    assert A._is_pyannote_segmentation(pa)
    assert A._is_pyannote_segmentation(pa / "pytorch_model.bin")
    assert A._pyannote_checkpoint(pa) == pa / "pytorch_model.bin"
    assert not A._is_pyannote_segmentation(sb)


def test_both_tasks_are_registered():
    from loom_exporter.registry import default_registry

    registry = default_registry()
    names = {(rec.task, rec.name) for entry in registry._entries.values() for rec in entry.recognizers}
    assert {("audio-embedding", "titanet"), ("audio-classification", "marblenet-vad"),
            ("audio-classification", "ecapa-tdnn-lid"),
            ("audio-classification", "pyannote-segmentation")} <= names
