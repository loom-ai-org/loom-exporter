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

Silero VAD's whole-clip rewrite is checked here against its own frame loop on random weights.

The real checkpoints are the gate half: `tests/gate/` would need five downloads, and the oracle numbers
are in loom.cpp's Epic-03.
"""
import io
import json
import tarfile
import types

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
    emb = A.audio_contract(A.AudioOutput.EMBEDDING, 16000, [], None, embedding_dim=192)
    assert emb["task"] == "audio-embedding" and emb["output.kind"] == "embeddings"
    assert emb["output.granularity"] == "clip" and "labels" not in emb
    assert emb["output.embedding_dim"] == 192


def test_a_frame_output_declares_its_rate_and_only_a_nonzero_offset():
    vad = A.audio_contract(A.AudioOutput.FRAME_CLASSES, 16000, ["non_speech", "speech"], 50.0)
    assert vad["output.frame_rate"] == 50.0 and "output.frame_offset" not in vad
    seg = A.audio_contract(A.AudioOutput.FRAME_CLASSES, 16000, ["a"], 59.26, 0.0225)
    assert seg["output.frame_offset"] == 0.0225
    # Floats, not ints: the GGUF writer picks the KV type off the Python type.
    assert isinstance(seg["output.frame_rate"], float)


def test_frame_embeddings_declare_their_rate_and_width_under_the_embedding_task():
    """WakeHuBERT's output: an `embeddings` kind at `frame` granularity. The rate puts a time on each
    row and the width is what the door cuts the flat answer by."""
    feats = A.audio_contract(A.AudioOutput.FRAME_EMBEDDINGS, 16000, [], 50.0, embedding_dim=128)
    assert feats == {"task": "audio-embedding", "input.kind": "audio", "output.kind": "embeddings",
                     "output.granularity": "frame", "sample_rate": 16000, "output.frame_rate": 50.0,
                     "output.embedding_dim": 128}
    # An int, not a float: the GGUF writer picks the KV type off the Python type, and the engine reads
    # an integer key.
    assert isinstance(feats["output.embedding_dim"], int)


def test_frame_embeddings_without_a_width_are_refused():
    """A frame file without its width is one no host can cut into rows -- refused at export, not
    discovered at the first call."""
    with pytest.raises(ValueError, match="embedding_dim"):
        A.audio_contract(A.AudioOutput.FRAME_EMBEDDINGS, 16000, [], 50.0)


def test_a_clip_embedding_without_a_width_stays_valid():
    """TitaNet's and ECAPA's published files predate the key; a clip answer is one row of whatever came
    back, so the key is written when known and not demanded."""
    emb = A.audio_contract(A.AudioOutput.EMBEDDING, 16000, [], None)
    assert "output.embedding_dim" not in emb


def test_the_width_is_read_off_the_traced_output():
    """`_check_output` records the last axis of the real tensor, so the declaration and the graph
    cannot disagree."""
    for output, shape in ((A.AudioOutput.EMBEDDING, (1, 192)), (A.AudioOutput.FRAME_EMBEDDINGS, (1, 7, 128))):
        spec = types.SimpleNamespace(output=output, architecture="toy", labels=[], embedding_dim=None)
        A._check_output(spec, torch.zeros(shape))
        assert spec.embedding_dim == shape[-1]


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


# -- Silero VAD: the whole-clip rewrite ------------------------------------------------------------------

def _silero_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"stft": (258, 1, 256), "conv1.weight": (128, 129, 3), "conv1.bias": (128,),
              "conv2.weight": (64, 128, 3), "conv2.bias": (64,), "conv3.weight": (64, 64, 3),
              "conv3.bias": (64,), "conv4.weight": (128, 64, 3), "conv4.bias": (128,),
              "lstm.weight_ih": (512, 128), "lstm.weight_hh": (512, 128), "lstm.bias_ih": (512,),
              "lstm.bias_hh": (512,), "final.weight": (1, 128, 1), "final.bias": (1,)}
    return {k: (torch.randn(*s, generator=g, dtype=torch.float64) * 0.1) for k, s in shapes.items()}


def _silero_streamed(w, wav):
    """`tinygrad_model.py`'s forward, one frame at a time with the previous frame's 64 samples in front
    and the state carried -- the reference the rewrite must equal."""
    cell = nn.LSTMCell(128, 128).double()
    with torch.no_grad():
        for t in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(cell, t).copy_(w["lstm." + t])
    x_all = F.pad(F.pad(wav, (0, (-wav.numel()) % 512)), (64, 0))
    h = c = torch.zeros(1, 128, dtype=torch.float64)
    out = []
    for i in range(64, x_all.numel(), 512):
        x = F.pad(x_all[i - 64:i + 512].view(1, 1, -1), (0, 64), mode="reflect")
        x = F.conv1d(x, w["stft"], stride=128)
        x = (x[:, :129] ** 2 + x[:, 129:] ** 2).sqrt()
        for k, (s, p) in enumerate(((1, 1), (2, 1), (2, 1), (1, 1))):
            x = F.relu(F.conv1d(x, w[f"conv{k + 1}.weight"], w[f"conv{k + 1}.bias"], stride=s, padding=p))
        h, c = cell(x.squeeze(-1), (h, c))
        out.append(torch.sigmoid(F.conv1d(F.relu(h).unsqueeze(-1), w["final.weight"], w["final.bias"])))
    return torch.cat(out).flatten()


@pytest.mark.parametrize("n", [1, 511, 512, 513, 512 * 6 + 77])
def test_the_whole_clip_silero_is_the_streamed_one(n):
    """The reflect pad folded into the STFT weights, the per-frame convolutions as dense pointwise maps
    and the cell as one `nn.LSTM`: the same function as the frame loop, to f64 rounding, at lengths that
    end mid-frame, on a boundary and one sample past it."""
    w = _silero_weights()
    model = A._SileroWholeClip(w, 256, 128)
    wav = torch.randn(n, generator=torch.Generator().manual_seed(n), dtype=torch.float64)
    with torch.no_grad():
        got = model(wav.unsqueeze(0))[0]
    want = _silero_streamed(w, wav)
    assert got.shape == (A._silero_frames(n), 2) == (want.numel(), 2)
    assert torch.allclose(got[:, 1], want, rtol=0, atol=1e-12)
    assert torch.allclose(got.sum(-1), torch.ones(got.shape[0], dtype=torch.float64))


def test_a_silero_stft_of_another_size_is_refused():
    w = _silero_weights()
    w["stft"] = torch.zeros(130, 1, 128, dtype=torch.float64)
    with pytest.raises(ValueError, match="four-column"):
        A._SileroWholeClip(w, 128, 64)


def test_the_silero_driver_counts_whole_frames():
    """`ceil(n / 512)` frames, as `(n - 1) // 512 + 1`, read by the sweep and the head alike."""
    from loom_exporter.driver_components import RecurrentCall, SubgraphCallComponent

    spec = A.SileroVadExportConfig(checkpoint="x", output_path="x.gguf")
    components = spec.driver_components()
    sweep = next(c for c in components if isinstance(c, RecurrentCall))
    head = [c for c in components if isinstance(c, SubgraphCallComponent)][-1]
    assert sweep.topology == "lstm_l0_fwd" and not sweep.reverse and sweep.retain
    assert sweep.seq_len == head.length
    assert sweep.seq_len.render() == "(math.floor((#waveform - 1) / 512) + 1)"
    contract = A.audio_contract(A.AudioOutput.FRAME_CLASSES, 16000, A._frame_vad_labels(["0", "1"]),
                                16000 / 512)
    assert contract["output.frame_rate"] == 31.25 and contract["labels"] == ["non_speech", "speech"]


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

    sv = tmp_path / "sv"
    sv.mkdir()
    assert not A._is_silero_vad(sv)
    (sv / "silero_vad.jit").write_bytes(b"")
    assert A._is_silero_vad(sv) and A._silero_checkpoint(sv) == sv / "silero_vad.jit"
    assert A._is_silero_vad(sv / "silero_vad.jit")
    assert not A._is_silero_vad(pa / "pytorch_model.bin")


# A WakeHuBERT-shaped checkpoint directory: `config.json` naming the module, loader and weights, and a
# `student.py` with upstream's attribute names (`mel.hop`, `stem.conv.stride`, `mel.basis`) at toy
# widths -- 32 channels, so a Q8_0 block fits the 1x1 projection's rows.
_TOY_STUDENT = """
import json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000


class CausalLogMel(nn.Module):
    def __init__(self, n_mels=8, n_fft=64, hop=32):
        super().__init__()
        k, f = torch.arange(n_fft), torch.arange(n_fft // 2 + 1)
        ang = 2 * math.pi * f[:, None] * k[None, :] / n_fft
        self.register_buffer("basis", torch.cat([torch.cos(ang), -torch.sin(ang)]).unsqueeze(1))
        self.register_buffer("mel", torch.rand(n_mels, n_fft // 2 + 1))
        self.pad, self.hop, self.n_freq = n_fft - hop, hop, n_fft // 2 + 1

    def forward(self, wav):
        spec = F.conv1d(F.pad(wav.unsqueeze(1), (self.pad, 0)), self.basis, stride=self.hop)
        power = spec[:, :self.n_freq].pow(2) + spec[:, self.n_freq:].pow(2)
        return torch.log(torch.matmul(self.mel, power) + 1e-6)


class CausalConv1d(nn.Module):
    def __init__(self, i, o, k, s):
        super().__init__()
        self.pad = k - s
        self.conv = nn.Conv1d(i, o, k, s, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = CausalLogMel()
        self.stem = CausalConv1d(8, 32, 4, 2)
        self.stem_bn = nn.BatchNorm1d(32)
        self.out = nn.Conv1d(32, 16, 1)

    def forward(self, wav):
        return self.out(F.relu(self.stem_bn(self.stem(self.mel(wav))))).transpose(1, 2)


def load_toy(path, config=None):
    torch.manual_seed(0)
    model = Toy()
    model.load_state_dict(torch.load(path))
    return model.eval()
"""


def _toy_wakehubert(tmp_path):
    d = tmp_path / "wakehubert"
    d.mkdir()
    (d / "student.py").write_text(_TOY_STUDENT)
    namespace = {}
    exec(compile(_TOY_STUDENT, "toy_student", "exec"), namespace)
    torch.manual_seed(0)
    torch.save(namespace["Toy"]().state_dict(), d / "weights.pt")
    (d / "config.json").write_text(json.dumps({
        "family": "WakeHuBERT", "input": {"sample_rate": 16000}, "output": {"hop_samples": 64},
        "pytorch": {"module": "student.py", "loader": "load_toy", "weights": "weights.pt"}}))
    return d


def test_the_wakehubert_recognizer_reads_the_config_family(tmp_path):
    d = _toy_wakehubert(tmp_path)
    assert A._is_wakehubert(d) and A._is_wakehubert(d / "config.json")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "config.json").write_text(json.dumps({"family": "Something"}))
    assert not A._is_wakehubert(tmp_path / "other")
    (d / "weights.pt").unlink()
    assert not A._is_wakehubert(d), "a config whose weights are missing is not a checkpoint"


def test_a_wakehubert_export_declares_frames_and_keeps_its_front_end_float(tmp_path):
    """The toy through the real compiler at F16 -- the precision that would otherwise pack the DFT
    basis (block size 1 aligns everything). The basis stays F32, the projection is packed, and the
    contract states one frame per hop."""
    from gguf import GGUFReader

    from loom_exporter.main_export import main_export

    out = tmp_path / "toy.gguf"
    main_export(str(_toy_wakehubert(tmp_path)), str(out), quantize="F16")
    reader = GGUFReader(str(out))
    types = {t.name: t.tensor_type.name for t in reader.tensors}
    assert types["model_mel_basis"] == "F32" and types["model_out_weight"] == "F16"

    def kv(key):
        field = reader.fields[key]
        value = field.parts[field.data[0]]
        return bytes(value).decode() if field.types[0].name == "STRING" else value.tolist()[0]

    assert kv("loom.output.kind") == "embeddings" and kv("loom.output.granularity") == "frame"
    assert kv("loom.output.frame_rate") == 16000 / 64
    # The toy's head is 16 wide: read off the traced output, and an INTEGER key the engine cuts by.
    assert kv("loom.output.embedding_dim") == 16
    assert reader.fields["loom.output.embedding_dim"].types[0].name in ("INT32", "UINT32")


def test_both_tasks_are_registered():
    from loom_exporter.registry import default_registry

    registry = default_registry()
    names = {(rec.task, rec.name) for entry in registry._entries.values() for rec in entry.recognizers}
    assert {("audio-embedding", "titanet"), ("audio-classification", "marblenet-vad"),
            ("audio-classification", "ecapa-tdnn-lid"),
            ("audio-classification", "pyannote-segmentation"),
            ("audio-classification", "silero-vad"), ("audio-embedding", "wakehubert")} <= names
