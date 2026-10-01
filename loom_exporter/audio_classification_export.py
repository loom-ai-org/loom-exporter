"""Small audio classifiers and embedders -- EXPORT-ROADMAP.md's family 13 (P5): a waveform in, and out of
it either one vector per clip (a speaker embedding), one class distribution per clip (language id), or
one class distribution per FRAME (voice activity, speaker segmentation).

**The contract is the family's real work, not the graphs.** Every leaf is one forward pass, and the
first two are a NeMo `ConvASREncoder` -- Citrinet's encoder, already exported -- under a different head.
What did not exist was any way for a file to say what its output MEANS: `class` meant "one per input
token" because family 12 was the only classifier, and a VAD's per-frame probabilities, a language id's
one row and a speaker embedding's vector are three different answers a host must read three different
ways. loom.cpp ADR-062 decides the names once for the family:

* `loom.output.kind` is `class` or `embeddings`, as before;
* `loom.output.granularity` says how many answers there are: `token` (family 12's, and what an absent
  key means, so no published file moves), `frame` or `clip`;
* `loom.output.frame_rate` is the frames per second of a `frame` output, so a host can put a time on
  each row without knowing the encoder's stride;
* `loom.labels` names the classes, exactly as family 12 already declares them.

**What the driver returns is the model's own distribution, not a decision.** A token classifier's
driver argmaxes because its door answers "which label", and the score is rarely wanted. A VAD's whole
point is the probability -- every consumer thresholds, smooths and hangs over it in its own way (the
checkpoint's own card says so) -- and a language id is routinely read top-k. So `frame` and `clip`
outputs are softmax probabilities, row-major `[n_rows, n_labels]`, and the decision is the host's.

Per-loader halves live below their template, the way `nemo_asr_export` holds family 1's.
"""
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase
from .nemo_asr_export import (
    _is_nemo_archive, _read_nemo_model_config, build_trace, prepare_conv_asr_encoder_for_trace,
    prepare_nemo_environment,
)
from .spec_protocol import Axis, Unchecked


class AudioOutput(Enum):
    """What one forward pass hands back, as the contract names it: `(output.kind, output.granularity)`.

    A closed set rather than two free strings, so a leaf cannot declare a pair no host reads -- an
    `embeddings` output per FRAME is a real thing (a frame-level encoder) and is not one of these until
    a leaf needs it and a door answers it.
    """

    # One fixed-width vector per clip: TitaNet, ECAPA speaker models.
    EMBEDDING = ("embeddings", "clip")
    # One probability row per encoder frame: VAD, segmentation.
    FRAME_CLASSES = ("class", "frame")
    # One probability row for the whole clip: language id.
    CLIP_CLASSES = ("class", "clip")

    @property
    def kind(self) -> str:
        return self.value[0]

    @property
    def granularity(self) -> str:
        return self.value[1]

    @property
    def task(self) -> str:
        return "audio-embedding" if self is AudioOutput.EMBEDDING else "audio-classification"


@dataclass(kw_only=True)
class AudioClassificationExportConfig(LoomExportConfig):
    """Family 13's `LoomExportConfig`: one traced graph over `(waveform, length)`, whose one output is
    the answer.

    The loader is the subclass's (`load_model`, `encoder_wrapper`); everything a HOST reads -- the
    output pair, the labels, the frame rate, the sample rate -- is collected here, read off the
    checkpoint by the loader during `build_trace` rather than declared by whoever builds the config.
    """

    checkpoint: str
    output: AudioOutput
    decomposition: Decomposition = field(default_factory=Flattened)
    # Raw audio samples, like every family-1 encoder this reuses the trace of.
    root_axis: str = "n_samples"

    # Read off the checkpoint during `build_trace`, never declared.
    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    labels: List[str] = field(default_factory=list, init=False, repr=False)
    frame_rate: Optional[float] = field(default=None, init=False, repr=False)
    frame_offset: float = field(default=0.0, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "checkpoint": Unchecked(
            "path to the checkpoint. Established by the recognizer's own detect(), which reads the "
            "archive's config rather than trusting the extension, and the loader raises on anything "
            "it cannot restore."
        ),
        "output": Unchecked(
            "chosen by the recognizer from the checkpoint's own restore target, and checked against "
            "what the traced forward returned by the loader's wrapper (`_check_output`), during "
            "tracing -- the only moment the real output exists."
        ),
        "sample_rate": Unchecked("READ off the checkpoint's preprocessor config during build_trace"),
        "labels": Unchecked("READ off the checkpoint during build_trace; the class count is checked "
                            "against the traced output's own width there"),
        "frame_rate": Unchecked("DERIVED from the preprocessor hop and the encoder's total stride "
                                "during build_trace, and checked against the frame count the traced "
                                "forward actually produced"),
        "frame_offset": Unchecked("DERIVED with frame_rate; 0 for every leaf whose frame i starts at "
                                  "i / frame_rate"),
    }

    def build_trace(self, model):
        return build_trace(self, model, self.sample_rate)

    def contract(self) -> dict:
        return audio_contract(self.output, self.sample_rate, self.labels, self.frame_rate,
                              self.frame_offset)

    def synthesized_builder_key(self) -> str:
        """Nothing to reduce: the tensor IS the answer -- the codec decoder's builder, under the name
        that says what it does for this family. Read by the exporter (through `backend_kwargs`) and by
        `component_registry.usage()`, which would otherwise credit these leaves with the causal LM's
        `argmax_epilogue` because they are `Flattened`."""
        return "ReturnOutput"

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=True, root_axis=self.root_axis,
                    driver_builder=self.synthesized_builder_key(), hparams=self.hparams())


def audio_contract(output: AudioOutput, sample_rate: Optional[int], labels: List[str],
                   frame_rate: Optional[float], frame_offset: float = 0.0) -> dict:
    """The task, the modality pair, and the facts a host needs to read the output (loom.cpp ADR-062).

    `output.granularity` is written for every leaf, including the `clip` ones a host could infer from a
    one-row answer: inferring it from the answer's SHAPE is what made `class` mean "per token" by
    accident, and a one-frame VAD answer is one row too.

    `output.frame_offset` is written only when it is not zero. Frame `i` of a `frame` output covers
    `[offset + i / rate, offset + (i + 1) / rate)` seconds: a strided encoder that pads its input
    (MarbleNet) starts at 0, and one that does not (pyannote's SincNet, whose first frame needs 991
    samples of context) starts later.
    """
    contract = {
        "task": output.task,
        "input.kind": "audio",
        "output.kind": output.kind,
        "output.granularity": output.granularity,
    }
    if sample_rate is not None:
        contract["sample_rate"] = int(sample_rate)
    if output is AudioOutput.FRAME_CLASSES and frame_rate is not None:
        contract["output.frame_rate"] = float(frame_rate)
        if frame_offset:
            contract["output.frame_offset"] = float(frame_offset)
    if labels:
        contract["labels"] = list(labels)
    return contract


def _check_output(spec: AudioClassificationExportConfig, out: torch.Tensor) -> None:
    """The traced output against what the config says it is -- run inside the wrapper's forward, during
    the trace, because that is the one moment the real tensor exists (`EncoderOutput.validate`'s
    argument). A head that is not the shape the contract will declare is a file whose labels or frame
    rate lie, and nothing downstream could tell."""
    if spec.output is AudioOutput.EMBEDDING:
        if out.dim() != 2 or out.shape[0] != 1:
            raise ValueError(f"{spec.architecture}: an embedding is one [1, D] row per clip; the "
                             f"traced forward returned {tuple(out.shape)}.")
        return
    want_rank = 3 if spec.output is AudioOutput.FRAME_CLASSES else 2
    if out.dim() != want_rank or out.shape[0] != 1:
        raise ValueError(f"{spec.architecture}: a {spec.output.granularity} class output is rank "
                         f"{want_rank} with batch 1; the traced forward returned {tuple(out.shape)}.")
    if spec.labels and int(out.shape[-1]) != len(spec.labels):
        raise ValueError(f"{spec.architecture}: the checkpoint names {len(spec.labels)} labels and its "
                         f"head emits {int(out.shape[-1])} classes.")


# -- NeMo: TitaNet (speaker embedding) and frame-VAD MarbleNet ----------------------------------------
#
# Both restore through NeMo's own model classes and both encoders are `ConvASREncoder`, so
# `prepare_conv_asr_encoder_for_trace` -- written for Citrinet -- makes them trace to one graph
# unchanged. What differs is the head after it.

_SPEAKER_TARGET = "EncDecSpeakerLabelModel"
_FRAME_TARGET = "EncDecFrameClassificationModel"

# The attentive pool's `get_statistics_with_mask` clamps the variance here (NeMo's default eps).
_POOL_VARIANCE_EPS = 1e-10
# What a masked frame's attention score becomes before the softmax. NeMo fills `-inf`; any value that
# `exp` takes to exactly 0 in f32 after the row max is subtracted is the same softmax, and a finite one
# keeps an `inf - inf` NaN out of a row that is all padding.
_MASKED_SCORE = -1e30


@dataclass(kw_only=True)
class NemoAudioClassificationExportConfig(AudioClassificationExportConfig):
    """TitaNet and frame-VAD MarbleNet: a `.nemo` archive whose `target` is a NeMo label model."""

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        import nemo.collections.asr as nemo_asr

        print(f"Loading NeMo model from {self.checkpoint}...")
        model = _restore_label_model(nemo_asr, self.checkpoint)
        model.eval()
        if not isinstance(model.encoder, nemo_asr.modules.ConvASREncoder):
            raise ValueError(f"{self.architecture}: expected a ConvASREncoder, got "
                             f"{type(model.encoder).__name__}; only that encoder's trace preparation "
                             f"exists for this family.")
        prepare_conv_asr_encoder_for_trace(model)
        if self.output is AudioOutput.EMBEDDING:
            _prepare_attentive_pool(model.decoder)

        cfg = model.cfg
        self.sample_rate = int(cfg.preprocessor.sample_rate)
        if self.output is AudioOutput.FRAME_CLASSES:
            if getattr(model, "crop_or_pad", None) is not None:
                raise ValueError(f"{self.architecture}: the checkpoint crops or pads every mel to a "
                                 f"fixed length (`crop_or_pad_augment`), which a dynamic-length graph "
                                 f"does not reproduce.")
            self.labels = _frame_vad_labels(list(cfg.get("labels") or []))
            hop = float(cfg.preprocessor.window_stride)
            self.frame_rate = 1.0 / (hop * _encoder_stride(cfg.encoder))
        return model

    def encoder_wrapper(self, model):
        if self.output is AudioOutput.EMBEDDING:
            return _NemoSpeakerEmbedding(model, self)
        return _NemoFrameClassifier(model, self)


def _restore_label_model(nemo_asr, checkpoint: str):
    """`ASRModel.restore_from`, tolerating exactly one missing key: the frame classifier's class-weighted
    loss.

    `EncDecFrameClassificationModel` builds `self.loss` with a `weight` buffer that the published VAD
    archive does not carry (it was saved before the buffer existed), so a strict restore refuses it.
    Non-strict on its own would also accept a checkpoint missing real weights, so the archive's own keys
    are compared against the model's and anything but `loss.*` still raises.
    """
    try:
        return nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu")
    except RuntimeError as exc:
        if "Missing key(s)" not in str(exc):
            raise
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu", strict=False)
    missing = sorted(set(model.state_dict()) - set(_archive_state_dict_keys(checkpoint)))
    unexplained = [key for key in missing if not key.startswith("loss.")]
    if unexplained:
        raise ValueError(f"{checkpoint}: the archive is missing weights {unexplained[:8]}; only the "
                         f"training loss's own buffers may be absent.")
    return model


def _archive_state_dict_keys(checkpoint: str) -> List[str]:
    import io
    import tarfile

    with tarfile.open(checkpoint) as archive:
        name = next(n for n in archive.getnames() if n.endswith("model_weights.ckpt"))
        state = torch.load(io.BytesIO(archive.extractfile(name).read()), map_location="cpu")
    return list(state)


def _encoder_stride(encoder_cfg) -> int:
    """The `ConvASREncoder`'s total time stride: the product of every block's stride, each repeated
    block striding only once (NeMo applies a block's stride on its first sub-layer)."""
    stride = 1
    for block in encoder_cfg.jasper:
        stride *= int(list(block.get("stride", [1]))[0])
    return stride


def _frame_vad_labels(labels: List[str]) -> List[str]:
    """NeMo's frame VAD names its two classes `'0'` and `'1'` -- the training targets, not names. The
    contract's labels are what a host prints, so the binary VAD pair is spelled out; anything else is
    passed through as the checkpoint wrote it."""
    if labels == ["0", "1"]:
        return ["non_speech", "speech"]
    return labels


class _NemoSpeakerEmbedding(nn.Module):
    """`EncDecSpeakerLabelModel.forward`, returning the embedding and not the training classifier's
    logits (`SpeakerDecoder` returns both; the logits are over the 16681 TRAINING speakers and mean
    nothing at inference)."""

    def __init__(self, model, spec):
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, waveform, length):
        out = self.model(input_signal=waveform, input_signal_length=length)[1]
        _check_output(self.spec, out)
        return out


class _NemoFrameClassifier(nn.Module):
    """`EncDecFrameClassificationModel.forward` with two changes: the frames are cut to the encoder's own
    `encoded_len`, and the logits go through NeMo's VAD softmax.

    The cut is Retro-068's, on a different head: the mel front end counts `floor(n / hop)` valid frames
    out of `floor(n / hop) + 1`, the stride-2 stem rounds the two apart again, and the last frame is
    computed from masked context and covers no audio. NeMo's forward returns it and `vad_utils` reads
    it; a host putting `frame_rate` times on the rows would put that one past the end of the clip.
    """

    def __init__(self, model, spec):
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, waveform, length):
        mel, mel_len = self.model.preprocessor(input_signal=waveform, length=length)
        encoded, encoded_len = self.model.encoder(audio_signal=mel, length=mel_len)
        out = F.softmax(self.model.decoder(encoded.transpose(1, 2))[:, :encoded_len[0]], dim=-1)
        _check_output(self.spec, out)
        return out


def _prepare_attentive_pool(decoder) -> None:
    """Replaces `SpeakerDecoder`'s attentive statistics pool with `_DecomposedAttentivePool`, the same
    arithmetic in a form that traces to one dynamic-length graph. NeMo's `TDNNModule` is
    convolution -> activation -> batch norm."""
    pool = decoder._pooling
    tdnn, tanh, conv_out = pool.attention_layer
    _install_decomposed_pool(pool, conv_in=tdnn.conv_layer,
                             after_conv_in=lambda h: tdnn.bn(tdnn.activation(h)),
                             tanh=tanh, conv_out=conv_out, eps=_POOL_VARIANCE_EPS,
                             whole_clip=False)


def _install_decomposed_pool(pool, *, conv_in, after_conv_in, tanh, conv_out, eps, whole_clip):
    """Swaps an ECAPA-style attentive statistics pool's `forward` for the same arithmetic in a form that
    traces to one dynamic-length graph, and checks on a probe that it is the same pool.

    Both loaders' pools (NeMo's `AttentivePoolLayer`, speechbrain's `AttentiveStatisticsPooling`) are
    one design with the same two problems, and neither problem is the arithmetic:

    1. **The global statistics are TILED over time** (`mean.unsqueeze(2).repeat(1, 1, L)`) and
       concatenated under the frames, so the attention's first 1x1 convolution sees `[x; mean; std]` --
       three times the encoder's channels, per frame. A live `.repeat()` by a dynamic count is the op
       Retro-065 found lowering as an identity. A 1x1 convolution over a concatenation is a sum of three
       convolutions, and two of the three see a time-constant input, so the same output is
       `conv(x) + W_mean @ mean + W_std @ std` broadcast over time: no repeat, and a third of the work.
    2. **The masked frames are filled with `-inf`** before the softmax. Kept as a mask, filled with a
       finite score `exp` underflows to zero (see `_MASKED_SCORE`).

    The mask itself stays LIVE. NeMo's mel counts `floor(n / hop)` of its `floor(n / hop) + 1` frames
    as real, so even an unpadded clip has a masked frame there, and on a real clip dropping the mask
    moves TitaNet's embedding by more than its own largest component (measured: 0.143 against an absmax
    of 0.077). speechbrain's whole-clip path (`whole_clip`) passes no length at all and counts
    every frame, which the probe below checks too.
    """
    if conv_in.kernel_size != (1,) or conv_out.kernel_size != (1,):
        raise ValueError("the attentive pool's decomposition into a frame term and a time-constant term "
                         "holds for 1x1 attention convolutions only; this pool's are "
                         f"{conv_in.kernel_size} and {conv_out.kernel_size}.")
    channels = conv_in.in_channels // 3
    w_x, w_mean, w_std = conv_in.weight.detach().split(channels, dim=1)
    pool._w_x = w_x.clone()
    pool._w_stats = torch.cat([w_mean, w_std], dim=1)[..., 0].clone()  # [A, 2C]
    pool._b = conv_in.bias.detach().clone()

    def forward(self, x, length=None, lengths=None):
        # NeMo passes `length`, absolute frames; speechbrain's whole-clip path passes neither, and
        # every frame is real (see `_SpeechbrainClassifier` for why it carries no length).
        n_valid = length if length is not None else lengths
        if n_valid is None:
            # Plain means over the frame axis, which is `ne[0]` here -- the one axis a run-time mean
            # lowers on.
            mean = x.mean(dim=2)
            std = torch.sqrt((x - mean.unsqueeze(2)).pow(2).mean(dim=2).clamp(min=eps))
            valid = None
        else:
            valid = (torch.arange(x.shape[2], device=x.device).unsqueeze(0) < n_valid.unsqueeze(1))
            valid = valid.unsqueeze(1).to(x.dtype)
            mean, std = _masked_statistics(x, valid / valid.sum(dim=2, keepdim=True), eps)
        stats = torch.matmul(self._w_stats, torch.cat([mean, std], dim=1).unsqueeze(2))
        hidden = F.conv1d(x, self._w_x, self._b) + stats
        scores = conv_out(tanh(after_conv_in(hidden)))
        if valid is not None:
            scores = scores * valid + (1.0 - valid) * _MASKED_SCORE
        mu, sigma = _masked_statistics(x, F.softmax(scores, dim=2), eps)
        return torch.cat((mu, sigma), dim=1).unsqueeze(2)

    probe = torch.randn(1, channels, 37, generator=torch.Generator().manual_seed(0))
    probe_len = None if whole_clip else torch.tensor([31])
    with torch.no_grad():
        want = pool(probe, probe_len)
        pool.forward = types.MethodType(forward, pool)
        got = pool(probe, probe_len)
    deviation = (want - got).abs().max().item()
    if deviation > 1e-5 * max(want.abs().max().item(), 1.0):
        raise ValueError(f"the rewritten attentive pool differs from the original by {deviation}. It "
                         f"exists to be the same pool, so this is a defect in it.")


def _masked_statistics(x, weights, eps):
    """`get_statistics_with_mask` / `_compute_statistics`: a weighted mean over time and the matching
    standard deviation, the variance clamped at `eps`."""
    mean = torch.sum(weights * x, dim=2)
    var = torch.sum(weights * (x - mean.unsqueeze(2)).pow(2), dim=2)
    return mean, torch.sqrt(var.clamp(min=eps))


# -- speechbrain: ECAPA-TDNN language id ---------------------------------------------------------------
#
# `speechbrain/lang-id-voxlingua107-ecapa`: speechbrain's own Fbank front end, a sentence-mean
# normalisation, ECAPA-TDNN, and an x-vector classifier ending in a log-softmax over 107 languages.
# Loaded through speechbrain's `EncoderClassifier`, so the checkpoint's own hyperparams.yaml builds it.


@dataclass(kw_only=True)
class SpeechbrainAudioClassificationExportConfig(AudioClassificationExportConfig):
    """A speechbrain `EncoderClassifier` directory: `hyperparams.yaml` + `label_encoder.txt`."""

    def load_model(self):
        import tempfile

        from speechbrain.inference.classifiers import EncoderClassifier

        print(f"Loading speechbrain EncoderClassifier from {self.checkpoint}...")
        # `from_hparams` links the checkpoint's files into `savedir`; a scratch directory keeps that
        # out of the checkpoint (which may sit on a read-only or nearly full drive).
        savedir = tempfile.mkdtemp(prefix="loom_speechbrain_")
        model = EncoderClassifier.from_hparams(source=self.checkpoint, savedir=savedir,
                                               run_opts={"device": "cpu"})
        model.eval()
        mods = model.mods
        features = mods.compute_features
        if features.deltas or features.context:
            raise ValueError(f"{self.architecture}: the Fbank front end appends deltas or a context "
                             f"window, which this loader does not reproduce.")
        norm = mods.mean_var_norm
        if norm.norm_type != "sentence" or norm.std_norm:
            raise ValueError(f"{self.architecture}: only a sentence-MEAN input normalisation is "
                             f"reproduced (this one is norm_type={norm.norm_type!r}, "
                             f"std_norm={norm.std_norm}).")
        _freeze_filterbank(features.compute_fbanks)
        asp = mods.embedding_model.asp
        _install_decomposed_pool(asp, conv_in=asp.tdnn.conv.conv,
                                 after_conv_in=lambda h: asp.tdnn.norm(asp.tdnn.activation(h)),
                                 tanh=asp.tanh, conv_out=asp.conv.conv, eps=asp.eps,
                                 whole_clip=True)
        self.sample_rate = int(features.compute_STFT.sample_rate)
        self.labels = _speechbrain_labels(model.hparams.label_encoder)
        return model

    def encoder_wrapper(self, model):
        return _SpeechbrainClassifier(model, self)

    def build_trace(self, model):
        """The waveform alone (see `_SpeechbrainClassifier`), over the same dynamic sample axis and trace
        length as every family-1 encoder."""
        import coremltools as ct
        import numpy as np

        from .nemo_asr_export import MAX_SECONDS, MIN_SECONDS, TRACE_SECONDS

        n_samples = int(TRACE_SECONDS * self.sample_rate)
        seq_dim = ct.RangeDim(int(MIN_SECONDS * self.sample_rate), int(MAX_SECONDS * self.sample_rate))
        mil_inputs = [ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32)]
        return self.encoder_wrapper(model), (torch.randn(1, n_samples),), mil_inputs


def _freeze_filterbank(filterbank) -> None:
    """speechbrain's `Filterbank` rebuilds its triangular mel matrix from `f_central`/`band` on every
    forward -- trainable filters, frozen in this checkpoint (`freeze=True`, no random jitter in eval).
    Built once here and multiplied as a constant, so the graph carries a `[n_freq, n_mels]` weight
    rather than the arithmetic that would rebuild it per call."""
    if not filterbank.freeze:
        raise ValueError("_freeze_filterbank: this Filterbank's filters are trainable "
                         "(`freeze=False`); only a frozen bank is a constant.")
    with torch.no_grad():
        f_central = filterbank.f_central.repeat(filterbank.all_freqs_mat.shape[1], 1).transpose(0, 1)
        band = filterbank.band.repeat(filterbank.all_freqs_mat.shape[1], 1).transpose(0, 1)
        matrix = filterbank._create_fbank_matrix(f_central, band).detach().clone()
    filterbank._loom_matrix = matrix

    def forward(self, spectrogram):
        fbanks = torch.matmul(spectrogram, self._loom_matrix)
        if not self.log_mel:
            return fbanks
        # `_amplitude_to_DB`, with its per-sequence `amax(dim=(-2, -1))` spelled as the global max:
        # the same number for the one clip this graph is traced over, and the only maximum the
        # exporter composes (a per-axis `reduce_max` has no ggml reduction).
        x_db = self.multiplier * torch.log10(torch.clamp(fbanks, min=self.amin))
        x_db = x_db - self.multiplier * self.db_multiplier
        return torch.max(x_db, x_db.max() - self.top_db)

    filterbank.forward = types.MethodType(forward, filterbank)


def _speechbrain_labels(encoder) -> List[str]:
    """The `CategoricalEncoder`'s labels by index, as the checkpoint writes them ('th: Thai')."""
    count = len(encoder)
    return [str(encoder.ind2lab[i]) for i in range(count)]


class _SpeechbrainClassifier(nn.Module):
    """`EncoderClassifier.classify_batch` for ONE WHOLE clip, returning probabilities.

    **The waveform is the only input, and nothing is masked**, which is `classify_batch`'s own default:
    with no `wav_lens` it counts every sample as real, and its blocks take `lengths=None` -- the
    squeeze-excite blocks then average with a plain `mean`, and the pool over every frame. Two ways of
    carrying a length were tried and neither is expressible today:

    * a `length` input, compared as `arange(n) < length`, is what the exporter's length-mask fold
      proves all-true and deletes -- leaving a topology with no `length` and a driver still binding one;
    * a RELATIVE length times a frame count (`lengths * x.shape[-1]`, speechbrain's own masks) traces to
      the shape READ AS DATA -- `SHAPE`, `GET_ROWS`, arithmetic -- and the engine's `SHAPE` is a
      four-element `ne` vector that `GET_ROWS` hands back whole, so those exports aborted dividing
      `[1]` by `[4]` and reshaping four elements into one.

    A padded batch is a different contract from this door's anyway: the host passes one clip.
    """

    def __init__(self, model, spec):
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, waveform):
        mods = self.model.mods
        feats = mods.compute_features(waveform)
        # `InputNormalization` (sentence, mean only) over every frame, as a sum over a count. A mean over
        # the frame axis does not lower (a run-time count is `ggml_mean`'s on `ne[0]` only, and the
        # transpose that would put frames there is folded back into the reduction), and the count
        # cannot be `feats.shape[1]` (see above). `feats * 0 + 1` is a ones tensor built by arithmetic;
        # the features are clamped dB values, so it is never `nan`.
        ones = feats[:, :, :1] * 0.0 + 1.0
        feats = feats - feats.sum(dim=1, keepdim=True) / ones.sum(dim=1, keepdim=True)
        embedding = mods.embedding_model(feats, None)
        out = torch.exp(mods.classifier(embedding).squeeze(1))
        _check_output(self.spec, out)
        return out


# -- pyannote: segmentation-3.0 (powerset speaker activity per frame) ----------------------------------
#
# SincNet over the raw waveform, a four-layer BiLSTM, two linear layers and a log-softmax over seven
# POWERSET classes -- no speaker, each of three local speakers, each pair. A recurrence is not a
# topology, so this is EnCodec's shape: a graph before the LSTM, the LSTM as cell topologies swept in
# C++, a graph after it.

#: pyannote's powerset of three local speakers, at most two at once, in `Powerset(3, 2).mapping`'s
#: order -- checked against the checkpoint's own mapping at load, not trusted.
_POWERSET_3_2 = ("non_speech", "speaker1", "speaker2", "speaker3",
                 "speaker1+speaker2", "speaker1+speaker3", "speaker2+speaker3")


@dataclass(kw_only=True)
class PyannoteSegmentationExportConfig(BaseMultiPhaseModelExportConfig):
    """pyannote's `PyanNet` segmentation model: `pytorch_model.bin` beside a `config.yaml`."""

    checkpoint: str
    architecture: Optional[str] = "pyannote-segmentation"
    driver_script_path: Path = Path(__file__).resolve().parent
    output: AudioOutput = AudioOutput.FRAME_CLASSES
    # Seconds the SincNet trace runs at, and the RangeDim ceiling. The checkpoint is TRAINED on 10 s
    # chunks and pyannote's own pipeline slides that window; the graph is length-agnostic and the LSTM a
    # loop, so the ceiling is a declaration, not a limit in the model.
    trace_seconds: float = 2.0
    max_seconds: float = 600.0

    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    labels: List[str] = field(default_factory=list, init=False, repr=False)
    frame_rate: Optional[float] = field(default=None, init=False, repr=False)
    frame_offset: float = field(default=0.0, init=False, repr=False)
    _model: Optional[object] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "checkpoint": Unchecked("path to `pytorch_model.bin`; the recognizer read its `config.yaml` "
                                "and pyannote's loader raises on anything it cannot load"),
        "architecture": Unchecked("the name the engine reads back"),
        "output": Unchecked("fixed: this model's head is a per-frame powerset distribution"),
        "trace_seconds": Unchecked("the concrete length torch.jit.trace runs at; the dynamic range is "
                                   "declared separately"),
        "max_seconds": Unchecked("the ct.RangeDim upper bound -- see the field's comment"),
        "sample_rate": Unchecked("READ off the checkpoint's hyperparameters at load"),
        "labels": Unchecked("the powerset names, checked against the checkpoint's own mapping"),
        "frame_rate": Unchecked("READ off the model's own receptive field (`step`)"),
        "frame_offset": Unchecked("READ off the model's own receptive field: the first frame's "
                                  "centre less half a step"),
        "_model": Unchecked("the loaded model, cached so phases() can build wrappers around it"),
    }

    def load_model(self):
        from pyannote.audio import Model
        from pyannote.audio.utils.powerset import Powerset

        print(f"Loading pyannote model from {self.checkpoint}...")
        model = Model.from_pretrained(self.checkpoint).eval()
        spec = model.specifications
        if spec.powerset_max_classes is None:
            raise ValueError(f"{self.architecture}: a multilabel (non-powerset) head is a different "
                             f"output -- independent sigmoids, not one distribution per frame.")
        mapping = Powerset(len(spec.classes), spec.powerset_max_classes).mapping
        if (len(spec.classes), spec.powerset_max_classes) != (3, 2):
            raise ValueError(f"{self.architecture}: only the 3-speaker/2-at-once powerset is named here, "
                             f"this checkpoint is {len(spec.classes)}/{spec.powerset_max_classes}.")
        expected = torch.tensor([[1.0 if f"speaker{k + 1}" in name.split("+") else 0.0 for k in range(3)]
                                 for name in _POWERSET_3_2])
        if not torch.equal(mapping.float(), expected):
            raise ValueError(f"{self.architecture}: pyannote's powerset mapping is not in the order the "
                             f"labels name it ({mapping.tolist()}).")
        if not model.hparams.lstm.get("monolithic", True):
            raise ValueError(f"{self.architecture}: a per-layer LSTM list traces to separate modules; "
                             f"only the monolithic nn.LSTM is wired here.")
        _freeze_sinc_filters(model.sincnet)
        field_ = model.receptive_field
        self.sample_rate = int(model.hparams.sample_rate)
        self.labels = list(_POWERSET_3_2)
        self.frame_rate = 1.0 / float(field_.step)
        self.frame_offset = float(field_.start) + float(field_.duration) / 2.0 - float(field_.step) / 2.0
        self._model = model
        return model

    def export_architecture(self) -> str:
        return self.architecture

    def phases(self):
        import coremltools as ct
        import numpy as np

        model = self._model if self._model is not None else self.load_model()
        lstm = model.lstm
        n_samples = int(self.trace_seconds * self.sample_rate)
        frames = int(model.num_frames(n_samples))
        width = 2 * lstm.hidden_size
        return [
            ExportPhase(
                name="sincnet", wrapper=_SincNetPhase(model), dummy_inputs=(torch.randn(1, n_samples),),
                root_axis="n_samples",
                mil_inputs=[ct.TensorType(name="waveform", dtype=np.float32, shape=(
                    1, ct.RangeDim(_SINCNET_MIN_SAMPLES, int(self.max_seconds * self.sample_rate))))],
            ),
            RecurrentPhase(name="lstm", module=lstm, number_layers=True),
            ExportPhase(
                name="head", wrapper=_PyannoteHead(model, self), root_axis="n_enc_frames",
                dummy_inputs=(torch.randn(1, frames, width),),
                mil_inputs=[ct.TensorType(name="lstm_out", dtype=np.float32, shape=(
                    1, ct.RangeDim(1, int(model.num_frames(int(self.max_seconds * self.sample_rate)))),
                    width))],
            ),
        ]

    def driver_components(self):
        """SincNet, one bidirectional sweep per LSTM layer, the head. No hand-written Lua.

        The frame count is SincNet's own arithmetic over the sample count (`frames_from_samples`), read
        off the model's layers rather than restated: the LSTM sweep needs it as a number, and the graph
        that computes it has already retained its output rather than handed it over.
        """
        from .driver_components import (
            CALLER, BiRecurrentCall, DriverInputs, DriverReturn, SubgraphCallComponent,
        )
        from .driver_ir import Len, OutputRef, Var

        model = self._model
        layers = int(model.lstm.num_layers) if model is not None else 1
        hidden = int(model.lstm.hidden_size) if model is not None else 1
        in_width = int(model.lstm.input_size) if model is not None else 1
        n_samples = Len(Var("waveform"))
        n_frames = _sincnet_frames_expr(model, n_samples)
        components = [
            DriverInputs(bindings=(("waveform", CALLER),), n_tokens=n_samples),
            SubgraphCallComponent(
                topology="sincnet", outputs=(), retain=True, length=n_samples,
                inputs={"waveform": Var("waveform")},
                note=("SincNet, emitted time-major for the sweeps below and RETAINED: its only "
                      "reader is the first LSTM layer, so it never becomes a Lua table."),
            ),
        ]
        previous = "sincnet"
        for layer in range(layers):
            forward = f"lstm_l{layer}_fwd"
            components.append(BiRecurrentCall(
                forward_topology=forward, backward_topology=f"lstm_l{layer}_bwd",
                out_var=f"lstm_{layer}_gen", sequence=OutputRef(previous), seq_len=n_frames,
                input_dim=in_width if layer == 0 else 2 * hidden, hidden_dim=hidden,
                note=("Both directions of one BiLSTM layer, swept in C++ and retained as one "
                      "[h_fwd | h_bwd] row per frame -- the layout the next layer and the head read."
                      if layer == 0 else None),
            ))
            previous = forward
        components.append(SubgraphCallComponent(
            topology="head", outputs=("probs",), length=n_frames,
            inputs={"lstm_out": OutputRef(previous)},
            note="The two linear layers and the powerset softmax; the probabilities are the answer.",
        ))
        components.append(DriverReturn(values=("probs",)))
        return components

    def contract(self) -> dict:
        return audio_contract(self.output, self.sample_rate, self.labels, self.frame_rate,
                              self.frame_offset)

    def backend_kwargs(self) -> dict:
        return dict(hparams=self.hparams())


# The shortest clip SincNet turns into one frame: its receptive field (251-tap sinc filters at stride
# 10, three max-pools of 3, two 5-tap convolutions). `_sincnet_frames_expr` is the same arithmetic.
_SINCNET_MIN_SAMPLES = 991


def _sincnet_frames_expr(model, n_samples):
    """SincNet's frame count as a driver expression: each valid convolution is `(n - k) // s + 1`, each
    max-pool `(n - k) // s + 1`, in the module's order. Read off the real layers, and checked against
    the model's own `num_frames` at three lengths, so a layer this does not model raises here rather
    than mis-sizing the LSTM sweep."""
    from .driver_ir import BinOp, Lit

    if model is None:
        return n_samples
    stages = []
    for conv, pool in zip(model.sincnet.conv1d, model.sincnet.pool1d):
        if hasattr(conv, "filterbank"):
            kernel, stride = conv.filterbank.kernel_size, conv.filterbank.stride
        else:
            kernel, stride = conv.kernel_size[0], conv.stride[0]
        stages += [(int(kernel), int(stride)), (int(pool.kernel_size), int(pool.stride))]

    def count(n):
        for kernel, stride in stages:
            n = (n - kernel) // stride + 1
        return n

    for n in (_SINCNET_MIN_SAMPLES, 48777, 160000):
        if count(n) != int(model.num_frames(n)):
            raise ValueError(f"_sincnet_frames_expr: {count(n)} frames for {n} samples, the model says "
                             f"{int(model.num_frames(n))}.")
    expr = n_samples
    for kernel, stride in stages:
        expr = BinOp("+", BinOp("floordiv", BinOp("-", expr, Lit(kernel)), Lit(stride)), Lit(1))
    return expr


def _freeze_sinc_filters(sincnet) -> None:
    """asteroid's `ParamSincFB` builds its 80 band-pass filters from learned cut-offs on every forward.
    Built once here and convolved as a constant weight -- the same filters, so the same output, and a
    graph with a `[80, 1, 251]` weight rather than the sinc arithmetic that would rebuild it per call."""
    encoder = sincnet.conv1d[0]
    if encoder.padding != 0 or not encoder.as_conv1d:
        raise ValueError("_freeze_sinc_filters: only an unpadded as_conv1d sinc encoder is reproduced.")
    with torch.no_grad():
        filters = encoder.get_filters().detach().clone()
    stride = int(encoder.stride)

    def forward(self, waveform):
        return F.conv1d(waveform, filters, stride=stride)

    probe = torch.randn(1, 1, 4000, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = encoder(probe)
        encoder.forward = types.MethodType(forward, encoder)
        got = encoder(probe)
    if not torch.allclose(want, got, rtol=0, atol=1e-6 * max(want.abs().max().item(), 1.0)):
        raise ValueError("_freeze_sinc_filters: the frozen filterbank differs from asteroid's.")


class _SincNetPhase(nn.Module):
    """`waveform [1, n]` -> SincNet's features, TIME-MAJOR `[1, n_frames, 60]` -- the layout
    `run_bi_recurrent_and_retain` reads a sequence in (`seq[t * input_dim + k]`).

    Both pyannote wrappers hold the SUBMODULES they run, never the `PyanNet` itself: it is a Lightning
    module, and `torch.jit.trace` walks a module's attributes -- reading `trainer` on one that has none
    raises."""

    def __init__(self, model):
        super().__init__()
        self.sincnet = model.sincnet

    def forward(self, waveform):
        return self.sincnet(waveform.unsqueeze(1)).transpose(1, 2)


class _PyannoteHead(nn.Module):
    """`PyanNet.forward` after the LSTM: two linear layers with leaky ReLU, the classifier, and the
    probabilities rather than pyannote's log-probabilities (`exp` of its `LogSoftmax`)."""

    def __init__(self, model, spec):
        super().__init__()
        self.linear = model.linear
        self.classifier = model.classifier
        self.activation = model.activation
        self._spec = [spec]  # a list, so trace does not walk the config as a submodule

    def forward(self, lstm_out):
        out = lstm_out
        for linear in self.linear:
            out = F.leaky_relu(linear(out))
        out = torch.exp(self.activation(self.classifier(out)))
        _check_output(self._spec[0], out)
        return out


# -- recognizers -----------------------------------------------------------------------------------


def _nemo_target(path: Path) -> str:
    if not _is_nemo_archive(path):
        return ""
    return str(_read_nemo_model_config(path).get("target", ""))


def _is_titanet(path: Path) -> bool:
    """A NeMo speaker-label model whose decoder pools attentively -- TitaNet. The same `target` covers
    ECAPA-style and x-vector NeMo checkpoints too, whose pools differ, so the pool mode is part of the
    claim rather than assumed."""
    if not _nemo_target(path).endswith(_SPEAKER_TARGET):
        return False
    decoder = _read_nemo_model_config(path).get("decoder") or {}
    return decoder.get("pool_mode") == "attention"


def _is_frame_vad(path: Path) -> bool:
    return _nemo_target(path).endswith(_FRAME_TARGET)


def _is_speechbrain_ecapa_classifier(path: Path) -> bool:
    """A speechbrain `EncoderClassifier` directory whose embedding model is ECAPA-TDNN and which carries
    a classifier and its label encoder -- a language id (or any clip classifier on that recipe). Read off
    `hyperparams.yaml` as text: parsing it would instantiate the model, which detection must not do."""
    hyperparams = path / "hyperparams.yaml"
    if not path.is_dir() or not hyperparams.is_file() or not (path / "label_encoder.txt").is_file():
        return False
    text = hyperparams.read_text(errors="replace")
    return "ECAPA_TDNN.ECAPA_TDNN" in text and "classifier:" in text and "EncoderClassifier" not in text


def _pyannote_checkpoint(path: Path) -> Optional[Path]:
    """`pytorch_model.bin` beside a `config.yaml` whose model is pyannote's `PyanNet` -- or the
    directory holding them. Read as text: detection must not import pyannote."""
    directory = path if path.is_dir() else path.parent
    weights, config = directory / "pytorch_model.bin", directory / "config.yaml"
    if not weights.is_file() or not config.is_file():
        return None
    if path.is_file() and path != weights:
        return None
    text = config.read_text(errors="replace")
    return weights if "pyannote.audio.models.segmentation.PyanNet" in text else None


def _is_pyannote_segmentation(path: Path) -> bool:
    return _pyannote_checkpoint(path) is not None


def _build_pyannote_segmentation(path: Path, output_path: str):
    return PyannoteSegmentationExportConfig(checkpoint=str(_pyannote_checkpoint(path)),
                                            output_path=output_path)


def _build_titanet(path: Path, output_path: str):
    return NemoAudioClassificationExportConfig(
        checkpoint=str(path), output=AudioOutput.EMBEDDING, architecture="titanet",
        output_path=output_path,
    )


def _build_frame_vad(path: Path, output_path: str):
    return NemoAudioClassificationExportConfig(
        checkpoint=str(path), output=AudioOutput.FRAME_CLASSES, architecture="marblenet-vad",
        output_path=output_path,
    )


def _build_ecapa_classifier(path: Path, output_path: str):
    return SpeechbrainAudioClassificationExportConfig(
        checkpoint=str(path), output=AudioOutput.CLIP_CLASSES, architecture="ecapa-tdnn-lid",
        output_path=output_path,
    )


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="audio-embedding",
        config_class=AudioClassificationExportConfig,
        recognizers=[
            ModelRecognizer(name="titanet", detect=_is_titanet, build_config=_build_titanet),
        ],
    ))
    registry.register(TaskRegistryEntry(
        task="audio-classification",
        config_class=AudioClassificationExportConfig,
        recognizers=[
            ModelRecognizer(name="marblenet-vad", detect=_is_frame_vad, build_config=_build_frame_vad),
            ModelRecognizer(name="ecapa-tdnn-lid", detect=_is_speechbrain_ecapa_classifier,
                            build_config=_build_ecapa_classifier),
        ],
    ))
    # Its own entry: a multi-phase export does not share `AudioClassificationExportConfig`'s shape, and
    # the task's base is the root for that reason (tasks.py).
    registry.register(TaskRegistryEntry(
        task="audio-classification",
        config_class=PyannoteSegmentationExportConfig,
        recognizers=[
            ModelRecognizer(name="pyannote-segmentation", detect=_is_pyannote_segmentation,
                            build_config=_build_pyannote_segmentation),
        ],
    ))
