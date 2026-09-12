"""The CNN + transformer + CTC family (EXPORT-ROADMAP.md's family 4, P5): a convolutional feature
encoder over the RAW waveform, a transformer, and one linear CTC head -- wav2vec 2.0, HuBERT and
data2vec-audio, plus every checkpoint fine-tuned from them.

**It is family 1's shape with a different front end and a different tokenizer, and both halves of that
sentence turned out to matter.** The roadmap's estimate was "family-1-shaped once the encoder template
generalizes past NeMo, and it needs no new head at all" (`EXPORT-ROADMAP.md` ordering note 6). The head
half was right -- `CtcGreedyBuilder` is reused verbatim, the driver is the same five statements, and the
engine's reduction (`loom.argmax_rows`) was already there. The generalization did not happen the way the
note implies, though: family 1's `build_trace` is a NeMo-shaped `(input_signal, input_signal_length)`
pair around a mel front end, and this family has neither a mel front end nor a length argument. So this
is a sibling template rather than a leaf of that one, and what the two genuinely share is the epilogue.

Three things are this family's own, and each is a place the export would otherwise be silently wrong.

* **THE WAVEFORM NORMALIZATION IS PART OF THE MODEL, AND IT IS NOT IN THE CHECKPOINT'S `forward`.**
  Every one of these checkpoints ships `do_normalize: true` in its `preprocessor_config.json`, and
  `Wav2Vec2FeatureExtractor` applies `(x - x.mean()) / sqrt(x.var() + 1e-7)` to the raw samples before
  the model ever sees them. Family 1's standing rule is that the front end is INSIDE the graph rather
  than in front of it -- that is what lets a host hand the engine a waveform and nothing else -- so the
  wrapper does it here, over the same axis and with the same population variance and the same epsilon.
  Leaving it out does not raise and does not change a shape: it feeds a correctly-shaped graph audio at
  the wrong scale, and a checkpoint trained on normalized input transcribes plausible nonsense from it.
  `do_normalize: false` is honoured (it is a real setting) by omitting the two ops.
* **The blank is the tokenizer's PAD token, and its id is not derivable from the class count.** NeMo's
  CTC convention is "blank is the last class", which is what family 1's `num_classes - 1` reads. This
  family's is HF's: `Wav2Vec2CTCTokenizer` uses `pad_token` as the blank, and it is id 0 in every
  checkpoint here. The two conventions disagree at both ends of the row, so the number is READ off the
  tokenizer (`pad_token` -> its id in `vocab.json`) rather than computed. It is not always spelled
  `<pad>` either: `omni-asr-ctc-300m-v2` declares `pad_token: "<s>"` and has a *separate* `<pad>` piece
  at id 1 that is not the blank, so resolving the name through the vocabulary is the whole of the check.
* **The attention mask is omitted, deliberately, and that is what keeps the length dynamic.** These
  models accept `attention_mask=None` for an unpadded single sequence -- and must be given exactly that,
  because `_get_feature_vector_attention_mask` builds its frame count from a Python-level
  `.shape[1]`, which a trace bakes. This is [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md)
  one modality over: a family whose door hands the model exactly the samples the caller recorded has no
  padding to describe, so the mask that would describe it is the thing that makes the graph static.
  `attn_implementation="eager"` for the same reason it is set in family 12 -- the sdpa path materialises
  a mask of the traced size even when it is handed none.

And one that is neither: **the positional convolution carries a weight-norm parametrization**, which
under `torch.jit.trace` records the `g * v / ||v||` recomputation as graph ops over constants rather
than as a weight. It folds, but it folds into an artifact whose tensor names no longer match the
checkpoint's; removing the parametrization first is one line and leaves a plain `Conv1d`.

Everything else is `AutoModelForCTC`'s, whatever it is. This family names no architecture: the
recognizer claims any HF directory whose `architectures` entry ends in `ForCTC`, and the three
checkpoints it is verified on are structurally different in the three places that could have needed a
branch and did not -- HuBERT's layer-norm convolutional stem and 128-wide positional convolution against
data2vec-audio's group-normed stem and 5-wide one, and `Wav2Vec2ForCTC` with a 10,288-piece multilingual
vocabulary against the two 32-character English ones.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .spec_protocol import Axis, Unchecked

# Dummy trace length and the dynamic sample-axis bounds, in SECONDS, resolved against the checkpoint's
# own declared sampling rate. The lower bound is this family's real floor rather than a round number:
# the convolutional stem has a 400-sample receptive field at stride 320, so anything shorter produces a
# zero-frame encoder output and a CTC decode with nothing to reduce. 30 s at the top matches what these
# checkpoints were fine-tuned on (LibriSpeech utterances) and is the same order as family 1's 20.
TRACE_SECONDS = 1.0
MIN_SECONDS = 0.1
MAX_SECONDS = 30.0

# `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`'s own epsilon, transcribed rather than rounded: it
# sits under a square root applied to a variance, so at a normal speech amplitude it changes the result
# in the last few significant digits and at digital silence it is the only thing standing between the
# division and a NaN.
NORM_EPS = 1e-7


class _CtcAsrWrapper(nn.Module):
    """Reduces any `AutoModelForCTC` to `waveform -> logits`, with the feature extractor's own
    normalization in front of it.

    No `attention_mask`, and no length input to build one from -- see the module docstring. The model is
    called with `input_values` alone, which is the argument name every member of this family declares.
    """

    def __init__(self, model, normalize: bool):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, waveform):
        if self.normalize:
            # Over the SAMPLE axis and over the whole clip, which is what the feature extractor does to
            # an unpadded array. `unbiased=False` is not a preference: numpy's `ndarray.var` is the
            # population variance, and torch's default is the sample one, so the default here would
            # divide by `n-1` and produce a scale that differs from the reference by a factor no
            # comparison would attribute to the graph.
            mean = waveform.mean(dim=-1, keepdim=True)
            var = waveform.var(dim=-1, keepdim=True, unbiased=False)
            waveform = (waveform - mean) / torch.sqrt(var + NORM_EPS)
        return self.model(input_values=waveform).logits


def strip_weight_norm(model) -> None:
    """Removes every `weight_norm` parametrization on the model, in place.

    `Wav2Vec2PositionalConvEmbedding` wraps its depthwise convolution in
    `nn.utils.parametrizations.weight_norm`, which is a forward-time recomputation of the weight from
    two parameters. A trace records that recomputation rather than the weight, so the exported graph
    carries a norm-and-scale over constants where the checkpoint has one convolution kernel. The values
    are identical either way; what differs is that the artifact's tensors stop corresponding to the
    checkpoint's.

    Structural, like everything else here: it asks each module whether it IS parametrized rather than
    looking for a module named `pos_conv_embed`, so a checkpoint that puts a weight-normed convolution
    somewhere else is covered by the same line.
    """
    import torch.nn.utils.parametrize as parametrize

    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def read_ctc_vocab(tokenizer_dir: str) -> dict:
    """A `Wav2Vec2CTCTokenizer` directory's `vocab.json` and the two ids that are not derivable from it.

    Returns `{"pieces": [...], "blank_id": int, "word_delimiter_id": int, "unk_id": int}` -- pieces
    indexed BY id, which is the convention every vocabulary in this schema already uses.

    **Both ids are resolved through the vocabulary rather than assumed.** The blank is `pad_token`'s id
    (see the module docstring for the checkpoint where that is not `<pad>`), and the word delimiter is
    `word_delimiter_token`'s -- `|` in the two English checkpoints and a literal space in the
    multilingual one, which is why the engine is told the ID and reads the spelling off the same array
    everything else does.
    """
    tok_dir = Path(tokenizer_dir)
    vocab = json.loads((tok_dir / "vocab.json").read_text())
    config = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = tok_dir / name
        if path.exists():
            # tokenizer_config wins where both name a token: it is the file the tokenizer class itself
            # is constructed from, and special_tokens_map is the older half of the same statement.
            config = {**json.loads(path.read_text()), **config}

    if not vocab:
        raise ValueError(f"{tok_dir}/vocab.json is empty; a CTC tokenizer with no pieces cannot decode")
    pieces = [None] * (max(vocab.values()) + 1)
    for piece, token_id in vocab.items():
        pieces[int(token_id)] = piece
    missing = [i for i, piece in enumerate(pieces) if piece is None]
    if missing:
        # A gap means two different ids would decode to the same nothing, and the row of the CTC head
        # that names one of them has no spelling at all. Raise rather than fill: this has never been
        # seen, and a placeholder would make it invisible the one time it happens.
        raise ValueError(
            f"{tok_dir}/vocab.json has no piece for id(s) {missing[:8]} of {len(pieces)}; a CTC "
            f"vocabulary is indexed by the head's own row number and cannot have holes")

    def _id_of(token_name: str, what: str) -> int:
        token = config.get(token_name)
        if token is None:
            raise ValueError(
                f"{tok_dir} declares no `{token_name}`, which is where this family's {what} comes from "
                f"(HF's CTC tokenizers name it there, not in vocab.json).")
        if isinstance(token, dict):  # the AddedToken serialization some checkpoints use
            token = token.get("content")
        if token not in vocab:
            raise ValueError(
                f"{tok_dir} declares `{token_name}` = {token!r}, which is not a piece in vocab.json -- "
                f"so the {what} cannot be resolved to a row of the CTC head.")
        return int(vocab[token])

    return {
        "pieces": pieces,
        "blank_id": _id_of("pad_token", "CTC blank"),
        "word_delimiter_id": _id_of("word_delimiter_token", "word delimiter"),
        "unk_id": _id_of("unk_token", "unknown piece"),
    }


@dataclass(kw_only=True)
class ASRCtcExportConfig(LoomExportConfig):
    """Any HF directory declaring a `*ForCTC` architecture -> Loom GGUF.

    One generic recognizer and no specific ones, for family 12's reason: there is nothing a caller could
    choose about this export and nothing a checkpoint declares that the family special-cases. A real
    exception goes in `_MODEL_TYPE_OVERRIDES`, not in a second recognizer.
    """

    architecture: Optional[str] = None
    model_dir: str
    tokenizer_dir: Optional[str] = None
    decomposition: Decomposition = None
    # `EXPORT-ROADMAP.md` R1: a waveform's own axis is raw audio samples, never a token count --
    # the same declaration family 1 makes, and checked the same way.
    root_axis: str = "n_samples"
    # Resolved from the checkpoint during `load_model()` / `build_trace()`.
    _resolved_architecture: Optional[str] = field(default=None, init=False, repr=False)
    _sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    _normalize: bool = field(default=True, init=False, repr=False)
    _blank_id: Optional[int] = field(default=None, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory. The recognizer's detect() already read its config.json -- that "
            "is how it claimed the checkpoint at all -- and AutoModelForCTC.from_pretrained raises on "
            "anything it cannot load."
        ),
        "tokenizer_dir": Unchecked(
            "defaults to model_dir. Whether that directory holds a CTC vocabulary is decided by "
            "`read_ctc_vocab` against the real files, which raises naming the file it could not read."
        ),
        "_resolved_architecture": Unchecked(
            "load_model()'s output, cached so export_architecture() can read it back. A field only "
            "because the config is a dataclass."
        ),
        "_sample_rate": Unchecked(
            "READ off the checkpoint's own feature extractor during build_trace, never declared -- it "
            "is what the trace length and the dynamic-axis bounds are derived from, so a caller value "
            "would be a second authority over the same number."
        ),
        "_normalize": Unchecked(
            "the feature extractor's own `do_normalize`, read alongside the sample rate from the same "
            "file. Same reason: there is no second authority for what the checkpoint's front end does."
        ),
        "_blank_id": Unchecked(
            "READ off the tokenizer (`pad_token` -> its id in vocab.json) during build_trace. The "
            "class count cannot supply it -- this family's blank is row 0, not the last row -- and "
            "nothing else in the artifact states it, so there is no second authority to check against."
        ),
    }

    def __post_init__(self):
        # Structural, not chosen: the convolutional stem, the transformer and the CTC head are one
        # graph with no boundary a caller could name. Defaulted here rather than with a
        # `field(default_factory=...)` so it keeps its place in the kw-only field order.
        if self.decomposition is None:
            self.decomposition = Flattened()

    def load_model(self):
        from transformers import AutoModelForCTC

        print(f"Loading CTC model from {self.model_dir}...")
        # `attn_implementation="eager"`, and load-bearing rather than conservative -- see the module
        # docstring, and ADR-019, which is the same finding for family 12's encoder.
        model = AutoModelForCTC.from_pretrained(
            self.model_dir, dtype=torch.float32, attn_implementation="eager").eval()
        self._resolved_architecture = self.architecture or getattr(model.config, "model_type", None)
        if not self._resolved_architecture:
            raise ValueError(
                "architecture could not be inferred from model.config.model_type; pass it explicitly")
        strip_weight_norm(model)
        return model

    def export_architecture(self) -> str:
        return self._resolved_architecture or self.architecture

    def build_trace(self, model):
        """`Flattened`'s hook: the wrapper, one clip of dummy audio, and the one MIL input declaration.

        The feature extractor is read here rather than in `load_model` because it answers two questions
        at once -- the sample rate the trace length is built from, and whether the waveform is
        normalized -- and both are needed exactly at this point.
        """
        from transformers import AutoFeatureExtractor

        extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self._sample_rate = int(extractor.sampling_rate)
        self._normalize = bool(getattr(extractor, "do_normalize", True))
        self._blank_id = int(read_ctc_vocab(self.tokenizer_dir or self.model_dir)["blank_id"])

        import coremltools as ct

        n_samples = int(TRACE_SECONDS * self._sample_rate)
        dummy_inputs = (torch.randn(1, n_samples, dtype=torch.float32),)
        print(f"Tracing the complete PyTorch graph (dummy n_samples={n_samples}, "
              f"normalize={self._normalize})...")
        seq_dim = ct.RangeDim(int(MIN_SECONDS * self._sample_rate),
                              int(MAX_SECONDS * self._sample_rate))
        mil_inputs = [ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32)]
        return _CtcAsrWrapper(model, self._normalize), dummy_inputs, mil_inputs

    def synthesized_builder_key(self) -> str:
        """The same answer family 1's CTC leaf gives, for the same reason it gives it (P4.0.17): this is
        a `Flattened` export and what differs from every other one is entirely what the host does with
        the single output -- reduce every frame and collapse, rather than reduce one row."""
        return "CtcGreedy"

    def hparams(self) -> dict:
        """The one number a host needs that the driver cannot hand it: what rate the samples it is
        about to pass must be at. `loom::ModelContract` already reads `loom.sample_rate` and falls back
        to 16000 for a file that declares none, which is the right number for every checkpoint in this
        family and is exactly why declaring it matters -- a fallback that happens to be correct is not a
        statement, and the next member of this family to be fine-tuned at 8 kHz would inherit it."""
        return {} if self._sample_rate is None else {"sample_rate": self._sample_rate}

    def backend_kwargs(self) -> dict:
        return dict(
            flat_namespace=True,
            root_axis=self.root_axis,
            driver_builder=self.synthesized_builder_key(),
            hparams=self.hparams(),
            tokenizer_dir=self.tokenizer_dir or self.model_dir,
            tokenizer_family="ctc",
            # Omitted rather than raised when the trace has not run, because `component_registry.usage()`
            # builds every registered config without ever tracing. The export path cannot slip through:
            # the exporter raises when asked for the CTC builder without a blank id.
            **({} if self._blank_id is None else {"ctc_blank_id": self._blank_id}),
        )


def _hf_config(path: Path) -> Optional[dict]:
    """An HF-style directory's own `config.json`, parsed, or None if `path` isn't one. Never raises:
    `detect()` runs against unidentified paths by construction."""
    cfg_path = path / "config.json"
    if not path.is_dir() or not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _is_hf_ctc_asr(path: Path) -> bool:
    """Any HF-style directory declaring a `model_type` AND an `architectures` entry ending in `ForCTC`.

    The same two-halves check as `token_classification_export._is_hf_token_classifier`, and load-bearing
    for the same reason: `TaskRegistry.detect` runs every recognizer against every path, so `model_type`
    alone would claim the pretrained-only checkpoints (`Wav2Vec2ForPreTraining`) that sit beside these on
    disk -- which have no CTC head and no tokenizer at all, and would export a GGUF whose "logits" are an
    encoder activation.

    Registered `fallback=True` -- consulted only when no specific recognizer matched -- so adding a
    specific one later cannot make this detection ambiguous.
    """
    cfg = _hf_config(path)
    if cfg is None or not cfg.get("model_type"):
        return False
    architectures = cfg.get("architectures") or []
    if not isinstance(architectures, list):
        return False
    return any(isinstance(arch, str) and arch.endswith("ForCTC") for arch in architectures)


# Per-`model_type` exceptions to the generic path's defaults, as `ASRCtcExportConfig` kwargs. Empty, and
# for the same reason family 12's copy is: nothing this family exports needs one yet.
_MODEL_TYPE_OVERRIDES: dict[str, dict] = {}


def _build_hf_ctc_asr(path: Path, output_path: str) -> LoomExportConfig:
    cfg = _hf_config(path) or {}
    overrides = _MODEL_TYPE_OVERRIDES.get(cfg.get("model_type") or "", {})
    return ASRCtcExportConfig(
        architecture=None, output_path=output_path, model_dir=str(path), **overrides,
    )


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRCtcExportConfig,
        recognizers=[
            ModelRecognizer(name="hf-ctc-asr", detect=_is_hf_ctc_asr,
                            build_config=_build_hf_ctc_asr, fallback=True),
        ],
    ))
