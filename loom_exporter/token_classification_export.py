"""The BERT-family token-classifier family (EXPORT-ROADMAP.md's family 12, P5): a bidirectional
encoder plus one linear head, `text` in and one class per token out.

**It is the smallest template in the roadmap, and it is here for what it PROVES rather than for what
it covers.** Every family before it produces audio or consumes it, so "the registry is task-shaped,
not audio-shaped" was a claim with no witness; `loom.task = "token-classification"` and
`loom.output.kind = "class"` are the first export where the contract's non-audio half is exercised
end to end (loom.cpp `docs/HIGH-LEVEL-API.md` §3, whose `Text2Class` door had been a `_Planned`
interface with nothing behind it). The models it covers -- punctuation restoration, truecasing, NER --
are also the post-processing half of an ASR pipeline, which is why the roadmap puts it first.

**No engine work, which is the acceptance criterion the roadmap states for a new family.** The
reduction the driver needs already exists: `loom.argmax_rows` was built for Conformer-CTC's frame-wise
head (P4.0.17), and one class per *token* is the same reduction over the same tensor shape -- the whole
difference is that a CTC head then collapses repeats and drops a blank and this one does not, which is
`TokenLabelsEpilogue` against `CtcGreedyEpilogue`, one component.

**THE WORK IN THIS FAMILY IS TRACING, NOT ARCHITECTURE**, and the one rule that covers all of it is:
*every tensor the model needs must be derived from an input, never computed from `.shape[1]`.* Each
place `transformers` breaks that rule is silently harmless at the traced length and only diverges past
it, which is what makes them worth naming individually:

* **The attention mask is derived from `tokens`, not built by the model.** `torch.ones_like(tokens)` --
  a real, all-ones padding mask, because this family's door hands the model exactly the tokens the
  caller wrote and there is no padding to hide. Passing one is what stops the model building its own:
  BERT would `torch.ones(input_shape)` at the traced length, and DistilBERT does the same inline. The
  2-D shape is load-bearing too. It reaches `ModuleUtilsMixin.get_extended_attention_mask`, which for
  an encoder is `mask[:, None, None, :]` and `(1 - x) * finfo.min` -- pure arithmetic on the input
  tensor. The route it must NOT take is the sdpa one: `_prepare_4d_attention_mask_for_sdpa` expands to
  a Python-level `tgt_len`, and under `torch.jit.trace` its `torch.all(mask == 1)` early-out is skipped
  (`is_tracing` is true), so even an all-ones mask becomes a baked `[1, 1, 128, 128]` constant. That is
  why `load_model` asks for `attn_implementation="eager"`.
* **`token_type_ids` is `tokens * 0`, where the model takes one at all.** `BertModel`'s own default is
  `self.embeddings.token_type_ids[:, :seq_length]`, a buffer slice at the traced length; `tokens * 0`
  is the same tensor of zeros -- single-segment input, which is what a token classifier takes -- built
  from an input whose length is dynamic. DistilBERT has no token-type embedding and no such argument,
  which is read off its `forward` signature rather than off its name.
* **`position_ids` is an input, delivered one of two ways.** Where the model accepts the argument
  (BERT, RoBERTa, ELECTRA) it is passed, for the same reason the causal-LM family passes
  `cache_position`. Where it does not (DistilBERT, whose `Embeddings.forward` reads
  `self.position_ids[:, :seq_length]` -- a buffer slice whose own comment says it "helps when tracing",
  which is true of a fixed-shape export and exactly wrong here), a forward pre-hook on the
  position-embedding table substitutes the caller's ids for the ones the model computed. **Both routes
  produce the same two graph inputs and the same driver**, which is the point: the difference is
  absorbed here rather than reaching the artifact.

Everything else is `AutoModelForTokenClassification`'s, whatever it is. This family names no
architecture: `base_model` is HF's own accessor for the encoder underneath a task head, and which
arguments that encoder takes is read from its signature. Verified on two structurally different
checkpoints -- BERT (token-type embeddings, `position_ids` argument, `.encoder`) and DistilBERT (none
of the three) -- which is what the "modular-export generality is unproven" item in the backlog asks of
a template and this one now has.
"""
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForTokenClassification

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .spec_protocol import Unchecked


def _accepts(fn, name: str) -> bool:
    """Whether `fn` declares a parameter called `name`.

    The whole of this family's per-architecture branching, and it is a question about the model rather
    than about its name -- which is what lets one wrapper cover BERT and DistilBERT without a table
    mapping `model_type` to a call signature.
    """
    return name in inspect.signature(fn).parameters


class _PositionHolder:
    """The caller's `position_ids`, for a model that has no argument to receive them through.

    A plain attribute rather than a buffer or a closure variable, and it works under
    `torch.jit.trace` for the reason tracing works at all: the trace follows the DATA, not the module
    boundaries, so a tensor built in `_TokenClassifierWrapper.forward` and read inside a submodule's
    pre-hook is one connected graph. It is per-wrapper state, so two exports cannot see each other's.
    """

    positions = None


class _TokenClassifierWrapper(torch.nn.Module):
    """Reduces any `AutoModelForTokenClassification` to `(tokens, position_ids) -> logits`.

    The constructor asks the encoder two questions and answers a third; see the module docstring for
    why each of them is a place the traced length would otherwise be baked in. Nothing here is keyed on
    an architecture name.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        base = model.base_model
        self._takes_position_ids = _accepts(base.forward, "position_ids")
        self._takes_token_type_ids = _accepts(base.forward, "token_type_ids")
        self._holder = _PositionHolder()
        if not self._takes_position_ids:
            self._redirect_position_lookup(base)

    def _redirect_position_lookup(self, base) -> None:
        """Feed the caller's ids to the position-embedding table, whatever the model computed.

        A forward pre-hook returning a replacement argument tuple, rather than a rewritten
        `Embeddings.forward`: the hook is four lines and reimplementing the embeddings would be a copy
        of one family's arithmetic that has to be kept in step with it. The ids the model computed
        become a dead constant and are pruned with every other dead node.

        Raises rather than exporting when the table cannot be found: a model with no `position_ids`
        argument AND no position table is one whose positions come from somewhere this has not seen, and
        guessing would produce a graph that is wrong only past the traced length -- the exact failure
        this whole path exists to prevent.
        """
        table = getattr(getattr(base, "embeddings", None), "position_embeddings", None)
        if table is None:
            raise ValueError(
                f"{type(base).__name__}.forward takes no `position_ids`, so its positions are computed "
                f"internally from the traced sequence length -- and it has no "
                f"`embeddings.position_embeddings` table to redirect, which is how that is worked "
                f"around for DistilBERT. Exporting it would bake the traced length into the graph. "
                f"Add the seam this model does have to `_redirect_position_lookup`."
            )
        table.register_forward_pre_hook(lambda module, args: (self._holder.positions,))

    def forward(self, tokens, position_ids):
        # Set unconditionally, so the one attribute is live whether or not a hook reads it -- a value
        # assigned only on the branch that needs it is a `None` waiting for the next model shape.
        self._holder.positions = position_ids.unsqueeze(0)
        kwargs = {}
        if self._takes_position_ids:
            kwargs["position_ids"] = position_ids
        if self._takes_token_type_ids:
            kwargs["token_type_ids"] = tokens * 0
        return self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), **kwargs).logits


@dataclass(kw_only=True)
class TokenClassificationExportConfig(LoomExportConfig):
    """Any HF directory declaring a `*ForTokenClassification` architecture -> Loom GGUF.

    One recognizer, generic, and no specific ones: unlike the causal-LM family there is nothing a
    caller could choose about this export (no monolithic/modular split, no sampler defaults) and
    nothing a checkpoint declares that the family has to special-case. When one turns up, it goes in
    `_MODEL_TYPE_OVERRIDES` the way the causal-LM family's table is shaped, not in a second recognizer.
    """

    architecture: Optional[str] = None
    model_dir: str
    tokenizer_dir: Optional[str] = None
    tokenizer_family: Optional[str] = None
    decomposition: Decomposition = None
    # Concrete length `torch.jit.trace` runs at; the dynamic range is declared separately through
    # `ct.convert`'s own `inputs=`, exactly as in the causal-LM family.
    seq_len: int = 64
    # The `ct.RangeDim` upper bound. `None` reads the checkpoint's own `max_position_embeddings`, which
    # is a HARD limit here rather than the soft one it is for a RoPE decoder: the position embedding is
    # a learned table, so a longer sequence indexes past its last row.
    max_seq_len: Optional[int] = None
    # Resolved from the checkpoint by `load_model()`.
    _resolved_architecture: Optional[str] = None
    _labels: Optional[List[str]] = None

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory. The recognizer's detect() already read its config.json -- that "
            "is how it claimed the checkpoint at all -- and AutoModelForTokenClassification."
            "from_pretrained raises on anything it cannot load."
        ),
        "tokenizer_dir": Unchecked(
            "defaults to model_dir. Whether that directory holds a tokenizer the exporter recognizes "
            "is decided by tokenizer_detect against the real files, not here."
        ),
        "tokenizer_family": Unchecked(
            "an override for the exporter's own auto-detection, which reads the tokenizer's real "
            "files. Nothing to cross-check it against that is not the detection it overrides."
        ),
        "seq_len": Unchecked(
            "the concrete length torch.jit.trace runs at. The dynamic range is declared separately via "
            "ct.convert's own inputs=, so this number constrains nothing the checkpoint could "
            "disagree with."
        ),
        "max_seq_len": Unchecked(
            "None means READ `max_position_embeddings` off the checkpoint, which is the case that "
            "needs no claim. An explicit value is a caller asking for a SHORTER range than the model "
            "supports, which is legitimate; asking for a longer one is not, and `build_trace` raises "
            "there rather than declaring a link that could only re-read the same number."
        ),
        "_resolved_architecture": Unchecked(
            "load_model()'s output, cached on the config so export_architecture() can read it back. A "
            "field only because the config is a dataclass."
        ),
        "_labels": Unchecked(
            "the checkpoint's own `id2label`, READ during load_model() rather than declared -- there "
            "is no second authority for a label set, and a caller who supplied one could only be "
            "renaming the model's classes."
        ),
    }

    def __post_init__(self):
        # Structural, not chosen: an encoder plus its head is one graph, so unlike the causal-LM family
        # there is no modular boundary a caller could name. Defaulted here rather than with a
        # `field(default_factory=...)` so it keeps its place in the kw-only field order.
        if self.decomposition is None:
            self.decomposition = Flattened()

    def load_model(self):
        print(f"Loading token classifier from {self.model_dir}...")
        # `attn_implementation="eager"` rather than the default, and it is load-bearing rather than
        # conservative: `BertModel.forward` routes a 2-D mask through
        # `_prepare_4d_attention_mask_for_sdpa`, which expands to a Python-level `tgt_len` and whose
        # all-ones early-out is skipped under tracing -- so the sdpa path bakes the traced length even
        # for the all-ones mask this family passes. Eager reaches `get_extended_attention_mask`
        # instead, which is arithmetic on the input tensor. The numbers are identical either way; what
        # differs is which MIL ops it lowers to (matmul/softmax/matmul rather than one fused
        # primitive), and this family fuses nothing anyway.
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir, dtype=torch.float32, attn_implementation="eager").eval()
        self._resolved_architecture = self.architecture or getattr(model.config, "model_type", None)
        if not self._resolved_architecture:
            raise ValueError(
                "architecture could not be inferred from model.config.model_type; pass it explicitly"
            )
        self._labels = _read_labels(model.config)
        return model

    def export_architecture(self) -> str:
        return self._resolved_architecture or self.architecture

    def build_trace(self, model):
        """`Flattened`'s hook: the wrapper, its dummy inputs, and the MIL input declarations.

        `tokens` and `position_ids` share ONE `ct.RangeDim` instance, so coremltools ties them to a
        single symbolic length -- the same one-symbol rule `_validate_input_axes` enforces and the same
        reason the causal-LM family shares its dim across three inputs. `tokens` is declared FIRST
        because `apply_monolithic_export` reads the root axis off the traced function's first input.
        """
        limit = _position_limit(model.config)
        max_seq_len = self.max_seq_len or limit
        if limit and max_seq_len > limit:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds this checkpoint's own max_position_embeddings="
                f"{limit}. A learned position table has no rows past that, so a longer sequence would "
                f"index off the end of it rather than degrade -- unlike a RoPE decoder, where a longer "
                f"range is merely extrapolation."
            )
        if self.seq_len > max_seq_len:
            raise ValueError(f"seq_len={self.seq_len} exceeds max_seq_len={max_seq_len}")

        print(f"Tracing the complete PyTorch graph (dummy seq_len={self.seq_len})...")
        dummy_inputs = (
            torch.zeros((1, self.seq_len), dtype=torch.long),
            torch.arange(self.seq_len, dtype=torch.long),
        )
        seq_len_dim = ct.RangeDim(1, max_seq_len)
        mil_inputs = [
            ct.TensorType(name="tokens", shape=(1, seq_len_dim), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(seq_len_dim,), dtype=np.int32),
        ]
        return _TokenClassifierWrapper(model), dummy_inputs, mil_inputs

    def synthesized_builder_key(self) -> str:
        """Which `driver_components.SYNTHESIZED_BUILDERS` entry assembles this config's driver.

        The second family to override this, and for the reason the first one records (P4.0.17): this is
        a `Flattened` export exactly like Qwen3, and what differs is entirely what the host does with
        the one output. `ArgmaxEpilogue` would reduce row `n_tokens - 1` and return the last token's
        label; the answer is every row's.
        """
        return "TokenLabels"

    def contract(self) -> dict:
        """The task contract, plus the label names.

        `loom.labels` is what makes the ids the driver returns readable, and it is a property of the
        CHECKPOINT by the tier-0 admission test in `docs/HIGH-LEVEL-API.md` §2 -- only the file knows
        that class 3 is `B-PER`. Written as one string array indexed BY id, which is the same
        id-indexed-array convention every vocabulary in this schema already uses, so a host needs no
        parallel id array to read it.
        """
        contract = super().contract()
        if self._labels:
            contract["labels"] = list(self._labels)
        return contract

    def backend_kwargs(self) -> dict:
        return dict(
            flat_namespace=True,
            driver_builder=self.synthesized_builder_key(),
            hparams=self.hparams(),
            tokenizer_dir=self.tokenizer_dir or self.model_dir,
            tokenizer_family=self.tokenizer_family,
        )


def _read_labels(config) -> List[str]:
    """`config.id2label` as an id-indexed list, or `[]` when the checkpoint names none.

    HF stores it as a dict whose keys are ints or strings depending on whether the config came from
    JSON, and nothing guarantees it is dense or ordered -- so the list is built by index and any gap is
    filled with the id itself rather than with an empty string, which would make two unnamed classes
    indistinguishable.
    """
    id2label = getattr(config, "id2label", None) or {}
    if not id2label:
        return []
    by_id = {int(key): str(value) for key, value in id2label.items()}
    return [by_id.get(i, f"LABEL_{i}") for i in range(max(by_id) + 1)]


def _position_limit(config) -> int:
    """`max_position_embeddings`, or 0 for a checkpoint that declares none."""
    return int(getattr(config, "max_position_embeddings", 0) or 0)


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


def _is_hf_token_classifier(path: Path) -> bool:
    """Any HF-style directory declaring a `model_type` AND an `architectures` entry ending in
    `ForTokenClassification`.

    The same two-halves check as `causal_lm_export._is_hf_causal_lm`, and load-bearing for the same
    reason: `TaskRegistry.detect` runs every recognizer against every path, so a check on `model_type`
    alone would claim the causal LMs sitting beside these on disk. `architectures` is the checkpoint's
    own statement of which `AutoModelFor*` class it loads through, which is exactly what the wrapper
    needs to be true.

    Registered `fallback=True` -- consulted only when no specific recognizer matched -- so adding a
    specific one later cannot make this detection ambiguous.
    """
    cfg = _hf_config(path)
    if cfg is None or not cfg.get("model_type"):
        return False
    architectures = cfg.get("architectures") or []
    if not isinstance(architectures, list):
        return False
    return any(isinstance(arch, str) and arch.endswith("ForTokenClassification")
               for arch in architectures)


# Per-`model_type` exceptions to the generic path's defaults, as `TokenClassificationExportConfig`
# kwargs. Empty, and for the same reason the causal-LM family's copy is: nothing this family exports
# needs one yet. It is where a real exception goes -- a `tokenizer_family` the detection cannot read
# off the directory, a `max_seq_len` a checkpoint's config overstates.
_MODEL_TYPE_OVERRIDES: dict[str, dict] = {}


def _build_hf_token_classifier(path: Path, output_path: str) -> LoomExportConfig:
    cfg = _hf_config(path) or {}
    overrides = _MODEL_TYPE_OVERRIDES.get(cfg.get("model_type") or "", {})
    return TokenClassificationExportConfig(
        architecture=None, output_path=output_path, model_dir=str(path), **overrides,
    )


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="token-classification",
        config_class=TokenClassificationExportConfig,
        recognizers=[
            ModelRecognizer(name="hf-token-classifier", detect=_is_hf_token_classifier,
                            build_config=_build_hf_token_classifier, fallback=True),
        ],
    ))
