"""Family 5's second leaf: Paraformer -- a SANM encoder, a **CIF predictor**, and a non-autoregressive
SANM decoder (`EXPORT-ROADMAP.md` family 5, P5).

It shares the first leaf's encoder shape and inherits its front end whole
(`sanm_asr_export.KaldiFbankLfrCmvn`, `sinusoidal_positions`), which is the part of "family 5" that is
actually a family. What it adds is the thing no model in this zoo had before: **an output whose LENGTH
depends on the VALUES rather than on any shape.**

## Continuous integrate-and-fire, and where it is computed

The predictor emits one scalar `alpha` per encoder frame and fires a token each time the running sum
crosses an integer. The number of tokens is therefore `floor(sum(alphas))` -- unknowable from any
shape, and not expressible as a traced graph's output extent.

The decomposition this module rests on splits it at exactly one seam:

* **The host decides the boundary**, because it is the only party that can -- see the float32 note
  below. From `alphas` it works out which frames fire and what fraction of each firing frame belongs to
  the token that just closed.
* **The graph does one matmul and contains no threshold, no cumulative sum and no gather.** Every
  token is a weighted sum of encoder frames whose weights depend on `alphas` alone, so the host -- which
  already has `alphas` -- hands over the whole `(n_tokens, n_frames)` resampling matrix:

      emb = cif_weights @ encoder_out

  That form was forced as well as preferred. FunASR DIFFERENCES two cumulative sums, which would need
  a cumulative sum along the graph's slow axis (`ggml_cumsum` only sums over `ne[0]`) and a row gather
  from the permuted result (`ggml_get_rows` needs a contiguous source, the same constraint the first
  leaf hit in `ggml_repeat`). It is also **more accurate than the reference**: differencing two
  cumulative sums is catastrophic cancellation by construction. Against an f64 evaluation of the same
  algebra, FunASR's own f32 result is 7.18e-07 away and this is 4.43e-08 -- sixteen times closer.

**A THRESHOLD CROSSING MUST BE COMPUTED IN EXACTLY ONE PLACE**, and that is why the weights arrive
whole rather than half-derived. The first working design had the host choose the firing frames and the
GRAPH recompute the fractions from its own float32 cumsum; the two agree everywhere except on the
frames sitting within an ulp of an integer, which is exactly the set that decides anything, and one
token would then have been assembled from two different boundaries.

**The reference's arithmetic is float32 and its ORDER is part of the contract** -- see `cif_fire.lua`,
which documents both traps. Getting either wrong does not raise: it moves the acoustic embeddings by
~2.6e-01, the logits by ~3.8e-01, and leaves a transcript that is still almost right.

## Two phases, and the edge between them

`encoder`: `waveform -> (hidden, alphas)`. `decode`: `(encoder_out, cif_weights) -> logits`. The hidden
states cross as an `OutputRef` -- retained, never marshalled, `T * 512` floats that nothing host-side
reads -- while `alphas` is bound as a VALUE, because the host genuinely computes with it and then
consumes it: it never reaches the second phase at all. That is exactly the split
[ADR-031](../../loom.cpp/docs/adrs/adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)
draws. **No new engine binding was needed for any of it**: `OutputRef`, `loom.get_output` and
`loom.output_shape` already existed, the last of them for a transducer's decode loop asking the same
question this driver asks.

The decode phase has TWO dynamic axes -- the token count it produces and the frame count it
cross-attends over -- so it declares the second through `declared_axes`, the same way `t5_export` does
for its cross-attention K/V.

## What it does NOT do yet

**The vocabulary.** `tokens.json` is 8404 flat decode-only pieces, but the decode rule is not a join:
FunASR merges `@@`-suffixed subword continuations, spaces Latin words and joins CJK bare. That cannot
be folded into an existing family, because `@@` marks "continues into the NEXT piece" where
SentencePiece's `▁` and WordPiece's `##` mark word START, and the same piece string appears in both
roles. Until the engine has a reader for it this export writes **no vocabulary at all**, and the driver
returns ids -- which is the honest state: a file that claimed a vocabulary it detokenized wrongly would
be worse than one that admits it has none.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import ExportPhase
from .sanm_asr_export import (
    KaldiFbankLfrCmvn, MAX_SECONDS, MIN_SECONDS, TRACE_SECONDS, _funasr_config, sinusoidal_positions,
)
from .spec_protocol import Axis, Unchecked

# The decode phase's dynamic bounds, in TOKENS. The floor is 1 -- a clip that fires nothing has no
# decode to run and the driver never reaches this phase -- and the ceiling is what 30 s of speech can
# produce at this predictor's rate with a wide margin.
MIN_TOKENS = 1
MAX_TOKENS = 1024


class _ParaformerEncoderPhase(nn.Module):
    """`waveform -> (hidden, alphas)`.

    The predictor's alphas ride out of this phase rather than being consumed inside it, because the
    boundary they imply is the host's to decide. Everything before them is the first leaf's encoder
    with the prompt rows removed.
    """

    def __init__(self, model, frontend: KaldiFbankLfrCmvn):
        super().__init__()
        self.model, self.frontend = model, frontend

    def forward(self, waveform):
        encoder = self.model.encoder
        x = self.frontend(waveform)
        x = x * (encoder.output_size() ** 0.5)
        x = x + sinusoidal_positions(x)
        for layer in encoder.encoders0:
            x = layer(x, None)[0]
        for layer in encoder.encoders:
            x = layer(x, None)[0]
        x = encoder.after_norm(x)

        predictor = self.model.predictor
        conv = torch.relu(predictor.cif_conv1d(predictor.pad(x.transpose(1, 2)))).transpose(1, 2)
        alphas = torch.sigmoid(predictor.cif_output(conv))
        alphas = torch.nn.functional.relu(
            alphas * predictor.smooth_factor - predictor.noise_threshold).squeeze(-1)
        # `tail_process_fn` for an unpadded single utterance, which is the only case this export has:
        # one extra frame carrying `tail_threshold`, so a final partial token still fires. The matching
        # zero row of `hidden` is appended in the decode phase, where the two are used together.
        tail = torch.full((1, 1), float(predictor.tail_threshold), dtype=alphas.dtype)
        return x, torch.cat([alphas, tail], dim=1)


class _ParaformerDecodePhase(nn.Module):
    """`(encoder_out, cif_weights) -> logits`.

    No threshold, no cumulative sum, no gather, no data-dependent branch: the boundary arrived as a
    resampling matrix and what is left is one matmul. See the module docstring for why that split is
    the design rather than a convenience.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_out, cif_weights):
        decoder = self.model.decoder
        # The whole of CIF, once the host has decided the boundary: each token is a weighted sum of
        # encoder frames, so the resampling is a matmul. No cumulative sum, no gather, no threshold.
        embeds = torch.matmul(cif_weights.unsqueeze(0), encoder_out)

        # Masks omitted for the reason family 5's first leaf omits them and ADR-019 states one modality
        # over: every mask here is `sequence_mask` over a single unpadded sequence, which is all ones,
        # and all ones is what the FSMN multiply and the attention `masked_fill` each do nothing with.
        x = embeds
        for layer in decoder.decoders:
            x = layer(x, None, encoder_out, None)[0]
        for layer in decoder.decoders3:
            x = layer(x, None, encoder_out, None)[0]
        return decoder.output_layer(decoder.after_norm(x))


@dataclass(kw_only=True)
class ParaformerExportConfig(LoomExportConfig):
    """A FunASR `Paraformer` checkpoint directory -> Loom GGUF, in two phases.

    Everything that could be declared is read off the checkpoint: the frontend geometry, the CMVN, the
    sample rate, the CIF threshold. What is left is the path.
    """

    # A plain default rather than `export_architecture()`: `MultiPhase.export` reads this FIELD (it
    # is what reaches the GGUF's `general.architecture`), and a config that answered only through the
    # method would export as the fallback `mil_model` with nothing raising.
    architecture: str = "paraformer"
    model_dir: str
    # `EXPORT-ROADMAP.md` R1 again: the ENCODER phase is measured in raw audio samples, and the decode
    # phase in the tokens the predictor fired. Two phases, two root axes, which is what `phases()`
    # declares per phase rather than once for the export.
    root_axis: str = "n_samples"
    # `MultiPhase` by construction: the encoder/decode split is not a boundary a caller could choose,
    # it is where the CIF boundary has to be decided, and a caller who picked otherwise would get a
    # graph that cannot be built.
    decomposition: Decomposition = field(default_factory=MultiPhase)
    _sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    _threshold: Optional[float] = field(default=None, init=False, repr=False)
    _auto = None

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked(
            "the GGUF's own architecture string; it names this export rather than describing the "
            "checkpoint, so there is nothing in the checkpoint to compare it against -- the same "
            "reading `t5_export` gives the same field."
        ),
        "model_dir": Unchecked(
            "path to the FunASR checkpoint directory; the recognizer's detect() already read its "
            "config.yaml and found model.pt beside it, and funasr's AutoModel raises on anything it "
            "cannot build."
        ),
        "decomposition": Unchecked("MultiPhase by construction -- see the field comment"),
        "_sample_rate": Unchecked(
            "READ off the checkpoint's own frontend during phases(), never declared -- it is what the "
            "trace length and the dynamic-axis bounds are derived from."
        ),
        "_threshold": Unchecked(
            "READ off the restored predictor (`CifPredictorV2.threshold`). It reaches the DRIVER, not "
            "the graph: the boundary it defines is decided host-side, which is `CifBoundary`'s whole "
            "subject."
        ),
    }

    def load_model(self):
        from funasr import AutoModel

        print(f"Loading FunASR model from {self.model_dir}...")
        auto = AutoModel(model=self.model_dir, device="cpu", disable_update=True)
        auto.model.eval()
        self._auto = auto
        return auto.model

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        frontend = self._auto.kwargs["frontend"]
        if frontend.cmvn is None:
            raise ValueError(
                f"{self.model_dir} declares no `cmvn_file`, so its frontend has no mean/variance "
                f"normalization to fold into the graph -- see sanm_asr_export, which raises the same "
                f"way for the same reason.")
        self._sample_rate = int(frontend.fs)
        self._threshold = float(model.predictor.threshold)

        width = int(model.encoder.output_size())
        n_samples = int(TRACE_SECONDS * self._sample_rate)
        sample_dim = ct.RangeDim(int(MIN_SECONDS * self._sample_rate),
                                  int(MAX_SECONDS * self._sample_rate))
        # The decode phase's two dynamic axes get their own RangeDims. The frame ceiling is what
        # MAX_SECONDS of audio becomes after the LFR stack (100 frames/s / lfr_n), with margin.
        frame_dim = ct.RangeDim(4, 4000)
        token_dim = ct.RangeDim(MIN_TOKENS, MAX_TOKENS)

        encoder = _ParaformerEncoderPhase(model, KaldiFbankLfrCmvn(
            frontend.cmvn, n_mels=frontend.n_mels, sample_rate=self._sample_rate,
            frame_length=float(frontend.frame_length), frame_shift=float(frontend.frame_shift),
            window_type=frontend.window, lfr_m=frontend.lfr_m, lfr_n=frontend.lfr_n,
            upscale_samples=bool(frontend.upsacle_samples)))

        trace_frames, trace_tokens = 40, 12
        return [
            ExportPhase(
                name="encoder",
                wrapper=encoder,
                dummy_inputs=(torch.randn(1, n_samples, dtype=torch.float32),),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, sample_dim), dtype=np.float32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="decode",
                wrapper=_ParaformerDecodePhase(model),
                dummy_inputs=(torch.randn(1, trace_frames, width, dtype=torch.float32),
                              torch.rand(trace_tokens, trace_frames, dtype=torch.float32)),
                mil_inputs=[
                    ct.TensorType(name="encoder_out", shape=(1, frame_dim, width), dtype=np.float32),
                    ct.TensorType(name="cif_weights", shape=(token_dim, frame_dim), dtype=np.float32),
                ],
                # The phase's OWN axis is the token count -- it is what the output is one row per.
                root_axis="n_tokens",
                # And the frames it cross-attends over are a SECOND dynamic axis: nothing about the
                # token count implies it, and no data-flow path inside this phase relates them.
                # Declared by input name and axis, exactly as `t5_export` declares its cross-attention
                # K/V -- both inputs carry it, which is what keeps them one symbol rather than two.
                declared_axes={
                    "encoder_out": {1: "n_enc_frames"},
                    "cif_weights": {1: "n_enc_frames"},
                },
            ),
        ]

    def driver_components(self) -> List:
        """Encoder once, the CIF boundary host-side, then one non-autoregressive decode.

        Three components and an epilogue, all of them IR. The epilogue is **family 12's**: a
        non-autoregressive decoder emits one token per row in row order, which is the reduction
        `TokenLabelsEpilogue` already performs and exactly not the one `CtcGreedyEpilogue` performs --
        collapsing consecutive duplicates here would delete a legitimately repeated character, and
        Chinese repeats characters.
        """
        from .driver_components import (
            CALLER, CifBoundary, DriverInputs, SubgraphCallComponent, TokenLabelsEpilogue,
        )
        from .lua_library import LuaLibrary
        from .driver_ir import FieldAccess, Len, OutputRef, Var

        waveform = FieldAccess("inputs", "waveform")
        return [
            DriverInputs(bindings=(("waveform", CALLER),), n_tokens=Len("waveform")),
            LuaLibrary(uses=("round_half_to_even", "to_f32", "cif_fire")),
            SubgraphCallComponent(
                topology="encoder",
                outputs=(),
                retain=True,
                inputs={"waveform": Var("waveform")},
                axes={"n_samples": Len("waveform")},
                note="Encoder + CIF predictor: one pass over the caller's waveform.",
            ),
            CifBoundary(encoder_module="encoder", threshold=self._threshold or 1.0),
            SubgraphCallComponent(
                topology="decode",
                outputs=(),
                retain=True,
                inputs={
                    # A retained reference: `T * 512` floats nothing host-side reads. `alphas` does
                    # NOT cross -- the host consumed it into the weights, which is the whole point.
                    "encoder_out": OutputRef("encoder", index=1),
                    # And the boundary, as the answer rather than the question.
                    "cif_weights": FieldAccess("_cif", "weights"),
                },
                axes={"n_tokens": FieldAccess("_cif", "n_tokens"),
                      "n_enc_frames": Var("_n_enc_frames")},
                multiline=True,
                note="One non-autoregressive pass over the fired tokens, cross-attending the encoder.",
            ),
            TokenLabelsEpilogue(retained_module="decode"),
        ]

    def hparams(self) -> dict:
        return {} if self._sample_rate is None else {"sample_rate": self._sample_rate}

    def backend_kwargs(self) -> dict:
        # No `tokenizer_dir`: this checkpoint's decode rule has no reader in the engine yet, and a file
        # that claimed a vocabulary it detokenized wrongly is worse than one that admits it has none.
        # See the module docstring.
        return dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams())


def _is_funasr_paraformer(path: Path) -> bool:
    """A FunASR checkpoint directory whose `config.yaml` declares `model: Paraformer`.

    Named, like the first leaf's recognizer and for the same reason stated there: `Paraformer` and
    `SenseVoiceSmall` share a directory layout and an `encoder: SANMEncoder`, so a structural check on
    the encoder would claim whichever checkpoint it met first for whichever template asked. The
    contextual and streaming Paraformer variants declare their own `model:` names and are deliberately
    NOT claimed -- they add a hotword encoder and a chunked predictor this template does not export.
    """
    cfg = _funasr_config(path)
    return bool(cfg) and cfg.get("model") == "Paraformer"


def _build_funasr_paraformer(path: Path, output_path: str) -> LoomExportConfig:
    return ParaformerExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ParaformerExportConfig,
        recognizers=[
            ModelRecognizer(name="funasr-paraformer", detect=_is_funasr_paraformer,
                            build_config=_build_funasr_paraformer),
        ],
    ))
