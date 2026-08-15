"""Export the real SupertonicTTS v2 checkpoint (`assets/pt/*.pt`, full pickled `nn.Module`s -- `torch.save
(self, path)`, so `torch.load(..., weights_only=False)` returns an ALREADY-CONSTRUCTED real module
directly, no hyperparameter guessing/reconstruction needed, unlike Matcha's checkpoint format) through
`TTSSupertonicExportConfig` (BACKLOG.md P3.3, migrated from `export_supertonic_mil.py`), tracing the REAL
`supertonic_tts.models.modules.*` submodules directly -- not the hand-built bespoke topology
`tools/convert_supertonic/convert_supertonic_*.py` constructs op-by-op via its own `TopologyBuilder` DSL
(that bespoke conversion's own `supertonic_common.py` was invaluable here as an independently-derived
oracle for cross-checking every architectural quirk found while reading source, even though this module
doesn't reuse any of its op-building code).

Four topologies (same names/roles `loom::SupertonicDriver`/`supertonic_driver.lua` already established --
real `SpeechGenerator.predict()` never calls the two style encoders itself, always taking PRECOMPUTED style
embeddings, so those are out of scope here too; see the voice-style note below for what that costs and
what P4.6b did about it):
  - dp:        real `DurationPredictor.forward(txt_ids, stl_emb, txt_msk)` -> scalar duration (seconds).
  - ttl_text:  real `TTLTextEncoder.forward(txt_ids, stl_emb, txt_msk)` -> txt_emb, ne=[T_TEXT,256].
  - vfe:       real `VectorFieldEstimator.compute_velocity(z_t, txt_emb, stl_emb, lat_msk, txt_msk, t)` --
               ONE Euler velocity evaluation (the `z += v*dt` update itself is the Euler sampler
               `TTSFlowMatchingModelExportConfig` generates, matching `supertonic_driver.lua`'s existing
               convention).
  - decoder:   real `SpeechDecoder.forward(latent)` -> flat waveform.

The checkpoint's own grapheme vocabulary travels in the GGUF (`backend_kwargs`, below): the static
`assets/onnx/unicode_indexer.json` codepoint table, which `loom::SupertonicTextVectorizer` reads back, so
this is the one TTS family whose artifact takes text rather than externally-produced phoneme ids. It is
data only -- the preprocessing pipeline around it is the engine's port of the real `TextVectorizer`. That
vocabulary was inspection-only until BACKLOG.md P4.6: the one text width was 10, and `<en>` + the pipeline's
inserted final period + `</en>` is exactly 10 ids for the EMPTY string, so every real sentence overflowed
and synthesis effectively still took ids directly. The axis is PADDED now, and traced at five widths.

Text-length scope limitation (REAL, carried forward from the bespoke conversion, not a new one introduced
here, and NOT merely a `loom::GraphBuilder` restriction -- see below): `T_TEXT` is FIXED at trace/export
time for every topology that touches text. Two INDEPENDENT reasons force this, not one:
  (1) `vfe` needs TWO independently-sized sequences at once (the CFM-iterated latent-frame count `T_lat`,
      and the text length `T_TEXT`) -- `loom::GraphBuilder::build(n_tokens, n_past)` only ever resolves
      ONE dynamic-length symbol per topology, so `T_lat` gets "$n_tokens" and `T_TEXT` must be static.
  (2) `dp`/`ttl_text` independently CAN'T be traced with a dynamic `T_TEXT` at all, regardless of (1) --
      confirmed empirically: `MultiHeadRelativeAttention._get_relative_embeddings`'s relative-position-
      table windowing (`components.py`) pads by a length-DERIVED amount (`pad_len = max(length-(ws+1),
      0)`), which coremltools' own torch frontend explicitly refuses once `length` is genuinely dynamic
      (`NotImplementedError: Dynamic padding for n-dimensional tensors is not supported`) -- a real
      coremltools/MIL limitation, not a gap in this project's own exporter. This is the SAME underlying
      reason the bespoke conversion fixed `T_TEXT` in the first place (its own hand-built rel-pos-attention
      windowing needs a static T to build its lookup tables at all), not a coincidence.
  Net effect: ALL FOUR topologies (`dp`/`ttl_text`/`vfe`/`decoder`) are consistent in using this same fixed
  bucket wherever text length appears (`decoder` doesn't touch text at all, only `T_lat`).

  What P4.6 changed is that the axis being fixed no longer means the TEXT has to be. `txt_msk` is a real
  input, the driver pads `txt_ids` up to the axis and builds the mask, and `_edge_fill` is what makes the
  padding inert -- see its docstring, which is where the measurement lives.

  What P4.6a changed is that there is no longer ONE axis. A static width costs the same on every call
  whatever the real text is, so a single width had to trade "long enough to be useful" against "cheap
  enough to always pay" -- 256 was that compromise, and the trade is what bucketing removes. Each of the
  three text-touching topologies is traced at every width in TEXT_BUCKETS, the driver picks the smallest
  that fits, and the LARGEST is the ceiling. So the ceiling went 256 -> 512 while an ordinary sentence
  got FASTER than it was at 256, which no single number could have done.

  `vfe` is bucketed too, and it is the one that must be: it runs once per CFM step, so leaving it at the
  ceiling would have given back most of what the other two save. That needed `FlowMatchingSpec` to learn
  a computed estimator name (`estimator_variants`) -- a generalization of the shared sampler template,
  not a Supertonic special case.

  Why this ladder: `<lang>` + the inserted final period costs 10 ids flat, so 32 is about 22 real
  characters ("Hi." is 12 ids, "hello world" 21), 128 a sentence (a 44-character one is 53), 512 a short
  paragraph (~490 characters). Doubling keeps the worst-case waste at just under half the axis while
  keeping the ladder short, and each rung costs only its own topology JSON: the GGUF writer dedups by
  CONTENT hash, so every bucket's weights are the same bytes and alias automatically.

  Text longer than the largest bucket is still not synthesizable in one call, and is refused by name
  rather than truncated. Chunking it is a separate question this export deliberately does not open: it
  is a decision about where sentences may be broken, which is preprocessing, not a model contract.

The default voice style travels in the GGUF too (`backend_kwargs`, `DEFAULT_VOICE_STYLE`), as two
`driver_weights` tensors the driver reads with `loom.get_weight` when the caller supplies no style.

That is worth being precise about, because the thing it fixed was easy to misread as the opposite.
`style_ttl`/`style_dp` have ALWAYS been `infer` inputs and a different pair has always selected a
different voice -- what was missing is that a published GGUF carried no styles at all, so every caller
had to have the upstream checkpoint repo to get any. The default makes the artifact usable on its own
and changes nothing about passing one; both paths are gated (P4.6b).

What is still out of scope is deriving a style from your own audio, which is a different thing again:
`SpeechGenerator.encode_voice_style` runs mel -> `SpeechEncoder` -> `lat_compressor` -> the two style
encoders, i.e. three more real modules than this export traces. Selecting among existing voices needs
none of it.

Trace-friendliness patches needed (same category as every prior MIL export in this project):
  - `lat_msk` (always all-ones -- the latent axis is the one axis these topologies size dynamically, so
    it is never padded) constructed via direct arithmetic on an already-real graph tensor
    (`z_t[:,:1,:]*0.0+1.0` -- NOT `torch.ones`/`torch.full`/`ones_like`), same "avoid a separate
    fill-shaped op" reasoning as every prior model's own mask construction. Unlike Matcha's Decoder
    (where every mask multiply was a provable no-op, so the mask was never even constructed),
    SupertonicTTS's masks are genuinely READ (softmax masking via `==0.0`/`masked_fill`, `.sum()` to
    recover fractional-RoPE sequence lengths) -- constructing a real all-ones tensor and letting those
    reads trace as real (structurally harmless, since the comparison is against a tensor of all 1s)
    EQUAL/SELECT/REDUCE_SUM ops is simpler and safer than trying to special-case every read site.
    `txt_msk` was built this same way until P4.6 and is a real traced input now, because the text axis
    is padded (see the text-length section below).
  - `nn.functional.pad(..., mode="replicate")` (every `ConvNextBlock` in this model pads this way before
    its depthwise conv) needed a NEW exporter capability, not a wrapper-level patch: see
    `loom_exporter/exporter.py`'s `pad` translation, `mode == "replicate"` branch -- ggml has no
    native replicate/edge-pad kernel (unlike PAD_1D/PAD_1D_REFLECT), so it's composed purely from
    already-existing primitives (VIEW the boundary column, REPEAT-broadcast it, CONCAT it on) rather than
    adding a new C++ op. Verified standalone (both symmetric and causal-only padding) via a small isolated
    trace/compare against real `F.pad` before ever touching the full model: 0.0 max abs diff.
  - Dilated depthwise conv (`ConvNextBlock`'s `big_convnext` groups, dilation up to 2**3=8) needed NO new
    engine work -- `CONV_1D_DW`'s existing `d0` (dilation) attribute already wraps `ggml_im2col`'s own
    dilation parameter correctly, confirmed by inspection (`src/ops/primitives_conv.cpp`), not newly added
    here.

No import-order stub needed for this checkpoint format -- the four `.pt` files are fully pickled
`nn.Module`s, and `supertonic_tts` is a plain installed package (no version-pin issue like Kokoro's/
Matcha's transformers/huggingface_hub dependencies), so there's no `ModelPatcher` subclass here.

Usage:
  loom-export /path/to/supertonic/assets/pt -o supertonic_mil.gguf --task text-to-speech --model supertonic
"""
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import coremltools as ct

from .paths import CONVERTERS, driver_dir
from .checkpoint_probe import probe_torch_checkpoint
from .flow_matching_export import FlowMatchingSpec
from .multi_phase_export import ExportPhase, TTSFlowMatchingModelExportConfig
from .spec_protocol import Unchecked
from .supertonic_tokenizer_export import INDEXER_RELPATH, find_indexer

# The padded text widths every text-touching topology is traced at. The driver picks the smallest that
# fits the caller's ids, so the LARGEST is the ceiling and the others are what keep short text from
# paying for it (BACKLOG.md P4.6a). One width, 10, until P4.6 -- which is the empty string after the
# `<lang>` wrap -- then a single 256 until P4.6a. See the module docstring for why a width is fixed at
# all, and for how the ladder was chosen.
TEXT_BUCKETS = (32, 64, 128, 256, 512)
T_TEXT_MAX = max(TEXT_BUCKETS)


# The voice style this export falls back to when a caller supplies none, and the tensor names it
# travels under. F1 rather than any other of the ten in `assets/voice_styles/`: it is the one the
# frozen end-to-end reference waveform was recorded with
# (`legacy_driver_reference/supertonic_driver_waveform_F1.npy`), so "call `infer` with no style at all"
# is gated by ground truth that already exists rather than by a new fixture recording this decision.
#
# A DEFAULT, not a restriction -- `style_ttl`/`style_dp` remain ordinary `infer` inputs and override it
# (BACKLOG.md P4.6b). The two are different sizes because they feed different encoders: TTL takes
# (50, 256) and DP takes (8, 16).
DEFAULT_VOICE_STYLE = "F1"
DEFAULT_STYLE_TTL_TENSOR = "loom.default_style.ttl"
DEFAULT_STYLE_DP_TENSOR = "loom.default_style.dp"


def find_voice_style(model_dir: Path, name: str = DEFAULT_VOICE_STYLE) -> Optional[Path]:
    """The `assets/voice_styles/<name>.json` belonging to the checkpoint at `model_dir`, or None.

    Same two spellings of the same place as `find_indexer`, and for the same reason: `model_dir` is
    the `assets/pt` directory, so the asset is a sibling of it -- reached from the repo root two
    levels up, or directly, which is what makes a copied-out `assets/` tree work on its own."""
    for candidate in (model_dir.parent.parent / "assets" / "voice_styles" / f"{name}.json",
                      model_dir.parent / "voice_styles" / f"{name}.json"):
        if candidate.is_file():
            return candidate
    return None


def load_voice_style(path: Path) -> dict:
    """`{tensor name: float32 array}` for one real `assets/voice_styles/*.json`.

    The file is the real `SpeechGenerator.export_voice_style` output -- `{"style_ttl": {"data": [...],
    "dims": [...]}, "style_dp": {...}}` -- so the dims are checked against what the traced graphs
    declare rather than trusted. A style of the wrong shape would otherwise reach `loom.get_weight`,
    come back as a flat array of the wrong length, and fail deep inside the engine as a shape mismatch
    on an input the caller never passed."""
    import json

    with path.open() as f:
        style = json.load(f)
    out = {}
    for field, tensor_name, dims in (("style_ttl", DEFAULT_STYLE_TTL_TENSOR, [1, 50, 256]),
                                      ("style_dp", DEFAULT_STYLE_DP_TENSOR, [1, 8, 16])):
        entry = style[field]
        if list(entry["dims"]) != dims:
            raise ValueError(f"{path}: {field} has dims {entry['dims']}, expected {dims} -- this is "
                             f"the shape the traced graphs declare for it.")
        out[tensor_name] = np.asarray(entry["data"], dtype=np.float32).reshape(-1)
    return out


def bucket_topology(prefix: str, t_text: int) -> str:
    """`"ttl_text", 64 -> "ttl_text_64"`. One function so the exported name and the name the driver
    computes are the same rule rather than two string formats that agree today."""
    return f"{prefix}_{t_text}"

# What the driver writes into the padded tail of `txt_ids`. Which id this is does not affect a single
# output value -- `x = x * txt_msk` zeroes every padded position's embedding before anything reads it,
# and ids 0/1/162 were measured to give bit-identical `txt_emb` and duration (BACKLOG.md P4.6). It is
# 162 because that is the vocabulary's one unused row: `SupertonicTextVectorizer::n_tokens()` is 162
# against an `nn.Embedding(163)`, so no codepoint maps to it and a dump of the padded ids reads
# unambiguously as "these are padding".
PAD_ID = 162

# The output sample rate of the SupertonicTTS v2 release. Unlike every other number this export binds
# into the driver, this one genuinely is NOT in the checkpoint: the four `.pt` files are pickled
# `nn.Module`s and none of them carries it -- it lives in `supertonic_tts.lightning`'s own
# `dec_sample_rate: int = 44100` default, i.e. in the training config, which is not shipped. So it is
# declared here rather than read, and it is declared HERE rather than restated by every host, which is
# the whole point of P4.0.8's first follow-up. A release at a different rate needs this line changed.
DEC_SAMPLE_RATE = 44100.0


class _TextMask:
    """The mask the patched `ConvNextBlock`s below read, set by the wrapper before it calls the real
    module.

    A holder rather than an argument because the mask has to reach code this export does not call:
    `DPTextEncoder`/`TTLTextPreEncoder` invoke their blocks as `block(x)` and apply the mask
    themselves afterwards (`x = block(x) * txt_msk`), so there is no argument to thread. The
    alternative was reimplementing both encoders' `forward` in a wrapper, which would put a copy of
    real checkpoint code in this file and make every future divergence silent."""

    __slots__ = ("mask",)

    def __init__(self):
        self.mask = None


def _edge_fill(x, msk):
    """Replace `x`'s padded tail with a copy of its last REAL column: (B,C,T), (B,1,T) -> (B,C,T).

    This is the whole of P4.6's correctness argument, so it is worth stating what it is for. A
    `ConvNextBlock` pads with `mode="replicate"` before its depthwise conv, and that conv is the only
    op in the block that reads across positions. On a MASKED tensor the "edge" it replicates is a zero
    column, whereas the reference implementation -- which runs at T = the real length, never padded,
    because `TextVectorizer.tokenize` pads only to the longest string in a batch and synthesis is a
    batch of one -- replicates the last real column. That difference is not small: measured in
    PyTorch, ten real ids padded to 256 moved `txt_emb` by 1.77 max-abs against a tensor whose own max
    is 1.82, and the predicted duration by 0.17%. Filling the tail this way makes every real position's
    conv window byte-for-byte what the unpadded run sees, which takes `txt_emb` to 2.6e-05 -- fp32
    reduction-order noise from the wider matmuls, not a residual mechanism.

    No gather, and that is deliberate: the last real index is data-dependent, but `msk` minus itself
    shifted left is one-hot at exactly that column, so `(x * edge).sum()` selects it with a multiply
    and a reduction the exporter already lowers. At an all-ones mask `edge` is one-hot at the last
    column and `1 - msk` is zero, so this is the identity -- confirmed to 0.0 max-abs before it was
    ever exported, which is why it costs the T_TEXT = 10 references nothing."""
    msk_next = torch.nn.functional.pad(msk[:, :, 1:], (0, 1))
    edge = msk - msk_next                                 # one-hot at the last real column
    last = (x * edge).sum(dim=2, keepdim=True)            # (B, C, 1)
    return x * msk + last.expand(-1, -1, x.shape[-1]) * (1.0 - msk)


def _edge_fill_convnext_forward(self, x, msk=None):
    """`ConvNextBlock.forward` with a mask-aware replicate pad -- see `_edge_fill`.

    A line-for-line copy of the real `components.ConvNextBlock.forward`'s `msk is None` path with one
    statement inserted, rather than a reimplementation: the two must not drift, and the diff being one
    line is what makes that checkable by reading. Only the TEXT encoders' blocks are patched (see
    `_patch_text_convnext`); the VectorFieldEstimator's own blocks take a real `msk` argument over the
    latent axis, which this export sizes dynamically and therefore never pads."""
    assert msk is None, "only the text encoders' blocks are patched, and they pass no mask"
    m = self._loom_text_mask.mask
    residual = x
    y = _edge_fill(x, m)  # <-- the only line the real ConvNextBlock does not have
    y = torch.nn.functional.pad(y, self.pad, mode="replicate")
    y = self.dwconv(y)
    y = y.transpose(1, 2)
    y = self.norm(y)
    y = y.transpose(1, 2)
    y = self.pwconv1(y)
    y = self.act(y)
    y = self.pwconv2(y)
    y = self.gamma * y
    return residual + y


def _patch_text_convnext(blocks, holder) -> None:
    """Give every block in `blocks` the mask-aware pad, reading its mask from `holder`.

    Per-INSTANCE (`types.MethodType`), not on the class: patching `ConvNextBlock.forward` would also
    catch the VectorFieldEstimator's blocks, which are correct as they stand. Per-instance also leaves
    the module tree and its state-dict paths untouched, which a wrapper module would not -- and those
    paths are the exported tensor names."""
    for block in blocks:
        block._loom_text_mask = holder
        block.forward = types.MethodType(_edge_fill_convnext_forward, block)


def _ones_mask_from_float(x):
    """(B,C,T) float -> (B,1,T) float, all-ones, derived via arithmetic on `x` itself (not torch.ones/
    ones_like/full) -- see module docstring.

    Only `lat_msk` is still built this way. The latent axis is the ONE dynamic axis these topologies
    have, so it is never padded and its mask is all-ones by construction; `txt_msk` was built the same
    way until P4.6 and is a real traced input now, because the text axis IS padded."""
    return x[:, :1, :] * 0.0 + 1.0


class DPWrapper(torch.nn.Module):
    """Real `DurationPredictor.forward` (assets/pt/duration_predictor.pt IS this class directly, not
    `DurationPredictorWrapper` -- confirmed via `tools/convert_supertonic/reference_forward_supertonic_dp.
    py`'s own usage, `dp(txt_ids, stl_emb, txt_msk)`). `stl_emb` (the DP style) is a precomputed input,
    matching `supertonic_driver.lua`'s own "dp" call (`stl_emb = inputs.style_dp`) -- `DPStyleEncoder` is
    out of scope here, same "basic synthesis from a precomputed style" precedent as every other model.

    `txt_msk` is a real forward argument, which the real module already accepted: P4.6 un-faked it
    rather than inventing it.
    """
    def __init__(self, dp):
        super().__init__()
        self.dp = dp
        self.text_mask = _TextMask()
        _patch_text_convnext(dp.sentence_encoder.convnext, self.text_mask)

    def forward(self, txt_ids, stl_emb, txt_msk):
        # `DPTextEncoder` prepends a learned `sentence_token` before its ConvNext stack and masks with
        # the (B,1,T+1) `full_mask` that results, so that -- not `txt_msk` -- is what its blocks see.
        # Restated here rather than reached into because it is one `cat`, and derived from `txt_msk`
        # rather than built with `torch.ones` for the same reason `_ones_mask_from_float` is.
        one = txt_msk[:, :, :1] * 0.0 + 1.0
        self.text_mask.mask = torch.cat([one, txt_msk], dim=2)
        duration = self.dp(txt_ids, stl_emb, txt_msk)  # (1,)
        return duration.reshape(-1)


class TTLTextWrapper(torch.nn.Module):
    """Real `TTLTextEncoder.forward` (assets/pt/text_encoder.pt). `stl_emb` (the TTL style, (1,50,256)) is
    precomputed, matching `supertonic_driver.lua`'s own "ttl_text" call -- `TTLStyleEncoder` out of scope.
    """
    def __init__(self, te):
        super().__init__()
        self.te = te
        self.text_mask = _TextMask()
        _patch_text_convnext(te.text_encoder.convnext, self.text_mask)

    def forward(self, txt_ids, stl_emb, txt_msk):
        self.text_mask.mask = txt_msk
        txt_emb = self.te(txt_ids, stl_emb, txt_msk)  # (1, 256, T_TEXT)
        return txt_emb.squeeze(0)  # (256, T_TEXT) -> ggml ne=[T_TEXT,256], T-fast


class VFEWrapper(torch.nn.Module):
    """Real `VectorFieldEstimator.compute_velocity` (assets/pt/vector_estimator.pt) -- ONE Euler velocity
    evaluation; the `z += v*dt` update itself is a Lua/host-side loop (`supertonic_driver/`), same
    split as the bespoke `supertonic_driver.lua`. `txt_emb`'s own T axis is FIXED at trace time
    (one of TEXT_BUCKETS) -- see module docstring for why. `t`: (1,) float fractional step in [0,1).

    `txt_msk` is a real input here for a second reason on top of DP's/TTL's: `txt_emb`'s padded columns
    are zero, so a mask synthesized from `txt_emb` would be all-ones no matter how much of it is
    padding, and `VFTextCrossAttention` reads the mask twice -- once to `masked_fill` the attention
    scores, once as `txt_len = txt_msk.sum()` for its fractional RoPE (`vector_field_estimator.py`).
    """
    def __init__(self, vfe):
        super().__init__()
        self.vfe = vfe

    def forward(self, z_t, txt_emb, stl_emb, t, txt_msk):
        lat_msk = _ones_mask_from_float(z_t)
        v = self.vfe.compute_velocity(z_t, txt_emb, stl_emb, lat_msk, txt_msk, t)  # (1, 144, L)
        return v.squeeze(0)  # (144, L) -> ggml ne=[L,144], T-fast


class DecoderWrapper(torch.nn.Module):
    """Real `SpeechDecoder.forward` (assets/pt/vocoder.pt), unmodified -- no masking anywhere in this
    module at all (pure causal-conv stack + folded BatchNorm + PReLU head), so no trace-friendliness
    patch needed here."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latent):
        wav = self.dec(latent)  # (1, T*6*512)
        return wav.reshape(-1)


def _load_pt(pt_dir: Path, name: str):
    mod = torch.load(pt_dir / name, weights_only=False, map_location="cpu")
    mod.eval()
    return mod


@dataclass(kw_only=True)
class TTSSupertonicExportConfig(TTSFlowMatchingModelExportConfig):
    """SupertonicTTS's own four-phase split (dp/ttl_text/vfe/decoder) plus the Euler CFM sampler over
    `vfe` -- see module docstring. `model_dir` is the `assets/pt` directory containing all four real
    checkpoint files."""

    model_dir: str

    # Derived from the real SpeechDecoder in `phases()` and bound into the driver as `ExportConstants`
    # (P4.0.8's first follow-up). Not constructor arguments: the module states them.
    lat_dim: Optional[int] = field(default=None, init=False, repr=False)
    compression_factor: Optional[int] = field(default=None, init=False, repr=False)
    base_chunk_size: Optional[int] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "the assets/pt directory holding all four .pt files. path to the real checkpoint(s). The recognizer's own detect() already established the structure this config depends on -- it probes the checkpoint's pickle opcodes without unpickling (checkpoint_probe) rather than trusting the filename -- and phases() raises on anything it cannot load. A 'this path exists' link would check the weaker property while reading as if it checked the stronger one."
        ),
        "lat_dim": Unchecked(
            "DERIVED in phases() from the restored SpeechDecoder's own `lat_channels * n_codebooks` "
            "(24 * 6), so there is no second authority to compare it against -- it IS the authority. "
            "It is checked hard and immediately anyway: the same number shapes the `vfe` and `decoder` "
            "phases this config traces, so a wrong one fails the trace"
        ),
        "compression_factor": Unchecked(
            "same -- the decoder's own `n_codebooks`, which is exactly the factor its forward "
            "interleaves codebooks into time by ((B,144,T) -> (B,24,T*6))"
        ),
        "base_chunk_size": Unchecked(
            "same -- the decoder head's own output width (`head_layer2.out_channels`), the samples each "
            "of those T*6 steps expands to. Nothing else in the tree states it, which is why it is "
            "read off the module rather than restated: `latent_size = base_chunk_size * "
            "compression_factor` is what turns a duration in seconds into a latent frame count"
        ),
    }
    # A DIRECTORY of `.lua` fragments -- Supertonic is peeled (P4.0.6/C.5). See `driver_components`.
    driver_script_path: Path = driver_dir("convert_supertonic", "supertonic_driver")

    def driver_components(self) -> List:
        """Supertonic's driver, as components (P4.0.6/C.5).

        Deliberately the second family peeled, and the point is that nothing new was written for it:
        three `SubgraphCallComponent`s, one `FlowMatchingSampler`, two `LuaFragment`s, one
        `DriverReturn` -- the same six classes Matcha uses, differing only in data. That is the reuse
        claim of P4.0.7's "marketplace" tested rather than asserted, and it is why the plan orders
        Supertonic straight after Matcha instead of after the harder families.

        The one Lua block that survives is the real reason it survives: `get_latent_mask` turns a
        predicted duration in seconds into a latent frame count, which is arithmetic on scalars the
        engine never sees."""
        from .driver_components import (
            DriverReturn, ExportConstants, FlowMatchingSampler, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import BinOp, FieldAccess, Lit, Var

        fragment = self.driver_script_path
        t_text, t_lat, lat_dim = Var("T_TEXT"), Var("t_lat"), Var("LAT_DIM")
        txt_ids, txt_msk = Var("txt_ids"), Var("txt_msk")
        sampler, = self.samplers()
        # `"dp_" .. T_TEXT` and friends: the bucket the fragment picked, turned into the topology name
        # by the same rule `bucket_topology` used to export it. `variants` states every name this can
        # produce, so each is checked against a real topology and its declared inputs.
        def bucket_call(prefix):
            return BinOp("..", Lit(f"{prefix}_"), t_text), tuple(
                bucket_topology(prefix, b) for b in TEXT_BUCKETS)

        dp_expr, dp_names = bucket_call("dp")
        ttl_expr, ttl_names = bucket_call("ttl_text")
        vfe_expr, _ = bucket_call("vfe")
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            # The numbers the caller used to have to supply (P4.0.8's first follow-up). The three the
            # SpeechDecoder states about itself, the release's sample rate -- which is genuinely not
            # in the checkpoint (see DEC_SAMPLE_RATE) -- PAD_ID, which joined them in P4.6 when the
            # text axis started being padded, and TEXT_BUCKETS, which joined in P4.6a when there
            # stopped being one text width. T_TEXT is no longer among them: it is chosen per call now.
            ExportConstants(values={
                "TEXT_BUCKETS": list(TEXT_BUCKETS),
                "PAD_ID": PAD_ID,
                "LAT_DIM": self.lat_dim,
                "SAMPLE_RATE": DEC_SAMPLE_RATE,
                "BASE_CHUNK_SIZE": self.base_chunk_size,
                "COMPRESSION_FACTOR": self.compression_factor,
            }),
            LuaFragment(fragment / "01_lengths.lua"),
            LuaFragment(fragment / "01_text_inputs.lua", reads=("TEXT_BUCKETS", "PAD_ID"),
                        defines=("n_txt", "T_TEXT", "txt_ids", "txt_msk")),
            LuaFragment(fragment / "01_styles.lua", defines=("style_ttl", "style_dp")),
            SubgraphCallComponent(
                topology=bucket_topology("dp", T_TEXT_MAX), topology_expr=dp_expr, variants=dp_names,
                outputs=("dur_arr",), length=t_text,
                inputs={"txt_ids": txt_ids, "stl_emb": Var("style_dp"),
                        "txt_msk": txt_msk},
                note="--- DurationPredictor: DPTextEncoder + MLP head -> scalar duration (seconds) ---"),
            LuaFragment(fragment / "02_latent_length.lua",
                        reads=("dur_arr", "SAMPLE_RATE", "BASE_CHUNK_SIZE", "COMPRESSION_FACTOR"),
                        defines=("duration", "wav_length", "latent_size", "t_lat")),
            SubgraphCallComponent(
                topology=bucket_topology("ttl_text", T_TEXT_MAX), topology_expr=ttl_expr,
                variants=ttl_names, outputs=("txt_emb",), length=t_text,
                inputs={"txt_ids": txt_ids, "stl_emb": Var("style_ttl"),
                        "txt_msk": txt_msk},
                note="--- TTLTextEncoder -> txt_emb, ne=[t_text,txt_dim] (T-fast, the traced module's\n"
                     "    own native torch layout -- no host-side layout crossing needed, unlike the\n"
                     "    bespoke driver's Layout A/B bridging, since \"vfe\" was traced expecting\n"
                     "    exactly this same layout for its own txt_emb input). ---"),
            FlowMatchingSampler(
                spec=sampler, result="z", length=t_lat, estimator=vfe_expr,
                n_elems=BinOp("*", t_lat, lat_dim), n_steps=FieldAccess("inputs", "n_steps"),
                step_inputs={"txt_emb": Var("txt_emb"), "stl_emb": Var("style_ttl"),
                             "txt_msk": txt_msk},
                note="--- Deterministic Euler CFM sampling over VectorFieldEstimator, at the same\n"
                     "    text bucket the two encoders above ran at -- see sample_vfe above. ---"),
            SubgraphCallComponent(
                topology="decoder", outputs=("waveform",), length=t_lat,
                inputs={"latent": Var("z")},
                note="--- SpeechDecoder: z (ne=[t_lat,lat_dim]) -> raw waveform ---"),
            DriverReturn(values=("waveform",)),
        ]

    def phases(self) -> List[ExportPhase]:
        pt_dir = Path(self.model_dir)
        print(f"Loading SupertonicTTS checkpoints from {pt_dir}...")
        dp = _load_pt(pt_dir, "duration_predictor.pt")
        te = _load_pt(pt_dir, "text_encoder.pt")
        vfe = _load_pt(pt_dir, "vector_estimator.pt")
        dec = _load_pt(pt_dir, "vocoder.pt")

        # The driver's constants, taken off the decoder rather than restated (P4.0.8's first
        # follow-up). Its forward is the authority on all three: it reshapes (B, lat_channels *
        # n_codebooks, T) into (B, lat_channels, T * n_codebooks) and its head then expands each of
        # those steps to `output_dim` samples -- so the latent width, the compression factor and the
        # chunk size are the module's own attributes, not numbers a host should be asked for.
        self.lat_dim = int(dec.lat_channels) * int(dec.n_codebooks)
        self.compression_factor = int(dec.n_codebooks)
        self.base_chunk_size = int(dec.head_layer2.out_channels)

        torch.manual_seed(0)
        dummy_dp_stl = torch.randn(1, 8, 16)
        dummy_ttl_stl = torch.randn(1, 50, 256)
        lat_seq_dim = ct.RangeDim(1, 512)
        dummy_L = 9
        dummy_z = torch.randn(1, 144, dummy_L)
        dummy_t = torch.tensor([0.3])

        # One wrapper per module, reused across every bucket. The wrappers patch their encoders'
        # ConvNext blocks in `__init__` (`_patch_text_convnext`), and patching the same block twice
        # would nest the patched forward inside itself -- so building three wrappers and tracing each
        # five times is not merely tidier here, it is the only correct order.
        dp_wrapper = DPWrapper(dp).eval()
        ttl_wrapper = TTLTextWrapper(te).eval()
        vfe_wrapper = VFEWrapper(vfe).eval()

        phases: List[ExportPhase] = []
        for t_text in TEXT_BUCKETS:
            print(f"Tracing DurationPredictor / TTLTextEncoder / VectorFieldEstimator at T_TEXT={t_text}...")
            dummy_txt_ids = torch.randint(1, 163, (1, t_text), dtype=torch.int64)
            # An all-ones dummy mask traces the same graph a padded one does -- every read of it is a
            # real op either way (multiply, `== 0.0`, `.sum()`), none of them shape-dependent. What the
            # dummy's CONTENT decides is nothing; what its SHAPE decides is the bucket.
            dummy_txt_msk = torch.ones(1, 1, t_text)
            dummy_txt_emb = torch.randn(1, 256, t_text)
            txt_msk_input = ct.TensorType(name="txt_msk", shape=(1, 1, t_text), dtype=np.float32)
            txt_ids_input = ct.TensorType(name="txt_ids", shape=(1, t_text), dtype=np.int32)

            phases += [
                ExportPhase(
                    name=bucket_topology("dp", t_text), wrapper=dp_wrapper,
                    dummy_inputs=(dummy_txt_ids, dummy_dp_stl, dummy_txt_msk),
                    mil_inputs=[
                        txt_ids_input,
                        ct.TensorType(name="stl_emb", shape=(1, 8, 16), dtype=np.float32),
                        txt_msk_input,
                    ],
                ),
                ExportPhase(
                    name=bucket_topology("ttl_text", t_text), wrapper=ttl_wrapper,
                    dummy_inputs=(dummy_txt_ids, dummy_ttl_stl, dummy_txt_msk),
                    mil_inputs=[
                        txt_ids_input,
                        ct.TensorType(name="stl_emb", shape=(1, 50, 256), dtype=np.float32),
                        txt_msk_input,
                    ],
                ),
                ExportPhase(
                    name=bucket_topology("vfe", t_text), wrapper=vfe_wrapper,
                    dummy_inputs=(dummy_z, dummy_txt_emb, dummy_ttl_stl, dummy_t, dummy_txt_msk),
                    mil_inputs=[
                        ct.TensorType(name="z_t", shape=(1, 144, lat_seq_dim), dtype=np.float32),
                        ct.TensorType(name="txt_emb", shape=(1, 256, t_text), dtype=np.float32),
                        ct.TensorType(name="stl_emb", shape=(1, 50, 256), dtype=np.float32),
                        ct.TensorType(name="t", shape=(1,), dtype=np.float32),
                        txt_msk_input,
                    ],
                ),
            ]

        print("Tracing SpeechDecoder...")
        dec_seq_dim = ct.RangeDim(1, 512)
        dummy_latent = torch.randn(1, 144, 4)
        # Not bucketed, and the only one of the four that isn't: it never touches the text axis, only
        # `T_lat`, which is dynamic.
        phases.append(ExportPhase(
            name="decoder", wrapper=DecoderWrapper(dec).eval(), dummy_inputs=(dummy_latent,),
            mil_inputs=[ct.TensorType(name="latent", shape=(1, 144, dec_seq_dim), dtype=np.float32)],
        ))

        return phases

    def backend_kwargs(self) -> dict:
        """The checkpoint's own grapheme vocabulary travels with the model, so the artifact has a text
        door on its own -- the one capability the standalone
        `tools/convert_supertonic/convert_supertonic_text_vectorizer.py` GGUF had that this MIL export did
        not. Same reasoning, and the same shape, as the NeMo families carrying their SentencePiece protobuf
        (`nemo_asr_export.py`, `transducer_export.py`).

        `tokenizer_family` is named rather than detected: `unicode_indexer.json` is not an HF tokenizer
        directory, so `tokenizer_detect.detect_vocab_family` would find no file it recognizes and raise.

        Warned-and-omitted rather than raised when the asset is missing, matching the NeMo families' own
        `if tokenizer_dir is not None`. The four traced graphs are the export; the vocabulary is a real
        improvement to it, but its absence makes the file worse rather than wrong -- and this method is
        also called by callers that never trace (`component_registry.usage()` builds every registered
        config). The warning is what keeps that from being silent, because the asset lives in a directory
        NEXT TO the one the caller names, and is easy to leave behind when copying a checkpoint out.
        """
        kwargs = {"hparams": self.hparams()}
        indexer = find_indexer(Path(self.model_dir))
        if indexer is not None:
            kwargs["tokenizer_dir"] = str(indexer.parent)
            kwargs["tokenizer_family"] = "supertonic"
        else:
            print(f"warning: no {INDEXER_RELPATH} found near {self.model_dir} -- exporting without a "
                  f"tokenizer, so this GGUF will take txt_ids only and `model.tokenizer` will be None",
                  file=sys.stderr)

        # The default voice style, on exactly the same terms as the vocabulary above: an asset from
        # NEXT TO the directory the caller names, warned-and-omitted when absent because its absence
        # makes the file worse rather than wrong. The four traced graphs are still the export.
        #
        # Without it this artifact cannot synthesize AT ALL on its own. `style_ttl`/`style_dp` have
        # always been `infer` inputs, so different styles were always possible -- but the values had to
        # come from the checkpoint repo, which a published GGUF is not distributed with, so the honest
        # description of the old state is "every caller must supply a style and nothing here provides
        # one" (BACKLOG.md P4.6b).
        style_path = find_voice_style(Path(self.model_dir))
        if style_path is not None:
            kwargs["driver_weights"] = load_voice_style(style_path)
        else:
            print(f"warning: no assets/voice_styles/{DEFAULT_VOICE_STYLE}.json found near "
                  f"{self.model_dir} -- exporting without a default voice style, so every `infer` call "
                  f"on this GGUF must supply style_ttl and style_dp itself", file=sys.stderr)
        return kwargs

    def hparams(self) -> dict:
        """The one number a HOST cannot proceed without: the MOST `txt_ids` this export accepts.

        Every text-touching topology here was traced at a FIXED text length (see the module
        docstring's two independent reasons), so a caller that sends more than this is calling a model
        that cannot run -- and the only place that used to say so was a comment in the driver and a
        literal in a C++ test header. Declaring it in the file is what makes the constraint checkable
        by whoever is actually building the input.

        It was an EXACT count until P4.6, when the driver started padding: `txt_len` is now a ceiling
        rather than a requirement, and every host that read it as the latter was rejecting text it can
        now synthesize. The name did not change, because what a host does with it did not: compare its
        id count against this number. Only the comparison did, from `==` to `<=`.

        P4.6a made the text axis bucketed, and this stayed ONE number for the same reason: a host's
        question is still "will my ids fit", and the answer is still the largest width. Which bucket a
        call lands in is the driver's business and deliberately not a host's -- publishing the ladder
        here would invite callers to pad to a bucket themselves, which is exactly the job the driver
        took over. It is readable anyway, by anyone who wants it, in the embedded driver's own
        `TEXT_BUCKETS` local."""
        return {"txt_len": T_TEXT_MAX}

    def contract(self) -> dict:
        """Supertonic takes TEXT, which is what makes it the exception in its own task.

        The `text-to-speech` default is `phoneme_ids`, because four of the five families in it consume
        ids a phonemiser produces outside the engine and have no vocabulary embedded to do otherwise.
        This one encodes graphemes itself -- its `TextVectorizer` is a unicode codepoint table that
        ships in the GGUF -- so declaring the task default here would be stating something false about
        the model, and a host reading it would refuse the text door this model actually has.

        Caught by the export sweep, which is the only thing that would have: every unit test on the
        default passed, and the wrong value is only visible next to the model it is wrong about.
        """
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        return contract

    def samplers(self) -> List[FlowMatchingSpec]:
        # Same shared family as Matcha's own sampler (EXPORT-IMPROVEMENT.md item 4) -- only the
        # estimator, the loop-carried input's name, and the per-step-constant inputs differ. Since
        # P4.6a the estimator is a SET rather than a name, because `vfe` is traced once per text
        # bucket; `estimator` is the canonical member every other link checks against, and the driver
        # passes the one it picked. Matcha's own sampler is unaffected -- it declares no variants and
        # its generated Lua is unchanged.
        return [FlowMatchingSpec(
            func_name="sample_vfe",
            estimator=bucket_topology("vfe", T_TEXT_MAX),
            estimator_variants=tuple(bucket_topology("vfe", b) for b in TEXT_BUCKETS),
            carried_input="z_t",
            time_input="t",
            fixed_inputs=["txt_emb", "stl_emb", "txt_msk"],
            note="Deterministic Euler CFM sampling over SupertonicTTS's VectorFieldEstimator:\n"
                 "z <- z + v(z, txt_emb, stl_emb, t) * dt, uniform dt = 1/n_steps.",
        )]


# Exactly the four `_load_pt` calls in `TTSSupertonicExportConfig.phases()` -- the style-encoder `.pt`
# files that sit beside them in the real `assets/pt` directory are not loaded by this export and are
# deliberately not required here.
_SUPERTONIC_REQUIRED_PT = ("duration_predictor.pt", "text_encoder.pt", "vector_estimator.pt", "vocoder.pt")


def _is_supertonic(path: Path) -> bool:
    """Real structural check (BACKLOG.md P4.0.1): the `assets/pt` directory `TTSSupertonicExportConfig.
    phases()` requires -- all four checkpoints present, and one of them really being a pickled
    SupertonicTTS module.

    The strongest signature of the five TTS families, and the reason is this family's otherwise
    inconvenient checkpoint format: these are `torch.save(module)` outputs, not state dicts, so the
    pickle names the real class it will reconstruct (`supertonic_tts.models.modules.
    text_to_latent_encoding.encoders.TTLTextEncoder`). Reading that reference is not the same as
    honoring it -- `probe_torch_checkpoint` walks pickle opcodes and never unpickles, so detection needs
    neither `torch.load` nor the `supertonic_tts` package importable."""
    if not path.is_dir() or not all((path / name).is_file() for name in _SUPERTONIC_REQUIRED_PT):
        return False
    probe = probe_torch_checkpoint(path / "text_encoder.pt")
    if probe is None:
        return False
    return any(ref.startswith("supertonic_tts.") for ref in probe.globals)


def _build_supertonic(path: Path, output_path: str) -> TTSSupertonicExportConfig:
    return TTSSupertonicExportConfig(architecture="supertonic_mil", output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=TTSFlowMatchingModelExportConfig,
        recognizers=[ModelRecognizer(name="supertonic", detect=_is_supertonic, build_config=_build_supertonic)],
    ))
