"""The audio-encoder + AR-cross-attention-decoder family (`EXPORT-ROADMAP.md` R5's family 2), on
Whisper -- BACKLOG.md P4.1.

**What is new here, and what is deliberately not.** Every earlier family either runs its model once
(`Flattened`: Qwen3, the NeMo encoders) or runs N phases in a fixed order with the loop over *data*
(`MultiPhase`: the TTS zoo, Parakeet's transducer). This family is the first whose second phase is a
KV-cached transformer decoding against a *first* phase's output -- an encoder run once, then a decode
loop that attends to its result at every step. That is one new fact for the export to carry (which
phase's attention is cached) and one new fact for the driver (which input holds the encoder's output);
everything else is the machinery those two families already have, which is why this module declares
phases and components rather than bringing a fourth `Decomposition`. See BACKLOG.md P4.1 for the
measurement behind that decision.

**The mel frontend is part of the exported graph, not the host's problem.** HF's `WhisperEncoder`
starts at `input_features` -- a log-mel spectrogram computed by `WhisperFeatureExtractor`, which is
numpy/torch code outside the model. Every other audio family in this tree traces its own frontend
(NeMo's `AudioToMelSpectrogramPreprocessor` is a real `nn.Module` and is traced with the encoder), and
the bespoke converter this replaces built the same mel in-graph from DFT-as-convolution kernels. So
`WhisperMelFrontend` reimplements the feature extractor's own arithmetic as a traceable module and the
exported encoder takes a **waveform**, keeping the "a host hands the engine audio, not features"
contract that makes the GGUF self-contained.

**The decoder is traced without a cache and decodes with one**, which is KV-CACHE.md's finding applied
unchanged: `fuse_loom_attention` turns each self-attention block into an `ATTENTION` node, the engine
supplies the past, and a decode step is the same graph at `n_tokens = 1`. Cross-attention is *not*
fused, and that is the correct outcome rather than a gap -- see `ASRWhisperExportConfig.phases`.
"""
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


class WhisperMelFrontend(nn.Module):
    """`WhisperFeatureExtractor`'s log-mel spectrogram, as a traceable `nn.Module`.

    Numerically identical to the extractor, not merely equivalent: the filterbank is the extractor's
    own `mel_filters` array (read off the checkpoint's `preprocessor_config.json`, never recomputed
    here), and the arithmetic below is `_torch_extract_fbank_features` line for line -- Hann-windowed
    STFT, drop the final frame, power spectrum, filterbank, log10 with a 1e-10 floor, clamp to 8 dB
    below the clip's own maximum, then `(x + 4) / 4`.

    **The global maximum is why this cannot be simplified.** `torch.maximum(log_spec, log_spec.max() -
    8)` makes every output element depend on the loudest bin in the whole 30 s clip, so the frontend is
    not separable per frame and cannot be pushed onto the host per chunk without changing the numbers.

    Takes `(1, n_samples)` rather than `(n_samples,)`: a 1-D input makes `torch.stft` trace an
    `aten::size` -> `aten::Int` chain over the sample axis, and the batch axis avoids it entirely.
    """

    def __init__(self, n_fft: int, hop_length: int, mel_filters: np.ndarray):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # The extractor stores `(n_freq, n_mels)`; the matmul below wants `(n_mels, n_freq)`.
        self.register_buffer(
            "filters", torch.from_numpy(np.asarray(mel_filters, dtype=np.float32)).T.contiguous()
        )

    def forward(self, waveform):
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=True, return_complex=True,
        )
        # Whisper drops the last STFT frame, so a 30 s clip yields 3000 frames and not 3001. The
        # extractor writes this as `stft[..., :-1].abs() ** 2`; taking the magnitude FIRST is the same
        # arithmetic on one fewer element and is the only order that converts -- coremltools' complex
        # dialect has no `slice_by_index` over a complex tensor, so slicing before `abs()` fails with
        # `expects tensor ... but got tensor[1, 201, 3001, complex64]`.
        magnitudes = (stft.abs() ** 2)[..., :-1]
        mel_spec = self.filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


class _WhisperEncoderWrapper(nn.Module):
    """`waveform -> encoder hidden states`, the one tensor the encoder phase exports.

    The mel frontend is inside the wrapper rather than beside it precisely so the traced graph contains
    it: this is the module docstring's "self-contained GGUF" claim in code.
    """

    def __init__(self, model, mel: WhisperMelFrontend):
        super().__init__()
        self.mel = mel
        self.encoder = model.model.encoder

    def forward(self, waveform):
        return self.encoder(self.mel(waveform)).last_hidden_state


class _CrossKvSlot(nn.Module):
    """Stands in for a cross-attention `k_proj`/`v_proj` and returns a tensor handed in from outside.

    The projection it replaces is a function of the ENCODER output alone, so its result is identical at
    every decode step. Traced as-is it is recomputed per token: for whisper-small that is 24 matmuls of
    [1500, 768] x [768, 768] every step, which measured **57.7% of the whole transcription** and made
    `MUL_MAT 768x1500` run 684 times for an 11-second clip where the encoder itself needs 48.

    Substituting the module rather than editing `WhisperAttention.forward` keeps this independent of how
    transformers spells its attention this month, and the substitution is BIT-EXACT -- the same tensor
    reaches the same consumer, computed once instead of N times.
    """

    def __init__(self, holder: list, key: int):
        super().__init__()
        # A plain list, deliberately not a buffer or submodule: it carries per-CALL tensors, and
        # registering them would make them state of the traced module.
        self._holder = holder
        self._key = key

    def forward(self, x):
        return self._holder[self._key]


class _WhisperCrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`, the cross-attention K/V for every decoder layer, once.

    **Construct this BEFORE `_WhisperDecoderWrapper`.** It holds the projection modules directly, so
    that the decoder wrapper may replace the `layer.encoder_attn.k_proj` attributes that used to reach
    them without this phase losing the weights it exports. Built the other way round it would trace
    `_CrossKvSlot`s and export nothing.

    Interleaved k,v per layer rather than all-k-then-all-v so the driver's `index` arithmetic is
    `2 * layer + 1` and reads as the pair it is.

    **The V half leaves here already head-split and TRANSPOSED**, `[1, heads, head_dim, frames]`
    rather than `[1, frames, d_model]`, and that asymmetry is the whole point of this class beyond
    hoisting the projection. `scores @ V` is the one matmul in a decoder layer with `transpose_y=False`,
    and `topology_ops._op_matmul_x_y` composes that as PERMUTE + CONT + MUL_MAT -- so the decoder's
    graph was materialising a 4.6 MB transpose of V **every token, in every layer**: 12 nodes x 27
    tokens for an 11-second clip, measured at **9.0% of a whole transcription and 47% of the decode
    loop** at one thread. It is a loop-invariant transform of a graph input that this phase already
    makes constant for the utterance, so it belongs on this side of the phase boundary, where it runs
    12 times per utterance instead of 324.
    
    Doing it here is only half of it: the decoder still TRACES the chain, because HF's attention
    reshapes and transposes whatever `v_proj` returns. `hoist_cross_v_transpose` deletes it from the
    traced topology afterwards, and raises if it does not find exactly what this emits. The two are one
    change and neither is correct alone.
    """

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        self.n_heads = int(cfg.decoder_attention_heads)
        self.head_dim = int(cfg.d_model) // self.n_heads
        self.projs = nn.ModuleList()
        for layer in model.model.decoder.layers:
            self.projs.append(layer.encoder_attn.k_proj)
            self.projs.append(layer.encoder_attn.v_proj)

    def forward(self, xa):
        out = []
        for i, proj in enumerate(self.projs):
            t = proj(xa)
            if i % 2:
                # The V half. `[b, frames, d_model] -> [b, heads, head_dim, frames]`: exactly the
                # layout `ggml_mul_mat` needs as its A operand to contract `scores @ V` over frames,
                # which is what the decoder was rebuilding per token. K is left alone -- `Q @ K^T` is
                # `transpose_y=True`, which composes to a bare MUL_MAT over the natural layout.
                b, frames, _ = t.shape
                t = t.view(b, frames, self.n_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()
            out.append(t)
        return tuple(out)


def hoist_cross_v_transpose(topo: dict, n_layers: int) -> int:
    """Delete the per-token transpose of each `xv_i` from the traced DECODER topology.

    The other half of `_WhisperCrossKvWrapper`'s V transpose, and the reason that one is not enough on
    its own: HF's attention reshapes and transposes whatever `v_proj` returns, so the trace still
    contains the chain whatever shape reaches it. What arrives here, per layer, is

        RESHAPE(xv_i)  ->  PERMUTE [0,2,1,3]  ->  PERMUTE [1,0,2,3]  ->  CONT  ->  MUL_MAT

    where the first two are HF's `.view(b, -1, heads, head_dim).transpose(1, 2)` and the second two are
    `topology_ops._op_matmul_x_y` composing `transpose_y=False`. With V arriving already in the layout
    the CONT was producing, the whole chain is the identity and the MUL_MAT can read `xv_i` directly.

    **This rewrites a traced graph, so it asserts every step and raises rather than skipping.** A
    transformers release that spells the reshape differently, or a change to the matmul composition,
    must break the export loudly here -- silently leaving the chain in place would cost only speed, but
    silently rewriting a chain that is NOT this one would be a wrong answer, and whisper's gate compares
    the encoder rather than the decoder ([Retro-006](tensor oracle, not token oracle) is the standing
    warning: a wrong decoder still emits a plausible transcript).

    Returns the number of layers rewritten, which the caller checks against `n_layers`.
    """
    nodes = topo["nodes"]
    producer = {out: (i, n) for i, n in enumerate(nodes) for out in n["outputs"]}
    consumers = {}
    for i, n in enumerate(nodes):
        for var in n["inputs"]:
            consumers.setdefault(var, []).append(i)

    def only_consumer(var, want_op, want_axes=None):
        users = consumers.get(var, [])
        if len(users) != 1:
            raise ValueError(
                f"whisper decoder: {var!r} has {len(users)} consumers, expected exactly 1 -- the "
                f"cross-attention V chain cannot be hoisted without duplicating work. Nodes: "
                f"{[nodes[i]['op'] for i in users]}")
        node = nodes[users[0]]
        if node["op"] != want_op:
            raise ValueError(f"whisper decoder: expected {want_op} consuming {var!r}, found "
                             f"{node['op']}")
        if want_axes is not None and node.get("attrs", {}).get("axes") != want_axes:
            raise ValueError(f"whisper decoder: expected {want_op}(axes={want_axes}) consuming "
                             f"{var!r}, found axes={node.get('attrs', {}).get('axes')}")
        return users[0], node

    drop, rename = set(), {}
    for layer in range(n_layers):
        xv = f"xv_{layer}"
        if xv not in consumers:
            raise ValueError(f"whisper decoder: no node consumes {xv!r}; the cross-attention inputs "
                             f"and the traced graph disagree")
        i_reshape, reshape = only_consumer(xv, "RESHAPE")
        i_perm1, perm1 = only_consumer(reshape["outputs"][0], "PERMUTE", [0, 2, 1, 3])
        i_perm2, perm2 = only_consumer(perm1["outputs"][0], "PERMUTE", [1, 0, 2, 3])
        i_cont, cont = only_consumer(perm2["outputs"][0], "CONT")
        drop.update((i_reshape, i_perm1, i_perm2, i_cont))
        # Everything downstream reads the CONT's output; it is now `xv_i` itself.
        rename[cont["outputs"][0]] = xv

    topo["nodes"] = [n for i, n in enumerate(nodes) if i not in drop]
    for n in topo["nodes"]:
        n["inputs"] = [rename.get(v, v) for v in n["inputs"]]
    if isinstance(topo.get("output"), str):
        topo["output"] = rename.get(topo["output"], topo["output"])
    topo["outputs"] = [rename.get(v, v) for v in topo.get("outputs", [])] or topo.get("outputs")
    if topo.get("outputs") is None:
        topo.pop("outputs", None)
    return n_layers


def cross_kv_input_names(n_layers: int) -> tuple:
    """`("xk_0", "xv_0", "xk_1", ...)` -- the decoder's per-layer cross-attention inputs, in the order
    `_WhisperCrossKvWrapper` returns them, which is the order their `index` binding assumes."""
    names = []
    for i in range(n_layers):
        names.append(f"xk_{i}")
        names.append(f"xv_{i}")
    return tuple(names)


class _WhisperDecoderWrapper(nn.Module):
    """`(tokens, position_ids, attention_mask, xk_0, xv_0, ...) -> logits`.

    The first three inputs are not a free choice. `position_ids` and `attention_mask` are passed
    explicitly for the reason `causal_lm_export._causal_mask` documents -- it is what keeps the token
    axis genuinely dynamic under `torch.jit.trace` -- and both names are already in
    `driver_components.POSITION_INPUT_NAMES`/`CAUSAL_MASK_INPUT_NAMES`, so the driver fills them in from
    `n_tokens`/`n_past` without this family declaring anything.

    **The rest are the `cross_kv` phase's outputs, and they replaced a single `xa`.** This step used to
    take the encoder output whole and project it to cross-attention K/V inside itself, every token --
    24 matmuls of [1500, 768] x [768, 768] per step for whisper-small, measured at **57.7% of a whole
    transcription**. They are a function of the encoder alone, so they are computed once by
    `_WhisperCrossKvWrapper` and copied in backend-side. The copy is not free (~110 MB per step for
    whisper-small) and is still an order of magnitude below recomputing them.

    `use_cache=False` is what makes the trace cache-free, which is the shape `fuse_loom_attention`
    matches; the cache appears at run time, in the engine, not in the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        # Every cross-attention projection becomes a slot fed from this call's arguments. See
        # `_CrossKvSlot`: the tensors are the same, they are just no longer recomputed per token.
        self._cross = [None] * (2 * len(self.decoder.layers))
        for i, layer in enumerate(self.decoder.layers):
            layer.encoder_attn.k_proj = _CrossKvSlot(self._cross, 2 * i)
            layer.encoder_attn.v_proj = _CrossKvSlot(self._cross, 2 * i + 1)

    def forward(self, tokens, position_ids, attention_mask, *cross):
        for i, tensor in enumerate(cross):
            self._cross[i] = tensor
        hidden = self.decoder(
            input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask,
            # `encoder_hidden_states` is what makes these blocks CROSS-attention
            # (`is_cross_attention = key_value_states is not None`), and nothing downstream of that
            # test reads it any more -- the projections that did are slots now. Passing `cross[0]`,
            # which has the encoder output's exact shape, keeps the flag true without declaring a
            # 4.6 MB input the graph would not otherwise use.
            encoder_hidden_states=cross[0], use_cache=False,
        ).last_hidden_state
        return self.proj_out(hidden)


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4D additive causal mask, the form transformers passes straight through to attention.

    Same tensor `causal_lm_export._causal_mask` builds, and for the same reason: an already-prepared 4D
    mask short-circuits `create_causal_mask` entirely, so the internal mask builder never derives a
    key length from a Python-level shape that tracing would bake in.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def decoder_prompt_constants(generation_config, n_audio_ctx: int, vocab_size: int,
                              n_text_ctx: int) -> dict:
    """The token ids a Whisper decode prompt is built from, read off the checkpoint's own generation
    config -- for the DRIVER to build the prompt with, not for a host to hardcode.

    Five numbers, and which of them exist is itself the capability statement this family needs:

    * `SOT` -- `decoder_start_token_id`. Always present; a prompt is at minimum this one token.
    * `LANG_LO` / `LANG_HI` -- the half-open id window the 98 language tokens occupy. Present only on a
      **multilingual** checkpoint. `0`/`0` on an English-only one (`whisper-*.en` has no language tokens
      at all), which is how the driver knows it must neither ask for a language nor try to detect one.
      Derived as `min..max+1` over `lang_to_id`'s own values rather than assumed contiguous -- and
      cross-checked below, because a non-contiguous block would make a restricted argmax able to return
      something that is not a language.
    * `TRANSCRIBE` / `TRANSLATE` -- `task_to_id`, likewise absent (`0`) on an English-only checkpoint.
    * `NO_TIMESTAMPS` -- `no_timestamps_token_id`, `0` if this checkpoint has no such token.

    **Not `forced_decoder_ids`**, which is the obvious-looking source and is wrong: a multilingual
    checkpoint leaves its language slot `None` there (`[[1, None], [2, 50359]]` on whisper-small) because
    HF fills it in from detection at generation time. Copying that list yields a prompt with a hole in it.
    """
    lang_to_id = getattr(generation_config, "lang_to_id", None) or {}
    task_to_id = getattr(generation_config, "task_to_id", None) or {}
    lang_ids = sorted(int(v) for v in lang_to_id.values())
    lang_lo, lang_hi = (lang_ids[0], lang_ids[-1] + 1) if lang_ids else (0, 0)
    if lang_ids and len(lang_ids) != lang_hi - lang_lo:
        # The driver detects a language with ONE restricted argmax over [LANG_LO, LANG_HI). A gap in the
        # block means some id in that window is not a language token, so detection could return it --
        # silently, as a plausible-looking prompt token. Say so at export time instead.
        raise ValueError(
            f"this checkpoint's {len(lang_ids)} language tokens do not occupy a contiguous id block: "
            f"{lang_lo}..{lang_hi - 1} spans {lang_hi - lang_lo} ids. Language detection is a single "
            f"argmax restricted to that window, which is only sound while every id in it is a language."
        )
    # The timestamp block: Whisper lays it out immediately above `<|notimestamps|>` and runs it to the
    # end of the vocabulary, one token per encoder frame plus one for the closing edge (`<|0.00|>` ..
    # `<|30.00|>` in 0.02 s steps = n_audio_ctx + 1 of them). Derived from that layout rather than by
    # looking tokens up by name, and then CHECKED against the model's own frame count -- a driver takes a
    # timestamp-restricted argmax over this window, so a wrong bound would silently return a word.
    no_timestamps = int(getattr(generation_config, "no_timestamps_token_id", None) or 0)
    ts_lo = ts_hi = 0
    if no_timestamps:
        ts_lo, ts_hi = no_timestamps + 1, int(vocab_size)
        # Unconditional once the checkpoint has a `<|notimestamps|>` at all -- a guarded version of this
        # check is what let `TS_HI = 0` ship once: `vocab_size` is on the model config, not the
        # generation config, so reading it from the wrong object produced a zero, the guard skipped
        # itself, and the driver silently stopped forcing timestamps. A bound this arithmetic depends on
        # gets verified or the export fails; it does not get defaulted.
        if ts_hi - ts_lo != n_audio_ctx + 1:
            raise ValueError(
                f"this checkpoint's timestamp block is {ts_hi - ts_lo} tokens ({ts_lo}..{ts_hi - 1}), but "
                f"its encoder emits {n_audio_ctx} frames, which should give {n_audio_ctx + 1} timestamps "
                f"(one per frame boundary). The bounds come from `no_timestamps_token_id` + 1 and "
                f"`vocab_size`, so one of those is not what this family assumes."
            )
    # `<|startofprev|>` and how much of the previous window's text may follow it. Whisper spends at most
    # HALF its text context on that carried-over context, leaving the other half for what this window is
    # about to say -- one token of the half goes to `<|startofprev|>` itself. Absent on a checkpoint with
    # no such token, which the driver reads as "cannot condition".
    prev_sot = int(getattr(generation_config, "prev_sot_token_id", None) or 0)
    return {
        "SOT": int(generation_config.decoder_start_token_id),
        "LANG_LO": lang_lo,
        "LANG_HI": lang_hi,
        "TRANSCRIBE": int(task_to_id.get("transcribe", 0)),
        "TRANSLATE": int(task_to_id.get("translate", 0)),
        "NO_TIMESTAMPS": no_timestamps,
        "TS_LO": ts_lo,
        "TS_HI": ts_hi,
        "PREV_SOT": prev_sot,
        "MAX_PREV": (n_text_ctx // 2 - 1) if prev_sot else 0,
    }


def load_feature_extractor(model_dir: str):
    """The checkpoint's own `WhisperFeatureExtractor`.

    Loaded from the directory rather than constructed with this family's idea of the defaults: the mel
    filterbank, the FFT geometry and the 30 s chunk length are all properties of the checkpoint, and
    `preprocessor_config.json` is where it states them.
    """
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(model_dir)


@dataclass
class ASRWhisperExportConfig(BaseMultiPhaseModelExportConfig):
    """Whisper as two traced phases -- `encoder` (waveform -> audio states) and `decoder` (a cached,
    cross-attending transformer step) -- plus a driver that runs the first once and loops the second.

    **Why this is a `MultiPhase` config and not a fourth `Decomposition`.**
    `EXPORT-PREPARATION.md` §5 decision 2 reserved a decomposition of its own for this shape, on the
    reasoning that a new orchestration needs a driver builder the family cannot supply. Building it
    found the orchestration to be the one `MultiPhase` already has: two independently traced phases, a
    component list, and `MultiPhaseDriverBuilder`. What genuinely differs is *two facts*, and both are
    now fields on the pieces that own them -- `ExportPhase.fuse_attention` (this decoder is cached and
    this encoder must not be) and `PrefillDecodeLoop.bound` (this step's `xa` comes from the encoder,
    not from the caller). A `Decomposition` subclass would have restated `MultiPhase.export` verbatim
    around those two. See BACKLOG.md P4.1.
    """

    model_dir: str = ""
    architecture: str = "whisper"
    output_path: str = "whisper_mil.gguf"
    # The decoder's token axis. The encoder phase declares `n_samples` instead -- raw audio, never a
    # token count -- the same distinction the NeMo family draws (EXPORT-ROADMAP.md R1).
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "whisper_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # Read off the checkpoint in `phases()`, which is the only moment the model and its feature
    # extractor are both in hand. Declared as fields rather than recomputed because the driver
    # components and `hparams()` need them after the trace.
    n_samples: Optional[int] = field(default=None, init=False, repr=False)
    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    n_audio_ctx: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    n_heads: Optional[int] = field(default=None, init=False, repr=False)
    max_target_positions: Optional[int] = field(default=None, init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)
    # Like `decoder_bindings`: filled in by `phases()`, and defaulted so `components()` can be
    # introspected without tracing anything (tests/ci/test_component_registry.py does exactly that).
    n_text_layers: Optional[int] = field(default=None, init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    prompt_constants: dict = field(default_factory=dict, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory, already established by the recognizer's own detect(), which "
            "reads its config.json `model_type`. WhisperForConditionalGeneration.from_pretrained "
            "raises on anything it cannot load."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the decoder ExportPhase's own Axis link, which is where the "
                               "value is actually used"),
        "driver_script_path": Unchecked("the one hand-written fragment here is a header comment; its "
                                        "contents are still parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction -- see the class docstring for why this "
                                   "shape did not need a fourth one"),
        "n_samples": Unchecked("READ off the checkpoint's own feature extractor in phases() "
                               "(`chunk_length * sampling_rate`), not declared"),
        "sample_rate": Unchecked("same -- the feature extractor's own `sampling_rate`"),
        "n_audio_ctx": Unchecked("same -- `config.max_source_positions`"),
        "d_model": Unchecked("same -- `config.d_model`"),
        "n_heads": Unchecked(
            "same -- `config.decoder_attention_heads`. Needed here rather than only inside the cross_kv "
            "wrapper because the DECODER declares V's head-split shape, and both must derive it from "
            "the one checkpoint field."
        ),
        "max_target_positions": Unchecked("same -- `config.max_target_positions`, which is the KV "
                                          "cache capacity a decode loop can address"),
        "decoder_bindings": Unchecked(
            "(name, kind) per decoder input, derived in phases() from the SAME mil_inputs list the "
            "trace is declared with, through `exporter._binding_kind` -- the one implementation of "
            "'is this input host-computed', which the flattened path already routes through -- so the "
            "driver cannot disagree with the trace about the order or the names, and this family "
            "cannot drift from the causal-LM one about which names the driver fills in. "
            "PrefillDecodeLoop's own `inputs` link re-checks them against the emitted topology anyway."
        ),
        "n_text_layers": Unchecked(
            "same -- `len(model.model.decoder.layers)`, read in phases(). It is a COUNT of exported "
            "outputs rather than a claim about them: `cross_kv` emits two per decoder layer."
        ),
        "cross_kv_names": Unchecked(
            "derived in phases() by `cross_kv_input_names(n_text_layers)`, which is also what orders "
            "`_WhisperCrossKvWrapper`'s return tuple -- one function, so the decoder's input names, "
            "the phase's output order and the driver's `index` arithmetic cannot disagree. "
            "PrefillDecodeLoop's `inputs` link re-checks the names against the emitted topology."
        ),
        "prompt_constants": Unchecked(
            "READ off the checkpoint's own generation config in phases() (`decoder_start_token_id`, "
            "`lang_to_id`, `task_to_id`, `no_timestamps_token_id`), never declared -- see "
            "`decoder_prompt_constants`, which DOES cross-check the one property a restricted argmax "
            "depends on: that the language tokens form a contiguous id block."
        ),
    }

    def prepare_environment(self) -> None:
        # transformers' hf-hub version gate, the same stub causal_lm_export installs at import time.
        mock_dep = types.ModuleType("dependency_versions_check")
        mock_dep.dep_version_check = lambda *args, **kwargs: None
        sys.modules.setdefault("transformers.dependency_versions_check", mock_dep)

    def load_model(self):
        from transformers import WhisperForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        return WhisperForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype=torch.float32
        ).eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        extractor = load_feature_extractor(self.model_dir)
        cfg = model.config
        self.n_samples = int(extractor.n_samples)
        self.sample_rate = int(extractor.sampling_rate)
        self.n_audio_ctx = int(cfg.max_source_positions)
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.decoder_attention_heads)
        self.max_target_positions = int(cfg.max_target_positions)
        self.prompt_constants = decoder_prompt_constants(
            model.generation_config, self.n_audio_ctx, cfg.vocab_size, self.max_target_positions)
        # Not from `decoder_prompt_constants`, which answers "which tokens does a prompt need" and has
        # a cross-check contract to match. This is a shape: how many `cross_kv` outputs the driver has
        # to bind, which is two per decoder layer.
        self.prompt_constants["N_TEXT_LAYERS"] = len(model.model.decoder.layers)
        # The language and task tables by NAME, kept for `contract()`. The driver needs only the id
        # WINDOW (it detects with one restricted argmax over it), which is why `decoder_prompt_constants`
        # returns bounds; a host asked for `language="en"` needs the individual ids, and only the
        # checkpoint's own tables have them. HF spells these keys either `"<|en|>"` or `"en"` depending
        # on version, so the brackets are stripped rather than assumed absent.
        def _bare(name: str) -> str:
            return name[2:-2] if name.startswith("<|") and name.endswith("|>") else name

        gen_cfg = model.generation_config
        self.lang_to_id = {_bare(str(k)): int(v)
                           for k, v in (getattr(gen_cfg, "lang_to_id", None) or {}).items()}
        self.task_to_id = {_bare(str(k)): int(v)
                           for k, v in (getattr(gen_cfg, "task_to_id", None) or {}).items()}

        mel = WhisperMelFrontend(extractor.n_fft, extractor.hop_length, np.array(extractor.mel_filters))

        # The trace length for the decoder. Free, and deliberately not 1: the graph must contain a real
        # token axis for the RangeDim below to make dynamic, and a length-1 trace gives coremltools a
        # size-1 axis it is entitled to fold away.
        trace_tokens = 8
        token_axis = ct.RangeDim(1, self.max_target_positions)
        self.n_text_layers = len(model.model.decoder.layers)
        self.cross_kv_names = cross_kv_input_names(self.n_text_layers)
        decoder_inputs = [
            ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
        ] + [
            # Fixed shape, every one of them: the encoder always emits `n_audio_ctx` frames, so nothing
            # about the cross-attention K/V varies with the token axis. That is the whole reason they
            # can be hoisted out of the step at all.
            #
            # **K and V do not have the same shape.** V arrives head-split and transposed, because that
            # is the layout `scores @ V` needs and rebuilding it per token cost 47% of the decode loop
            # -- see `_WhisperCrossKvWrapper`. The trace still contains the chain that would have built
            # it (HF reshapes whatever `v_proj` returns, whatever shape reaches it); `topology_rewrite`
            # below deletes it. Shapes here are what the DRIVER copies into, so they must be the shapes
            # `cross_kv` emits, not the ones HF's code reads.
            ct.TensorType(name=name, shape=self._cross_kv_shape(name), dtype=np.float32)
            for name in self.cross_kv_names
        ]
        self.decoder_bindings = tuple(
            (t.name, _binding_kind(t.name)) for t in decoder_inputs
        )

        # ORDER IS LOAD-BEARING: `_WhisperCrossKvWrapper` captures the real projection modules, and
        # `_WhisperDecoderWrapper.__init__` then replaces the attributes that reached them. Built the
        # other way round the cross_kv phase would trace `_CrossKvSlot`s and export no weights at all.
        cross_kv_wrapper = _WhisperCrossKvWrapper(model).eval()
        decoder_wrapper = _WhisperDecoderWrapper(model).eval()

        return [
            ExportPhase(
                name="encoder",
                wrapper=_WhisperEncoderWrapper(model, mel).eval(),
                dummy_inputs=(torch.zeros(1, self.n_samples),),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, self.n_samples), dtype=np.float32)],
                # Every shape in this phase is a compile-time constant -- Whisper always sees exactly
                # 30 s of audio -- so this axis names the sample count without anything varying over it.
                root_axis="n_samples",
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv_wrapper,
                dummy_inputs=(torch.zeros(1, self.n_audio_ctx, self.d_model),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, self.n_audio_ctx, self.d_model),
                                          dtype=np.float32)],
                # Nothing here varies with the token axis -- this phase never sees a token. It runs
                # once per window, straight after the encoder, so its one axis is the encoder's frame
                # count (fixed at `n_audio_ctx`; `axes.py` is the vocabulary this name comes from).
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder_wrapper,
                dummy_inputs=(
                    torch.zeros((1, trace_tokens), dtype=torch.long),
                    torch.arange(trace_tokens).unsqueeze(0),
                    causal_mask(trace_tokens),
                ) + tuple(torch.zeros(self._cross_kv_shape(name))
                          for name in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                # The other half of the V transpose. See `hoist_cross_v_transpose`: it deletes the
                # chain HF's attention traces around each `xv_i`, and RAISES if that chain is not
                # exactly the one `_WhisperCrossKvWrapper` was written against.
                topology_rewrite=lambda topo: hoist_cross_v_transpose(topo, self.n_text_layers),
                root_axis=self.root_axis,
                # The self-attention blocks become cached ATTENTION nodes; the cross-attention blocks do
                # not, and that is correct rather than a miss. `fuse_loom_attention` anchors on the
                # `add(scores, mask)` that only a masked block has, and Whisper's cross-attention has no
                # mask at all -- it attends over the whole encoder output, every step. A cache there
                # would be wrong twice over: the K/V it would store are the encoder's, identical at
                # every step, and `layer` indices are assigned in occurrence order, so a cached
                # cross-attention block would consume cache slots the self-attention blocks address.
                #
                # "Identical at every step" used to be an argument for leaving them alone; it is the
                # reason they left. They are now the `cross_kv` phase, run once per window and copied
                # in -- what was wrong was not the absence of a CACHE but the presence of the
                # PROJECTION, 24 [1500, 768] x [768, 768] matmuls per token, 57.7% of a transcription.
                fuse_attention=True,
                kv_cache_size=self.max_target_positions,
            ),
        ]

    def _cross_kv_shape(self, name: str) -> tuple:
        """The shape `cross_kv` emits for one of its outputs, which is what the decoder declares.

        K keeps the natural `[1, frames, d_model]`; V is `[1, heads, head_dim, frames]`. Keyed on the
        NAME rather than on position so this cannot drift out of step with `cross_kv_input_names`, which
        is the same list the driver's `index` arithmetic assumes.
        """
        if name.startswith("xv_"):
            return (1, self.n_heads, self.d_model // self.n_heads, self.n_audio_ctx)
        return (1, self.n_audio_ctx, self.d_model)

    def hparams(self) -> dict:
        """What a HOST must know to call this driver at all.

        `n_samples` is the load-bearing one: Whisper is trained on exactly 30 s of audio and the encoder
        graph is built at that length, so a caller has to pad or trim to it before calling. Until this
        existed for the bespoke path, that number lived in a C++ test header
        (`test_e2e_whisper_lua_driver.cpp` sizes its input from a hardcoded `WhisperConfig`), which is
        precisely the "self-contained GGUF" claim being false (P4.0.8's first follow-up).

        `sample_rate` is here for two host jobs, both of which are otherwise a guess: the audio has to be
        resampled to it before `waveform` means anything, and a **timestamp token** has to be turned into
        a time. Whisper's timestamp tokens step by one encoder frame, so that step is
        `(n_samples / sample_rate) / n_audio_ctx` -- 0.02 s -- and a host that had to hardcode either
        number would be carrying a per-model constant, which is what these KVs exist to stop.
        """
        return {
            "n_samples": self.n_samples,
            "n_audio_ctx": self.n_audio_ctx,
            "n_text_ctx": self.max_target_positions,
            "sample_rate": self.sample_rate,
        }

    def contract(self) -> dict:
        """The task's default pair, plus the ASR decode table -- the ids that make a transcription loop
        this model's rather than Whisper's.

        Everything here already existed as a number this export computed; none of it is newly derived.
        What changes is WHERE it is written down. The engine used to recover the same facts by spelling
        Whisper's tokens -- `piece_to_id("<|0.00|>")`, `"<|" + language + "|>"`, `<|notimestamps|>` --
        which worked because Whisper is the only timestamped family exported so far, and would have cost
        engine code for the second one: Canary, Qwen3-ASR and Granite-Speech spell all three differently.
        A checkpoint's own token ids are a property of the checkpoint, so they belong in the file
        (loom.cpp docs/HIGH-LEVEL-API.md §2/§3).

        Omission is meaningful and is used: an English-only checkpoint has no language tokens at all, so
        the language table is empty and is left out entirely rather than written as an empty array. A
        host reads that as "this model has no languages to name", which is true, and different from
        "this export predates the table".
        """
        contract = super().contract()
        c = self.prompt_constants
        if c["TS_LO"]:
            contract["asr.timestamp_first_id"] = int(c["TS_LO"])
            # Seconds per timestamp token: one encoder frame. The engine used to derive this from three
            # separate hparams, which is the same arithmetic in a place that cannot check it -- and it is
            # only Whisper's arithmetic. A family whose timestamps step differently declares the number.
            contract["asr.timestamp_step_sec"] = (self.n_samples / self.sample_rate) / self.n_audio_ctx
        if c["NO_TIMESTAMPS"]:
            # Dropped before detokenizing: the model stating something about the decode rather than a
            # word that was spoken. EOS is NOT listed -- every vocabulary family names it in
            # `tokenizer.ggml.eos_token_id`, so it needs no per-task declaration to be found.
            contract["asr.control_ids"] = [int(c["NO_TIMESTAMPS"])]
        if self.lang_to_id:
            # Parallel arrays, because a GGUF array is homogeneous and there is no map type. Sorted so
            # the export is deterministic -- two runs of the same checkpoint must produce byte-identical
            # files, and dict order is not something to bet that on.
            names = sorted(self.lang_to_id)
            contract["asr.language_names"] = names
            contract["asr.language_ids"] = [self.lang_to_id[n] for n in names]
        if self.task_to_id:
            names = sorted(self.task_to_id)
            contract["asr.task_names"] = names
            contract["asr.task_ids"] = [self.task_to_id[n] for n in names]
        contract["asr.prev_context"] = int(self.max_target_positions)
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        # The tokenizer travels with the model. `tokenizer_pre` is named rather than left to the hash
        # cascade because Whisper's tokenizer.json is not in llama.cpp's chkhsh table, so detection warns
        # and falls back to "qwen2" -- a pretokenizer whose digit and letter runs differ from the
        # byte-level GPT-2 one Whisper actually uses. Nothing in this family's own decode path notices
        # (ASR only ever maps ids BACK to text), which is exactly why a wrong value here would sit
        # unnoticed until something encoded text with it.
        return dict(tokenizer_dir=self.model_dir, tokenizer_pre="gpt-2", hparams=self.hparams())

    def driver_components(self) -> List:
        """Encoder once, then the decode loop -- two components and a header.

        The two calls are IR rather than text, which is what lets each be checked against the real
        traced topologies: the encoder's input names and output arity by `SubgraphCallComponent`, the
        decoder's by `PrefillDecodeLoop`'s own exact `inputs` link.

        The prompt is the exception, and it earns it: choosing a language is branching control flow
        (given / detect / neither), which is exactly what a `LuaFragment` is for -- the same split
        Parakeet's decode loop takes. Its own `run_subgraph_and_retain('decoder', ...)` detection call is
        parsed out of the text and declared against the traced topology regardless.
        """
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Lit, OutputRef, Var

        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values=dict(self.prompt_constants)),
            SubgraphCallComponent(
                topology="encoder",
                # Retained, not bound to a local. The encoder emits `n_audio_ctx * d_model` floats --
                # 1.15M for whisper-small -- and the decode loop reads them at every step, so a Lua
                # table here would marshal them once per generated token. `OutputRef` below copies
                # backend-side instead (BACKLOG.md P4.0.12).
                outputs=(),
                retain=True,
                inputs={"waveform": FieldAccess("inputs", "waveform")},
                # A literal, not `#inputs.waveform`: this phase's sample count is a compile-time
                # constant (Whisper always sees exactly 30 s), so binding the axis to the length the
                # graph was built at states that, where reading the caller's array would imply the
                # encoder could run at some other length.
                axes={"n_samples": Lit(self.n_samples), "n_past": Lit(0)},
                note="Encoder: one fixed-shape pass over 30 s of audio -- mel frontend, conv stem, "
                     "transformer stack.",
            ),
            SubgraphCallComponent(
                topology="cross_kv",
                # Retained like the encoder, and for a sharper version of the same reason: these are
                # the tensors the decode loop reads at every step, and the whole point of the phase is
                # that they are produced ONCE per window.
                outputs=(),
                retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": Lit(self.n_audio_ctx), "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, computed once from the encoder "
                     "output instead of re-projected at every token.",
            ),
            LuaFragment(
                self.driver_script_path / "01_prompt.lua",
                reads=("SOT", "LANG_LO", "LANG_HI", "TRANSCRIBE", "NO_TIMESTAMPS", "TS_LO", "TS_HI",
                       "PREV_SOT", "MAX_PREV", "N_TEXT_LAYERS"),
                defines=("_prompt", "_language", "_gen0", "_eos", "_decoder_inputs"),
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                # The cross-attention K/V, held constant across every step -- `cross_kv` output
                # `2 * layer + 1` is that layer's K and `+ 2` its V, which is the interleaving
                # `_WhisperCrossKvWrapper` returns and `cross_kv_input_names` names. Everything else the
                # loop needs it already computes: tokens from the previous step, positions and the
                # causal mask from n_tokens/n_past.
                bound={name: OutputRef("cross_kv", index=i + 1)
                       for i, name in enumerate(self.cross_kv_names)},
                # The prefix the fragment above built, not `inputs.tokens`: this driver owns its own
                # prompt, so a caller passes audio and at most a language.
                prompt=Var("_prompt"),
                # The forced opening timestamp, when there is one: the model chose it, so
                # it is part of what this call generated even though it is fed back in.
                generated_prefix=Var("_gen0"),
            ),
        ]


def _hf_model_type(path: Path) -> Optional[str]:
    """An HF-style directory's own `config.json`'s `model_type`, or None if `path` isn't one. Never
    raises: `detect()` runs against unidentified paths by construction."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return config.get("model_type") if isinstance(config, dict) else None


def _is_whisper(path: Path) -> bool:
    """A real structural check (BACKLOG.md P3.2): an HF directory declaring `model_type == "whisper"`.

    Deliberately claims distil-whisper and every fine-tune too -- they are the same architecture with
    fewer decoder layers, which this family reads off the checkpoint rather than assuming. It does not
    collide with `hf-causal-lm`'s fallback, which requires a `*ForCausalLM` architecture entry and so
    rejects `WhisperForConditionalGeneration` by construction.
    """
    return _hf_model_type(path) == "whisper"


def _build_whisper(path: Path, output_path: str) -> ASRWhisperExportConfig:
    return ASRWhisperExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRWhisperExportConfig,
        recognizers=[ModelRecognizer(name="whisper", detect=_is_whisper, build_config=_build_whisper)],
    ))
