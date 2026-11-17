"""OmniVoice (k2-fsa): a masked-diffusion TTS -- a BIDIRECTIONAL Qwen3-0.6B that fills in eight codebooks
of HiggsAudio-v2 codec tokens over a fixed number of unmasking steps, then the codec's decoder.

Nothing in this model is autoregressive. A generation is

    prompt = style tokens + text tokens [+ reference codes] + T target frames, all MASK
    32 times:  one forward over the whole prompt (conditional) and over the target alone
               (unconditional), classifier-free guidance on the log-probabilities, then the k most
               confident still-masked cells (k from a shifted schedule) take their argmax
    codes -> HiggsAudio-v2 decoder -> 24 kHz

so the cost is `steps x (L + T)` row-forwards of the 28-layer stack, with no KV cache: every step
re-reads the whole sequence, because bidirectional attention lets every row see the cells that just
changed.

**One phase per step, and both guidance passes are ONE sequence.** The reference batches the
conditional row (`L` long) with the unconditional one (the target alone, `T` long, padded to `L`).
Here the two are concatenated into one `L + T` sequence with a block-diagonal mask and the
unconditional positions restarting at 0 -- the same attention, row for row, with the weights read
once (ADR-118's argument for Chatterbox, where the two halves needed a double-width cache; here there
is no cache at all). The head runs on the `2T` rows a step reads, gathered by index, and the whole
scoring -- guidance, log-softmax, the mask id banned, argmax, confidence, the layer penalty and the
position noise -- is in the graph, so a step hands the driver `8 x T` ids and `8 x T` scores.

**One file, not ADR-022's pair.** The codec (HiggsAudio-v2's tokenizer) is used by no other model in
this tree, the reference defines its output at the WAVEFORM (each long-text chunk decoded on its own and
cross-faded, then silences trimmed and the level set), and cloning needs its encoder anyway (ADR-038).
See loom.cpp ADR-131.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


OMNIVOICE_MISSING = (
    "OmniVoice ships as the `omnivoice` package (Apache-2.0, github.com/k2-fsa/OmniVoice), which needs "
    "transformers >= 5.3 for `HiggsAudioV2TokenizerModel`. Put a clone on PYTHONPATH (or "
    "LOOM_OMNIVOICE_REPO) rather than `pip install omnivoice`: its requirements pull gradio and "
    "webdataset for a training UI this export never touches. It also imports `pydub` at module scope."
)

N_CODEBOOKS = 8
# The reference's own generation defaults (`OmniVoiceGenerationConfig`), the driver's fallbacks.
NUM_STEP = 32
GUIDANCE_SCALE = 2.0
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
POSITION_TEMPERATURE = 5.0
AUDIO_CHUNK_DURATION = 15.0
AUDIO_CHUNK_THRESHOLD = 30.0
PAD_DURATION = 0.1
FADE_DURATION = 0.1


def load_omnivoice(model_dir: str):
    import os
    import sys

    repo = os.environ.get("LOOM_OMNIVOICE_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from omnivoice import OmniVoice
    except ImportError as exc:                       # pragma: no cover - env-dependent
        raise ImportError(OMNIVOICE_MISSING) from exc
    import transformers

    # `sdpa`, for the reason `qwen3_tts_export.load_model` gives: the attention fusion matches the scale
    # folded onto Q, which is what the sdpa path emits.
    model = OmniVoice.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    model.audio_tokenizer.eval()
    return model


def _log_softmax(x):
    """`log_softmax` over the last axis, spelled so it cannot underflow: `x - (m + log sum exp(x - m))`.

    torch's own converts to `log(softmax(x))`, and a code the model all but rules out has a softmax of 0 in
    f32 -- `log` makes it -inf, in BOTH guidance halves, and `c + g (c - u)` is then `-inf - -inf` = NaN.
    One NaN score in the step's top-k sort is enough to unmask the wrong cell: the first OmniVoice run
    went NaN on one cell at step 11 of 32 and every later step drifted from the reference."""
    m = torch.amax(x, dim=-1, keepdim=True)
    return x - (m + torch.log(torch.exp(x - m).sum(dim=-1, keepdim=True)))


class _StepWrapper(nn.Module):
    """One unmasking step: `(ids, audio_mask, position_ids, segment, cond_rows, uncond_rows, uniforms, knobs) -> (ids,
    scores)` for the `8 x T` target cells.

    Inputs, all built by the driver:

    * `ids` `[8, N]`, `N = L + T`: the conditional sequence (text ids repeated in every row, codes in
      the audio region) followed by its own last `T` columns -- the unconditional row. The text
      embedding reads row 0, the audio embedding all eight, as `_prepare_embed_inputs` does.
    * `audio_mask`, `segment` `[1, N]` floats: which positions are audio, and which half (1 = cond).
      The mask is the outer product `s s' + (1-s)(1-s')`, block-diagonal, built here so the driver
      ships `N` numbers instead of `N^2`.
    * `position_ids` `[1, N]`: `0..L-1` then `0..T-1`, the positions each half had in the batch.
    * `cond_rows`, `uncond_rows` `[1, T]`: which positions' hidden states the head reads -- the
      target's in each half -- gathered by index rather than sliced by shape (a shape-derived slice
      mints a fresh symbol).
    * `uniforms` `[8, T]`: the position noise's draws, `rand_like` in the reference.
    * `knobs` `[1, 4]`: guidance scale, layer penalty, position temperature, noise on (0/1).

    Out: the argmax id per cell and its score, `(max log-prob - layer * penalty) / temperature +
    gumbel(u)` -- the reference's `_predict_tokens_with_scoring` + penalty + `_gumbel_sample`, in its
    order and at f32. Which cells take their id (still masked, top k) is the driver's: it needs the
    schedule and the cells' state, and it is a sort of `8T` numbers.
    """

    def __init__(self, model):
        super().__init__()
        self.llm = model.llm
        self.audio_embeddings = model.audio_embeddings
        self.audio_heads = model.audio_heads
        self.vocab = int(model.config.audio_vocab_size)
        self.mask_id = int(model.config.audio_mask_id)
        self.register_buffer("offsets", (torch.arange(N_CODEBOOKS) * self.vocab).view(-1, 1))
        self.register_buffer("layers", torch.arange(N_CODEBOOKS, dtype=torch.float32).view(-1, 1))

    def forward(self, ids, audio_mask, position_ids, segment, cond_rows, uncond_rows, uniforms, knobs):
        text = self.llm.embed_tokens(ids[0:1])                                  # [1, N, H]
        mask = audio_mask.unsqueeze(-1)                                         # [1, N, 1]
        shifted = ids * audio_mask.to(ids.dtype) + self.offsets                 # [8, N]
        audio = self.audio_embeddings(shifted).sum(dim=0, keepdim=True)         # [1, N, H]
        embeds = audio * mask + text * (1.0 - mask)
        # Block-diagonal: 1 where both rows are in the same half. Outer products of f32 vectors, the
        # form ADR-110's mask took for the same reason (a two-sided broadcast of a compare aborts ggml).
        col = segment.transpose(0, 1)                                           # [N, 1]
        same = col @ segment + (1.0 - col) @ (1.0 - segment)                    # [N, N]
        # `unsqueeze`, not `view(1, 1, N, N)`: a view spelled from shape reads lost one N in the topology
        # (RESHAPE to [N,1,1,1] of an N x N tensor).
        additive = ((same - 1.0) * 1e9).unsqueeze(0).unsqueeze(0)
        hidden = self.llm(inputs_embeds=embeds, attention_mask=additive, position_ids=position_ids,
                          use_cache=False).last_hidden_state[0]                 # [N, H]
        # Two gathers rather than one `2T` gather split in half: `chunk` on a dynamic axis is a
        # shape-derived split coremltools cannot size (`constantchunk` floats a symbol).
        cond = self.audio_heads(hidden.index_select(0, cond_rows[0])).view(-1, N_CODEBOOKS, self.vocab)
        uncond = self.audio_heads(hidden.index_select(0, uncond_rows[0])).view(-1, N_CODEBOOKS,
                                                                               self.vocab)
        c = _log_softmax(cond)
        u = _log_softmax(uncond)
        guided = _log_softmax(c + knobs[0, 0] * (c - u))
        # The mask id is banned by -inf in the reference AFTER the log-softmax, so it still counts in
        # the normaliser; dropping its column here and taking max/argmax over the rest is the same.
        real = guided[..., : self.mask_id].permute(1, 0, 2)                    # [8, T, V-1]
        confidence, ids_out = real.max(dim=-1)                                  # [8, T]
        scores = (confidence - self.layers * knobs[0, 1]) / knobs[0, 2]
        gumbel = -torch.log(-torch.log(uniforms + 1e-10) + 1e-10)
        return ids_out.to(torch.int32), scores + gumbel * knobs[0, 3]


class _DecodeWrapper(nn.Module):
    """`codes [8, T] -> waveform [1, T * hop]`: `HiggsAudioV2TokenizerModel.decode`, its body."""

    def __init__(self, tokenizer):
        super().__init__()
        self.quantizer = tokenizer.quantizer
        self.fc2 = tokenizer.fc2
        self.decoder = tokenizer.acoustic_decoder

    def forward(self, codes):
        quantized = None
        for i, quantizer in enumerate(self.quantizer.quantizers):
            q = quantizer.decode(codes[i:i + 1])                                # [1, D, T]
            quantized = q if quantized is None else quantized + q
        acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        return self.decoder(acoustic).reshape(1, -1)


class _EncodeWrapper(nn.Module):
    """`waveform [1, 960 T] at 24 kHz -> codes [8, T]`: `HiggsAudioV2TokenizerModel.encode` at its default
    (largest) bandwidth, for a clip whose length is a whole number of codec frames -- the driver trims to
    one, as `create_voice_clone_prompt` does.

    Spelled out rather than called, for three things a trace cannot carry as written:

    * the 24 -> 16 kHz resample is torchaudio's own sinc kernel as a constant strided convolution; its
      `[..., :ceil(2n/3)]` crop is shape-derived, and on a whole-frame clip (n a multiple of 3) the
      convolution makes exactly two samples more, so the crop is the static `[..., :-2]`;
    * the acoustic encoder's "pad when the lengths disagree" is a Python branch on shapes; it is
      decided ONCE here, at trace time, and `phases` checks the same decision holds at other lengths;
    * the RVQ's nearest code is `argmax(2 x.e - |e|^2)` -- the reference's `-(|x|^2 - 2 x.e + |e|^2)`
      without the term that is constant along the code axis -- and the residual chain stays in-graph.
    """

    def __init__(self, tokenizer, pad_acoustic: bool):
        super().__init__()
        import torchaudio.functional.functional as taf

        config = tokenizer.config
        self.semantic_model = tokenizer.semantic_model
        self.encoder_semantic = tokenizer.encoder_semantic
        self.acoustic_encoder = tokenizer.acoustic_encoder
        self.fc = tokenizer.fc
        self.quantizers = tokenizer.quantizer.quantizers
        self.n_q = int(tokenizer.quantizer.get_num_quantizers_for_bandwidth(config.target_bandwidths[-1]))
        self.pad = tokenizer.pad if pad_acoustic else 0
        self.downsample = int(config.semantic_downsample_factor)
        orig, new = int(config.sample_rate), int(config.semantic_sample_rate)
        gcd = math.gcd(orig, new)
        kernel, width = taf._get_sinc_resample_kernel(orig, new, gcd, dtype=torch.float32)
        self.register_buffer("resample_kernel", kernel)
        self.width = int(width)
        self.orig, self.new = orig // gcd, new // gcd

    def forward(self, waveform, semantic_rows):
        x = F.pad(waveform, (self.width, self.width + self.orig))
        resampled = F.conv1d(x.unsqueeze(1), self.resample_kernel, stride=self.orig)
        resampled = resampled.transpose(1, 2).reshape(1, -1)[:, : -self.new]
        hidden = self.semantic_model(F.pad(resampled, (160, 160)), output_hidden_states=True).hidden_states
        # `[:, ::2]` (HuBERT's 50 Hz to the codec's 25) as a gather at rows the driver names
        # (`0, 2, ..., 2T-2`): a strided slice has no lowering. HuBERT makes exactly 2T rows here.
        semantic = torch.stack(hidden, dim=1).mean(dim=1).index_select(1, semantic_rows[0])
        e_semantic = self.encoder_semantic(semantic.transpose(1, 2))
        audio = waveform.unsqueeze(1)
        if self.pad:
            audio = F.pad(audio, (self.pad, self.pad))
        e_acoustic = self.acoustic_encoder(audio)
        embeddings = self.fc(torch.cat([e_acoustic, e_semantic], dim=1).transpose(1, 2))    # [1, T, D]
        residual = embeddings[0]
        codes = []
        for quantizer in self.quantizers[: self.n_q]:
            x = quantizer.project_in(residual)                                 # [T, d]
            embed = quantizer.codebook.embed                                    # [K, d]
            score = 2.0 * (x @ embed.t()) - (embed * embed).sum(dim=1).unsqueeze(0)
            index = score.argmax(dim=-1)
            codes.append(index.to(torch.int32).unsqueeze(0))
            residual = residual - quantizer.project_out(embed.index_select(0, index))
        return torch.cat(codes, dim=0)


@dataclass(kw_only=True)
class OmniVoiceExportConfig(BaseMultiPhaseModelExportConfig):
    """A `k2-fsa/OmniVoice` directory (the LM at the root, the codec in `audio_tokenizer/`) -> one GGUF."""

    architecture: str = "omnivoice"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "omnivoice_driver"
    trace_prompt: int = 37
    trace_target: int = 11
    max_seq: int = 4096
    max_frames: int = 2048
    # 20 s, upstream's own warning threshold for a reference clip (it recommends 3-10 s).
    max_ref_frames: int = 500
    _sample_rate: int = field(default=0, init=False, repr=False)
    _hop: int = field(default=0, init=False, repr=False)
    _frame_rate: int = field(default=0, init=False, repr=False)
    _semantic_downsample: int = field(default=0, init=False, repr=False)
    _mask_id: int = field(default=0, init=False, repr=False)

    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the checkpoint; the recognizer read its config.json"),
        "root_axis": Unchecked("the step phase's sequence"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "driver_script_path": Unchecked("hand-written fragments, checked against the topologies"),
        "trace_prompt": Unchecked("the concrete L the trace runs at; deliberately not 8, 16 or 128 "
                                  "(the K/V heads, the query heads, the head width)"),
        "trace_target": Unchecked("the concrete T the trace runs at"),
        "max_seq": Unchecked("the RangeDim bound on L + T"),
        "max_frames": Unchecked("the RangeDim bound on a decode's frames"),
        "max_ref_frames": Unchecked("the RangeDim bound on a reference clip's frames"),
        "_sample_rate": Unchecked("read off the codec's config in phases()"),
        "_hop": Unchecked("read off the codec's config in phases()"),
        "_frame_rate": Unchecked("read off the codec's config in phases()"),
        "_semantic_downsample": Unchecked("read off the codec's config in phases()"),
        "_mask_id": Unchecked("read off the LM's config in phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = load_omnivoice(self.model_dir)
        tokenizer = model.audio_tokenizer
        self._sample_rate = int(tokenizer.config.sample_rate)
        self._hop = int(tokenizer.config.hop_length)
        self._frame_rate = int(tokenizer.config.frame_rate)
        self._semantic_downsample = int(tokenizer.config.semantic_downsample_factor)
        self._mask_id = int(model.config.audio_mask_id)

        # The reference clip's encoder: HuBERT's attention traced EAGER (the fused sdpa path is what
        # ADR-019 measured wrong for an encoder) and its positional convolution's weight norm folded.
        from .ctc_asr_export import strip_weight_norm

        tokenizer.semantic_model.set_attn_implementation("eager")
        strip_weight_norm(tokenizer)
        # `encode` pads the acoustic input when its convolutions would not land on the semantic frame
        # count. On a whole-frame clip -- the only kind the driver hands `ref_encode` -- they always do,
        # which `_EncodeWrapper(pad_acoustic=False)` bakes in; checked here at several lengths.
        for frames in (2, 7, 23, 100, 501):
            if tokenizer._get_conv1d_output_lengths(frames * self._hop, tokenizer.acoustic_encoder) != frames:
                raise ValueError(f"the acoustic encoder makes a different frame count at {frames} frames; "
                                 "`ref_encode` would need `encode`'s padding branch")

        L, T = self.trace_prompt, self.trace_target
        N = L + T
        seq = ct.RangeDim(3, self.max_seq)
        target = ct.RangeDim(1, self.max_frames)
        ids = torch.full((N_CODEBOOKS, N), int(model.config.audio_mask_id), dtype=torch.long)
        ids[:, :L - T] = 100
        audio_mask = torch.zeros(1, N)
        audio_mask[:, L - T:] = 1.0
        positions = torch.cat([torch.arange(L), torch.arange(T)]).view(1, -1)
        segment = torch.cat([torch.ones(L), torch.zeros(T)]).view(1, -1)
        cond_rows = torch.arange(L - T, L).view(1, -1)
        uncond_rows = torch.arange(L, N).view(1, -1)
        uniforms = torch.rand(N_CODEBOOKS, T)
        knobs = torch.tensor([[GUIDANCE_SCALE, LAYER_PENALTY, POSITION_TEMPERATURE, 1.0]])
        frames = ct.RangeDim(1, self.max_frames)
        return [
            ExportPhase(
                name="lm_step",
                wrapper=_StepWrapper(model).eval(),
                dummy_inputs=(ids, audio_mask, positions, segment, cond_rows, uncond_rows, uniforms, knobs),
                mil_inputs=[
                    ct.TensorType(name="ids", shape=(N_CODEBOOKS, seq), dtype=np.int32),
                    ct.TensorType(name="audio_mask", shape=(1, seq), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="segment", shape=(1, seq), dtype=np.float32),
                    ct.TensorType(name="cond_rows", shape=(1, target), dtype=np.int32),
                    ct.TensorType(name="uncond_rows", shape=(1, target), dtype=np.int32),
                    ct.TensorType(name="uniforms", shape=(N_CODEBOOKS, target), dtype=np.float32),
                    ct.TensorType(name="knobs", shape=(1, 4), dtype=np.float32),
                ],
                declared_axes={"cond_rows": {1: "n_codes"}, "uncond_rows": {1: "n_codes"},
                               "uniforms": {1: "n_codes"}},
                dense_attention=True,
            ),
            ExportPhase(
                name="codec_decode",
                wrapper=_DecodeWrapper(tokenizer).eval(),
                dummy_inputs=(torch.randint(0, 1024, (N_CODEBOOKS, 13)),),
                mil_inputs=[ct.TensorType(name="codes", shape=(N_CODEBOOKS, frames),
                                          dtype=np.int32)],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="ref_encode",
                wrapper=_EncodeWrapper(tokenizer, pad_acoustic=False).eval(),
                dummy_inputs=(torch.randn(1, self._hop * 23) * 0.1,
                              (torch.arange(23) * tokenizer.config.semantic_downsample_factor).view(1, -1)),
                mil_inputs=[ct.TensorType(name="waveform",
                                          shape=(1, ct.RangeDim(self._hop * 2, self._hop * self.max_ref_frames)),
                                          dtype=np.float32),
                            ct.TensorType(name="semantic_rows",
                                          shape=(1, ct.RangeDim(2, self.max_ref_frames)), dtype=np.int32)],
                root_axis="n_codes",
                declared_axes={"waveform": {1: f"{self._hop}*n_codes"}, "semantic_rows": {1: "n_codes"}},
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import DriverReturn, ExportConstants, LuaFragment
        from .lua_library import LuaLibrary

        fragments = self.driver_script_path
        return [
            LuaFragment(fragments / "00_header.lua", top_level=True),
            ExportConstants(values=self._constants(), top_level=True),
            LuaLibrary(uses=("round_half_to_even", "to_f32")),
            LuaFragment(fragments / "00_text.lua", top_level=True),
            LuaFragment(fragments / "01_audio.lua", top_level=True),
            LuaFragment(fragments / "02_stream.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish")),
            LuaFragment(fragments / "04_encode.lua", top_level=True),
            LuaFragment(fragments / "03_generate.lua", defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def _constants(self) -> dict:
        """The reference's tables as numbers -- ExportConstants binds nothing else, so strings travel
        as `(offsets, utf-8 bytes)` pairs and character sets as codepoints."""
        if not self.model_dir:
            return {}
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_dir)
        ids = lambda s: tok(s, add_special_tokens=False).input_ids
        special = lambda s: int(tok.convert_tokens_to_ids(s))
        text = _upstream_text_tables()
        languages = self.languages()
        lang_ids, lang_offsets = [], [0]
        for code in ["None"] + languages:
            lang_ids += ids(code)
            lang_offsets.append(len(lang_ids))
        starts, values = _duration_weight_runs()
        en, zh, category = [], [], []
        for index, cat in enumerate(text["categories"]):
            for e, z in cat.items():
                en.append(e)
                zh.append(z)
                category.append(index)
        values_out = {
            "N_CODEBOOKS": N_CODEBOOKS, "MASK_ID": self._mask_id,
            "SAMPLE_RATE": self._sample_rate, "FRAME_RATE": self._frame_rate, "HOP": self._hop,
            "SEMANTIC_DOWNSAMPLE": self._semantic_downsample,
            "NUM_STEP": NUM_STEP, "GUIDANCE_SCALE": GUIDANCE_SCALE, "T_SHIFT": T_SHIFT,
            "LAYER_PENALTY": LAYER_PENALTY, "POSITION_TEMPERATURE": POSITION_TEMPERATURE,
            "AUDIO_CHUNK_DURATION": AUDIO_CHUNK_DURATION, "AUDIO_CHUNK_THRESHOLD": AUDIO_CHUNK_THRESHOLD,
            "PAD_DURATION": PAD_DURATION, "FADE_DURATION": FADE_DURATION,
            "DENOISE_ID": special("<|denoise|>"),
            "LANG_START_ID": special("<|lang_start|>"), "LANG_END_ID": special("<|lang_end|>"),
            "INSTRUCT_START_ID": special("<|instruct_start|>"),
            "INSTRUCT_END_ID": special("<|instruct_end|>"),
            "TEXT_START_ID": special("<|text_start|>"), "TEXT_END_ID": special("<|text_end|>"),
            "NONE_IDS": ids("None"),
            "LANG_TOKEN_IDS": lang_ids, "LANG_TOKEN_OFFSETS": lang_offsets,
            "WEIGHT_STARTS": starts, "WEIGHT_VALUES": values,
            # `_estimate_target_tokens`' fallback reference, for a text with none.
            "FALLBACK_REF_TEXT": [ord(c) for c in "Nice to meet you."], "FALLBACK_REF_FRAMES": 25,
            "SPLIT_PUNCTUATION": sorted(ord(c) for c in text["split"]),
            "CLOSING_MARKS": sorted(ord(c) for c in text["closing"]),
            # Single characters only: the check is `text[-1] in END_PUNCTUATION`, which the two-
            # character "……" in that set can never satisfy.
            "END_PUNCTUATION": sorted(ord(c) for c in text["end"] if len(c) == 1),
            "INSTRUCT_CATEGORY": category,
        }
        for name, strings in (("ABBREVIATION", sorted(text["abbreviations"])), ("TAG", text["tags"]),
                              ("INSTRUCT_EN", en), ("INSTRUCT_ZH", zh),
                              ("ACCENT", sorted(text["accents"])), ("DIALECT", sorted(text["dialects"]))):
            offsets, data = _byte_table(strings)
            values_out[f"{name}_OFFSETS"] = offsets
            values_out[f"{name}_BYTES"] = data
        return values_out

    def languages(self) -> list:
        """The language codes the reference accepts (`LANG_IDS`), sorted: a host's `language=` is a
        1-based index into this list, 0 meaning none."""
        from omnivoice.utils.lang_map import LANG_IDS

        return sorted(LANG_IDS)

    def hparams(self) -> dict:
        # `tts.reference`: a host's `reference=` passes the clip, its transcript and the text as three
        # inputs (the driver joins them the reference's way). `tts.instruct`: the file designs a voice
        # from words. Both are hparams by ADR-020's split: they say how the host builds the door's inputs.
        return {"hop_length": self._hop, "tts.reference": "transcript", "tts.instruct": "text"}

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = self._sample_rate or 24000
        if self.model_dir:
            contract["languages"] = self.languages()
        return contract

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                    tokenizer_dir=self.model_dir)


def _upstream_text_tables() -> dict:
    """The reference's own character sets and word lists, read off its modules rather than copied."""
    import omnivoice.models.omnivoice as om
    from omnivoice.utils import text as text_utils
    from omnivoice.utils import voice_design as vd

    tags = om._NONVERBAL_PATTERN.pattern
    inner = tags[tags.index("(") + 1:tags.rindex(")")]
    categories = [c for c in vd._INSTRUCT_CATEGORIES if isinstance(c, dict)]
    flat = [c for c in vd._INSTRUCT_CATEGORIES if not isinstance(c, dict)]
    return dict(
        split=text_utils.SPLIT_PUNCTUATION, closing=text_utils.CLOSING_MARKS,
        end=text_utils.END_PUNCTUATION, abbreviations=text_utils.ABBREVIATIONS,
        tags=[f"[{t}]" for t in inner.replace("|\n", "|").split("|")],
        categories=categories, accents=flat[0], dialects=flat[1],
    )


def _duration_weight_runs():
    """`RuleDurationEstimator._get_char_weight` for every codepoint, as runs: `starts[i]` begins a run
    of weight `values[i]`. unicodedata's categories and the script table folded into one lookup the
    driver can binary-search."""
    from omnivoice.utils.duration import RuleDurationEstimator

    estimator = RuleDurationEstimator()
    starts, values = [], []
    for cp in range(0x110000):
        w = float(estimator._get_char_weight.__wrapped__(estimator, chr(cp)))
        if not values or values[-1] != w:
            starts.append(cp)
            values.append(w)
    return starts, values


def _byte_table(strings) -> tuple:
    offsets, data = [0], []
    for s in strings:
        data += list(s.encode("utf-8"))
        offsets.append(len(data))
    return offsets, data


def _is_omnivoice(path: Path) -> bool:
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists() or not (path / "audio_tokenizer").is_dir():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "omnivoice"


def _build_omnivoice(path: Path, output_path: str) -> OmniVoiceExportConfig:
    return OmniVoiceExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=OmniVoiceExportConfig,
        recognizers=[ModelRecognizer(name="omnivoice", detect=_is_omnivoice,
                                     build_config=_build_omnivoice)],
    ))
