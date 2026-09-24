"""Export Chatterbox (`ResembleAI/chatterbox`, the English checkpoint) -- family 9's fourth leaf, and the
first one that is an AR token LM and a flow-matching decoder in ONE file.

Chatterbox is two models in sequence:

    text -> T3 (Llama-520M, guided) -> S3 speech tokens (25 Hz)
         -> S3Gen (conformer -> guided 10-step ODE, in-filled after the voice's prompt) -> mel
         -> HiFT (NSF source + iSTFTNet) -> 24 kHz waveform

EXPORT-ROADMAP's correction 4 says families 9 and 10 are two stages of one pipeline for most of the
remaining TTS leaves, and this is the first of them here. Neither half needed a new template: T3 is
family 10's shape (a KV-cached decoder, a private second stream for guidance, a hand-written loop --
`dia_export`, `qwen3_tts_export`), and S3Gen is family 9's (`FlowMatchingSpec` with F5-TTS's caller
schedule, guidance and caller noise). What is new is only that they share one GGUF and one driver.

Six phases:
  - `t3_prefill_embed`: the voice conditioning (speaker vector, 150 prompt speech tokens through the
                        perceiver resampler, the emotion scalar), the text ids, and two BOS rows -> the
                        prefill's input embeddings. Called twice, once per guidance stream: the
                        unconditional one differs ONLY in its text embeddings being zeroed (their
                        POSITION embeddings stay), which is the `text_keep` input.
  - `t3_lm`:            the 30-layer Llama, KV-cached, returning the speech head's logits for the LAST
                        row only. Its unconditional twin `t3_lm_uncond` is an `extra_streams` alias with
                        a private cache (loom.cpp ADR-023), which is how Dia runs guidance too.
  - `t3_step_embed`:    one generated token -> `speech_emb(token) + speech_pos_emb(position)`, the next
                        step's input for both streams.
  - `flow_encoder`:     prompt tokens + generated tokens -> `mu` (`[2n, 80]`, frame-major) and the
                        projected speaker vector `spks`. The 6+4-block espnet conformer with its 2x
                        upsample in the middle.
  - `estimator`:        one velocity evaluation of the causal Matcha-style U-Net. The ENGINE runs it
                        twice per step under guidance (loom.cpp ADR-040), on the cosine schedule.
  - `vocoder`:          HiFT: F0 predictor, the NSF sine source, and the iSTFTNet decoder. Its two
                        random draws are INPUTS (see `HiftVocoderPhase`).

**The watermark is deliberately absent.** The reference runs Resemble's Perth implicit watermarker over
every output; it is a separate neural model applied after synthesis, and loom does not ship it -- for
local inference and dev kits a smaller, faster model is the target. Decided 2026-09-23 and stated on
the model card, so no reader is told the audio is watermarked.

**The built-in voice travels in the GGUF** as `driver_weights` (`conds.pt`, the checkpoint's own default
voice), read by the driver with `loom.get_weight` when a caller passes none -- Kokoro's and Supertonic's
arrangement. That is what makes `Text2Speech.infer(text)` work with no reference clip, which is the door
F5-TTS does not have yet. Cloning a new voice needs three more models (the voice encoder, the S3
tokenizer, CAMPPlus) and is not in this export.

Usage:
  loom-export ~/Dev/models/chatterbox -o chatterbox.gguf --task text-to-speech --model chatterbox
"""
import importlib.metadata as _metadata
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .flow_matching_export import FlowMatchingSpec
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Axis, Unchecked

# Where the reference checkout lives. `chatterbox-tts` on PyPI pins torch 2.6, transformers 5.2 and
# gradio for a handful of `nn.Module`s -- the trade `f5_tts_export` and `matcha_export` record, with the
# same resolution: a git clone on `sys.path`.
CHATTERBOX_REPO = "/home/flavio/Dev/chatterbox/src"

SAMPLE_RATE = 24000
N_MEL = 80
# Mel frames per speech token: S3 tokens are 25 Hz and S3Gen's mel is 50 Hz.
TOKEN_MEL_RATIO = 2
# Output samples per mel frame: HiFT's upsample rates (8, 5, 3) times its iSTFT hop (4).
SAMPLES_PER_FRAME = 480
# The vocabulary of speech tokens S3Gen can embed; T3's head is wider (8194) and ids at or above this
# are control tokens, which the reference drops before S3Gen sees them.
SPEECH_VOCAB = 6561
START_SPEECH, STOP_SPEECH = 6561, 6562
START_TEXT, STOP_TEXT = 255, 0
# `T3Config.speech_cond_prompt_len`: how many of the voice's own speech tokens condition T3.
COND_PROMPT_LEN = 150
# `CFM_PARAMS` and `S3Token2Wav.flow_inference`'s defaults.
FLOW_STEPS = 10
FLOW_CFG_RATE = 0.7
# `ChatterboxTTS.generate`'s defaults, which is the only place they are written down.
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MIN_P = 0.05
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_EXAGGERATION = 0.5
DEFAULT_MAX_NEW_TOKENS = 1000
# How many positions T3's KV cache holds. The prefix is 36 rows plus the text; the default generation
# budget is 1000 tokens. 2048 covers both with a long sentence to spare, and costs 30 layers x 2 x 1024
# x 2048 x 4 bytes = 480 MB per stream -- two streams under guidance.
T3_MAX_POSITIONS = 2048
# The two BOS rows at the end of every prefill: `prepare_input_embeds` appends one (its
# `speech_tokens` argument is `[BOS]`) and `T3.inference` appends a second before its first forward.
# Reproduced as the reference does it, not deduplicated.
N_BOS_ROWS = 2
# Rows ahead of the text in the prefill: speaker (1) + perceiver queries (32) + emotion (1).
N_COND_ROWS = 34

# The trace lengths. Odd and distinct from every static dimension in the graphs, so no fusion can
# confuse a sequence axis with a head or channel axis (the Qwen3-TTS lesson about length 8).
TRACE_TEXT = 13
TRACE_TOKENS = 23
TRACE_FRAMES = 2 * TRACE_TOKENS
TRACE_STEPS = 7


def import_chatterbox():
    """Put the reference checkout on `sys.path`, with its two import-time side effects neutralised.

    `chatterbox/__init__.py` reads its own installed version, and a checkout is not installed; and
    `tts.py` imports `perth`, the watermarker this export deliberately does not ship."""
    if not getattr(_metadata, "_loom_chatterbox_patched", False):
        real_version = _metadata.version
        _metadata.version = lambda name: "0.0.0" if name == "chatterbox-tts" else real_version(name)
        _metadata._loom_chatterbox_patched = True
    if "perth" not in sys.modules:
        perth = types.ModuleType("perth")

        class _NoWatermark:
            def apply_watermark(self, wav, sample_rate):
                return wav

        perth.PerthImplicitWatermarker = _NoWatermark
        sys.modules["perth"] = perth
    if CHATTERBOX_REPO not in sys.path:
        sys.path.insert(0, CHATTERBOX_REPO)


def install_patches() -> None:
    """The two rewrites the reference needs before it traces. Both are exact.

    **Llama's `rotate_half`**, for Dia's reason verbatim (`dia_export.install_rotate_half_patch`): a
    `x.shape[-1] // 2` slice bound traces to `aten::floor_divide` feeding `aten::Int`, which
    coremltools cannot convert. `chunk` asks for a count.

    **espnet's `rel_shift` crop**, for the same reason one module over: it keeps
    `x[..., : x.size(-1) // 2 + 1]` of a `(.., time1, 2*time1 - 1)` score matrix. For self-attention
    that bound IS `time1`, the query length, which the tensor carries as its own axis -- so the crop is
    spelled with that axis rather than with arithmetic on the other one. It is only valid when query
    and key lengths agree, which every call here satisfies (no cache, self-attention), and the patched
    body checks it outside tracing.
    """
    from transformers.models.llama import modeling_llama
    from chatterbox.models.s3gen.transformer import attention

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_llama.rotate_half = rotate_half

    def rel_shift(self, x):
        if not torch.jit.is_tracing():
            assert x.size(-1) == 2 * x.size(2) - 1, (
                f"rel_shift expects (.., time1, 2*time1-1) scores, got {tuple(x.shape)}")
        zero_pad = torch.zeros_like(x[..., :1])
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        # `view` with the four sizes rather than `view_as`, which coremltools has no handler for.
        return x_padded[:, :, 1:].reshape(x.size(0), x.size(1), x.size(2), x.size(3))[:, :, :, : x.size(2)]

    attention.RelPositionMultiHeadedAttention.rel_shift = rel_shift


def fold_weight_norm(module: nn.Module) -> None:
    """Fold every `torch.nn.utils.parametrizations.weight_norm` into a plain weight.

    HiFT and its F0 predictor are built with the parametrization form, so the trace would otherwise
    carry `g * v / ||v||` for every convolution at every call. Folding is exact (the reference computes
    the same product each forward) and is what every vocoder export here does before tracing."""
    from torch.nn.utils import parametrize

    for sub in module.modules():
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)


# ------------------------------------------------------------------------------------------- T3 --

class T3PrefillEmbedPhase(nn.Module):
    """`(speaker_emb, prompt_tokens, emotion_adv, text_ids, text_keep) -> the prefill's embeddings`.

    `T3.prepare_input_embeds` plus the second BOS row `T3.inference` appends, for one stream at a time.
    `text_keep` is 1.0 for the conditional stream and 0.0 for the unconditional one: the reference
    zeroes `text_emb[1]` BEFORE the position embeddings are added, so the unconditional prefill still
    carries the text's positions -- a multiply on the token embeddings alone reproduces exactly that.

    The two learned position tables are sliced by length rather than looked up with an `arange`; the
    prompt's rows are a fixed `[0, 150)` and the BOS rows are both position 0.
    """

    def __init__(self, t3):
        super().__init__()
        self.cond_enc = t3.cond_enc
        self.text_emb = t3.text_emb
        self.speech_emb = t3.speech_emb
        self.register_buffer("text_pos", t3.text_pos_emb.emb.weight.detach().clone())
        self.register_buffer("prompt_pos",
                             t3.speech_pos_emb.emb.weight[:COND_PROMPT_LEN].detach().clone())
        bos = t3.speech_emb.weight[START_SPEECH] + t3.speech_pos_emb.emb.weight[0]
        self.register_buffer("bos_rows", bos.detach().clone().expand(1, N_BOS_ROWS, -1).contiguous())

    def forward(self, speaker_emb, prompt_tokens, emotion_adv, text_ids, text_keep):
        ce = self.cond_enc
        spk = ce.spkr_enc(speaker_emb).unsqueeze(1)                       # (1, 1, 1024)
        prompt = self.speech_emb(prompt_tokens) + self.prompt_pos         # (1, 150, 1024)
        prompt = ce.perceiver(prompt)                                     # (1, 32, 1024)
        emotion = ce.emotion_adv_fc(emotion_adv).unsqueeze(1)             # (1, 1, 1024)
        text = self.text_emb(text_ids) * text_keep + self.text_pos[: text_ids.shape[1]]
        return torch.cat([spk, prompt, emotion, text, self.bos_rows], dim=1)


class T3LMPhase(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> speech logits of the LAST row`, `(1, 1, 8194)`.

    The whole head, not a trimmed one (compare `qwen3_tts_export._TalkerWrapper`): the reference draws
    over all 8194 ids and FEEDS BACK whatever it drew, control ids included, dropping them only on the
    way to S3Gen. A trimmed head would be a different sampler.

    Last row only, because only the last row is ever sampled; the prefill would otherwise project every
    prefix row through a 1024 x 8194 head for nothing.
    """

    def __init__(self, t3):
        super().__init__()
        self.model = t3.tfmr
        self.head = t3.speech_head

    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                            attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.head(hidden[:, -1:])


class T3StepEmbedPhase(nn.Module):
    """`(token, position) -> speech_emb(token) + speech_pos_emb(position)`, one row."""

    def __init__(self, t3):
        super().__init__()
        self.speech_emb = t3.speech_emb
        self.speech_pos = t3.speech_pos_emb.emb

    def forward(self, token, position):                                   # (1, 1) i32, (1, 1) i32
        return self.speech_emb(token) + self.speech_pos(position)


# ---------------------------------------------------------------------------------------- S3Gen --

class FlowEncoderPhase(nn.Module):
    """`(tokens, embedding) -> (mu, spks)`: `CausalMaskedDiffWithXvec.inference` up to the decoder.

    `tokens` is the voice's prompt tokens followed by the generated ones -- the reference concatenates
    them before embedding, so the conformer sees one sequence and the in-fill boundary is invisible to
    it. `mu` comes back FRAME-major, `(1, 2n, 80)`, because the ODE's state is frame-major (so that the
    driver's slice of the generated frames is a contiguous suffix -- F5-TTS's Retro-052 lesson about
    choosing ONE layout at every join).

    The encoder is re-spelled for one unpadded utterance: every mask the reference builds is all ones,
    and `forward_attention` treats a `(0, 0, 0)` mask as "no mask" by design, so that is what each
    layer is handed. Nothing is approximated -- the masked branches fill nothing when nothing is
    padded.
    """

    def __init__(self, flow):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.spk_affine = flow.spk_embed_affine_layer
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj

    def forward(self, tokens, embedding):                                 # (1, n) i32, (1, 192)
        spks = self.spk_affine(F.normalize(embedding, dim=1))             # (1, 80)
        enc = self.encoder
        no_mask = torch.ones((0, 0, 0), dtype=torch.bool)
        xs, pos_emb, _ = enc.embed(self.input_embedding(tokens), no_mask)
        xs = enc.pre_lookahead_layer(xs)
        for layer in enc.encoders:
            xs, _, _, _ = layer(xs, no_mask, pos_emb, no_mask)
        xs = xs.transpose(1, 2)
        up = enc.up_layer
        xs = F.interpolate(xs, scale_factor=float(up.stride), mode="nearest")
        xs = F.pad(xs, (up.stride * 2, 0), value=0.0)
        xs = up.conv(xs).transpose(1, 2)
        xs, pos_emb, _ = enc.up_embed(xs, no_mask)
        for layer in enc.up_encoders:
            xs, _, _, _ = layer(xs, no_mask, pos_emb, no_mask)
        xs = enc.after_norm(xs)
        return self.encoder_proj(xs), spks                                # (1, 2n, 80), (1, 80)


def _causal_block(block, x):
    """`CausalBlock1D.forward` with its all-ones mask multiplies dropped (see `matcha_export`'s
    `_block1d_forward_nomask`, which found that even constructing the mask traces badly)."""
    return block.block(x)


def _causal_resnet(resnet, x, t_emb):
    h = _causal_block(resnet.block1, x)
    h = h + resnet.mlp(t_emb).unsqueeze(-1)
    h = _causal_block(resnet.block2, h)
    return h + resnet.res_conv(x)


class EstimatorPhase(nn.Module):
    """`(x, mu, spks, cond, t) -> dx/dt`, all frame-major: `ConditionalDecoder.forward` for one
    unpadded utterance, transposed in and out.

    `matcha_export._decoder_forward_traceable`'s rewrite on S3Gen's causal variant: the masks are all
    ones, so the resnet/Block1D multiplies are dropped and the transformer blocks get
    `attention_mask=None` (`static_chunk_size` is 0, so the reference's own attention mask is a
    padding mask and nothing else). This checkpoint has ONE down level whose "downsample" is a causal
    conv of stride 1, so the up path's `x[:, :, :skip.shape[-1]]` crop is the identity and is omitted.

    `spks` is broadcast over time by ADDITION to a zero tensor shaped like `mu` rather than by einops'
    `repeat`, which traces to a dynamic `tile` (Kokoro's `duration_style_concat` lesson).
    """

    def __init__(self, decoder):
        super().__init__()
        if len(decoder.down_blocks) != 1 or decoder.meanflow:
            raise NotImplementedError(
                f"this S3Gen decoder has {len(decoder.down_blocks)} down levels (meanflow="
                f"{decoder.meanflow}); the export's up path assumes the English checkpoint's single "
                f"stride-1 level, where the skip crop is the identity. A deeper U-Net needs the crop.")
        self.d = decoder

    def forward(self, x, mu, spks, cond, t):
        d = self.d
        x, mu, cond = x.transpose(1, 2), mu.transpose(1, 2), cond.transpose(1, 2)
        t_emb = d.time_mlp(d.time_embeddings(t))
        spk = torch.zeros_like(mu) + spks.unsqueeze(-1)
        h = torch.cat([x, mu, spk, cond], dim=1)                          # (1, 320, T)
        (resnet, blocks, downsample), = d.down_blocks
        h = _causal_resnet(resnet, h, t_emb)
        h = h.transpose(1, 2)
        for block in blocks:
            h = block(hidden_states=h, attention_mask=None, timestep=t_emb)
        skip = h.transpose(1, 2)
        h = downsample(skip)
        for resnet, blocks in d.mid_blocks:
            h = _causal_resnet(resnet, h, t_emb)
            h = h.transpose(1, 2)
            for block in blocks:
                h = block(hidden_states=h, attention_mask=None, timestep=t_emb)
            h = h.transpose(1, 2)
        (resnet, blocks, upsample), = d.up_blocks
        h = _causal_resnet(resnet, torch.cat([h, skip], dim=1), t_emb)
        h = h.transpose(1, 2)
        for block in blocks:
            h = block(hidden_states=h, attention_mask=None, timestep=t_emb)
        h = upsample(h.transpose(1, 2))
        h = _causal_block(d.final_block, h)
        return d.final_proj(h).transpose(1, 2)                            # (1, T, 80)


class HiftVocoderPhase(nn.Module):
    """`(mel, nsf_phase, nsf_noise) -> waveform`: `HiFTGenerator.inference` plus S3Gen's 40 ms fade-in.

    **The NSF source's two random draws are INPUTS.** `SineGen` draws a phase per harmonic (the
    fundamental's is pinned to 0) and a unit Gaussian per harmonic per SAMPLE; torch's RNG and the
    engine's are different algorithms, so the only way to reproduce a reference waveform is to hand it
    the reference's draws -- F5-TTS's `caller_noise` lesson, applied to a vocoder. The driver draws both
    when a caller does not supply them.

    Three substitutions, each for a named reason and each exact:
      * `% 1` is `x - floor(x)`: coremltools lowers `remainder(x, 1)` to `sub(x, x)`, which is always
        zero (`kokoro_export._f02sine_traceable` found it; it would silence the whole source).
      * `f0 > threshold` is `clamp((f0 - threshold) * 1e30, 0, 1)`: the exporter maps no `greater`.
        Exact for every f32 input -- the nearest f32 neighbours of 10 are ~9.5e-7 away, which the
        factor takes past 1.
      * the harmonic "outer product" is a concatenation of per-harmonic scales, because `ggml_mul`
        cannot broadcast along two axes at once (Kokoro again).
    `torch.stft` of the source is two strided convolutions over a reflect-padded signal (real and
    imaginary), and `torch.istft` is this project's `ISTFT`.
    """

    def __init__(self, hift, trim_fade):
        super().__init__()
        from .istft import ISTFT

        self.h = hift
        src = hift.m_source
        self.harmonics = src.l_sin_gen.harmonic_num + 1
        self.sine_amp = float(src.l_sin_gen.sine_amp)
        self.noise_std = float(src.l_sin_gen.noise_std)
        self.voiced_threshold = float(src.l_sin_gen.voiced_threshold)
        self.sr = float(src.l_sin_gen.sampling_rate)
        n_fft = hift.istft_params["n_fft"]
        hop = hift.istft_params["hop_len"]
        self.n_fft, self.hop = n_fft, hop
        window = hift.stft_window.to(torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        angle = 2 * torch.pi * k * n / n_fft
        self.register_buffer("stft_re", (window * torch.cos(angle)).float()[:, None, :])
        self.register_buffer("stft_im", (-window * torch.sin(angle)).float()[:, None, :])
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
        self.register_buffer("trim_fade", trim_fade.detach().clone().float())

    def source(self, f0, nsf_phase, nsf_noise):
        """`SourceModuleHnNSF.forward` -> `(1, 1, n_samples)`."""
        f0 = F.interpolate(f0.unsqueeze(1), scale_factor=float(SAMPLES_PER_FRAME), mode="nearest")
        f_mat = torch.cat([f0 * ((i + 1) / self.sr) for i in range(self.harmonics)], dim=1)
        c = torch.cumsum(f_mat, dim=-1)
        theta = 2 * torch.pi * (c - torch.floor(c))
        sine = self.sine_amp * torch.sin(theta + nsf_phase)
        uv = torch.clamp((f0 - self.voiced_threshold) * 1e30, 0.0, 1.0)
        noise_amp = uv * self.noise_std + (1 - uv) * (self.sine_amp / 3)
        sine = sine * uv + noise_amp * nsf_noise                          # (1, 9, n)
        lin = self.h.m_source.l_linear
        merged = torch.tanh(F.conv1d(sine, lin.weight.unsqueeze(-1), lin.bias))
        return merged                                                     # (1, 1, n)

    def forward(self, mel, nsf_phase, nsf_noise):
        # mel (1, m, 80) frame-major; nsf_phase (1, 9, 1); nsf_noise (1, 9, 480 m)
        h = self.h
        x = mel.transpose(1, 2)                                           # (1, 80, m)
        f0 = h.f0_predictor(x)                                            # (1, m)
        s = self.source(f0, nsf_phase, nsf_noise).squeeze(1)              # (1, n)
        pad = self.n_fft // 2
        s = F.pad(s.unsqueeze(1), (pad, pad), mode="reflect")
        s_stft = torch.cat([F.conv1d(s, self.stft_re, stride=self.hop),
                            F.conv1d(s, self.stft_im, stride=self.hop)], dim=1)
        x = h.conv_pre(x)
        for i in range(h.num_upsamples):
            x = F.leaky_relu(x, h.lrelu_slope)
            x = h.ups[i](x)
            if i == h.num_upsamples - 1:
                x = h.reflection_pad(x)
            si = h.source_resblocks[i](h.source_downs[i](s_stft))
            x = x + si
            xs = None
            for j in range(h.num_kernels):
                r = h.resblocks[i * h.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / h.num_kernels
        x = F.leaky_relu(x)
        x = h.conv_post(x)
        half = self.n_fft // 2 + 1
        magnitude = torch.clip(torch.exp(x[:, :half, :]), max=1e2)
        phase = torch.sin(x[:, half:, :])
        wave = self.istft(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        wave = torch.clamp(wave, -h.audio_limit, h.audio_limit).reshape(1, -1)
        n_fade = self.trim_fade.shape[0]
        return torch.cat([wave[:, :n_fade] * self.trim_fade, wave[:, n_fade:]], dim=1)


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `create_causal_mask` passes straight through (see
    `dia_export.causal_mask`, whose reason this is)."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def _load_reference(model_dir: str):
    """`ChatterboxTTS.from_local`, which is the only loader that knows the checkpoint's file layout,
    plus the patches that must be in place before any trace."""
    import_chatterbox()
    from chatterbox.tts import ChatterboxTTS

    install_patches()
    tts = ChatterboxTTS.from_local(model_dir, "cpu")
    if tts.conds is None:
        raise FileNotFoundError(
            f"{model_dir} has no `conds.pt`. It is the checkpoint's built-in voice, which this export "
            f"ships as the default so that `infer(text)` needs no reference clip; without it every "
            f"caller would need a voice-cloning path this export does not have.")
    fold_weight_norm(tts.s3gen.mel2wav)
    return tts


@dataclass(kw_only=True)
class ChatterboxExportConfig(BaseMultiPhaseModelExportConfig):
    """A `ResembleAI/chatterbox` directory (the English `t3_cfg` + `s3gen` pair) -> one Loom GGUF."""

    architecture: str = "chatterbox"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "chatterbox_driver"
    _voice: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _trim_fade_len: int = field(default=0, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked(
            "path to the checkpoint directory; the recognizer already found `t3_cfg.safetensors`, "
            "`s3gen.safetensors`, `tokenizer.json` and `conds.pt` in it"),
        "decomposition": Unchecked("MultiPhase by construction -- six graphs and two loops"),
        "driver_script_path": Unchecked("the hand-written fragments here are still parsed and "
                                         "checked against the traced topologies by LuaFragment"),
        "_voice": Unchecked(
            "READ off `conds.pt` during phases(); shipped as driver weights and shape-checked there"),
        "_trim_fade_len": Unchecked("read off S3Gen's own `trim_fade` buffer during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        tts = _load_reference(self.model_dir)
        t3, s3 = tts.t3, tts.s3gen
        conds = tts.conds
        self._voice = {
            "voice.speaker_emb": conds.t3.speaker_emb.reshape(-1).detach().float().numpy(),
            "voice.cond_prompt_tokens": conds.t3.cond_prompt_speech_tokens.reshape(-1).detach().float().numpy(),
            "voice.prompt_token": conds.gen["prompt_token"].reshape(-1).detach().float().numpy(),
            # Frame-major, `(n_prompt_frames, 80)` flattened: the layout the estimator's `cond` wants.
            "voice.prompt_feat": conds.gen["prompt_feat"][0].reshape(-1).detach().float().numpy(),
            "voice.embedding": conds.gen["embedding"].reshape(-1).detach().float().numpy(),
        }
        self._check_voice(t3)
        self._trim_fade_len = int(s3.trim_fade.shape[0])
        hidden = t3.cfg.hidden_size

        text_dim = ct.RangeDim(1, t3.hp.max_text_tokens)
        seq_dim = ct.RangeDim(1, T3_MAX_POSITIONS)
        code_dim = ct.RangeDim(4, t3.hp.max_speech_tokens)
        frame_dim = ct.RangeDim(8, 2 * t3.hp.max_speech_tokens)
        gen_dim = ct.RangeDim(2, 2 * t3.hp.max_speech_tokens)
        harmonics = s3.mel2wav.m_source.l_sin_gen.harmonic_num + 1
        return [
            ExportPhase(
                name="t3_prefill_embed",
                wrapper=T3PrefillEmbedPhase(t3).eval(),
                dummy_inputs=(torch.randn(1, t3.hp.speaker_embed_size),
                              torch.randint(0, SPEECH_VOCAB, (1, COND_PROMPT_LEN), dtype=torch.int32),
                              torch.tensor([[DEFAULT_EXAGGERATION]]),
                              torch.randint(1, t3.hp.text_tokens_dict_size, (1, TRACE_TEXT),
                                            dtype=torch.int32),
                              torch.tensor([1.0])),
                mil_inputs=[
                    ct.TensorType(name="speaker_emb", shape=(1, t3.hp.speaker_embed_size),
                                  dtype=np.float32),
                    ct.TensorType(name="prompt_tokens", shape=(1, COND_PROMPT_LEN), dtype=np.int32),
                    ct.TensorType(name="emotion_adv", shape=(1, 1), dtype=np.float32),
                    ct.TensorType(name="text_ids", shape=(1, text_dim), dtype=np.int32),
                    ct.TensorType(name="text_keep", shape=(1,), dtype=np.float32),
                ],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="t3_lm",
                wrapper=T3LMPhase(t3).eval(),
                dummy_inputs=(torch.zeros(1, TRACE_STEPS, hidden),
                              torch.arange(TRACE_STEPS).view(1, -1),
                              causal_mask(TRACE_STEPS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, seq_dim, hidden), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=T3_MAX_POSITIONS,
                # **A stream with its own KV cache.** Guidance runs the same decoder over a second
                # prefill (text embeddings zeroed) and the two caches then diverge for the whole
                # generation, so one cache cannot serve both -- Dia's `decoder_uncond`, exactly.
                extra_streams=("t3_lm_uncond",),
            ),
            ExportPhase(
                name="t3_step_embed",
                wrapper=T3StepEmbedPhase(t3).eval(),
                dummy_inputs=(torch.tensor([[START_SPEECH]], dtype=torch.int32),
                              torch.tensor([[3]], dtype=torch.int32)),
                mil_inputs=[ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
                            ct.TensorType(name="position", shape=(1, 1), dtype=np.int32)],
            ),
            ExportPhase(
                name="flow_encoder",
                wrapper=FlowEncoderPhase(s3.flow).eval(),
                dummy_inputs=(torch.randint(0, SPEECH_VOCAB, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.randn(1, 192)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, code_dim), dtype=np.int32),
                            ct.TensorType(name="embedding", shape=(1, 192), dtype=np.float32)],
                root_axis="n_codes",
            ),
            ExportPhase(
                name="estimator",
                wrapper=EstimatorPhase(s3.flow.decoder.estimator).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL), torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.randn(1, N_MEL), torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.tensor([0.3])),
                mil_inputs=[
                    ct.TensorType(name="x", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="mu", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="spks", shape=(1, N_MEL), dtype=np.float32),
                    ct.TensorType(name="cond", shape=(1, frame_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="t", shape=(1,), dtype=np.float32),
                ],
                # `n_tokens`, not `n_enc_frames`: `FlowMatchingSampler` binds its length under that name
                # (F5-TTS's estimator is declared the same way), and the vocoder below is where the
                # acoustic frame count gets its own name.
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="vocoder",
                wrapper=HiftVocoderPhase(s3.mel2wav, s3.trim_fade).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, N_MEL),
                              torch.rand(1, harmonics, 1),
                              torch.randn(1, harmonics, TRACE_FRAMES * SAMPLES_PER_FRAME)),
                mil_inputs=[
                    ct.TensorType(name="mel", shape=(1, gen_dim, N_MEL), dtype=np.float32),
                    ct.TensorType(name="nsf_phase", shape=(1, harmonics, 1), dtype=np.float32),
                    ct.TensorType(name="nsf_noise",
                                  shape=(1, harmonics, ct.RangeDim(2 * SAMPLES_PER_FRAME,
                                                                   2 * t3.hp.max_speech_tokens
                                                                   * SAMPLES_PER_FRAME)),
                                  dtype=np.float32),
                ],
                root_axis="n_enc_frames",
                # The noise's sample axis is the mel's frame axis times 480 -- one symbol, not two.
                declared_axes={"nsf_noise": {2: f"{SAMPLES_PER_FRAME} * n_enc_frames"}},
            ),
        ]

    def _check_voice(self, t3) -> None:
        """The built-in voice's shapes, against the model that will read them. A wrong-width voice
        reaches `loom.get_weight` flat and fails deep in the engine (`kokoro_export._default_voice`)."""
        want = {
            "voice.speaker_emb": t3.hp.speaker_embed_size,
            "voice.cond_prompt_tokens": COND_PROMPT_LEN,
            "voice.embedding": 192,
        }
        for name, width in want.items():
            if self._voice[name].size != width:
                raise ValueError(f"conds.pt's {name} has {self._voice[name].size} values, expected {width}")
        n_prompt_tokens = self._voice["voice.prompt_token"].size
        n_prompt_frames = self._voice["voice.prompt_feat"].size // N_MEL
        if n_prompt_frames != TOKEN_MEL_RATIO * n_prompt_tokens:
            raise ValueError(
                f"conds.pt's prompt is {n_prompt_tokens} tokens and {n_prompt_frames} mel frames; "
                f"S3Gen in-fills after the prompt's frames and needs exactly {TOKEN_MEL_RATIO} per "
                f"token, which is what `S3Token2Mel.embed_ref` trims a reference to.")

    def samplers(self) -> List[FlowMatchingSpec]:
        return [FlowMatchingSpec(
            func_name="sample_estimator",
            estimator="estimator",
            carried_input="x",
            time_input="t",
            fixed_inputs=["mu", "spks", "cond"],
            # `1 - cos(pi/2 * t)` over a linspace -- `CausalConditionalCFM.forward`'s cosine schedule.
            schedule="caller",
            # `(1 + rate) * v_cond - rate * v_uncond` with the unconditional run seeing mu, spks and
            # cond all zeroed: `solve_euler`'s own batch-of-two, which is ADR-040's form exactly.
            guidance=True,
            # See `HiftVocoderPhase`: a reference waveform is only reproducible from its own draws.
            caller_noise=True,
            note="Euler over S3Gen's causal U-Net on the cosine schedule, under classifier-free\n"
                 "guidance: the unconditional evaluation sees mu, spks and cond zeroed.",
        )]

    def driver_components(self) -> List:
        from .driver_components import (
            CALLER, DriverInputs, DriverReturn, ExportConstants, FlowMatchingSampler, LuaFragment,
            SubgraphCallComponent,
        )
        from .driver_ir import BinOp, FieldAccess, Len, OutputRef, Var

        fragment = self.driver_script_path
        n_frames, n_gen = Var("n_frames"), Var("n_gen_frames")
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True,
                        defines=("chatterbox_voice", "chatterbox_cosine_times", "chatterbox_zeros")),
            ExportConstants(values={
                "START_TEXT": START_TEXT, "STOP_TEXT": STOP_TEXT,
                "START_SPEECH": START_SPEECH, "STOP_SPEECH": STOP_SPEECH,
                "SPEECH_VOCAB": SPEECH_VOCAB,
                "N_COND_ROWS": N_COND_ROWS, "N_BOS_ROWS": N_BOS_ROWS,
                "T3_MAX_POSITIONS": T3_MAX_POSITIONS,
                "TOKEN_MEL_RATIO": TOKEN_MEL_RATIO, "N_MEL": N_MEL,
                "SAMPLES_PER_FRAME": SAMPLES_PER_FRAME,
                "NSF_HARMONICS": 9,
                "FLOW_STEPS": FLOW_STEPS, "FLOW_CFG_RATE": FLOW_CFG_RATE,
                # `ChatterboxTTS.generate`'s own defaults: the numbers every published sample used.
                "DEFAULT_CFG_WEIGHT": DEFAULT_CFG_WEIGHT,
                "DEFAULT_TEMPERATURE": DEFAULT_TEMPERATURE,
                "DEFAULT_MIN_P": DEFAULT_MIN_P,
                "DEFAULT_TOP_P": DEFAULT_TOP_P,
                "DEFAULT_REPETITION_PENALTY": DEFAULT_REPETITION_PENALTY,
                "DEFAULT_EXAGGERATION": DEFAULT_EXAGGERATION,
                "DEFAULT_MAX_NEW_TOKENS": DEFAULT_MAX_NEW_TOKENS,
            }),
            # The text's ids are the one required input; every other `inputs.*` the fragments read
            # (the voice, the sampling knobs, the draws) is optional and defaulted in place.
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "01_t3.lua",
                        reads=("tokens", "START_TEXT", "STOP_TEXT", "START_SPEECH",
                               "STOP_SPEECH", "SPEECH_VOCAB", "N_COND_ROWS", "N_BOS_ROWS",
                               "T3_MAX_POSITIONS", "DEFAULT_CFG_WEIGHT", "DEFAULT_TEMPERATURE",
                               "DEFAULT_MIN_P", "DEFAULT_TOP_P", "DEFAULT_REPETITION_PENALTY",
                               "DEFAULT_EXAGGERATION", "DEFAULT_MAX_NEW_TOKENS"),
                        defines=("speech_tokens",)),
            LuaFragment(fragment / "02_flow_plan.lua",
                        reads=("speech_tokens", "TOKEN_MEL_RATIO", "N_MEL", "FLOW_STEPS",
                               "FLOW_CFG_RATE"),
                        defines=("flow_tokens", "flow_embedding", "n_codes", "n_frames",
                                 "n_prompt_frames", "n_gen_frames", "step_cond", "zero_cond",
                                 "zero_spks", "times", "flow_cfg")),
            SubgraphCallComponent(
                topology="flow_encoder", outputs=(), retain=True, length=Var("n_codes"),
                inputs={"tokens": Var("flow_tokens"), "embedding": Var("flow_embedding")},
                note="--- mu (frame-major) and the projected speaker vector, both RETAINED: the\n"
                     "    sampler hands them to every conditional evaluation. ---"),
            FlowMatchingSampler(
                spec=self.samplers()[0], result="_mel", length=n_frames,
                n_elems=BinOp("*", n_frames, Var("N_MEL")),
                n_steps=None, times=Var("times"),
                step_inputs={"mu": OutputRef("flow_encoder", index=1),
                             "spks": OutputRef("flow_encoder", index=2),
                             "cond": Var("step_cond")},
                uncond_inputs={"mu": Var("zero_cond"), "spks": Var("zero_spks"),
                               "cond": Var("zero_cond")},
                guidance_scale=Var("flow_cfg"),
                state=FieldAccess("inputs", "noise"),
                note="--- The whole mel grid, in-filled after the voice's prompt frames. ---"),
            LuaFragment(fragment / "03_mel_tail.lua",
                        reads=("n_prompt_frames", "n_gen_frames", "N_MEL"),
                        defines=("mel_tail",), retains=("estimator",)),
            LuaFragment(fragment / "04_nsf.lua",
                        reads=("n_gen_frames", "NSF_HARMONICS", "SAMPLES_PER_FRAME"),
                        defines=("nsf_phase", "nsf_noise")),
            SubgraphCallComponent(
                topology="vocoder", outputs=("wave",), length=n_gen,
                inputs={"mel": Var("mel_tail"), "nsf_phase": Var("nsf_phase"),
                        "nsf_noise": Var("nsf_noise")},
                note="--- HiFT: the generated frames -> 24 kHz waveform, fade-in included. ---"),
            DriverReturn(values=("wave",)),
        ]

    def hparams(self) -> dict:
        return {"n_mel": N_MEL}

    def contract(self) -> dict:
        contract = super().contract()
        # Text, not phoneme ids: T3 reads its own BPE ids, and the table ships in the GGUF.
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        contract["tts.default_steps"] = FLOW_STEPS
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                      # Named rather than auto-detected: a `tokenizer.json` BPE is what "gpt2" looks
                      # like on disk, and this one is not byte-level.
                      tokenizer_dir=self.model_dir, tokenizer_family="chatterbox")
        if self._voice is not None:
            kwargs["driver_weights"] = dict(self._voice)
        return kwargs


def _is_chatterbox(path: Path) -> bool:
    """The English Chatterbox release: T3, S3Gen, the tokenizer and the built-in voice side by side.

    `t3_cfg.safetensors` is the discriminator -- the multilingual T3 ships as `t3_mtl23ls_*` and
    Chatterbox Turbo as a GPT-2 T3, and neither is this export."""
    return path.is_dir() and all((path / n).is_file() for n in (
        "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"))


def _build_chatterbox(path: Path, output_path: str) -> LoomExportConfig:
    return ChatterboxExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=ChatterboxExportConfig,
        recognizers=[
            ModelRecognizer(name="chatterbox", detect=_is_chatterbox, build_config=_build_chatterbox),
        ],
    ))
