"""Export Voxtral-4B-TTS (`mistralai/Voxtral-4B-TTS-2603`) -- family 9's eighth leaf: an autoregressive LM
over FRAMES of codes, each frame's acoustic half integrated by a small flow-matching transformer.

    prompt  [BOS] [BEGIN_AUDIO] voice rows [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
    LM      Ministral-3B (26 layers, 3072 wide, GQA 32/8, RoPE 1e6), KV-cached; its last row per frame
    frame   the flow head on that row: an argmax over the 8192-code semantic codebook (plus [END_AUDIO]),
            and 36 acoustic values from ONE unit draw, integrated over 7 Euler steps with CFG (alpha
            1.2), clamped and rounded to 21 levels. The frame's 37 codes, each through its codebook's
            embedding and summed, are the LM's next row; [END_AUDIO] ends the loop
    codec   the audio tokenizer's causal decoder: 12.5 Hz frames -> ALiBi sliding-window transformers
            and x2 transposed convolutions -> 100 Hz x 240-sample patches -> 24 kHz

Five phases, each a re-spelling of the upstream module it replaces, and checked against it:

  - `embed_prompt`: `(ids, voice, voice_mask) -> rows`. The prompt's text ids through the LM's table, with
                    the voice's rows where `voice_mask` is set -- vllm-omni's `tts_preprocess`, which
                    writes a preset voice's embeddings over the prompt's [AUDIO] slots.
  - `embed_frame`:  `codes [1, n, 37] -> rows`, the frame's codes through `audio_codebook_embeddings`
                    (one table, each codebook at its offset) and summed: `encode_tokens`.
  - `lm`:           the cached LM, `inputs_embeds -> its last row, normed`. Mistral's checkpoint rotates
                    INTERLEAVED pairs; the export permutes `wq`/`wk` to rotate-half, as vllm does on load,
                    because that is the spelling `fuse_loom_attention` recognises -- and asserts the
                    permuted stack equal to the native one before tracing.
  - `acoustic`:     `(hidden, noise, cfg) -> (semantic logits, 36 acoustic codes)`. The whole of
                    `decode_one_frame` in one graph, its seven Euler steps unrolled: the state is 36 floats,
                    the step count is fixed, and the update is f32 arithmetic the driver's LuaJIT cannot
                    reproduce (the Pocket-TTS lesson). The semantic draw is the driver's, an argmax over
                    `[END_AUDIO, codes]` (`argmax_row_range`), because the acoustic integration does not
                    read it -- `should_decode` only masks a frame the loop is about to stop at anyway.
  - `codec`:        codes `[1, T, 37]` -> the 24 kHz waveform, every frame in one call. The attention
                    masks (ALiBi plus a causal window of 2/4/8/16 positions per rate) are built from
                    `pos1`/`pos2`/`pos4`/`pos8` inputs the driver hands over, Pocket-TTS's way, rather
                    than from a shape. Long audio is decoded in chunks with left context by the driver;
                    see `CODEC_CONTEXT`.

**Voices are prompt rows, and the default ships in the file.** A preset voice (`voice_embedding/*.pt`) is
`[n, 3072]` embeddings that replace the prompt's n [AUDIO] slots. `DEFAULT_VOICE` is a driver weight; the
other nineteen are voice files (`voxtral_tts_voices`, loom.cpp ADR-045) whose `voice` tensor is the driver
input of that name. The open checkpoint has no codec ENCODER, so a voice cannot be cloned from a clip --
upstream says so too ("the open-source variant only supports preset voices").

**The reference is vllm-omni's own two module files**, imported with `vllm` stubbed out
(`import_upstream`): the flow head and the codec are plain torch there, and the export asserts each
wrapper equal to the module it re-spells, on real inputs, before tracing. The LM is vllm's
`MistralForCausalLM`, which vllm-omni does not carry; `MistralNative` below is it in the checkpoint's
own layout (loom.cpp `scripts/voxtral_tts_reference.py` runs the same code end to end).

Weights and voices are CC BY-NC 4.0 (the model card: the voices come from EARS, CML-TTS, IndicVoices-R
and the Arabic Natural Audio dataset, and the model inherits their licence).

Usage (4B parameters at F32: export it on a machine with ~40 GB of RAM):
  loom-export ~/Dev/models/voxtral-4b-tts-2603 -o voxtral_tts.gguf --task text-to-speech --model voxtral-tts
"""
import importlib.util
import json
import logging
import math
import os
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
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Axis, Unchecked

# vllm-omni's `model_executor/models/voxtral_tts` directory (Apache-2.0), read-only. `VLLM_OMNI_VOXTRAL`
# overrides it on a machine where the clone lives elsewhere.
VLLM_OMNI_SRC = os.environ.get(
    "VLLM_OMNI_VOXTRAL", "/home/flavio/Dev/vllm-omni/vllm_omni/model_executor/models/voxtral_tts")
UPSTREAM_MODULE = "vllm_omni.model_executor.models.voxtral_tts"

SAMPLE_RATE = 24000
# Waveform samples per 12.5 Hz frame: the codec's 240-sample patch times its x8 upsampling.
SAMPLES_PER_FRAME = 1920
# The deploy config's stage-0 defaults (`vllm_omni/deploy/voxtral_tts.yaml`): CFG alpha, and a frame
# budget of `max_tokens`. `n_decoding_steps` is absent from params.json and vllm-omni's parser sets 7.
DEFAULT_CFG = 1.2
N_DECODING_STEPS = 7
MAX_FRAMES = 2048
# The model card's own example voice.
DEFAULT_VOICE = "casual_male"
# How many positions the LM's cache holds: the deploy config's `max_model_len`. 26 layers x 2 x 4096 x
# 1024 x 4 bytes = 872 MB at F32.
LM_MAX_POSITIONS = 4096
# The codec decodes up to CODEC_CHUNK frames per call, each call re-reading CODEC_CONTEXT frames before
# it and dropping their samples. Its receptive field is finite (causal convolutions, windowed attention),
# so with enough context a chunked decode equals a one-call decode; measured on the reference, 16 frames
# of context still left 1.3e-4 and the field is ~18 frames, so 32 is the context used here.
CODEC_CHUNK = 256
CODEC_CONTEXT = 32
# Mistral's special ids (tekken.json's markers). Checked against the tokenizer file at export.
BOS, AUDIO, BEGIN_AUDIO = 1, 24, 25
REPEAT_AUDIO_TEXT, NEXT_AUDIO_TEXT = 35, 36
# `AudioSpecialTokens`: codes 0 and 1 are [EMPTY_AUDIO] and [END_AUDIO], and every real code is +2 --
# so an [EMPTY_AUDIO] acoustic code, as the flow head writes it into an [END_AUDIO] frame, is 0 + 2.
END_AUDIO, N_AUDIO_SPECIAL = 1, 2

TRACE_TOKENS = 7    # not 8: the GQA fusion must tell 8 K/V heads from the sequence axis
TRACE_FRAMES = 5


# ----------------------------------------------------------------------------- upstream, imported --

def _stub_vllm() -> None:
    """Just enough of `vllm.*` for vllm-omni's two Voxtral module files to import: every name they
    mention at import time. None of it runs in the paths the export uses."""
    if "vllm.config" in sys.modules and getattr(sys.modules["vllm.config"], "_loom_stub", False):
        return

    class _Anything:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, *args, **kwargs):
            pass

    def mod(name, **attrs):
        m = sys.modules.get(name) or types.ModuleType(name)
        m.__path__ = []
        m._loom_stub = True
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    def default_weight_loader(param, loaded_weight):
        if param.shape != loaded_weight.shape:
            raise ValueError(f"weight shape {tuple(loaded_weight.shape)} for a {tuple(param.shape)} parameter")
        param.data.copy_(loaded_weight)

    class _Registry:
        def register_processor(self, *args, **kwargs):
            return lambda cls: cls

    anything = lambda *names: {n: type(n, (_Anything,), {}) for n in names}  # noqa: E731
    mod("vllm")
    mod("vllm.config", **anything("VllmConfig"))
    mod("vllm.inputs", MultiModalDataDict=dict)
    mod("vllm.logger", init_logger=logging.getLogger)
    mod("vllm.model_executor")
    mod("vllm.model_executor.model_loader")
    mod("vllm.model_executor.model_loader.weight_utils", default_weight_loader=default_weight_loader)
    mod("vllm.model_executor.models")
    mod("vllm.model_executor.models.interfaces", **anything("SupportsMultiModal"))
    mod("vllm.model_executor.models.utils", flatten_bn=None, init_vllm_registered_model=None,
        maybe_prefix=lambda p, n: f"{p}.{n}" if p else n)
    mod("vllm.multimodal", MULTIMODAL_REGISTRY=_Registry())
    mod("vllm.multimodal.inputs", **anything("MultiModalFieldConfig", "MultiModalKwargsItems"),
        NestedTensors=object)
    mod("vllm.multimodal.parse", **anything("AudioProcessorItems", "MultiModalDataItems", "MultiModalDataParser"))
    mod("vllm.multimodal.processing", **anything("BaseDummyInputsBuilder", "BaseMultiModalProcessor"))
    mod("vllm.multimodal.processing.processor", **anything("BaseProcessingInfo", "ProcessorInputs",
                                                           "PromptReplacement", "PromptUpdate"))
    mod("vllm.sequence", **anything("IntermediateTensors"))
    mod("vllm.tokenizers", cached_tokenizer_from_config=None)
    mod("vllm.tokenizers.mistral", **anything("MistralTokenizer"))
    for name in ("vllm_omni", "vllm_omni.model_executor", "vllm_omni.model_executor.models", UPSTREAM_MODULE,
                 "vllm_omni.quantization"):
        mod(name)
    mod("vllm_omni.quantization.component_config", **anything("ComponentQuantizationConfig"))
    mod("vllm_omni.platforms", current_omni_platform=types.SimpleNamespace(device_type="cpu"))
    # `mistral_common` is imported at module scope for the prompt processor, which the export never
    # runs. Real when installed (the reference script needs it); a stub otherwise, since installing it
    # would move pydantic and numpy under the export venv's other pins.
    try:
        import mistral_common.protocol.instruct.chunk  # noqa: F401
        import mistral_common.tokens.tokenizers.audio  # noqa: F401
    except ImportError:
        for name in ("mistral_common", "mistral_common.protocol", "mistral_common.protocol.instruct",
                     "mistral_common.tokens", "mistral_common.tokens.tokenizers"):
            mod(name)
        mod("mistral_common.protocol.instruct.chunk", **anything("AudioChunk", "RawAudio"))
        mod("mistral_common.tokens.tokenizers.audio", **anything("Audio", "AudioEncoder"))


def import_upstream(src: str = VLLM_OMNI_SRC):
    """vllm-omni's `voxtral_tts_audio_generation` and `voxtral_tts_audio_tokenizer`, as they are."""
    src_dir = Path(src)
    if not (src_dir / "voxtral_tts_audio_generation.py").is_file():
        raise FileNotFoundError(f"vllm-omni's voxtral_tts modules are not at {src_dir}; clone "
                                "github.com/vllm-project/vllm-omni and set VLLM_OMNI_VOXTRAL")
    _stub_vllm()
    loaded = []
    for name in ("voxtral_tts_audio_generation", "voxtral_tts_audio_tokenizer"):
        full = f"{UPSTREAM_MODULE}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, src_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[full]        # a half-imported module would answer the next import
                raise
        loaded.append(sys.modules[full])
    return loaded[0], loaded[1]


def read_params(model_dir: str) -> dict:
    return json.loads((Path(model_dir) / "params.json").read_text())


def audio_model_args(params: dict) -> dict:
    args = json.loads(json.dumps(params["multimodal"]["audio_model_args"]))
    if args["acoustic_transformer_args"].get("n_decoding_steps") is None:
        args["acoustic_transformer_args"]["n_decoding_steps"] = N_DECODING_STEPS
    return args


def load_checkpoint(model_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {k: v.to(torch.float32) for k, v in load_file(str(Path(model_dir) / "consolidated.safetensors")).items()}


def load_upstream(model_dir: str, weights: Dict[str, torch.Tensor], src: str = VLLM_OMNI_SRC,
                  dtype=torch.float32):
    """`(flow, codec)`: vllm-omni's `FlowMatchingAudioTransformer` and `VoxtralTTSAudioTokenizer` at
    `dtype` with the checkpoint's weights, loaded through their own `load_weight`."""
    gen_mod, tok_mod = import_upstream(src)
    params = read_params(model_dir)
    args = audio_model_args(params)
    flow = gen_mod.FlowMatchingAudioTransformer(json.loads(json.dumps(args)))
    flow_params = dict(flow.named_parameters())
    loaded = set()
    for name, w in weights.items():
        if name.startswith("acoustic_transformer."):
            short = name[len("acoustic_transformer."):]
            if short not in flow_params:
                raise KeyError(f"{name} is not a flow-head parameter")
            loaded.add(flow.load_weight((short, w)))
    if any(k.startswith("acoustic_transformer.") for k in weights) and loaded != set(flow_params):
        raise KeyError(f"flow-head parameters with no checkpoint tensor: {sorted(set(flow_params) - loaded)}")
    hf_config = types.SimpleNamespace(
        audio_config={"codec_args": params["multimodal"]["audio_tokenizer_args"], "audio_model_args": args},
        text_config=types.SimpleNamespace(hidden_size=params["dim"]))
    codec = tok_mod.VoxtralTTSAudioTokenizer(
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=hf_config)))
    for name, w in weights.items():
        if name.startswith("audio_tokenizer."):
            codec.load_weight((name[len("audio_tokenizer."):], w))
        elif name.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
            codec.load_weight(("audio_token_embedding.embeddings.weight", w))
    flow, codec = flow.to(dtype).eval(), codec.to(dtype).eval()
    # The semantic centroids are plain attributes `load_weight` set, which `.to` does not move.
    sc = codec.quantizer.semantic_codebook
    sc.embedding_sum, sc.cluster_usage, sc._embedding = sc.embedding_sum.to(dtype), sc.cluster_usage.to(dtype), None
    return flow, codec


def fold_weight_norm(module: nn.Module) -> None:
    """The codec's convolutions use `parametrizations.weight_norm`; bake each into a plain weight."""
    from torch.nn.utils import parametrize

    for sub in module.modules():
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)


# ------------------------------------------------------------------------------------------ the LM --

def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _norm(norm: nn.Module, x):
    """`torch.nn.RMSNorm` (what upstream uses without apex), spelled out: `aten::rms_norm` has no MIL
    lowering, and the product is the same `x * rsqrt(mean(x^2) + eps) * weight`."""
    return _rms(x, norm.weight, norm.eps)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x, n_rep: int):
    """HF's `repeat_kv`, spelled as HF spells it: the window `passes.py`'s GQA fusion strips, so the
    cache stores the checkpoint's 8 K/V heads."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def permute_for_rotate_half(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """vllm's `permute` for Mistral-format Q/K weights: each head's interleaved pairs `(2i, 2i+1)` become
    `(i, i + d/2)`, which is what rotate-half RoPE rotates together. An exact re-indexing of rows."""
    out_dim, in_dim = w.shape
    return w.view(n_heads, out_dim // n_heads // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


class MistralNative(nn.Module):
    """vllm's `MistralForCausalLM` body over the checkpoint AS SHIPPED: interleaved-pair RoPE (Mistral's
    complex rotation), no permutation. Only the equivalence check runs it."""

    def __init__(self, params: dict, weights: Dict[str, torch.Tensor], n_layers: Optional[int] = None):
        super().__init__()
        self.p, self.w = params, weights
        self.n_layers = params["n_layers"] if n_layers is None else n_layers

    def _rope(self, x, positions):                                   # x [s, h, d]
        d = self.p["head_dim"]
        freqs = 1.0 / (self.p["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        angle = positions.float()[:, None] * freqs[None, :]
        cos, sin = torch.cos(angle)[:, None, :], torch.sin(angle)[:, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)

    def forward(self, x):                                            # [s, dim] -> [s, dim], normed
        p, w, s = self.p, self.w, x.shape[0]
        H, KV, D, eps = p["n_heads"], p["n_kv_heads"], p["head_dim"], p["norm_eps"]
        pos = torch.arange(s)
        for i in range(self.n_layers):
            L = f"layers.{i}."
            h = _rms(x, w[L + "attention_norm.weight"], eps)
            q = self._rope((h @ w[L + "attention.wq.weight"].T).view(s, H, D), pos)
            k = self._rope((h @ w[L + "attention.wk.weight"].T).view(s, KV, D), pos)
            v = (h @ w[L + "attention.wv.weight"].T).view(s, KV, D)
            k, v = k.repeat_interleave(H // KV, 1).transpose(0, 1), v.repeat_interleave(H // KV, 1).transpose(0, 1)
            scores = (q.transpose(0, 1) @ k.transpose(1, 2)) / math.sqrt(D)
            scores = scores.masked_fill(~torch.ones(s, s, dtype=torch.bool).tril(), float("-inf"))
            x = x + (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(s, -1) @ w[L + "attention.wo.weight"].T
            h = _rms(x, w[L + "ffn_norm.weight"], eps)
            x = x + (F.silu(h @ w[L + "feed_forward.w1.weight"].T) * (h @ w[L + "feed_forward.w3.weight"].T)) \
                @ w[L + "feed_forward.w2.weight"].T
        return _rms(x, w["norm.weight"], eps)


class LMPhase(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> the last row, normed` -- vllm's
    `MistralForCausalLM.model`, rotate-half after the permutation, HF's attention spelling for the
    fusion. RoPE's cos/sin are ROWS of a table gathered by position (the reference's f32 angles)."""

    def __init__(self, params: dict, weights: Dict[str, torch.Tensor], n_layers: Optional[int] = None):
        super().__init__()
        self.H, self.KV, self.D = params["n_heads"], params["n_kv_heads"], params["head_dim"]
        self.eps = params["norm_eps"]
        n = params["n_layers"] if n_layers is None else n_layers
        dim = params["dim"]

        def linear(t):
            layer = nn.Linear(t.shape[1], t.shape[0], bias=False)
            layer.weight = nn.Parameter(t.contiguous(), requires_grad=False)
            return layer

        self.layers = nn.ModuleList()
        for i in range(n):
            L = f"layers.{i}."
            m = nn.Module()
            m.wq = linear(permute_for_rotate_half(weights[L + "attention.wq.weight"], self.H))
            m.wk = linear(permute_for_rotate_half(weights[L + "attention.wk.weight"], self.KV))
            m.wv, m.wo = linear(weights[L + "attention.wv.weight"]), linear(weights[L + "attention.wo.weight"])
            m.w1, m.w2 = linear(weights[L + "feed_forward.w1.weight"]), linear(weights[L + "feed_forward.w2.weight"])
            m.w3 = linear(weights[L + "feed_forward.w3.weight"])
            m.register_buffer("attention_norm", weights[L + "attention_norm.weight"].clone())
            m.register_buffer("ffn_norm", weights[L + "ffn_norm.weight"].clone())
            self.layers.append(m)
        self.register_buffer("norm", weights["norm.weight"].clone())
        d = self.D
        freqs = 1.0 / (params["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        angle = torch.arange(LM_MAX_POSITIONS, dtype=torch.float32)[:, None] * freqs[None, :]
        self.register_buffer("cos_table", torch.cat([torch.cos(angle)] * 2, dim=-1))
        self.register_buffer("sin_table", torch.cat([torch.sin(angle)] * 2, dim=-1))
        del dim

    def forward(self, inputs_embeds, position_ids, attention_mask, pasts=None):
        x = inputs_embeds
        b, s, _ = x.shape
        cos = F.embedding(position_ids, self.cos_table).unsqueeze(1)    # (1, 1, s, 128)
        sin = F.embedding(position_ids, self.sin_table).unsqueeze(1)
        scale = 1.0 / math.sqrt(self.D)
        for i, m in enumerate(self.layers):
            h = _rms(x, m.attention_norm, self.eps)
            q = m.wq(h).view(b, s, self.H, self.D).transpose(1, 2)
            k = m.wk(h).view(b, s, self.KV, self.D).transpose(1, 2)
            v = m.wv(h).view(b, s, self.KV, self.D).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if pasts is not None:
                past = pasts[i]
                if past[0] is not None:
                    k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
                past[0], past[1] = k, v
            k, v = _repeat_kv(k, self.H // self.KV), _repeat_kv(v, self.H // self.KV)
            scores = torch.matmul(q * scale, k.transpose(-1, -2)) + attention_mask
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
            x = x + m.wo(ctx.transpose(1, 2).reshape(b, s, self.H * self.D))
            h = _rms(x, m.ffn_norm, self.eps)
            x = x + m.w2(F.silu(m.w1(h)) * m.w3(h))
        return _rms(x, self.norm, self.eps)[:, -1:, :]


def check_lm_permutation(params, weights, n_layers: int = 2, n: int = 9, atol: float = 1e-4) -> float:
    """The traced LM (permuted, rotate-half) against the checkpoint's native spelling on real rows."""
    torch.manual_seed(0)
    ids = torch.randint(1000, 20000, (n,))
    x = weights["mm_audio_embeddings.tok_embeddings.weight"][ids]
    with torch.no_grad():
        want = MistralNative(params, weights, n_layers)(x)[-1]
        got = LMPhase(params, weights, n_layers)(x[None], torch.arange(n)[None], causal_mask(n))[0, 0]
    diff = float((got - want).abs().max())
    if not diff <= atol * max(float(want.abs().max()), 1.0):
        raise AssertionError(f"the permuted LM is {diff:.3e} from the native one; the Q/K permutation or "
                             "the RoPE layout is wrong")
    return diff


# -------------------------------------------------------------------------------------- the frames --

class EmbedPromptPhase(nn.Module):
    """`(ids [1, n], voice [1, n, D], voice_mask [1, n, 1]) -> rows`: the text table, with the voice's
    rows written over the slots the mask marks (`tts_preprocess`'s `embed_input_ids`)."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = nn.Embedding(table.shape[0], table.shape[1])
        self.table.weight = nn.Parameter(table, requires_grad=False)

    def forward(self, ids, voice, voice_mask):
        return self.table(ids) * (1.0 - voice_mask) + voice * voice_mask


class EmbedFramePhase(nn.Module):
    """`codes [1, n, 37] -> rows`: `MultiVocabEmbeddings` summed over the codebooks (`encode_tokens`).
    The codes carry the +2 special-token offset, as the flow head emits them and the table expects."""

    def __init__(self, codec):
        super().__init__()
        emb = codec.audio_token_embedding
        self.register_buffer("table", emb.embeddings.weight.detach().clone())
        # FLOAT offsets, for MOSS's reason: an int buffer is written as F32, and the gather index must
        # come out I32 (loom.cpp Retro-060's neighbour).
        self.register_buffer("offsets", emb.offsets.detach().to(torch.float32).clone())
        self.n_codebooks = int(emb.offsets.numel())

    def forward(self, codes):
        n = codes.shape[1]
        # The index flattened to one axis: `ggml_get_rows` takes a rank-3 index only when its third axis
        # matches the table's (moss_tts_export's note).
        index = (codes.float() + self.offsets).to(torch.int32).reshape(n * self.n_codebooks)
        return self.table[index].reshape(1, n, self.n_codebooks, self.table.shape[1]).sum(dim=2)


class _BidirectionalLayer(nn.Module):
    """`AcousticTransformerBlock`: pre-norm, GQA attention without RoPE or mask, SwiGLU. The GQA is the
    GROUPED spelling (voxcpm2_export's): each K/V head's `n_rep` adjacent query heads regrouped as one
    `[b, n_kv, n_rep * s, d]` block, which is `repeat_interleave`'s pairing with no tile traced."""

    def __init__(self, block):
        super().__init__()
        a = block.attention
        self.wq, self.wk, self.wv, self.wo = a.wq, a.wk, a.wv, a.wo
        self.H, self.KV, self.D = a.n_local_heads, a.n_local_kv_heads, a.head_dim
        self.attention_norm, self.ffn_norm = block.attention_norm, block.ffn_norm
        self.ff = block.feed_forward

    def forward(self, x):                                            # [b, s, dim]
        b, s, _ = x.shape
        h = _norm(self.attention_norm, x)
        q = self.wq(h).view(b, s, self.H, self.D).transpose(1, 2)
        k = self.wk(h).view(b, s, self.KV, self.D).transpose(1, 2)
        v = self.wv(h).view(b, s, self.KV, self.D).transpose(1, 2)
        rep = self.H // self.KV
        q = q.reshape(b, self.KV, rep * s, self.D)
        scores = torch.matmul(q * (1.0 / math.sqrt(self.D)), k.transpose(-1, -2))
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v).reshape(b, self.H, s, self.D)
        x = x + self.wo(ctx.transpose(1, 2).reshape(b, s, self.H * self.D))
        h = _norm(self.ffn_norm, x)
        return x + self.ff.w2(F.silu(self.ff.w1(h)) * self.ff.w3(h))


class AcousticPhase(nn.Module):
    """`(hidden [1, 1, D], noise [1, 36], cfg [1, 1]) -> (semantic logits [1, 8320], codes [1, 36])`.

    `FlowMatchingAudioTransformer.forward` minus the argmax: the semantic head's raw row, and
    `decode_one_frame`'s acoustic codes -- the draw scaled by `_noise_scale`, seven Euler steps over
    `linspace(0, 1, 8)`, each running the conditional and the unconditional (zero LLM row) sequences as
    one batch of two and mixing them `cfg * v + (1 - cfg) * v_uncond`, then clamp, scale to the 21
    levels, round, +2. The time rows are the reference's own `time_projection(time_embedding(t))`,
    computed here once at F32 as the reference caches them."""

    def __init__(self, flow):
        super().__init__()
        self.semantic = flow.semantic_codebook_output
        self.input_projection, self.llm_projection = flow.input_projection, flow.llm_projection
        self.layers = nn.ModuleList(_BidirectionalLayer(flow.layers[str(i)]) for i in flow.layers_ids)
        self.norm, self.out = flow.norm, flow.acoustic_codebook_output
        with torch.no_grad():
            t = flow._timesteps.float()
            self.register_buffer("t_proj", flow.time_projection(flow.time_embedding(t.view(-1, 1))).clone())
            self.dts = [float(v) for v in (t[1:] - t[:-1])]
        self.noise_scale = float(flow._noise_scale)
        self.levels = int(flow.acoustic_embeddings_levels)
        self.n_special = N_AUDIO_SPECIAL

    def velocity(self, x, t_row, llm):                               # [2, 36], [1, D], [2, D]
        seq = torch.cat([self.input_projection(x).unsqueeze(1), torch.cat([t_row, t_row], dim=0).unsqueeze(1),
                         llm.unsqueeze(1)], dim=1)                     # [2, 3, D]
        for layer in self.layers:
            seq = layer(seq)
        return self.out(_norm(self.norm, seq)[:, 0, :])

    def forward(self, hidden, noise, cfg):
        h = hidden.reshape(1, hidden.shape[-1])
        logits = self.semantic(h)
        p = self.llm_projection(h)
        llm = torch.cat([p, p * 0.0], dim=0)
        x = noise * self.noise_scale
        for i, dt in enumerate(self.dts):
            v = self.velocity(torch.cat([x, x], dim=0), self.t_proj[i:i + 1], llm)
            # The big operand first: ggml broadcasts only the second one.
            x = x + (v[0:1] * cfg + v[1:2] * (1.0 - cfg)) * dt
        x = torch.clamp(x, -1.0, 1.0)
        codes = torch.round(((x + 1.0) / 2.0) * (self.levels - 1)) + self.n_special
        return logits, codes


# ------------------------------------------------------------------------------------------ the codec --

class _CodecLayer(nn.Module):
    """`TransformerBlock` of the codec: q/k RMS norms over all heads, ALiBi + a causal window (the bias
    comes in), layer scale on both residual branches."""

    def __init__(self, block):
        super().__init__()
        a = block.attention
        self.wq, self.wk, self.wv, self.wo = a.wq, a.wk, a.wv, a.wo
        self.q_norm, self.k_norm = a.q_norm, a.k_norm
        self.H, self.D = a.n_local_heads, a.args.head_dim
        if a.n_local_kv_heads != a.n_local_heads:
            raise NotImplementedError("the codec's attention is expected to have as many K/V heads as Q")
        self.attention_norm, self.ffn_norm, self.ff = block.attention_norm, block.ffn_norm, block.feed_forward
        self.register_buffer("attention_scale", block.attention_scale.detach().clone())
        self.register_buffer("ffn_scale", block.ffn_scale.detach().clone())

    def forward(self, x, bias):                                      # [1, s, dim], [1, H, s, s]
        b, s, _ = x.shape
        h = _norm(self.attention_norm, x)
        q = _norm(self.q_norm, self.wq(h)).view(b, s, self.H, self.D).transpose(1, 2)
        k = _norm(self.k_norm, self.wk(h)).view(b, s, self.H, self.D).transpose(1, 2)
        v = self.wv(h).view(b, s, self.H, self.D).transpose(1, 2)
        scores = torch.matmul(q * (1.0 / math.sqrt(self.D)), k.transpose(-1, -2)) + bias
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(b, s, self.H * self.D)
        x = x + self.wo(ctx) * self.attention_scale
        h = _norm(self.ffn_norm, x)
        return x + self.ff.w2(F.silu(self.ff.w1(h)) * self.ff.w3(h)) * self.ffn_scale


class CodecPhase(nn.Module):
    """`(codes [1, T, 37], pos1, pos2, pos4, pos8) -> the 24 kHz waveform, `[1, 8T, 240]` row-major.

    `VoxtralTTSAudioTokenizer.decode`: `MistralAudioCodebook.decode` (the semantic code's centroid --
    `embedding_sum / cluster_usage`, a constant table here -- beside the 36 acoustic codes rescaled to
    [-1, 1]), then `_forward_decoder`: a k3 causal convolution with REPLICATE padding, four 2-layer
    transformers at 12.5/25/50/100 Hz with x2 causal transposed convolutions between them, and a k7
    causal convolution with REFLECT padding to 240-sample patches. `posN` is `0 .. N*T - 1`, the
    positions at each rate, from which the ALiBi bias and the window masks are built."""

    def __init__(self, codec):
        super().__init__()
        fold_weight_norm(codec)
        q = codec.quantizer
        self.register_buffer("centroids", q.semantic_codebook.embedding.detach().clone())
        self.levels = int(q.acoustic_codebook.n_levels)
        self.blocks = nn.ModuleList()
        self.kinds = []
        for block in codec.decoder_blocks:
            kind = type(block).__name__
            if kind == "CausalConv1d":
                self.blocks.append(block.conv)
                self.kinds.append(("conv", block.pad_mode, block._padding_total, block._stride))
            elif kind == "CausalConvTranspose1d":
                self.blocks.append(block.conv)
                k, s = block.conv.kernel_size[0], block.conv.stride[0]
                right = math.ceil((k - s) * block.trim_ratio)
                self.kinds.append(("convtr", k - s - right, right))
            else:
                self.blocks.append(nn.ModuleList(_CodecLayer(l) for l in block.layers.values()))
                self.kinds.append(("attn", block.args.attn_sliding_window_size, block.args.causal))
        out = codec.output_proj
        self.output_proj = out.conv
        self.out_pad = (out.pad_mode, out._padding_total, out._stride)
        self.patch = int(codec.patch_size)
        self.slopes = [float(v) for v in codec.decoder_blocks[1].layers["0"].attention.alibi_slopes]

    @staticmethod
    def _pad_left(x, mode, pad):                                    # x [1, C, t]
        if pad == 0:
            return x
        if mode == "replicate":
            return torch.cat([x[:, :, :1]] * pad + [x], dim=-1)
        if mode == "reflect":
            # `F.pad(..., mode="reflect")` on the left: x[pad], ..., x[1], then x. Static slices, since a
            # reversed slice is a negative step the lowering has no op for.
            return torch.cat([x[:, :, i:i + 1] for i in range(pad, 0, -1)] + [x], dim=-1)
        raise NotImplementedError(f"causal convolution padding mode {mode!r}")

    def _bias(self, positions, window, causal):
        """`Attention._native_attention`'s bias: `slope_h * (j - i)` inside the window, -1e30 outside --
        the reference's -inf, which softmax turns to the same exact 0 since every row keeps its own
        diagonal. `j - i` as two K=1 outer products (Pocket-TTS's note: ggml broadcasts one operand), and
        each head's slope a constant factor, stacked -- no `s * s` reshape for the shape walk to derive."""
        if not causal:
            raise NotImplementedError("the decoder's attention is causal in every released codec")
        pos = positions.to(torch.float32)[0]
        s = pos.shape[0]
        ones = torch.ones_like(pos)
        rel = torch.matmul(ones.view(-1, 1), pos.view(1, -1)) - torch.matmul(pos.view(-1, 1), ones.view(1, -1))
        outside = torch.relu(rel) + torch.relu(-rel - window)            # j > i, or i - j > window
        alibi = torch.stack([rel * slope for slope in self.slopes], dim=0).unsqueeze(0)
        return alibi + (torch.clamp(outside, 0.0, 1.0) * -1e30).view(1, 1, s, s)

    def forward(self, codes, pos1, pos2, pos4, pos8):
        t = codes.shape[1]
        c = codes.float() - N_AUDIO_SPECIAL
        semantic = self.centroids[c[:, :, 0].to(torch.int32).reshape(t)].reshape(1, t, self.centroids.shape[1])
        # `_rescale`'s own spelling, `2c / (levels - 1) - 1`: `c * 0.1 - 1` is one f32 ulp off for some
        # codes, and the decoder carries an input ulp to its output ~60x larger (measured at f64: every
        # block exact to 1e-14 once the inputs agreed, 3e-6 at the output while they did not).
        acoustic = c[:, :, 1:] * 2.0 / (self.levels - 1) - 1.0
        x = torch.cat([semantic, acoustic], dim=-1).transpose(1, 2)     # [1, 292, T]
        positions = {1: pos1, 2: pos2, 4: pos4, 8: pos8}
        rate = 1
        for block, kind in zip(self.blocks, self.kinds):
            if kind[0] == "conv":
                _, mode, pad, stride = kind
                if stride != 1:
                    raise NotImplementedError("the decoder's plain convolutions are stride 1")
                x = block(self._pad_left(x, mode, pad))
            elif kind[0] == "convtr":
                _, left, right = kind
                if left:
                    raise NotImplementedError("a transposed convolution trimmed on the left")
                # `[..., :-right]`, a static end: a slice to `y.shape[-1] - right` would be a
                # shape-derived bound, the thing the shape walk mis-substitutes (loom.cpp Retro-051).
                y = block(x)
                x = y[:, :, :-right] if right else y
                rate *= block.stride[0]
            else:
                bias = self._bias(positions[rate], kind[1], kind[2])
                h = x.transpose(1, 2)
                for layer in block:
                    h = layer(h, bias)
                x = h.transpose(1, 2)
        mode, pad, _ = self.out_pad
        x = self.output_proj(self._pad_left(x, mode, pad))              # [1, 240, 8T]
        # `b (c h) t -> b c (t h)`: patch-major `[1, 8T, 240]` IS the waveform in row-major order, so
        # the transpose is the whole rearrange and no reshape (with its inferred length) is traced.
        return x.transpose(1, 2)


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `fuse_loom_attention` expects added to `Q @ K^T`."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(1, 1, seq_len, seq_len)


def positions(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32).view(1, -1)


# ------------------------------------------------------------------------------------- the checks --

def check_wrappers(model_dir: str, weights: Dict[str, torch.Tensor], src: str = VLLM_OMNI_SRC,
                   n_frames: int = 24) -> Dict[str, float]:
    """Each wrapper against the upstream module it re-spells, both at FLOAT64, on the real weights.

    At f64 because the codec cannot be checked at f32: it carries an input ulp to its output ~60x larger,
    so random codes put two exact spellings 2e-5 apart, which no tolerance separates from a real defect.
    At f64 every block agreed to 1e-14 (the one real defect found this way was a rescale spelled
    `c * 0.1`, see `CodecPhase.forward`). The export's own phases are built separately, at F32."""
    w64 = {k: v.double() for k, v in weights.items()
           if k.startswith(("acoustic_transformer.", "audio_tokenizer.", "mm_audio_embeddings.audio"))}
    flow, codec = load_upstream(model_dir, w64, src, torch.float64)
    _, codec_ref = load_upstream(model_dir, {k: v for k, v in w64.items() if not k.startswith("acoustic")},
                                 src, torch.float64)
    return compare_wrappers(flow, codec, codec_ref, n_frames)


def compare_wrappers(flow, codec, codec_ref, n_frames: int = 24, tolerance: float = 1e-9) -> Dict[str, float]:
    """The comparison `check_wrappers` runs, on any f64 upstream modules: `codec` is consumed (the codec
    wrapper folds its weight norms in place), `codec_ref` must be a separate copy of the same weights.
    Raises past a relative `tolerance`."""
    acoustic, embed_frame, codec_phase = AcousticPhase(flow), EmbedFramePhase(codec), CodecPhase(codec)
    n_semantic = flow.model_args.semantic_codebook_size
    n_levels = flow.acoustic_embeddings_levels
    n_acoustic = flow.model_args.n_acoustic_codebook
    torch.manual_seed(0)
    out = {}
    with torch.no_grad():
        hidden = torch.randn(1, flow.acoustic_transformer_args.input_dim, dtype=torch.float64) * 2.0
        noise = torch.randn(1, n_acoustic, dtype=torch.float64)
        original = torch.randn
        torch.randn = lambda *shape, **kw: noise.clone()
        try:
            want = flow(llm_hidden=hidden, cfg_alpha=torch.full((1,), DEFAULT_CFG, dtype=torch.float64))
        finally:
            torch.randn = original
        logits, codes = acoustic(hidden.view(1, 1, -1), noise, torch.tensor([[DEFAULT_CFG]], dtype=torch.float64))
        masked = logits.clone()
        masked[:, 0] = -float("inf")
        masked[:, N_AUDIO_SPECIAL + n_semantic:] = -float("inf")
        got = torch.cat([masked.argmax(-1, keepdim=True).to(codes.dtype), codes], dim=1)
        if not torch.equal(got.long(), want.long()) and int(want[0, 0]) != END_AUDIO:
            raise AssertionError(f"acoustic wrapper codes {got.long().tolist()} != flow head {want.tolist()}")
        want_logits = flow.semantic_codebook_output(hidden)
        out["semantic_logits"] = float((logits - want_logits).abs().max()) / float(want_logits.abs().max())

        frame_codes = torch.cat([torch.randint(2, 2 + n_semantic, (1, n_frames, 1)),
                                 torch.randint(2, 2 + n_levels, (1, n_frames, n_acoustic))], dim=-1)
        want_rows = codec_ref.encode_tokens([frame_codes.transpose(1, 2)])[0]
        got_rows = embed_frame(frame_codes.to(torch.int32))[0]
        out["embed_frame"] = float((got_rows - want_rows).abs().max()) / float(want_rows.abs().max())

        want_wave = codec_ref.decode((frame_codes - N_AUDIO_SPECIAL).transpose(1, 2), dtype=torch.float64).reshape(-1)
        got_wave = codec_phase(frame_codes.to(torch.int32), positions(n_frames), positions(2 * n_frames),
                               positions(4 * n_frames), positions(8 * n_frames)).reshape(-1)
        out["codec"] = float((got_wave - want_wave).abs().max()) / float(want_wave.abs().max())
    for key, value in out.items():
        if not value <= tolerance:
            raise AssertionError(f"the {key} wrapper is {value:.3e} (relative, f64) from its upstream module")
    return out


def read_voice(model_dir: str, name: str) -> np.ndarray:
    v = torch.load(Path(model_dir) / "voice_embedding" / f"{name}.pt", map_location="cpu")
    if v.ndim != 2 or v.shape[1] != read_params(model_dir)["dim"]:
        raise ValueError(f"voice {name!r} is {tuple(v.shape)}, not [n, dim]")
    return v.float().numpy()


def check_special_ids(model_dir: str) -> None:
    spec = json.loads((Path(model_dir) / "tekken.json").read_text())
    names = {e["rank"]: e["token_str"] for e in spec["special_tokens"]}
    expect = {BOS: "<s>", AUDIO: "[AUDIO]", BEGIN_AUDIO: "[BEGIN_AUDIO]",
              REPEAT_AUDIO_TEXT: "[REPEAT_AUDIO_TEXT]", NEXT_AUDIO_TEXT: "[NEXT_AUDIO_TEXT]"}
    for i, s in expect.items():
        if names.get(i) != s:
            raise ValueError(f"tekken.json's marker {i} is {names.get(i)!r}, not {s!r}")


# ------------------------------------------------------------------------------------------ export --

@dataclass(kw_only=True)
class VoxtralTTSExportConfig(BaseMultiPhaseModelExportConfig):
    """A `mistralai/Voxtral-4B-TTS-2603` checkpoint directory -> one Loom GGUF."""

    architecture: str = "voxtral_tts"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "voxtral_tts_driver"
    voice: str = DEFAULT_VOICE
    vllm_omni_src: str = VLLM_OMNI_SRC
    _driver_weights: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _voice_rows: int = field(default=0, init=False, repr=False)
    _voice_compat: Optional[str] = field(default=None, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to a Voxtral-TTS directory; the recognizer found params.json "
                               "(model_type voxtral_tts), consolidated.safetensors and tekken.json in it"),
        "decomposition": Unchecked("MultiPhase by construction -- five graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                        "against the traced topologies by LuaFragment"),
        "voice": Unchecked("the built-in voice's file stem under `voice_embedding/`; read and shape-"
                           "checked by `read_voice`"),
        "vllm_omni_src": Unchecked("the reference modules' directory; `import_upstream` raises without it"),
        "_driver_weights": Unchecked("read off the checkpoint during phases() and shipped as driver weights"),
        "_voice_rows": Unchecked("the built-in voice's length, read during phases()"),
        "_voice_compat": Unchecked("the LM weights' fingerprint, read once by contract()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        check_special_ids(self.model_dir)
        params = read_params(self.model_dir)
        print(f"Loading Voxtral-TTS from {self.model_dir}...")
        weights = load_checkpoint(self.model_dir)
        diff = check_lm_permutation(params, weights)
        print(f"  permuted LM vs the checkpoint's native RoPE (2 layers): max |d| {diff:.2e}")
        print(f"  wrappers vs vllm-omni at f64: {check_wrappers(self.model_dir, weights, self.vllm_omni_src)}")
        flow, codec = load_upstream(self.model_dir, weights, self.vllm_omni_src)
        acoustic, embed_frame, codec_phase = AcousticPhase(flow).eval(), EmbedFramePhase(codec).eval(), \
            CodecPhase(codec).eval()
        voice = read_voice(self.model_dir, self.voice)
        from .voxtral_tts_voices import expected_rows

        if expected_rows(self.model_dir).get(self.voice) != voice.shape[0]:
            raise ValueError(f"voice {self.voice!r} has {voice.shape[0]} rows, not the tokenizer's slot count")
        self._voice_rows = int(voice.shape[0])
        self._driver_weights = {"voice.default": voice.reshape(-1)}
        dim = params["dim"]
        n_cb = int(embed_frame.n_codebooks)
        lm = LMPhase(params, weights).eval()
        text_table = weights["mm_audio_embeddings.tok_embeddings.weight"]
        del weights
        seq = ct.RangeDim(1, LM_MAX_POSITIONS)
        frames = ct.RangeDim(1, MAX_FRAMES)
        return [
            ExportPhase(
                name="embed_prompt",
                wrapper=EmbedPromptPhase(text_table).eval(),
                dummy_inputs=(torch.randint(1000, 20000, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.randn(1, TRACE_TOKENS, dim), torch.zeros(1, TRACE_TOKENS, 1)),
                mil_inputs=[ct.TensorType(name="ids", shape=(1, seq), dtype=np.int32),
                            ct.TensorType(name="voice", shape=(1, seq, dim), dtype=np.float32),
                            ct.TensorType(name="voice_mask", shape=(1, seq, 1), dtype=np.float32)],
            ),
            ExportPhase(
                name="embed_frame",
                wrapper=embed_frame,
                dummy_inputs=(torch.randint(2, 20, (1, TRACE_FRAMES, n_cb), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="codes", shape=(1, frames, n_cb), dtype=np.int32)],
            ),
            ExportPhase(
                name="lm",
                wrapper=lm,
                dummy_inputs=(torch.randn(1, TRACE_TOKENS, dim), positions(TRACE_TOKENS), causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, seq, dim), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq, seq), dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="acoustic",
                wrapper=acoustic,
                dummy_inputs=(torch.randn(1, 1, dim), torch.randn(1, 36), torch.tensor([[DEFAULT_CFG]])),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, 1, dim), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(1, 36), dtype=np.float32),
                            ct.TensorType(name="cfg", shape=(1, 1), dtype=np.float32)],
            ),
            ExportPhase(
                name="codec",
                wrapper=codec_phase,
                dummy_inputs=(torch.randint(2, 20, (1, TRACE_FRAMES, n_cb), dtype=torch.int32),
                              positions(TRACE_FRAMES), positions(2 * TRACE_FRAMES), positions(4 * TRACE_FRAMES),
                              positions(8 * TRACE_FRAMES)),
                mil_inputs=[
                    ct.TensorType(name="codes", shape=(1, frames, n_cb), dtype=np.int32),
                    ct.TensorType(name="pos1", shape=(1, frames), dtype=np.int32),
                    ct.TensorType(name="pos2", shape=(1, ct.RangeDim(2, 2 * MAX_FRAMES)), dtype=np.int32),
                    ct.TensorType(name="pos4", shape=(1, ct.RangeDim(4, 4 * MAX_FRAMES)), dtype=np.int32),
                    ct.TensorType(name="pos8", shape=(1, ct.RangeDim(8, 8 * MAX_FRAMES)), dtype=np.int32),
                ],
                root_axis="n_codes",
                declared_axes={"pos2": {1: "2 * n_codes"}, "pos4": {1: "4 * n_codes"}, "pos8": {1: "8 * n_codes"}},
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        constants = {
            "BOS": BOS, "AUDIO": AUDIO, "BEGIN_AUDIO": BEGIN_AUDIO,
            "REPEAT_AUDIO_TEXT": REPEAT_AUDIO_TEXT, "NEXT_AUDIO_TEXT": NEXT_AUDIO_TEXT,
            "END_AUDIO": END_AUDIO, "EMPTY_AUDIO_CODE": N_AUDIO_SPECIAL, "N_SEMANTIC": 8192 + N_AUDIO_SPECIAL,
            "N_CODEBOOKS": 37, "N_ACOUSTIC": 36, "DIM": 3072,
            "DEFAULT_CFG": DEFAULT_CFG, "MAX_FRAMES": MAX_FRAMES, "LM_MAX_POSITIONS": LM_MAX_POSITIONS,
            "SAMPLES_PER_FRAME": SAMPLES_PER_FRAME, "CODEC_CHUNK": CODEC_CHUNK, "CODEC_CONTEXT": CODEC_CONTEXT,
        }
        fragment = self.driver_script_path
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "00_header.lua", top_level=True, defines=("voxtral_slice", "voxtral_decode")),
            LuaFragment(fragment / "01_generate.lua", reads=("tokens",) + tuple(constants), defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        # What a voice file must match (`voxtral_tts_voices`, loom.cpp ADR-045) -- a fact about THESE
        # weights, so read only when there are weights to read: an architecture-only query
        # (test_tts_text_door's nonexistent path) opens nothing -- and the voice the file carries.
        from .voxtral_tts_voices import weights_fingerprint

        weights = Path(self.model_dir) / "consolidated.safetensors"
        if self._voice_compat is None and weights.is_file():
            self._voice_compat = weights_fingerprint(weights)
        if self._voice_compat is not None:
            contract["voice.compat"] = self._voice_compat
        contract["tts.voices"] = [self.voice]
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                      tokenizer_dir=self.model_dir, tokenizer_family="tekken")
        if self._driver_weights is not None:
            kwargs["driver_weights"] = dict(self._driver_weights)
        return kwargs


def _is_voxtral_tts(path: Path) -> bool:
    """A Voxtral-TTS directory: Mistral's layout (`params.json` + `consolidated.safetensors` +
    `tekken.json`) with `model_type: voxtral_tts`."""
    if not (path.is_dir() and (path / "params.json").is_file() and (path / "consolidated.safetensors").is_file()
            and (path / "tekken.json").is_file()):
        return False
    try:
        return read_params(str(path)).get("model_type") == "voxtral_tts"
    except (OSError, ValueError):
        return False


def _build_voxtral_tts(path: Path, output_path: str) -> LoomExportConfig:
    return VoxtralTTSExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=VoxtralTTSExportConfig,
        recognizers=[ModelRecognizer(name="voxtral-tts", detect=_is_voxtral_tts, build_config=_build_voxtral_tts)],
    ))
