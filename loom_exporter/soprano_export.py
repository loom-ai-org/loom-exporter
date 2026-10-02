"""Export Soprano TTS (`ekwek/Soprano-1.1-80M`) -- family 9's eighth leaf, and the first whose vocoder
reads the LM's HIDDEN STATES rather than the tokens it draws.

Soprano is a 17-layer Qwen3 (512 wide, 4 query heads over 1 K/V head) whose vocabulary is 8000 audio
ids `[0]..[7999]` beside a small character BPE for the text:

    text -> clean_text -> sentences -> `[STOP][TEXT]{sentence}[START]` ids
         -> per step: logits -> draw (repetition penalty 1.2 over prompt AND generated ids, T=0.001,
                      top-p 0.95) -> the drawn id is the next input; the final-norm hidden row that
                      produced it is KEPT, unless the draw is `[STOP]`
         -> Vocos decoder over every kept row (linear x4 upsample, 8 ConvNeXt blocks, ISTFT head:
            n_fft 2048, hop 512) -> 2048 samples per kept row at 32 kHz

**The drawn ids are never decoded.** They exist to drive the LM; the audio is a function of the hidden
rows alone (`SopranoTTS.infer_batch` hands `response['hidden_state']` to the decoder). So the decoder
phase's input is the LM phase's SECOND output, gathered per step by the driver.

Two phases:
  - `lm`:       `(input_ids, position_ids, attention_mask) -> (logits, hidden)` for the LAST row, the
                hidden row after the final RMSNorm -- HF's `hidden_states[-1]`, which is what the
                reference keeps (checked: `lm_head(hidden_states[-1]) == logits`). KV-cached.
  - `decoder`:  `SopranoDecoder` over every kept row in one call, as the reference's non-streaming
                `infer` does: L rows -> `2048 * (L - 1)` samples.

Usage:
  loom-export ~/Dev/models/soprano-1.1-80m -o soprano.gguf --task text-to-speech --model soprano
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Axis, Unchecked

# Where the reference checkout lives. `soprano-tts` on PyPI is CUDA-only ("Install with wheel
# (CUDA-only for now)"), and the decoder's module definition is in the repository, not on the Hub.
SOPRANO_REPO = "/home/flavio/Dev/soprano"

SAMPLE_RATE = 32000
# `SopranoTTS.TOKEN_SIZE`: waveform samples per kept hidden row -- the decoder's x4 upsample times the
# ISTFT's hop of 512.
SAMPLES_PER_ROW = 2048
UPSCALE = 4
# The reference's `TransformersModel.infer`: `max_new_tokens=512`, and the prompt is truncated at 512.
MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 512
# `max_position_embeddings`: a 512-id prompt plus 512 drawn ids is the most the reference can ask of it.
LM_MAX_POSITIONS = 1024
# `SopranoTTS.infer`'s defaults. `temperature=0.0` is replaced by 0.001 in the backend ("temp must be
# nonzero"), and that is the number the reference samples at.
DEFAULT_TEMPERATURE = 0.001
# `generate`'s own default: `generation_config.json` sets no `top_k`, so `GenerationConfig`'s 50 applies
# whenever `do_sample=True`, which the reference always passes.
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.2

# Trace lengths: odd and distinct from every static dimension.
TRACE_TOKENS = 7
TRACE_ROWS = 5


def import_soprano():
    if SOPRANO_REPO not in sys.path:
        sys.path.insert(0, SOPRANO_REPO)


def load_decoder(model_dir: str):
    """The reference's `SopranoDecoder` with `decoder.pth` loaded -- what `SopranoTTS.__init__` builds."""
    import_soprano()
    from soprano.vocos.decoder import SopranoDecoder

    decoder = SopranoDecoder()
    state = torch.load(str(Path(model_dir) / "decoder.pth"), map_location="cpu", weights_only=True)
    decoder.load_state_dict(state)
    return decoder.eval()


def load_lm(model_dir: str):
    """The Qwen3 LM, f32 on CPU -- the dtype the reference's transformers backend loads on CPU."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    return model.eval()


def prompt_ids(model_dir: str):
    """`[TEXT]`, `[START]` and `[STOP]`'s ids: the three added tokens `_preprocess_text`'s prompt
    `[STOP][TEXT]{sentence}[START]` is spelled with."""
    import json

    spec = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    added = {t["content"]: int(t["id"]) for t in spec["added_tokens"]}
    return added["[TEXT]"], added["[START]"], added["[STOP]"]


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask: an already-prepared 4-D mask short-circuits HF's mask builder,
    which would otherwise read the key length off a shape the trace bakes in."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class LMPhase(nn.Module):
    """`(input_ids, position_ids, attention_mask) -> (logits, hidden)` for the LAST row.

    Logits first, because `loom.sample_row` reduces a module's output 0. The hidden row is output 2,
    and it is the row the vocoder reads -- after the final norm, which is HF's `hidden_states[-1]`."""

    def __init__(self, lm):
        super().__init__()
        self.model = lm.model
        self.lm_head = lm.lm_head

    def forward(self, input_ids, position_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, position_ids=position_ids,
                            attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = hidden[:, -1:]
        return self.lm_head(last), last


def linear_upsample_kernel(upscale: int) -> torch.Tensor:
    """`F.interpolate(mode='linear', align_corners=True)` to `upscale * (T - 1) + 1` as a transposed
    convolution's kernel: output `upscale*i + r` is `x[i] * (1 - r/upscale) + x[i+1] * (r/upscale)`, so
    every input sample spreads a triangle of `2*upscale - 1` taps, peaking at 1 on its own position.

    The interpolation's source coordinate is `j * (T-1) / (upscale*(T-1))`, which is exactly
    `j / upscale` for every T > 1 (and T = 1 is a single sample both ways), so the weights do not
    depend on the length and one kernel serves every call."""
    k = torch.arange(1, 2 * upscale, dtype=torch.float64)
    return (1.0 - (k - upscale).abs() / upscale).to(torch.float32)


class DecoderPhase(nn.Module):
    """`(hidden) -> waveform`: `SopranoDecoder.forward` over every kept row, then the reference's trim.

    Three things differ from the module as written, each exact:

    * **The interpolation is a depthwise transposed convolution** (`linear_upsample_kernel`). Its
      output size is computed from `T` in Python, which the trace would bake in.
    * **`ISTFTHead` builds a complex tensor and calls `torch.istft`**, which has no coremltools
      handler; the head's real and imaginary parts go to this project's traceable `ISTFT` instead, as
      F5-TTS's Vocos does. The reference zeroes the DC and Nyquist bins first (`spec[:,0] = 0`,
      `spec[:,-1] = 0`, its `padding == "center"` branch), which here is a constant 0/1 mask.
    * **The reference's trim is not here because it trims nothing**: `audio[-(L*2048 - 2048):]` of a
      waveform that is exactly `2048 * (L - 1)` samples long (see `forward`) is the whole waveform.

    Input is FRAME-major (`[1, n, 512]`), the layout the LM's hidden rows come in; the transpose to
    the decoder's channel-major convention is the graph's own first op."""

    def __init__(self, decoder):
        super().__init__()
        from .istft import ISTFT

        if decoder.upscale != UPSCALE or decoder.hop_length * decoder.upscale != SAMPLES_PER_ROW:
            raise NotImplementedError(f"decoder upscale {decoder.upscale} / hop {decoder.hop_length}: "
                                      f"the driver's sample arithmetic assumes x{UPSCALE} and 512")
        if getattr(decoder.head.istft, "padding", None) != "center":
            raise NotImplementedError("only the ISTFT head's 'center' branch is implemented")
        channels = decoder.decoder_initial_channels
        kernel = linear_upsample_kernel(decoder.upscale)
        self.register_buffer("upsample", kernel.view(1, 1, -1).repeat(channels, 1, 1).contiguous())
        self.upscale = decoder.upscale
        self.channels = channels
        self.backbone = decoder.decoder
        self.out = decoder.head.out
        n_freq = decoder.n_fft // 2 + 1
        edges = torch.ones(1, n_freq, 1)
        edges[:, 0] = 0.0
        edges[:, -1] = 0.0
        self.register_buffer("edges", edges)
        self.istft = ISTFT(n_fft=decoder.head.istft.n_fft, hop_length=decoder.head.istft.hop_length,
                           win_length=decoder.head.istft.win_length, center=True)

    def forward(self, hidden):                                      # (1, n, 512)
        x = hidden.transpose(1, 2)
        # Unpadded, then cropped by `upscale - 1` at both ends -- `padding=` itself is the same crop,
        # but the depthwise `conv_transpose` lowering takes only `pad = 0` and a separate slice.
        x = F.conv_transpose1d(x, self.upsample, stride=self.upscale, groups=self.channels)
        x = x[..., self.upscale - 1:1 - self.upscale]
        x = self.backbone(x)
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        # `audio[-(L*2048 - 2048):]`: the ISTFT of `4(L-1)+1` frames at hop 512, centred, is exactly
        # `512 * 4(L-1)` = `2048 * (L - 1)` samples, so the reference's trim keeps every one of them
        # and there is nothing to slice here.
        return self.istft(mag * torch.cos(p) * self.edges, mag * torch.sin(p) * self.edges)


@dataclass(kw_only=True)
class SopranoExportConfig(BaseMultiPhaseModelExportConfig):
    """A Soprano checkpoint directory (`model.safetensors`, `decoder.pth`, `tokenizer.json`) -> one
    Loom GGUF."""

    architecture: str = "soprano"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "soprano_driver"
    _eos_id: int = field(default=3, init=False, repr=False)
    _text_id: int = field(default=1, init=False, repr=False)
    _start_id: int = field(default=2, init=False, repr=False)
    _hidden_size: int = field(default=512, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the checkpoint directory; the recognizer found `decoder.pth` "
                               "and a qwen3 `config.json` in it"),
        "decomposition": Unchecked("MultiPhase by construction -- two graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                         "against the traced topologies by LuaFragment"),
        "_eos_id": Unchecked("`config.json`'s eos_token_id, read during phases()"),
        "_text_id": Unchecked("`[TEXT]`'s id in `tokenizer.json`, read during phases()"),
        "_start_id": Unchecked("`[START]`'s id in `tokenizer.json`, read during phases()"),
        "_hidden_size": Unchecked("`config.json`'s hidden_size, read during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        lm = load_lm(self.model_dir)
        decoder = load_decoder(self.model_dir)
        self._eos_id = int(lm.config.eos_token_id)
        self._text_id, self._start_id, stop_id = prompt_ids(self.model_dir)
        if stop_id != self._eos_id:
            raise ValueError(f"`[STOP]` is id {stop_id} but eos_token_id is {self._eos_id}: the driver "
                             f"splits sentences on the id that both the prompt opens with and EOS is")
        hidden_size = self._hidden_size = int(lm.config.hidden_size)
        seq_dim = ct.RangeDim(1, LM_MAX_POSITIONS)
        row_dim = ct.RangeDim(1, MAX_NEW_TOKENS)
        return [
            ExportPhase(
                name="lm",
                wrapper=LMPhase(lm).eval(),
                dummy_inputs=(torch.randint(4, 8000, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="input_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="position_ids", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, seq_dim, seq_dim),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=LM_MAX_POSITIONS,
            ),
            ExportPhase(
                name="decoder",
                wrapper=DecoderPhase(decoder).eval(),
                dummy_inputs=(torch.randn(1, TRACE_ROWS, hidden_size),),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, row_dim, hidden_size),
                                          dtype=np.float32)],
                root_axis="n_codes",
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        fragment = self.driver_script_path
        return [
            ExportConstants(values={
                "EOS_ID": self._eos_id,
                "TEXT_ID": self._text_id,
                "START_ID": self._start_id,
                "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS,
                "LM_MAX_POSITIONS": LM_MAX_POSITIONS,
                "HIDDEN_SIZE": self._hidden_size,
                "DEFAULT_TEMPERATURE": DEFAULT_TEMPERATURE,
                "DEFAULT_TOP_K": DEFAULT_TOP_K,
                "DEFAULT_TOP_P": DEFAULT_TOP_P,
                "DEFAULT_REPETITION_PENALTY": DEFAULT_REPETITION_PENALTY,
            }),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "01_generate.lua",
                        reads=("tokens", "EOS_ID", "TEXT_ID", "START_ID", "MAX_NEW_TOKENS",
                               "MAX_PROMPT_TOKENS", "LM_MAX_POSITIONS", "HIDDEN_SIZE",
                               "DEFAULT_TEMPERATURE", "DEFAULT_TOP_K", "DEFAULT_TOP_P",
                               "DEFAULT_REPETITION_PENALTY"),
                        defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        return contract

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                    tokenizer_dir=self.model_dir, tokenizer_family="soprano")


def _is_soprano(path: Path) -> bool:
    """A Soprano checkpoint: a qwen3 `config.json` beside the reference's `decoder.pth`."""
    if not (path.is_dir() and (path / "decoder.pth").is_file() and (path / "config.json").is_file()
            and (path / "model.safetensors").is_file()):
        return False
    import json

    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return config.get("model_type") == "qwen3"


def _build_soprano(path: Path, output_path: str) -> LoomExportConfig:
    return SopranoExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=SopranoExportConfig,
        recognizers=[
            ModelRecognizer(name="soprano", detect=_is_soprano, build_config=_build_soprano),
        ],
    ))
