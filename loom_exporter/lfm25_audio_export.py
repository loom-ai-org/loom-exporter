"""Export LFM2.5-Audio-1.5B's speech-to-text door -- family 3's third leaf, and its first hybrid LM.

`LiquidAI/LFM2.5-Audio-1.5B` (liquid-audio 1.3) is an audio encoder, an adapter and LFM2.5-1.2B:

    16 kHz audio -> NeMo log-mel (128 bins) -> FastConformer (canary-180m-flash, x8)
                 -> adapter MLP (LayerNorm, 512 -> 2048 -> GELU -> 2048) -> one LM row per 80 ms
    "<|startoftext|><|im_start|>system\\nPerform ASR.<|im_end|>\\n<|im_start|>user\\n" [audio rows]
    "<|im_end|>\\n<|im_start|>assistant\\n" -> greedy text until <|im_end|>

which is `generate_sequential` with the README's fixed ASR system prompt. The model also SPEAKS (a
depthformer over 8 Mimi codebooks and an LFM2-based detokenizer); that door is not this export.

**The encoder is NeMo's own.** liquid-audio vendors NeMo's `ConformerEncoder` and
`AudioToMelSpectrogramPreprocessor`, and the checkpoint's 692 conformer tensors load key for key into
NeMo's class -- so this builds NeMo's modules from the checkpoint's `encoder`/`preprocessor` configs
and uses Canary's encoder path, whose masks were already fixed against NeMo (Retro-065). Dither is
training-only in both.

**Not the family-3 template, though it is a family-3 model.** `BaseSpeechLMExportConfig` holds an
encoder to "rows grow linearly with whole chunks of samples"; NeMo's centred STFT gives
`floor(n / 160) + 1` frames and the x8 subsampling `ceil(frames / 8)` rows, so k whole chunks become
k + 1 rows and the template's geometry check fails by construction. So the encoder takes the whole
waveform and its length, as Canary's does (`transcribe`'s one-pass branch), and the driver walks the
prompt in segments as family 3's does. The embed, decoder and head wrappers are family 3's.

**The LM is a hybrid**: 10 short-convolution blocks and 6 attention blocks. The decoder phase fuses
both (`fuse_attention`, `fuse_conv`), so the file carries a KV cache and a conv-state cache.

Phases: `encoder` (waveform, length -> rows), `embed` (ids -> embeddings), `decoder` (embeddings ->
hidden, cached), `lm_head` (hidden -> logits, the tied embedding).

Usage:
  loom-export ~/Dev/models/lfm2.5-audio-1.5b -o lfm25_audio_asr.gguf \\
      --task automatic-speech-recognition --model lfm2.5-audio
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
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .speech_lm_export import _DecoderWrapper, _EmbedWrapper, _LMHeadWrapper, causal_mask
from .spec_protocol import Axis, Unchecked

# The README's fixed ASR system prompt, and the turn scaffolding `ChatState` adds around it.
ASR_SYSTEM_PROMPT = "Perform ASR."
PROMPT_HEAD = ["<|startoftext|>", "<|im_start|>system\n", ASR_SYSTEM_PROMPT, "<|im_end|>\n",
               "<|im_start|>user\n"]
PROMPT_TAIL = ["<|im_end|>\n", "<|im_start|>assistant\n"]
END_OF_TURN = "<|im_end|>"
# `generate_sequential(max_new_tokens=512)`, the README's ASR call.
MAX_NEW_TOKENS = 512
# The KV cache: the prompt is ~20 text positions plus 12.5 audio rows a second, then the transcript.
MAX_SEQ_LEN = 4096
# The encoder's sample axis: 280 s at 16 kHz is 3500 audio rows, which with the ~19 prompt positions
# and a 512-id transcript fits the 4096-position cache above (300 s would not: 3750 + 19 + 512).
MAX_SECONDS = 280

TRACE_SAMPLES = 16000 * 3 + 173
TRACE_TOKENS = 7


def load_model(model_dir: str):
    """`(preprocessor, encoder, adapter, lfm, config)`: NeMo's preprocessor and conformer and the
    adapter MLP, with the checkpoint's weights, and the LFM2 backbone as transformers' `Lfm2Model`."""
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
    from safetensors.torch import load_file
    from transformers import Lfm2Config, Lfm2Model

    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    weights = load_file(str(d / "model.safetensors"))
    pre = cfg["preprocessor"]
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=pre["sample_rate"], normalize=pre["normalize"], window_size=pre["window_size"],
        window_stride=pre["window_stride"], window=pre["window"], features=pre["features"],
        n_fft=pre["n_fft"], log=pre["log"], frame_splicing=pre["frame_splicing"], dither=pre["dither"],
        pad_to=pre["pad_to"], pad_value=pre["pad_value"]).eval()
    encoder = ConformerEncoder(**cfg["encoder"]).eval()
    encoder.load_state_dict({k[len("conformer."):]: v.float() for k, v in weights.items()
                             if k.startswith("conformer.")})
    hidden = int(cfg["lfm"]["hidden_size"])
    adapter = nn.Sequential(nn.LayerNorm(encoder._feat_out), nn.Linear(encoder._feat_out, hidden), nn.GELU(),
                            nn.Linear(hidden, hidden)).eval()
    adapter.load_state_dict({k[len("audio_adapter.model."):]: v.float() for k, v in weights.items()
                             if k.startswith("audio_adapter.model.")})
    lfm_cfg = dict(cfg["lfm"])
    lfm_cfg["torch_dtype"] = "float32"
    lfm = Lfm2Model(Lfm2Config(**lfm_cfg)).eval()
    lfm.load_state_dict({k[len("lfm."):]: v.float() for k, v in weights.items() if k.startswith("lfm.")})
    return preprocessor, encoder, adapter, lfm, cfg


class EncoderPhase(nn.Module):
    """`(waveform, length) -> audio rows [n_rows, hidden]`: NeMo's log-mel and FastConformer over the
    real samples, then the adapter -- `LFM2AudioModel._prefill`'s audio half, which keeps the
    `audio_in_len` rows the conformer reports."""

    def __init__(self, preprocessor, encoder, adapter):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, waveform, length):
        features, feature_len = self.preprocessor(input_signal=waveform, length=length)
        # `ChatState.add_audio` ignores the length NeMo returns, `floor(n / 160)`, and hands the
        # conformer EVERY frame the centred STFT made -- one more. Measured on jfk.wav: with NeMo's
        # length the last frame is masked and the rows differ by 0.32 (absmax 3.4); with this one they
        # are identical.
        encoded, encoded_len = self.encoder(audio_signal=features, length=feature_len + 1)
        return self.adapter(encoded.permute(0, 2, 1))[0, :encoded_len[0]]


class _TiedHead(nn.Module):
    """`nn.functional.linear(hidden, embed_tokens.weight)`: the text logits `generate_sequential`
    reads, through the input embedding (LFM2 ties them)."""

    def __init__(self, embed_tokens):
        super().__init__()
        self.weight = embed_tokens.weight

    def forward(self, hidden):
        return nn.functional.linear(hidden, self.weight)


def prompt_ids(model_dir: str):
    """`(head, tail, end_of_turn)` ids: each piece of the scaffolding tokenized ON ITS OWN, as
    `ChatState.add_text` tokenizes it (`add_special_tokens=False`)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    head = [i for piece in PROMPT_HEAD for i in enc(piece)]
    tail = [i for piece in PROMPT_TAIL for i in enc(piece)]
    eot = enc(END_OF_TURN)
    if len(eot) != 1:
        raise ValueError(f"{END_OF_TURN!r} is {len(eot)} ids; the driver stops on one")
    return head, tail, eot[0]


@dataclass(kw_only=True)
class Lfm25AudioAsrExportConfig(BaseMultiPhaseModelExportConfig):
    """An LFM2.5-Audio checkpoint directory -> one Loom GGUF for its speech-to-text door."""

    architecture: str = "lfm2.5-audio-asr"
    model_dir: str
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "lfm25_audio_driver"
    _facts: dict = field(default_factory=dict, init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the checkpoint directory; the recognizer found its "
                               "`Lfm2AudioForConditionalGeneration` config.json"),
        "decomposition": Unchecked("MultiPhase by construction -- four graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragment is parsed and checked against the "
                                         "traced topologies by LuaFragment"),
        "_facts": Unchecked("the checkpoint's numbers, read during phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        preprocessor, encoder, adapter, lfm, cfg = load_model(self.model_dir)
        head, tail, eot = prompt_ids(self.model_dir)
        hidden = int(lfm.config.hidden_size)
        self._facts = {"head": head, "tail": tail, "eot": eot, "hidden": hidden,
                       "sample_rate": int(cfg["preprocessor"]["sample_rate"])}
        n_samples = ct.RangeDim(400, self._facts["sample_rate"] * MAX_SECONDS)
        token_axis = ct.RangeDim(1, MAX_SEQ_LEN)
        return [
            ExportPhase(
                name="encoder",
                wrapper=EncoderPhase(preprocessor, encoder, adapter).eval(),
                dummy_inputs=(torch.randn(1, TRACE_SAMPLES) * 0.1, torch.tensor([TRACE_SAMPLES])),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, n_samples), dtype=np.float32),
                            ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="embed",
                wrapper=_EmbedWrapper(lfm).eval(),
                dummy_inputs=(torch.randint(0, 1000, (1, TRACE_TOKENS), dtype=torch.int32),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32)],
            ),
            ExportPhase(
                name="decoder",
                wrapper=_DecoderWrapper(lfm).eval(),
                dummy_inputs=(torch.randn(1, TRACE_TOKENS, hidden),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              causal_mask(TRACE_TOKENS)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, token_axis, hidden), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
                ],
                fuse_attention=True,
                fuse_conv=True,
                kv_cache_size=MAX_SEQ_LEN,
            ),
            ExportPhase(
                name="lm_head",
                wrapper=_LMHeadWrapper(_TiedHead(lfm.embed_tokens)).eval(),
                dummy_inputs=(torch.randn(1, 1, hidden),),
                mil_inputs=[ct.TensorType(name="hidden", shape=(1, ct.RangeDim(1, MAX_SEQ_LEN), hidden),
                                          dtype=np.float32)],
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        f = self._facts
        constants = {"PROMPT_HEAD": f.get("head", [1]), "PROMPT_TAIL": f.get("tail", [7]),
                     "END_OF_TURN": f.get("eot", 7), "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                     "MAX_SEQ_LEN": MAX_SEQ_LEN}
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("waveform", CALLER), ("length", CALLER)), n_tokens=Len("waveform")),
            LuaFragment(self.driver_script_path / "01_transcribe.lua",
                        reads=("waveform", "length") + tuple(constants), defines=("ids",)),
            DriverReturn(values=("ids",)),
        ]

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = int(self._facts.get("sample_rate", 16000))
        return contract

    def backend_kwargs(self) -> dict:
        return dict(flat_namespace=False, root_axis=self.root_axis, tokenizer_dir=self.model_dir,
                    eos_token_id=int(self._facts.get("eot", 7)))


def _is_lfm25_audio(path: Path) -> bool:
    cfg = path / "config.json"
    if not (path.is_dir() and cfg.is_file() and (path / "model.safetensors").is_file()):
        return False
    try:
        c = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    return "Lfm2AudioForConditionalGeneration" in c.get("architectures", [])


def _build_lfm25_audio(path: Path, output_path: str) -> LoomExportConfig:
    return Lfm25AudioAsrExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=Lfm25AudioAsrExportConfig,
        recognizers=[ModelRecognizer(name="lfm2.5-audio", detect=_is_lfm25_audio, build_config=_build_lfm25_audio)],
    ))
