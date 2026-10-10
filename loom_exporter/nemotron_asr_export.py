"""NVIDIA Nemotron 3.5 ASR (`nemotron3_5_asr`): a cache-aware streaming FastConformer encoder with an
RNN-T head and language-ID prompt conditioning, exported as the third transducer leaf.

The checkpoint is the `transformers` layout (5.13+, the ovos venv), not a `.nemo` archive: the card's
NeMo requirement (26.06) is past what either export venv has, and `transformers` implements it whole.
What follows `transformers`' OFFLINE mode -- the whole clip through the encoder once, then Parakeet's
greedy transducer loop -- which is what loom's one-pass `speech2text` door is. Chunked streaming
(the encoder's K/V and conv caches) is not exported.

**Everything after the encoder is the transducer template unchanged.** The prediction network is a
2-layer `nn.LSTM` behind an embedding, and the joint is `head(relu(enc + dec))`. `transformers` splits
NeMo's joint differently -- `encoder_projector` after the prompt, `decoder_projector` after the LSTM --
so `_NemotronJoint` presents it as the template's `enc`/`pred`/`joint_net` triple, with `enc` the
identity because the projection already ran in the encoder phase (once per frame, not once per symbol).

**The encoder phase is the leaf's own, for three reasons:**

1. **The mel front end is traced.** `NemotronAsrStreamingFeatureExtractor` is preemphasis, a centred
   STFT with ZERO padding, power, a librosa Slaney filterbank and `log(x + 2^-24)` -- no normalisation,
   which is what makes it traceable at all. It marks the LAST frame invalid (`L // 160` of
   `L // 160 + 1`) and zeroes it; `_NemotronMel` pads `(256, 96)` instead of `(256, 256)`, which yields
   exactly the valid frames with no shape-derived slice. Dropping that frame is exact rather than an
   approximation: it is zero in the reference, every conv after it is causal, and its attention key is
   masked, so it reaches no valid output (the subsampling's right pad of `stride - 1` sees zero either
   way).
2. **The chunked-limited attention mask is built here, as a BOOLEAN**, by `_chunked_limited_mask`.
   `transformers` 5.14's own encoder builds it through `create_bidirectional_mask`, which returns an
   additive float mask under `attn_implementation="eager"`, and the attention then calls
   `masked_fill_(attention_mask.logical_not(), -inf)` on it: `logical_not(0.0)` is True, so eager blanks
   exactly the positions that should be visible. JFK under eager is "Your American do you for your
   country?"; under `sdpa` (a boolean mask) it is the sentence. Passing a boolean mask to the layers
   directly gives the `sdpa` semantics whatever implementation is traced.
3. **The language prompt.** `prompt_projector.linear_1([h; onehot(p)])` is `W_h h + W_p[:, p] + b`, so
   the one-hot concat becomes an embedding lookup of `W_p`'s column -- a `[1]` int input the driver
   fills from `inputs.language` (the contract's language table maps names to prompt ids), else the
   checkpoint's own default prompt, `auto`.

**The model writes language tags** (`<en-US>` after each sentence); `transformers`' processor drops
them with `skip_special_tokens`. They are declared `asr.control_ids`, which the engine drops before
detokenizing, for the same result.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .multi_phase_export import ExportPhase
from .spec_protocol import Unchecked
from .transducer_export import BaseTransducerExportConfig, TransducerParts

# The checkpoint's model_type. The English-only sibling (`nemotron_asr_streaming`) has no prompt
# projector; it is the same encoder and would be this leaf minus the prompt, not yet exported.
MODEL_TYPE = "nemotron3_5_asr"
# `log(mel + 2^-24)`, the feature extractor's own `LOG_ZERO_GUARD_VALUE`.
LOG_ZERO_GUARD = 2.0 ** -24
# Seconds of dummy audio the encoder is traced on, and the range coremltools is told the sample axis
# takes. That range is a trace-time statement only: the file declares no length bound and computes
# its relative positions rather than reading `transformers`' 5000-frame (400 s) table, so a longer
# clip runs -- at a cost that grows faster than its length (the attention is over the whole clip).
TRACE_SECONDS = 4.0
MIN_SECONDS, MAX_SECONDS = 0.5, 300.0


class _NemotronMel(nn.Module):
    """`NemotronAsrStreamingFeatureExtractor` for one clip, as traceable arithmetic: `[1, n] -> [1, T, M]`
    with `T = n // hop`. Every number is read off the real extractor."""

    def __init__(self, extractor):
        super().__init__()
        self.n_fft = int(extractor.n_fft)
        self.hop = int(extractor.hop_length)
        self.win_length = int(extractor.win_length)
        self.preemphasis = float(extractor.preemphasis or 0.0)
        # `center=True` pads n_fft // 2 each side and yields n // hop + 1 frames, the last of them
        # invalid; padding the right side by `n_fft // 2 - hop` yields exactly the n // hop valid ones.
        self.pad_left = self.n_fft // 2
        self.pad_right = self.n_fft // 2 - self.hop
        if self.pad_right < 0:
            raise ValueError(f"hop {self.hop} > n_fft / 2 = {self.n_fft // 2}: the valid-frame padding "
                             f"would be negative")
        # `torch.hann_window(win_length, periodic=False)`, which `torch.stft` centres inside n_fft.
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        # `(M, n_fft // 2 + 1)`, the extractor's librosa Slaney filterbank, used as `fb @ power`.
        self.register_buffer("fb", extractor.mel_filters.detach().clone().to(torch.float32))
        # y[0] = x[0], y[i] = x[i] - a * x[i - 1]: a 2-tap conv over x left-padded by one zero, which is
        # the reference's `cat([x[:1], x[1:] - a * x[:-1]])` without slicing a dynamic axis.
        self.register_buffer("preemph", torch.tensor([[[-self.preemphasis, 1.0]]], dtype=torch.float32))

    def forward(self, waveform):
        x = waveform.unsqueeze(1)                                          # [1, 1, n]
        if self.preemphasis:
            x = nn.functional.conv1d(nn.functional.pad(x, (1, 0)), self.preemph)
        x = nn.functional.pad(x, (self.pad_left, self.pad_right))[:, 0]    # [1, n + n_fft - hop]
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                          window=self.window, center=False, return_complex=True)
        power = spec.abs() ** 2                                           # [1, F, T]
        mel = torch.matmul(self.fb, power)                                 # [1, M, T]
        return torch.log(mel + LOG_ZERO_GUARD).transpose(1, 2)             # [1, T, M]


def _chunked_limited_mask(n: int, chunk: int, left_chunks: int, device=None):
    """`chunked_limited_mask_function(left, right)` as a BOOLEAN `[1, 1, n, n]` over a length the graph
    reads off its own input: query `q` sees key `k` when `0 <= q // chunk - k // chunk <= left_chunks`.

    The pairwise difference is one matmul, `[c, 1] @ [1, -c]^T`, for the reason `moonshine_export.
    window_mask` gives: the engine broadcasts one operand into the other, never both. Chunk indices are
    integers far below 2^24, so the product is exact in f32."""
    idx = torch.arange(n, device=device).to(torch.float32)
    c = torch.floor(idx / chunk)
    ones = c * 0.0 + 1.0
    d = torch.matmul(torch.stack([c, ones], dim=1), torch.stack([ones, -c], dim=0))
    return ((d > -0.5) & (d < left_chunks + 0.5)).view(1, 1, n, n)


def _relative_positions(n: int, inv_freq):
    """`NemotronAsrStreamingEncoderRelPositionalEncoding` with no cache: sin/cos of the relative
    distances `n - 1` down to `-(n - 1)`, interleaved, `[1, 2n - 1, hidden]`.

    The reference writes the distances as `arange(n - 1, -n, -1)`. A NEGATIVE-step range reaches
    the engine's RANGE_1D, which in loom 1.0.0rc16 and earlier forces `end > start` and returns
    -1 elements, so this spells the same values with a positive step, `-arange(1 - n, n)`, whose
    bounds are both symbolic expressions of the length. (`(n - 1) - arange(2n - 1)` does not do: the
    shape-derived scalar `n - 1` reaches the graph as a 4-wide tensor.) Identical numbers; the file
    then runs on the released wheels."""
    distances = -torch.arange(1 - n, n, device=inv_freq.device).to(torch.float32)
    freqs = distances[:, None] * inv_freq[None, :]
    return torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(1, 2 * n - 1, -1)


class _NemotronEncoderWrapper(nn.Module):
    """`(waveform [1, n], prompt [1]) -> [1, T, decoder_hidden]`: mel, subsampling, the conformer layers
    under the chunked-limited mask, the language prompt, and the encoder projection."""

    def __init__(self, model, extractor, num_lookahead_tokens: int):
        super().__init__()
        encoder = model.encoder
        self.mel = _NemotronMel(extractor)
        self.subsampling = encoder.subsampling
        self.input_scale = float(encoder.input_scale)
        self.register_buffer("inv_freq", encoder.encode_positions.inv_freq.detach().clone().float())
        self.layers = encoder.layers
        left, right = encoder._resolve_attn_context(num_lookahead_tokens)
        self.chunk = int(right) + 1
        self.left_chunks = int(left) // self.chunk

        projector = model.prompt_projector
        hidden = int(model.config.encoder_config.hidden_size)
        weight = projector.linear_1.weight.detach()                      # [inter, hidden + n_prompts]
        self.prompt_hidden = nn.Linear(hidden, weight.shape[0])
        self.prompt_hidden.weight = nn.Parameter(weight[:, :hidden].clone())
        self.prompt_hidden.bias = nn.Parameter(projector.linear_1.bias.detach().clone())
        # Row p of this table is W_p[:, p]: the one-hot's contribution, looked up rather than multiplied.
        self.prompt_table = nn.Embedding.from_pretrained(weight[:, hidden:].t().contiguous(), freeze=True)
        self.prompt_act = projector.act
        self.prompt_out = projector.linear_2
        self.encoder_projector = model.encoder_projector

    def forward(self, waveform, prompt):
        features = self.mel(waveform)
        # No attention mask: every frame the front end emits is valid (see `_NemotronMel`), so the
        # subsampling's per-stage length masks have nothing to zero.
        hidden = self.subsampling(features, None) * self.input_scale
        positions = _relative_positions(hidden.shape[1], self.inv_freq)
        mask = _chunked_limited_mask(hidden.shape[1], self.chunk, self.left_chunks, hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=positions)
        fused = self.prompt_hidden(hidden) + self.prompt_table(prompt).unsqueeze(1)
        fused = self.prompt_out(self.prompt_act(fused))
        return self.encoder_projector(fused)


class _NemotronJoint(nn.Module):
    """`transformers`' joint as the template's triple: `joint_net(enc(f) + pred(g))`. `enc` is the
    identity because `encoder_projector` ran in the encoder phase; `pred` is `decoder_projector`, which
    `transformers` applies after the LSTM and NeMo applies inside the joint -- the same linear map."""

    def __init__(self, model):
        super().__init__()
        self.enc = nn.Identity()
        self.pred = model.decoder.decoder_projector
        self.joint_net = nn.Sequential(model.joint.activation, model.joint.head)


@dataclass
class ASRNemotronExportConfig(BaseTransducerExportConfig):
    """Nemotron 3.5 ASR (`nemotron3_5_asr`), an HF directory loaded through `transformers`."""

    architecture: str = "nemotron-asr"
    output_path: str = "nemotron-asr.gguf"
    # Filled during `phases()`, READ off the checkpoint.
    default_prompt: Optional[int] = None
    prompt_dictionary: Optional[dict] = None
    language_tag_ids: Tuple[int, ...] = ()
    num_lookahead_tokens: Optional[int] = None

    __unchecked__ = {
        **BaseTransducerExportConfig.__unchecked__,
        "default_prompt": Unchecked("READ off config.json (`default_prompt_id`, the `auto` prompt) during "
                                    "phases(); checked to be a row of the prompt table"),
        "prompt_dictionary": Unchecked("READ off processor_config.json, the processor's own language -> "
                                       "prompt-id table; every id is checked to be a row of the table"),
        "language_tag_ids": Unchecked("READ off tokenizer.json: the added special tokens spelled "
                                      "`<xx-YY>`, which the model writes after a sentence"),
        "num_lookahead_tokens": Unchecked("READ off processor_config.json's "
                                          "`default_num_lookahead_tokens`, the processor's own default "
                                          "and so the one `generate()` runs with"),
    }

    def load_model(self):
        from transformers import Nemotron3_5AsrForRNNT

        print(f"Loading Nemotron ASR from {self.checkpoint}...")
        # The attention implementation does not matter to the trace -- `_NemotronEncoderWrapper` passes
        # a boolean mask -- but eager is the one that traces to plain ops.
        return Nemotron3_5AsrForRNNT.from_pretrained(
            self.checkpoint, dtype=torch.float32, attn_implementation="eager").eval()

    def transducer_parts(self, model) -> TransducerParts:
        if not isinstance(model.decoder.lstm, nn.LSTM):
            raise ValueError("Nemotron's prediction network is expected to be an nn.LSTM")
        return TransducerParts(embed=model.decoder.embedding, lstm=model.decoder.lstm,
                               joint=_NemotronJoint(model))

    def tokenizer_dir(self) -> Optional[str]:
        return self.checkpoint if (Path(self.checkpoint) / "tokenizer.json").is_file() else None

    def encoder_phase(self, model):
        import coremltools as ct
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        extractor = processor.feature_extractor
        self.num_lookahead_tokens = int(processor.default_num_lookahead_tokens)
        self.prompt_dictionary = dict(processor.prompt_dictionary)
        self.default_prompt = int(model.config.default_prompt_id)
        n_prompts = int(model.config.num_prompts)
        bad = {k: v for k, v in self.prompt_dictionary.items() if not 0 <= int(v) < n_prompts}
        if bad or not 0 <= self.default_prompt < n_prompts:
            raise ValueError(f"prompt ids outside the {n_prompts}-row prompt table: {bad or self.default_prompt}")
        self.language_tag_ids = _language_tag_ids(Path(self.checkpoint) / "tokenizer.json")

        sample_rate = int(extractor.sampling_rate)
        wrapper = _NemotronEncoderWrapper(model, extractor, self.num_lookahead_tokens).eval()
        n_samples = int(TRACE_SECONDS * sample_rate)
        dummy = (torch.randn(1, n_samples, dtype=torch.float32) * 0.1,
                 torch.tensor([self.default_prompt], dtype=torch.int64))
        seq_dim = ct.RangeDim(int(MIN_SECONDS * sample_rate), int(MAX_SECONDS * sample_rate))
        mil_inputs = [ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
                      ct.TensorType(name="prompt", shape=(1,), dtype=np.int32)]
        phase = ExportPhase(name="encoder", wrapper=wrapper, dummy_inputs=dummy,
                            mil_inputs=mil_inputs, root_axis=self.root_axis)
        return phase, int(model.config.decoder_hidden_size)

    def backend_kwargs(self) -> dict:
        """The base's, with the vocabulary read off `tokenizer.json`: this checkpoint ships no `.model`
        protobuf. It is a SentencePiece BPE (Metaspace, `<0xNN>` byte pieces), written as llama.cpp's
        "llama" vocabulary like Parakeet's. Named rather than detected, because `tokenizer_detect`
        reads any BPE `tokenizer.json` as byte-level GPT-2 BPE, whose detokenizer leaves the `▁`
        word boundaries out (JFK came back as one word)."""
        kwargs = super().backend_kwargs()
        kwargs["tokenizer_family"] = "sentencepiece_json"
        return kwargs

    def encoder_inputs(self) -> dict:
        from .driver_ir import Var

        return {"waveform": Var("_waveform"), "prompt": Var("_prompt")}

    def encoder_prelude(self) -> List:
        from .driver_components import ExportConstants, LuaFragment

        return [
            ExportConstants(values={"DEFAULT_PROMPT": self.default_prompt}),
            LuaFragment(Path(__file__).resolve().parent / "nemotron_asr_driver" / "00_prompt.lua",
                        reads=("DEFAULT_PROMPT",), defines=("_prompt",)),
        ]

    def contract(self) -> dict:
        """The language table (names -> prompt ids, the processor's own dictionary, `auto` included)
        and the language tags as control ids, which the engine drops from the transcript."""
        contract = super().contract()
        if self.prompt_dictionary:
            names = sorted(self.prompt_dictionary)
            contract["asr.language_names"] = names
            contract["asr.language_ids"] = [int(self.prompt_dictionary[n]) for n in names]
        if self.language_tag_ids:
            contract["asr.control_ids"] = list(self.language_tag_ids)
        return contract


def _language_tag_ids(tokenizer_json: Path) -> Tuple[int, ...]:
    """The added special tokens spelled `<xx-YY>`: the language tags the model writes."""
    import re

    added = json.loads(tokenizer_json.read_text()).get("added_tokens") or []
    return tuple(sorted(int(t["id"]) for t in added
                        if t.get("special") and re.fullmatch(r"<[a-z]{2,3}-[A-Z]{2}>", t["content"])))


def _is_nemotron_asr(path: Path) -> bool:
    config = path / "config.json"
    if not path.is_dir() or not config.is_file():
        return False
    try:
        return json.loads(config.read_text()).get("model_type") == MODEL_TYPE
    except (json.JSONDecodeError, OSError):
        return False


def _build_nemotron_asr(path: Path, output_path: str) -> ASRNemotronExportConfig:
    return ASRNemotronExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRNemotronExportConfig,
        recognizers=[ModelRecognizer(name="nemotron-asr", detect=_is_nemotron_asr,
                                     build_config=_build_nemotron_asr)],
    ))
