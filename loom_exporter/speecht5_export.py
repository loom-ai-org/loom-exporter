"""Export SpeechT5 (`microsoft/speecht5_tts` + `microsoft/speecht5_hifigan`) -- family 9b's first leaf,
and the first model here whose autoregressive loop emits MEL FRAMES.

    text -> SentencePiece CHARACTER ids (+ `</s>`) -> 12-layer encoder with Shaw relative-key attention
         -> per step: previous mel frame -> prenet (two ReLU layers with ALWAYS-ON dropout) + scaled
                      position + speaker x-vector -> 6-layer KV-cached decoder, cross-attending
                   -> 2 mel frames (`reduction_factor`) + 2 stop logits
         -> 5-layer convolutional postnet over the whole spectrogram -> HiFi-GAN -> 16 kHz

Five phases:
  - `encoder`:  `(input_ids, position_ids, rel_index) -> [1, n, 768]`.
  - `cross_kv`: the decoder's cross-attention K and V for every layer, once (`t5_export`'s split).
  - `decoder`:  one step, KV-cached; `-> (spectrum [1, n, 160], stop_logits [1, n, 2])`.
  - `postnet`:  `spectrogram + postnet(spectrogram)`, `[1, frames, 80]`.
  - `vocoder`:  the separate HiFi-GAN checkpoint, `[1, frames, 80] -> [1, 256 * frames]`.

The loop is hand-written Lua (`speecht5_driver/`), Pocket-TTS's shape: a KV-cached step over a
continuous frame, a stop head read back, nothing sampled.

**Three things are this family's own.**

* **The encoder's relative position bias depends on the QUERY.** SpeechT5 adds
  `q_scaled . pe_k[clip(i - j) + 160]` to every score (Shaw et al.), so unlike T5's bucketed table it
  cannot be summed into the mask on the host. The driver hands the graph the `[n, n]` INDEX matrix
  (`speecht5_relative_index`), the graph gathers `pe_k` rows by it and forms the bias as one batched
  matmul over the query axis. The index is data rather than a shape read, which is what keeps it out
  of the shape walk's way.
* **The prenet's dropout is always on**, at inference too (the reference says so at the call: Tacotron
  2 §2.2). Its two masks per step are graph INPUTS the driver draws (`loom.uniform_array`), which is
  loom.cpp ADR-042's rule -- the driver draws graph-input noise -- and what lets a caller pin them
  (`inputs.masks`) for a gate. Each mask is `{0, 1}` and the graph applies the `1 / (1 - p)` scale.
* **The voice is a 512-d x-vector** (SpeechBrain `spkrec-xvect-voxceleb`), L2-normalised in the graph.
  The default ships as a driver weight: `Matthijs/cmu-arctic-xvectors`'s `slt` utterance that every
  published SpeechT5 example uses. A caller's own x-vector is `inputs.speaker`.

The postnet's batch norms are applied as explicit affine ops (eval statistics), and the speaker
projection `Linear(cat(h, spk))` is split into `W_h h + W_s spk` so the speaker row broadcasts instead
of being expanded to a traced length.

Usage:
  loom-export ~/Dev/models/speecht5-tts -o speecht5.gguf --task text-to-speech --model speecht5
(the directory must hold the vocoder under `hifigan/` and the voice set under
`xvectors/spkrec-xvect.zip`, F5-TTS's precedent for a second checkpoint the export needs.)
"""
import io
import json
import math
import zipfile
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

SAMPLE_RATE = 16000
# The vocoder's total upsampling (4 x 4 x 4 x 4): waveform samples per mel frame.
HOP_LENGTH = 256
# `_generate_speech`'s defaults.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MINLENRATIO = 0.0
DEFAULT_MAXLENRATIO = 20.0
# The utterance whose x-vector every published SpeechT5 example uses (`embeddings_dataset[7306]`).
DEFAULT_VOICE = "cmu_us_slt_arctic-wav-arctic_a0508"
XVECTOR_ZIP = Path("xvectors") / "spkrec-xvect.zip"

# Trace lengths: odd, distinct from each other and from every static dimension.
TRACE_TOKENS = 13
TRACE_STEPS = 7
TRACE_SRC = 11
TRACE_FRAMES = 9


def _heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """`[b, t, heads * head_dim] -> [b, heads, t, head_dim]`, with every extent written out (no `-1`)."""
    b, t, _ = x.shape
    return x.view(b, t, heads, head_dim).transpose(1, 2)


def _merge(ctx: torch.Tensor) -> torch.Tensor:
    b, h, t, d = ctx.shape
    return ctx.transpose(1, 2).reshape(b, t, h * d)


def _encoder_self_attention(attn, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """`SpeechT5Attention` with a relative position bias, no mask. `pos` is `pe_k` gathered by the
    relative index, `[n, n, head_dim]`.

    The reference forms the bias as `matmul(q^T, pos^T)` batched over the QUERY position -- for each
    `i`, `q[:, i, :] @ pos[i]^T` -- with `q` already scaled by `head_dim ** -0.5`. Same here, in four
    transposes and one matmul."""
    h, d = attn.num_heads, attn.head_dim
    q = _heads(attn.q_proj(x) * attn.scaling, h, d)                   # [1, h, n, d]
    k = _heads(attn.k_proj(x), h, d)
    v = _heads(attn.v_proj(x), h, d)
    bias = torch.matmul(q[0].transpose(0, 1), pos.transpose(1, 2))     # [n, h, n]
    scores = torch.matmul(q, k.transpose(-1, -2)) + bias.transpose(0, 1).unsqueeze(0)
    ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
    return attn.out_proj(_merge(ctx))


def _decoder_self_attention(attn, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The causal self-attention in the form `fuse_loom_attention` matches: Q scaled, `Q @ K^T + mask`,
    softmax, `@ V`. Fused, it becomes an `ATTENTION` node with a KV cache."""
    h, d = attn.num_heads, attn.head_dim
    q = _heads(attn.q_proj(x) * attn.scaling, h, d)
    k = _heads(attn.k_proj(x), h, d)
    v = _heads(attn.v_proj(x), h, d)
    scores = torch.matmul(q, k.transpose(-1, -2)) + mask
    return attn.out_proj(_merge(torch.matmul(torch.softmax(scores, dim=-1), v)))


def _cross_attention(attn, x: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    """Cross-attention over K/V projected once by `cross_kv`. No mask (one unpadded source sequence,
    loom.cpp ADR-019), so it is not fused and holds no cache."""
    h, d = attn.num_heads, attn.head_dim
    q = _heads(attn.q_proj(x) * attn.scaling, h, d)
    k, v = _heads(xk, h, d), _heads(xv, h, d)
    scores = torch.matmul(q, k.transpose(-1, -2))
    return attn.out_proj(_merge(torch.matmul(torch.softmax(scores, dim=-1), v)))


class EncoderPhase(nn.Module):
    """`(input_ids [1, n], position_ids [1, n], rel_index [n, n]) -> [1, n, 768]`.

    `SpeechT5EncoderWithTextPrenet`: the embedding plus `alpha * pe[position]` (the scaled positional
    encoding, gathered by a position INPUT rather than sliced by a traced length), the stack's input
    layer norm, then twelve post-norm layers sharing one relative bias table."""

    def __init__(self, model):
        super().__init__()
        encoder = model.speecht5.encoder
        prenet = encoder.prenet
        self.embed_tokens = prenet.embed_tokens
        self.register_buffer("pe", prenet.encode_positions.pe[0].detach().clone())
        self.alpha = prenet.encode_positions.alpha
        stack = encoder.wrapped_encoder
        self.layer_norm = stack.layer_norm
        self.pe_k = stack.embed_positions.pe_k
        self.layers = stack.layers

    def forward(self, input_ids, position_ids, rel_index):
        x = self.embed_tokens(input_ids) + self.alpha * F.embedding(position_ids, self.pe)
        x = self.layer_norm(x)
        # Gathered through a FLAT index: `ggml_get_rows` reads a 2-D index as a batch of 1-D lookups
        # (one per row of a 3-D table), so `pe_k(rel_index)` on the `[n, n]` matrix aborts the engine.
        # `n` is read off `position_ids`' axis 1, NOT `rel_index.shape[0]`: the shape walk reads a
        # torch axis 0 that derives to the root axis as a batch size of 1 (`value_facts`' batch
        # guess), and the reshape below came out `[1, 1, 64]`.
        n = position_ids.shape[1]
        pos = self.pe_k(rel_index.reshape(n * n)).view(n, n, self.pe_k.embedding_dim)   # [n, n, head_dim]
        for layer in self.layers:
            x = layer.layer_norm(x + _encoder_self_attention(layer.attention, x, pos))
            x = layer.final_layer_norm(x + layer.feed_forward(x))
        return x


class CrossKvPhase(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`: every decoder layer's cross-attention K and V, once."""

    def __init__(self, model):
        super().__init__()
        self.projs = nn.ModuleList()
        for layer in model.speecht5.decoder.wrapped_decoder.layers:
            self.projs.append(layer.encoder_attn.k_proj)
            self.projs.append(layer.encoder_attn.v_proj)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


def cross_kv_input_names(n_layers: int) -> tuple:
    names = []
    for i in range(n_layers):
        names += [f"xk_{i}", f"xv_{i}"]
    return tuple(names)


class DecoderPhase(nn.Module):
    """One decoder step: `(frame, position_ids, prenet_mask_0, prenet_mask_1, speaker, attention_mask,
    xk_0, xv_0, ...) -> (spectrum [1, n, 2 * 80], stop_logits [1, n, 2])`.

    `SpeechT5SpeechDecoderPrenet` + the wrapped decoder + `feat_out`/`prob_out`. The reference runs the
    prenet over the WHOLE output sequence every step and keeps its last row; a row depends on nothing
    but its own frame, position and masks, so running it on the new row alone is the same number."""

    def __init__(self, model):
        super().__init__()
        prenet = model.speecht5.decoder.prenet
        self.prenet_layers = prenet.layers
        self.final_layer = prenet.final_layer
        self.register_buffer("pe", prenet.encode_positions.pe[0].detach().clone())
        self.alpha = prenet.encode_positions.alpha
        self.dropout_scale = 1.0 / (1.0 - float(model.config.speech_decoder_prenet_dropout))
        hidden = model.config.hidden_size
        w = prenet.speaker_embeds_layer.weight.detach()
        self.speaker_h = nn.Linear(hidden, hidden, bias=False, dtype=w.dtype)
        self.speaker_s = nn.Linear(w.shape[1] - hidden, hidden, dtype=w.dtype)
        self.speaker_h.weight.data.copy_(w[:, :hidden])
        self.speaker_s.weight.data.copy_(w[:, hidden:])
        self.speaker_s.bias.data.copy_(prenet.speaker_embeds_layer.bias.detach())
        self.layers = model.speecht5.decoder.wrapped_decoder.layers
        self.feat_out = model.speech_decoder_postnet.feat_out
        self.prob_out = model.speech_decoder_postnet.prob_out

    def forward(self, frame, position_ids, prenet_mask_0, prenet_mask_1, speaker, attention_mask, *cross):
        x = frame
        for layer, mask in zip(self.prenet_layers, (prenet_mask_0, prenet_mask_1)):
            x = F.relu(layer(x)) * mask * self.dropout_scale
        x = self.final_layer(x) + self.alpha * F.embedding(position_ids, self.pe)
        spk = speaker / torch.sqrt((speaker * speaker).sum(dim=-1, keepdim=True))
        x = F.relu(self.speaker_h(x) + self.speaker_s(spk).unsqueeze(1))
        for i, layer in enumerate(self.layers):
            x = layer.self_attn_layer_norm(x + _decoder_self_attention(layer.self_attn, x, attention_mask))
            x = layer.encoder_attn_layer_norm(
                x + _cross_attention(layer.encoder_attn, x, cross[2 * i], cross[2 * i + 1]))
            x = layer.final_layer_norm(x + layer.feed_forward(x))
        return self.feat_out(x), self.prob_out(x)


class PostnetPhase(nn.Module):
    """`spectrogram [1, frames, 80] -> spectrogram + postnet(spectrogram)`.

    Five `Conv1d -> BatchNorm1d (-> tanh)` layers. The batch norm is the eval-mode affine
    `(y - mean) * (gamma / sqrt(var + eps)) + beta`, spelled out so no `batch_norm` op traces."""

    def __init__(self, model):
        super().__init__()
        self.convs = nn.ModuleList()
        self.tanh = []
        layers = model.speech_decoder_postnet.layers
        for i, layer in enumerate(layers):
            bn = layer.batch_norm
            self.convs.append(layer.conv)
            inv = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
            self.register_buffer(f"bn_mean_{i}", bn.running_mean.detach().clone().view(1, -1, 1))
            self.register_buffer(f"bn_scale_{i}", inv.clone().view(1, -1, 1))
            self.register_buffer(f"bn_shift_{i}", bn.bias.detach().clone().view(1, -1, 1))
            self.tanh.append(layer.activation is not None)

    def forward(self, spectrogram):
        x = spectrogram.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            x = (conv(x) - getattr(self, f"bn_mean_{i}")) * getattr(self, f"bn_scale_{i}") \
                + getattr(self, f"bn_shift_{i}")
            if self.tanh[i]:
                x = torch.tanh(x)
        return spectrogram + x.transpose(1, 2)


class VocoderPhase(nn.Module):
    """`mel [1, frames, 80] -> waveform [1, 256 * frames]`: `SpeechT5HifiGan.forward`, unchanged."""

    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel):
        return self.vocoder(mel)


def _check_fused_attention(topo: dict, n_layers: int) -> int:
    """Exactly `n_layers` ATTENTION nodes in the decoder, all cached: the self-attention blocks. A
    fused cross-attention block would take a KV cache slot the self-attention blocks address."""
    total = [n for n in topo["nodes"] if n["op"] == "ATTENTION"]
    cached = [n for n in total if n.get("attrs", {}).get("kv_cache", True)]
    if len(total) != n_layers or len(cached) != n_layers:
        raise ValueError(f"speecht5 decoder: fused {len(total)} ATTENTION node(s) ({len(cached)} cached) "
                         f"for {n_layers} self-attention blocks; cross-attention must stay unfused")
    return n_layers


def read_xvector(model_dir: Path, name: str, dim: int = 512) -> np.ndarray:
    """One x-vector out of `<model>/xvectors/spkrec-xvect.zip` (`Matthijs/cmu-arctic-xvectors`), by
    utterance name."""
    path = Path(model_dir) / XVECTOR_ZIP
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. SpeechT5 needs a speaker x-vector and its checkpoint ships none; "
            f"download `spkrec-xvect.zip` from the `Matthijs/cmu-arctic-xvectors` dataset into "
            f"{path.parent}/ (do not unzip it: 7931 files).")
    with zipfile.ZipFile(path) as z:
        member = f"spkrec-xvect/{name}.npy"
        if member not in z.namelist():
            raise KeyError(f"{path.name} has no {member}")
        x = np.load(io.BytesIO(z.read(member))).astype(np.float32)
    if x.shape != (dim,) or not np.isfinite(x).all():
        raise ValueError(f"{member}: expected {dim} finite floats (the checkpoint's "
                         f"`speaker_embedding_dim`), got {x.shape}")
    return x


@dataclass(kw_only=True)
class SpeechT5ExportConfig(BaseMultiPhaseModelExportConfig):
    """A `microsoft/speecht5_tts` directory (with `hifigan/` and `xvectors/`) -> one Loom GGUF."""

    architecture: str = "speecht5"
    model_dir: str
    voice: str = DEFAULT_VOICE
    root_axis: str = "n_tokens"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "speecht5_driver"
    _driver_weights: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _n_layers: int = field(default=0, init=False, repr=False)
    _max_text: int = field(default=0, init=False, repr=False)
    _max_speech: int = field(default=0, init=False, repr=False)
    _max_relative: int = field(default=0, init=False, repr=False)
    _num_mel_bins: int = field(default=80, init=False, repr=False)
    _reduction: int = field(default=2, init=False, repr=False)
    _prenet_units: int = field(default=256, init=False, repr=False)
    _eos_token_id: int = field(default=2, init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)

    __links__ = {"root_axis": Axis()}
    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string; it names this export"),
        "model_dir": Unchecked("path to the speecht5_tts directory; the recognizer read its config.json"),
        "voice": Unchecked("an utterance name in the x-vector zip; read and shape-checked by read_xvector"),
        "decomposition": Unchecked("MultiPhase by construction -- five graphs and a hand-written loop"),
        "driver_script_path": Unchecked("the hand-written fragments are still parsed and checked "
                                        "against the traced topologies by LuaFragment"),
        "_driver_weights": Unchecked("READ during phases() and shipped as driver weights"),
        "_n_layers": Unchecked("READ off the checkpoint in phases()"),
        "_max_text": Unchecked("READ: `config.max_text_positions`, the encoder's position table"),
        "_max_speech": Unchecked("READ: `config.max_speech_positions`, the decoder's position table "
                                 "and so the most steps a loop can take"),
        "_max_relative": Unchecked("READ: `config.encoder_max_relative_position`; cross-checked "
                                   "against `pe_k`'s row count in phases()"),
        "_num_mel_bins": Unchecked("READ: `config.num_mel_bins`"),
        "_reduction": Unchecked("READ: `config.reduction_factor`"),
        "_prenet_units": Unchecked("READ: `config.speech_decoder_prenet_units`"),
        "_eos_token_id": Unchecked("READ: `config.eos_token_id`, which the tokenizer appends"),
        "cross_kv_names": Unchecked("derived in phases() by `cross_kv_input_names`, which also orders "
                                    "`CrossKvPhase`'s outputs"),
    }

    def load_models(self):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan

        # Checked here, where a checkpoint is read, and not when the config is built: the component
        # catalogue builds every registered config with no checkpoint in hand.
        vocoder_dir = Path(self.model_dir) / "hifigan"
        if not (vocoder_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"{vocoder_dir} does not exist. SpeechT5 predicts mel spectrograms; the waveform comes "
                f"from the separate `microsoft/speecht5_hifigan` checkpoint, which the export folds into "
                f"the same GGUF. Download it into {vocoder_dir}/.")
        model = SpeechT5ForTextToSpeech.from_pretrained(self.model_dir, dtype=torch.float32).eval()
        vocoder = SpeechT5HifiGan.from_pretrained(str(Path(self.model_dir) / "hifigan"),
                                                  dtype=torch.float32).eval()
        return model, vocoder

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model, vocoder = self.load_models()
        cfg = model.config
        self._n_layers = int(cfg.decoder_layers)
        self._max_text = int(cfg.max_text_positions)
        self._max_speech = int(cfg.max_speech_positions)
        self._max_relative = int(cfg.encoder_max_relative_position)
        self._num_mel_bins = int(cfg.num_mel_bins)
        self._reduction = int(cfg.reduction_factor)
        self._prenet_units = int(cfg.speech_decoder_prenet_units)
        self._eos_token_id = int(cfg.eos_token_id)
        if int(cfg.speech_decoder_prenet_layers) != 2:
            raise NotImplementedError(f"the decoder takes one mask input per prenet layer and declares "
                                      f"two; this checkpoint has {cfg.speech_decoder_prenet_layers}")
        pe_k_rows = model.speecht5.encoder.wrapped_encoder.embed_positions.pe_k.num_embeddings
        if pe_k_rows != 2 * self._max_relative:
            raise ValueError(f"pe_k has {pe_k_rows} rows for a max relative position of "
                             f"{self._max_relative}; the driver's index assumes 2 * max")
        self._driver_weights = {"speaker": read_xvector(Path(self.model_dir), self.voice,
                                                        int(cfg.speaker_embedding_dim))}
        self.cross_kv_names = cross_kv_input_names(self._n_layers)

        hidden, mels, units = int(cfg.hidden_size), self._num_mel_bins, self._prenet_units
        text_dim = ct.RangeDim(1, self._max_text)
        src_dim = ct.RangeDim(1, self._max_text)
        step_dim = ct.RangeDim(1, self._max_speech)
        frame_dim = ct.RangeDim(1, self._reduction * self._max_speech)
        decoder_inputs = [
            ct.TensorType(name="frame", shape=(1, step_dim, mels), dtype=np.float32),
            ct.TensorType(name="position_ids", shape=(1, step_dim), dtype=np.int32),
            ct.TensorType(name="prenet_mask_0", shape=(1, step_dim, units), dtype=np.float32),
            ct.TensorType(name="prenet_mask_1", shape=(1, step_dim, units), dtype=np.float32),
            ct.TensorType(name="speaker", shape=(1, int(cfg.speaker_embedding_dim)), dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=(1, 1, step_dim, step_dim), dtype=np.float32),
        ] + [ct.TensorType(name=name, shape=(1, src_dim, hidden), dtype=np.float32)
             for name in self.cross_kv_names]

        # ORDER IS LOAD-BEARING, as in t5_export: nothing here replaces modules, but keep the cross
        # projections' owner built before the decoder's for the day something does.
        cross_kv = CrossKvPhase(model).eval()
        decoder = DecoderPhase(model).eval()
        mask = torch.triu(torch.full((TRACE_STEPS, TRACE_STEPS), float("-inf")), diagonal=1)
        rel = torch.arange(TRACE_TOKENS).view(-1, 1) - torch.arange(TRACE_TOKENS).view(1, -1)
        rel = rel.clamp(-self._max_relative, self._max_relative - 1) + self._max_relative
        return [
            ExportPhase(
                name="encoder",
                wrapper=EncoderPhase(model).eval(),
                dummy_inputs=(torch.randint(4, 70, (1, TRACE_TOKENS), dtype=torch.int32),
                              torch.arange(TRACE_TOKENS, dtype=torch.int32).view(1, -1),
                              rel.to(torch.int32)),
                mil_inputs=[
                    ct.TensorType(name="input_ids", shape=(1, text_dim), dtype=np.int32),
                    ct.TensorType(name="position_ids", shape=(1, text_dim), dtype=np.int32),
                    ct.TensorType(name="rel_index", shape=(text_dim, text_dim), dtype=np.int32),
                ],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv,
                dummy_inputs=(torch.randn(1, TRACE_SRC, hidden),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, src_dim, hidden), dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder,
                dummy_inputs=(
                    torch.randn(1, TRACE_STEPS, mels),
                    torch.arange(TRACE_STEPS, dtype=torch.int32).view(1, -1),
                    (torch.rand(1, TRACE_STEPS, units) < 0.5).float(),
                    (torch.rand(1, TRACE_STEPS, units) < 0.5).float(),
                    torch.randn(1, int(cfg.speaker_embedding_dim)),
                    mask.view(1, 1, TRACE_STEPS, TRACE_STEPS),
                ) + tuple(torch.randn(1, TRACE_SRC, hidden) for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis="n_tokens",
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                topology_rewrite=lambda topo: _check_fused_attention(topo, self._n_layers),
                fuse_attention=True,
                kv_cache_size=self._max_speech,
            ),
            ExportPhase(
                name="postnet",
                wrapper=PostnetPhase(model).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, mels),),
                mil_inputs=[ct.TensorType(name="spectrogram", shape=(1, frame_dim, mels),
                                          dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="vocoder",
                wrapper=VocoderPhase(vocoder).eval(),
                dummy_inputs=(torch.randn(1, TRACE_FRAMES, mels),),
                mil_inputs=[ct.TensorType(name="mel", shape=(1, frame_dim, mels), dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import CALLER, DriverInputs, DriverReturn, ExportConstants, LuaFragment
        from .driver_ir import Len

        fragment = self.driver_script_path
        constants = {
            "N_LAYERS": self._n_layers,
            "NUM_MEL_BINS": self._num_mel_bins,
            "REDUCTION_FACTOR": self._reduction,
            "PRENET_UNITS": self._prenet_units,
            "MAX_TEXT_POSITIONS": self._max_text,
            "MAX_SPEECH_POSITIONS": self._max_speech,
            "MAX_RELATIVE_POSITION": self._max_relative,
            "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
            "DEFAULT_MINLENRATIO": DEFAULT_MINLENRATIO,
            "DEFAULT_MAXLENRATIO": DEFAULT_MAXLENRATIO,
        }
        return [
            ExportConstants(values=constants),
            DriverInputs(bindings=(("tokens", CALLER),), n_tokens=Len("tokens")),
            LuaFragment(fragment / "00_header.lua", top_level=True,
                        defines=("speecht5_relative_index", "speecht5_draw_mask")),
            LuaFragment(fragment / "01_speech.lua", reads=("tokens",) + tuple(constants),
                        defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def hparams(self) -> dict:
        return {"n_ctx": self._max_text} if self._max_text else {}

    def contract(self) -> dict:
        contract = super().contract()
        contract["input.kind"] = "text"
        contract["text.frontend"] = "vocab"
        contract["sample_rate"] = SAMPLE_RATE
        contract["tts.voices"] = [self.voice]
        return contract

    def backend_kwargs(self) -> dict:
        # `SpeechT5Tokenizer.build_inputs_with_special_tokens` appends `</s>` to every sequence; the
        # protobuf records no such flag, so the export states it (t5_export's reason).
        kwargs = dict(flat_namespace=False, root_axis=self.root_axis, hparams=self.hparams(),
                      tokenizer_dir=self.model_dir, add_eos_token=True, eos_token_id=self._eos_token_id)
        if self._driver_weights is not None:
            kwargs["driver_weights"] = dict(self._driver_weights)
        return kwargs


def _is_speecht5_tts(path: Path) -> bool:
    """An HF directory whose `config.json` is a `speecht5` TEXT-TO-SPEECH checkpoint. The ASR and
    voice-conversion heads share the `model_type` and are not this export."""
    config_path = path / "config.json"
    if not (path.is_dir() and config_path.is_file()):
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (isinstance(config, dict) and config.get("model_type") == "speecht5"
            and "SpeechT5ForTextToSpeech" in (config.get("architectures") or []))


def _build_speecht5(path: Path, output_path: str) -> LoomExportConfig:
    return SpeechT5ExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=SpeechT5ExportConfig,
        recognizers=[ModelRecognizer(name="speecht5", detect=_is_speecht5_tts,
                                     build_config=_build_speecht5)],
    ))
