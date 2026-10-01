"""NVIDIA Canary -- NeMo's `EncDecMultiTaskModel`, a FastConformer encoder feeding a transformer decoder
that cross-attends to it -- on `nvidia/canary-1b-v2`.

**Two families already in this tree, joined.** The encoder half is family 1's: NeMo's own mel front end
and a FastConformer, traced over `(waveform, length)` exactly as Parakeet's encoder phase is. The
decoder half is family 2/6's: an `encoder` run once, a `cross_kv` phase that projects its output into
every decoder layer's cross-attention K/V once, and a KV-cached `decoder` step in a loop. T5
(`t5_export`) is the closest module to read beside this one, because its encoder length is dynamic too
and its decoder therefore carries the second symbol `n_enc_frames`.

What is Canary's own, and nothing else is:

* **The decoder is NeMo's, not transformers'.** `TransformerEmbedding.forward` builds its position ids
  from `torch.arange(start_pos, start_pos + seq_len)` with Python ints, which a trace bakes. The wrapper
  therefore composes the embedding's three parts itself (token table, fixed sinusoid table indexed by a
  `position_ids` INPUT, LayerNorm) and walks the decoder blocks directly with the engine's causal mask
  -- the same reason Whisper and the causal LMs take `position_ids`/`attention_mask` as inputs.
* **The prompt is NeMo's own encoding of the turn `transcribe(source_lang=, target_lang=)` builds**,
  not a template written here: `▁<|startofcontext|><|startoftranscript|><|emo:undefined|><|src|>
  <|tgt|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`. The leading `▁` is SentencePiece's dummy-prefix
  space, produced because the formatter encodes the template as TEXT; it is in every prompt NeMo feeds
  the model, and a template spelled out by hand here left it out (the export's own cross-check caught
  that before anything was traced). The two language slots are found by encoding a second pair and
  diffing.
* **The encoder output is cut to `encoded_len` inside the graph.** NeMo masks cross-attention with
  `lens_to_mask(encoded_len)`, and the encoder can emit one frame past it (see
  `nemo_asr_export.EncoderOutput.select`). Cut here, every encoder row the decoder sees is one NeMo lets
  it see, and cross-attention needs no mask at all.
* **Decoding is NeMo's `beam_size: 1` beam search, which is greedy**: one top-1 continuation per step,
  ended by `<|endoftext|>` or `<pad>`, for at most `min(max_sequence_length, n_enc + max_generation_delta)
  - len(prompt)` tokens. The driver states that budget rather than the loop's generic 16.

**What this export does not do.** NeMo's `transcribe()` chunks audio longer than 40 s (the checkpoint's
training ceiling) into overlapping windows; one call here is one window, so a longer clip decodes as one
sequence, which is not what NeMo would return. No timestamps: those come from a SECOND model inside the
`.nemo` (a CTC aligner), which is not exported and whose restore is skipped.

**The language written is a choice separate from the language heard** -- `target_language`, a role this
family introduced. It defaults to ENGLISH (the user's decision, 2026-10-01), so German audio with no
arguments comes back translated; `task="transcribe"`, or a target equal to the source, keeps it German.
"""
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .nemo_asr_export import (
    MAX_SECONDS, MIN_SECONDS, TRACE_SECONDS, _is_nemo_archive, _read_nemo_model_config,
    extract_nemo_tokenizer_dir, prepare_nemo_environment,
)
from .spec_protocol import Unchecked

# The 25 languages canary-1b-v2 is trained on, from its model card's `language:` list. The vocabulary
# carries ~180 language tokens (`<|aa|>` onwards) that the checkpoint was never trained to transcribe,
# so the vocabulary cannot answer which ones are real; the card is the only authority, and every one of
# these is checked to exist as a token before it is declared.
CANARY_V2_LANGUAGES = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
)

# A language pair different from the default on BOTH sides, encoded only to find where the two slots
# sit in the prompt. Any two trained languages that differ from each other and from the default work.
PROBE_PAIR = ("de", "fr")
EOS_PIECE = "<|endoftext|>"
PAD_PIECE = "<pad>"


class _CanaryEncoderWrapper(nn.Module):
    """`(waveform, length) -> encoder states [1, encoded_len, d]`: NeMo's preprocessor, the FastConformer
    and the encoder-to-decoder projection, which is `EncDecMultiTaskModel.forward` up to the point where
    the decoder takes over.

    The slice is the point (see the module docstring): it is what makes cross-attention maskless."""

    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor
        self.encoder = model.encoder
        self.encoder_decoder_proj = model.encoder_decoder_proj

    def forward(self, waveform, length):
        features, feature_len = self.preprocessor(input_signal=waveform, length=length)
        encoded, encoded_len = self.encoder(audio_signal=features, length=feature_len)
        return self.encoder_decoder_proj(encoded.permute(0, 2, 1))[:, :encoded_len[0]]


class _CrossKvSlot(nn.Module):
    """Stands in for a cross-attention `key_net`/`value_net` and returns the tensor handed in from
    outside -- `t5_export._CrossKvSlot`, for the same reason: the projection is a function of the
    encoder alone, so recomputing it per token is 16 matmuls of `[n_enc, 1024] x [1024, 1024]` a step."""

    def __init__(self, holder: list, key: int):
        super().__init__()
        self._holder = holder
        self._key = key

    def forward(self, x):
        return self._holder[self._key]


class _CanaryCrossKvWrapper(nn.Module):
    """`xa -> (k_0, v_0, k_1, v_1, ...)`, every decoder layer's cross-attention K/V, once.

    **Construct this BEFORE `_CanaryDecoderWrapper`**, which replaces the attributes these modules are
    reached through -- built the other way round this phase would trace slots and export no weights.
    The K here is the projection alone: NeMo's `MultiHeadAttention` divides it by `attn_scale` after
    the head split, inside the decoder step, and that division stays there."""

    def __init__(self, decoder_layers):
        super().__init__()
        self.projs = nn.ModuleList()
        for layer in decoder_layers:
            self.projs.append(layer.second_sub_layer.key_net)
            self.projs.append(layer.second_sub_layer.value_net)

    def forward(self, xa):
        return tuple(proj(xa) for proj in self.projs)


def cross_kv_input_names(n_layers: int) -> tuple:
    """`("xk_0", "xv_0", "xk_1", ...)`, the order `_CanaryCrossKvWrapper` returns them in."""
    names = []
    for i in range(n_layers):
        names.append(f"xk_{i}")
        names.append(f"xv_{i}")
    return tuple(names)


class _CanaryDecoderWrapper(nn.Module):
    """`(tokens, position_ids, attention_mask, xk_0, xv_0, ...) -> logits`.

    `TransformerEmbedding.forward` and `TransformerDecoder.forward` are both bypassed, for one reason
    each: the first derives position ids from Python ints (baked by the trace), and the second builds
    its masks from a padding mask with `form_attention_mask` (baked likewise, and a cross-attention mask
    this export does not need -- the encoder output is already cut to its valid length). What runs is
    their parts, in their order: token embedding + sinusoid, LayerNorm, each pre-LN block, the final
    LayerNorm, the tied token classifier -- WITHOUT its log-softmax. The search only ever takes the
    argmax of a step, which the log-softmax does not move, and coremltools lowers `log_softmax` as
    `log(softmax(x))`, which underflows to -inf below about -103: nine of 688k JFK log-probs came out
    -inf where NeMo's were finite. Logits have no such floor, and the step skips a softmax and a log
    over 16384 classes.

    Each block is called with `decoder_keys = decoder_query`, which is what NeMo's own uncached forward
    does; the engine's KV cache then supplies the past keys a cached NeMo step reads from its `mems`.
    The two are the same because a block's K and V are per-position functions of its input."""

    def __init__(self, model):
        super().__init__()
        self.embedding = model.transf_decoder.embedding
        decoder = model.transf_decoder.decoder
        self.layers = decoder.layers
        self.final_layer_norm = decoder.final_layer_norm
        self.head = model.log_softmax
        self._cross = [None] * (2 * len(self.layers))
        for i, layer in enumerate(self.layers):
            layer.second_sub_layer.key_net = _CrossKvSlot(self._cross, 2 * i)
            layer.second_sub_layer.value_net = _CrossKvSlot(self._cross, 2 * i + 1)

    def forward(self, tokens, position_ids, attention_mask, *cross):
        for i, tensor in enumerate(cross):
            self._cross[i] = tensor
        emb = self.embedding
        hidden = emb.layer_norm(emb.token_embedding(tokens) + emb.position_embedding(position_ids))
        for layer in self.layers:
            # `cross[0]` as the encoder states: the slots above ignore what they are handed, and the
            # tensor only has to have the encoder output's shape for `MultiHeadAttention` to reach them.
            # `None` as the encoder mask: every row is valid (the encoder phase cut the rest).
            hidden = layer(hidden, attention_mask, hidden, cross[0], None)[0]
        with self.head.with_log_softmax_enabled(False):
            return self.head(hidden_states=self.final_layer_norm(hidden))


def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def _check_fused_attention(topo: dict, n_layers: int) -> int:
    """Exactly one cached ATTENTION node per decoder layer: the self-attention blocks.

    Cross-attention must NOT fuse. It has no mask add (the decoder step passes `None`), which is what
    leaves it an ordinary softmax; a block that did fuse would take a KV-cache slot the self-attention
    blocks address by occurrence order. The same check `t5_export._check_fused_attention` makes, for
    the same reason."""
    n_fused = sum(1 for node in topo.get("nodes", []) if node.get("op") == "ATTENTION")
    if n_fused != n_layers:
        raise ValueError(
            f"canary decoder: expected {n_layers} fused ATTENTION nodes (one self-attention block per "
            f"layer) and found {n_fused}. Fewer means a self-attention block was not recognised and "
            f"would run uncached; more means a cross-attention block fused and would claim a KV-cache "
            f"slot the self-attention blocks address."
        )
    return 0


def long_form_policy(sample_rate: int, subsampling_factor: int, window_stride: float) -> dict:
    """How NeMo decodes audio past the 40 s Canary was trained on, as the numbers loom.cpp's
    `transcribe` reads (`loom.asr.window_*`, `loom.asr.merge_*`; loom.cpp asr_long_form.h).

    **Read off NeMo, not restated.** The window bounds and the overlap are the defaults of NeMo's own
    `PromptedAudioToTextLhotseDataset._find_optimal_chunk_size(min_sec=30, max_sec=40, overlap_sec=1.0)`,
    taken from its signature, so an export follows the NeMo it was made against. The search runs in
    whole seconds (`range(min_sec, max_sec + 1)`).

    The merge widths are `merge_parallel_chunks`'s: `delay` is the encoder frames in one second
    (`int(1 / (subsampling_factor / 100))`, which assumes NeMo's 10 ms feature stride -- checked below),
    the search covers `delay * max_steps_per_timestep` tokens (`max_steps_per_timestep=2`), and only
    `int(delay * 0.6)` tokens of each new window take part ("approximately 60% of the tokens are non
    blank"). Those two literals live inside the function body, where no signature can be read.

    Without this, one decode runs over the whole file: on 79 s of LibriSpeech that returned 116 of 181
    words (WER 0.43), and on 304 s it looped on "of the world" until the token budget ran out.
    """
    import inspect

    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextLhotseDataset

    defaults = {name: p.default for name, p in
                inspect.signature(PromptedAudioToTextLhotseDataset._find_optimal_chunk_size).parameters.items()
                if p.default is not inspect.Parameter.empty}
    if abs(window_stride - 0.01) > 1e-9:
        raise ValueError(f"canary: NeMo's chunk merge assumes a 10 ms feature stride (it divides the "
                         f"subsampling factor by 100); this checkpoint's is {window_stride} s.")
    delay = int(1 / (subsampling_factor / 100))
    return {
        "window_max_samples": int(defaults["max_sec"] * sample_rate),
        "window_min_samples": int(defaults["min_sec"] * sample_rate),
        "window_search_step_samples": int(sample_rate),
        "window_overlap_samples": int(defaults["overlap_sec"] * sample_rate),
        "merge_search_tokens": int(delay * 2),
        "merge_head_tokens": int(delay * 0.6),
    }


@dataclass(kw_only=True)
class ASRCanaryExportConfig(BaseMultiPhaseModelExportConfig):
    """Canary as three traced phases -- `encoder`, `cross_kv`, `decoder` -- and a driver that runs the
    first two once and loops the third."""

    checkpoint: str = ""
    architecture: str = "canary"
    output_path: str = "canary.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "canary_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # The decoder's trace length, and the encoder-frame count its cross-attention inputs are traced at.
    # Free, not 1 (a size-1 axis may be folded away), and different from each other so the two symbols
    # cannot be confused by the trace.
    trace_tokens: int = 4
    trace_enc: int = 6

    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    n_layers: Optional[int] = field(default=None, init=False, repr=False)
    max_positions: Optional[int] = field(default=None, init=False, repr=False)
    max_generation_delta: Optional[int] = field(default=None, init=False, repr=False)
    prompt_ids: tuple = field(default=(), init=False, repr=False)
    source_slot: Optional[int] = field(default=None, init=False, repr=False)
    target_slot: Optional[int] = field(default=None, init=False, repr=False)
    default_language_id: Optional[int] = field(default=None, init=False, repr=False)
    # The language written when the caller names neither a target nor a task: ENGLISH, by the user's
    # decision (2026-10-01), even for English audio, where it is simply a transcript. Not the
    # checkpoint's default turn, which says "same as the source".
    default_target: str = "en"
    lang_to_id: dict = field(default_factory=dict, init=False, repr=False)
    eos_token_id: Optional[int] = field(default=None, init=False, repr=False)
    pad_token_id: Optional[int] = field(default=None, init=False, repr=False)
    cross_kv_names: tuple = field(default=(), init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)

    __unchecked__ = {
        "checkpoint": Unchecked("path to the .nemo archive, already established by the recognizer's own "
                                "detect(), which reads the archive's config; restore_from raises on "
                                "anything it cannot load"),
        "architecture": Unchecked("the GGUF's own architecture string; no second authority"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("the two hand-written fragments are parsed and cross-checked "
                                        "by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "trace_tokens": Unchecked("a property of the TRACE, not of the model"),
        "trace_enc": Unchecked("same, for the encoder-frame axis"),
        "sample_rate": Unchecked("READ off the checkpoint (`cfg.preprocessor.sample_rate`)"),
        "d_model": Unchecked("READ off the checkpoint (`model_defaults.lm_dec_hidden`)"),
        "n_layers": Unchecked("READ off the checkpoint: the decoder's own layer count"),
        "max_positions": Unchecked("READ off the checkpoint: the decoder's `max_sequence_length`, which "
                                   "is both the sinusoid table's length and the KV-cache capacity"),
        "max_generation_delta": Unchecked("READ off the checkpoint's own decoding config"),
        "prompt_ids": Unchecked("NeMo's own PromptFormatter encoding of the turn transcribe() builds"),
        "source_slot": Unchecked("found by diffing two encodings, and checked to be one of exactly two "
                                 "positions where they differ"),
        "target_slot": Unchecked("same"),
        "default_language_id": Unchecked("READ off the checkpoint's `prompt_defaults`"),
        "default_target": Unchecked("a product decision, not a checkpoint fact -- see the field"),
        "lang_to_id": Unchecked("CANARY_V2_LANGUAGES resolved through the vocabulary; a name with no "
                                "token raises in phases()"),
        "eos_token_id": Unchecked("READ off the vocabulary (`<|endoftext|>`), and the same id NeMo's "
                                  "decoder is handed as `eos`"),
        "pad_token_id": Unchecked("same, for `<pad>`"),
        "cross_kv_names": Unchecked("derived by `cross_kv_input_names`, the function that also orders "
                                    "the cross_kv phase's outputs"),
        "decoder_bindings": Unchecked("(name, kind) per decoder input, derived from the same mil_inputs "
                                      "the trace is declared with"),
    }

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        """`ASRModel.restore_from`, without the second model inside the archive.

        `EncDecMultiTaskModel.__init__` also restores `timestamps_asr_model_*` -- a 600M-parameter CTC
        model NeMo uses for forced-alignment timestamps, 2.5 GB more to hold while tracing a model that
        never calls it. Its private restore hook is replaced for the duration of the load only."""
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

        hook = "_EncDecMultiTaskModel__restore_timestamps_asr_model"
        original = getattr(EncDecMultiTaskModel, hook)
        setattr(EncDecMultiTaskModel, hook, lambda self: None)
        try:
            print(f"Loading NeMo model from {self.checkpoint}...")
            model = nemo_asr.models.ASRModel.restore_from(self.checkpoint, map_location="cpu")
        finally:
            setattr(EncDecMultiTaskModel, hook, original)
        return model.eval()

    def _read_prompt(self, model) -> None:
        """The prompt, its two language slots, the language table and the two stop ids.

        The prompt is NeMo's: the single `user` turn `transcribe(source_lang=, target_lang=)` builds --
        the checkpoint's default slots with the two languages filled in -- encoded by the checkpoint's
        own `PromptFormatter`. Encoding a second pair (`PROBE_PAIR`) and diffing locates the slots; the
        two encodings must differ in exactly those two positions, or the prompt is not the fixed row
        of tokens the driver treats it as."""
        tok = model.tokenizer
        vocab = {tok.ids_to_tokens([i])[0]: i for i in range(tok.vocab_size)}

        def piece(name):
            if name not in vocab:
                raise ValueError(f"canary: the vocabulary has no {name!r} token.")
            return vocab[name]

        if str(model.prompt_format) != "canary2":
            raise ValueError(f"canary: prompt format {model.prompt_format!r}; this export builds the "
                             f"`canary2` prompt.")
        self.lang_to_id = {lang: piece(f"<|{lang}|>") for lang in CANARY_V2_LANGUAGES}
        defaults = next((dict(t["slots"]) for t in model.prompt.get_default_dialog_slots()
                         if t["role"] == "user"), None)
        if defaults is None:
            raise ValueError("canary: the checkpoint's prompt format declares no default `user` turn.")
        source, target = str(defaults["source_lang"]), str(defaults["target_lang"])
        if source != target:
            raise ValueError(f"canary: the checkpoint's default turn translates ({source} -> {target}); "
                             f"the driver's default is a transcript, so this needs a decision first.")

        def encode(src, tgt):
            slots = dict(defaults, source_lang=src, target_lang=tgt)
            return [int(i) for i in model.prompt.encode_dialog(
                turns=[{"role": "user", "slots": slots}])["context_ids"]]

        base = encode(source, source)
        probe_src, probe_tgt = (f"<|{lang}|>" for lang in PROBE_PAIR)
        probe = encode(probe_src, probe_tgt)
        diff = [i for i, (a, b) in enumerate(zip(base, probe)) if a != b]
        if len(base) != len(probe) or len(diff) != 2:
            raise ValueError(
                f"canary: the prompt for {source}->{source} is {base} and for {probe_src}->{probe_tgt} "
                f"is {probe}; they should differ in exactly the two language slots."
            )
        self.source_slot = probe.index(piece(probe_src))
        self.target_slot = probe.index(piece(probe_tgt))
        if sorted((self.source_slot, self.target_slot)) != diff:
            raise ValueError(f"canary: the language slots ({self.source_slot}, {self.target_slot}) are "
                             f"not where the two prompts differ ({diff}).")
        self.prompt_ids = tuple(base)
        self.default_language_id = piece(source)
        self.eos_token_id = piece(EOS_PIECE)
        self.pad_token_id = piece(PAD_PIECE)

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        if getattr(model, "use_transf_encoder", False):
            raise ValueError("canary: this checkpoint has a transformer encoder between the FastConformer "
                             "and the decoder (`transf_encoder.num_layers > 0`), which the encoder phase "
                             "does not trace.")
        self._read_prompt(model)
        self.sample_rate = int(model.cfg.preprocessor.sample_rate)
        decoder = model.transf_decoder.decoder
        self.n_layers = len(decoder.layers)
        self.d_model = int(model.cfg.model_defaults.lm_dec_hidden)
        self.max_positions = int(model.transf_decoder.embedding.position_embedding.pos_enc.shape[0])
        self.max_generation_delta = int(model.cfg.decoding.beam.max_generation_delta)
        self.long_form = long_form_policy(self.sample_rate, int(model.encoder.subsampling_factor),
                                          float(model.cfg.preprocessor.window_stride))
        self.cross_kv_names = cross_kv_input_names(self.n_layers)

        n_samples = int(TRACE_SECONDS * self.sample_rate)
        sample_axis = ct.RangeDim(int(MIN_SECONDS * self.sample_rate), int(40 * self.sample_rate))
        token_axis = ct.RangeDim(1, self.max_positions)
        enc_axis = ct.RangeDim(1, self.max_positions)
        decoder_inputs = [
            ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
        ] + [
            ct.TensorType(name=name, shape=(1, enc_axis, self.d_model), dtype=np.float32)
            for name in self.cross_kv_names
        ]
        self.decoder_bindings = tuple((t.name, _binding_kind(t.name)) for t in decoder_inputs)

        # ORDER IS LOAD-BEARING -- see `_CanaryCrossKvWrapper`.
        cross_kv_wrapper = _CanaryCrossKvWrapper(decoder.layers).eval()
        decoder_wrapper = _CanaryDecoderWrapper(model).eval()
        trace_tokens, trace_enc = int(self.trace_tokens), int(self.trace_enc)

        return [
            ExportPhase(
                name="encoder",
                wrapper=_CanaryEncoderWrapper(model).eval(),
                dummy_inputs=(torch.randn(1, n_samples), torch.tensor([n_samples], dtype=torch.int64)),
                mil_inputs=[
                    ct.TensorType(name="waveform", shape=(1, sample_axis), dtype=np.float32),
                    ct.TensorType(name="length", shape=(1,), dtype=np.int32),
                ],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="cross_kv",
                wrapper=cross_kv_wrapper,
                dummy_inputs=(torch.zeros(1, trace_enc, self.d_model),),
                mil_inputs=[ct.TensorType(name="xa", shape=(1, enc_axis, self.d_model),
                                          dtype=np.float32)],
                root_axis="n_enc_frames",
            ),
            ExportPhase(
                name="decoder",
                wrapper=decoder_wrapper,
                dummy_inputs=(
                    torch.zeros((1, trace_tokens), dtype=torch.long),
                    torch.arange(trace_tokens).unsqueeze(0),
                    causal_mask(trace_tokens),
                ) + tuple(torch.zeros(1, trace_enc, self.d_model) for _ in self.cross_kv_names),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                declared_axes={name: {1: "n_enc_frames"} for name in self.cross_kv_names},
                topology_rewrite=lambda topo: _check_fused_attention(topo, self.n_layers),
                fuse_attention=True,
                kv_cache_size=self.max_positions,
            ),
        ]

    def hparams(self) -> dict:
        """`sample_rate`, because a host must resample to it before `waveform` means anything; `n_ctx`,
        the KV-cache capacity and the longest decode (prompt included) this file can run."""
        if not self.sample_rate:
            return {}
        return {"sample_rate": self.sample_rate, "n_ctx": self.max_positions}

    def contract(self) -> dict:
        """The ASR decode table: languages heard, languages written, and the task table.

        **`task` ids are TARGET-LANGUAGE ids**, because that is what a task is in this prompt format:
        `transcribe` is "the target is the source" (0, which the driver reads as exactly that), and
        `translate` is "the target is English" -- the meaning Whisper's `translate` has and the only one
        the canonical `task` role can carry. `<pad>` is a control id: NeMo's search ends on it as it
        does on `<|endoftext|>`, and neither is a word."""
        contract = super().contract()
        if self.lang_to_id:
            names = sorted(self.lang_to_id)
            contract["asr.language_names"] = names
            contract["asr.language_ids"] = [self.lang_to_id[n] for n in names]
            contract["asr.task_names"] = ["transcribe", "translate"]
            contract["asr.task_ids"] = [0, self.lang_to_id["en"]]
            contract["asr.control_ids"] = [self.pad_token_id]
            # Every trained language can be written as well as heard: canary-1b-v2 translates
            # English <-> each of the other 24.
            contract["asr.target_language_names"] = names
            contract["asr.target_language_ids"] = [self.lang_to_id[n] for n in names]
        for key, value in (getattr(self, "long_form", None) or {}).items():
            contract[f"asr.{key}"] = value
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        kwargs = dict(hparams=self.hparams())
        tokenizer_dir = extract_nemo_tokenizer_dir(self.checkpoint)
        if tokenizer_dir is not None:
            kwargs["tokenizer_dir"] = tokenizer_dir
            kwargs["tokenizer_family"] = "sentencepiece_proto"
        if self.eos_token_id is not None:
            # The proto declares no eos (`eos_id = -1`); the canary2 format's is `<|endoftext|>`.
            kwargs["eos_token_id"] = self.eos_token_id
        return kwargs

    def driver_components(self) -> List:
        """Encoder once, the frame count, cross-attention K/V once, the prompt, then the decode loop."""
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Len, Lit, OutputRef, Var

        waveform = FieldAccess("inputs", "waveform")
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "PROMPT": list(self.prompt_ids),
                # 1-based, for Lua.
                "SOURCE_SLOT": (self.source_slot or 0) + 1,
                "TARGET_SLOT": (self.target_slot or 0) + 1,
                "DEFAULT_LANGUAGE": self.default_language_id or 0,
                "DEFAULT_TARGET": self.lang_to_id.get(self.default_target, 0),
                "MAX_POSITIONS": self.max_positions or 0,
                "MAX_GENERATION_DELTA": self.max_generation_delta or 0,
            }),
            SubgraphCallComponent(
                topology="encoder",
                outputs=(),
                retain=True,
                inputs={"waveform": waveform, "length": FieldAccess("inputs", "length")},
                axes={"n_samples": Len(waveform), "n_past": Lit(0)},
                note="Encoder: mel front end, FastConformer, projection; cut to NeMo's encoded_len.",
            ),
            LuaFragment(
                self.driver_script_path / "01_prompt.lua",
                reads=("PROMPT", "SOURCE_SLOT", "TARGET_SLOT", "DEFAULT_LANGUAGE", "DEFAULT_TARGET",
                       "MAX_POSITIONS",
                       "MAX_GENERATION_DELTA"),
                defines=("_n_enc", "_prompt", "_max_new"),
            ),
            SubgraphCallComponent(
                topology="cross_kv",
                outputs=(),
                retain=True,
                inputs={"xa": OutputRef("encoder")},
                axes={"n_enc_frames": Var("_n_enc"), "n_past": Lit(0)},
                note="Cross-attention K/V for every decoder layer, computed once.",
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                bound={name: OutputRef("cross_kv", index=i + 1)
                       for i, name in enumerate(self.cross_kv_names)},
                extra_axes={"n_enc_frames": Var("_n_enc")},
                prompt=Var("_prompt"),
                default_max_new_tokens=self.max_positions or 16,
                default_eos_token=self.eos_token_id if self.eos_token_id is not None else -1,
                extra_eos_tokens=(self.pad_token_id,) if self.pad_token_id is not None else (),
            ),
        ]


def _is_canary(path: Path) -> bool:
    """A `.nemo` archive whose model is `EncDecMultiTaskModel` with the `canary2` prompt format. The
    first Canary (`canary-1b`) uses the older `canary` format, a different prompt, and is not claimed."""
    if not _is_nemo_archive(path):
        return False
    cfg = _read_nemo_model_config(path)
    return (str(cfg.get("target", "")).endswith("EncDecMultiTaskModel")
            and str(cfg.get("prompt_format", "")) == "canary2")


def _build_canary(path: Path, output_path: str) -> ASRCanaryExportConfig:
    return ASRCanaryExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRCanaryExportConfig,
        recognizers=[ModelRecognizer(name="canary", detect=_is_canary, build_config=_build_canary)],
    ))
