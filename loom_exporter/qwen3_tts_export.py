"""Qwen3-TTS's talker: family 10's second leaf, and the first with a SECOND autoregressive loop
inside the first.

Family 10 is an AR LM that emits codec tokens rather than text, and Dia is its first leaf. This one
shares the shape and almost none of the mechanics, because Dia emits all nine of its codebooks from
one row of one head and this emits sixteen from two models:

    talker step  ->  codebook 0          (28 layers, 1024 wide, its own KV cache, one step per frame)
    code predictor -> codebooks 1..15    (5 layers, 1024 wide, a cache RESET EVERY FRAME, 15 steps)

So one audio frame is **sixteen transformer forwards**, not one, and the driver's loop is nested. At
12.5 Hz a ten-second utterance is 125 talker steps and 1875 predictor steps.

**The prefill is ten positions regardless of how long the text is.** This is a streaming model: the
text is fed one token per generation step through `trailing_text_hidden`, added to the frame's own
embedding sum. Only the first text token is in the prompt. That is why there is no text encoder here
and no cross-attention -- family 2's shape, which Dia has, does not appear at all.

**The mrope is decorative, and that is a measured claim rather than a reading.** `config.json`
declares `rope_scaling: {interleaved: true, mrope_section: [24, 20, 20]}`, and
`apply_multimodal_rotary_pos_emb` builds its cos/sin by picking channels from three different rows of
a 3-row table. But `get_rope_index` is `attention_mask.cumsum(-1) - 1` expanded to three IDENTICAL
rows, at prefill and at every decode step alike -- there is no second modality in this model to make
them differ. Picking channels from three copies of one row gives that row back. `install_patches`
replaces the function with plain RoPE and **asserts the three rows are equal** rather than trusting
the argument.

**What it emits is `audio_codes`, not a waveform** ([ADR-020]): the codes are decoded by
`qwen3-tts-tokenizer-12hz`, exported through `audio_codec_export` as family 11's fourth leaf, and the
pair is two GGUFs by [ADR-022]'s argument -- one codec serves every size and variant of this talker.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .bpe_tokenizer_export import read_sampling_defaults
from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


QWEN3_TTS_MISSING = (
    "Qwen3-TTS is not a transformers architecture -- it ships in Alibaba's own Apache-2.0 `qwen-tts` "
    "package, which this module imports lazily so that no other model's export needs it. Install it "
    "with `pip install --no-deps qwen-tts`: its pins (`transformers==4.57.3`, `accelerate==1.12.0`) "
    "plus gradio, onnxruntime and sox would otherwise move the export venv underneath every other "
    "model in the tree. The piper venv's transformers 4.57.6 satisfies what the package imports."
)


def install_patches(modeling) -> None:
    """The class-level rewrites the talker needs to trace. Every one of them is exact, and
    three of the four are one shape read away from being unnecessary.

    **`rotate_half`, because a shape read becomes `aten::Int`.** Dia's failure verbatim
    (`dia_export.install_rotate_half_patch`, [Retro-030]): `x[..., : x.shape[-1] // 2]` traces as
    `aten::floor_divide` feeding `aten::Int`, which coremltools' `_int` handler kills with "only
    0-dimensional arrays can be converted to Python scalars". `torch.chunk` asks for a COUNT.

    **`AttentiveStatisticsPooling.forward`, because its mask is ALL ONES and only looks dynamic.**
    The pooling builds one from `lengths * seq_length` with `lengths = torch.ones(batch)` -- so the
    mask is `arange(L) < L`, true everywhere, because a single unpadded clip has nothing to filter.
    Under tracing `seq_length` is a 0-d Tensor, and the `arange`/`expand`/compare around it emits a
    RESHAPE that resolves to nonsense at any length but the traced one: the export, the write and the
    load all succeed, and the first call dies with `RESHAPE: target shape [1,1] has 1 elements but
    input has 4`. [Retro-047]'s failure again, in a third spelling.

    The replacement builds the mask by ARITHMETIC on the hidden states (`h[:, :1] * 0 + 1`), which
    carries their shape and reads none, and replaces `.repeat(1, 1, seq_length)` with `expand_as` for
    the same reason. `masked_fill(mask == 0, -inf)` is DROPPED rather than approximated: it fills
    nothing when the mask is all ones. Verified bit-identical to the checkpoint's own pooling at three
    lengths before anything was traced.

    **`apply_multimodal_rotary_pos_emb`, because the interleave is unexpressible AND unnecessary.**
    Its body is

        x_t[..., beg:end:modality] = x[beg, ..., beg:end:modality]

    -- a strided assignment into a clone, three times, which is not something a traced graph can carry.
    It is also the identity here: every caller's `position_ids` is one row expanded to three, so
    `x[0]`, `x[1]` and `x[2]` are the same tensor and picking channels between them gives `x[0]` back.
    The replacement asserts that equality on the tensors it is actually handed, so a checkpoint or a
    call path where the rows DIFFER fails loudly instead of silently exporting a model that ignores
    two thirds of its own position encoding.
    """
    import torch as _torch

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return _torch.cat((-x2, x1), dim=-1)

    def rotary_forward(self, x, position_ids):
        """`cos`/`sin` at RANK 3, from row 0 of the three identical position rows.

        The checkpoint builds them at rank 4 (`[3, batch, seq, head_dim]`) so the interleave can pick
        channels between rows. Collapsing it HERE rather than in the rope application is what makes
        the talker's attention op-for-op identical to the code predictor's -- and that is not
        cosmetic: with a `cos[0]` slice in the way, `passes.py`'s `repeat_kv` fusion matched the
        predictor's blocks and not the talker's, and the export failed with 28 blocks declaring 16
        K/V heads and 5 declaring 8, which one KvCache cannot serve.
        """
        if not _torch.jit.is_tracing():
            assert _torch.equal(position_ids[0], position_ids[1]) and \
                   _torch.equal(position_ids[0], position_ids[2]), (
                "the three mrope position rows differ, so this model's rope is NOT decorative and "
                "the interleave cannot be dropped -- see install_patches"
            )
        pos = position_ids[0]
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        freqs = (inv_freq @ pos[:, None, :].float()).transpose(1, 2)
        emb = _torch.cat((freqs, freqs), dim=-1)
        return ((emb.cos() * self.attention_scaling).to(x.dtype),
                (emb.sin() * self.attention_scaling).to(x.dtype))

    def apply_rope(q, k, cos, sin, mrope_section=None, mrope_interleaved=False, unsqueeze_dim=1):
        c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)

    def pooling_forward(self, hidden_states):
        # `mask` built by ARITHMETIC on the hidden states rather than from a length: elementwise on a
        # real tensor, so it carries that tensor's shape and reads none.
        mask = hidden_states[:, :1] * 0.0 + 1.0
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        # `expand_as` rather than `.repeat(1, 1, seq_length)`, for the same reason: the repeat count
        # was a traced shape read, and the target shape is a tensor already in hand.
        attention = _torch.cat([hidden_states,
                                mean.unsqueeze(2).expand_as(hidden_states),
                                std.unsqueeze(2).expand_as(hidden_states)], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention)))
        # The checkpoint's `masked_fill(mask == 0, -inf)` is dropped, not approximated: `mask` is all
        # ones, so it fills nothing.
        attention = _torch.nn.functional.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        return _torch.cat((mean, std), dim=1).unsqueeze(2)

    def repeat_kv(hidden_states, n_rep):
        # `repeat_interleave` on the heads axis, where HF writes `unsqueeze -> expand -> reshape`.
        #
        # That idiom's expand is a BROADCAST, and coremltools' own default pipeline folds it away
        # before `passes.fuse_gqa_repeat_kv` ever runs -- which leaves the reshape behind, alone,
        # asking for 16 heads of a tensor that still has 8. The export, the write and the load all
        # succeed; the first call dies with `RESHAPE: target shape [128,10,16,1] has 20480 elements
        # but input has 10240`. The same failure then reappears one level up as two cached phases
        # disagreeing about their K/V geometry, which is the symptom, not the cause.
        #
        # `repeat_interleave` states the repetition as a repetition rather than as a broadcast that
        # happens to be reshaped, so there is nothing for a broadcast-folding pass to remove. It is
        # the same tensor either way -- HF's own docstring says the idiom IS `torch.repeat_interleave(
        # x, dim=1, repeats=n_rep)`.
        if n_rep == 1:
            return hidden_states
        return hidden_states.repeat_interleave(n_rep, dim=1)

    # BOTH, because the two attention paths reach different copies: the checkpoint's own
    # `eager_attention_forward` uses the one in its module, and the sdpa path this export traces
    # through uses `transformers.integrations.sdpa_attention`'s.
    from transformers.integrations import sdpa_attention as _sdpa

    _sdpa.repeat_kv = repeat_kv
    modeling.repeat_kv = repeat_kv
    modeling.rotate_half = rotate_half
    modeling.apply_multimodal_rotary_pos_emb = apply_rope
    modeling.Qwen3TTSTalkerRotaryEmbedding.forward = _torch.no_grad()(rotary_forward)
    modeling.AttentiveStatisticsPooling.forward = pooling_forward



def materialise_gqa(module, config) -> None:
    """Duplicate `k_proj`/`v_proj` so the model has as many K/V heads as query heads, in place.

    **This removes `repeat_kv` from the graph rather than trying to convert it, and three attempts at
    converting it are why.** HF writes the repeat as `unsqueeze -> expand -> reshape`; coremltools'
    own pipeline folds the expand (it is a broadcast) before `passes.fuse_gqa_repeat_kv` can match the
    pair, leaving a reshape that asks for 16 heads of a tensor with 8. Rewriting it as
    `repeat_interleave` keeps a real `tile` -- and then the reshape that merges `(n_kv, n_rep)` back
    into one axis comes out as `[128, n_tokens, n_tokens, 8]`, the root axis substituted into the head
    slot, which is [Retro-044]'s failure exactly.

    With `num_key_value_heads == num_attention_heads` the repeat is the identity and never traces at
    all. What it costs is honest and small: `k_proj` and `v_proj` grow from `[1024, 1024]` to
    `[2048, 1024]`, which is +4.2 M parameters on a 917 M model, and the KV cache doubles. What it
    buys is that the one op in this architecture the converter cannot carry is simply not there.

    Interleaved, not concatenated: `repeat_kv` repeats each K/V head `n_rep` times ADJACENTLY
    (`[h0, h0, h1, h1, ...]`), which is what pairs head `2i` and `2i+1` of the query with K/V head
    `i`. A concatenated duplicate would pair them with the wrong halves and is the one way to get
    this wrong silently.
    """
    n_rep = config.num_attention_heads // config.num_key_value_heads
    if n_rep == 1:
        return
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    for layer in module.layers:
        attention = layer.self_attn
        for name in ("k_proj", "v_proj"):
            projection = getattr(attention, name)
            weight = projection.weight.data.view(config.num_key_value_heads, head_dim, -1)
            widened = weight.repeat_interleave(n_rep, dim=0).reshape(-1, weight.shape[-1])
            replacement = nn.Linear(widened.shape[1], widened.shape[0],
                                    bias=projection.bias is not None)
            with torch.no_grad():
                replacement.weight.copy_(widened)
                if projection.bias is not None:
                    bias = projection.bias.data.view(config.num_key_value_heads, head_dim)
                    replacement.bias.copy_(bias.repeat_interleave(n_rep, dim=0).reshape(-1))
            setattr(attention, name, replacement)
        attention.num_key_value_groups = 1
    config.num_key_value_heads = config.num_attention_heads


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4-D additive causal mask, the form `create_causal_mask` passes straight through.

    The same tensor and the same reason as `dia_export.causal_mask` and `causal_lm_export._causal_mask`:
    an already-prepared 4-D mask short-circuits the internal mask builder entirely, so it never derives
    a key length from a Python-level shape that tracing would bake in.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def merged_predictor_tables(code_predictor) -> tuple:
    """The code predictor's 15 embeddings and 15 heads, each CONCATENATED into one tensor.

    **This is the decision that keeps the predictor to one topology instead of fifteen.** Its
    `forward` selects a module by step -- `get_input_embeddings()[steps - 1]` on the way in and
    `lm_head[steps]` on the way out -- which no single traced graph can express, because the step is
    data the driver holds rather than a shape.

    Stacked, both selections become index arithmetic the DRIVER does, which is where a model constant
    belongs ([ADR-006], and [ADR-020]'s reasoning about Dia's delay pattern):

    * the head is `[15 * 2048, 1024]`, so the logits for group `g` are the window
      `[g * 2048, (g + 1) * 2048)` -- exactly what `loom.sample_row`'s `lo`/`hi` restricts a draw to,
      which family 10 added for Dia and which this reuses unchanged.
    * the embedding is `[15 * 2048, 1024]`, so group `g`'s token `t` is row `g * 2048 + t` -- and
      since a windowed draw returns the ABSOLUTE index, `sample_row`'s answer IS that row. The offset
      never has to be added or removed anywhere.

    It costs nothing in weights: the same 15 tensors, written once each, contiguously.
    """
    embeddings = torch.cat([e.weight for e in code_predictor.get_input_embeddings()], dim=0)
    heads = torch.cat([h.weight for h in code_predictor.lm_head], dim=0)
    return embeddings, heads


class _SpeakerEncoderWrapper(nn.Module):
    """`waveform -> x-vector`, mel front end included.

    The whole speaker-conditioning path in one phase, because its two halves have no other caller: the
    128-bin mel at 24 kHz is built for this encoder alone, and the ECAPA stack reads nothing else.

    **The reference computes the mel with `librosa`'s filter bank inside `extract_speaker_embedding`,
    not with a `transformers` feature extractor**, so the constants here are read off that call rather
    than off a preprocessor config: `n_fft=1024, hop=256, win=1024, 128 mels, fmin=0, fmax=12000`.
    """

    def __init__(self, speaker_encoder, mel_fn):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.mel_fn = mel_fn

    def forward(self, waveform):
        mels = self.mel_fn(waveform).transpose(1, 2)
        return self.speaker_encoder(mels)[0].view(1, -1)


class _PrefillEmbedWrapper(nn.Module):
    """`(text token ids, language id, x-vector) -> (the prompt's embeddings, the text schedule)`.

    **This phase exists because the prompt is ten positions long however long the text is.** The
    talker is a streaming model: only the first text token is in the prompt, and every later one is
    added to a generated frame's own embedding one step at a time. So there are two outputs, not one,
    and the second is a SCHEDULE rather than a sequence.

    What it reproduces is the block in `Qwen3TTSForConditionalGeneration.generate` that builds
    `talker_input_embed`, and every constant in it is the checkpoint's:

        rows 0..2   the rendered `<|im_start|>assistant\\n`, projected text embeddings
        rows 3..8   five `tts_pad` and one `tts_bos`, PLUS the codec control run
                    (think, think_bos, <language>, think_eos, the x-vector, codec_pad)
        row 9       the first text token, plus `codec_bos`

    The x-vector sits in the middle of the codec run as if it were a codec embedding, which is the
    whole of how this model is conditioned on a voice -- there is no speaker table and no adapter.

    **`language_id` is an INPUT rather than a constant**, so one GGUF serves all ten languages. The
    control ids around it are constants, concatenated in-graph.

    The second output is `trailing_text_hidden` with `tts_pad` appended as its last row. The reference
    branches -- `trailing[step]` while `step < len`, `tts_pad_embed` after -- and appending the pad
    turns that branch into `min(step, len)`, an index the driver computes instead of a condition the
    graph cannot carry.
    """

    def __init__(self, talker, config):
        super().__init__()
        self.talker = talker
        self.register_buffer("specials", torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]]),
            persistent=False)
        talker_config = config.talker_config
        self.register_buffer("control_head", torch.tensor(
            [[talker_config.codec_think_id, talker_config.codec_think_bos_id]]), persistent=False)
        self.register_buffer("control_tail", torch.tensor(
            [[talker_config.codec_think_eos_id]]), persistent=False)
        self.register_buffer("codec_tail", torch.tensor(
            [[talker_config.codec_pad_id, talker_config.codec_bos_id]]), persistent=False)

    def _text(self, ids):
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids))

    def forward(self, text_ids, language_id, x_vector):
        codec_embedding = self.talker.get_input_embeddings()
        bos_e, eos_e, pad_e = self._text(self.specials).chunk(3, dim=1)
        control = torch.cat([self.control_head, language_id.view(1, 1), self.control_tail], dim=1)
        codec_in = torch.cat([codec_embedding(control), x_vector.view(1, 1, -1),
                              codec_embedding(self.codec_tail)], dim=1)
        body = torch.cat([pad_e.expand(-1, codec_in.shape[1] - 2, -1), bos_e], dim=1) + codec_in[:, :-1]
        text = self._text(text_ids)
        prefill = torch.cat([text[:, :3], body, text[:, 3:4] + codec_in[:, -1:]], dim=1)
        # `[:, 4:-5]` is the text after its first token and before the closing
        # `<|im_end|>\n<|im_start|>assistant\n`, which is five tokens in this checkpoint's template.
        # Both bounds are counted from an END, so neither bakes the traced length in.
        schedule = torch.cat([self._text(text_ids[:, 4:-5]), eos_e, pad_e], dim=1)
        return prefill, schedule


class _TalkerWrapper(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> (logits over the DRAWABLE ids, last hidden)`.

    The 28-layer decoder, KV-cached, one step per audio frame. Two outputs because both are read: the
    logits pick codebook 0, and the hidden state is what the code predictor is conditioned on.

    **Logits first, because `loom.sample_row` reduces a module's output 0.** The hidden state is
    output 1 and the driver reaches it by index, as Dia's decoder reaches its cross-attention K/V.

    **The head is TRIMMED to the ids a draw may return, and that is what removes this model's
    `suppress_tokens` from the driver entirely.** The checkpoint bans `[vocab - 1024, vocab)` except
    `codec_eos`, leaving `[0, 2048) + {2150}` -- a set with a hole in it, which `sample_row`'s
    `lo`/`hi` window cannot express and which would otherwise have needed a new engine option. So the
    export writes a head of 2049 rows: the 2048 real codes, then `codec_eos`. The window is the whole
    row, the hole is gone, and the driver's only mapping is that index 2048 means stop.

    It also makes `min_new_tokens = 2` expressible with the window that already exists: `hi = 2048`
    for the first two steps bans EOS and nothing else.

    The INPUT embedding is untouched at 3072 rows -- the control ids the prompt is built from live up
    there, and they are read, never written.
    """

    def __init__(self, talker, eos_token_id: int, n_codes: int):
        super().__init__()
        self.model = talker.model
        head = talker.codec_head
        trimmed = nn.Linear(head.in_features, n_codes + 1, bias=False)
        with torch.no_grad():
            trimmed.weight.copy_(torch.cat(
                [head.weight[:n_codes], head.weight[eos_token_id:eos_token_id + 1]], dim=0))
        self.codec_head = trimmed

    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                            attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = hidden[:, -1:]
        return self.codec_head(last), last


class _PredictorWrapper(nn.Module):
    """`(codebook-0 id, talker hidden[, the rows drawn so far]) -> logits over all fifteen heads`.

    **The code predictor is NOT KV-cached, and that is a decision rather than an omission.** One
    KvCache is allocated per model with one per-layer width, so two cached phases must agree on their
    K/V geometry -- and these two do not: the talker's 28 fused ATTENTION blocks report 16 heads where
    this one's 5 report 8, because the talker's `repeat_kv` survives into its topology and this one's
    does not. Rather than force them to agree, this phase stops needing the cache.

    It can afford to. The predictor never sees more than 16 positions, so re-running its prefix costs
    `2 + 3 + ... + 16 = 135` row-forwards of a FIVE-layer, 1024-wide stack per audio frame, against a
    cached 16. That is a fraction of the one 28-layer talker step beside it, and it buys a phase whose
    cost is bounded and whose cache cannot disagree with anything.

    **The prefix is rebuilt IN-GRAPH from ids**, which is what keeps the driver's per-frame traffic to
    integers even without a cache: `cat(talker_hidden, codec_embedding(first), merged_table(rows))`.
    The driver hands the same `rows` array it has been accumulating, one longer each step, and the
    numbers in it are the absolute merged-table rows a windowed draw already returned.

    Two topologies rather than one, and only because step 0 has no rows yet: an empty axis is not a
    shape MIL will carry. They share every weight, which the writer aliases.
    """

    def __init__(self, code_predictor, talker, heads, with_rows: bool, embeddings):
        super().__init__()
        self.model = code_predictor.model
        self.codec_embedding = talker.get_input_embeddings()
        self.projection = code_predictor.small_to_mtp_projection
        self.with_rows = with_rows
        if with_rows:
            self.row_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            with torch.no_grad():
                self.row_embedding.weight.copy_(embeddings)
        merged = nn.Linear(heads.shape[1], heads.shape[0], bias=False)
        with torch.no_grad():
            merged.weight.copy_(heads)
        self.lm_head = merged

    def forward(self, first_id, talker_hidden, position_ids, attention_mask, rows=None):
        prefix = [talker_hidden, self.codec_embedding(first_id)]
        if self.with_rows:
            prefix.append(self.row_embedding(rows))
        embeds = self.projection(torch.cat(prefix, dim=1))
        hidden = self.model(inputs_embeds=embeds, position_ids=position_ids,
                            attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.lm_head(hidden[:, -1:]).view(1, -1)


class _FrameEmbedWrapper(nn.Module):
    """`(a frame's sixteen ids, a schedule row, the schedule) -> the talker's next input embedding`.

    One audio frame's contribution to the talker is the SUM of its sixteen codebook embeddings, plus
    one row of the text schedule. Doing that in Lua would be sixteen gathers and sixteen 1024-wide
    adds per frame, marshalled both ways -- which is the cost [ADR-031] exists to remove, and at 12.5
    Hz it is per 80 ms of audio.

    `codes[0]` indexes the TALKER's own 3072-row codec embedding; `codes[1:]` index the merged
    15-table and are already absolute rows, because that is what a windowed draw returned. So the two
    gathers need no offsets and the driver passes the fifteen numbers it was handed, unchanged.

    The schedule arrives as a reference to `prefill_embed`'s second output and is indexed here, so the
    text hidden never crosses the boundary either -- the driver supplies a row NUMBER.
    """

    def __init__(self, talker, embeddings):
        super().__init__()
        self.codec_embedding = talker.get_input_embeddings()
        self.predictor_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        with torch.no_grad():
            self.predictor_embedding.weight.copy_(embeddings)

    def forward(self, codes, schedule_row, schedule):
        summed = (self.codec_embedding(codes[:, :1]).sum(dim=1)
                  + self.predictor_embedding(codes[:, 1:]).sum(dim=1))
        return summed.unsqueeze(1) + self.embed_schedule(schedule, schedule_row)

    @staticmethod
    def embed_schedule(schedule, schedule_row):
        # `index_select` over the row axis rather than a slice: the index is a graph INPUT, and a
        # slice bound derived from one is exactly the shape-algebra hole Retro-047 records.
        return torch.index_select(schedule, 1, schedule_row.view(-1))


@dataclass
class TextToCodesQwen3TTSExportConfig(BaseMultiPhaseModelExportConfig):
    """Qwen3-TTS's talker as seven traced phases plus a nested decode loop.

    Seven is more than any other family, and the reason is that this model's generation step is not
    one forward pass. Grouped by what runs when:

        once, per utterance    speaker_encoder     waveform -> x-vector
                               prefill_embed       ids + language + x-vector -> prompt, text schedule
        once per frame         talker              28 layers, cached -> logits + hidden
                               predictor_prompt    (codebook-0 id, hidden) -> the predictor's prompt
                               frame_embed         16 ids + a schedule row -> the next talker input
        fifteen times a frame  predictor           5 layers, cached, RESET each frame
                               predictor_step      the merged table's row -> that row

    **Nothing but integers crosses the Lua boundary inside the loop**, which is [ADR-031]'s rule
    applied to a model that would otherwise pay it 16 times per 80 ms of audio: every embedding, hidden
    state and logit row stays in the engine, referenced by module and index the way Dia's
    cross-attention K/V are. The driver's per-frame traffic is 16 ids out and 2 ids in.

    The two `lo`/`hi` uses are the whole of the step bookkeeping, and both were already in the engine:
    the merged head makes group `g` a window, and the trimmed talker head makes `min_new_tokens` one.
    What was NOT already there is the repetition penalty -- see `hparams`.
    """

    model_dir: str = ""
    architecture: str = "qwen3_tts"
    output_path: str = "qwen3_tts_mil.gguf"
    # Both cached phases count the same thing the cache is indexed by, so both are `n_tokens` for the
    # mechanical reason `dia_export` records: `GraphBuilder` reads `n_tokens` out of `SymbolEnv` to
    # size the cache's cell index, and a cached phase naming its axis anything else builds a graph
    # whose cache addresses an axis nothing bound. The honest name is in `contract()`.
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "qwen3_tts_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # The concrete lengths the trace runs at -- free, and deliberately not 1, so coremltools has a real
    # axis to make dynamic rather than a size-1 one it may fold away.
    trace_text_len: int = 24
    trace_samples: int = 24000
    max_text_len: int = 2048
    # 1024 frames is 82 s of audio at 12.5 Hz, and the number is a MEMORY decision rather than a
    # generous default: one KvCache is allocated for the whole model, so this sizes the talker's 28
    # layers AND the code predictor's 5 -- 33 layers x 1024 cells x 1024 wide x K and V x 4 bytes,
    # about 277 MB at F32. At 4096 it would be 1.1 GB, for utterances no TTS caller asks for.
    max_frames: int = 1024
    max_ref_seconds: int = 30

    _n_code_groups: Optional[int] = None
    _codebook_size: Optional[int] = None
    _eos_index: Optional[int] = None
    _language_ids: Optional[dict] = None
    _sample_rate: Optional[int] = None

    __unchecked__ = {
        "model_dir": Unchecked("path to the HF directory; `load_model` raises on anything it cannot "
                               "load, and the recognizer already read its config.json"),
        "root_axis": Unchecked("`n_tokens` is forced for a KV-cached phase -- see the field comment"),
        "trace_text_len": Unchecked("the concrete length torch.jit.trace runs at; the dynamic range "
                                    "is declared separately, so this constrains nothing"),
        "trace_samples": Unchecked("same, for the speaker encoder's waveform"),
        "max_text_len": Unchecked("the ct.RangeDim upper bound on the prompt's token count"),
        "max_frames": Unchecked("the ct.RangeDim upper bound on the talker's cached axis"),
        "max_ref_seconds": Unchecked(
            "how long a reference clip the speaker-encoder graph accepts. A ceiling on the EXPORT, "
            "not a claim about the checkpoint -- an ECAPA stack pools over time and is length-agnostic "
            "-- and 30 s is a generous reading of a model whose own card advertises 3-second cloning."
        ),
        "_n_code_groups": Unchecked("read off talker_config.num_code_groups by load_model"),
        "_codebook_size": Unchecked("read off code_predictor_config.vocab_size by load_model"),
        "_eos_index": Unchecked(
            "where `codec_eos` lands in the TRIMMED head, which is `codebook_size` by construction -- "
            "_TalkerWrapper writes the 2048 real codes and then that one row. Derived from the shape "
            "the export itself builds, so there is no second authority to check it against."
        ),
        "_language_ids": Unchecked("read off talker_config.codec_language_id by load_model"),
        "_sample_rate": Unchecked("read off speaker_encoder_config.sample_rate by load_model"),
    }

    def load_model(self):
        try:
            import qwen_tts.core.models.modeling_qwen3_tts as modeling
            from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
        except ImportError as exc:                       # pragma: no cover - env-dependent
            raise ImportError(QWEN3_TTS_MISSING) from exc

        install_patches(modeling)
        print(f"Loading Qwen3-TTS talker from {self.model_dir}...")
        # **`sdpa`, and it is not a performance choice -- it is what the ATTENTION FUSION matches.**
        # `passes.py`'s window is anchored on a softmax preceded by `mul(q, 1/sqrt(head_dim))`: the
        # scale folded onto Q, which is what the sdpa path emits. HF's EAGER path scales the SCORES
        # instead (`matmul(q, k^T) * scaling`), so the window never matches and the block stays raw --
        # and then `repeat_kv`'s expand-then-reshape survives into the topology, where MIL has folded
        # the expand away and the reshape asks for 16 heads of a tensor that has 8. The export, the
        # write and the load all succeed; the first call dies with `RESHAPE: target shape
        # [128,10,16,1] has 20480 elements but input has 10240`.
        #
        # A prepared 4-D mask reaches the sdpa path as `attn_mask` just as it reaches the eager one,
        # so nothing about the masking changes.
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float32, attn_implementation="sdpa",
            local_files_only=True).eval()
        if model.speaker_encoder is None:
            raise NotImplementedError(
                f"this checkpoint declares tts_model_type={model.config.tts_model_type!r} and so has "
                "no speaker encoder. Only the `base` (voice-clone) checkpoints export through this "
                "module; an instruct/custom-voice one conditions on a speaker TABLE and is a "
                "different prompt, not a longer one."
            )
        talker_config = model.config.talker_config
        materialise_gqa(model.talker.model, talker_config)
        materialise_gqa(model.talker.code_predictor.model, talker_config.code_predictor_config)
        self._n_code_groups = int(talker_config.num_code_groups)
        self._codebook_size = int(talker_config.code_predictor_config.vocab_size)
        self._eos_index = self._codebook_size
        self._language_ids = {k: int(v) for k, v in talker_config.codec_language_id.items()}
        self._sample_rate = int(model.config.speaker_encoder_config.sample_rate)
        self._model = model
        return model

    def hparams(self) -> dict:
        """What a host needs to call this driver, or to read what comes back.

        `codec.n_codebooks` and `codec.codebook_size` are deliberately spelled the way
        `audio_codec_export` spells them: they are the same two numbers seen from the two ends of one
        pipeline, and a host piping this into `qwen3-tts-tokenizer-12hz` compares them.

        **`sampling.repetition_penalty` is not decoration, and this model is the proof.** The
        checkpoint declares 1.05, `transformers` applies it as a PROCESSOR rather than a warper -- so
        it runs under greedy decoding too -- and a greedy decode without it never emits EOS: it runs
        to the token cap, 200 frames where the reference stops at 42. It is why `loom.sample_row`
        gained `repetition_penalty`/`penalized`.
        """
        if self._n_code_groups is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        hparams = {
            "codec.n_codebooks": self._n_code_groups,
            "codec.codebook_size": self._codebook_size,
            "n_text_ctx": self.max_text_len,
            "n_codes_ctx": self.max_frames,
            "sample_rate": self._sample_rate,
        }
        hparams.update({f"sampling.{k}": v for k, v in read_sampling_defaults(self.model_dir).items()})
        return hparams

    def contract(self) -> dict:
        """`text -> audio_codes`, by [ADR-020]: what comes back is what a codec decodes, not a waveform.

        `text.frontend = "vocab"` because this checkpoint ships a real HF tokenizer and needs nothing
        else -- which is the sentence `Epic-07` already wrote about this model while it was unstarted.
        """
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        return contract

    def backend_kwargs(self) -> dict:
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams())

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        talker = model.talker
        predictor = talker.code_predictor
        hidden_size = int(talker.config.hidden_size)
        embeddings, heads = merged_predictor_tables(predictor)
        groups, codebook = self._n_code_groups, self._codebook_size

        mel_fn = _mel_frontend(model)
        text_dim = ct.RangeDim(10, self.max_text_len)
        frames_dim = ct.RangeDim(1, self.max_frames)
        steps_dim = ct.RangeDim(3, groups)
        rows_dim = ct.RangeDim(1, groups - 2)

        probe_text = torch.zeros((1, self.trace_text_len), dtype=torch.long)
        probe_spk = torch.zeros((1, hidden_size))
        probe_lang = torch.tensor([next(iter(self._language_ids.values()))])

        return [
            ExportPhase(
                name="speaker_encoder",
                wrapper=_SpeakerEncoderWrapper(model.speaker_encoder, mel_fn).eval(),
                dummy_inputs=(torch.zeros((1, self.trace_samples)),),
                mil_inputs=[ct.TensorType(
                    name="waveform",
                    shape=(1, ct.RangeDim(self._sample_rate // 4,
                                          self._sample_rate * self.max_ref_seconds)),
                    dtype=np.float32)],
                root_axis="n_samples",
            ),
            ExportPhase(
                name="prefill_embed",
                wrapper=_PrefillEmbedWrapper(talker, model.config).eval(),
                dummy_inputs=(probe_text, probe_lang, probe_spk),
                mil_inputs=[
                    ct.TensorType(name="text_ids", shape=(1, text_dim), dtype=np.int32),
                    ct.TensorType(name="language_id", shape=(1,), dtype=np.int32),
                    ct.TensorType(name="x_vector", shape=(1, hidden_size), dtype=np.float32),
                ],
            ),
            ExportPhase(
                name="talker",
                wrapper=_TalkerWrapper(talker, int(talker.config.codec_eos_token_id),
                                       codebook).eval(),
                # **Seven, not eight, and the number matters.** `repeat_kv` reshapes through
                # `[1, n_kv, n_rep, seq, head_dim]`, and this model has 8 K/V heads -- so a trace at
                # length 8 makes the sequence axis and the head axis the same number, and the GQA
                # fusion cannot tell them apart. Unfused, `repeat_kv` survives into the topology and
                # the ATTENTION node reports 16 K/V heads where the code predictor's five report 8,
                # which one KvCache cannot serve.
                dummy_inputs=(torch.zeros(1, 7, hidden_size),
                              torch.arange(7).view(1, -1),
                              causal_mask(7)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, frames_dim, hidden_size),
                                  dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, frames_dim), dtype=np.int32),
                    # The SAME `RangeDim` instance on both mask axes, which is what every cached
                    # decoder in this tree declares: two instances are two independent symbols, and
                    # `_validate_input_axes` refuses a topology with two roots. The mask a cached step
                    # is actually handed is `1 x (n_past + 1)`; the fused ATTENTION node resolves the
                    # key extent from `n_past`, not from this declaration.
                    ct.TensorType(name="attention_mask",
                                  shape=(1, 1, frames_dim, frames_dim),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=self.max_frames,
            ),
            ExportPhase(
                name="predictor_prefill",
                wrapper=_PredictorWrapper(predictor, talker, heads, False, embeddings).eval(),
                dummy_inputs=(torch.zeros((1, 1), dtype=torch.long),
                              torch.zeros(1, 1, hidden_size),
                              torch.arange(2).view(1, -1),
                              causal_mask(2)),
                mil_inputs=[
                    ct.TensorType(name="first_id", shape=(1, 1), dtype=np.int32),
                    ct.TensorType(name="talker_hidden", shape=(1, 1, hidden_size),
                                  dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, 2), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, 2, 2), dtype=np.float32),
                ],
            ),
            ExportPhase(
                name="predictor_steps",
                wrapper=_PredictorWrapper(predictor, talker, heads, True, embeddings).eval(),
                dummy_inputs=(torch.zeros((1, 1), dtype=torch.long),
                              torch.zeros(1, 1, hidden_size),
                              torch.arange(5).view(1, -1),
                              causal_mask(5),
                              torch.zeros((1, 3), dtype=torch.long)),
                mil_inputs=[
                    ct.TensorType(name="first_id", shape=(1, 1), dtype=np.int32),
                    ct.TensorType(name="talker_hidden", shape=(1, 1, hidden_size),
                                  dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, steps_dim), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, steps_dim, steps_dim),
                                  dtype=np.float32),
                    # `rows` is two shorter than the prefix it becomes: the talker hidden and
                    # codebook 0 are the other two positions.
                    ct.TensorType(name="rows", shape=(1, rows_dim), dtype=np.int32),
                ],
                declared_axes={"rows": {1: "n_tokens - 2"}},
            ),
            ExportPhase(
                name="frame_embed",
                wrapper=_FrameEmbedWrapper(talker, embeddings).eval(),
                dummy_inputs=(torch.zeros((1, groups), dtype=torch.long),
                              torch.zeros((1,), dtype=torch.long),
                              torch.zeros(1, 4, hidden_size)),
                mil_inputs=[
                    ct.TensorType(name="codes", shape=(1, groups), dtype=np.int32),
                    ct.TensorType(name="schedule_row", shape=(1,), dtype=np.int32),
                    ct.TensorType(name="schedule",
                                  shape=(1, ct.RangeDim(1, self.max_text_len), hidden_size),
                                  dtype=np.float32),
                ],
            ),
        ]


    def driver_components(self) -> List:
        """Two constants blocks' worth of checkpoint facts, and one hand-written loop.

        A `LuaFragment` rather than `PrefillDecodeLoop`, and for a stronger version of Dia's reason:
        that component reduces one row to one token and feeds the token back, where every step here
        runs a SECOND cached model fifteen times, resets its cache, and reassembles sixteen ids into
        one embedding. There is no part of it that the shared loop expresses. Its own
        `run_subgraph_and_retain` call sites are parsed out of the text and declared against the real
        traced topologies regardless, which is what `LuaFragment` is for.
        """
        from .driver_components import DriverReturn, ExportConstants, LuaFragment

        sampling = read_sampling_defaults(self.model_dir) if self.model_dir else {}
        # `.get` throughout, for the reason Dia's block records: `component_registry.usage()` builds
        # every registered config with no checkpoint in hand, and `phases()` is what fills these in.
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "N_GROUPS": self._n_code_groups or 0,
                "CODEBOOK_SIZE": self._codebook_size or 0,
                # The trimmed head's last row. Both a stop condition and the end of the draw window,
                # which is the point of trimming -- see `_TalkerWrapper`.
                "EOS_INDEX": self._eos_index or 0,
                # Ten, and it does not depend on the text: only the first text token is in the prompt.
                "PREFILL_LEN": PREFILL_LEN,
                # `transformers` installs `MinNewTokensLengthLogitsProcessor` from the talker's own
                # `min_new_tokens: 2`, and it runs under greedy like every other processor.
                "MIN_NEW_TOKENS": MIN_NEW_TOKENS,
                "MAX_NEW_TOKENS": int(sampling.get("max_new_tokens", 4096)),
                "DEFAULT_LANGUAGE_ID": int((self._language_ids or {}).get("english", 0)),
                # The checkpoint's own decoding defaults as the driver's `or`-fallbacks -- the same
                # numbers `hparams()` writes for the host, rendered twice from one attribute set.
                "TEMPERATURE": sampling.get("temperature", 0.0),
                "TOP_K": sampling.get("top_k", 0),
                "TOP_P": sampling.get("top_p", 1.0),
                "REPETITION_PENALTY": sampling.get("repetition_penalty", 1.0),
                # The code predictor draws with its own three, which `generation_config.json` states
                # separately under `subtalker_*`. They are genuinely different knobs on the same file.
                "SUB_TEMPERATURE": _subtalker(self.model_dir, "temperature", 0.0),
                "SUB_TOP_K": _subtalker(self.model_dir, "top_k", 0),
                "SUB_TOP_P": _subtalker(self.model_dir, "top_p", 1.0),
            }),
            LuaFragment(
                self.driver_script_path / "01_generate.lua",
                reads=("N_GROUPS", "CODEBOOK_SIZE", "EOS_INDEX", "PREFILL_LEN", "MIN_NEW_TOKENS",
                       "MAX_NEW_TOKENS", "DEFAULT_LANGUAGE_ID", "TEMPERATURE", "TOP_K", "TOP_P",
                       "REPETITION_PENALTY", "SUB_TEMPERATURE", "SUB_TOP_K", "SUB_TOP_P"),
                defines=("_codes",),
            ),
            DriverReturn(values=("_codes",)),
        ]


# The prompt's length, which is a property of this model's STREAMING design rather than of any text:
# three template tokens, five `tts_pad` plus `tts_bos` over the codec control run, and the first text
# token. Everything after it arrives one token per generated frame.
PREFILL_LEN = 10
# `generation_config.json` does not state it; `Qwen3TTSForConditionalGeneration.generate` passes
# `min_new_tokens: 2` in `talker_kwargs`, so the authority is that call site.
MIN_NEW_TOKENS = 2


def _subtalker(model_dir: str, key: str, fallback):
    """One `subtalker_*` decoding default off `generation_config.json`.

    Not `read_sampling_defaults`, which reads the standard names: this checkpoint declares a SECOND
    set of three for the code predictor, and they are different numbers for a different draw.
    """
    if not model_dir:
        return fallback
    path = Path(model_dir) / "generation_config.json"
    if not path.exists():
        return fallback
    try:
        config = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return fallback
    if not config.get("subtalker_dosample", False) and key == "temperature":
        return 0.0                                   # the engine's spelling of greedy
    return config.get(f"subtalker_{key}", fallback)


def _is_qwen3_tts(path: Path) -> bool:
    """An HF directory declaring `model_type == "qwen3_tts"` -- the TALKER, not its codec.

    The codec is the `speech_tokenizer/` subfolder and declares `qwen3_tts_tokenizer_12hz`;
    `audio_codec_export` owns that recognizer. Two model types, two exports, two GGUFs.
    """
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "qwen3_tts"


def _build_qwen3_tts(path: Path, output_path: str) -> TextToCodesQwen3TTSExportConfig:
    return TextToCodesQwen3TTSExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-codes",
        config_class=TextToCodesQwen3TTSExportConfig,
        recognizers=[ModelRecognizer(name="qwen3-tts", detect=_is_qwen3_tts,
                                     build_config=_build_qwen3_tts)],
    ))


def _mel_frontend(model):
    """The 128-bin mel the speaker encoder is fed, as a module the wrapper can call.

    `extract_speaker_embedding` computes it inline with `librosa`'s filter bank rather than through a
    feature extractor, so these constants are read off that call site and not off a config -- there is
    no preprocessor file in this checkpoint that states them.
    """
    import qwen_tts.core.models.modeling_qwen3_tts as modeling

    def mel(waveform):
        return modeling.mel_spectrogram(waveform, n_fft=1024, num_mels=128, sampling_rate=24000,
                                        hop_size=256, win_size=1024, fmin=0, fmax=12000)

    return mel
