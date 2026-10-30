"""Audio8-TTS (`arktts`, Edge0's Apache-2.0 DualAR): family 10's fifth leaf, text in and 10-codebook
`audio_codes` out at 21.5 frames a second.

The codes are what the checkpoint's own `codec.pth` decodes (`audio_codec_export`, family 11's
`ARKTTS` member) -- two GGUFs by [ADR-022], and the codec is this checkpoint's `companions()`. The
shape is Fish Audio S2's DualAR, which is Qwen3-TTS's shape with the sizes moved:

    slow stack  ->  one SEMANTIC token per frame      (24 layers, 896 wide, GQA 14/2, KV-cached)
    fast stack  ->  the frame's other nine codebooks   (4 layers, 896 wide, re-run per codebook)

Codebook 0 IS the semantic token, offset into the vocabulary at `semantic_begin_id`, so the fast stack
draws nine ids, not ten.

**The RoPE is INTERLEAVED and its table is BF16.** `_precompute_rope` builds `cos`/`sin` in f32 and
casts the table to bfloat16, and the attention multiplies by those rounded numbers at whatever dtype
the model runs in. That rounding is about 2^-9 of every rotation, which is not noise a greedy decode
survives, so the export reads the table off the module rather than recomputing it: `cos`/`sin` are
two gathered constant tables, indexed by `position_ids`, holding exactly the reference's values.

The interleave itself is a ROW PERMUTATION. Rotating pairs `(2j, 2j+1)` of q and k is rotate-half
rotation of `q[perm]` and `k[perm]` with `perm = [0, 2, ..., 62, 1, 3, ..., 63]`, and `q . k` is
invariant under applying one permutation to both. So each head's q and k rows of `wqkv` (and its
bias) are permuted at export time and the graph runs plain rotate-half, which is what `chunk` traces
(Dia's patch, [Retro-030]). V and the output projection are untouched.

**The head is trimmed to the ids a draw may return.** `ArkttsSemanticLogitsProcessor` keeps the 4096
semantic ids and `<|im_end|>`, so the export writes those 4097 rows of the tied 155776-row table,
EOS LAST -- `qwen3_tts_export._TalkerWrapper`'s trick, and here it removes a processor that is a hole
in the vocabulary rather than a window.

**The fast stack is not cached**, for Qwen3-TTS's reason: one KvCache per model, and it never sees
more than ten positions. Its input is the slow stack's NORMED last hidden state (`norm_fastlayer_input`)
followed by the codes drawn so far through ONE shared 4096-row table, and its head is one shared
4096-row linear -- there is no per-codebook offset anywhere in this model, so the driver hands back
exactly the ids it drew.

**The input embedding is the token embedding plus ten codebook embeddings, on semantic rows only.**
The reference masks the codebook sum with `where(semantic, sum, 0)`; here each codebook's block of the
table gains a zero row at `codebook_size` and every text row carries that code -- MOSS-TTS's absent-id
trick (`moss_tts_export._EmbedWrapper`), and exact, since the reference adds the same 0.0.

**Voice cloning takes the reference's CODES.** The clone prompt puts the reference transcript and its
frames in the system turn; the codes come from the codec's encoder, which is not exported:
`audio8_tts_voices` runs it once per voice and writes a voice file (loom.cpp ADR-045).

**The sampler is the reference's "legacy" one, which is not `transformers`'**: top-k and top-p are
measured on the UNTEMPERED row, top-p over the whole row and EXCLUDING the id that crosses it, and only
the survivors are divided by the temperature. A semantic draw that repeats one of the last ten is
redrawn at the checkpoint's `ras_*` settings. See the driver and loom.cpp ADR-095.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bpe_tokenizer_export import read_sampling_defaults
from .decomposition import Decomposition, MultiPhase
from .export_config import CompanionExport
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked

MODEL_TYPE = "arktts"

# The processor's template pieces (`processing_arktts.ArkttsProcessor._prompt_segments`), verbatim.
# Each is encoded on its own and the ids concatenated, which is what the processor does -- not the same
# as encoding the rendered string.
NO_REFERENCE_HEAD = ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                     "<|im_start|>user\n"]
REFERENCE_HEAD = ["<|im_start|>system\n",
                  "convert the provided text to speech reference to the following:\n\nText:\n"]
REFERENCE_SPEECH = ["\n\nSpeech:\n"]
REFERENCE_TO_TEXT = ["<|im_end|>\n", "<|im_start|>user\n"]
TAIL = ["<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]


def prompt_segments(tokenizer) -> dict:
    """The ids the driver wraps the caller's ids in, encoded with the checkpoint's own tokenizer. The
    reference transcript between `reference_head` and `reference_speech` is a voice file's own
    `reference_tokens`, speaker prefix included (`audio8_tts_voices` says why it is tokenized there)."""
    def enc(parts):
        out = []
        for part in parts:
            out.extend(int(i) for i in tokenizer.encode(part, add_special_tokens=False))
        return out

    return dict(no_reference_head=enc(NO_REFERENCE_HEAD), reference_head=enc(REFERENCE_HEAD),
                reference_speech=enc(REFERENCE_SPEECH), reference_to_text=enc(REFERENCE_TO_TEXT),
                tail=enc(TAIL))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(freqs: torch.Tensor):
    """`[L, head_dim/2, 2]` (the reference's bf16 `freqs_cis`) -> rotate-half `cos`, `sin`, `[L, head_dim]`."""
    f = freqs.detach().float()
    return torch.cat([f[..., 0], f[..., 0]], -1), torch.cat([f[..., 1], f[..., 1]], -1)


def _deinterleave_rows(linear: nn.Linear, n_q: int, n_kv: int, head_dim: int) -> nn.Linear:
    """`wqkv` with every q and k head's rows reordered evens-then-odds -- see the module docstring.
    V's rows are untouched. Still one Linear: the uneven `(896, 128, 128)` split after it is what
    `topology_ops._op_split` learned `split_sizes` for."""
    perm = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    rows = [h * head_dim + perm for h in range(n_q + n_kv)]
    rows.append(torch.arange((n_q + n_kv) * head_dim, (n_q + 2 * n_kv) * head_dim))
    rows = torch.cat(rows)
    out = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        out.weight.copy_(linear.weight.detach()[rows])
        if linear.bias is not None:
            out.bias.copy_(linear.bias.detach()[rows])
    return out


class _Attention(nn.Module):
    """`ArkttsAttention` re-spelled for the trace: de-interleaved rows and rotate-half RoPE from the
    gathered tables, `repeat_kv`'s expand-and-merge (which `passes.fuse_gqa_repeat_kv` undoes, so the
    cache holds the checkpoint's 2 K/V heads), and SDPA with an additive mask -- the form the attention
    fusion anchors on. The reference's `repeat_interleave` is the same head order."""

    def __init__(self, attention):
        super().__init__()
        if attention.qk_norm:
            raise NotImplementedError("qk-norm would have to be permuted with the rows; no Audio8 "
                                      "release uses it")
        self.n_q, self.n_kv, self.d = attention.n_head, attention.n_local_heads, attention.head_dim
        self.wqkv = _deinterleave_rows(attention.wqkv, self.n_q, self.n_kv, self.d)
        self.wo = attention.wo

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q, k, v = self.wqkv(x).split((self.n_q * self.d, self.n_kv * self.d, self.n_kv * self.d), dim=-1)
        q = q.view(b, t, self.n_q, self.d).transpose(1, 2)
        k = k.view(b, t, self.n_kv, self.d).transpose(1, 2)
        v = v.view(b, t, self.n_kv, self.d).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = self.n_q // self.n_kv
        k = k[:, :, None].expand(b, self.n_kv, rep, t, self.d).reshape(b, self.n_q, t, self.d)
        v = v[:, :, None].expand(b, self.n_kv, rep, t, self.d).reshape(b, self.n_q, t, self.d)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, self.n_q * self.d))


class _Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.attention = _Attention(block.attention)
        self.feed_forward = block.feed_forward
        self.attention_norm = block.attention_norm
        self.ffn_norm = block.ffn_norm

    def forward(self, x, cos, sin, mask):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class _EmbedWrapper(nn.Module):
    """`rows [1, n, 1 + n_codebooks] -> inputs_embeds [1, n, dim]`: the token embedding plus the ten
    codebook embeddings, a text row's codes being the absent code whose row is zero."""

    def __init__(self, model):
        super().__init__()
        config = model.config
        self.n, self.size = int(config.num_codebooks), int(config.codebook_size) + 1
        width = int(config.dim)
        self.text = model.embeddings
        table = model.codebook_embeddings.weight.detach().float().view(self.n, self.size - 1, width)
        blocks = [torch.cat([table[i], torch.zeros(1, width)], 0) for i in range(self.n)]
        self.register_buffer("table", torch.cat(blocks, 0))
        # FLOAT, for `moss_tts_export._EmbedWrapper`'s reason: an int buffer is written as F32 and the
        # gather index must come out I32.
        self.register_buffer("offsets", torch.arange(self.n, dtype=torch.float32) * self.size)

    def forward(self, rows):
        n = rows.shape[1]
        text = self.text(rows[:, :, :1].reshape(n)).reshape(1, n, self.text.weight.shape[1])
        index = (rows[:, :, 1:].float() + self.offsets).to(torch.int32).reshape(n * self.n)
        codes = self.table[index].reshape(1, n, self.n, self.table.shape[1]).sum(dim=2)
        return text + codes


class _SlowWrapper(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> (logits over the 4097 drawable ids, normed
    last hidden)`. Logits first: `loom.sample_row` reduces a module's output 0."""

    def __init__(self, model):
        super().__init__()
        config = model.config
        self.layers = nn.ModuleList(_Block(b) for b in model.layers)
        self.norm = model.norm
        if not config.norm_fastlayer_input:
            raise NotImplementedError("the fast stack is fed the UN-normed hidden state; the export "
                                      "returns the normed one")
        cos, sin = rope_tables(model.freqs_cis)
        self.cos = nn.Embedding.from_pretrained(cos, freeze=True)
        self.sin = nn.Embedding.from_pretrained(sin, freeze=True)
        weight = model.embeddings.weight.detach()
        begin, size, eos = int(config.semantic_begin_id), int(config.codebook_size), int(config.eos_token_id)
        if int(config.semantic_end_id) != begin + size - 1:
            raise ValueError(f"semantic ids [{begin}, {config.semantic_end_id}] are not one codebook of "
                             f"{size}; codebook 0 = semantic - begin would not hold")
        self.head = nn.Linear(weight.shape[1], size + 1, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.cat([weight[begin:begin + size], weight[eos:eos + 1]], 0))

    def forward(self, inputs_embeds, position_ids, attention_mask):
        cos, sin = self.cos(position_ids), self.sin(position_ids)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        last = self.norm(hidden[:, -1:])
        return self.head(last), last


class _FastWrapper(nn.Module):
    """`(slow hidden, the codes drawn so far, position_ids, attention_mask) -> logits [1, 4096]` for the
    next codebook. The reference's position-0 step only fills its cache; here the prefix is re-run."""

    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(_Block(b) for b in model.fast_layers)
        self.project = model.fast_project_in
        self.embeddings = model.fast_embeddings
        self.norm = model.fast_norm
        self.output = model.fast_output
        cos, sin = rope_tables(model.fast_freqs_cis)
        self.cos = nn.Embedding.from_pretrained(cos, freeze=True)
        self.sin = nn.Embedding.from_pretrained(sin, freeze=True)

    def forward(self, slow_hidden, codes, position_ids, attention_mask):
        hidden = torch.cat([self.project(slow_hidden), self.embeddings(codes)], dim=1)
        cos, sin = self.cos(position_ids), self.sin(position_ids)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        return self.output(self.norm(hidden[:, -1:])).view(1, -1)


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(1, 1, seq_len, seq_len)


def load_reference(model_dir: str):
    """The checkpoint's own model at F32, through its remote code."""
    import transformers

    return transformers.AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                                  dtype=torch.float32).eval()


def greedy_with_wrappers(embed, slow, fast, config, prompt_rows: torch.Tensor, n_frames: int) -> list:
    """The driver's greedy loop in torch, over the three wrappers, uncached: `n_frames` frames of ten
    codes (fewer if EOS comes first). What `check_equivalence` compares to the reference's `generate`."""
    n, size = int(config.num_codebooks), int(config.codebook_size)
    begin = int(config.semantic_begin_id)
    sequence = embed(prompt_rows)
    frames = []
    for _ in range(n_frames):
        length = sequence.shape[1]
        logits, hidden = slow(sequence, torch.arange(length).view(1, -1), causal_mask(length))
        semantic = int(logits[0, 0].argmax())
        if semantic == size:
            break
        frame = [semantic]
        for k in range(1, n):
            out = fast(hidden, torch.tensor([frame]), torch.arange(k + 1).view(1, -1), causal_mask(k + 1))
            frame.append(int(out.argmax()))
        frames.append(frame)
        row = torch.tensor([[[semantic + begin] + frame]])
        sequence = torch.cat([sequence, embed(row)], dim=1)
    return frames


def check_equivalence(model, embed, slow, fast, tokenizer, n_frames: int = 4) -> None:
    """The wrappers' greedy codes against the reference's own `generate(do_sample=False)` on a real
    prompt, id for id; raises on any difference. Exact rather than within a tolerance because the
    rewrites are exact up to summation order, and a greedy argmax is what would expose one that is not."""
    config = model.config
    seg = prompt_segments(tokenizer)
    text = [int(i) for i in tokenizer.encode("Hello there.", add_special_tokens=False)]
    ids = seg["no_reference_head"] + text + seg["tail"]
    prompt = torch.zeros((1, int(config.num_codebooks) + 1, len(ids)), dtype=torch.long)
    prompt[0, 0] = torch.tensor(ids)
    with torch.inference_mode():
        ref = model.generate(input_ids=prompt, max_new_tokens=n_frames, do_sample=False)[0].T.tolist()
        rows = torch.full((1, len(ids), int(config.num_codebooks) + 1), int(config.codebook_size))
        rows[0, :, 0] = torch.tensor(ids)
        got = greedy_with_wrappers(embed, slow, fast, config, rows, n_frames)
    if got != ref:
        raise ValueError(f"the export's wrappers draw {got} where the reference's greedy generate draws "
                         f"{ref}")


@dataclass
class TextToCodesAudio8ExportConfig(BaseMultiPhaseModelExportConfig):
    """Audio8-TTS as three traced phases and a nested loop.

        once per utterance   embed (the prompt rows)      slow (prefill, cached)
        once per frame       nine x fast -> codebooks 1..9
                             embed (the new row)          slow (one cached step)

    Nothing but integers crosses the Lua boundary inside the loop ([ADR-031]): the slow stack's hidden
    state is a retained reference, and every draw is from a retained row.
    """

    model_dir: str = ""
    architecture: str = "audio8_tts"
    output_path: str = "audio8_tts.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "audio8_tts_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # Not 2, 7 or 14: the GQA fusion must tell the sequence axis from the K/V heads and the repeat.
    trace_len: int = 5

    _n_codebooks: Optional[int] = None
    _codebook_size: Optional[int] = None
    _dim: Optional[int] = None
    _max_seq_len: Optional[int] = None
    _semantic_begin: Optional[int] = None
    _sample_rate: Optional[int] = None
    _segments: Optional[dict] = None
    _ras: Optional[dict] = None

    __unchecked__ = {
        "model_dir": Unchecked("path to the HF directory; `load_model` raises on anything it cannot load"),
        "root_axis": Unchecked("`n_tokens` is forced for a KV-cached phase -- qwen3_tts_export says why"),
        "trace_len": Unchecked("the concrete length torch.jit.trace runs at; see the field comment"),
        "_n_codebooks": Unchecked("read off config.num_codebooks by read_constants"),
        "_codebook_size": Unchecked("read off config.codebook_size by read_constants"),
        "_dim": Unchecked("read off config.dim by read_constants"),
        "_max_seq_len": Unchecked("read off config.max_seq_len: the reference's own cache and its "
                                  "rope table both stop there, so the KvCache does too"),
        "_semantic_begin": Unchecked("read off config.semantic_begin_id by read_constants"),
        "_sample_rate": Unchecked("read off config.codec_sample_rate by read_constants"),
        "_segments": Unchecked("encoded by read_constants with the checkpoint's own tokenizer"),
        "_ras": Unchecked("read off config.ras_* by read_constants"),
    }

    def load_model(self):
        import transformers

        print(f"Loading Audio8-TTS from {self.model_dir}...")
        model = load_reference(self.model_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir)
        self.read_constants(model.config, tokenizer)
        self._wrappers = (_EmbedWrapper(model).eval(), _SlowWrapper(model).eval(), _FastWrapper(model).eval())
        check_equivalence(model, *self._wrappers, tokenizer)
        print("  wrappers vs the reference's greedy generate: identical")
        self._model = model
        return model

    def read_constants(self, config, tokenizer) -> None:
        """Everything the driver and the contract need that is not a weight. Split from `load_model` so
        a driver can be rendered without the weights."""
        if bool(getattr(config, "scale_codebook_embeddings", False)):
            raise NotImplementedError("scale_codebook_embeddings is not re-spelled by `_EmbedWrapper`")
        self._n_codebooks = int(config.num_codebooks)
        self._codebook_size = int(config.codebook_size)
        self._dim = int(config.dim)
        self._max_seq_len = int(config.max_seq_len)
        self._semantic_begin = int(config.semantic_begin_id)
        self._sample_rate = int(config.codec_sample_rate)
        self._segments = prompt_segments(tokenizer)
        self._ras = dict(window=int(config.ras_window_size), temperature=float(config.ras_temperature),
                         top_p=float(config.ras_top_p))

    def hparams(self) -> dict:
        if self._n_codebooks is None:
            return {}   # built without a checkpoint, e.g. by component_registry.usage()
        hparams = {
            # Spelled as `audio_codec_export` spells them: the same numbers from the other end of the pipe.
            "codec.n_codebooks": self._n_codebooks,
            "codec.codebook_size": self._codebook_size,
            "n_codes_ctx": self._max_seq_len,
            "sample_rate": self._sample_rate,
        }
        hparams.update({f"sampling.{k}": v for k, v in read_sampling_defaults(self.model_dir).items()})
        return hparams

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        # What a voice file must match (`audio8_tts_voices`, loom.cpp ADR-045): a code means something
        # only to the codec that produced it.
        from .audio8_tts_voices import codec_fingerprint

        contract["voice.compat"] = codec_fingerprint(self.model_dir)
        return contract

    def backend_kwargs(self) -> dict:
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams())

    def companions(self) -> List[CompanionExport]:
        """The codec, which this checkpoint ships as one file beside its weights."""
        codec = Path(self.model_dir) / "codec.pth"
        if not codec.is_file():
            return []
        return [CompanionExport(
            name="audio8-tts-codec",
            checkpoint=codec,
            why="this model emits codec TOKENS, not audio; its decoder ships in the same checkpoint",
        )]

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        self.load_model()
        embed, slow, fast = self._wrappers
        n, D, T = self._n_codebooks, self._dim, self.trace_len
        positions = ct.RangeDim(1, self._max_seq_len)
        rows_dim = ct.RangeDim(1, self._max_seq_len)
        steps = ct.RangeDim(2, n)
        return [
            ExportPhase(
                name="embed",
                wrapper=embed,
                dummy_inputs=(torch.zeros((1, T, n + 1), dtype=torch.long),),
                mil_inputs=[ct.TensorType(name="rows", shape=(1, rows_dim, n + 1), dtype=np.int32)],
            ),
            ExportPhase(
                name="slow",
                wrapper=slow,
                dummy_inputs=(torch.zeros(1, T, D), torch.arange(T).view(1, -1), causal_mask(T)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, positions, D), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, positions), dtype=np.int32),
                    # The SAME `RangeDim` on both mask axes: two instances are two roots.
                    ct.TensorType(name="attention_mask", shape=(1, 1, positions, positions),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=self._max_seq_len,
            ),
            ExportPhase(
                name="fast",
                wrapper=fast,
                dummy_inputs=(torch.zeros(1, 1, D), torch.zeros((1, 3), dtype=torch.long),
                              torch.arange(4).view(1, -1), causal_mask(4)),
                mil_inputs=[
                    ct.TensorType(name="slow_hidden", shape=(1, 1, D), dtype=np.float32),
                    ct.TensorType(name="codes", shape=(1, ct.RangeDim(1, n - 1)), dtype=np.int32),
                    ct.TensorType(name="position_ids", shape=(1, steps), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, steps, steps), dtype=np.float32),
                ],
                # The slow hidden state is the first position; the codes are the rest.
                declared_axes={"codes": {1: "n_tokens - 1"}},
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import DriverReturn, ExportConstants, LuaFragment

        seg = self._segments or {k: [0] for k in ("no_reference_head", "reference_head",
                                                  "reference_speech", "reference_to_text", "tail")}
        sampling = read_sampling_defaults(self.model_dir) if self.model_dir else {}
        ras = self._ras or {"window": 0, "temperature": 1.0, "top_p": 1.0}
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "N_CODEBOOKS": self._n_codebooks or 0,
                "CODEBOOK_SIZE": self._codebook_size or 0,
                # The trimmed head's last row: both the stop condition and the end of the draw window.
                "EOS_INDEX": self._codebook_size or 0,
                "SEMANTIC_BEGIN": self._semantic_begin or 0,
                "MAX_POSITIONS": self._max_seq_len or 0,
                # `generation_config.json`'s own cap, in frames: 512 is ~24 s of audio.
                "MAX_NEW_TOKENS": _generation_value(self.model_dir, "max_new_tokens", 512),
                "PROMPT_NO_REFERENCE_HEAD": seg["no_reference_head"],
                "PROMPT_REFERENCE_HEAD": seg["reference_head"],
                "PROMPT_REFERENCE_SPEECH": seg["reference_speech"],
                "PROMPT_REFERENCE_TO_TEXT": seg["reference_to_text"],
                "PROMPT_TAIL": seg["tail"],
                "TEMPERATURE": sampling.get("temperature", 0.0),
                "TOP_K": sampling.get("top_k", 0),
                "TOP_P": sampling.get("top_p", 1.0),
                "RAS_WINDOW": ras["window"],
                "RAS_TEMPERATURE": ras["temperature"],
                "RAS_TOP_P": ras["top_p"],
            }),
            LuaFragment(
                self.driver_script_path / "01_generate.lua",
                reads=("N_CODEBOOKS", "CODEBOOK_SIZE", "EOS_INDEX", "SEMANTIC_BEGIN", "MAX_POSITIONS",
                       "MAX_NEW_TOKENS", "PROMPT_NO_REFERENCE_HEAD", "PROMPT_REFERENCE_HEAD",
                       "PROMPT_REFERENCE_SPEECH", "PROMPT_REFERENCE_TO_TEXT", "PROMPT_TAIL",
                       "TEMPERATURE", "TOP_K", "TOP_P", "RAS_WINDOW", "RAS_TEMPERATURE", "RAS_TOP_P"),
                defines=("_codes",),
            ),
            DriverReturn(values=("_codes",)),
        ]


def _generation_value(model_dir: str, key: str, fallback):
    """One key of `generation_config.json`, which `read_sampling_defaults` does not carry."""
    path = Path(model_dir) / "generation_config.json"
    if not model_dir or not path.is_file():
        return fallback
    return json.loads(path.read_text()).get(key, fallback)


def _is_audio8_tts(path: Path) -> bool:
    """An HF directory declaring `model_type == "arktts"` -- the DualAR, not its `codec.pth`, which
    `audio_codec_export` recognizes as a FILE."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        return json.loads(config_path.read_text()).get("model_type") == MODEL_TYPE
    except (json.JSONDecodeError, OSError):
        return False


def _build_audio8_tts(path: Path, output_path: str) -> TextToCodesAudio8ExportConfig:
    return TextToCodesAudio8ExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-codes",
        config_class=TextToCodesAudio8ExportConfig,
        recognizers=[ModelRecognizer(name="audio8-tts", detect=_is_audio8_tts,
                                     build_config=_build_audio8_tts)],
    ))
