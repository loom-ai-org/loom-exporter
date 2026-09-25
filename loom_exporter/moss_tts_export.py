"""MOSS-TTS-Local-Transformer-v1.5: family 10's third leaf, text in and 12-codebook `audio_codes` out.

The codes are what MOSS-Audio-Tokenizer-v2 decodes (`moss_audio_tokenizer_export`, family 11), and the
pair is two files by [ADR-022]. Structurally this is Qwen3-TTS's shape -- a big stack that runs once per
audio frame and a small one that runs once per codebook -- with the sizes moved:

    global stack   ->  one hidden state per frame   (Qwen3, 36 layers, 2560 wide, GQA 32/8, KV-cached)
    local stack    ->  continue/stop, then 12 codes (GPT-2, 1 layer, 2560 wide, re-run per codebook)

**The input embedding is a sum, and it is folded like the codec's quantizer.** A row is one text id
plus twelve audio ids, and the reference adds the text embedding to the twelve audio embeddings with
every `audio_pad_code` (1024) MASKED out. Here each codebook's table gains a zero row at 1024, so the
mask is the lookup: the same absent-id trick as `moss_audio_tokenizer_export.fold_quantizer`, for the
same reason, on the id MOSS already uses. One phase embeds the prompt and every generated frame alike.

**The global stack is transformers' own `Qwen3Model`, not the checkpoint's.** MOSS ships its own
`MossQwen3Model` (remote code) whose attention builds boolean masks from shapes. The arithmetic is
Qwen3's exactly -- q/k RMSNorm before rotate-half RoPE at theta 1e6, SwiGLU, pre-norm -- so the weights
load into `transformers.Qwen3Model` unchanged (`assign=True`, no copy), which is the stack
`causal_lm_export` already caches and fuses with GQA. `load_model` ASSERTS the two agree on a real
prefix before anything is traced, so this is checked rather than trusted.

**The local stack is not cached, for Qwen3-TTS's reason.** One KvCache per model has one per-layer
width: the global stack's is 8 K/V heads x 128, the local's 32 x 80. The local never sees more than 13
positions (the hidden state, then up to 11 drawn codes), so re-running its prefix costs 78 row-forwards
of ONE layer per frame, beside a 36-layer step. Its head is merged: the twelve audio heads (tied to the
audio embeddings) then the two-way continue/stop head, `[12 * 1024 + 2, 2560]`, so every draw is a
`lo`/`hi` window of one output -- `qwen3_tts_export.merged_predictor_tables`' trick.

**The text head is not exported.** `text_lm_head` is the 151936-row tied copy of the input embedding,
and this release decides continue/stop with `local_text_lm_head`'s two rows (`local_text_head_mode:
binary`). 1.56 GB at F32 that no draw reads.

**The prompt template ships as ids, pre-encoded here segment by segment.** The processor encodes each
template piece SEPARATELY and concatenates the ids -- which is not the same as encoding the rendered
string -- so the export does exactly that with the checkpoint's own tokenizer, once per supported
language, and the driver wraps the caller's text ids in them. Voice cloning (a reference clip's codes
in the prompt) needs the codec's ENCODER, which is not exported; this is the reference-free mode.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked

# The processor's template pieces (`processing_moss_tts.py`), verbatim. Each is encoded on its own.
USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"

# The README's supported-languages table, as the NAMES its examples pass (`language="French"`): the
# template renders the value it is given verbatim, so the name is what the model was trained to read.
# The codes are what the contract declares, the same vocabulary Whisper's `languages` uses.
LANGUAGES = [
    ("zh", "Chinese"), ("yue", "Cantonese"), ("en", "English"), ("ar", "Arabic"), ("cs", "Czech"),
    ("da", "Danish"), ("nl", "Dutch"), ("fi", "Finnish"), ("fr", "French"), ("de", "German"),
    ("el", "Greek"), ("he", "Hebrew"), ("hi", "Hindi"), ("hu", "Hungarian"), ("it", "Italian"),
    ("ja", "Japanese"), ("ko", "Korean"), ("mk", "Macedonian"), ("ms", "Malay"), ("fa", "Persian"),
    ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"), ("ru", "Russian"), ("es", "Spanish"),
    ("sw", "Swahili"), ("sv", "Swedish"), ("tl", "Tagalog"), ("th", "Thai"), ("tr", "Turkish"),
    ("vi", "Vietnamese"),
]

# The README's recommended audio sampling, which is what its own example passes to `generate`. The
# continue/stop draw uses `generate`'s text defaults, which that example leaves alone.
AUDIO_TEMPERATURE, AUDIO_TOP_P, AUDIO_TOP_K = 1.7, 0.8, 25
TEXT_TEMPERATURE, TEXT_TOP_P, TEXT_TOP_K = 1.0, 1.0, 50


def after_reference(language: Optional[str]) -> str:
    """`processing_moss_tts._render_user_prompt_after_reference` with every optional field unset."""
    lang = "None" if language is None else str(language).strip() or "None"
    return ("\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone"
            "\n- Ambient Sound:\nNone\n- Language:\n" + lang + "\n- Text:\n")


def prompt_segments(tokenizer, config) -> dict:
    """The ids the driver wraps the caller's text in: `head`, one `after` per language (index 0 is
    no language), and `tail`. Encoded exactly as `_build_generation_or_voice_clone_codes` does."""
    def enc(text):
        return list(tokenizer.encode(text, add_special_tokens=False))

    head = [int(config.im_start_token_id)] + enc(USER_ROLE_PREFIX) + \
        enc(USER_TEMPLATE_REFERENCE_PREFIX) + enc("None")
    tail = enc(USER_TEMPLATE_SUFFIX) + [int(config.im_end_token_id)] + enc(ASSISTANT_TURN_PREFIX) + \
        [int(config.im_start_token_id)] + enc(ASSISTANT_ROLE_PREFIX) + [int(config.audio_start_token_id)]
    after = [enc(after_reference(None))] + [enc(after_reference(name)) for _, name in LANGUAGES]
    return dict(head=head, after=after, tail=tail)


class _EmbedWrapper(nn.Module):
    """`rows [1, n, 1 + n_vq] -> inputs_embeds [1, n, hidden]`: text embedding plus the twelve audio
    embeddings, with the pad code's zero row standing in for the reference's mask."""

    def __init__(self, text_embedding, audio_embeddings, codebook_size: int):
        super().__init__()
        self.text = nn.Embedding(text_embedding.weight.shape[0], text_embedding.weight.shape[1])
        self.text.weight = text_embedding.weight
        width = text_embedding.weight.shape[1]
        self.size = codebook_size + 1
        self.n_vq = len(audio_embeddings)
        blocks = [torch.cat([e.weight.detach().float(), torch.zeros(1, width)], 0)
                  for e in audio_embeddings]
        self.register_buffer("table", torch.cat(blocks, 0))
        # FLOAT, for `moss_audio_tokenizer_export`'s reason: an int weight is written as F32, and the
        # gather index must come out I32 (loom.cpp Retro-060's neighbour).
        self.register_buffer("offsets", torch.arange(self.n_vq, dtype=torch.float32) * self.size)

    def forward(self, rows):
        n = rows.shape[1]
        # The index flattened to one axis: `ggml_get_rows` takes a rank-3 index only when its third
        # axis matches the table's, and `[1, n, 1]` does not -- the load succeeds and the first call
        # aborts inside ggml.
        text = self.text(rows[:, :, :1].reshape(n)).reshape(1, n, self.text.weight.shape[1])
        index = (rows[:, :, 1:].float() + self.offsets).to(torch.int32).reshape(n * self.n_vq)
        audio = self.table[index].reshape(1, n, self.n_vq, self.table.shape[1]).sum(dim=2)
        return text + audio


class _GlobalWrapper(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> the last position's hidden state`, normed.

    The one output is what the local stack is conditioned on, and it stays in the engine: the driver
    hands it to `local_*` as a reference ([ADR-031])."""

    def __init__(self, qwen3):
        super().__init__()
        self.model = qwen3

    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                            attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, -1:]


class _LocalWrapper(nn.Module):
    """`(global hidden, position_ids, attention_mask[, rows]) -> logits over every head`.

    The GPT-2 block re-spelled for the trace, and exact against `MossTTSNanoGPT2Model`: pre-LN,
    `c_attn` split into Q/K/V (a `split` is fine, but three Linears keep every select out of the graph),
    interleaved-pair RoPE at the block's own base, SiLU MLP, then `ln_f`. `rows` are ABSOLUTE rows of
    the merged audio table -- group `g` code `c` is `g * 1024 + c`, which is what a windowed draw on the
    merged head returns -- so the driver never adds or removes an offset inside the loop.
    """

    def __init__(self, local, heads, embeddings, with_rows: bool):
        super().__init__()
        block = local.h[0]
        if len(local.h) != 1:
            raise NotImplementedError(f"this release's local stack has 1 layer; this one has {len(local.h)}")
        attn = block.attn
        self.H, self.Dh, self.D = attn.num_heads, attn.head_dim, attn.embed_dim
        self.scale = 1.0 / math.sqrt(self.Dh) if attn.scale_attn_weights else 1.0
        if attn.scale_attn_by_inverse_layer_idx:
            raise NotImplementedError("scale_attn_by_inverse_layer_idx is not re-spelled here")
        if attn.rotary_emb is None:
            raise NotImplementedError("the local stack is expected to use RoPE")
        self.ln_1, self.ln_2, self.ln_f = block.ln_1, block.ln_2, local.ln_f
        self.mlp = block.mlp
        self.c_proj = attn.c_proj
        w, b = attn.c_attn.weight.detach(), attn.c_attn.bias.detach()
        self.q, self.k, self.v = (self._linear(w[i * self.D:(i + 1) * self.D],
                                               b[i * self.D:(i + 1) * self.D]) for i in range(3))
        inv = 1.0 / (attn.rotary_emb.base ** (torch.arange(0, self.Dh, 2, dtype=torch.float32) / self.Dh))
        self.register_buffer("inv_freq", torch.repeat_interleave(inv, 2).view(1, -1))
        self.with_rows = with_rows
        if with_rows:
            self.rows = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            with torch.no_grad():
                self.rows.weight.copy_(embeddings)
        self.head = self._linear(heads, None)

    @staticmethod
    def _linear(weight, bias):
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            layer.weight.copy_(weight)
            if bias is not None:
                layer.bias.copy_(bias)
        return layer

    def _rope(self, x, cos, sin):                          # x [1, L, H, Dh], pairs interleaved
        L = x.shape[1]
        pairs = x.reshape(1, L, self.H * self.Dh // 2, 2)
        rotated = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(1, L, self.H, self.Dh)
        return x * cos + rotated * sin

    def forward(self, global_hidden, position_ids, attention_mask, rows=None):
        x = global_hidden
        if self.with_rows:
            x = torch.cat([x, self.rows(rows)], dim=1)
        L = x.shape[1]
        # An outer product rather than a broadcast multiply: ggml broadcasts one operand only.
        # A reshape, not a transpose: a transposed view reaches `ggml_mul_mat` as a non-contiguous
        # second operand and aborts (`nb10 == ggml_type_size`).
        angle = (position_ids.float().reshape(L, 1) @ self.inv_freq).reshape(1, L, 1, self.Dh)
        cos, sin = torch.cos(angle), torch.sin(angle)
        h = self.ln_1(x)
        q = self._rope(self.q(h).reshape(1, L, self.H, self.Dh), cos, sin).permute(0, 2, 1, 3)
        k = self._rope(self.k(h).reshape(1, L, self.H, self.Dh), cos, sin).permute(0, 2, 3, 1)
        v = self.v(h).reshape(1, L, self.H, self.Dh).permute(0, 2, 1, 3)
        scores = (q @ k) * self.scale + attention_mask
        out = (torch.softmax(scores, dim=-1) @ v).permute(0, 2, 1, 3).reshape(1, L, self.D)
        x = x + self.c_proj(out)
        x = x + self.mlp(self.ln_2(x))
        return self.head(self.ln_f(x)[:, -1:]).reshape(1, self.head.weight.shape[0])


def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def load_reference(model_dir: str):
    """The checkpoint's own model at F32 on the SDPA path, and its config."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.attn_implementation = "sdpa"
    config.local_transformer_attn_implementation = "sdpa"
    config.qwen3_config._attn_implementation = "sdpa"
    model = transformers.AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    return model, config


def as_hf_qwen3(model, config) -> nn.Module:
    """The global stack as `transformers.Qwen3Model`, sharing the checkpoint's tensors (no copy)."""
    import transformers

    qcfg = transformers.Qwen3Config(**config.qwen3_config.to_dict())
    qcfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = transformers.Qwen3Model(qcfg)
    hf.load_state_dict(model.transformer.state_dict(), strict=True, assign=True)
    hf.rotary_emb = transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding(config=qcfg)
    return hf.eval()


def check_global_equivalence(model, hf, n: int = 6, atol: float = 2e-4) -> float:
    """The reference's own stack against the transformers one on a real prefix; raises past `atol`."""
    torch.manual_seed(0)
    embeds = torch.randn(1, n, hf.config.hidden_size) * 0.02
    with torch.no_grad():
        want = model.transformer(inputs_embeds=embeds, use_cache=False).last_hidden_state
        got = hf(inputs_embeds=embeds, position_ids=torch.arange(n).view(1, -1),
                 attention_mask=causal_mask(n), use_cache=False).last_hidden_state
    diff = float((got - want).abs().max())
    scale = float(want.abs().max())
    if not diff <= atol * max(scale, 1.0):
        raise AssertionError(f"transformers' Qwen3Model is {diff:.3e} from MossQwen3Model (peak "
                             f"{scale:.3e}); the global stack is not plain Qwen3 after all")
    return diff


@dataclass
class TextToCodesMossTTSExportConfig(BaseMultiPhaseModelExportConfig):
    """MOSS-TTS-Local-Transformer-v1.5 as four traced phases and a nested loop.

        once per utterance   embed (the prompt rows)    global (prefill, cached)
        once per frame       local_first -> continue/stop and codebook 0
                             eleven x local_steps -> codebooks 1..11
                             embed (the new row)        global (one cached step)

    Nothing but integers crosses the Lua boundary inside the loop ([ADR-031]): the hidden state is a
    retained reference, and every draw is a window of one merged head.
    """

    model_dir: str = ""
    architecture: str = "moss_tts_local"
    output_path: str = "moss_tts_local.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "moss_tts_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    trace_len: int = 7          # not 8: the GQA fusion must tell 8 K/V heads from the sequence axis
    # Prompt + generated frames share one KvCache. 2048 is ~2.5 min of audio after a long prompt; the
    # cache is 36 layers x 2048 x 1024 x K,V x 4 bytes = 604 MB at F32.
    max_positions: int = 2048

    _n_vq: Optional[int] = None
    _codebook_size: Optional[int] = None
    _hidden: Optional[int] = None
    _segments: Optional[dict] = None
    _slot_id: Optional[int] = None
    _pad_code: Optional[int] = None
    _sample_rate: Optional[int] = None

    __unchecked__ = {
        "model_dir": Unchecked("path to the HF directory; `load_model` raises on anything it cannot load"),
        "root_axis": Unchecked("`n_tokens` is forced for a KV-cached phase -- qwen3_tts_export says why"),
        "trace_len": Unchecked("the concrete length torch.jit.trace runs at; see the field comment"),
        "max_positions": Unchecked("the ct.RangeDim bound on the cached axis, a memory decision"),
        "_n_vq": Unchecked("read off config.n_vq by load_model"),
        "_codebook_size": Unchecked("read off config.audio_codebook_sizes by load_model"),
        "_hidden": Unchecked("read off config.hidden_size by load_model"),
        "_segments": Unchecked("encoded by load_model with the checkpoint's own tokenizer"),
        "_slot_id": Unchecked("read off config.audio_assistant_slot_token_id by load_model"),
        "_pad_code": Unchecked("read off config.audio_pad_code by load_model"),
        "_sample_rate": Unchecked("read off config.sampling_rate by load_model"),
    }

    def load_model(self):
        import transformers

        print(f"Loading MOSS-TTS-Local from {self.model_dir}...")
        model, config = load_reference(self.model_dir)
        if str(getattr(config, "local_text_head_mode", "")).lower() != "binary":
            raise NotImplementedError("only the binary continue/stop head (v1.5) is exported")
        sizes = set(int(s) for s in config.audio_codebook_sizes)
        if len(sizes) != 1:
            raise NotImplementedError(f"codebooks of different sizes {sorted(sizes)} are not merged here")
        self._n_vq = int(config.n_vq)
        self._codebook_size = sizes.pop()
        self._hidden = int(config.hidden_size)
        self._slot_id = int(config.audio_assistant_slot_token_id)
        self._pad_code = int(config.audio_pad_code)
        if self._pad_code != self._codebook_size:
            raise NotImplementedError("the zero row is placed at `codebook_size`; the pad code is not there")
        self._sample_rate = int(config.sampling_rate)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir)
        self._segments = prompt_segments(tokenizer, config)
        hf = as_hf_qwen3(model, config)
        diff = check_global_equivalence(model, hf)
        print(f"  transformers Qwen3Model vs MossQwen3Model: max |d| {diff:.2e}")
        self._model, self._hf = model, hf
        return model

    def hparams(self) -> dict:
        if self._n_vq is None:
            return {}
        return {
            # Spelled as `audio_codec_export` spells them: the same numbers from the other end of the
            # pipe. 12 here against the codec's 32 is what `codec.absent_code` exists for (ADR-050).
            "codec.n_codebooks": self._n_vq,
            "codec.codebook_size": self._codebook_size,
            "n_codes_ctx": self.max_positions,
            "sample_rate": self._sample_rate,
        }

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "vocab"
        # Read by `ModelContract` as `loom.text.languages` (the writer adds the prefix) -- the same key
        # SenseVoice declares. Index i+1 of this list is `PROMPT_AFTER` segment i+1 in the driver.
        contract["text.languages"] = [code for code, _ in LANGUAGES]
        return contract

    def backend_kwargs(self) -> dict:
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams())

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        model = self.load_model()
        n_vq, V, D = self._n_vq, self._codebook_size, self._hidden
        heads = torch.cat([e.weight.detach().float() for e in model.audio_embeddings]
                          + [model.local_text_lm_head.weight.detach().float()], 0)
        merged = torch.cat([e.weight.detach().float() for e in model.audio_embeddings], 0)
        positions = ct.RangeDim(1, self.max_positions)
        rows_dim = ct.RangeDim(1, self.max_positions)
        steps = ct.RangeDim(2, n_vq)
        return [
            ExportPhase(
                name="embed",
                wrapper=_EmbedWrapper(model.transformer.embed_tokens, model.audio_embeddings, V).eval(),
                dummy_inputs=(torch.zeros((1, self.trace_len, n_vq + 1), dtype=torch.long),),
                mil_inputs=[ct.TensorType(name="rows", shape=(1, rows_dim, n_vq + 1), dtype=np.int32)],
            ),
            ExportPhase(
                name="global",
                wrapper=_GlobalWrapper(self._hf).eval(),
                dummy_inputs=(torch.zeros(1, self.trace_len, D),
                              torch.arange(self.trace_len).view(1, -1),
                              causal_mask(self.trace_len)),
                mil_inputs=[
                    ct.TensorType(name="inputs_embeds", shape=(1, positions, D), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, positions), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, positions, positions),
                                  dtype=np.float32),
                ],
                fuse_attention=True,
                kv_cache_size=self.max_positions,
            ),
            ExportPhase(
                name="local_first",
                wrapper=_LocalWrapper(model.local_transformer, heads, merged, False).eval(),
                dummy_inputs=(torch.zeros(1, 1, D), torch.zeros((1, 1), dtype=torch.long),
                              torch.zeros(1, 1, 1, 1)),
                mil_inputs=[
                    ct.TensorType(name="global_hidden", shape=(1, 1, D), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, 1, 1), dtype=np.float32),
                ],
            ),
            ExportPhase(
                name="local_steps",
                wrapper=_LocalWrapper(model.local_transformer, heads, merged, True).eval(),
                dummy_inputs=(torch.zeros(1, 1, D), torch.arange(5).view(1, -1), causal_mask(5),
                              torch.zeros((1, 4), dtype=torch.long)),
                mil_inputs=[
                    ct.TensorType(name="global_hidden", shape=(1, 1, D), dtype=np.float32),
                    ct.TensorType(name="position_ids", shape=(1, steps), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 1, steps, steps), dtype=np.float32),
                    ct.TensorType(name="rows", shape=(1, ct.RangeDim(1, n_vq - 1)), dtype=np.int32),
                ],
                declared_axes={"rows": {1: "n_tokens - 1"}},
            ),
        ]

    def driver_components(self) -> List:
        from .driver_components import DriverReturn, ExportConstants, LuaFragment

        seg = self._segments or {"head": [0], "after": [[0]], "tail": [0]}
        after_flat, after_offsets = [], []
        for ids in seg["after"]:
            after_offsets.append(len(after_flat))
            after_flat.extend(ids)
        after_offsets.append(len(after_flat))
        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values={
                "N_VQ": self._n_vq or 0,
                "CODEBOOK_SIZE": self._codebook_size or 0,
                "PAD_CODE": self._pad_code or 0,
                "SLOT_ID": self._slot_id or 0,
                "PROMPT_HEAD": seg["head"],
                "PROMPT_TAIL": seg["tail"],
                # One `after` per language, flattened: language i is `[OFFSETS[i+1], OFFSETS[i+2])`
                # in Lua's 1-based terms, and index 0 is "no language".
                "PROMPT_AFTER": after_flat,
                "PROMPT_AFTER_OFFSETS": after_offsets,
                "MAX_POSITIONS": self.max_positions,
                "MAX_NEW_TOKENS": 4096,
                "AUDIO_TEMPERATURE": AUDIO_TEMPERATURE, "AUDIO_TOP_K": AUDIO_TOP_K,
                "AUDIO_TOP_P": AUDIO_TOP_P,
                "TEXT_TEMPERATURE": TEXT_TEMPERATURE, "TEXT_TOP_K": TEXT_TOP_K,
                "TEXT_TOP_P": TEXT_TOP_P,
            }),
            LuaFragment(
                self.driver_script_path / "01_generate.lua",
                reads=("N_VQ", "CODEBOOK_SIZE", "PAD_CODE", "SLOT_ID", "PROMPT_HEAD", "PROMPT_TAIL",
                       "PROMPT_AFTER", "PROMPT_AFTER_OFFSETS", "MAX_POSITIONS", "MAX_NEW_TOKENS",
                       "AUDIO_TEMPERATURE", "AUDIO_TOP_K", "AUDIO_TOP_P", "TEXT_TEMPERATURE",
                       "TEXT_TOP_K", "TEXT_TOP_P"),
                defines=("_codes",),
            ),
            DriverReturn(values=("_codes",)),
        ]


def _is_moss_tts_local(path: Path) -> bool:
    cfg_path = path / "config.json"
    if not path.is_dir() or not cfg_path.exists():
        return False
    try:
        return json.loads(cfg_path.read_text()).get("model_type") == "moss_tts_local"
    except (json.JSONDecodeError, OSError):
        return False


def _build_moss_tts_local(path: Path, output_path: str) -> TextToCodesMossTTSExportConfig:
    return TextToCodesMossTTSExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-codes",
        config_class=TextToCodesMossTTSExportConfig,
        recognizers=[ModelRecognizer(name="moss-tts-local", detect=_is_moss_tts_local,
                                     build_config=_build_moss_tts_local)],
    ))
