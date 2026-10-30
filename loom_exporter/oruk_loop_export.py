"""oruk-loop-153M: Qwen3-ASR-1.7B distilled into two looped stacks, as a leaf over family 3's template.

The architecture is the teacher's -- a 24-layer window-attention audio encoder (1024-wide), a projector,
a 28-layer Qwen3 decoder (2048-wide, GQA 16/8) -- but each stack stores **one** block and runs it at
every depth. A looped linear layer computes `W x + b[d] + B[d] (A[d] x)`: `W` is shared by all depths,
the bias and the rank-r adapter (64 in the encoder, 16 in the decoder) belong to depth `d`, and so do
the LayerNorm / RMSNorm gains. The vocabulary is cut to 36,974 English tokens, one tied table for the
embedding and the head.

**The loop is kept, not merged.** The checkpoint's own `merge_loops()` materialises `W + B[d] A[d]` at
every depth, which is what its CUDA path runs -- and which would export a 1.75B-parameter file from a
153M-parameter checkpoint. Traced as written instead, each depth is a shared-weight matmul plus a
two-matmul low-rank branch, and the GGUF writer's content addressing stores `W` once however many
depths read it. That is ~200M parameters on disk; the arithmetic is the teacher's either way (the
model card says so too), plus the adapters' small rank-r products.

**Token ids are REMAPPED to the kept vocabulary**, which is the one structural liberty this leaf takes.
The checkpoint keeps Qwen3's 151,936-id space and returns `-inf` for the 114,962 ids it dropped, by
scattering 36,974 real logits into a full-width row. A scatter is not an engine primitive, and nothing
needs the full width: the kept ids are sorted (`keep[i] < keep[i+1]`, checked), so kept index `i` IS
token `keep[i]` and the order of every logit that can win is unchanged. The head therefore emits the
36,974 kept logits, the embedding takes kept indices, and the GGUF carries a tokenizer whose id `i` is
the piece of original id `keep[i]` -- rewritten from the checkpoint's own `tokenizer.json`
(`_write_kept_tokenizer`), with every merge that needs a dropped piece removed. The prompt and stop
ids pass through the same map (`lut`), and a prompt id the table does not keep is an error.

**The prompt is transformers 5.17's, not the installed version's.** The model card is tested with 5.17,
where `apply_transcription_request(language="English")` FORCES the language: an empty system turn,
then the assistant turn pre-filled with `language English<asr_text>`, so the model writes only the
transcript. 5.14.1 (the `ovos` venv) instead puts `English` in the system turn and lets the model write
the preamble itself -- a different prompt from the one the card's numbers come from. The modeling code
is the same in both (diffed: no numerical change), so the prompt is built here explicitly and the
installed version does not decide it. The card also says the model was trained and evaluated only
with this English prompt.

**Requires `trust_remote_code`**: the checkpoint ships `modeling_oruk_loop.py` (MIT), a subclass of the
stock `Qwen3ASRForConditionalGeneration`. Its `from_pretrained` is its own and ignores
`attn_implementation`, so the language model keeps the default (sdpa) attention that
`fuse_loom_attention` matches -- see `qwen3_asr_export.ASRQwen3SpeechLMExportConfig.load_model`.

**Weights license: PolyForm Small Business 1.0.0 or PolyForm Noncommercial 1.0.0** (Oruk AI), not the
Apache-2.0 of the teacher. Redistribution with changes is allowed provided the terms and the
`Required Notice:` line travel with every copy, which is the card's job, not this module's.
"""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_asr_export import ASRQwen3SpeechLMExportConfig, WindowedAudioEncoder
from .spec_protocol import Unchecked

# What `apply_transcription_request(language=...)` pre-fills the assistant turn with in transformers
# 5.17 (`processing_qwen3_asr.py`: `f"language {lang}<asr_text>"`, rendered with
# `continue_final_message=True`). The only language this checkpoint was trained with.
_FORCED_LANGUAGE = "English"


class LoopedWindowedAudioEncoder(WindowedAudioEncoder):
    """`WindowedAudioEncoder` over one shared block: before each depth's layer runs, the checkpoint's
    own depth cursor is moved, which is how every `LoopLinear` / `LoopLayerNorm` in the block picks its
    depth's bias, adapter and gains. Tracing runs this generator once, so each depth becomes its own
    unrolled layer in the graph, reading the shared weight and that depth's own small tensors."""

    def __init__(self, mel, tower, projector, audio_config, cursor):
        super().__init__(mel, tower, projector, audio_config)
        self.cursor = cursor

    def layers(self):
        for depth in range(self.cursor.n):
            self.cursor.depth = depth
            yield self.tower.loop


class _KeptEmbedding(nn.Module):
    """Kept index -> embedding row. The checkpoint's `TableEmbedding` takes ORIGINAL ids and maps them
    through `lut` first; after the remap the caller already holds the kept index."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(table, requires_grad=False)

    def forward(self, tokens):
        return F.embedding(tokens, self.weight)


class _KeptHead(nn.Module):
    """Hidden -> the 36,974 kept logits, in kept-index order: `TableHead` without its scatter into the
    full 151,936-wide row of `-inf`, which no argmax over the kept ids can ever pick."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(table, requires_grad=False)

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


def _write_kept_tokenizer(model_dir: Path, keep: np.ndarray, out_dir: Path) -> None:
    """The checkpoint's `tokenizer.json` restricted to the kept ids and renumbered to kept indices.

    Pieces and added tokens keep their text and take `lut[id]`; a merge survives only when both halves
    and their concatenation are kept, which is the condition HF's BPE model itself requires of a merge
    (the card says the kept set is closed under merges, so no kept piece loses the merge that builds
    it -- `_check_kept_tokenizer` verifies the decode side). `tokenizer_config.json` names its special
    tokens by TEXT, so it is copied unchanged.
    """
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    lut = {int(orig): i for i, orig in enumerate(keep.tolist())}
    vocab = {piece: lut[idx] for piece, idx in tok["model"]["vocab"].items() if idx in lut}
    added = [dict(t, id=lut[t["id"]]) for t in tok.get("added_tokens", []) if t["id"] in lut]
    pieces = set(vocab)
    merges = []
    for m in tok["model"]["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        if a in pieces and b in pieces and a + b in pieces:
            merges.append(m)
    ids = sorted([*vocab.values(), *(t["id"] for t in added)])
    if ids != list(range(len(keep))):
        raise ValueError(
            f"the kept tokenizer must number its pieces 0..{len(keep) - 1} exactly once; it has "
            f"{len(ids)} ids over {len(set(ids))} distinct values. A kept id with no piece (or two "
            f"pieces) would make the remap ambiguous."
        )
    n_merges = len(tok["model"]["merges"])
    tok["model"]["vocab"] = vocab
    tok["model"]["merges"] = merges
    tok["added_tokens"] = added
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    (out_dir / "tokenizer_config.json").write_text((model_dir / "tokenizer_config.json").read_text())
    print(f"  kept tokenizer: {len(vocab)} pieces + {len(added)} added, "
          f"{len(merges)} of {n_merges} merges")


def _check_kept_tokenizer(model_dir: Path, keep: np.ndarray, kept_dir: Path) -> None:
    """Every kept index must decode to exactly what its original id decodes to -- the one property the
    driver relies on, since it detokenizes kept indices. Checked over the whole table, piece by piece
    and on one long run (byte-level pieces only form text in sequence)."""
    from tokenizers import Tokenizer

    orig = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    kept = Tokenizer.from_file(str(kept_dir / "tokenizer.json"))
    keep_list = keep.tolist()
    bad = [i for i, o in enumerate(keep_list) if kept.id_to_token(i) != orig.id_to_token(o)]
    if bad:
        raise ValueError(f"{len(bad)} kept indices name a different piece than their original id, "
                         f"first {bad[:5]}")
    run = list(range(0, len(keep_list), 7))
    if kept.decode(run, skip_special_tokens=False) != orig.decode([keep_list[i] for i in run],
                                                                 skip_special_tokens=False):
        raise ValueError("the kept tokenizer decodes a run of kept indices differently from the "
                         "original tokenizer on the same tokens")


@dataclass
class ASROrukLoopExportConfig(ASRQwen3SpeechLMExportConfig):
    """oruk-loop-153M's loader: the stock Qwen3-ASR leaf, with the loop kept and the vocabulary remapped."""

    architecture: str = "oruk-loop"
    output_path: str = "oruk_loop_mil.gguf"

    # One piece's worth of positions, not the template's 4096: see `prompt_segment_constants`, which
    # derives it from the piece length and token budget and replaces this before any phase uses it.
    max_seq_len: int = 1024

    _keep: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _kept_tokenizer_dir: Optional[str] = field(default=None, init=False, repr=False)
    _quiet_split: Optional[dict] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        **ASRQwen3SpeechLMExportConfig.__unchecked__,
        "_keep": Unchecked(
            "the checkpoint's own `token_table.keep` buffer (kept index -> original id), read in "
            "load_model and checked sorted there; every id the driver carries goes through it"
        ),
        "_kept_tokenizer_dir": Unchecked(
            "a staging copy of the checkpoint's tokenizer renumbered to kept indices, written and "
            "decode-checked against the original in load_model"
        ),
        "_quiet_split": Unchecked(
            "the reference's long-form numbers (`loom.asr.split_*`, `loom.asr.tokens_*`), read off the "
            "checkpoint's own `split_points` signature in load_model; `prompt_segment_constants` sizes "
            "the cache from them and checks the result fits"
        ),
    }

    def load_model(self):
        from transformers import AutoModel

        print(f"Loading model from {self.model_dir}...")
        # fp32 on the CPU with `merge=False`: the adapters stay separate, which is the point -- see the
        # module docstring. The checkpoint's loader merges only the token table (fp32 here).
        model = AutoModel.from_pretrained(
            self.model_dir, trust_remote_code=True, device="cpu", dtype=torch.float32, merge=False,
        ).eval()
        lm = model.model.language_model
        table = lm.token_table
        keep = table.keep.cpu().numpy().astype(np.int64)
        if not (np.diff(keep) > 0).all():
            raise ValueError(
                "token_table.keep is not sorted, so kept index order is not original id order and an "
                "argmax tie would break differently from the checkpoint's own head. The remap needs "
                "a sorted table, or a permutation this leaf does not carry."
            )
        self._keep = keep
        weight = table.weight(torch.float32).detach().clone()
        lm.embed_tokens = _KeptEmbedding(weight)
        model.lm_head = _KeptHead(weight)

        self._quiet_split = _quiet_split_policy(model)

        if self._kept_tokenizer_dir is None:
            staging = Path(tempfile.mkdtemp(prefix="oruk_loop_tokenizer_"))
            _write_kept_tokenizer(Path(self.model_dir), keep, staging)
            _check_kept_tokenizer(Path(self.model_dir), keep, staging)
            self._kept_tokenizer_dir = str(staging)
        return model

    def audio_encoder(self, model, mel) -> nn.Module:
        return LoopedWindowedAudioEncoder(
            mel, model.model.audio_tower, model.model.multi_modal_projector, model.config.audio_config,
            model.enc_cursor,
        )

    def _kept(self, token_id: int) -> int:
        """Original id -> kept index; a prompt or stop id the table dropped cannot be expressed."""
        index = int(np.searchsorted(self._keep, token_id))
        if index >= len(self._keep) or int(self._keep[index]) != int(token_id):
            raise ValueError(f"token id {token_id} is not in the kept vocabulary")
        return index

    def prompt_segment_constants(self, model) -> dict:
        """The stock leaf's constants with 5.17's forced-language pre-fill appended to the suffix, then
        every id mapped to its kept index.

        The base renders the template with no system text -- which is also 5.17's system turn, since
        the language no longer goes there -- so only the assistant pre-fill is added: its text, tokenized
        by the checkpoint's own tokenizer with no special tokens (`<asr_text>` is an added token and
        comes back as one id). `TRANSCRIPT_AFTER` becomes None: the marker is in the prompt, the model
        never emits it, and what it does emit is already the transcript.
        """
        constants = super().prompt_segment_constants(model)
        tokenizer = self._processor_for().tokenizer
        prefill = tokenizer.encode(f"language {_FORCED_LANGUAGE}<asr_text>", add_special_tokens=False)
        suffix = [*constants["AUDIO_SUFFIX"], *prefill]

        # THE CACHE HOLDS ONE PIECE. `transcribe` never hands this model more than `split_max_samples`
        # at once (the quiet-point scheme, `contract`), so a call's positions are that piece's audio
        # rows, the two prompt runs and the largest token budget -- 390 + 18 + 448 = 856 at the
        # checkpoint's numbers, rounded up to 1024. The template's 4096 was a cache of 940 MB at F32
        # for a 243 MB model; this is 235 MB.
        policy = self._quiet_split
        hop, chunk_frames, rows_per_chunk = self._row_formula
        mel = policy["split_max_samples"] // hop
        rows = (mel // chunk_frames) * rows_per_chunk
        rest = mel % chunk_frames
        for _ in range(3):
            rest = (rest - 1) // 2 + 1
        needed = rows + rest + len(constants["AUDIO_PREFIX"]) + len(suffix) + policy["tokens_cap"]
        self.max_seq_len = -(-needed // 256) * 256
        return {
            "EOS": self._kept(constants["EOS"]),
            "EOS_EXTRA": tuple(self._kept(e) for e in constants["EOS_EXTRA"]),
            "AUDIO_PREFIX": [self._kept(t) for t in constants["AUDIO_PREFIX"]],
            "AUDIO_SUFFIX": [self._kept(t) for t in suffix],
            "TRANSCRIPT_AFTER": None,
        }

    def contract(self) -> dict:
        """The stock ASR contract plus the reference's long-form decode -- cut at quiet points into
        pieces of at most 30 s, each decoded alone with its own token budget, texts joined. See
        loom.cpp `asr_long_form.h`, which implements it; the numbers are this checkpoint's."""
        contract = super().contract()
        for key, value in (self._quiet_split or {}).items():
            contract[f"asr.{key}"] = value
        return contract

    def backend_kwargs(self) -> dict:
        # The remapped tokenizer. NOT the remapped stop ids, though both are known here: the decode loop
        # has both bound already, and `write_bpe_vocab` would make the generation config's FIRST one --
        # `<|endoftext|>` -- the scalar `tokenizer.ggml.eos_token_id`, which is the one id loom.cpp's
        # `transcribe` strips from the text. The model ends its turn on `<|im_end|>`, so every
        # transcript came back ending in it. Left to the tokenizer config's `eos_token`, as the stock
        # leaf leaves it, the scalar is `<|im_end|>`.
        return dict(
            tokenizer_dir=self._kept_tokenizer_dir, hparams=self.hparams(),
            # The tied table is written twice, `embed.` and `lm_head.`, and only the head's copy is a
            # MUL_MAT operand; this quantizes the embedding's too, so the two alias (see the backend).
            quantize_gathered=True,
        )


def _quiet_split_policy(model) -> dict:
    """How the checkpoint's own `transcribe` cuts long audio, as loom.cpp's `transcribe` reads it.

    **The window numbers are read off its `split_points` signature** (`max_s=30.0, search_s=5.0,
    frame=0.1`, at `SAMPLE_RATE`), so an export follows the modeling code it was made against. The
    other four are literals inside `transcribe`'s body, where no signature can be read: a piece under
    1600 samples (`if len(piece) < 1600:  # <0.1 s tail`) is skipped, and each piece may generate
    `min(448, ceil(10 * dur) + 24)` tokens.
    """
    import inspect
    import sys

    module = sys.modules[type(model).__module__]
    rate = int(module.SAMPLE_RATE)
    defaults = {name: p.default for name, p in inspect.signature(module.split_points).parameters.items()
                if p.default is not inspect.Parameter.empty}
    if int(defaults["sr"]) != rate:
        raise ValueError(f"split_points defaults to {defaults['sr']} Hz but the model runs at {rate}")
    return {
        "split_max_samples": int(round(defaults["max_s"] * rate)),
        "split_search_samples": int(round(defaults["search_s"] * rate)),
        "split_frame_samples": int(round(defaults["frame"] * rate)),
        "split_min_samples": 1600,
        "tokens_per_second": 10.0,
        "tokens_base": 24,
        "tokens_cap": 448,
    }


def _is_oruk_loop(path: Path) -> bool:
    """A `qwen3_asr` directory in the transformers layout that declares the `oruk_loop` block and the
    class that reads it. Both, because the block is what the loader depends on and the class is what
    `trust_remote_code` will actually run."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_asr":
        return False
    return ("oruk_loop" in config and "audio_config" in config and "text_config" in config
            and "OrukLoopForConditionalGeneration" in (config.get("architectures") or []))


def _build_oruk_loop(path: Path, output_path: str) -> ASROrukLoopExportConfig:
    return ASROrukLoopExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASROrukLoopExportConfig,
        recognizers=[ModelRecognizer(
            name="oruk-loop", detect=_is_oruk_loop, build_config=_build_oruk_loop,
        )],
    ))
