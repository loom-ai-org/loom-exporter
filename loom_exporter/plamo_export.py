"""PLaMo 3 (pfnet; `plamo-3-translate-0.8b`): a causal LM with sliding-window attention, exported through
the causal-LM family with its own trace wrapper, its own tokenizer and a translation prompt.

What differs from a plain `AutoModelForCausalLM`, and why each needs this module:

* **Remote code.** `Plamo3ForCausalLM` ships beside the checkpoint (`trust_remote_code`).
* **Its RoPE is not traceable as written.** `Attention.forward` builds cos/sin from `arange(kv_len)` of
  its own cache and gathers them by position; under a trace that length is a constant. The wrapper
  computes them from the positions the driver passes (`cache_position`), with each layer's own
  `inv_freq` and YaRN `concentration` -- the full layers' YaRN table and the local layers' plain one.
  PLaMo renumbers a cropped sliding cache from 0 (its "local coordinate system"), which is the same
  attention: a rotary score depends only on the distance between query and key.
* **Its norms carry offsets** -- `(offset + w) * x / rms(x)`, offset 1 everywhere but the two post-norms
  (1/5 and 1/5^1.5) -- folded into the weight the export writes.
* **Its window layout is not in `config.json`'s HF spelling.** Every 8th layer (`sliding_window_pattern`)
  is full, the others see 2048 positions (`window_size`): the per-block widths the family's
  `attention_windows` takes, computed here rather than read off `layer_types`.
* **Its tokenizer** is PLaMo's own (`plamo_tokenizer_export`, loom::PlamoVocab), and its chat template
  is PFML3 with arguments a role-tag template cannot express -- so no chat template ships; the driver
  carries `translation_prompt` instead (`plamo_driver/translation_prompt.lua`, loom.cpp ADR-132).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .causal_lm_export import LMCausalModelExportConfig, _causal_mask, _hf_config
from .spec_protocol import Unchecked


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _Layer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        mixer = layer.mixer
        self.qkv = mixer.qkv_proj
        self.o = mixer.o_proj
        self.gate_up = layer.mlp.gate_up_proj
        self.down = layer.mlp.down_proj
        self.n_head, self.n_kv, self.head_dim = mixer.q_num_heads, mixer.k_num_heads, mixer.qk_dim
        self.split = [mixer.q_proj_dim, mixer.k_proj_dim, mixer.v_proj_dim]
        self.eps = float(config.rms_norm_eps)
        for name, norm in (("pre_mixer", layer.pre_mixer_norm), ("post_mixer", layer.post_mixer_norm),
                           ("pre_mlp", layer.pre_mlp_norm), ("post_mlp", layer.post_mlp_norm),
                           ("q_norm", mixer.q_norm), ("k_norm", mixer.k_norm)):
            self.register_buffer(name, (norm.offset + norm.weight.detach()).clone())
        self.register_buffer("inv_freq", mixer.rotary_emb.inv_freq.detach().float().clone())
        self.concentration = float(mixer.rotary_emb.concentration)

    def forward(self, h, positions, mask):
        b, s, _ = h.shape
        x = _rms(h, self.pre_mixer, self.eps)
        q, k, v = torch.split(self.qkv(x), self.split, dim=-1)
        q = _rms(q.view(b, s, self.n_head, self.head_dim).transpose(1, 2), self.q_norm, self.eps)
        k = _rms(k.view(b, s, self.n_kv, self.head_dim).transpose(1, 2), self.k_norm, self.eps)
        v = v.view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)          # [s, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.concentration).unsqueeze(0).unsqueeze(0)
        sin = (emb.sin() * self.concentration).unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        # HF's `repeat_kv` spelling, which `passes.fuse_gqa_repeat_kv` folds into the fused node's own
        # K/V head broadcast (MQA: one K/V head for sixteen queries).
        rep = self.n_head // self.n_kv
        k = k[:, :, None].expand(b, self.n_kv, rep, s, self.head_dim).reshape(b, self.n_head, s, self.head_dim)
        v = v[:, :, None].expand(b, self.n_kv, rep, s, self.head_dim).reshape(b, self.n_head, s, self.head_dim)
        scores = torch.matmul(q * (self.head_dim ** -0.5), k.transpose(-1, -2)) + mask
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        attn = self.o(ctx.transpose(1, 2).reshape(b, s, self.n_head * self.head_dim))
        h = h + _rms(attn, self.post_mixer, self.eps)
        g, u = self.gate_up(_rms(h, self.pre_mlp, self.eps)).chunk(2, dim=-1)
        return h + _rms(self.down(nn.functional.silu(g) * u), self.post_mlp, self.eps)


class _PlamoWrapper(nn.Module):
    """`(tokens, cache_position, attention_mask) -> logits`, the causal-LM family's trace signature."""

    def __init__(self, model):
        super().__init__()
        inner, config = model.model, model.config
        self.embed = inner.embed_tokens
        self.scale = float(config.hidden_size) ** 0.5 if getattr(config, "scale_embedding", False) else 1.0
        self.layers = nn.ModuleList(_Layer(layer, config) for layer in inner.layers.layers)
        self.register_buffer("norm", (inner.norm.offset + inner.norm.weight.detach()).clone())
        self.eps = float(config.rms_norm_eps)
        self.head = model.lm_head

    def forward(self, tokens, cache_position, attention_mask):
        h = self.embed(tokens)
        if self.scale != 1.0:
            h = h * self.scale
        for layer in self.layers:
            h = layer(h, cache_position, attention_mask)
        return self.head(_rms(h, self.norm, self.eps))


@dataclass(kw_only=True)
class PlamoExportConfig(LMCausalModelExportConfig):
    """A `pfnet/plamo-3-*` directory -> one Loom GGUF (the causal-LM family, Flattened)."""

    __unchecked__ = dict(LMCausalModelExportConfig.__unchecked__)

    def load_model(self):
        from transformers import AutoModelForCausalLM

        print(f"Loading PLaMo from {self.model_dir} (remote code)...")
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, trust_remote_code=True,
                                                     dtype=torch.float32).eval()
        self._resolved_architecture = self.architecture or "plamo3"
        return model

    def build_trace(self, model):
        wrapper, dummy, mil_inputs = super().build_trace(model)
        return _PlamoWrapper(model).eval(), dummy, mil_inputs

    def _attention_windows(self):
        cfg = _hf_config(Path(self.model_dir)) or {}
        n, pattern, window = (cfg.get("num_hidden_layers"), cfg.get("sliding_window_pattern"),
                              cfg.get("window_size"))
        if not (n and pattern and window):
            return None
        # `is_full_attn(pattern, i)`: layer i is full when (i + 1) % pattern == 0.
        return [0 if (i + 1) % int(pattern) == 0 else int(window) for i in range(int(n))]

    def backend_kwargs(self) -> dict:
        kwargs = super().backend_kwargs()
        # PFML3 is a template with ARGUMENTS (target_lang, glossary, ...) that the role/prefix/suffix
        # chat template cannot carry; the driver's `translation_prompt` is that template instead.
        kwargs["chat_template"] = False
        kwargs["tokenizer_family"] = "plamo"
        source = self.tokenizer_dir or self.model_dir
        # The component registry builds every config over a placeholder path: nothing to read there.
        if Path(source).is_dir():
            kwargs["driver_appendix"] = translation_prompt_lua(source)
        return kwargs


TAGS = ("bos", "tag", "msg", "key", "val", "index", "item")


def translation_prompt_lua(tokenizer_dir: str) -> str:
    """The driver's `translation_prompt`, with the PFML3 tag ids it needs bound above it."""
    from .plamo_tokenizer_export import read_plamo_vocab

    pieces = read_plamo_vocab(tokenizer_dir)["pieces"]
    ids = {}
    for tag in TAGS:
        spelled = f"<|plamo:{tag}|>"
        if spelled not in pieces:
            raise ValueError(f"PLaMo's vocabulary has no {spelled}; the PFML3 prompt cannot be built")
        ids[tag] = pieces.index(spelled)
    table = ", ".join(f"{k} = {v}" for k, v in ids.items())
    body = (Path(__file__).resolve().parent / "plamo_driver" / "translation_prompt.lua").read_text()
    return f"local PLAMO_TAG_IDS = {{{table}}}\n" + body


def _is_plamo(path: Path) -> bool:
    cfg = _hf_config(path)
    return cfg is not None and cfg.get("model_type") == "plamo3"


def _build_plamo(path: Path, output_path: str):
    from .decomposition import Flattened

    return PlamoExportConfig(output_path=output_path, decomposition=Flattened(), model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-generation",
        config_class=PlamoExportConfig,
        recognizers=[ModelRecognizer(name="plamo3", detect=_is_plamo, build_config=_build_plamo)],
    ))
