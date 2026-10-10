"""sanoTTS -- Ampixa's tiny distilled TTS voices (`ampixa/sanoTTS`, 0.5M-2.3M parameters): the
**piperlite** line, a duration net, an acoustic net and a HiFi-GAN-shaped decoder distilled from a
Piper/VITS teacher at 22.05 kHz, one package per voice.

**The checkpoint is a package, not a torch file.** Each voice directory is `manifest.json` (each
component's config and every tensor's name, shape and byte offset) plus `weights.fp16.bin`, one flat
fp16 blob; upstream ships no `.pt`. The tensor names are the `state_dict` of the torch modules in
upstream's GPL-3.0 training scripts (`tools/train_roota_piper_{duration,latent,decoder}_student.py`),
so the export builds those modules from a pinned clone of the repository on `sys.path` -- never
vendored here -- and loads the blob into them `strict=True`. Upstream's own C-runtime golden exporters
load the same scripts the same way. The exported GGUF carries the voice's weights and is GPL-3.0 like
them.

The reference is upstream's torch path end to end: `predict_durations` (`round(exp(log_d).clamp_min(1)
* length_scale)`, clamped to `max_duration`), `expand_features`, the latent student, the decoder. The
pip package's numpy runtime matches it to 1.3e-5 on amy and is what upstream ships; it is a port of it.

What runs where:

* **`duration`** (graph) -- ids and three per-token features (position, length hint, valid) to the
  per-token log-duration.
* **The driver** -- the ids clamped into each net's vocabulary, the features, the rounding (half to
  even, which is `torch.round`'s), and the expansion's index and per-frame features. These are counts
  and positions over a data-dependent frame count, which stays host-side; the float32 the reference
  computes them in is emulated (`to_f32`) where a rounding decides an integer.
* **`synth`** (graph) -- the acoustic net's token stack, the token-to-frame expansion as a gather on a
  host-built index, its frame stack, and the decoder, to the waveform: 256 samples per frame.

**Two vocabulary details upstream's runtimes decide and its training code does not.** An id at or past
a net's vocabulary is remapped to 59 (schwa), as the pip package, the ESP32 and the WASM ports all do
(the torch modules would index out of bounds; the duration and acoustic nets can differ in size, so
the clamp is per net). And the phoneme framing is piper-phonemize's, `[BOS, blank, p1, blank, ...,
pn, blank, EOS]`, a blank right after BOS -- what piper1-gpl's `phonemes_to_ids` builds and the
students were distilled on, and one id longer than Piper's old python runtime (which the VITS export
follows). Declared as `blank_after_bos`.

Not reproduced: upstream's optional "sibilant injection" (inference-time fricative noise, beta 0.6 in
three packages) is applied by its dashboard and its Arduino runtime but not by the pip package, which
is the documented Python path; this export follows the pip package.
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked

# The upstream checkout whose training scripts define the modules. A git clone, never vendored: the
# scripts are GPL-3.0 and this repository is MIT. Pinned, because the modules ARE the architecture and
# a moved script is a different export.
SANOTTS_REPO = os.environ.get("LOOM_SANOTTS_REPO", "/home/flavio/Dev/sanoTTS")
SANOTTS_COMMIT = "50a9c82235b4faf8d4191b225e3c79f2d9734574"

# What an id outside a net's vocabulary becomes: schwa, upstream runtimes' own fallback.
FALLBACK_ID = 59
# The decoder's three transposed convolutions, 8 x 8 x 4: waveform samples per acoustic frame.
HOP = 256
# The decoder-config keys `DecoderStudent` takes, as upstream's `export_piperlite_golden.py` lists them.
_DECODER_KEYS = (
    "in_channels", "channels", "res_layers", "variant", "rank_ratio", "activation", "stage_affine",
    "factorized_pre_rank", "piper_res_factor_rank_ratio", "res_bank_scale_mode", "stage0_branches",
    "stage1_branches", "stage2_branches", "stage3_branches", "post_filter_channels", "post_filter_layers",
    "post_filter_kernel", "post_filter_scale", "stage_projection_bottlenecks",
)


def _check_clone(repo: Path) -> None:
    if not (repo / "tools" / "train_roota_piper_decoder_student.py").is_file():
        raise FileNotFoundError(
            f"sanoTTS: no checkout at {repo}. The export builds upstream's own modules from it: "
            f"`git clone https://github.com/Ampixa/sanoTTS {repo}` and check out {SANOTTS_COMMIT[:12]} "
            f"(or point LOOM_SANOTTS_REPO at one).")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != SANOTTS_COMMIT:
        raise RuntimeError(f"sanoTTS: {repo} is at {head[:12] or '?'}, the export is pinned to "
                           f"{SANOTTS_COMMIT[:12]}. Check that commit out, or move the pin after "
                           f"re-verifying the export against the new one.")


def load_sanotts_modules(repo: Optional[str] = None):
    """`(duration, latent, decoder)` -- upstream's three training scripts, imported the way its own
    golden exporters import them (`spec_from_file_location`). `tools/` goes on `sys.path` too, because
    the latent and decoder scripts import their siblings (`qat_ste`, `roota_fsd_blocks`)."""
    import importlib.util
    import pathlib

    repo = Path(repo or SANOTTS_REPO)
    _check_clone(repo)
    # Upstream's checkpoints were pickled on Windows; its loaders do this too. Harmless here, where no
    # pickle is read, but the scripts' module scope expects it.
    pathlib.WindowsPath = pathlib.PosixPath
    tools = str(repo / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    modules = []
    for name, script in (("sanotts_duration_student", "train_roota_piper_duration_student.py"),
                         ("sanotts_latent_student", "train_roota_piper_latent_student.py"),
                         ("sanotts_decoder_student", "train_roota_piper_decoder_student.py")):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, repo / "tools" / script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        modules.append(sys.modules[name])
    return tuple(modules)


def read_package(package_dir: Path) -> tuple:
    """`(manifest, {component: state_dict})` -- the fp16 blob sliced by the manifest's offsets and
    widened to f32, the dtype the modules are built in."""
    manifest = json.loads((package_dir / "manifest.json").read_text())
    if manifest.get("format") != "roota.raw-fp16.v1":
        raise ValueError(f"sanoTTS: {package_dir} is format {manifest.get('format')!r}; only the "
                         f"piperlite package (`roota.raw-fp16.v1`) is read here.")
    blob = (package_dir / manifest["weights_file"]).read_bytes()
    if len(blob) != int(manifest["weights_size_bytes"]):
        raise ValueError(f"sanoTTS: {manifest['weights_file']} is {len(blob)} bytes, the manifest says "
                         f"{manifest['weights_size_bytes']}.")
    states = {}
    for component in ("duration", "acoustic", "decoder"):
        state = {}
        for t in manifest["components"][component]["tensors"]:
            if t["dtype"] != "float16":
                raise ValueError(f"sanoTTS: {component}.{t['name']} is {t['dtype']}, not float16.")
            count = int(np.prod(t["shape"])) if t["shape"] else 1
            array = np.frombuffer(blob, dtype=np.float16, count=count, offset=int(t["offset_bytes"]))
            state[t["name"]] = torch.from_numpy(array.astype(np.float32).reshape(t["shape"]))
        states[component] = state
    return manifest, states


def build_models(package_dir: Path, repo: Optional[str] = None) -> tuple:
    """`(manifest, duration, latent, decoder)`, each upstream's own module with the package's weights,
    in eval mode. Refuses any architecture the C ports also refuse."""
    duration_mod, latent_mod, decoder_mod = load_sanotts_modules(repo)
    manifest, states = read_package(package_dir)
    components = manifest["components"]
    dc, ac, xc = (components[c]["config"] for c in ("duration", "acoustic", "decoder"))
    if dc.get("architecture") != "duration_conv":
        raise ValueError(f"sanoTTS: duration architecture {dc.get('architecture')!r}; only duration_conv.")
    if ac.get("architecture") != "token_context":
        raise ValueError(f"sanoTTS: acoustic architecture {ac.get('architecture')!r}; only token_context "
                         f"(no shipped piperlite voice carries a calibrated adapter).")
    if xc.get("variant") != "piperlite":
        raise ValueError(f"sanoTTS: decoder variant {xc.get('variant')!r}; only piperlite.")

    duration = duration_mod.DurationStudent(
        vocab_size=dc["vocab_size"], hidden=dc["hidden"], depth=dc["depth"],
        kernel_size=dc["kernel_size"], max_tokens=dc["max_tokens"])
    latent = latent_mod.ContextualLatentStudent(
        vocab_size=ac["vocab_size"], hidden=ac["hidden"], depth=ac["depth"],
        token_depth=ac["token_depth"], kernel_size=ac.get("kernel_size", 5),
        out_channels=ac["out_channels"])
    decoder = decoder_mod.DecoderStudent(**{k: v for k, v in xc.items() if k in _DECODER_KEYS})
    for module, component in ((duration, "duration"), (latent, "acoustic"), (decoder, "decoder")):
        module.load_state_dict(states[component], strict=True)
        module.eval()
    return manifest, duration, latent, decoder


class _DurationWrapper(nn.Module):
    """`(tokens [1, n], features [1, 3, n]) -> log_duration [1, n]`: `DurationStudent.forward` with the
    features the host computed and the mask all-valid (one sequence), under which every
    `ResidualConvBlock`'s mask product is the identity."""

    def __init__(self, duration):
        super().__init__()
        self.embedding = duration.embedding
        self.input_proj = duration.input_proj
        self.blocks = duration.blocks
        self.output = duration.output

    def forward(self, tokens, features):
        x = self.input_proj(torch.cat([self.embedding(tokens).transpose(1, 2), features], dim=1))
        for block in self.blocks:
            x = x + block.scale * block.net(x)
        return self.output(x).squeeze(1)


class _SynthWrapper(nn.Module):
    """`(tokens [1, n], token_features [1, 2, n], frame_index [T], frame_features [1, 3, T]) ->
    waveform [1, 256 T]`: `ContextualLatentStudent.forward` then `DecoderStudent.forward`, with the
    token-to-frame `repeat_interleave` spelled as a gather on the index the host built from the same
    durations."""

    def __init__(self, latent, decoder):
        super().__init__()
        self.embedding = latent.embedding
        self.token_input_proj = latent.token_input_proj
        self.token_blocks = latent.token_blocks
        self.frame_input_proj = latent.frame_input_proj
        self.frame_blocks = latent.frame_blocks
        self.output = latent.output
        self.decoder = decoder

    def mel(self, tokens, token_features, frame_index, frame_features):
        """The acoustic student alone: `[1, out_channels, T]` -- piperlite's latent, nano's mel-100."""
        x = self.token_input_proj(torch.cat([self.embedding(tokens).transpose(1, 2), token_features], dim=1))
        for block in self.token_blocks:
            x = block(x)
        frames = torch.index_select(x.squeeze(0).transpose(0, 1), 0, frame_index)
        x = self.frame_input_proj(torch.cat([frames.transpose(0, 1).unsqueeze(0), frame_features], dim=1))
        for block in self.frame_blocks:
            x = block(x)
        return self.output(x)

    def forward(self, tokens, token_features, frame_index, frame_features):
        return self.decoder(self.mel(tokens, token_features, frame_index, frame_features)).reshape(1, -1)


def _bcp47(code: str) -> str:
    """`en_US` (the manifest's) -> `en-US` (what a phonemizer's language argument takes)."""
    return code.replace("_", "-")


@dataclass(kw_only=True)
class TTSSanoTTSExportConfig(BaseMultiPhaseModelExportConfig):
    """One sanoTTS piperlite voice package -> one GGUF: `duration` and `synth`, and the driver between."""

    package_dir: str = ""
    architecture: str = "sanotts-piperlite"
    output_path: str = "sanotts.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "sanotts_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    trace_tokens: int = 23
    trace_frames: int = 61
    max_tokens: int = 2048
    max_frames: int = 16384

    manifest: Optional[dict] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "package_dir": Unchecked("a voice directory; `read_package` checks the format, the blob's size "
                                 "and every tensor's dtype, and `load_state_dict(strict=True)` the rest"),
        "architecture": Unchecked("the GGUF's own architecture string"),
        "output_path": Unchecked("where to write"),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "trace_tokens": Unchecked("a property of the TRACE, not of the model"),
        "trace_frames": Unchecked("same"),
        "max_tokens": Unchecked("the token axis's declared upper bound; the nets are convolutional and "
                                "take any length"),
        "max_frames": Unchecked("the frame axis's declared upper bound (16384 frames = 190 s at 22.05 "
                                "kHz); the same"),
        "manifest": Unchecked("READ off the package"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        manifest, duration, latent, decoder = build_models(Path(self.package_dir))
        self.manifest = manifest
        if int(manifest["hop_length"]) != HOP:
            raise ValueError(f"sanoTTS: hop_length {manifest['hop_length']}; the decoder upsamples by {HOP}.")
        dc = manifest["components"]["duration"]["config"]
        ac = manifest["components"]["acoustic"]["config"]
        n, t = int(self.trace_tokens), int(self.trace_frames)
        token_axis = ct.RangeDim(1, self.max_tokens)
        frame_axis = ct.RangeDim(1, self.max_frames)
        g = torch.Generator().manual_seed(0)
        return [
            ExportPhase(
                name="duration", wrapper=_DurationWrapper(duration).eval(),
                dummy_inputs=(torch.randint(0, int(dc["vocab_size"]), (1, n), generator=g),
                              torch.rand(1, 3, n, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="features", shape=(1, 3, token_axis), dtype=np.float32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="synth", wrapper=_SynthWrapper(latent, decoder).eval(),
                dummy_inputs=(torch.randint(0, int(ac["vocab_size"]), (1, n), generator=g),
                              torch.rand(1, 2, n, generator=g),
                              torch.sort(torch.randint(0, n, (t,), generator=g)).values,
                              torch.rand(1, 3, t, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="token_features", shape=(1, 2, token_axis), dtype=np.float32),
                            ct.TensorType(name="frame_index", shape=(frame_axis,), dtype=np.int32),
                            ct.TensorType(name="frame_features", shape=(1, 3, frame_axis), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"tokens": {1: "n_tokens"}, "token_features": {2: "n_tokens"}},
            ),
        ]

    def hparams(self) -> dict:
        if not self.manifest:
            return {}
        return {"sample_rate": int(self.manifest["sample_rate"])}

    def contract(self) -> dict:
        contract = super().contract()
        contract["text.frontend"] = "phonemes"
        contract["text.phoneme_alphabet"] = "ipa"
        if self.manifest and self.manifest.get("language"):
            contract["text.languages"] = [_bcp47(self.manifest["language"])]
        return contract

    def phoneme_table(self) -> dict:
        """The voice's own `piper-phoneme-config.json`, read as the VITS export reads Piper's, with
        piper-phonemize's framing (see the module docstring)."""
        config = json.loads((Path(self.package_dir) / "piper-phoneme-config.json").read_text())
        id_map = config["phoneme_id_map"]
        multi = {sym: ids for sym, ids in id_map.items() if len(ids) != 1}
        if multi:
            raise ValueError(f"sanoTTS: phoneme_id_map maps {len(multi)} symbols to several ids "
                             f"({list(multi)[:4]}); one id per symbol is what the vocabulary models.")
        framing = {"^": 1, "$": 2, "_": 0}
        for symbol, want in framing.items():
            if id_map.get(symbol) != [want]:
                raise ValueError(f"sanoTTS: {symbol!r} is {id_map.get(symbol)} in the phoneme map, not "
                                 f"[{want}] -- the framing ids upstream's front end hardcodes.")
        symbols = sorted(id_map, key=lambda sym: id_map[sym][0])
        return {"symbols": symbols, "ids": [id_map[sym][0] for sym in symbols],
                "bos": 1, "eos": 2, "blank": 0, "interleave_blank": True, "blank_after_bos": True}

    def driver_components(self) -> List:
        from .driver_components import (
            DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import Lit, Var
        from .lua_library import LuaLibrary

        components = (self.manifest or {}).get("components", {})
        dc = components.get("duration", {}).get("config", {})
        ac = components.get("acoustic", {}).get("config", {})
        inference = (self.manifest or {}).get("inference", {})
        fragment = self.driver_script_path
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            LuaLibrary(uses=("round_half_to_even", "to_f32")),
            ExportConstants(values={
                "DUR_VOCAB": int(dc.get("vocab_size", 0)),
                "AC_VOCAB": int(ac.get("vocab_size", 0)),
                "FALLBACK_ID": FALLBACK_ID,
                "MAX_TOKENS": int(dc.get("max_tokens", 0)),
                "MAX_DURATION": int(dc.get("max_duration", 0)),
                "LENGTH_SCALE": float(inference.get("duration_length_scale", 1.0)),
            }),
            LuaFragment(fragment / "01_tokens.lua",
                        reads=("DUR_VOCAB", "AC_VOCAB", "FALLBACK_ID", "MAX_TOKENS"),
                        defines=("_n", "_dur_tokens", "_dur_features", "_ac_tokens", "linspace01")),
            SubgraphCallComponent(
                topology="duration", outputs=("_log_duration",),
                inputs={"tokens": Var("_dur_tokens"), "features": Var("_dur_features")},
                axes={"n_tokens": Var("_n"), "n_past": Lit(0)},
                note="Duration net: per-token log-duration."),
            LuaFragment(fragment / "02_expand.lua",
                        reads=("_n", "_log_duration", "LENGTH_SCALE", "MAX_DURATION", "linspace01"),
                        defines=("_n_frames", "_token_features", "_frame_index", "_frame_features")),
            SubgraphCallComponent(
                topology="synth", outputs=("waveform",),
                inputs={"tokens": Var("_ac_tokens"), "token_features": Var("_token_features"),
                        "frame_index": Var("_frame_index"), "frame_features": Var("_frame_features")},
                axes={"n_tokens": Var("_n"), "n_enc_frames": Var("_n_frames"), "n_past": Lit(0)},
                note="Acoustic net (token stack, expansion, frame stack) and decoder: the waveform."),
            DriverReturn(values=("waveform",)),
        ]



# ---------------------------------------------------------------------------------------------------------
# The NANO line (heart, heart-nano): the same duration and acoustic students, emitting mel-100, and a
# Vocos-shaped decoder -- ConvNeXt1D, a magnitude/phase head, iSTFT -- fed four channels of noise.
#
# **The decoder's torch module was never published** (upstream's `train_tiny_vocos_student.py` is in no
# commit of the repository), so `_TinyVocos` below is this export's own, written to the definition
# upstream DOES publish: the pip package's numpy runtime (`pypkg/sanotts/nano.py`), which upstream gates
# against its PyTorch reference. The user's call (2026-10-10), the Silero precedent: published weights
# plus a plain-Python definition. The front end's modules are upstream's own torch classes, as for
# piperlite.
#
# **The weights are a C runtime's blobs**: `front_*.bin` / `model_*.bin` addressed by the generated
# `nano_q8_meta.h` -- per layer an `_W8` region of rows padded to 16 bytes, an `_SCALE` and a `_BIAS`, or
# one `_F32` region. heart ships float32 rows (`NANO_WEIGHT_FORMAT 1`, unit scales); heart-nano ships
# int8 with a per-row scale, which is DEQUANTISED here, as the pip package does: its C runtime also
# quantises activations, which a float graph does not reproduce, and upstream gates both against the same
# float reference rather than against each other.
# ---------------------------------------------------------------------------------------------------------

NANO_LAYER_NORM_EPS = 1e-6   # upstream's `make_norm`, NOT torch's 1e-5 (nano.py: "cost 0.06 of correlation")
NANO_MAG_MAX = 1e2           # the head's exp(magnitude) is clipped here
# The DC blocker after the iSTFT: H(z) = (1 - z^-1) / (1 - R z^-1), which upstream applies as a 4096-tap
# truncation of its impulse response. The driver runs the recursion itself; the truncated tail is below
# R^4096 = 1.6e-5 of the signal.
NANO_DC_BLOCK_R = 0.9973
# Upstream's default seed (`engine.DEFAULT_NANO_SEED`); ATen keeps its low 32 bits.
NANO_DEFAULT_SEED = 2236265385529901705 & 0xFFFFFFFF


def _parse_nano_header(path: Path) -> dict:
    import re

    out = {}
    pattern = re.compile(r"^#define\s+([A-Z0-9_]+)\s+(-?\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    if "NANO_FRONT_BYTES" not in out:
        raise ValueError(f"sanoTTS: {path} is not a nano offsets header")
    return out


class _NanoBlobs:
    """One nano package's two blobs and its header: a region by name, as float32."""

    def __init__(self, package_dir: Path):
        self.meta_json = json.loads((package_dir / "meta.json").read_text())
        self.h = _parse_nano_header(package_dir / "nano_q8_meta.h")
        self.front = (package_dir / self.meta_json["front"]).read_bytes()
        self.dec = (package_dir / self.meta_json["dec"]).read_bytes()
        for blob, key in ((self.front, "NANO_FRONT_BYTES"), (self.dec, "NANO_DEC_BYTES")):
            if len(blob) != self.h[key]:
                raise ValueError(f"sanoTTS: a nano blob is {len(blob)} bytes, the header says {self.h[key]}.")
        if self.h.get("NANO_NORM_TYPE", 0) != 0 or self.h.get("NANO_ACT_TYPE", 0) != 0:
            raise ValueError("sanoTTS: a DyT/ReLU nano decoder (NANO_NORM_TYPE/NANO_ACT_TYPE != 0); only "
                             "LayerNorm + GELU, the shipped voices' operators, is reproduced.")

    def rows(self, blob: str, prefix: str, name: str, out_ch: int, in_flat: int) -> tuple:
        """`(weight [out_ch, in_flat], bias [out_ch])` of one row-major layer, dequantised."""
        h, data = self.h, (self.front if blob == "front" else self.dec)
        n16 = h[f"NANO_{name}_N16"]
        w_off, s_off, b_off = (h[f"{prefix}_{name}_{k}"] for k in ("W8", "SCALE", "BIAS"))
        if h.get("NANO_WEIGHT_FORMAT", 0) == 1:
            rows = np.frombuffer(data, np.float32, out_ch * n16, w_off).reshape(out_ch, n16)
            weight = rows[:, :in_flat].astype(np.float32)
        else:
            q = np.frombuffer(data, np.int8, out_ch * n16, w_off).reshape(out_ch, n16)
            scale = np.frombuffer(data, np.float32, out_ch, s_off)
            weight = q[:, :in_flat].astype(np.float32) * scale[:, None]
        bias = np.frombuffer(data, np.float32, out_ch, b_off).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(weight)), torch.from_numpy(bias.copy())

    def f32(self, blob: str, prefix: str, name: str, count: int) -> torch.Tensor:
        data = self.front if blob == "front" else self.dec
        return torch.from_numpy(np.frombuffer(data, np.float32, count, self.h[f"{prefix}_{name}_F32"]).copy())


def _load_conv(conv: nn.Conv1d, weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Rows are torch's `[C_out, C_in, K]` flattened, kernel fastest -- the device kernel's layout."""
    conv.weight.data.copy_(weight.reshape(conv.weight.shape))
    conv.bias.data.copy_(bias)


def _load_front(blobs: _NanoBlobs, duration, latent) -> None:
    h = blobs.h
    V = h["NANO_VOCAB"]
    DH, DK = h["NANO_DUR_HIDDEN"], h["NANO_DUR_KERNEL"]
    duration.embedding.weight.data.copy_(blobs.f32("front", "NOFF", "DUR_EMB", V * DH).reshape(V, DH))
    _load_conv(duration.input_proj, *blobs.rows("front", "NOFF", "DUR_PROJ", DH, DH + 3))
    for b, block in enumerate(duration.blocks):
        _load_conv(block.net[0], *blobs.rows("front", "NOFF", f"DUR_B{b}_C0", DH, DH * DK))
        _load_conv(block.net[2], *blobs.rows("front", "NOFF", f"DUR_B{b}_C1", DH, DH * DK))
        block.scale.data.copy_(blobs.f32("front", "NOFF", f"DUR_B{b}_SCALE", 1)[0])
    _load_conv(duration.output, *blobs.rows("front", "NOFF", "DUR_OUT", 1, DH))

    AH, AK = h["NANO_AC_HIDDEN"], h["NANO_AC_KERNEL"]
    latent.embedding.weight.data.copy_(blobs.f32("front", "NOFF", "AC_EMB", V * AH).reshape(V, AH))
    _load_conv(latent.token_input_proj, *blobs.rows("front", "NOFF", "AC_TPROJ", AH, AH + 2))
    for b, block in enumerate(latent.token_blocks):
        _load_conv(block.net[0], *blobs.rows("front", "NOFF", f"AC_TB{b}_C0", AH, AH * AK))
        _load_conv(block.net[2], *blobs.rows("front", "NOFF", f"AC_TB{b}_C1", AH, AH * AK))
        block.scale.data.copy_(blobs.f32("front", "NOFF", f"AC_TB{b}_SCALE", 1)[0])
    _load_conv(latent.frame_input_proj, *blobs.rows("front", "NOFF", "AC_FPROJ", AH, AH + 3))
    for b, block in enumerate(latent.frame_blocks):
        _load_conv(block.net[0], *blobs.rows("front", "NOFF", f"AC_FB{b}_C0", AH, AH * AK))
        _load_conv(block.net[2], *blobs.rows("front", "NOFF", f"AC_FB{b}_C1", AH, AH * AK))
        block.scale.data.copy_(blobs.f32("front", "NOFF", f"AC_FB{b}_SCALE", 1)[0])
    _load_conv(latent.output, *blobs.rows("front", "NOFF", "AC_OUT", h["NANO_MELS"], AH))


class _TinyVocos(nn.Module):
    """The nano decoder: `(mel [1, 100, T], noise [1, 4, T]) -> waveform [1, 256 T]`, as `nano.py`'s
    `decoder_forward` and `istft` compute it."""

    def __init__(self, blobs: _NanoBlobs):
        super().__init__()
        from .istft import ISTFT

        h = blobs.h
        dim, ek, dk, hid = h["NANO_DIM"], h["NANO_EMBED_KERNEL"], h["NANO_DW_KERNEL"], h["NANO_PW_HIDDEN"]
        mels, noise_ch, bins = h["NANO_MELS"], h["NANO_NOISE_CH"], h["NANO_BINS"]
        if h["NANO_HEAD_OUT"] != 2 * bins or bins != h["NANO_N_FFT"] // 2 + 1:
            raise ValueError("sanoTTS: the nano head is not [magnitude; phase] over n_fft/2 + 1 bins.")
        self.bins = bins

        def norm(prefix):
            ln = nn.LayerNorm(dim, eps=NANO_LAYER_NORM_EPS)
            ln.weight.data.copy_(blobs.f32("dec", "DOFF", f"{prefix}_W", dim))
            ln.bias.data.copy_(blobs.f32("dec", "DOFF", f"{prefix}_B", dim))
            return ln

        def linear(name, out_f, in_f):
            lin = nn.Linear(in_f, out_f)
            w, b = blobs.rows("dec", "DOFF", name, out_f, in_f)
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            return lin

        self.embed = nn.Conv1d(mels, dim, ek, padding=ek // 2)
        _load_conv(self.embed, *blobs.rows("dec", "DOFF", "EMBED", dim, mels * ek))
        self.noise = nn.Conv1d(noise_ch, dim, ek, padding=ek // 2)
        _load_conv(self.noise, *blobs.rows("dec", "DOFF", "NOISE", dim, noise_ch * ek))
        self.norm = norm("NORM")
        self.dw, self.block_norm, self.pw0, self.pw1 = (nn.ModuleList() for _ in range(4))
        gammas = []
        for b in range(h["NANO_BLOCKS"]):
            dw = nn.Conv1d(dim, dim, dk, padding=dk // 2, groups=dim)
            dw.weight.data.copy_(blobs.f32("dec", "DOFF", f"B{b}_DW_W", dim * dk).reshape(dim, 1, dk))
            dw.bias.data.copy_(blobs.f32("dec", "DOFF", f"B{b}_DW_B", dim))
            self.dw.append(dw)
            self.block_norm.append(norm(f"B{b}_NORM"))
            self.pw0.append(linear(f"B{b}_PW0", hid, dim))
            self.pw1.append(linear(f"B{b}_PW1", dim, hid))
            gammas.append(blobs.f32("dec", "DOFF", f"B{b}_GAMMA", dim))
        self.register_buffer("gamma", torch.stack(gammas))
        self.final_norm = norm("FNORM")
        self.head = linear("HEAD", 2 * bins, dim)
        # Bin 0 and Nyquist are zeroed: upstream's magnitude/phase parametrisation collapses there.
        edges = torch.ones(1, bins, 1)
        edges[0, 0, 0] = edges[0, -1, 0] = 0.0
        self.register_buffer("edges", edges)
        self.istft = ISTFT(n_fft=h["NANO_N_FFT"], hop_length=h["NANO_HOP"], win_length=h["NANO_N_FFT"],
                           center=True)

    def forward(self, mel, noise):
        x = self.embed(mel) + self.noise(noise)                       # [1, dim, T]
        x = self.norm(x.transpose(1, 2))                              # [1, T, dim]
        for b in range(len(self.dw)):
            h = self.dw[b](x.transpose(1, 2)).transpose(1, 2)
            h = self.pw1[b](F.gelu(self.pw0[b](self.block_norm[b](h))))
            x = x + h * self.gamma[b]
        out = self.head(self.final_norm(x)).transpose(1, 2)           # [1, 2 bins, T]
        mag = torch.clamp(torch.exp(out[:, : self.bins]), max=NANO_MAG_MAX) * self.edges
        phase = out[:, self.bins:]
        return self.istft(mag * torch.cos(phase), mag * torch.sin(phase))


class _NanoSynthWrapper(nn.Module):
    """`(tokens, token_features, frame_index, frame_features, noise [1, 4, T]) -> waveform [1, 256 T]`:
    the acoustic student to mel-100, the expansion a gather as in `_SynthWrapper`, then `_TinyVocos`."""

    def __init__(self, latent, decoder):
        super().__init__()
        self.acoustic = _SynthWrapper(latent, decoder=None)
        self.decoder = decoder

    def forward(self, tokens, token_features, frame_index, frame_features, noise):
        return self.decoder(self.acoustic.mel(tokens, token_features, frame_index, frame_features), noise)


def build_nano_models(package_dir: Path, repo: Optional[str] = None) -> tuple:
    """`(blobs, duration, latent, decoder)`: upstream's front-end modules and `_TinyVocos`, loaded."""
    duration_mod, latent_mod, _ = load_sanotts_modules(repo)
    blobs = _NanoBlobs(package_dir)
    h = blobs.h
    duration = duration_mod.DurationStudent(
        vocab_size=h["NANO_VOCAB"], hidden=h["NANO_DUR_HIDDEN"], depth=h["NANO_DUR_DEPTH"],
        kernel_size=h["NANO_DUR_KERNEL"], max_tokens=h["NANO_DUR_MAX_TOKENS"])
    latent = latent_mod.ContextualLatentStudent(
        vocab_size=h["NANO_VOCAB"], hidden=h["NANO_AC_HIDDEN"], depth=h["NANO_AC_DEPTH"],
        token_depth=h["NANO_AC_TOKEN_DEPTH"], kernel_size=h["NANO_AC_KERNEL"], out_channels=h["NANO_MELS"])
    _load_front(blobs, duration, latent)
    decoder = _TinyVocos(blobs)
    for module in (duration, latent, decoder):
        module.eval()
    return blobs, duration, latent, decoder


def nano_vocabulary(blobs: _NanoBlobs, repo: Optional[str] = None) -> dict:
    """The package's symbol table: its own when `meta.json` carries one, else upstream's frozen default
    -- `nano_frontend.vocabulary_for`, read from the clone rather than copied here."""
    clone = Path(repo or SANOTTS_REPO)
    pypkg = str(clone / "pypkg")
    if pypkg not in sys.path:
        sys.path.insert(0, pypkg)
    from sanotts.nano_frontend import vocabulary_for

    vocab = vocabulary_for(blobs.meta_json)
    if len(vocab) != blobs.h["NANO_VOCAB"]:
        raise ValueError(f"sanoTTS: the vocabulary has {len(vocab)} symbols, the header {blobs.h['NANO_VOCAB']}.")
    return vocab


@dataclass(kw_only=True)
class TTSSanoNanoExportConfig(TTSSanoTTSExportConfig):
    """One sanoTTS nano voice package (heart, heartnano) -> one GGUF."""

    architecture: str = "sanotts-nano"
    driver_script_path: Path = Path(__file__).resolve().parent / "sanotts_nano_driver"

    blobs: Optional[object] = field(default=None, init=False, repr=False)

    __unchecked__ = dict(TTSSanoTTSExportConfig.__unchecked__, blobs=Unchecked("READ off the package"))

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        blobs, duration, latent, decoder = build_nano_models(Path(self.package_dir))
        self.blobs = blobs
        h = blobs.h
        n, t = int(self.trace_tokens), int(self.trace_frames)
        token_axis = ct.RangeDim(1, self.max_tokens)
        frame_axis = ct.RangeDim(1, self.max_frames)
        g = torch.Generator().manual_seed(0)
        return [
            ExportPhase(
                name="duration", wrapper=_DurationWrapper(duration).eval(),
                dummy_inputs=(torch.randint(0, h["NANO_VOCAB"], (1, n), generator=g),
                              torch.rand(1, 3, n, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="features", shape=(1, 3, token_axis), dtype=np.float32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="synth", wrapper=_NanoSynthWrapper(latent, decoder).eval(),
                dummy_inputs=(torch.randint(0, h["NANO_VOCAB"], (1, n), generator=g),
                              torch.rand(1, 2, n, generator=g),
                              torch.sort(torch.randint(0, n, (t,), generator=g)).values,
                              torch.rand(1, 3, t, generator=g),
                              torch.randn(1, h["NANO_NOISE_CH"], t, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="token_features", shape=(1, 2, token_axis), dtype=np.float32),
                            ct.TensorType(name="frame_index", shape=(frame_axis,), dtype=np.int32),
                            ct.TensorType(name="frame_features", shape=(1, 3, frame_axis), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(1, h["NANO_NOISE_CH"], frame_axis),
                                          dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"tokens": {1: "n_tokens"}, "token_features": {2: "n_tokens"}},
            ),
        ]

    def hparams(self) -> dict:
        if not self.blobs:
            return {}
        return {"sample_rate": int(self.blobs.meta_json["sample_rate"])}

    def contract(self) -> dict:
        contract = BaseMultiPhaseModelExportConfig.contract(self)
        contract["text.frontend"] = "phonemes"
        contract["text.phoneme_alphabet"] = "ipa"
        contract["text.languages"] = ["en-US"]
        return contract

    def phoneme_table(self) -> dict:
        """The 62-symbol misaki-normalised table, framed `[BOS, p1, ..., pn, EOS]` with no blank -- what
        upstream's `phonemes_to_token_ids` builds."""
        vocab = nano_vocabulary(self.blobs or _NanoBlobs(Path(self.package_dir)))
        for symbol, want in (("<pad>", 0), ("<bos>", 1), ("<eos>", 2)):
            if vocab.get(symbol) != want:
                raise ValueError(f"sanoTTS: {symbol} is {vocab.get(symbol)} in the nano vocabulary, not {want}.")
        symbols = sorted(vocab, key=vocab.get)
        return {"symbols": symbols, "ids": [vocab[s] for s in symbols],
                "bos": 1, "eos": 2, "blank": -1, "interleave_blank": False}

    def driver_components(self) -> List:
        from .driver_components import (
            DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import Lit, Var
        from .lua_library import LuaLibrary

        h = self.blobs.h if self.blobs else {}
        fragment = self.driver_script_path
        piperlite = TTSSanoTTSExportConfig.driver_script_path
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            LuaLibrary(uses=("round_half_to_even", "to_f32", "aten_randn")),
            ExportConstants(values={
                "DUR_VOCAB": int(h.get("NANO_VOCAB", 0)),
                "AC_VOCAB": int(h.get("NANO_VOCAB", 0)),
                "FALLBACK_ID": FALLBACK_ID,
                "MAX_TOKENS": int(h.get("NANO_DUR_MAX_TOKENS", 0)),
                "MAX_DURATION": int(h.get("NANO_DUR_MAX_DURATION", 0)),
                "LENGTH_SCALE": 1.0,
                "NOISE_CH": int(h.get("NANO_NOISE_CH", 0)),
                "DEFAULT_SEED": NANO_DEFAULT_SEED,
                "DC_BLOCK_R": NANO_DC_BLOCK_R,
            }),
            LuaFragment(fragment / "01_limit.lua", reads=("MAX_TOKENS", "DUR_VOCAB")),
            # The token features and the expansion are piperlite's, statement for statement.
            LuaFragment(piperlite / "01_tokens.lua",
                        reads=("DUR_VOCAB", "AC_VOCAB", "FALLBACK_ID", "MAX_TOKENS"),
                        defines=("_n", "_dur_tokens", "_dur_features", "_ac_tokens", "linspace01")),
            SubgraphCallComponent(
                topology="duration", outputs=("_log_duration",),
                inputs={"tokens": Var("_dur_tokens"), "features": Var("_dur_features")},
                axes={"n_tokens": Var("_n"), "n_past": Lit(0)},
                note="Duration net: per-token log-duration."),
            LuaFragment(piperlite / "02_expand.lua",
                        reads=("_n", "_log_duration", "LENGTH_SCALE", "MAX_DURATION", "linspace01"),
                        defines=("_n_frames", "_token_features", "_frame_index", "_frame_features")),
            LuaFragment(fragment / "03_noise.lua", reads=("_n_frames", "NOISE_CH", "DEFAULT_SEED"),
                        defines=("_noise",)),
            SubgraphCallComponent(
                topology="synth", outputs=("_wave",),
                inputs={"tokens": Var("_ac_tokens"), "token_features": Var("_token_features"),
                        "frame_index": Var("_frame_index"), "frame_features": Var("_frame_features"),
                        "noise": Var("_noise")},
                axes={"n_tokens": Var("_n"), "n_enc_frames": Var("_n_frames"), "n_past": Lit(0)},
                note="Acoustic net to mel-100, then the ConvNeXt decoder and the iSTFT."),
            LuaFragment(fragment / "04_dc_block.lua", reads=("_wave", "DC_BLOCK_R"), defines=("waveform",)),
            DriverReturn(values=("waveform",)),
        ]


def _is_sanotts_nano(path: Path) -> bool:
    return path.is_dir() and (path / "nano_q8_meta.h").is_file() and (path / "meta.json").is_file()


def _build_sanotts_nano(path: Path, output_path: str) -> TTSSanoNanoExportConfig:
    return TTSSanoNanoExportConfig(package_dir=str(path), output_path=output_path)

def _is_sanotts(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not path.is_dir() or not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text()).get("format") == "roota.raw-fp16.v1"
    except (OSError, ValueError):
        return False


def _build_sanotts(path: Path, output_path: str) -> TTSSanoTTSExportConfig:
    return TTSSanoTTSExportConfig(package_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=TTSSanoTTSExportConfig,
        recognizers=[ModelRecognizer(name="sanotts", detect=_is_sanotts, build_config=_build_sanotts)],
    ))
    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=TTSSanoNanoExportConfig,
        recognizers=[ModelRecognizer(name="sanotts-nano", detect=_is_sanotts_nano,
                                     build_config=_build_sanotts_nano)],
    ))
