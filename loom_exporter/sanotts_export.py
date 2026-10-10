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

    def forward(self, tokens, token_features, frame_index, frame_features):
        x = self.token_input_proj(torch.cat([self.embedding(tokens).transpose(1, 2), token_features], dim=1))
        for block in self.token_blocks:
            x = block(x)
        frames = torch.index_select(x.squeeze(0).transpose(0, 1), 0, frame_index)
        x = self.frame_input_proj(torch.cat([frames.transpose(0, 1).unsqueeze(0), frame_features], dim=1))
        for block in self.frame_blocks:
            x = block(x)
        return self.decoder(self.output(x)).reshape(1, -1)


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
