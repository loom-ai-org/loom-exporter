"""Vocetta2 -- VantoraLabs' tiny distilled English TTS (`VantoraLabs/Vocetta2-276k`, `-500k`, `-1m`; 0.28M,
0.50M and 1.03M parameters): a duration net, an acoustic net to mel-100 and a noise-fed ConvNeXt1D/iSTFT
decoder at 24 kHz, MIT, one checkpoint directory per size.

**It is sanoTTS's nano line, re-trained** (sanotts_export.py, `TTSSanoNanoExportConfig`): the same three
students, the same frozen 62-symbol misaki-normalised table (checked equal symbol for symbol), the same
LayerNorm epsilon, magnitude clip, zeroed edge bins and DC blocker; the 276k's decoder has heart-nano's
dimensions. What differs is everything around the modules, so this is its own config rather than a
subclass:

* **The checkpoint is torch.** `duration.pt`, `acoustic.pt`, `decoder.pt` are `state_dict`s of the
  classes in the checkpoint's own `microtts/models.py` (MIT), which the export imports from the
  checkpoint directory -- the definition that ships beside the weights and that upstream's runtime runs.
  Only the 1m's `config.json` states the dimensions, so they are read off the weights' shapes and
  cross-checked against `architecture_dims` where it exists.
* **`tilt.npy`** (500k, 1m): a per-band mel offset upstream's runtime adds between the acoustic net and
  the decoder by default, and the published scores were measured with. Folded into the acoustic output
  conv's bias. The 1m's is all zeros.
* **Zero noise by default.** Upstream renders `noise_scale=0.0` ("the best measured quality"), so the
  default here feeds the decoder zeros, and `seed` changes nothing; a caller who names `noise_scale`
  gets `aten_randn(seed)` times it. (sanoTTS's nano voices draw seeded noise by default.)
* **Its features round where upstream's code rounds them**, which is not where sanoTTS's do: the length
  hint is a Python double rounded once, and a frame's token position and in-token position are
  `torch.linspace` values (sanoTTS divides).
* **No clip on the waveform**, and upstream's 4096-tap truncation of the DC blocker reproduced exactly.

**One deliberate difference: no 2400-frame cut.** Upstream's `synthesize_ids` truncates the mel at 2400
frames (~25.6 s, "the training cap") after the acoustic net has seen all of them. An utterance that long
needs more than 11.6 frames per symbol at the 207-symbol limit, so upstream's own durations essentially
never reach it, and through `length_scale` (a loom addition, default 1.0) cutting a slowed sentence's end
off is not a behaviour worth reproducing. Every utterance under 2400 frames is upstream's.
"""
import importlib.util
import json
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

SAMPLE_RATE = 24000
WEIGHT_FILES = ("duration.pt", "acoustic.pt", "decoder.pt")
# The DC blocker after the iSTFT, H(z) = (1 - z^-1) / (1 - R z^-1), truncated by upstream to this many
# taps of its impulse response (`DecoderStudent.dc_block`).
DC_BLOCK_R = 0.9973
DC_BLOCK_TAPS = 4096
# The dimensions `config.json` may state (`architecture_dims`; only the 1m does), under the names `_dims`
# reads them off the weights with.
_CONFIG_DIMS = ("duration_hidden", "duration_depth", "acoustic_hidden", "acoustic_depth",
                "acoustic_token_depth", "decoder_dim", "decoder_pw", "decoder_blocks", "noise_channels",
                "n_mels")


def is_vocetta2(path: Path) -> bool:
    config = path / "config.json"
    if not path.is_dir() or not config.is_file() or not all((path / f).is_file() for f in WEIGHT_FILES):
        return False
    try:
        return json.loads(config.read_text()).get("model_type") == "vocetta2"
    except (OSError, ValueError):
        return False


def load_upstream_models(model_dir: Path):
    """The checkpoint's own `microtts/models.py`, imported by path. Only that file: the package's
    `__init__` pulls in the G2P, which the export does not need."""
    name = "vocetta2_upstream_models"
    if name not in sys.modules:
        script = model_dir / "microtts" / "models.py"
        if not script.is_file():
            raise FileNotFoundError(f"Vocetta2: {script} is missing; the export builds the checkpoint's own "
                                    f"modules from it.")
        spec = importlib.util.spec_from_file_location(name, script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _count(state: dict, prefix: str) -> int:
    """How many `prefix.N.` blocks a state_dict holds."""
    return len({k[len(prefix):].split(".")[0] for k in state if k.startswith(prefix)})


def _dims(states: dict) -> dict:
    d, a, x = states["duration"], states["acoustic"], states["decoder"]
    return {
        "vocab": int(d["embed.weight"].shape[0]),
        "duration_hidden": int(d["embed.weight"].shape[1]),
        "duration_depth": _count(d, "blocks."),
        "duration_kernel": int(d["blocks.0.0.weight"].shape[-1]),
        "acoustic_vocab": int(a["embed.weight"].shape[0]),
        "acoustic_hidden": int(a["embed.weight"].shape[1]),
        "acoustic_token_depth": _count(a, "token_blocks."),
        "acoustic_depth": _count(a, "frame_blocks."),
        "acoustic_kernel": int(a["token_blocks.0.0.weight"].shape[-1]),
        "n_mels": int(a["out.weight"].shape[0]),
        "decoder_dim": int(x["embed.weight"].shape[0]),
        "decoder_embed_kernel": int(x["embed.weight"].shape[-1]),
        "decoder_pw": int(x["blocks.0.pw0.weight"].shape[0]),
        "decoder_blocks": _count(x, "blocks."),
        "decoder_dw_kernel": int(x["blocks.0.dw.weight"].shape[-1]),
        "noise_channels": int(x["noise_adapter.weight"].shape[1]),
        "bins": int(x["head.weight"].shape[0]) // 2,
    }


def check_config(dims: dict, config: dict) -> None:
    """Refuse a checkpoint whose `config.json` disagrees with its weights, or that is not the line."""
    declared = config.get("architecture_dims", {})
    for key in _CONFIG_DIMS:
        if key in declared and int(declared[key]) != dims[key]:
            raise ValueError(f"Vocetta2: config.json says {key}={declared[key]}, the weights {dims[key]}.")
    if int(config.get("sample_rate", SAMPLE_RATE)) != SAMPLE_RATE:
        raise ValueError(f"Vocetta2: a {config['sample_rate']} Hz checkpoint; the line is {SAMPLE_RATE} Hz.")
    if dims["vocab"] != dims["acoustic_vocab"]:
        raise ValueError(f"Vocetta2: the duration net has {dims['vocab']} symbols, the acoustic net "
                         f"{dims['acoustic_vocab']}; upstream's runtime feeds both the same ids.")


def build_models(model_dir: Path) -> tuple:
    """`(dims, tilt, duration, acoustic, decoder)`: upstream's modules, loaded `strict=True`, in eval mode.
    `tilt` is the `[n_mels]` offset, or None."""
    models = load_upstream_models(model_dir)
    states = {c: torch.load(model_dir / f"{c}.pt", map_location="cpu", weights_only=True)
              for c in ("duration", "acoustic", "decoder")}
    dims = _dims(states)
    check_config(dims, json.loads((model_dir / "config.json").read_text()))

    duration = models.DurationStudent(vocab=dims["vocab"], hidden=dims["duration_hidden"],
                                      depth=dims["duration_depth"], kernel=dims["duration_kernel"])
    acoustic = models.AcousticStudent(vocab=dims["vocab"], hidden=dims["acoustic_hidden"],
                                      token_depth=dims["acoustic_token_depth"], depth=dims["acoustic_depth"],
                                      kernel=dims["acoustic_kernel"], n_mels=dims["n_mels"])
    decoder = models.DecoderStudent(n_mels=dims["n_mels"], dim=dims["decoder_dim"],
                                    blocks=dims["decoder_blocks"], pw_hidden=dims["decoder_pw"],
                                    dw_kernel=dims["decoder_dw_kernel"],
                                    embed_kernel=dims["decoder_embed_kernel"],
                                    noise_ch=dims["noise_channels"])
    if decoder.bins != dims["bins"] or decoder.bins != decoder.n_fft // 2 + 1:
        raise ValueError(f"Vocetta2: the head has {dims['bins']} bins, not n_fft/2 + 1 = {decoder.n_fft // 2 + 1}.")
    for module, component in ((duration, "duration"), (acoustic, "acoustic"), (decoder, "decoder")):
        module.load_state_dict(states[component], strict=True)
        module.eval()

    tilt = None
    if (model_dir / "tilt.npy").is_file():
        tilt = torch.from_numpy(np.load(model_dir / "tilt.npy").astype(np.float32)).reshape(-1)
        if tilt.numel() != dims["n_mels"]:
            raise ValueError(f"Vocetta2: tilt.npy has {tilt.numel()} values for {dims['n_mels']} mel bands.")
    return dims, tilt, duration, acoustic, decoder


class _DurationWrapper(nn.Module):
    """`(tokens [1, n], features [1, 3, n]) -> log_duration [1, n]`: `DurationStudent.forward` with
    `round_out=False` (what upstream's runtime calls), the features the host computed."""

    def __init__(self, duration):
        super().__init__()
        self.embed, self.proj, self.blocks, self.scales, self.out = (
            duration.embed, duration.proj, duration.blocks, duration.scales, duration.out)

    def forward(self, tokens, features):
        x = self.proj(torch.cat([self.embed(tokens).transpose(1, 2), features], dim=1))
        for (c0, c1), scale in zip(self.blocks, self.scales):
            x = x + scale * c1(F.silu(c0(x)))
        return self.out(x).squeeze(1)


class _SynthWrapper(nn.Module):
    """`(tokens [1, n], token_features [1, 2, n], frame_index [T], frame_features [1, 3, T], noise
    [1, C, T]) -> waveform [1, 256 (T - 1)]`: `AcousticStudent.forward` with its `repeat_interleave` a
    gather on the index the host built from the same durations, the tilt in the output conv's bias,
    `DecoderStudent.forward`, and the iSTFT. The DC blocker is the driver's."""

    def __init__(self, acoustic, decoder, tilt: Optional[torch.Tensor]):
        super().__init__()
        from .istft import ISTFT

        self.embed, self.tproj, self.token_blocks, self.token_scales = (
            acoustic.embed, acoustic.tproj, acoustic.token_blocks, acoustic.token_scales)
        self.fproj, self.frame_blocks, self.frame_scales = (
            acoustic.fproj, acoustic.frame_blocks, acoustic.frame_scales)
        # `mel + tilt` after the conv is `conv(x) + (bias + tilt)`: one rounding of the sum moves, a
        # fraction of an ulp of the mel.
        self.out = nn.Conv1d(acoustic.out.in_channels, acoustic.out.out_channels, 1)
        self.out.weight.data.copy_(acoustic.out.weight.data)
        self.out.bias.data.copy_(acoustic.out.bias.data + (tilt if tilt is not None else 0.0))
        self.decoder = decoder
        self.bins = decoder.bins
        # Bin 0 and Nyquist are zeroed: upstream's `mag[0] = mag[-1] = 0`.
        edges = torch.ones(1, decoder.bins, 1)
        edges[0, 0, 0] = edges[0, -1, 0] = 0.0
        self.register_buffer("edges", edges)
        self.istft = ISTFT(n_fft=decoder.n_fft, hop_length=decoder.hop, win_length=decoder.n_fft, center=True)

    def mel(self, tokens, token_features, frame_index, frame_features):
        x = self.tproj(torch.cat([self.embed(tokens).transpose(1, 2), token_features], dim=1))
        for (c0, c1), scale in zip(self.token_blocks, self.token_scales):
            x = x + scale * c1(F.silu(c0(x)))
        frames = torch.index_select(x.squeeze(0).transpose(0, 1), 0, frame_index)
        x = self.fproj(torch.cat([frames.transpose(0, 1).unsqueeze(0), frame_features], dim=1))
        for (c0, c1), scale in zip(self.frame_blocks, self.frame_scales):
            x = x + scale * c1(F.silu(c0(x)))
        return self.out(x)

    def forward(self, tokens, token_features, frame_index, frame_features, noise):
        dec = self.decoder
        x = dec.embed(self.mel(tokens, token_features, frame_index, frame_features)) + dec.noise_adapter(noise)
        x = dec.norm(x.transpose(1, 2))                                   # [1, T, dim]
        for block in dec.blocks:
            h = block.dw(x.transpose(1, 2)).transpose(1, 2)
            h = block.pw1(F.gelu(block.pw0(block.norm(h))))
            x = x + h * block.gamma
        out = dec.head(dec.fnorm(x)).transpose(1, 2)                      # [1, 2 bins, T]
        mag = torch.clamp(torch.exp(out[:, : self.bins]), max=1e2) * self.edges
        phase = out[:, self.bins:]
        return self.istft(mag * torch.cos(phase), mag * torch.sin(phase))


def vocabulary(model_dir: Path) -> dict:
    """The checkpoint's frozen table, `microtts/frontend.py`'s `DEFAULT_VOCABULARY`, read without
    importing the package (its `__init__` loads the G2P)."""
    name = "vocetta2_upstream_frontend"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, model_dir / "microtts" / "frontend.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return dict(sys.modules[name].DEFAULT_VOCABULARY)


@dataclass(kw_only=True)
class TTSVocetta2ExportConfig(BaseMultiPhaseModelExportConfig):
    """One Vocetta2 checkpoint directory -> one GGUF: `duration` and `synth`, and the driver between."""

    model_dir: str = ""
    architecture: str = "vocetta2"
    output_path: str = "vocetta2.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "vocetta2_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    trace_tokens: int = 23
    trace_frames: int = 61
    max_tokens: int = 2048
    max_frames: int = 16384

    dims: Optional[dict] = field(default=None, init=False, repr=False)
    max_symbols: Optional[int] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked("a checkpoint directory; `build_models` loads every component "
                               "`strict=True` and cross-checks the declared dimensions"),
        "architecture": Unchecked("the GGUF's own architecture string"),
        "output_path": Unchecked("where to write"),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link"),
        "driver_script_path": Unchecked("parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "trace_tokens": Unchecked("a property of the TRACE, not of the model"),
        "trace_frames": Unchecked("same"),
        "max_tokens": Unchecked("the token axis's declared upper bound; the driver refuses past the "
                                "duration net's own 207"),
        "max_frames": Unchecked("the frame axis's declared upper bound (16384 frames = 175 s)"),
        "dims": Unchecked("READ off the weights"),
        "max_symbols": Unchecked("READ off the duration module (`max_tokens`, its length hint's scale)"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        dims, tilt, duration, acoustic, decoder = build_models(Path(self.model_dir))
        self.dims, self.max_symbols = dims, int(duration.max_tokens)
        n, t = int(self.trace_tokens), int(self.trace_frames)
        token_axis = ct.RangeDim(1, self.max_tokens)
        frame_axis = ct.RangeDim(1, self.max_frames)
        g = torch.Generator().manual_seed(0)
        return [
            ExportPhase(
                name="duration", wrapper=_DurationWrapper(duration).eval(),
                dummy_inputs=(torch.randint(0, dims["vocab"], (1, n), generator=g),
                              torch.rand(1, 3, n, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="features", shape=(1, 3, token_axis), dtype=np.float32)],
                root_axis="n_tokens",
            ),
            ExportPhase(
                name="synth", wrapper=_SynthWrapper(acoustic, decoder, tilt).eval(),
                dummy_inputs=(torch.randint(0, dims["vocab"], (1, n), generator=g),
                              torch.rand(1, 2, n, generator=g),
                              torch.sort(torch.randint(0, n, (t,), generator=g)).values,
                              torch.rand(1, 3, t, generator=g),
                              torch.randn(1, dims["noise_channels"], t, generator=g)),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
                            ct.TensorType(name="token_features", shape=(1, 2, token_axis), dtype=np.float32),
                            ct.TensorType(name="frame_index", shape=(frame_axis,), dtype=np.int32),
                            ct.TensorType(name="frame_features", shape=(1, 3, frame_axis), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(1, dims["noise_channels"], frame_axis),
                                          dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"tokens": {1: "n_tokens"}, "token_features": {2: "n_tokens"}},
            ),
        ]

    def hparams(self) -> dict:
        """The rate, and `tts.phoneme_style` (loom.cpp ADR-071): distilled on misaki's normalised IPA, as
        sanoTTS's nano voices were."""
        return {"sample_rate": SAMPLE_RATE, "tts.phoneme_style": "misaki"}

    def contract(self) -> dict:
        contract = super().contract()
        contract["sample_rate"] = SAMPLE_RATE
        contract["text.frontend"] = "phonemes"
        contract["text.phoneme_alphabet"] = "ipa"
        contract["text.languages"] = ["en-US"]
        return contract

    def phoneme_table(self) -> dict:
        """The 62-symbol table, framed `[BOS, p1, ..., pn, EOS]` with no blank -- upstream's
        `phonemes_to_token_ids`."""
        vocab = vocabulary(Path(self.model_dir))
        for symbol, want in (("<pad>", 0), ("<bos>", 1), ("<eos>", 2)):
            if vocab.get(symbol) != want:
                raise ValueError(f"Vocetta2: {symbol} is {vocab.get(symbol)} in the vocabulary, not {want}.")
        if self.dims and len(vocab) != self.dims["vocab"]:
            raise ValueError(f"Vocetta2: the table has {len(vocab)} symbols, the embeddings {self.dims['vocab']}.")
        symbols = sorted(vocab, key=vocab.get)
        return {"symbols": symbols, "ids": [vocab[s] for s in symbols],
                "bos": 1, "eos": 2, "blank": -1, "interleave_blank": False}

    def driver_components(self) -> List:
        from .driver_components import (
            DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import Lit, Var
        from .lua_library import LuaLibrary

        dims = self.dims or {}
        fragment = self.driver_script_path
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            LuaLibrary(uses=("round_half_to_even", "to_f32", "aten_randn")),
            ExportConstants(values={
                "VOCAB": int(dims.get("vocab", 0)),
                "MAX_TOKENS": int(self.max_symbols or 0),
                "NOISE_CH": int(dims.get("noise_channels", 0)),
                "DC_BLOCK_R": DC_BLOCK_R,
                "DC_BLOCK_TAPS": DC_BLOCK_TAPS,
            }),
            LuaFragment(fragment / "01_tokens.lua", reads=("VOCAB", "MAX_TOKENS"),
                        defines=("_n", "_tokens", "_dur_features", "linspace01")),
            SubgraphCallComponent(
                topology="duration", outputs=("_log_duration",),
                inputs={"tokens": Var("_tokens"), "features": Var("_dur_features")},
                axes={"n_tokens": Var("_n"), "n_past": Lit(0)},
                note="Duration net: per-token log-duration."),
            LuaFragment(fragment / "02_expand.lua", reads=("_n", "_log_duration", "linspace01"),
                        defines=("_n_frames", "_token_features", "_frame_index", "_frame_features")),
            LuaFragment(fragment / "03_noise.lua", reads=("_n_frames", "NOISE_CH"), defines=("_noise",)),
            SubgraphCallComponent(
                topology="synth", outputs=("_wave",),
                inputs={"tokens": Var("_tokens"), "token_features": Var("_token_features"),
                        "frame_index": Var("_frame_index"), "frame_features": Var("_frame_features"),
                        "noise": Var("_noise")},
                axes={"n_tokens": Var("_n"), "n_enc_frames": Var("_n_frames"), "n_past": Lit(0)},
                note="Acoustic net to mel-100 (tilt folded in), the ConvNeXt decoder and the iSTFT."),
            LuaFragment(fragment / "04_dc_block.lua", reads=("_wave", "DC_BLOCK_R", "DC_BLOCK_TAPS"),
                        defines=("waveform",)),
            DriverReturn(values=("waveform",)),
        ]


def _build_vocetta2(path: Path, output_path: str) -> TTSVocetta2ExportConfig:
    return TTSVocetta2ExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=TTSVocetta2ExportConfig,
        recognizers=[ModelRecognizer(name="vocetta2", detect=is_vocetta2, build_config=_build_vocetta2)],
    ))
