"""kale-tts-1-tiny (Kale, Apache-2.0): a 45.7M-parameter English TTS with ten designed voices, 24 kHz. Exported
from the PyTorch release (`pytorch/tiny/*.pt`), the teacher it was distilled from (`KaleComputer/kale-tts-1-base`,
whose text encoder and duration encoder it keeps frozen) and the word encoder it reads with
(`jhu-clsp/ettin-encoder-17m`, MIT).

Upstream (`recipe/textside/textside.py`'s `export10` graph, the one its ONNX files hold) for one sentence:

  sentence + previous sentence -> Ettin (ModernBERT, 7 layers) -> mean of its last two layers' states
      -> averaged over each word's sub-tokens -> one 256-wide row per word
  phoneme ids -> K20Text (embedding, 3 conv blocks, + the phoneme's word row, 2 transformer blocks)
      -> d_en: the 512-wide input Kokoro's ALBERT + bert_encoder used to give the predictor
  d_en, style -> Kokoro's DurationEncoder (3x BiLSTM + AdaLayerNorm) -> BiLSTM -> durations
      -> the prosody states repeated by the durations -> F0Ntrain -> F0, N
      -> the end clamp: in the last `tail` voiced frames, F0 held at or above the voice's floor; then
         every frame under the voice's unvoicing floor set to 0 (rendered as noise)
  ids -> Kokoro's TextEncoder (CNN + BiLSTM) -> repeated by the durations
  asr, F0, N, the voice's 16-number row -> Paradee's decoder trunk (Kokoro's AdaIN layout at width 256)
      and its source-filter head -> 24 kHz

**Everything after d_en is a module this exporter already traces.** The predictor and the text encoder are
Kokoro's classes (`kokoro_export.build_prosody_phases`), and the decoder is `model.Dec`, which is Paradee's
`VDecoder` layout -- the same attribute names, the same `SourceFilterHead2` (Kale vendors Paradee's
`vhead.py` unchanged) -- so `paradee_export._SynthWrapper` takes it as it is. What is new is the text side:
the Ettin encoder as one graph, and K20Text as another.

**The word averaging is one matrix the driver builds.** Upstream averages Ettin's states per word
(`wavg @ h`, `wavg` [words, sub-tokens]) and then gathers each phoneme's word (`widx`). A row gather
commutes with the matmul, so the driver hands the graph `wtok = wavg[widx]` ([phonemes, sub-tokens]) and the
graph does `wtok @ h`: the same products, and a phase with two axes instead of three.

**The voice is a row of three tables**, carried in the GGUF and picked by the driver's `voice` input: the
predictor's style (Kokoro's pack layout, one row per phoneme count, `pack[min(n, 510) - 1]`; the teacher's
pack plus the pitch heads' per-voice offset `pv`, which is what the release's `voices/<id>.bin` holds), the
decoder's 16-number row, and the four end-clamp numbers the release's `voices.json` names per voice.

The sine bank's phase, the noise and the STFTs are Paradee's (see `paradee_export`): upstream's `HeadX` is
Paradee's, copied unchanged.
"""
import ast
import json
import os
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

DRIVER_DIR = Path(__file__).resolve().parent / "kale_tts_driver"

SAMPLE_RATE = 24000
FRAME_HOP = 300
DURATION_HOP = 600
# Kokoro's packs are indexed by the sentence's phoneme count; a sentence past the last row uses it.
PACK_ROWS = 510
# K20Text's learned positions (`nn.Embedding(512, c)`), the two 0 pads included.
MAX_IDS = 512
# Upstream's decoder sizes (`kline.KSIZES["k"]`): preset A, a 160-wide head of 4 blocks, 16-number rows.
DECODER_PRESET, HEAD_DIM, HEAD_BLOCKS, DECODER_STYLE_DIM = "A", 160, 4, 16

TEACHER_REPO = "KaleComputer/kale-tts-1-base"
ETTIN_REPO = "jhu-clsp/ettin-encoder-17m"
# The commit Kale's export used (`recipe/textside/config/textside.json` "ettin_revision"): its weights match the
# release graph bit for bit on every encoder tensor.
ETTIN_REVISION = "987607455c61e7a5bbc85f7758e0512ea6d0ae4c"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"

PACKS = "loom.kale.prosody_packs"
DECODER_ROWS = "loom.kale.voice_sound"
CLAMPS = "loom.kale.voice_clamp"
# The voice files hold the clamp numbers in this order after `voice_sound` (release `voices.json` "voice_inputs").
CLAMP_FIELDS = ("pitch_floor_hz", "clamp_tail_frames", "clamp_thr_hz", "unvoice_hz")

CHECKPOINTS = {
    "text_side": "pytorch/tiny/ettin_text_side_fle.pt",
    "pitch": "pytorch/tiny/pitch_heads_flp.pt",
    "duration": "pytorch/tiny/duration_head_fld.pt",
    "decoder": "pytorch/tiny/decoder_flkd.pt",
}


def _strip(module):
    from .paradee_export import _strip as strip
    strip(module)


def _k20text_class(model_dir: Path):
    """`K20Text` from upstream's own source (`recipe/textside/textside.py`, `_mods`), not a copy: that file is a
    Modal app that imports `modal` at the top, so the one function is compiled out of it and run here."""
    src = (model_dir / "recipe" / "textside" / "textside.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mods"), None)
    if fn is None:
        raise ValueError(f"{model_dir}: recipe/textside/textside.py has no `_mods`; this export reads K20Text from it")
    scope: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(model_dir / "recipe/textside/textside.py"), "exec"), scope)
    return scope["_mods"]()[1]


def _resolve(given: Optional[str], sibling: Path, repo: str, revision: Optional[str], patterns) -> Path:
    """The named directory, else a sibling checkout of that name, else the Hub at the pinned revision."""
    if given:
        return Path(given)
    if sibling.is_dir():
        return sibling
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def load_kale(model_dir: Path, teacher_dir: Optional[str] = None, ettin_dir: Optional[str] = None):
    """(modules, voices) from the release, the teacher and Ettin, weight norm folded."""
    model_dir = Path(model_dir)
    train = model_dir / "recipe" / "train"
    for p in (str(train / "third_party" / "paradee"), str(train)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from . import kokoro_export as ke
    ke.load_kokoro()                     # Kokoro's trace-friendly AdainResBlk1d before anything is built
    import model as kale_model           # recipe/train/model.py: Dec over Paradee's head
    from huggingface_hub import hf_hub_download

    teacher = _resolve(teacher_dir, model_dir.parent / "kale-tts-1-base", TEACHER_REPO, None,
                       ["kale-tts-1-base.pth", "voices/*", "voices.json"])
    ettin = _resolve(ettin_dir, model_dir.parent / "ettin-encoder-17m", ETTIN_REPO, ETTIN_REVISION,
                     ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json"])
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731

    # The teacher's text encoder and predictor, as `kline.load_kmodel` loads them: kikiri's weight-norm names
    # renamed to the PyPI kokoro's, every tensor required (its own loader is strict=False and skipped some).
    km = ke.KModel(repo_id=KOKORO_REPO, config=hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REVISION),
                   model=False)
    sd = load(teacher / "kale-tts-1-base.pth")
    for key in ("predictor", "text_encoder"):
        state = {}
        for k, v in sd[key].items():
            k = k[7:] if k.startswith("module.") else k
            k = k.replace(".parametrizations.weight.original0", ".weight_g").replace(
                ".parametrizations.weight.original1", ".weight_v")
            state[k] = v
        res = getattr(km, key).load_state_dict(state, strict=False)
        bad = [m for m in res.missing_keys if not m.endswith((".norm.weight", ".norm.bias"))]
        if bad or res.unexpected_keys:
            raise ValueError(f"kale: the teacher's {key} does not fit Kokoro's: missing {bad[:5]}, "
                             f"unexpected {res.unexpected_keys[:5]}")
    dur = load(model_dir / CHECKPOINTS["duration"])
    km.predictor.lstm.load_state_dict(dur["lstm"])
    km.predictor.duration_proj.load_state_dict(dur["duration_proj"])
    pitch = load(model_dir / CHECKPOINTS["pitch"])
    for m in ("shared", "F0", "N", "F0_proj", "N_proj"):
        getattr(km.predictor, m).load_state_dict(pitch[m])
    km = km.eval()
    _strip(km)

    text = _k20text_class(model_dir)(out=int(km.predictor.text_encoder.lstms[1].channels)).eval()
    text.load_state_dict(load(model_dir / CHECKPOINTS["text_side"])["model"], strict=True)

    dec_state = load(model_dir / CHECKPOINTS["decoder"])["model"]
    dec = kale_model.Dec(DECODER_PRESET, int(km.text_encoder.lstm.hidden_size) * 2, DECODER_STYLE_DIM,
                         HEAD_DIM, HEAD_BLOCKS).eval()
    dec.load_state_dict({k[4:]: v for k, v in dec_state.items() if k.startswith("dec.")}, strict=True)
    if not dec.head.burst or dec.head.mod != 2 or dec.head.K != 64:
        raise ValueError("kale: this export traces Paradee's 1.1+ head (two pulsing-noise pairs, 64 harmonics, "
                         "the 3 ms noise path); this checkpoint's head differs")
    _strip(dec)

    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(str(ettin), attn_implementation="eager").eval()

    # The voices, in the release's order (`voices.json`): the pack + pv, the decoder row, the clamp numbers.
    vj = json.loads((model_dir / "voices.json").read_text())
    base = json.loads((teacher / "voices.json").read_text())
    pack_of = {e["name"]: teacher / e["file"] for e in base}
    order = list(pitch["voices"])
    if list(dur["voices"]) != order or not torch.equal(dur["pv"], pitch["pv"]):
        raise ValueError("kale: the duration head and the pitch heads disagree on the voices' offsets")
    style = dec_state["style"].float().numpy()
    voices = []
    for v in vj["voices"]:
        row = order.index(v["id"])
        if row != int(v.get("order", row)):
            raise ValueError(f"kale: voice {v['id']} is row {row} of the checkpoints and order {v['order']} in voices.json")
        pack = load(pack_of[v["name"]]).squeeze(1).float().numpy()
        if pack.shape != (PACK_ROWS, 256):
            raise ValueError(f"kale: {pack_of[v['name']].name} has shape {pack.shape}, expected ({PACK_ROWS}, 256)")
        prosody = pack[:, 128:] + pitch["pv"][row].float().numpy()[None]
        voices.append(dict(id=v["id"], name=v["name"], accent=v.get("accent", "en-us"),
                           description=v.get("description", ""), prosody=prosody.astype(np.float32),
                           sound=style[row].astype(np.float32),
                           clamp=np.array([float(v[k]) for k in CLAMP_FIELDS], np.float32)))
    return dict(km=km, text=text, dec=dec, encoder=encoder, ettin_dir=ettin, teacher_dir=teacher), voices


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _EttinWrapper(nn.Module):
    """`(ids [S]) -> [S, 256]`: the mean of the encoder's last two hidden states (upstream's `WordFeat`), for the
    sentence with the previous one before it, `[CLS] ... [SEP]`, every position attended.

    `ModernBertModel.forward` with its eager attention, batch-free: HF views the fused projection as
    [batch, seq, 3, heads, dim], one axis more than ggml has, so it is viewed here as [seq, 3, heads, dim]. The
    rest is HF's arithmetic in HF's order -- the rotary tables per layer from its own `inv_freq` (local and
    global layers carry their own theta), `rotate_half`, the scale, and on the local layers the sliding window
    as HF builds it (`finfo.min` beyond `local_attention // 2` either side). The distance is an outer
    difference spelled as two K = 1 matmuls (ggml broadcasts only an op's second operand). The final norm is
    not applied: `hidden_states` holds every layer's output before it."""

    def __init__(self, encoder):
        super().__init__()
        cfg = encoder.config
        self.embeddings, self.layers = encoder.embeddings, encoder.layers
        self.heads, self.window = cfg.num_attention_heads, cfg.local_attention // 2
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.neg = torch.finfo(torch.float32).min

    def forward(self, tokens):
        x = self.embeddings.drop(self.embeddings.norm(self.embeddings.tok_embeddings(tokens.long())))   # [S, C]
        # The length off the INPUT: read off `x` (after the LayerNorm) the exporter's shape walk resolved it
        # to the trace's constant, and `arange` became a static one-row RANGE_1D -- every position 0.
        S = tokens.shape[0]
        pos = torch.arange(S).float()
        ones = torch.ones(S, 1)
        dist = torch.abs(pos[:, None] @ ones.T - ones @ pos[None, :])
        window = torch.where(dist <= self.window, torch.zeros_like(dist), torch.full_like(dist, self.neg))
        states = []
        for layer in self.layers:
            attn = layer.attn
            qkv = attn.Wqkv(layer.attn_norm(x)).view(S, 3, self.heads, self.head_dim).permute(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]                                   # [H, S, D]
            freqs = pos[:, None] @ attn.rotary_emb.inv_freq.float()[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attn.rotary_emb.attention_scaling
            sin = emb.sin() * attn.rotary_emb.attention_scaling
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            w = (q @ k.transpose(1, 2)) * (self.head_dim ** -0.5)
            if attn.local_attention != (-1, -1):
                w = w + window
            y = (torch.softmax(w, dim=-1) @ v).permute(1, 0, 2).reshape(S, self.heads * self.head_dim)
            x = x + attn.out_drop(attn.Wo(y))
            x = x + layer.mlp(layer.mlp_norm(x))
            states.append(x)
        return (states[-1] + states[-2]) / 2


class _TextSideWrapper(nn.Module):
    """`(tokens [T], wtok [T, S], h [S, 256]) -> d_en [T, 512]`: K20Text with every phoneme's word row as
    `wtok @ h` (upstream's `gather(wavg @ h, widx)`, the gather moved before the product). One sentence, no
    padding: every mask is all False and the `* -1e4` terms are zeros, dropped. The transformer blocks are
    `TFBlock.forward` batch-free, for the 4-axis reason `_EttinWrapper` states."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    @staticmethod
    def _block(b, x):                                                          # x [T, C]
        T, C = x.shape
        hd = C // b.nh
        qkv = b.qkv(b.n1(x)).reshape(T, 3, b.nh, hd).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(1, 2)) * (hd ** -0.5)
        y = (a.softmax(-1) @ v).permute(1, 0, 2).reshape(T, C)
        x = x + b.drop(b.o(y))
        return x + b.drop(b.ff(b.n2(x)))

    def forward(self, tokens, wtok, h):
        t = self.text
        x = tokens.long()[None]
        m = torch.zeros_like(x, dtype=torch.bool)
        e = t.emb(x)
        for b in t.convs:
            e = b(e, m)
        e = (e[0] + t.wproj(wtok @ h) + t.pos(torch.arange(x.shape[1])))
        for b in t.tf:
            e = self._block(b, e)
        return t.out(e)


@dataclass(kw_only=True)
class KaleTtsExportConfig(BaseMultiPhaseModelExportConfig):
    """A kale-tts-1-tiny release directory (`voices.json` + `pytorch/tiny/`) -> one Loom GGUF."""

    architecture: str = "kale_tts"
    model_dir: str
    teacher_dir: Optional[str] = None
    ettin_dir: Optional[str] = None
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = DRIVER_DIR
    trace_frames: int = 12
    _voices: Optional[list] = field(default=None, init=False, repr=False)
    _dims: Optional[dict] = field(default=None, init=False, repr=False)
    _ettin_dir: Optional[Path] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "architecture": Unchecked("the GGUF's architecture string"),
        "model_dir": Unchecked("path to the release; the recognizer found voices.json + pytorch/tiny/ in it"),
        "teacher_dir": Unchecked("kale-tts-1-base, or None for a sibling checkout or the Hub"),
        "ettin_dir": Unchecked("ettin-encoder-17m, or None for a sibling checkout or the Hub at the pinned commit"),
        "decomposition": Unchecked("MultiPhase by construction"),
        "driver_script_path": Unchecked("hand-written fragments, checked against the topologies"),
        "trace_frames": Unchecked("the concrete duration-frame count the synth trace runs at"),
        "_voices": Unchecked("the voice tables, read in phases()"),
        "_dims": Unchecked("prosody_dims of the real modules, read in phases()"),
        "_ettin_dir": Unchecked("where the Ettin checkpoint was found, read in phases()"),
    }

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct
        from . import kokoro_export as ke
        from .paradee_export import _SynthWrapper

        mods, voices = load_kale(Path(self.model_dir), self.teacher_dir, self.ettin_dir)
        self._voices, self._ettin_dir = voices, mods["ettin_dir"]
        km = mods["km"]
        self._dims = ke.prosody_dims(km.text_encoder, km.predictor)
        g = torch.Generator().manual_seed(0)
        width = mods["encoder"].config.hidden_size
        d_model = self._dims["d_model"]

        ettin = _EttinWrapper(mods["encoder"]).eval()
        text_side = _TextSideWrapper(mods["text"]).eval()
        ids = torch.randint(1, 1000, (9,), generator=g, dtype=torch.int32)
        phon = torch.randint(1, 178, (7,), generator=g, dtype=torch.int32)
        wtok = torch.rand(7, 9, generator=g)
        h = torch.randn(9, width, generator=g)
        frames = self.trace_frames
        synth = _SynthWrapper(mods["dec"]).eval()
        f0 = torch.rand(2 * frames, generator=g) * 200 + 80
        dummy = (torch.randn(1, 2 * self._dims["hidden_per_dir"], frames, generator=g), f0[None],
                 torch.randn(1, 2 * frames, generator=g), torch.randn(DECODER_STYLE_DIM, generator=g),
                 torch.rand(2 * frames + 1, 2, generator=g), torch.randn(DURATION_HOP * frames, generator=g),
                 torch.randn(DURATION_HOP * frames, generator=g))
        with torch.no_grad():               # eager sanity checks before tracing
            ettin(ids)
            text_side(phon, wtok, h)
            synth(*dummy)
        f0_axis = ct.RangeDim(2, 4000)
        asr_dim = 2 * self._dims["hidden_per_dir"]
        # One RangeDim per length, shared by the inputs that carry it: the phonemes' and the sub-tokens'.
        phonemes, subs = ct.RangeDim(1, MAX_IDS), ct.RangeDim(2, 2048)
        return [
            ExportPhase(
                name="ettin", wrapper=ettin, dummy_inputs=(ids,),
                mil_inputs=[ct.TensorType(name="tokens", shape=(ct.RangeDim(2, 2048),), dtype=np.int32)],
            ),
            ExportPhase(
                name="text_side", wrapper=text_side, dummy_inputs=(phon, wtok, h),
                mil_inputs=[ct.TensorType(name="tokens", shape=(phonemes,), dtype=np.int32),
                            ct.TensorType(name="wtok", shape=(phonemes, subs), dtype=np.float32),
                            ct.TensorType(name="h", shape=(subs, width), dtype=np.float32)],
                declared_axes={"wtok": {1: "n_enc_frames"}, "h": {0: "n_enc_frames"}},
            ),
            *ke.build_prosody_phases(km.text_encoder, km.predictor),
            ExportPhase(
                name="synth", wrapper=synth, dummy_inputs=dummy,
                mil_inputs=[ct.TensorType(name="asr", shape=(1, asr_dim, ct.RangeDim(1, 2000)), dtype=np.float32),
                            ct.TensorType(name="f0", shape=(1, f0_axis), dtype=np.float32),
                            ct.TensorType(name="n", shape=(1, f0_axis), dtype=np.float32),
                            ct.TensorType(name="style", shape=(DECODER_STYLE_DIM,), dtype=np.float32),
                            ct.TensorType(name="phase", shape=(ct.RangeDim(3, 4001), 2), dtype=np.float32),
                            ct.TensorType(name="noise", shape=(ct.RangeDim(600, 1200000),), dtype=np.float32),
                            ct.TensorType(name="noise_rev", shape=(ct.RangeDim(600, 1200000),), dtype=np.float32)],
                root_axis="n_enc_frames",
                declared_axes={"asr": {2: "n_enc_frames"}, "f0": {1: "2*n_enc_frames"}, "n": {1: "2*n_enc_frames"},
                               "phase": {0: "2*n_enc_frames+1"},
                               "noise": {0: "600*n_enc_frames"}, "noise_rev": {0: "600*n_enc_frames"}},
            ),
        ]

    def backend_kwargs(self) -> dict:
        kwargs = super().backend_kwargs()
        # The text front end (`kale_tts_tokenizer_export`): the release's own files, and Ettin's vocabulary.
        kwargs.update(tokenizer_dir=self.model_dir, tokenizer_family="kale_tts",
                      kale_ettin_dir=str(self._ettin_dir) if self._ettin_dir else None)
        if self._voices:
            kwargs["driver_weights"] = {
                PACKS: np.stack([v["prosody"] for v in self._voices]).astype(np.float32),
                DECODER_ROWS: np.stack([v["sound"] for v in self._voices]).astype(np.float32),
                CLAMPS: np.stack([v["clamp"] for v in self._voices]).astype(np.float32),
            }
        return kwargs

    def driver_components(self) -> List:
        from .driver_components import ComputedCall, DriverReturn, ExportConstants, HelperCall, LuaFragment
        from .lua_library import LuaLibrary

        dims = self._dims or {"style_dim": 0, "d_model": 0, "hidden_per_dir": 0}
        return [
            LuaLibrary(uses=("array_slice", "array_sum", "run_bi_lstm", "run_resblk_stack", "run_proj1x1",
                             "predict_durations", "aten_randn")),
            ExportConstants(values={
                "STYLE_DIM": dims["style_dim"], "D_MODEL": dims["d_model"],
                "HIDDEN_PER_DIR": dims["hidden_per_dir"], "N_VOICES": len(self._voices or []),
                "PACK_ROWS": PACK_ROWS, "DECODER_STYLE_DIM": DECODER_STYLE_DIM, "N_CLAMP": len(CLAMP_FIELDS),
                "FRAME_HOP": FRAME_HOP, "SAMPLE_RATE": SAMPLE_RATE, "MAX_IDS": MAX_IDS,
            }, top_level=True),
            LuaFragment(DRIVER_DIR / "00_kale.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish"),
                        drives=(
                            HelperCall("run_bi_lstm", tuple(f"duration_lstm_{i}" for i in range(3)),
                                       written='"duration_lstm_" .. i'),
                            ComputedCall(topologies=tuple(f"duration_adaln_{i}" for i in range(3)),
                                         inputs=("x", "style"), written='"duration_adaln_" .. i'),
                            HelperCall("run_bi_lstm", "top_lstm"),
                            HelperCall("run_bi_lstm", "text_encoder_lstm"),
                            HelperCall("run_bi_lstm", "f0n_shared_lstm"),
                            HelperCall("run_resblk_stack", "f0n_f0"),
                            HelperCall("run_resblk_stack", "f0n_n"),
                            HelperCall("run_proj1x1", "f0n_f0_proj"),
                            HelperCall("run_proj1x1", "f0n_n_proj"),
                        )),
            LuaFragment(DRIVER_DIR / "01_generate.lua", defines=("wave",)),
            DriverReturn(values=("wave",)),
        ]

    def hparams(self) -> dict:
        return {"sample_rate": SAMPLE_RATE}

    def contract(self) -> dict:
        contract = super().contract()
        contract["sample_rate"] = SAMPLE_RATE
        # The file's own front end: text in, the packed records out (`loom::KaleTtsVocab`).
        contract["text.frontend"] = "vocab"
        voices = self._voices or []
        # The driver's `voice` is a row of the tables, named here in that order; each voice reads one accent
        # (upstream picks the front end by the voice: Amy and Jack read the British one).
        contract["tts.voices"] = [v["name"].lower() for v in voices]
        contract["tts.voice_languages"] = [_bcp47(v["accent"]) for v in voices]
        return contract


def _bcp47(accent: str) -> str:
    lang, _, region = accent.partition("-")
    return f"{lang}-{region.upper()}" if region else lang


def _is_kale_tts(path: Path) -> bool:
    if not (path.is_dir() and (path / "voices.json").is_file()
            and all((path / p).is_file() for p in CHECKPOINTS.values())):
        return False
    try:
        return bool(json.loads((path / "voices.json").read_text()).get("ettin_tokenizer"))
    except (OSError, ValueError):
        return False


def _build_kale_tts(path: Path, output_path: str) -> KaleTtsExportConfig:
    return KaleTtsExportConfig(output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech", config_class=KaleTtsExportConfig,
        recognizers=[ModelRecognizer(name="kale-tts", detect=_is_kale_tts, build_config=_build_kale_tts)],
    ))
