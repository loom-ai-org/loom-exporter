"""Vaniq-Edge (upstream VITS) and AuroraTTS-100K -- the hermetic half.

Neither checkpoint is in CI, so what is tested here is what does not need one: the upstream-VITS
recognizer and the architecture it reads off a training config, the GRU cell topology against
`torch.nn.GRU`, and the three pieces of AuroraTTS's DSP that the export re-expresses -- the per-sample
interpolation as transposed convolutions, the phase's closed form, and the envelope lookup as a hat sum.

The real checkpoints are the gate half (2026-10-25, scratch oracles `vaniq_ref.py`/`aurora_ref.py`):
Vaniq-Edge through the released rc19 engine with every draw pinned is 3.9e-6..1.0e-5 from upstream at
f64, inside upstream's own f32 spread (3.9e-6..1.5e-5); AuroraTTS's wrappers at f64 are 5e-7 from
upstream's f64 waveform with identical durations.
"""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from loom_exporter import auroratts_export as A
from loom_exporter import vits_export as V
from loom_exporter.recurrent import build_gru_cell_topologies


# -- Vaniq-Edge / upstream VITS ---------------------------------------------------------------------

_VANIQ_CONFIG = {
    "train": {"segment_size": 8192},
    "data": {"sampling_rate": 24000, "filter_length": 1024, "hop_length": 256, "n_speakers": 0,
             "add_blank": True, "cleaned_text": True},
    "model": {"inter_channels": 128, "hidden_channels": 128, "filter_channels": 384, "n_heads": 2,
              "n_layers": 4, "kernel_size": 3, "p_dropout": 0.1, "resblock": "1",
              "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5]] * 3,
              "upsample_rates": [8, 8, 2, 2], "upsample_initial_channel": 256,
              "upsample_kernel_sizes": [16, 16, 4, 4], "use_spectral_norm": False, "use_sdp": True},
}


def _upstream_dir(tmp_path, weights="model.pth", inference=None):
    (tmp_path / "config.json").write_text(json.dumps(_VANIQ_CONFIG))
    (tmp_path / weights).write_bytes(b"")
    if inference is not None:
        (tmp_path / "inference.py").write_text(inference)
    return tmp_path


def test_an_upstream_vits_directory_is_recognized(tmp_path):
    assert V._is_upstream_vits(_upstream_dir(tmp_path))
    assert not V._is_vaniq_edge(tmp_path)


def test_upstream_names_its_generator_g_step(tmp_path):
    assert V._is_upstream_vits(_upstream_dir(tmp_path, weights="G_100000.pth"))


def test_a_config_without_vits_keys_is_not_vits(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (tmp_path / "model.pth").write_bytes(b"")
    assert not V._is_upstream_vits(tmp_path)


def test_vaniq_edge_is_told_apart_by_its_own_inference_script(tmp_path):
    _upstream_dir(tmp_path, inference="class VaniqTTS:\n    pass\n")
    assert V._is_vaniq_edge(tmp_path)
    config = V._build_vaniq_edge(tmp_path, "out.gguf")
    assert (config.noise_scale, config.noise_scale_w) == (0.333, 0.8)
    assert config.hparams()["tts.phoneme_style"] == "espeak"


def test_the_architecture_is_read_off_the_training_config(tmp_path):
    config = V.TTSUpstreamVitsExportConfig.from_directory(_upstream_dir(tmp_path), "out.gguf")
    hp = config.hp
    assert hp["n_vocab"] == len(V.VITS_SYMBOLS) == 178
    assert hp["spec_channels"] == 513 and hp["segment_size"] == 32
    assert hp["resblock_dilation_sizes"] == ((1, 3, 5),) * 3 and hp["upsample_rates"] == (8, 8, 2, 2)
    assert "use_spectral_norm" not in hp
    assert config.sample_rate == 24000 and config.hparams() == {"sample_rate": 24000}
    # The class's defaults are upstream VITS's notebook's, not Piper's or Vaniq-Edge's.
    assert (config.noise_scale, config.noise_scale_w) == (0.667, 0.8)


def test_the_framing_is_intersperse(tmp_path):
    """`commons.intersperse(ids, 0)` = `[0, p1, 0, ..., pn, 0]`: the opening 0 as BOS, a blank after
    every phoneme, no EOS -- what Matcha (the same front end) declares."""
    table = V.TTSUpstreamVitsExportConfig.from_directory(_upstream_dir(tmp_path), "o").phoneme_table()
    assert (table["bos"], table["eos"], table["blank"], table["interleave_blank"]) == (0, -1, 0, True)
    assert table["symbols"][0] == "_" and table["symbols"][16] == " "


def test_a_multi_speaker_config_is_refused(tmp_path):
    _upstream_dir(tmp_path)
    multi = dict(_VANIQ_CONFIG, data=dict(_VANIQ_CONFIG["data"], n_speakers=109))
    (tmp_path / "config.json").write_text(json.dumps(multi))
    with pytest.raises(NotImplementedError, match="speakers"):
        V.TTSUpstreamVitsExportConfig.from_directory(tmp_path, "o")


# -- the GRU cell ------------------------------------------------------------------------------------

def _interpret(topology, weights, x, h):
    env = {"layer_input": x, "h_prev": h, **weights}
    for node in topology["nodes"]:
        a = [env[i] for i in node["inputs"]]
        op = node["op"]
        if op == "MUL_MAT":
            v = a[0] @ a[1]
        elif op == "VIEW":
            k = node["attrs"]["offset"] // 4
            v = a[0][k:k + node["attrs"]["shape"][0]]
        else:
            v = {"ADD": lambda: a[0] + a[1], "SUB": lambda: a[0] - a[1], "MUL": lambda: a[0] * a[1],
                 "SIGMOID": lambda: 1 / (1 + np.exp(-a[0])), "TANH": lambda: np.tanh(a[0])}[op]()
        env[node["outputs"][0]] = v
    return env[topology["outputs"][0]]


def test_the_gru_cell_steps_torch_gru_in_both_directions():
    torch.manual_seed(0)
    gru = torch.nn.GRU(7, 5, batch_first=True, bidirectional=True).double()
    cells = build_gru_cell_topologies(gru, "ctx.")
    assert cells["forward"]["outputs"] == ["h_new"] and cells["backward"]["outputs"] == ["h_new"]
    assert [i["name"] for i in cells["forward"]["inputs"]] == ["layer_input", "h_prev"]
    weights = {k: v.astype(np.float64) for k, v in cells["weights"].items()}
    x = torch.randn(1, 11, 7, dtype=torch.float64)
    want = gru(x)[0][0].detach().numpy()
    seq = x[0].numpy()
    h, fwd = np.zeros(5), []
    for t in range(11):
        h = _interpret(cells["forward"], weights, seq[t], h)
        fwd.append(h)
    h, bwd = np.zeros(5), [None] * 11
    for t in reversed(range(11)):
        h = _interpret(cells["backward"], weights, seq[t], h)
        bwd[t] = h
    got = np.concatenate([np.stack(fwd), np.stack(bwd)], 1)
    # The weights are float32 in the file; that rounding is the whole difference.
    assert np.abs(got - want).max() < 1e-6


def test_a_stacked_gru_is_refused():
    with pytest.raises(NotImplementedError, match="num_layers=2"):
        build_gru_cell_topologies(torch.nn.GRU(4, 3, num_layers=2), "x.")


# -- AuroraTTS's DSP re-expressions ------------------------------------------------------------------

def _frames(seed=0, t=9, c=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, c, t, generator=g, dtype=torch.float64) * 300 + 80


def test_the_lerp_kernel_is_upstreams_align_corners_interpolate():
    frames = _frames()
    t = frames.shape[-1]
    kernel = torch.tensor(A._frame_kernels()["lerp"]).view(1, 1, -1).repeat(3, 1, 1)
    got = A._SynthWrapper._per_sample(frames, kernel)
    want = F.interpolate(frames, size=(t - 1) * A.HOP + 1, mode="linear", align_corners=True)[..., :-1]
    assert got.shape == want.shape and torch.abs(got - want).max() < 1e-9


def test_the_phase_kernels_are_the_running_sum_of_the_interpolated_f0():
    """Each sample's phase = (its frame's start, from the host) + (the closed-form sum inside the
    frame); together that is upstream's float64 `cumsum(f0s * 2*pi/sr)`."""
    f0 = _frames(1, t=12, c=1)
    t, w = f0.shape[-1], 2 * math.pi / 22050
    f0s = F.interpolate(f0, size=(t - 1) * A.HOP + 1, mode="linear", align_corners=True)[0, 0, :-1]
    want = torch.cumsum(f0s * w, 0)
    starts = [0.0]
    p = 0.0
    for j in range(t - 1):
        starts.append(p)
        p += (A.HOP * f0[0, 0, j] + (A.HOP - 1) / 2 * (f0[0, 0, j + 1] - f0[0, 0, j])).item() * w
    k = {n: torch.tensor(v).view(1, 1, -1) for n, v in A._frame_kernels().items()}
    got = (A._SynthWrapper._per_sample(torch.tensor([[starts]], dtype=torch.float64), k["box_next"])
           + A._SynthWrapper._per_sample(f0, k["phase"]) * w)[0, 0]
    assert torch.abs(got - want).max() < 1e-9


def test_the_hat_sum_is_upstreams_gather_and_lerp():
    kh = 40
    g = torch.Generator().manual_seed(2)
    env = torch.randn(1, 7, kh, generator=g, dtype=torch.float64)               # (B, T, kh)
    pos = torch.rand(1, 7, 80, generator=g, dtype=torch.float64) * (kh - 1)
    pos[0, 0, :3] = torch.tensor([0.0, kh - 1.0, 17.0])                         # both ends and an integer
    i0 = pos.floor().clamp(max=kh - 2)
    fr = pos - i0
    i0 = i0.long()
    want = env.gather(-1, i0) * (1 - fr) + env.gather(-1, i0 + 1) * fr       # HNSynth.forward
    bands = torch.arange(kh, dtype=torch.float64).view(1, 1, -1, 1)
    p = pos.transpose(1, 2)                                                     # (B, H, T)
    got = (F.relu(1 - (p[:, :, None, :] - bands).abs()) * env.transpose(1, 2)[:, None]).sum(2)
    assert torch.abs(got.transpose(1, 2) - want).max() < 1e-12


def test_the_symbol_table_is_upstreams_with_the_boundary_spelled_bar():
    assert len(A.SYMBOLS) == 79 and A.SYMBOLS[:2] == ["<pad>", "<sil>"] and A.SYMBOLS[71] == "|"
    assert A.SYMBOLS.index("DH") == 5 and A.SYMBOLS.index("AH0") == 32 and A.SYMBOLS[73] == "."
    assert len(set(A.SYMBOLS)) == 79
