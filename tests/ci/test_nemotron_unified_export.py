"""Nemotron-Labs-Unified-ASR's leaf, the parts that need no checkpoint: which recognizer claims a `.nemo`,
and the prompt fold the encoder phase is built on.

The streaming step's mask is the other (loom.cpp ADR-110): NeMo's `chunked_limited_with_rc` mask with the
context sizes as tensors, graded against a transcription of NeMo's own code, and shadowed onto the
shared encoder only while the step runs.

The fold is the leaf's first piece of arithmetic: NeMo writes `prompt_kernel(cat([h, onehot(p)]))` with a
`scatter_`, and the leaf computes `W_h h + W_p[:, p] + b` with an embedding lookup. Checked here against
NeMo's own spelling on a toy module, so a transposed table or a swapped half fails in CI rather than in
a transcript.

Run: ~/.venvs/piper/bin/python3 -m pytest tests/ci/test_nemotron_unified_export.py
"""
import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom_exporter.nemo_asr_export import _is_parakeet_rnnt, _is_parakeet_tdt  # noqa: E402
from loom_exporter.nemotron_unified_export import (  # noqa: E402
    UnifiedEncoderStepPhase, _UnifiedEncoderWrapper, _is_nemotron_unified, chunked_masks,
)


def _archive(tmp_path: Path, name: str, config: dict) -> Path:
    path = tmp_path / f"{name}.nemo"
    data = yaml.safe_dump(config).encode()
    with tarfile.open(path, "w") as t:
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return path


def test_each_recognizer_claims_only_its_own_archive(tmp_path):
    unified = _archive(tmp_path, "unified", {
        "target": "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt",
        "model_defaults": {"num_prompts": 128}})
    rnnt = _archive(tmp_path, "rnnt", {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel", "model_defaults": {}})
    assert _is_nemotron_unified(unified)
    assert not _is_parakeet_rnnt(unified) and not _is_parakeet_tdt(unified)
    assert _is_parakeet_rnnt(rnnt) and not _is_nemotron_unified(rnnt)


class _Passthrough(nn.Module):
    """Stands in for NeMo's preprocessor and encoder: hands `[B, D, T]` features straight through -- with
    one extra ZEROED frame from the preprocessor, as NeMo's centred STFT emits (`n // hop + 1` frames, the
    last past the valid length), which the wrapper must drop."""

    def __init__(self, hidden: int, frames: int):
        super().__init__()
        self.hidden, self.frames = hidden, frames

    def forward(self, input_signal=None, length=None, audio_signal=None):
        if audio_signal is not None:
            return audio_signal, length
        features = input_signal.reshape(1, self.hidden, self.frames)
        return torch.cat([features, torch.zeros(1, self.hidden, 1)], dim=2), length


@pytest.mark.parametrize("prompt", [0, 5, 7])
def test_the_prompt_fold_is_the_concat_it_replaces(prompt):
    torch.manual_seed(0)
    hidden, n_prompts, frames, inter = 6, 8, 5, 10
    kernel = nn.Sequential(nn.Linear(hidden + n_prompts, inter), nn.ReLU(), nn.Linear(inter, hidden))
    model = SimpleNamespace(
        preprocessor=_Passthrough(hidden, frames), encoder=_Passthrough(hidden, frames),
        prompt_kernel=kernel, num_prompts=n_prompts,
        cfg=SimpleNamespace(encoder=SimpleNamespace(d_model=hidden)),
    )
    wrapper = _UnifiedEncoderWrapper(model).eval()
    x = torch.randn(1, hidden * frames)
    with torch.no_grad():
        got = wrapper(x, torch.tensor([frames]), torch.tensor([prompt]))
        assert got.shape[1] == frames                    # the preprocessor's extra frame is gone
        # NeMo's own spelling (`EncDecRNNTBPEModelWithPrompt.forward`): a scattered one-hot, concatenated.
        h = x.reshape(1, hidden, frames).transpose(1, 2)
        onehot = torch.zeros(1, frames, n_prompts)
        onehot.scatter_(2, torch.tensor([prompt]).view(1, 1, 1).expand(-1, frames, -1), 1.0)
        want = kernel(torch.cat([h, onehot], dim=-1))
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_a_kernel_of_another_shape_is_refused():
    model = SimpleNamespace(
        preprocessor=_Passthrough(4, 2), encoder=_Passthrough(4, 2),
        prompt_kernel=nn.Sequential(nn.Linear(4 + 3, 4)), num_prompts=3,
        cfg=SimpleNamespace(encoder=SimpleNamespace(d_model=4)),
    )
    with pytest.raises(ValueError, match="Linear, activation, Linear"):
        _UnifiedEncoderWrapper(model)


def _nemo_chunked_masks(att_context_size, padding_length, max_audio_length):
    """`ConformerEncoder._create_masks` for `chunked_limited_with_rc`, transcribed from NeMo 3.0
    (`modules/conformer_encoder.py`; main is the same) with its `offset=None` branch: the reference."""
    att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool)
    left_context_frames, chunk_size_frames, right_context_frames = att_context_size
    frame_idx = torch.arange(0, max_audio_length, dtype=torch.int)
    chunk_idx = torch.div(frame_idx, chunk_size_frames, rounding_mode="trunc")
    window_start = chunk_idx * chunk_size_frames - left_context_frames
    window_start = torch.maximum(window_start, torch.zeros_like(window_start))
    window_end = chunk_idx * chunk_size_frames + chunk_size_frames - 1 + right_context_frames
    window_end = torch.minimum(window_end, torch.full_like(window_end, max_audio_length - 1))
    j_indices = frame_idx.unsqueeze(0)
    chunked_limited_mask = torch.logical_and(j_indices >= window_start.unsqueeze(1),
                                             j_indices <= window_end.unsqueeze(1))
    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
    pad_mask = torch.arange(0, max_audio_length).expand(padding_length.size(0), -1) < padding_length.unsqueeze(-1)
    pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
    pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
    att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask[:, :max_audio_length, :max_audio_length])
    return ~pad_mask, ~att_mask


@pytest.mark.parametrize("context", [(70, 7, 7), (70, 1, 3), (70, 2, 5), (0, 1, 1), (3, 13, 13), (200, 26, 13)])
@pytest.mark.parametrize("frames,valid", [(84, 84), (84, 80), (12, 12), (95, 95), (1, 1)])
def test_the_chunked_mask_is_nemos(context, frames, valid):
    want_pad, want_att = _nemo_chunked_masks(list(context), torch.tensor([valid]), frames)
    got_pad, got_att = chunked_masks(*(torch.tensor([float(v)]) for v in context), torch.tensor([valid]), frames)
    assert torch.equal(got_pad, want_pad)
    assert torch.equal(got_att, want_att)


def test_a_wrong_window_edge_is_caught():
    # The grader can fail: the window one frame short on the right is a different mask.
    want = _nemo_chunked_masks([70, 7, 7], torch.tensor([84]), 84)[1]
    got = chunked_masks(torch.tensor([70.0]), torch.tensor([7.0]), torch.tensor([6.0]), torch.tensor([84]), 84)[1]
    assert not torch.equal(got, want)


class _MaskRecorder(nn.Module):
    """Stands in for NeMo's encoder: asks itself for its masks the way `forward_internal` does, and
    records what it got; its own `_create_masks` is the offline one (nothing masked)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden, self.seen = hidden, []

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        none = torch.zeros(1, max_audio_length, max_audio_length, dtype=torch.bool)
        return torch.zeros(1, max_audio_length, dtype=torch.bool), none

    def forward(self, audio_signal=None, length=None):
        frames = audio_signal.shape[2]
        _, att = self._create_masks([-1, -1, -1], length, frames, None, audio_signal.device)
        self.seen.append(att)
        return audio_signal, length


def test_the_step_masks_only_while_it_runs():
    torch.manual_seed(0)
    hidden, n_prompts, frames = 4, 3, 20
    encoder = _MaskRecorder(hidden)
    model = SimpleNamespace(
        preprocessor=_Passthrough(hidden, frames), encoder=encoder,
        prompt_kernel=nn.Sequential(nn.Linear(hidden + n_prompts, 6), nn.ReLU(), nn.Linear(6, hidden)),
        num_prompts=n_prompts, cfg=SimpleNamespace(encoder=SimpleNamespace(d_model=hidden)),
    )
    wrapper = _UnifiedEncoderWrapper(model).eval()
    step = UnifiedEncoderStepPhase(wrapper).eval()
    x, length, prompt = torch.randn(1, hidden * frames), torch.tensor([frames]), torch.tensor([1])
    with torch.no_grad():
        offline = wrapper(x, length, prompt)
        streamed = step(x, length, prompt, torch.tensor([4.0]), torch.tensor([3.0]), torch.tensor([2.0]))
        again = wrapper(x, length, prompt)
    assert torch.equal(offline, streamed) and torch.equal(offline, again)   # a passthrough ignores masks
    assert not encoder.seen[0].any() and not encoder.seen[2].any()          # the offline masks, before and after
    want = _nemo_chunked_masks([4, 3, 2], length, frames)[1]
    assert torch.equal(encoder.seen[1], want)                                # the stream's, while it ran
    assert "_create_masks" not in vars(encoder)                              # nothing left shadowing it


class _MaskedSum(nn.Module):
    """The step's mask over a dynamic frame count, used the way NeMo's attention uses it (a masked fill
    of `[T, T]` scores), so the conversion keeps all of it."""

    def forward(self, x, att_left, att_chunk, att_right):
        frames = x.shape[1]
        _, att = chunked_masks(att_left, att_chunk, att_right, torch.tensor([frames]), frames)
        scores = torch.matmul(x, x.transpose(1, 2))
        return scores.masked_fill(att, -10000.0)


def _integer_operands(topology: dict) -> list:
    """The MUL_MATs one of whose operands is an integer tensor: an `i32` CAST or a bare RANGE_1D, seen
    through the shape ops between it and the matmul. ggml's CPU matmul refuses one (an int32 arange
    reached the engine this way once)."""
    nodes = topology["nodes"]
    producer = {out: n for n in nodes for out in n.get("outputs", [])}
    shape_ops = {"RESHAPE", "PERMUTE", "CONT", "VIEW", "REPEAT", "TRANSPOSE"}

    def integer(name):
        n = producer.get(name)
        while n is not None and n["op"] in shape_ops:
            n = producer.get(n["inputs"][0])
        if n is None:
            return False
        return n["op"] == "RANGE_1D" or (n["op"] == "CAST" and n["attrs"].get("dtype") == "i32")

    return [n for n in nodes if n["op"] == "MUL_MAT" and any(integer(i) for i in n["inputs"])]


def test_the_mask_lowers_to_float_matmuls():
    import coremltools as ct
    import numpy as np

    from loom_exporter.multi_phase_export import ExportPhase
    from loom_exporter.phase_conversion import convert_phase

    frames = ct.RangeDim(1, 64)
    phase = ExportPhase(
        name="masked", wrapper=_MaskedSum().eval(),
        dummy_inputs=(torch.randn(1, 20, 4), torch.tensor([4.0]), torch.tensor([3.0]), torch.tensor([2.0])),
        mil_inputs=[ct.TensorType(name="x", shape=(1, frames, 4), dtype=np.float32)]
        + [ct.TensorType(name=n, shape=(1,), dtype=np.float32) for n in ("att_left", "att_chunk", "att_right")],
    )
    topology = convert_phase(phase).topologies["masked"]
    assert [n for n in topology["nodes"] if n["op"] == "MUL_MAT"], "the outer products were folded away"
    assert _integer_operands(topology) == []
