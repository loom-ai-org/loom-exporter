"""Nemotron-Labs-Unified-ASR's leaf, the parts that need no checkpoint: which recognizer claims a `.nemo`,
and the prompt fold the encoder phase is built on.

The fold is the leaf's one piece of arithmetic: NeMo writes `prompt_kernel(cat([h, onehot(p)]))` with a
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
from loom_exporter.nemotron_unified_export import _UnifiedEncoderWrapper, _is_nemotron_unified  # noqa: E402


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
