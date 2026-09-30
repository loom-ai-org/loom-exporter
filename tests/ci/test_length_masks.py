"""Length-validity masks: which `arange(T) < bound` the exporter may bake all-true, and which it must not.

NeMo re-masks after every stride-2 subsampling conv with `floor((L - 1)/2) + 1`, where `L` is the mel
front end's valid length -- one less than the frame count. The conv's output is one frame longer than
that bound whenever `L` is even, and NeMo zeroes that frame. The exporter used to bake every such mask
all-true (Retro-013's "a single utterance is never padded"), which left the frame in at about half of
all lengths (Retro-065). Only a bound that provably IS the range may be baked.
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter


class SubsamplingMasks(torch.nn.Module):
    """The two shapes of mask a NeMo encoder traces, reduced to one stride-2 stage."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, waveform, length):
        # Waveform mask: the bound IS the range, so it is all-true at every length.
        t = waveform.shape[1]
        wave_mask = (torch.arange(t) < length).to(waveform.dtype)
        x = waveform * wave_mask
        # The front end's convention: one fewer valid frame than the tensor holds.
        valid = length - 1
        h = self.conv(x.unsqueeze(1))
        stage_len = (valid + 2 - 3) // 2 + 1
        stage_mask = (torch.arange(h.shape[2]) < stage_len).to(h.dtype)
        # Keeps `length` a live input even when both masks are baked, so a guard that bakes too much
        # fails the assertion below rather than the link check.
        return h * stage_mask * (length > 0).to(h.dtype)


def _topology(module, example, inputs):
    prog = ct.convert(torch.jit.trace(module.eval(), example), inputs=inputs, convert_to="milinternal")
    out = Path("test_length_masks.gguf")
    try:
        exporter = LoomGGUFExporter(prog, output_path=str(out), architecture="length_mask_test")
        exporter.export()
    finally:
        out.unlink(missing_ok=True)
    return next(iter(exporter.topologies.values()))


class TestSubsamplingMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 64
        cls.topo = _topology(
            SubsamplingMasks(),
            (torch.randn(1, n), torch.tensor([n], dtype=torch.int64)),
            [ct.TensorType(name="waveform", shape=(1, ct.RangeDim(16, 4096)), dtype=np.float32),
             ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
        )

    def test_the_stage_mask_stays_a_real_comparison(self):
        """Range `floor((T-1)/2)+1`, bound `floor((T-2)/2)+1`: equal at even T, one apart at odd T. The
        old probe-based guard saw "not always off by exactly one" and baked it all-true."""
        less = [n for n in self.topo["nodes"] if n["op"] == "LESS"]
        self.assertEqual(len(less), 1, [n["outputs"] for n in less])

    def test_the_waveform_mask_is_still_baked(self):
        baked = [n for n in self.topo["nodes"]
                 if n["op"] == "REPEAT" and n["inputs"][0].endswith("_always_valid_scalar")]
        self.assertEqual(len(baked), 1)


if __name__ == "__main__":
    unittest.main()
