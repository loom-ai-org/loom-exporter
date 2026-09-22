"""
Confirms EXPORT-IMPROVEMENT-BACKLOG.md item 4's STFT/complex-dialect finding actually holds end-to-end:
`torch.stft` decomposes via coremltools' own `common::lower_complex_dialect_ops` pass into ops this
exporter now fully covers (the new "pad"/"conv_transpose" handling in exporter.py), and the new
`loom_exporter/istft.py` module lets the inverse transform -- which coremltools' torch frontend
has no handler for at all -- flow through the same standard pipeline instead of needing a bespoke
hand-derived path (the way Kokoro's own STFT/ISTFT does today, outside this compiler entirely).
"""
import unittest

import torch
import coremltools as ct

import sys
from pathlib import Path

from loom_exporter.paths import CONVERTERS, driver_dir
import loom_exporter  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_exporter.exporter import LoomGGUFExporter
from loom_exporter.istft import ISTFT


class RoundTripModule(torch.nn.Module):
    """torch.stft -> ISTFT (this module's own, since torch.istft itself can't be traced) round trip,
    mirroring the shape a real vocoder's STFT-domain processing would take."""

    def __init__(self, n_fft=400, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, center=True)

    def forward(self, x):
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, return_complex=True, center=True,
        )
        return self.istft(spec.real, spec.imag)


class TestStftExport(unittest.TestCase):
    def test_stft_istft_round_trip_exports_without_bespoke_ops(self):
        m = RoundTripModule().eval()
        x = torch.randn(1, 16000)

        traced = torch.jit.trace(m, (x,))
        prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal")

        exporter = LoomGGUFExporter(prog, output_path="test_stft_output.gguf", architecture="stft_test")
        try:
            path = exporter.export()
            self.assertTrue(Path(path).exists())

            topo = next(iter(exporter.topologies.values()))
            ops_used = {node["op"] for node in topo["nodes"]}
            # Every op the STFT->ISTFT decomposition needs must be a real Loom primitive -- if any of
            # these were still missing, .export() itself would have raised NotImplementedError above.
            self.assertIn("PAD_1D_REFLECT", ops_used)  # STFT's center-framing reflect-pad
            self.assertIn("CONV_1D", ops_used)  # STFT's DFT-as-convolution
            self.assertIn("CONV_TRANSPOSE_1D", ops_used)  # ISTFT's synthesis + wsum normalization
        finally:
            Path("test_stft_output.gguf").unlink(missing_ok=True)

    def test_istft_matches_real_torch_istft_numerically(self):
        """The exported topology's own correctness rests on ISTFT's math matching torch.istft -- verified
        directly here (not just via export success) on random, non-self-consistent magnitude/phase, the
        same rigor kokoro_stft_common.py's own docstring requires of the equivalent ggml-graph reduction."""
        torch.manual_seed(0)
        n_fft, hop, n_frames, batch = 400, 160, 47, 2
        n_freq = n_fft // 2 + 1

        real = torch.randn(batch, n_freq, n_frames)
        imag = torch.randn(batch, n_freq, n_frames)
        window = torch.hann_window(n_fft, periodic=True)

        ref = torch.istft(
            torch.complex(real, imag), n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, center=True,
        )
        got = ISTFT(n_fft=n_fft, hop_length=hop, center=True).eval()(real, imag)

        self.assertEqual(ref.shape, got.shape)
        self.assertLess((ref - got).abs().max().item(), 1e-4)


class L2NormModule(torch.nn.Module):
    """A complex magnitude, spelled the way that reliably reaches `reduce_l2_norm`.

    `torch.stft(...).abs()` lowers through coremltools' complex dialect to the same op, which is how
    F5-TTS's `power=1` mel front end found that it had no ggml mapping -- but whether it SURVIVES to
    the emitted program depends on later MIL passes, and on that model it no longer does (its mel is
    `square/square/add/sqrt` today). So this exercises the rule directly rather than through a
    spectrogram that may or may not still produce it.
    """

    def forward(self, x):
        return torch.linalg.vector_norm(x, ord=2, dim=-1)


class TestComplexMagnitude(unittest.TestCase):
    """`reduce_l2_norm` had no ggml mapping until family 9.

    Whisper's frontend never reached it because it writes `abs() ** 2` -- the square cancels the square
    root and coremltools emits `reduce_sum_square` instead. F5-TTS's mel is `power=1`
    (`get_vocos_mel_spectrogram`), the first un-squared complex magnitude in the zoo, and its export
    stopped with `MIL op 'reduce_l2_norm' is missing a ggml mapping`.
    """

    def test_an_l2_norm_lowers_to_a_composition_of_existing_ops(self):
        m = L2NormModule().eval()
        x = torch.randn(1, 8, 5)

        traced = torch.jit.trace(m, (x,))
        prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)],
                          convert_to="milinternal")
        self.assertIn("reduce_l2_norm",
                      {op.op_type for f in prog.functions.values() for op in f.operations})

        exporter = LoomGGUFExporter(prog, output_path="test_l2_output.gguf",
                                     architecture="l2_norm_test")
        try:
            self.assertTrue(Path(exporter.export()).exists())
            topo = next(iter(exporter.topologies.values()))
            # Composed, not given a primitive of its own: `sqrt(sum(x**2))` over one axis, three ops
            # that all already existed. A new engine op would have to be maintained on every backend
            # for a shape ggml already computes.
            self.assertEqual([n["op"] for n in topo["nodes"]], ["MUL", "REDUCE_SUM", "SQRT"])
        finally:
            Path("test_l2_output.gguf").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
