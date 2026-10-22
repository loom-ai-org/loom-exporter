"""
Confirms EXPORT-IMPROVEMENT-BACKLOG.md item 4's STFT/complex-dialect finding actually holds end-to-end:
`torch.stft` decomposes via coremltools' own `common::lower_complex_dialect_ops` pass into ops this
exporter now fully covers (the new "pad"/"conv_transpose" handling in exporter.py), and the new
`loom_exporter/istft.py` module lets the inverse transform -- which coremltools' torch frontend
has no handler for at all -- flow through the same standard pipeline instead of needing a bespoke
hand-derived path (the way Kokoro's own STFT/ISTFT does today, outside this compiler entirely).
"""
import os
import unittest

import numpy as np
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


class _NoFft:
    """LOOM_EXPORT_NO_FFT for the duration: both the trace (ISTFT) and the conversion (STFT) read it."""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get("LOOM_EXPORT_NO_FFT")
        os.environ["LOOM_EXPORT_NO_FFT"] = "1" if self.on else "0"

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("LOOM_EXPORT_NO_FFT")
        else:
            os.environ["LOOM_EXPORT_NO_FFT"] = self.old


def _export_round_trip(no_fft: bool, module=None, n=16000):
    m = (module or RoundTripModule()).eval()
    x = torch.randn(1, n)
    with _NoFft(no_fft):
        traced = torch.jit.trace(m, (x,))
        prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal",
                          compute_precision=ct.precision.FLOAT32)
        exporter = LoomGGUFExporter(prog, output_path="test_stft_output.gguf", architecture="stft_test")
        try:
            path = exporter.export()
            assert Path(path).exists()
        finally:
            Path("test_stft_output.gguf").unlink(missing_ok=True)
    return exporter, next(iter(exporter.topologies.values()))


class TestStftExport(unittest.TestCase):
    def test_stft_istft_round_trip_exports_without_bespoke_ops(self):
        _, topo = _export_round_trip(no_fft=True)
        ops_used = {node["op"] for node in topo["nodes"]}
        # Every op the STFT->ISTFT decomposition needs must be a real Loom primitive -- if any of
        # these were still missing, .export() itself would have raised NotImplementedError above.
        self.assertIn("PAD_1D_REFLECT", ops_used)  # STFT's center-framing reflect-pad
        self.assertIn("CONV_1D", ops_used)  # STFT's DFT-as-convolution
        self.assertIn("CONV_TRANSPOSE_1D", ops_used)  # ISTFT's synthesis + wsum normalization
        self.assertNotIn("ISTFT", ops_used)
        self.assertNotIn("STFT", ops_used)

    def test_stft_lowers_to_one_stft_node(self):
        """ADR-080: `torch.stft` is one engine STFT node with two outputs and no convolution, its
        window a periodic Hann of win_length centred in n_fft zeros, as torch.stft pads it."""
        class Spec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("window", torch.hann_window(400))

            def forward(self, x):
                s = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=self.window,
                               return_complex=True, center=True)
                return s.real * s.real + s.imag * s.imag

        exporter, topo = _export_round_trip(no_fft=False, module=Spec())
        nodes = [n for n in topo["nodes"] if n["op"] == "STFT"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(nodes[0]["outputs"]), 2)
        self.assertNotIn("CONV_1D", {n["op"] for n in topo["nodes"]})
        self.assertEqual(nodes[0]["attrs"], {"n_fft": 512, "hop": 160})
        window = np.asarray(exporter.weights[nodes[0]["inputs"][1]], dtype=np.float64).reshape(-1)
        want = np.zeros(512)
        want[56:456] = torch.hann_window(400).numpy()   # the model's own f32 buffer, carried exactly
        np.testing.assert_array_equal(window, want)

    def test_istft_traces_to_one_istft_node(self):
        """ADR-080: under a trace, ISTFT is one engine ISTFT node and no convolutions, with the window
        as its third input and the frame count carried into its output length."""
        exporter, topo = _export_round_trip(no_fft=False)
        self.assertEqual(sum(n["op"] == "STFT" for n in topo["nodes"]), 1)
        nodes = [n for n in topo["nodes"] if n["op"] == "ISTFT"]
        self.assertEqual(len(nodes), 1)
        self.assertNotIn("CONV_TRANSPOSE_1D", {n["op"] for n in topo["nodes"]})
        self.assertEqual(nodes[0]["attrs"], {"n_fft": 400, "hop": 160, "center": 1})
        window = np.asarray(exporter.weights[nodes[0]["inputs"][2]], dtype=np.float64).reshape(-1)
        np.testing.assert_array_equal(window, ISTFT(400, 160).window.numpy())   # istft.py's f64 Hann, cast

    def test_the_traced_op_computes_what_the_module_does(self):
        """`loom::istft`'s torch body is the engine ISTFT's contract; hold it against the module's
        convolutions, center and not, on spectra that are not self-consistent."""
        torch.manual_seed(1)
        for n_fft, hop, center in ((400, 160, True), (16, 4, False), (20, 5, True)):
            m = ISTFT(n_fft=n_fft, hop_length=hop, center=center).eval()
            real = torch.randn(1, n_fft // 2 + 1, 23)
            imag = torch.randn(1, n_fft // 2 + 1, 23)
            want = m(real, imag)
            got = torch.ops.loom.istft(real, imag, m.window, n_fft, hop, center)
            self.assertEqual(want.shape, got.shape)
            finite = torch.isfinite(want)
            self.assertTrue(torch.equal(finite, torch.isfinite(got)))   # center=False: 0/0 at sample 0
            self.assertLess((want[finite] - got[finite]).abs().max().item(), 1e-5)

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


class MagnitudeModule(torch.nn.Module):
    """`|stft(x)|` at N=1024, the Qwen3-TTS speaker encoder's front end without its mel."""

    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length

    def forward(self, x):
        spec = torch.stft(x, self.n_fft, hop_length=self.hop_length, window=None, center=False,
                          return_complex=True)
        return torch.view_as_real(spec).pow(2).sum(-1)


class TestDftBasisPrecision(unittest.TestCase):
    """coremltools folds the DFT basis from `outer(k, n) * 2*pi / N` in fp32, and `k*n` is large enough
    by N=1024 that the angle has lost ~0.25 rad of precision before the divide: the basis came out
    1.4e-4 from the true one. `torch_patches` rebuilds it from `k*n mod N` at f64."""

    def test_the_folded_basis_is_the_dft_to_f32_precision(self):
        n_fft = 1024
        x = torch.randn(1, 4 * n_fft)
        # the basis exists only where the STFT is not an FFT (ADR-080)
        with _NoFft(True):
            prog = ct.convert(torch.jit.trace(MagnitudeModule(n_fft).eval(), (x,)),
                              inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal",
                              compute_precision=ct.precision.FLOAT32)
            exporter = LoomGGUFExporter(prog, output_path="test_dft_basis.gguf", architecture="dft_test")
            try:
                exporter.export()
            finally:
                Path("test_dft_basis.gguf").unlink(missing_ok=True)

        rows = n_fft // 2 + 1
        k, n = np.arange(rows)[:, None], np.arange(n_fft)[None]
        angle = 2 * np.pi * ((k * n) % n_fft) / n_fft
        # The axis pad (ADR-081) widens the basis to 516 rows for the direct conv's OC % 4; its record
        # says how many are real, and the rest must be zeros.
        bases = []
        for name, w in exporter.weights.items():
            w = np.asarray(w, dtype=np.float64)
            if name in exporter.packing.padded:
                real = exporter.packing.padded[name][0]
                self.assertFalse(np.any(w[real:]), f"{name}'s padded rows must be zero")
                w = w[:real]
            if w.size == rows * n_fft:
                bases.append(w.reshape(rows, n_fft))
        self.assertEqual(len(bases), 2, "expected one cos and one sin kernel")
        for basis in bases:
            err = min(np.abs(basis - np.cos(angle)).max(), np.abs(basis - np.sin(angle)).max(),
                      np.abs(basis + np.sin(angle)).max())
            # f32 rounding of a unit-magnitude value is 6e-8; the unpatched basis is 1.4e-4 out.
            self.assertLess(err, 1e-7)


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
