"""`torch_patches.py`'s third patch: `torch.reciprocal` must mean `1/x`.

coremltools lowers `aten::reciprocal` to MIL's `inverse`, whose `epsilon` input defaults to **1e-4**
and is ADDED to the operand -- `y = 1 / (x + epsilon)`. torch's own reciprocal adds nothing, so
without the patch every `.reciprocal()` in a traced model computes a different function, and does it
silently: the operand is usually a parameter, so the expression const-folds into a weight and no
shape, no op count and no error distinguishes a folded `1/(a + 1e-4)` from a folded `1/a`.

It was found on SNAC, whose Snake activation is `x + (1/(a + 1e-9)) * sin(a*x)^2`. With `a` around
0.2-0.8 the folded constant was off by 2.3e-4 relative, which put the decoded waveform 9.2e-3 from the
reference in relative RMS -- against 1.8e-6 for the same graph's own f32-vs-f64 spread. The error is
unbounded as `a` approaches the epsilon.
"""
import numpy as np
import pytest


def _folded_constant(size: int = 3):
    """Trace `x * (a + 1e-9).reciprocal()` and return the constant the converter folded it to."""
    import torch

    import loom_exporter  # noqa: F401  applies the patches at import
    import coremltools as ct

    alpha = torch.tensor([[0.2], [0.5], [2.0]])

    class _Reciprocal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(alpha)

        def forward(self, x):
            return x * (self.a + 1e-9).reciprocal()

    traced = torch.jit.trace(_Reciprocal().eval(), (torch.ones(size, 1),))
    program = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=(size, 1))],
                         convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
    folded = [np.asarray(op.outputs[0].val).reshape(-1)
              for function in program.functions.values() for op in function.operations
              if op.op_type == "const" and op.outputs[0].val is not None
              and np.asarray(op.outputs[0].val).size == size]
    assert len(folded) == 1, f"expected one folded constant, got {len(folded)}"
    return alpha.numpy().reshape(-1), folded[0]


def test_a_traced_reciprocal_folds_to_one_over_x():
    pytest.importorskip("coremltools")
    alpha, folded = _folded_constant()
    exact = 1.0 / (alpha.astype(np.float32) + np.float32(1e-9))
    assert np.allclose(folded, exact, rtol=1e-6, atol=0), f"{folded} != {exact}"


def test_the_default_the_patch_overrides_is_still_the_one_worth_overriding():
    """The control arm, stated as the number rather than by un-registering the patch: `inverse`'s
    epsilon default is what makes the patch load-bearing, and if coremltools ever changes it this is
    the test that says so rather than the patch quietly becoming a no-op (or, if they change it to
    something else, quietly becoming wrong)."""
    pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(3,))])
    def program(x):
        return mb.inverse(x=x)

    epsilon = program.functions["main"].find_ops(op_type="inverse")[0].epsilon.val
    assert float(epsilon) == pytest.approx(1e-4), (
        "MIL `inverse` no longer defaults to 1e-4 -- re-read torch_patches.py's third patch"
    )
