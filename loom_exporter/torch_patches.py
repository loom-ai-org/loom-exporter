"""
Torch-frontend robustness patches for coremltools' PyTorch->MIL converter.

No patch here is model-specific -- the first two are what any HF causal-LM traced through this
pipeline wants, and the third is a plain correctness fix to one op's default -- so they're applied
once at `import loom_exporter` time (see `__init__.py`) rather than re-pasted into every export
script. The first two were previously duplicated verbatim across export_lfm2_monolithic.py, the
now-retired export_lfm2_atomic.py, and tools/convert_lfm/make_lfm2_gguf.py
(EXPORT-IMPROVEMENT-BACKLOG.md item 1).
"""
import numpy as np

_PATCHED = False


def apply_torch_frontend_patches() -> None:
    """Installs both coremltools torch-frontend patches. Idempotent -- safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from coremltools.converters.mil.mil import Builder as mb

    # 1. Support 1-element numpy array conversion in the cast op, so a compile-time-constant
    #    `.item()`-style cast folds to a const instead of producing a dynamic cast node.
    from coremltools.converters.mil.frontend.torch import ops as mil_ops
    _original_cast = mil_ops._cast

    def _robust_cast(context, node, dtype, dtype_name):
        inputs = mil_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray):
            if x.val.size == 1:
                scalar_val = dtype(x.val.item())
                res = mb.const(val=scalar_val, name=node.name)
                context.add(res, node.name)
                return
        _original_cast(context, node, dtype, dtype_name)

    mil_ops._cast = _robust_cast

    # 2. Pre-tile K/V before SDPA decomposition so grouped-query attention (mismatched Q/K head
    #    counts) traces correctly.
    from coremltools.converters.mil.frontend import _utils as mil_frontend_utils
    _original_decompose_sdpa = mil_frontend_utils._decompose_scaled_dot_product_attention

    def _robust_decompose_sdpa(q, k, v, mask, name, scale=None, before_op=None):
        q_shape = list(q.shape)
        k_shape = list(k.shape)
        rank = len(q_shape)

        if rank == 4:
            q_heads = q_shape[1]
            k_heads = k_shape[1]
            if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
                ratio = q_heads // k_heads
                if ratio > 1:
                    k = mb.tile(x=k, reps=[1, ratio, 1, 1], before_op=before_op)
                    v = mb.tile(x=v, reps=[1, ratio, 1, 1], before_op=before_op)
        elif rank == 3:
            q_heads = q_shape[0]
            k_heads = k_shape[0]
            if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
                ratio = q_heads // k_heads
                if ratio > 1:
                    k = mb.tile(x=k, reps=[ratio, 1, 1], before_op=before_op)
                    v = mb.tile(x=v, reps=[ratio, 1, 1], before_op=before_op)

        return _original_decompose_sdpa(q, k, v, mask, name, scale, before_op)

    mil_frontend_utils._decompose_scaled_dot_product_attention = _robust_decompose_sdpa

    # 3. `torch.reciprocal` must mean 1/x, not 1/(x + 1e-4).
    #
    #    coremltools' frontend lowers `aten::reciprocal` to MIL `inverse`, whose `epsilon` input
    #    DEFAULTS TO 1e-4 ("for stability") and is added to the operand: `y = 1 / (x + epsilon)`.
    #    torch's own reciprocal adds nothing, so every `.reciprocal()` that reaches this pipeline is
    #    silently computing a different function -- and it is silent twice over, because the operand is
    #    usually a parameter, so the whole thing const-folds into a weight and nothing downstream can
    #    tell the difference between a folded 1/(a+1e-4) and a folded 1/a.
    #
    #    FOUND ON SNAC, whose Snake activation is `x + (1/(a + 1e-9)) * sin(a*x)^2` with a learned
    #    per-channel `a`. The exported constant was off by up to 2.3e-4 in relative terms, which put
    #    the decoded waveform 9.2e-3 from the reference in relative RMS where the same graph at f64
    #    agrees with itself to 1.8e-6 -- a 3700x gap that reads exactly like ordinary float noise until
    #    you difference the folded constant against numpy. `a` was ~0.2-0.8 here; the error grows
    #    without bound as `a` approaches the epsilon, so a model with small alphas would be visibly
    #    wrong rather than subtly.
    #
    #    The other two `epsilon`-carrying unary ops are fine as they stand: `rsqrt` defaults to 1e-12
    #    (below f32's resolution for any operand big enough to take a square root of) and `log` to
    #    1e-45 (a denormal). `inverse` is the outlier, and 1e-4 is not a rounding-scale number.
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

    @register_torch_op(override=True)
    def reciprocal(context, node):
        inputs = mil_ops._get_inputs(context, node, expected=1)
        # Typed from the operand rather than passed as a Python float: `inverse`'s own
        # `default_inputs` builds epsilon with `nptype_from_builtin(self.x.dtype)`, and an fp16
        # program handed an fp32 const here fails type inference rather than casting.
        epsilon = types.nptype_from_builtin(inputs[0].dtype)(0.0)
        context.add(mb.inverse(x=inputs[0], epsilon=epsilon, name=node.name))

    # 4. The DFT basis `torch.stft` lowers to must be the DFT's, to f32 precision.
    #
    #    coremltools' `lower_complex_dialect_ops` builds it as `cos(outer(k, n) * 2*pi / N)` with every
    #    step in fp32. `k*n` reaches (N/2)*(N-1) -- 523,776 at N=1024 -- so once it is multiplied by
    #    2*pi the product's ulp is 0.25 rad, and dividing by N afterwards cannot give the precision back:
    #    the folded basis is off by up to 1.4e-4 where an f32 one is good to 6e-8. The error is a leak,
    #    ~1e-5 of each frame's energy spread across its other bins, so it is invisible in loud bins and
    #    dominates quiet ones -- and a log-mel then reads it as a 0.4 error on the floor bins.
    #
    #    FOUND ON QWEN3-TTS's SPEAKER ENCODER, whose x-vector sat 7.5e-4 from the reference at f64
    #    where the reference's own f32 run sat 1.3e-6. The same bound applies to every other model
    #    that traces a `torch.stft`. The replacement reduces `k*n` modulo N in integers first, so the
    #    angle is formed from a number below N, and takes the cosine at f64 before casting.
    from coremltools.converters.mil.mil.passes.defs import lower_complex_dialect_ops as _complex
    _original_dft_matrix = _complex._calculate_dft_matrix

    def _exact_dft_matrix(n_fft, onesided=False):
        size = getattr(n_fft, "val", None)
        if size is None:
            return _original_dft_matrix(n_fft, onesided)
        size = int(np.asarray(size).reshape(-1)[0])
        rows = size // 2 + 1 if onesided else size
        phase = np.outer(np.arange(rows, dtype=np.int64), np.arange(size, dtype=np.int64)) % size
        angle = (2.0 * np.pi / size) * phase
        return (mb.const(val=np.cos(angle).astype(np.float32)),
                mb.const(val=np.sin(angle).astype(np.float32)))

    _complex._calculate_dft_matrix = _exact_dft_matrix
