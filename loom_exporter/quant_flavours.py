"""Quantization flavours: `--quantize` names that are not a GGML type of their own.

`Q8_0_PC` (ADR-076) is per-channel int8 written as ordinary Q8_0: every 32-element block of a row
carries the same F16 scale, the row's absmax / 127. Any backend reads the file as plain Q8_0, unchanged.
A CPU kernel that wants one scale per output channel (ADR-079's ARMv6 dynamic int8 GEMMs, XNNPACK's
`qd8`) takes it from the first block of each row, and the tensors written this way are listed under
`PER_CHANNEL_KEY` so an engine knows which ones it can do that for without scanning the scales.

Only the weights such a kernel can read are written per channel: `mul_mat` operands and pointwise
convolution kernels (`LoomGGUFExporter._int8_readable_weight_names`). The flavour's other eligible weights
are ordinary block Q8_0, which is finer, and which ggml's convolutions read the same either way.
"""
import numpy as np

#: flavour -> the GGML type it is written as.
FLAVOURS = {"Q8_0_PC": "Q8_0"}

#: GGUF key: the names of the tensors written row-uniform (canonical names only; an alias resolves to one).
PER_CHANNEL_KEY = "loom.quantize.per_channel"


def ggml_type_name(quantize: str) -> str:
    """The GGML type a `--quantize` name is written as: itself, unless it is a flavour."""
    return FLAVOURS.get(quantize, quantize)


def is_per_channel(quantize: str) -> bool:
    return quantize == "Q8_0_PC"


def quantize_q8_0_per_channel(array: np.ndarray) -> np.ndarray:
    """`array` (float, last axis a multiple of 32) as Q8_0 bytes with one scale per row.

    The arithmetic is `gguf.quants`' Q8_0 with the row in place of the block: the scale is
    absmax / 127, the values are rounded half away from zero by its unrounded inverse, and the scale is
    stored as F16. So a row whose blocks all share the row's absmax comes out byte-identical to Q8_0.
    """
    from gguf.quants import np_roundf

    n = array.shape[-1]
    if n % 32:
        raise ValueError(f"Q8_0_PC needs a last axis that is a multiple of 32, got {array.shape}")
    rows = np.asarray(array, dtype=np.float32).reshape(-1, n)
    d = np.abs(rows).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1 / d)
    qs = np_roundf(rows * inv).astype(np.int8).view(np.uint8).reshape(rows.shape[0], n // 32, 32)
    d16 = np.broadcast_to(d.astype(np.float16).view(np.uint8)[:, None, :], (rows.shape[0], n // 32, 2))
    blocks = np.concatenate([d16, qs], axis=2)
    return np.ascontiguousarray(blocks.reshape(*array.shape[:-1], n // 32 * 34))
