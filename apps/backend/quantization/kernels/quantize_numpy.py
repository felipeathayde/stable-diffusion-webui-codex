from __future__ import annotations

import numpy as np

from ..core import QuantType


def _np_roundf(values: np.ndarray) -> np.ndarray:
    """Round like ggml (round-half-up) for quant packing."""
    abs_values = np.abs(values)
    floored = np.floor(abs_values)
    delta = floored + np.floor(2 * (abs_values - floored))
    return np.sign(values) * delta


def quantize_blocks_q8_0(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q8_0 packed blocks (n, 34)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q8_0 quantize expects (n_blocks, 32), got {blocks.shape}")
    x = blocks.astype(np.float32, copy=False)
    d = np.abs(x).max(axis=1, keepdims=True) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    qs = _np_roundf(x * inv)
    header = d.astype(np.float16).view(np.uint8)
    payload = qs.astype(np.int8).view(np.uint8)
    return np.concatenate([header, payload], axis=1)


def _pack_k_scale_min(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Pack Q4_K/Q5_K scale+min 6-bit tables into the 12-byte GGML layout."""
    if scales.dtype != np.uint8 or mins.dtype != np.uint8:
        raise TypeError("scales/mins must be uint8 arrays")
    if scales.shape != mins.shape:
        raise ValueError(f"scales/mins shape mismatch: {scales.shape} vs {mins.shape}")
    if scales.ndim != 2 or scales.shape[1] != 8:
        raise ValueError(f"expected (n_blocks, 8) scales/mins, got {scales.shape}")

    sc0 = scales[:, :4]
    sc1 = scales[:, 4:]
    mn0 = mins[:, :4]
    mn1 = mins[:, 4:]

    if np.any(sc0 > 0x3F) or np.any(sc1 > 0x3F) or np.any(mn0 > 0x3F) or np.any(mn1 > 0x3F):
        raise ValueError("scales/mins contain values outside 6-bit range (0..63)")

    d = (sc0 & np.uint8(0x3F)) | ((sc1 & np.uint8(0x30)) << np.uint8(2))
    m = (mn0 & np.uint8(0x3F)) | ((mn1 & np.uint8(0x30)) << np.uint8(2))
    m_d = ((mn1 & np.uint8(0x0F)) << np.uint8(4)) | (sc1 & np.uint8(0x0F))

    packed = np.stack([d, m, m_d], axis=1)  # (n_blocks, 3, 4)
    return packed.reshape((scales.shape[0], 12))


def quantize_blocks_q4_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q4_K packed blocks (n, 144)."""
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"Q4_K quantize expects (n_blocks, 256), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    n_blocks = x.shape[0]
    groups = x.reshape((n_blocks, 8, 32))

    g_min = groups.min(axis=-1)
    g_max = groups.max(axis=-1)
    g_min = np.minimum(g_min, 0.0)
    dm = -g_min

    scale = (g_max - g_min) / np.float32(15.0)

    max_scale = scale.max(axis=-1, keepdims=True)
    d = (max_scale / np.float32(63.0)).astype(np.float32)
    sc = np.where(d == 0, 0, np.rint(scale / d)).astype(np.int32)
    sc = np.clip(sc, 0, 63).astype(np.uint8)

    max_dm = dm.max(axis=-1, keepdims=True)
    dmin = (max_dm / np.float32(63.0)).astype(np.float32)
    mn = np.where(dmin == 0, 0, np.rint(dm / dmin)).astype(np.int32)
    mn = np.clip(mn, 0, 63).astype(np.uint8)

    scale_q = d * sc.astype(np.float32)
    dm_q = dmin * mn.astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            scale_q[:, :, None] == 0,
            0,
            np.rint((groups + dm_q[:, :, None]) / scale_q[:, :, None]),
        )
    q = np.clip(q, 0, 15).astype(np.uint8)

    header_d = d.astype(np.float16).view(np.uint8)
    header_dmin = dmin.astype(np.float16).view(np.uint8)
    scales_packed = _pack_k_scale_min(sc, mn)

    even = q[:, 0:8:2, :]
    odd = q[:, 1:8:2, :]
    qs = (even & np.uint8(0x0F)) | (odd << np.uint8(4))
    qs = qs.reshape((n_blocks, 128))

    return np.concatenate([header_d, header_dmin, scales_packed, qs], axis=-1)


def quantize_blocks_q5_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q5_K packed blocks (n, 176)."""
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"Q5_K quantize expects (n_blocks, 256), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    n_blocks = x.shape[0]
    groups = x.reshape((n_blocks, 8, 32))

    g_min = groups.min(axis=-1)
    g_max = groups.max(axis=-1)
    g_min = np.minimum(g_min, 0.0)
    dm = -g_min

    scale = (g_max - g_min) / np.float32(31.0)

    max_scale = scale.max(axis=-1, keepdims=True)
    d = (max_scale / np.float32(63.0)).astype(np.float32)
    sc = np.where(d == 0, 0, np.rint(scale / d)).astype(np.int32)
    sc = np.clip(sc, 0, 63).astype(np.uint8)

    max_dm = dm.max(axis=-1, keepdims=True)
    dmin = (max_dm / np.float32(63.0)).astype(np.float32)
    mn = np.where(dmin == 0, 0, np.rint(dm / dmin)).astype(np.int32)
    mn = np.clip(mn, 0, 63).astype(np.uint8)

    scale_q = d * sc.astype(np.float32)
    dm_q = dmin * mn.astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            scale_q[:, :, None] == 0,
            0,
            np.rint((groups + dm_q[:, :, None]) / scale_q[:, :, None]),
        )
    q = np.clip(q, 0, 31).astype(np.uint8)

    ql = q & np.uint8(0x0F)
    qh_bits = (q >> np.uint8(4)) & np.uint8(1)

    even = ql[:, 0:8:2, :]
    odd = ql[:, 1:8:2, :]
    qs = (even & np.uint8(0x0F)) | (odd << np.uint8(4))
    qs = qs.reshape((n_blocks, 128))

    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape((1, 8, 1))
    qh = (qh_bits.astype(np.uint8) * weights).sum(axis=1).astype(np.uint8)  # (n_blocks, 32)

    header_d = d.astype(np.float16).view(np.uint8)
    header_dmin = dmin.astype(np.float16).view(np.uint8)
    scales_packed = _pack_k_scale_min(sc, mn)

    return np.concatenate([header_d, header_dmin, scales_packed, qh, qs], axis=-1)


QUANTIZE_NUMPY_BY_TYPE: dict[QuantType, callable] = {
    QuantType.Q8_0: quantize_blocks_q8_0,
    QuantType.Q4_K: quantize_blocks_q4_k,
    QuantType.Q5_K: quantize_blocks_q5_k,
}

