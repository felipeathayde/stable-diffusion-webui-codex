from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from typing import Optional, Tuple


def split_heads(x, num_heads: int) -> Tuple[object, int]:
    """x: [B, L, C] -> [B, H, L, D]."""
    import torch
    B, L, C = x.shape
    D = C // num_heads
    return x.view(B, L, num_heads, D).permute(0, 2, 1, 3).contiguous(), D


def merge_heads(x):
    """x: [B, H, L, D] -> [B, L, H*D]."""
    import torch
    B, H, L, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)


def sdpa_cross(q, k, v, *, dropout_p: float = 0.0, is_causal: bool = False):
    import torch
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p, is_causal=is_causal
    )


def cross_attention(x, ctx, *, wq, bq, wk, bk, wv, bv, wo, bo, num_heads: int, norm_q_w, norm_k_w, dtype: Optional[str] = None, scale=None, shift=None):
    """Cross-attention block with RMSNorm on q/k and SDPA flash.

    Shapes: x=[B,L,C], ctx=[B,Lc,C], weights are dequantized on-demand.
    """
    from .ops import linear, rms_norm

    q_in = rms_norm(x, norm_q_w, dtype=dtype)
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = linear(q_in, wq, bq, dtype=dtype)
    k = linear(rms_norm(ctx, norm_k_w, dtype=dtype), wk, bk, dtype=dtype)
    v = linear(ctx, wv, bv, dtype=dtype)

    qh, _ = split_heads(q, num_heads)
    kh, _ = split_heads(k, num_heads)
    vh, _ = split_heads(v, num_heads)

    ah = sdpa_cross(qh, kh, vh, dropout_p=0.0, is_causal=False)
    a = merge_heads(ah)
    out = linear(a, wo, bo, dtype=dtype)
    return x + out


def self_attention(x, *, wq, bq, wk, bk, wv, bv, wo, bo, num_heads: int, norm_q_w, norm_k_w, dtype: Optional[str] = None, scale=None, shift=None):
    """Self-attention with RMSNorm and optional scale/shift from time modulation."""
    from .ops import linear, rms_norm

    q_in = rms_norm(x, norm_q_w, dtype=dtype)
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = linear(q_in, wq, bq, dtype=dtype)
    k = linear(rms_norm(x, norm_k_w, dtype=dtype), wk, bk, dtype=dtype)
    v = linear(x, wv, bv, dtype=dtype)

    qh, _ = split_heads(q, num_heads)
    kh, _ = split_heads(k, num_heads)
    vh, _ = split_heads(v, num_heads)

    ah = sdpa_cross(qh, kh, vh, dropout_p=0.0, is_causal=False)
    a = merge_heads(ah)
    out = linear(a, wo, bo, dtype=dtype)
    return x + out
