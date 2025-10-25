from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from typing import Optional


def _resolve_dtype(dtype: Optional[str]):
    import torch
    if dtype is None:
        return None
    return {
        "bf16": getattr(torch, "bfloat16", torch.float16),
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.float16)


def to_dtype(x, dtype: Optional[str]):
    if dtype is None:
        return x
    target = _resolve_dtype(dtype)
    return x.to(target)


def _prepare_param(tensor, *, device, dtype: Optional[str]):
    if tensor is None:
        return None
    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor
    if hasattr(tensor, "gguf_cls"):
        tensor = dequantize_tensor(tensor)
    if dtype is not None:
        tensor = to_dtype(tensor, dtype)
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def linear(x, w, b=None, *, dtype: Optional[str] = None):
    """F.linear with on-demand dequant and dtype cast; x=[B,L,C], w=[C,C] or [out,in]."""
    import torch

    device = x.device
    target_dtype = dtype

    x_in = to_dtype(x, target_dtype)
    if x_in.device != device:
        x_in = x_in.to(device)

    W = _prepare_param(w, device=device, dtype=target_dtype)
    bias = _prepare_param(b, device=device, dtype=target_dtype) if b is not None else None

    # torch.nn.functional.linear expects shape [..., in] @ [out,in]^T
    y = torch.nn.functional.linear(x_in, W, bias=bias)
    return y


def rms_norm(x, weight, eps: float = 1e-6, *, dtype: Optional[str] = None):
    import torch
    # x: [B, L, C], weight: [C]
    device = x.device
    if dtype is not None:
        x = to_dtype(x, dtype)
    if x.device != device:
        x = x.to(device)
    w = _prepare_param(weight, device=device, dtype=dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(var + eps)
    return x_hat * w
