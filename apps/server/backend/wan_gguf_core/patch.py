from __future__ import annotations

from typing import Tuple


def patch_embed(video, w, b):
    """Apply 3D patch embedding conv: video[B,C,T,H,W] -> tokens[B, T*H'*W', C_out].

    Uses kernel/stride from weight shape heuristics: k=(kT,kH,kW), stride=(1,kH,kW).
    """
    import torch
    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = video.device
    dtype = video.dtype

    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    W = W.to(device=device, dtype=dtype)

    bias = None
    if b is not None:
        bias = b.to(device=device, dtype=dtype)

    video_in = video.to(device=device, dtype=dtype)
    B, C, T, H, Wd = video.shape
    # weight: [C_out, C_in, kT, kH, kW]
    kCout, kCin, kT, kH, kW = W.shape
    if C != kCin:
        raise RuntimeError(f"patch_embed: C_in mismatch: video C={C} vs weight {kCin}")
    stride = (1, kH, kW)
    pad = (0, 0, 0)
    y = torch.nn.functional.conv3d(video_in, W, bias=bias, stride=stride, padding=pad)
    # y: [B, C_out, T, H', W'] → tokens [B, L, C_out]
    B2, Cout, T2, H2, W2 = y.shape
    tokens = y.permute(0, 2, 3, 4, 1).contiguous().view(B2, T2 * H2 * W2, Cout)
    return tokens, (T2, H2, W2)


def patch_unembed(tokens, w, out_shape: Tuple[int, int, int]):
    """Inverse of patch_embed via conv3d_transpose: tokens[B,L,C_out] -> video[B,C_in,T,H,W].

    Expects out_shape=(T,H',W'), uses kernel/stride from weight and assumes no padding.
    """
    import torch
    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = tokens.device
    dtype = tokens.dtype

    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    W = W.to(device=device, dtype=dtype)

    B, L, Cout = tokens.shape
    kCout, kCin, kT, kH, kW = W.shape
    if Cout != kCout:
        raise RuntimeError(f"patch_unembed: C_out mismatch: tokens C={Cout} vs weight {kCout}")
    T2, H2, W2 = out_shape
    y = tokens.view(B, T2, H2, W2, Cout).permute(0, 4, 1, 2, 3).contiguous().to(device=device, dtype=dtype)  # [B,Cout,T,H,W]
    stride = (1, kH, kW)
    pad = (0, 0, 0)
    video = torch.nn.functional.conv_transpose3d(y, W, bias=None, stride=stride, padding=pad)
    return video  # [B, C_in, T, H, W]
