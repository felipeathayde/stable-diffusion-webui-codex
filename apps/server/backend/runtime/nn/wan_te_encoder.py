from __future__ import annotations

"""
Experimental WAN T5 FP8 encoder (CUDA-backed)

This module wires the FP8 weights (uint8+scale) with our CUDA extension:
- Linear FP8 (tile dequant + GEMM)
- Attention: first pass uses PyTorch SDPA (compute fp16/bf16); projections in FP8

Strict: raises explicit errors on missing tensors or unsupported shapes. No silent fallback.
"""

from typing import Optional, Tuple
import os
import logging
import torch

from .wan_te_loader import load_umt5_xxl_fp8, WanTEFp8Weights
from .wan_te_cuda import linear_fp8, attn_fp8, available as te_ext_available

log = logging.getLogger("wan22.te.encoder")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[wan22.te.encoder] %(levelname)s: %(message)s'))
    log.addHandler(h)
log.setLevel(logging.INFO)
log.propagate = False


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w


def _embedding_fp8(indices: torch.Tensor, embed_u8: torch.Tensor, embed_scale: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # indices: [B,L], embed_u8: [V,C] (uint8 CPU), embed_scale: [V] or [1] (float)
    # Strategy: gather on CPU -> pin -> async copy to GPU -> dequantize on GPU -> cast to dtype
    B, L = indices.shape
    V, C = embed_u8.shape
    rows = indices.reshape(-1).to(torch.long).cpu()
    u8_rows = embed_u8.index_select(0, rows)  # [B*L, C] (CPU u8)
    if u8_rows.device.type == 'cpu':
        try:
            u8_rows = u8_rows.pin_memory()
        except Exception:
            pass
    sc_rows = embed_scale.index_select(0, rows).to(torch.float32).view(-1, 1)
    if sc_rows.device.type == 'cpu':
        try:
            sc_rows = sc_rows.pin_memory()
        except Exception:
            pass
    u8_gpu = u8_rows.to(device, non_blocking=True)
    sc_gpu = sc_rows.to(device, non_blocking=True)
    em = u8_gpu.to(torch.float32).mul_(sc_gpu).to(dtype)
    return em.view(B, L, C)


def _proj_fp8(x: torch.Tensor, pack, device: torch.device) -> torch.Tensor:
    # x: [B,L,Cin]; pack.w (u8+scale), pack.b optional
    w_u8 = pack.w.w_u8.to('cpu') if pack.w.w_u8.is_cuda else pack.w.w_u8
    w_scale = pack.w.scale.to('cpu') if pack.w.scale.is_cuda else pack.w.scale
    b = pack.b.to(device, x.dtype) if (pack.b is not None) else None
    # linear_fp8 expects tensors; it will tile-dequant onto device
    return linear_fp8(x, pack.w, b)


_ATTN_LOG_ONCE = False

def _attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # q/k/v: [B,H,L,D]
    B, H, L, D = q.shape
    impl_env = os.environ.get('WAN_TE_ATTN', '').lower().strip()
    # Auto-select: use CUDA path if extension available and not explicitly forced to 'sdpa'
    impl = 'cuda' if (impl_env in ('', 'auto') and te_ext_available()) else (impl_env or 'sdpa')
    global _ATTN_LOG_ONCE
    if not _ATTN_LOG_ONCE:
        _ATTN_LOG_ONCE = True
        log.info("attention_impl=%s (env=%s, ext=%s)", impl, (impl_env or 'auto'), te_ext_available())
    if impl == 'cuda' and te_ext_available():
        out, _ = attn_fp8(q, k, v, None, bool(causal))
        return out
    else:
        q2 = q.reshape(B*H, L, D)
        k2 = k.reshape(B*H, L, D)
        v2 = v.reshape(B*H, L, D)
        out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
        return out.reshape(B, H, L, D)


def encode_fp8(
    te_weights_path: str,
    *,
    input_ids: torch.Tensor,          # [B,L] long
    attention_mask: Optional[torch.Tensor],  # [B,1,1,L] or [B,L]
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    d_kv: int,
    log_metrics: bool = True,
) -> torch.Tensor:
    """Encode tokens via FP8 weights in tiles (projs) + SDPA attention.

    Returns last_hidden_state [B,L,C].
    """
    weights: WanTEFp8Weights = load_umt5_xxl_fp8(te_weights_path)
    B, L = input_ids.shape
    C = weights.d_model
    if C <= 0:
        C = d_kv * num_heads
    dt = dtype
    dev = device

    # Embedding
    h = _embedding_fp8(input_ids, weights.embed.w_u8, weights.embed.scale, dt, dev)  # [B,L,C]

    # Attention mask format: convert [B,L] -> [B,1,1,L] bool
    am = None
    if attention_mask is not None:
        am = attention_mask
        if am.dim() == 2:
            am = am.to(torch.bool).view(B, 1, 1, L)
        else:
            am = am.to(torch.bool)

    H = num_heads
    D = d_kv

    # Iterate encoder blocks with lightweight telemetry
    def _mem(label: str) -> None:
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() // (1024*1024)
                reserved = torch.cuda.memory_reserved() // (1024*1024)
                log.info("mem[%s]: alloc=%dMB reserved=%dMB", label, alloc, reserved)
        except Exception:
            pass

    _mem('te-start')
    for i in range(weights.num_layers):
        blk = weights.blocks.get(i)
        if blk is None:
            raise RuntimeError(f"TE FP8: missing block {i}")
        # LayerNorm 1 (if present)
        ln1_w = blk.get('ln1_w').b if blk.get('ln1_w') else None  # stored as bias field for convenience
        x = h
        if ln1_w is not None:
            h = _rms_norm(h, ln1_w.to(dev, dt))

        # QKV projections (FP8)
        q = _proj_fp8(h, blk['q'], dev)
        k = _proj_fp8(h, blk['k'], dev)
        v = _proj_fp8(h, blk['v'], dev)
        # reshape to [B,H,L,D]
        q = q.view(B, L, H, D).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, L, H, D).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, L, H, D).permute(0, 2, 1, 3).contiguous()
        # Attention (SDPA)
        a = _attn(q, k, v, causal=False, attn_mask=None)
        a = a.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        o = _proj_fp8(a, blk['o'], dev)
        h = x + o

        # LayerNorm 2 + FFN (gated if wi_1 present)
        ln2_w = blk.get('ln2_w').b if blk.get('ln2_w') else None
        x2 = h
        if ln2_w is not None:
            h = _rms_norm(h, ln2_w.to(dev, dt))
        wi = _proj_fp8(h, blk['wi'], dev)
        if 'wi_1' in blk:
            wi_1 = _proj_fp8(h, blk['wi_1'], dev)
            wi = torch.nn.functional.gelu(wi) * wi_1
        else:
            wi = torch.nn.functional.gelu(wi)
        wo = _proj_fp8(wi, blk['wo'], dev)
        h = x2 + wo
        if (i % 4) == 3:
            _mem(f'blk{i}')

    return h
