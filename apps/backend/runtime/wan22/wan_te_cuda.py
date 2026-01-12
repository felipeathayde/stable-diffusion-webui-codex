"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN T5 CUDA FP8 extension bridge (optional runtime path).
Loads the `wan_te_cuda` extension and exposes minimal FP8 ops used by WAN text-encoder code paths.

Symbols (top-level; keep in sync; no ghosts):
- `available` (function): Returns True when the extension is available (may attempt a JIT build).
- `last_error` (function): Returns the last extension load/build error message, when present.
- `Fp8Weight` (class): Container describing an FP8 packed weight tensor and scale.
- `linear_fp8` (function): Calls the extension linear kernel for FP8 weights.
- `attn_fp8` (function): Calls the extension attention kernel for FP8 q/k/v (returns output + optional probs).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import torch

log = logging.getLogger("backend.runtime.wan22.te.cuda")

_ext = None
_last_error: Optional[str] = None


def _try_load_ext(build: bool = False) -> None:
    global _ext
    global _last_error
    if _ext is not None:
        return
    try:
        import wan_te_cuda as _loaded
        _ext = _loaded
        log.info("loaded wan_te_cuda extension (prebuilt)")
        return
    except Exception as ex:
        log.info("wan_te_cuda prebuilt not found: %s", ex)
    # Try in-place build location (apps/backend/runtime/kernels/wan_t5)
    try:
        this_dir = os.path.dirname(__file__)
        ext_dir = os.path.join(os.path.dirname(this_dir), 'kernels', 'wan_t5')
        if os.path.isdir(ext_dir) and ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import wan_te_cuda as _loaded
        _ext = _loaded
        log.info("loaded wan_te_cuda extension from in-place build (%s)", ext_dir)
        return
    except Exception as ex:
        log.info("wan_te_cuda not found in in-place dir: %s", ex)
    if not build:
        return
    # JIT build (requires nvcc toolchain present)
    try:
        from torch.utils.cpp_extension import load
        this_dir = os.path.dirname(__file__)
        src_dir = os.path.join(os.path.dirname(this_dir), 'kernels', 'wan_t5')
        def _src(p: str) -> str:
            return os.path.join(src_dir, p)
        sources = [
            _src('te_binding.cpp'),
            _src('te_attention_fp8.cu'),
            _src('te_linear_fp8.cu'),
        ]
        kern = _src('te_attention_fp8_kernel.cu')
        if os.path.isfile(kern):
            sources.append(kern)
        _mod = load(name='wan_te_cuda', sources=sources,
                    extra_cflags=['-O3'],
                    extra_cuda_cflags=['-O3', '--use_fast_math'])
        _ext = _mod
        log.info("built wan_te_cuda extension via JIT")
    except Exception as ex:
        _last_error = f"failed to build wan_te_cuda via JIT: {ex}"
        log.error(_last_error)
        _ext = None


def available() -> bool:
    # Always attempt JIT build if not found — exceptional fallback only
    _try_load_ext(build=True)
    return _ext is not None


def last_error() -> Optional[str]:
    return _last_error


class Fp8Weight:
    def __init__(self, w_u8: torch.Tensor, scale: torch.Tensor, fp8_format: str = 'e4m3fn') -> None:
        self.w_u8 = w_u8
        self.scale = scale
        self.format = 0 if fp8_format == 'e4m3fn' else 1


def linear_fp8(x: torch.Tensor, w: Fp8Weight, b: Optional[torch.Tensor]) -> torch.Tensor:
    if _ext is None:
        raise RuntimeError("wan_te_cuda extension not available")
    return torch.ops.wan_te_cuda.linear_fp8_forward(x, w.w_u8, w.scale, b, int(w.format))


def attn_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], causal: bool, fp8_format: str = 'e4m3fn') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _ext is None:
        raise RuntimeError("wan_te_cuda extension not available")
    fmt = 0 if fp8_format == 'e4m3fn' else 1
    out, probs = torch.ops.wan_te_cuda.attn_fp8_forward(q, k, v, attn_mask if attn_mask is not None else None, bool(causal), int(fmt))
    return out, probs if probs.numel() > 0 else None
