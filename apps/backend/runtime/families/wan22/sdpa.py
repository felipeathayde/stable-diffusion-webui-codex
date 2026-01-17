"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDPA backend selection helpers for WAN runtimes.
Provides a configurable `sdpa(...)` wrapper with optional chunking and an env/config-driven backend policy.

Symbols (top-level; keep in sync; no ghosts):
- `_SDPA_SETTINGS` (constant): Mutable config dict storing current SDPA policy and chunk size.
- `set_sdpa_settings` (function): Applies policy/chunk settings (explicit args override env overrides).
- `sdpa` (function): Calls PyTorch SDPA using the configured backend policy and optional chunking.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

_LOG_ONCE = {
    "sdpa": False,
}
_SDPA_LOG_COUNT = 0

_SDPA_SETTINGS = {
    "policy": "mem_efficient",
    "chunk": 0,
}


def set_sdpa_settings(policy: Optional[str], chunk: Optional[int]) -> None:
    pol = (policy or _SDPA_SETTINGS["policy"]).strip().lower()
    if pol not in ("mem_efficient", "flash", "math"):
        pol = _SDPA_SETTINGS["policy"]
    ch = int(chunk) if (chunk is not None and int(chunk) > 0) else 0
    _SDPA_SETTINGS["policy"] = pol
    _SDPA_SETTINGS["chunk"] = ch


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol = str(_SDPA_SETTINGS["policy"]).strip().lower()
    ch = int(_SDPA_SETTINGS["chunk"])

    ctx = nullcontext()
    eff = "unknown"
    try:
        if q.is_cuda:
            from torch.nn.attention import SDPBackend  # type: ignore[attr-defined]
            from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # type: ignore[attr-defined]

            backend = {
                "flash": SDPBackend.FLASH_ATTENTION,
                "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH,
                "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
            }.get(pol, SDPBackend.EFFICIENT_ATTENTION)
            ctx = _sdpa_kernel(backend)
            eff = {
                SDPBackend.FLASH_ATTENTION: "flash",
                SDPBackend.EFFICIENT_ATTENTION: "mem_efficient",
                SDPBackend.MATH: "math",
                getattr(SDPBackend, "CUDNN_ATTENTION", SDPBackend.EFFICIENT_ATTENTION): "cudnn",
            }.get(backend, pol)
    except Exception:
        try:
            if q.is_cuda and hasattr(torch.backends, "cuda"):
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=(pol == "flash"),
                    enable_math=(pol == "math"),
                    enable_mem_efficient=(pol == "mem_efficient"),
                )
                try:
                    _b = torch.backends.cuda
                    if _b.is_flash_sdp_enabled():
                        eff = "flash"
                    elif _b.is_mem_efficient_sdp_enabled():
                        eff = "mem_efficient"
                    elif _b.is_math_sdp_enabled():
                        eff = "math"
                except Exception:
                    eff = pol
        except Exception:
            ctx = nullcontext()

    global _LOG_ONCE, _SDPA_LOG_COUNT
    _SDPA_LOG_COUNT += 1
    should_log = not _LOG_ONCE.get("sdpa", False)
    _LOG_ONCE["sdpa"] = True
    if should_log:
        try:
            import logging

            logging.getLogger("backend.runtime.wan22.sdpa").info(
                "sdpa[n=%d]: policy=%s effective=%s chunk=%d device=%s dtype=%s qkv=%s",
                _SDPA_LOG_COUNT,
                pol,
                eff,
                ch,
                str(q.device),
                str(q.dtype),
                (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
            )
        except Exception:
            pass

    if ch and ch > 0:
        with ctx:
            B, H, L, D = q.shape
            out_chunks = []
            for s in range(0, L, ch):
                e = min(L, s + ch)
                out_chunks.append(
                    torch.nn.functional.scaled_dot_product_attention(q[:, :, s:e], k, v, is_causal=causal)
                )
            return torch.cat(out_chunks, dim=2)
    with ctx:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
