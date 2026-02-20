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

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend

_LOG_ONCE = {
    "sdpa": False,
    "cross_attn_sliding_fallback": False,
}
_SDPA_LOG_COUNT = 0

_SDPA_SETTINGS = {
    "policy": "mem_efficient",
    "mode": "global",
    "chunk": 0,
}


def set_sdpa_settings(policy: Optional[str], chunk: Optional[int], attention_mode: Optional[str] = None) -> None:
    pol = str(policy if policy is not None else "mem_efficient").strip().lower()
    if pol not in ("mem_efficient", "flash", "math"):
        pol = "mem_efficient"
    mode = str(attention_mode if attention_mode is not None else "global").strip().lower()
    if mode not in ("global", "sliding"):
        raise RuntimeError(f"WAN22 SDPA: unsupported attention mode {attention_mode!r} (expected 'global' or 'sliding').")
    ch = int(chunk) if (chunk is not None and int(chunk) > 0) else 0
    _SDPA_SETTINGS["policy"] = pol
    _SDPA_SETTINGS["mode"] = mode
    _SDPA_SETTINGS["chunk"] = ch


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol = str(_SDPA_SETTINGS["policy"]).strip().lower()
    mode = str(_SDPA_SETTINGS["mode"]).strip().lower()
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
                "sdpa[n=%d]: policy=%s mode=%s effective=%s chunk=%d device=%s dtype=%s qkv=%s",
                _SDPA_LOG_COUNT,
                pol,
                mode,
                eff,
                ch,
                str(q.device),
                str(q.dtype),
                (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
            )
        except Exception:
            pass

    if mode == "sliding":
        if ch <= 0:
            raise RuntimeError("WAN22 SDPA: sliding attention mode requires gguf_attn_chunk > 0.")
        q_length = int(q.shape[2])
        kv_length = int(k.shape[2])
        if kv_length != q_length:
            if not _LOG_ONCE.get("cross_attn_sliding_fallback", False):
                _LOG_ONCE["cross_attn_sliding_fallback"] = True
                try:
                    import logging

                    logging.getLogger("backend.runtime.wan22.sdpa").warning(
                        "sliding mode fallback: q_len=%d differs from kv_len=%d; using full K/V per query chunk",
                        q_length,
                        kv_length,
                    )
                except Exception:
                    pass
            with ctx:
                out_chunks = []
                for start in range(0, q_length, ch):
                    end = min(q_length, start + ch)
                    out_chunks.append(
                        attention_function_pre_shaped(
                            q[:, :, start:end],
                            k,
                            v,
                            is_causal=causal,
                            backend=AttentionBackend.PYTORCH,
                        )
                    )
                return torch.cat(out_chunks, dim=2)
        with ctx:
            _, _, length, _ = q.shape
            out_chunks = []
            for start in range(0, length, ch):
                end = min(length, start + ch)
                window_start = max(0, start - ch)
                window_end = min(length, end + ch)
                out_chunks.append(
                    attention_function_pre_shaped(
                        q[:, :, start:end],
                        k[:, :, window_start:window_end],
                        v[:, :, window_start:window_end],
                        is_causal=causal,
                        backend=AttentionBackend.PYTORCH,
                    )
                )
            return torch.cat(out_chunks, dim=2)

    if mode == "global" and ch > 0:
        with ctx:
            _, _, length, _ = q.shape
            out_chunks = []
            for start in range(0, length, ch):
                end = min(length, start + ch)
                out_chunks.append(
                    attention_function_pre_shaped(
                        q[:, :, start:end],
                        k,
                        v,
                        is_causal=causal,
                        backend=AttentionBackend.PYTORCH,
                    )
                )
            return torch.cat(out_chunks, dim=2)
    with ctx:
        return attention_function_pre_shaped(
            q,
            k,
            v,
            is_causal=causal,
            backend=AttentionBackend.PYTORCH,
        )
