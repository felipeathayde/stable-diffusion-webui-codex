"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDPA backend selection helpers for WAN runtimes.
Provides a configurable `sdpa(...)` wrapper with optional chunking and strict policy validation, delegating per-call SDPA execution to the central attention dispatcher.

Symbols (top-level; keep in sync; no ghosts):
- `_SDPA_SETTINGS` (constant): Mutable config dict storing current SDPA policy and chunk size.
- `set_sdpa_settings` (function): Applies policy/chunk settings (explicit args override env overrides).
- `sdpa` (function): Calls PyTorch SDPA using the configured backend policy and optional chunking.
"""

from __future__ import annotations

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
    "policy": "auto",
    "mode": "global",
    "chunk": 0,
}


def set_sdpa_settings(policy: Optional[str], chunk: Optional[int], attention_mode: Optional[str] = None) -> None:
    if policy is not None and not isinstance(policy, str):
        raise TypeError(f"WAN22 SDPA: policy must be a string when provided, got {type(policy).__name__}.")
    pol = str(policy if policy is not None else "auto").strip().lower()
    if pol not in ("auto", "mem_efficient", "flash", "math"):
        raise RuntimeError(
            "WAN22 SDPA: unsupported policy "
            f"{policy!r} (expected one of: 'auto', 'mem_efficient', 'flash', 'math')."
        )
    mode = str(attention_mode if attention_mode is not None else "global").strip().lower()
    if mode not in ("global", "sliding"):
        raise RuntimeError(f"WAN22 SDPA: unsupported attention mode {attention_mode!r} (expected 'global' or 'sliding').")
    if chunk is None:
        ch = 0
    else:
        try:
            chunk_value = int(chunk)
        except Exception as exc:
            raise RuntimeError(f"WAN22 SDPA: chunk must be an integer when provided, got {chunk!r}.") from exc
        ch = chunk_value if chunk_value > 0 else 0
    _SDPA_SETTINGS["policy"] = pol
    _SDPA_SETTINGS["mode"] = mode
    _SDPA_SETTINGS["chunk"] = ch


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol = str(_SDPA_SETTINGS["policy"]).strip().lower()
    mode = str(_SDPA_SETTINGS["mode"]).strip().lower()
    ch = int(_SDPA_SETTINGS["chunk"])

    global _LOG_ONCE, _SDPA_LOG_COUNT
    _SDPA_LOG_COUNT += 1
    should_log = not _LOG_ONCE.get("sdpa", False)
    _LOG_ONCE["sdpa"] = True
    if should_log:
        try:
            import logging

            logging.getLogger("backend.runtime.wan22.sdpa").info(
                "sdpa[n=%d]: policy=%s mode=%s chunk=%d device=%s dtype=%s qkv=%s",
                _SDPA_LOG_COUNT,
                pol,
                mode,
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
                        sdpa_policy=pol,
                    )
                )
            return torch.cat(out_chunks, dim=2)
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
                    sdpa_policy=pol,
                )
            )
        return torch.cat(out_chunks, dim=2)

    if mode == "global" and ch > 0:
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
                    sdpa_policy=pol,
                )
            )
        return torch.cat(out_chunks, dim=2)
    return attention_function_pre_shaped(
        q,
        k,
        v,
        is_causal=causal,
        backend=AttentionBackend.PYTORCH,
        sdpa_policy=pol,
    )
