"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDPA backend selection helpers for WAN runtimes.
Provides a configurable `sdpa(...)` wrapper with optional chunking and strict policy validation,
delegating per-call SDPA execution to the central attention dispatcher and carrying fused-attention mode.

Symbols (top-level; keep in sync; no ghosts):
- `_SDPA_SETTINGS_CTX` (constant): Context-local SDPA settings tuple (`policy`, `mode`, `chunk`, `fused_mode`).
- `_normalize_fused_mode` (function): Validates and normalizes WAN fused-attention mode input.
- `_normalize_sdpa_settings` (function): Validates and normalizes SDPA policy/chunk/mode/fused_mode inputs.
- `set_sdpa_settings` (function): Applies policy/chunk/mode/fused settings (explicit args override env when provided).
- `_get_sdpa_settings` (function): Reads effective context-local SDPA settings tuple.
- `get_wan_fused_mode` (function): Returns the effective WAN fused-attention mode (`off|auto|force`) for the active context.
- `sdpa` (function): Calls PyTorch SDPA using the configured backend policy and optional chunking.
"""

from __future__ import annotations

from contextvars import ContextVar
import os
from typing import Optional

import torch

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend

_LOG_ONCE = {
    "sdpa": False,
    "cross_attn_sliding_fallback": False,
}
_SDPA_LOG_COUNT = 0

_FUSED_MODE_ENV = "CODEX_WAN22_FUSED_ATTN_V1_MODE"

_SDPA_SETTINGS_CTX: ContextVar[tuple[str, str, int, str]] = ContextVar(
    "wan22_sdpa_settings",
    default=("auto", "global", 0, "off"),
)


def _normalize_fused_mode(fused_mode: Optional[str]) -> str:
    raw = fused_mode
    if raw is None:
        raw = os.environ.get(_FUSED_MODE_ENV, "off")
    normalized = str(raw).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "auto",
        "true": "auto",
        "yes": "auto",
        "on": "auto",
        "auto": "auto",
        "force": "force",
        "required": "force",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        raise RuntimeError(
            "WAN22 SDPA: unsupported fused attention mode "
            f"{raw!r} (expected one of: 'off', 'auto', 'force')."
        )
    return mapped


def _normalize_sdpa_settings(
    policy: Optional[str],
    chunk: Optional[int],
    attention_mode: Optional[str],
    fused_mode: Optional[str],
) -> tuple[str, str, int, str]:
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
    fused = _normalize_fused_mode(fused_mode)
    return pol, mode, ch, fused


def set_sdpa_settings(
    policy: Optional[str],
    chunk: Optional[int],
    attention_mode: Optional[str] = None,
    fused_mode: Optional[str] = None,
) -> None:
    _SDPA_SETTINGS_CTX.set(_normalize_sdpa_settings(policy, chunk, attention_mode, fused_mode))


def _get_sdpa_settings() -> tuple[str, str, int, str]:
    return _SDPA_SETTINGS_CTX.get()


def get_wan_fused_mode() -> str:
    _pol, _mode, _chunk, fused_mode = _get_sdpa_settings()
    return fused_mode


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol, mode, ch, fused_mode = _get_sdpa_settings()

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
            logging.getLogger("backend.runtime.wan22.sdpa").info(
                "wan_fused_mode=%s env=%s",
                fused_mode,
                _FUSED_MODE_ENV,
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
            out_accum: torch.Tensor | None = None
            for start in range(0, q_length, ch):
                end = min(q_length, start + ch)
                chunk_out = attention_function_pre_shaped(
                    q[:, :, start:end],
                    k,
                    v,
                    is_causal=causal,
                    backend=AttentionBackend.PYTORCH,
                    sdpa_policy=pol,
                )
                if out_accum is None:
                    out_accum = torch.empty(
                        (
                            int(chunk_out.shape[0]),
                            int(chunk_out.shape[1]),
                            q_length,
                            int(chunk_out.shape[3]),
                        ),
                        device=chunk_out.device,
                        dtype=chunk_out.dtype,
                    )
                out_accum[:, :, start:end, :] = chunk_out
            if out_accum is None:
                raise RuntimeError("WAN22 SDPA: sliding cross-attention fallback produced no output chunks.")
            return out_accum
        _, _, length, _ = q.shape
        out_accum: torch.Tensor | None = None
        for start in range(0, length, ch):
            end = min(length, start + ch)
            window_start = max(0, start - ch)
            window_end = min(length, end + ch)
            chunk_out = attention_function_pre_shaped(
                q[:, :, start:end],
                k[:, :, window_start:window_end],
                v[:, :, window_start:window_end],
                is_causal=causal,
                backend=AttentionBackend.PYTORCH,
                sdpa_policy=pol,
            )
            if out_accum is None:
                out_accum = torch.empty(
                    (
                        int(chunk_out.shape[0]),
                        int(chunk_out.shape[1]),
                        length,
                        int(chunk_out.shape[3]),
                    ),
                    device=chunk_out.device,
                    dtype=chunk_out.dtype,
                )
            out_accum[:, :, start:end, :] = chunk_out
        if out_accum is None:
            raise RuntimeError("WAN22 SDPA: sliding self-attention produced no output chunks.")
        return out_accum

    if mode == "global" and ch > 0:
        _, _, length, _ = q.shape
        out_accum: torch.Tensor | None = None
        for start in range(0, length, ch):
            end = min(length, start + ch)
            chunk_out = attention_function_pre_shaped(
                q[:, :, start:end],
                k,
                v,
                is_causal=causal,
                backend=AttentionBackend.PYTORCH,
                sdpa_policy=pol,
            )
            if out_accum is None:
                out_accum = torch.empty(
                    (
                        int(chunk_out.shape[0]),
                        int(chunk_out.shape[1]),
                        length,
                        int(chunk_out.shape[3]),
                    ),
                    device=chunk_out.device,
                    dtype=chunk_out.dtype,
                )
            out_accum[:, :, start:end, :] = chunk_out
        if out_accum is None:
            raise RuntimeError("WAN22 SDPA: global chunked attention produced no output chunks.")
        return out_accum
    return attention_function_pre_shaped(
        q,
        k,
        v,
        is_causal=causal,
        backend=AttentionBackend.PYTORCH,
        sdpa_policy=pol,
    )
