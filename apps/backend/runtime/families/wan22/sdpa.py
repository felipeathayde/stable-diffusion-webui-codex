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
import logging
import os
from typing import Optional
from uuid import uuid4

import torch

from apps.backend.runtime.attention import attention_function_pre_shaped, set_attention_request_id
from apps.backend.runtime.memory.config import AttentionBackend

_LOGGER = logging.getLogger("backend.runtime.wan22.sdpa")
_SDPA_CALL_COUNT_CTX: ContextVar[int] = ContextVar("wan22_sdpa_call_count", default=0)
_CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX: ContextVar[bool] = ContextVar(
    "wan22_cross_attn_sliding_fallback_logged",
    default=False,
)

_FUSED_MODE_ENV = "CODEX_WAN22_FUSED_ATTN_V1_MODE"

_SDPA_SETTINGS_CTX: ContextVar[tuple[str, str, int, str, str]] = ContextVar(
    "wan22_sdpa_settings",
    default=("auto", "global", 0, "off", "wan22-unknown"),
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
    request_id: Optional[str] = None,
) -> None:
    pol, mode, ch, fused = _normalize_sdpa_settings(policy, chunk, attention_mode, fused_mode)
    rid = str(request_id or "").strip() or f"wan22-{uuid4().hex[:12]}"
    _SDPA_SETTINGS_CTX.set((pol, mode, ch, fused, rid))
    _SDPA_CALL_COUNT_CTX.set(0)
    _CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.set(False)
    set_attention_request_id(rid)
    _LOGGER.info(
        "[wan22.sdpa][req=%s] configured policy=%s mode=%s chunk=%d backend=%s fused_mode=%s",
        rid,
        pol,
        mode,
        ch,
        AttentionBackend.PYTORCH.value,
        fused,
    )


def _get_sdpa_settings() -> tuple[str, str, int, str, str]:
    return _SDPA_SETTINGS_CTX.get()


def get_wan_fused_mode() -> str:
    _pol, _mode, _chunk, fused_mode, _request_id = _get_sdpa_settings()
    return fused_mode


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol, mode, ch, fused_mode, request_id = _get_sdpa_settings()
    set_attention_request_id(request_id)
    call_count = int(_SDPA_CALL_COUNT_CTX.get()) + 1
    _SDPA_CALL_COUNT_CTX.set(call_count)
    if call_count == 1:
        _LOGGER.info(
            "[wan22.sdpa][req=%s] first_call policy=%s mode=%s chunk=%d device=%s dtype=%s qkv=%s fused_mode=%s env=%s",
            request_id,
            pol,
            mode,
            ch,
            str(q.device),
            str(q.dtype),
            (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
            fused_mode,
            _FUSED_MODE_ENV,
        )

    if mode == "sliding":
        if ch <= 0:
            raise RuntimeError("WAN22 SDPA: sliding attention mode requires gguf_attn_chunk > 0.")
        q_length = int(q.shape[2])
        kv_length = int(k.shape[2])
        if kv_length != q_length:
            if not bool(_CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.get()):
                _CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.set(True)
                _LOGGER.warning(
                    "[wan22.sdpa][req=%s] sliding mode fallback: q_len=%d differs from kv_len=%d; using full K/V per query chunk",
                    request_id,
                    q_length,
                    kv_length,
                )
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
