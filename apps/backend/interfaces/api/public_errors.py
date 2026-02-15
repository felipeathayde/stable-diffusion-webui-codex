"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public-safe error message normalization for API payloads/SSE.
Prevents leaking raw exception text (for example prompt fragments) to client-visible task status/event channels,
while preserving actionable terminal classes like cancellation and out-of-memory.

Symbols (top-level; keep in sync; no ghosts):
- `public_task_error_message` (function): Convert a raw exception/string into a public-safe task error message.
- `public_http_error_detail` (function): Convert a raw exception/string into a public-safe HTTP detail string.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_INTERNAL_ERROR_ID_RE = re.compile(r"^internal error \(error_id=[0-9a-f]{12}\)$")
_OOM_WORD_RE = re.compile(r"\boom\b", flags=re.IGNORECASE)
_OOM_HINTS = (
    "out of memory",
    "cuda oom",
    "not enough memory",
    "alloc_failed",
    "allocation failed",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "oom while",
    "oom during",
    "oom on ",
    "core construction oom",
)
_INTEGRITY_HINTS = ("sha256 mismatch",)


def _normalize_error_text(err: Any) -> str:
    text = str(err or "").strip()
    if not text:
        return "internal error"
    return text


def _is_oom_error(*, err: Any, lowered_message: str) -> bool:
    if isinstance(err, MemoryError):
        return True
    if any(marker in lowered_message for marker in _OOM_HINTS):
        return True
    if _OOM_WORD_RE.search(lowered_message):
        return True
    return False


def public_task_error_message(err: Any) -> str:
    raw = _normalize_error_text(err)
    lowered = raw.lower()

    if lowered == "cancelled":
        return "cancelled"
    if _INTERNAL_ERROR_ID_RE.fullmatch(raw):
        return raw
    if any(marker in lowered for marker in _INTEGRITY_HINTS):
        return "sha256 mismatch"
    if _is_oom_error(err=err, lowered_message=lowered):
        return "out of memory"

    error_id = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"internal error (error_id={error_id})"


def public_http_error_detail(err: Any, *, fallback: str) -> str:
    public = public_task_error_message(err)
    if public == "out of memory":
        return public
    return str(fallback or "invalid request")


__all__ = ["public_http_error_detail", "public_task_error_message"]
