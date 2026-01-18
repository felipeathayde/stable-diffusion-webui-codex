"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Serialization helpers for API responses.
Keeps response shapes stable for checkpoint listings.

Symbols (top-level; keep in sync; no ghosts):
- `_serialize_checkpoint` (function): Serializes a checkpoint record for `/api/models` responses (hash/path/name metadata + core-only hints).
"""

from __future__ import annotations

from typing import Any, Dict


def _serialize_checkpoint(info) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    short_hash = getattr(info, "short_hash", None) or getattr(info, "shorthash", None)
    payload: Dict[str, Any] = {
        "title": info.title,
        "name": info.name,
        "model_name": info.model_name,
        "hash": short_hash,
        "filename": info.filename,
        "metadata": info.metadata,
        "core_only": bool(getattr(info, "core_only", False)),
    }
    core_only_reason = getattr(info, "core_only_reason", None)
    if isinstance(core_only_reason, str) and core_only_reason.strip():
        payload["core_only_reason"] = core_only_reason.strip()
    family_hint = getattr(info, "family_hint", None)
    if isinstance(family_hint, str) and family_hint.strip():
        payload["family_hint"] = family_hint.strip()
    return payload
