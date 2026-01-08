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
- `_serialize_checkpoint` (function): Serializes a checkpoint record for `/api/models` responses (hash/path/name metadata).
"""

from __future__ import annotations

from typing import Any, Dict


def _serialize_checkpoint(info) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    short_hash = getattr(info, "short_hash", None) or getattr(info, "shorthash", None)
    return {
        "title": info.title,
        "name": info.name,
        "model_name": info.model_name,
        "hash": short_hash,
        "filename": info.filename,
        "metadata": info.metadata,
    }
