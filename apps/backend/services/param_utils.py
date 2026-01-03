"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request payload parsing helpers for the API layer.
Provides small converters used to validate required fields and coerce basic types from dict payloads.

Symbols (top-level; keep in sync; no ghosts):
- `require` (function): Returns a required key from a payload or raises a `ValueError`.
- `as_list` (function): Returns a value as a list (wrapping scalars, defaulting to empty list).
- `as_int` (function): Coerces a payload field to `int`, optionally providing a default.
- `as_float` (function): Coerces a payload field to `float`, optionally providing a default.
- `as_float_optional` (function): Best-effort float coercion returning `None`/default on failure.
- `as_bool` (function): Coerces a payload field to `bool`, supporting common string/int forms.
"""

from __future__ import annotations

from typing import Any, List, Optional


def require(payload: dict, key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def as_list(payload: dict, key: str) -> List[Any]:
    v = payload.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def as_int(payload: dict, key: str, default: Optional[int] = None) -> int:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing int field: {key}")
        return int(default)
    return int(v)


def as_float(payload: dict, key: str, default: Optional[float] = None) -> float:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing float field: {key}")
        return float(default)
    return float(v)


def as_float_optional(payload: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = payload.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def as_bool(payload: dict, key: str, default: Optional[bool] = None) -> bool:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing bool field: {key}")
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")
