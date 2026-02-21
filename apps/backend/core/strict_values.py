"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Strict value parsing helpers shared across backend orchestration layers.
Provides fail-loud conversion utilities for loose object payloads that may come from API requests, persisted options, or runtime metadata.

Symbols (top-level; keep in sync; no ghosts):
- `_TRUE_BOOL_TOKENS` (constant): Accepted truthy string tokens for strict bool parsing.
- `_FALSE_BOOL_TOKENS` (constant): Accepted falsy string tokens for strict bool parsing.
- `parse_bool_value` (function): Parses a strict boolean from bool/int/float/string inputs and fails loud on invalid values.
- `parse_int_value` (function): Parses an integer from int/float/string inputs with optional bounds and fails loud on invalid values.
- `__all__` (constant): Explicit export list for strict value parsers.
"""

from __future__ import annotations

_TRUE_BOOL_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_BOOL_TOKENS = frozenset({"0", "false", "no", "off"})


def parse_bool_value(value: object, *, field: str, default: bool | None = None) -> bool:
    if value is None:
        if default is not None:
            return bool(default)
        raise RuntimeError(f"Invalid '{field}': expected boolean, got null.")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_TOKENS:
            return True
        if normalized in _FALSE_BOOL_TOKENS:
            return False
    raise RuntimeError(
        f"Invalid '{field}': expected bool or one of "
        f"('true','false','1','0','yes','no','on','off'), got {value!r}."
    )


def parse_int_value(
    value: object,
    *,
    field: str,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    parsed: int
    if value is None:
        if default is None:
            raise RuntimeError(f"Invalid '{field}': expected integer, got null.")
        parsed = int(default)
    elif isinstance(value, bool):
        raise RuntimeError(f"Invalid '{field}': expected integer, got boolean.")
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise RuntimeError(f"Invalid '{field}': expected integer value, got {value!r}.")
        parsed = int(value)
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            raise RuntimeError(f"Invalid '{field}': expected integer literal, got empty string.")
        try:
            parsed = int(token, 10)
        except ValueError as exc:
            raise RuntimeError(f"Invalid '{field}': expected integer literal, got {value!r}.") from exc
    else:
        raise RuntimeError(f"Invalid '{field}': expected integer, got {type(value).__name__}.")

    if minimum is not None and parsed < minimum:
        raise RuntimeError(f"Invalid '{field}': expected value >= {minimum}, got {parsed}.")
    if maximum is not None and parsed > maximum:
        raise RuntimeError(f"Invalid '{field}': expected value <= {maximum}, got {parsed}.")
    return parsed


__all__ = ["parse_bool_value", "parse_int_value"]
