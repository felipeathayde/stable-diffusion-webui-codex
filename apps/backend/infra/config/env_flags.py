"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared env-flag parsing helpers for backend/runtime modules.
Centralizes flag parsing semantics (truthy/falsy sets, defaults, and numeric clamping) so subsystems don't drift on debug/feature toggles.
Reads from `bootstrap_env` overrides first (set at backend startup) to avoid runtime `os.environ` mutation.

Symbols (top-level; keep in sync; no ghosts):
- `_TRUE` (constant): Truthy token set for env flags.
- `_FALSE` (constant): Falsy token set for env flags.
- `env_flag` (function): Reads a boolean env var with consistent truthiness and default fallback.
- `env_int` (function): Reads an integer env var with default fallback and optional clamping.
- `env_str` (function): Reads a string env var with default fallback and optional allowed-set validation.
- `__all__` (constant): Explicit export list for env flag helpers.
"""

from __future__ import annotations

import os
from typing import Optional

from .bootstrap_env import get_bootstrap_env

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean env flag.

    Semantics:
    - missing → default
    - truthy tokens → True
    - falsy tokens → False
    - unknown/empty → default
    """

    raw = get_bootstrap_env(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return bool(default)


def env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """Return an int env var with optional clamping.

    - missing/invalid/empty → default
    - min_value/max_value apply after parsing.
    """

    raw = get_bootstrap_env(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        value = int(default)
    else:
        s = str(raw).strip()
        if not s:
            value = int(default)
        else:
            try:
                value = int(s)
            except Exception:
                value = int(default)

    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def env_str(name: str, default: str = "", *, allowed: Optional[set[str]] = None) -> str:
    """Return a normalized string env var.

    Semantics:
    - missing/empty → default
    - if allowed is provided: unknown → default
    - normalization: strip + lowercase
    """

    raw = get_bootstrap_env(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        value = str(default)
    else:
        value = str(raw).strip()
        if not value:
            value = str(default)

    normalized = value.strip().lower()
    if allowed is None:
        return normalized

    allowed_normalized = {str(v).strip().lower() for v in allowed}
    if normalized in allowed_normalized:
        return normalized
    return str(default).strip().lower()


__all__ = ["env_flag", "env_int", "env_str"]
