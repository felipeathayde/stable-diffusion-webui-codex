"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Process-local “env override” store for runtime bootstrap (no os.environ mutation).
Some subsystems historically read configuration via `os.getenv(...)`. Instead of mutating `os.environ` at runtime, the API bootstrap can
publish resolved values (after CLI/env/settings precedence) into this module, and `env_flags` / config readers can consult it first.

Symbols (top-level; keep in sync; no ghosts):
- `set_bootstrap_env` (function): Set an override value for an env key (stringified).
- `get_bootstrap_env` (function): Get an override value for an env key, or None.
- `clear_bootstrap_env` (function): Clear all overrides (test utility).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from typing import Dict, Optional

_BOOTSTRAP_ENV: Dict[str, str] = {}


def set_bootstrap_env(key: str, value: object) -> None:
    """Set a bootstrap env override value for this process."""
    _BOOTSTRAP_ENV[str(key)] = str(value)


def get_bootstrap_env(key: str) -> Optional[str]:
    """Return the bootstrap env override value for `key` if set."""
    return _BOOTSTRAP_ENV.get(str(key))


def clear_bootstrap_env() -> None:
    """Clear all bootstrap env overrides (intended for tests)."""
    _BOOTSTRAP_ENV.clear()


__all__ = ["clear_bootstrap_env", "get_bootstrap_env", "set_bootstrap_env"]

