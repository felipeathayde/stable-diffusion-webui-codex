"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Safeweights policy helpers for upscalers (env-controlled).
When `CODEX_SAFE_WEIGHTS=1`, the upscalers runtime and API must reject non-SafeTensors weights (`.pt/.pth`) to prevent
unsafe pickle-based loads.

Symbols (top-level; keep in sync; no ghosts):
- `safeweights_enabled` (function): Returns whether safeweights mode is enabled via `CODEX_SAFE_WEIGHTS`.
- `allowed_upscaler_weight_suffixes` (function): Returns the allowed weight suffixes for upscalers in this process.
"""

from __future__ import annotations

import os


_DEFAULT_ALLOWED_SUFFIXES: tuple[str, ...] = (".safetensors", ".pt", ".pth")


def _env_truthy(key: str, *, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be boolean (got {raw!r}).")


def safeweights_enabled() -> bool:
    return _env_truthy("CODEX_SAFE_WEIGHTS", default=False)


def allowed_upscaler_weight_suffixes() -> tuple[str, ...]:
    return (".safetensors",) if safeweights_enabled() else _DEFAULT_ALLOWED_SUFFIXES


__all__ = ["safeweights_enabled", "allowed_upscaler_weight_suffixes"]

