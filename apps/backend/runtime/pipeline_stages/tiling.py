"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tiling toggles for VAE decoding/encoding in generation pipelines.
Applies prompt-derived tiling controls by temporarily mutating the runtime VAE tiling flag and restoring it afterwards.

Symbols (top-level; keep in sync; no ghosts):
- `apply_tiling_if_requested` (function): Enable tiling when requested, returning (applied, previous_value).
- `finalize_tiling` (function): Restore the tiling flag when it was applied.
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.runtime.memory import memory_management


def apply_tiling_if_requested(processing: Any, controls: Mapping[str, Any]) -> tuple[bool, bool]:
    """Enable VAE tiling temporarily when prompts request it."""
    old_value = memory_management.manager.vae_always_tiled
    applied = False
    if controls.get("tiling") is True:
        memory_management.manager.vae_always_tiled = True
        applied = True
    return applied, old_value


def finalize_tiling(applied: bool, previous: bool) -> None:
    """Restore VAE tiling flag."""
    if applied:
        memory_management.manager.vae_always_tiled = previous
