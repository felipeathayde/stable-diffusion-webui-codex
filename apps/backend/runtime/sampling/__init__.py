"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Import-light sampling public surface (catalog only).
Re-exports the canonical sampler/scheduler catalog used by the API/UI without importing torch-bound sampling internals at import time.

Symbols (top-level; keep in sync; no ghosts):
- `SAMPLER_OPTIONS` (constant): UI-facing sampler option table (canonical name + optional scheduler allowlists) (re-export).
- `SUPPORTED_SAMPLERS` (constant): Set of supported sampler canonical names (re-export).
- `SCHEDULER_OPTIONS` (constant): UI-facing scheduler option table (canonical name only) (re-export).
- `SUPPORTED_SCHEDULERS` (constant): Set of supported scheduler canonical names (re-export).
- `SAMPLER_DEFAULT_SCHEDULER` (constant): Default scheduler per sampler (re-export).
- `__all__` (constant): Explicit export list for the import-light public sampling surface.
"""

from __future__ import annotations

from .catalog import (
    SAMPLER_DEFAULT_SCHEDULER,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
)

__all__ = [
    "SAMPLER_OPTIONS",
    "SUPPORTED_SAMPLERS",
    "SCHEDULER_OPTIONS",
    "SUPPORTED_SCHEDULERS",
    "SAMPLER_DEFAULT_SCHEDULER",
]
