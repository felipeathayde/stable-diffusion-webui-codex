"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared engine runtime lifecycle helpers.
Centralizes runtime availability checks so engine implementations keep consistent fail-fast errors without repeating
boilerplate across Flux/Z-Image/SDXL/WAN engines.

Symbols (top-level; keep in sync; no ghosts):
- `require_runtime` (function): Return the runtime or raise a consistent “call load() first” error.
"""

from __future__ import annotations

from typing import TypeVar

TRuntime = TypeVar("TRuntime")


def require_runtime(runtime: TRuntime | None, *, label: str) -> TRuntime:
    if runtime is None:
        raise RuntimeError(f"{label} runtime is not initialised; call load() first.")
    return runtime

