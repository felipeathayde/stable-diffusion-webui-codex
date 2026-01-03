"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Import-light sampling facade for Codex engines.
Exposes torch-bound inner-loop helpers via a lazy `__getattr__` hook so API/UI tooling can import sampling catalogs without pulling heavy runtime modules.

Symbols (top-level; keep in sync; no ghosts):
- `__getattr__` (function): Lazy import hook exposing `sampling_*` helpers from `inner_loop.py` on first access.
- `__all__` (constant): Export list for sampling helpers exposed via the facade.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - import-time dispatch
    if name in {
        "sampling_function",
        "sampling_function_inner",
        "sampling_prepare",
        "sampling_cleanup",
    }:
        from . import inner_loop as _inner_loop

        value = getattr(_inner_loop, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "sampling_cleanup",
    "sampling_function",
    "sampling_function_inner",
    "sampling_prepare",
]
