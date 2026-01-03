"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public API for parsing checkpoint state dicts into Codex components.
Exposes `parse_state_dict(...)` which resolves the correct family plan, executes it over a (possibly lazy) state dict, and returns a
`CodexEstimatedConfig` describing the parsed components and defaults.

Symbols (top-level; keep in sync; no ghosts):
- `parse_state_dict` (function): Parses a state dict using a `ModelSignature` and returns a `CodexEstimatedConfig`.
- `CodexEstimatedConfig` (class): Parse output structure (re-export from `.specs`).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from apps.backend.runtime.model_registry.specs import ModelSignature

from .specs import CodexEstimatedConfig


def parse_state_dict(state_dict: MutableMapping[str, Any], signature: ModelSignature) -> CodexEstimatedConfig:
    # Lazy import to avoid circular dependency during unit tests and lightweight callers.
    from .families import resolve_plan
    from .plan import execute_plan

    bundle = resolve_plan(signature)
    context = execute_plan(bundle.plan, state_dict, signature=signature)
    return bundle.build_config(context)

__all__ = ["parse_state_dict", "CodexEstimatedConfig"]
