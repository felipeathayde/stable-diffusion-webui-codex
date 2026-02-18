"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-side context managers for stage-scoped model residency.
Centralizes the “load if needed → run → unload only if we loaded it” pattern used by conditioning and other small stages
when smart-offload is enabled, preventing VRAM churn between adjacent calls (e.g. cond/uncond).
Delegates smart-offload load/unload event emission to `runtime.memory.manager` (single source for generic actions).

Symbols (top-level; keep in sync; no ghosts):
- `stage_scoped_model_load` (context manager): Load a model for the duration of a stage and unload only if this context loaded it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol

class _ModelManager(Protocol):
    def is_model_loaded(self, model: object) -> bool: ...
    def load_model(
        self,
        model: object,
        *,
        source: str = "runtime.memory.manager.load_model",
        stage: str | None = None,
        component_hint: str | None = None,
        event_reason: str | None = None,
    ) -> None: ...
    def unload_model(
        self,
        model: object,
        *,
        source: str = "runtime.memory.manager.unload_model",
        stage: str | None = None,
        component_hint: str | None = None,
        event_reason: str | None = None,
    ) -> None: ...


@contextmanager
def stage_scoped_model_load(
    model: object,
    *,
    smart_offload_enabled: bool,
    manager: _ModelManager,
) -> Iterator[None]:
    source = "engines.common.model_scopes.stage_scoped_model_load"
    already_loaded = manager.is_model_loaded(model)
    manager.load_model(
        model,
        source=source,
        stage="scope_enter",
    )
    unload = bool(smart_offload_enabled) and not already_loaded
    try:
        yield
    finally:
        if unload:
            manager.unload_model(
                model,
                source=source,
                stage="scope_exit",
            )


__all__ = ["stage_scoped_model_load"]
