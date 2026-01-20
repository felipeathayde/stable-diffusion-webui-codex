"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lazy export name groups for backend package facades.
Defines frozen export sets used by backend `__getattr__`/`__all__` wiring to expose subsystems on demand without heavy imports during startup.
Patchers/services are intentionally excluded (empty sets) so dependencies remain explicit.

Symbols (top-level; keep in sync; no ghosts):
- `LazyExports` (dataclass): Frozen sets of export names grouped by subsystem (engines/runtime/text-processing); patchers/services are intentionally empty.
- `LAZY_EXPORTS` (constant): Singleton instance of `LazyExports`.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class LazyExports:
    """Groups of exports loaded lazily to avoid heavy import costs."""
    
    ENGINES: FrozenSet[str] = frozenset({
        "register_default_engines",
        "Wan2214BEngine",
        "Wan225BEngine",
    })
    
    TEXT_PROCESSING: FrozenSet[str] = frozenset({
        "ClassicTextProcessingEngine",
        "EmbeddingDatabase",
        "T5TextProcessingEngine",
        "embedding_from_b64",
        "embedding_to_b64",
        "text_emphasis",
        "text_parsing",
        "textual_inversion",
    })
    
    RUNTIME: FrozenSet[str] = frozenset({
        "attention",
        "logging",
        "memory_management",
        "models",
        "nn",
        "ops",
        "stream",
        "text_processing",
        "utils",
    })
    
    # Patcher symbols are intentionally not re-exported from `apps.backend` anymore.
    # Import patchers from `apps.backend.patchers.*` to keep dependencies explicit.
    PATCHERS: FrozenSet[str] = frozenset()
    
    # Service classes are intentionally not re-exported from `apps.backend` anymore.
    # Import services from `apps.backend.services.*` to keep dependencies explicit.
    SERVICES: FrozenSet[str] = frozenset()


# Singleton instance
LAZY_EXPORTS = LazyExports()

__all__ = ["LazyExports", "LAZY_EXPORTS"]
