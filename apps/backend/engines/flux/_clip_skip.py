"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux-family clip-skip helper.
Centralizes clip-skip parsing/validation, runtime application (0=reset sentinel), cache invalidation, and consistent logging
for Flux-family engines.

Symbols (top-level; keep in sync; no ghosts):
- `apply_flux_clip_skip` (function): Validate and apply `clip_skip` to a `FluxEngineRuntime` and clear the engine conditioning cache.
"""

from __future__ import annotations

import logging
from typing import Any

from apps.backend.engines.flux.spec import FluxEngineRuntime


def apply_flux_clip_skip(
    *,
    engine: Any,
    runtime: FluxEngineRuntime,
    clip_skip: int,
    logger: logging.Logger,
    label: str,
) -> None:
    try:
        requested = int(clip_skip)
    except Exception as exc:  # noqa: BLE001
        raise TypeError("clip_skip must be an integer") from exc
    if requested < 0:
        raise ValueError("clip_skip must be >= 0")

    runtime.set_clip_skip(requested)

    # Cached conditioning depends on clip_skip (pooled CLIP output changes).
    try:
        engine._cond_cache.clear()  # noqa: SLF001 (engine-owned cache; explicit invalidation)
    except Exception:
        pass

    if requested == 0:
        logger.debug("Clip skip reset to default for %s.", label)
    else:
        logger.debug("Clip skip set to %d for %s.", requested, label)


__all__ = ["apply_flux_clip_skip"]

