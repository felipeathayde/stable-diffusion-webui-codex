"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared SD-family clip-skip helpers.
Centralizes clip-skip validation, runtime application (0=reset), cache invalidation, and consistent logging for SD engines.

Symbols (top-level; keep in sync; no ghosts):
- `apply_sd_clip_skip` (function): Validate and apply `clip_skip` to an `SDEngineRuntime` and clear the engine conditioning cache.
"""

from __future__ import annotations

import logging
from typing import Any

from apps.backend.engines.sd.spec import SDEngineRuntime


def apply_sd_clip_skip(
    *,
    engine: Any,
    runtime: SDEngineRuntime,
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

    if requested == 0:
        runtime.reset_clip_skip()
        try:
            effective = int(runtime.primary_classic().clip_skip)
        except Exception:
            effective = None
        if effective is None:
            logger.debug("Clip skip reset to default for %s.", label)
        else:
            logger.debug("Clip skip reset to default (%d) for %s.", effective, label)
    else:
        runtime.set_clip_skip(requested)
        logger.debug("Clip skip set to %d for %s.", requested, label)

    # Cached conditioning depends on clip-skip (text encoder pooled output changes).
    try:
        engine._cond_cache.clear()  # noqa: SLF001 (engine-owned cache; explicit invalidation)
    except Exception:
        pass

