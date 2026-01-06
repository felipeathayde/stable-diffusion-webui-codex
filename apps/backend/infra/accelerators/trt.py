"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: TensorRT accelerator adapter stub (import-safe).
Provides a minimal `TensorRTAccelerator` with an availability probe and a placeholder `apply_to_diffusers(...)` hook.

Symbols (top-level; keep in sync; no ghosts):
- `TensorRTAccelerator` (class): TensorRT adapter with runtime availability probe (`tensorrt` import or `CODEX_TRT_AVAILABLE`) and a no-op `apply_to_diffusers`.
"""

from __future__ import annotations

from typing import Any

import os


class TensorRTAccelerator:
    def __init__(self) -> None:
        self._available = False
        # Allow a manual override for testing environments
        env_flag = os.getenv("CODEX_TRT_AVAILABLE")
        if env_flag is not None:
            self._available = env_flag.strip().lower() in {"1", "true", "yes", "on"}
            return
        # Heuristic probe: try importing TensorRT only to detect presence
        try:
            import tensorrt as _trt  # type: ignore  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return bool(self._available)

    def apply_to_diffusers(self, pipe: Any) -> None:
        """Apply TensorRT optimizations to a diffusers pipeline.

        This stub intentionally does nothing when unavailable. If available,
        a real implementation should be plugged here in the future.
        """
        if not self._available:
            return
        # TODO: Integrate a proper TensorRT application path when supported
        return
