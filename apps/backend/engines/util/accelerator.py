"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-side accelerator selection for diffusers pipelines.
Applies an accelerator backend (or no-op) based on explicit input (payload-driven).

Symbols (top-level; keep in sync; no ghosts):
- `_get_selected_accelerator` (function): Reads accelerator selection from explicit input (defaults to `"none"`).
- `apply_to_diffusers_pipeline` (function): Applies the selected accelerator to a diffusers pipeline or raises with cause.
"""

from __future__ import annotations

from typing import Any, Optional


def _get_selected_accelerator() -> str:
    # Accelerator selection is payload-driven; default to none.
    return "none"


def apply_to_diffusers_pipeline(pipe: Any, *, accelerator: Optional[str] = None, logger=None) -> str:
    """Apply a selected accelerator to a diffusers pipeline or raise with cause.

    - 'none'/'off' performs no acceleration.
    - 'tensorrt' requires the built-in accelerator to be available; otherwise an error is raised.
    """
    choice = (accelerator or _get_selected_accelerator()).lower().strip()
    if choice in ("", "none", "off"):
        return "none"

    if choice == "tensorrt":
        from ...accelerators.trt import TensorRTAccelerator  # type: ignore
        acc = TensorRTAccelerator()
        if not acc.is_available():
            raise RuntimeError("TensorRT accelerator requested but not available in this environment")
        try:
            acc.apply_to_diffusers(pipe)
        except Exception as ex:
            raise RuntimeError(f"Failed to apply TensorRT accelerator: {ex}") from ex
        if logger:
            logger.info("accelerator applied: TensorRT (built-in)")
        return "tensorrt"

    raise ValueError(f"Unsupported accelerator '{choice}'. Allowed: none, tensorrt")
