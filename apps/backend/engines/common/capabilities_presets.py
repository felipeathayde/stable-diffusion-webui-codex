"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared EngineCapabilities tuple presets for common image engines.
Provides small constants to avoid repeating the same `tasks/devices/precision` literals across multiple engines without
hiding capability fields.

Symbols (top-level; keep in sync; no ghosts):
- `IMAGE_TASKS` (constant): Default tasks for image engines (txt2img/img2img).
- `DEFAULT_IMAGE_DEVICES` (constant): Default devices tuple for image engines.
- `DEFAULT_IMAGE_PRECISION` (constant): Default precision tuple for image engines.
- `__all__` (constant): Export list.
"""

from __future__ import annotations

from apps.backend.core.engine_interface import TaskType

IMAGE_TASKS: tuple[TaskType, TaskType] = (TaskType.TXT2IMG, TaskType.IMG2IMG)
DEFAULT_IMAGE_DEVICES: tuple[str, str] = ("cpu", "cuda")
DEFAULT_IMAGE_PRECISION: tuple[str, str, str] = ("fp16", "bf16", "fp32")

__all__ = [
    "DEFAULT_IMAGE_DEVICES",
    "DEFAULT_IMAGE_PRECISION",
    "IMAGE_TASKS",
]

