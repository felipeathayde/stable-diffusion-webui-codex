"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler/scheduler availability policy for engines and tasks.
Defines default sampler/scheduler lists and per-engine overrides, exposing a stable API for UI gating and backend request validation.

Symbols (top-level; keep in sync; no ghosts):
- `SAMPLER_NAME` (constant): Mapping of `SamplerKind` to UI display names.
- `DEFAULT_IMAGE_SAMPLERS` (constant): Baseline sampler display names for image tasks.
- `DEFAULT_IMAGE_SCHEDULERS` (constant): Baseline scheduler display names for image tasks.
- `DEFAULT_VIDEO_SAMPLERS` (constant): Baseline sampler display names for video tasks.
- `DEFAULT_VIDEO_SCHEDULERS` (constant): Baseline scheduler display names for video tasks.
- `ENGINE_OVERRIDES` (constant): Per-engine overrides keyed by engine key and `TaskType`.
- `ENGINE_ALIAS` (constant): Engine key alias map (normalized to override keys).
- `_base_for_task` (function): Returns baseline lists for the given task kind (image vs video).
- `allowed_samplers` (function): Returns allowed sampler display names for an engine/task.
- `allowed_schedulers` (function): Returns allowed scheduler display names for an engine/task.
"""

from __future__ import annotations

from typing import Dict, List

from .engine_interface import TaskType
from apps.backend.engines.util.schedulers import SamplerKind


# Friendly display names for SamplerKind values
SAMPLER_NAME: Dict[SamplerKind, str] = {
    SamplerKind.AUTOMATIC: "Automatic",
    SamplerKind.EULER: "Euler",
    SamplerKind.EULER_A: "Euler a",
    SamplerKind.DDIM: "DDIM",
    SamplerKind.DPM2M: "DPM++ 2M",
    SamplerKind.DPM2M_SDE: "DPM++ 2M SDE",
    SamplerKind.PLMS: "PLMS",
    SamplerKind.PNDM: "PNDM",
    SamplerKind.UNI_PC: "UniPC",
}

# Baseline sets used unless an engine overrides them.
DEFAULT_IMAGE_SAMPLERS: List[str] = [
    SAMPLER_NAME[SamplerKind.AUTOMATIC],
    SAMPLER_NAME[SamplerKind.EULER_A],
    SAMPLER_NAME[SamplerKind.EULER],
    SAMPLER_NAME[SamplerKind.DPM2M],
    SAMPLER_NAME[SamplerKind.DPM2M_SDE],
    SAMPLER_NAME[SamplerKind.DDIM],
    SAMPLER_NAME[SamplerKind.PLMS],
    SAMPLER_NAME[SamplerKind.PNDM],
    SAMPLER_NAME[SamplerKind.UNI_PC],
]
DEFAULT_IMAGE_SCHEDULERS: List[str] = ["Automatic", "Karras", "Simple"]

DEFAULT_VIDEO_SAMPLERS: List[str] = [
    SAMPLER_NAME[SamplerKind.AUTOMATIC],
    SAMPLER_NAME[SamplerKind.EULER_A],
    SAMPLER_NAME[SamplerKind.EULER],
    SAMPLER_NAME[SamplerKind.DPM2M],
    SAMPLER_NAME[SamplerKind.DPM2M_SDE],
    SAMPLER_NAME[SamplerKind.DDIM],
    SAMPLER_NAME[SamplerKind.PNDM],
    SAMPLER_NAME[SamplerKind.UNI_PC],
]
DEFAULT_VIDEO_SCHEDULERS: List[str] = ["Automatic", "Karras"]

# Per-engine overrides (lowercase engine keys)
ENGINE_OVERRIDES: Dict[str, Dict[TaskType, Dict[str, List[str]]]] = {
    "flux1": {
        TaskType.TXT2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
        TaskType.IMG2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
    },
    "flux1_chroma": {
        TaskType.TXT2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
        TaskType.IMG2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
    },
    "wan22_14b": {
        TaskType.TXT2VID: {"samplers": DEFAULT_VIDEO_SAMPLERS, "schedulers": DEFAULT_VIDEO_SCHEDULERS},
        TaskType.IMG2VID: {"samplers": DEFAULT_VIDEO_SAMPLERS, "schedulers": DEFAULT_VIDEO_SCHEDULERS},
    },
    "wan22_5b": {
        TaskType.TXT2VID: {"samplers": DEFAULT_VIDEO_SAMPLERS, "schedulers": DEFAULT_VIDEO_SCHEDULERS},
        TaskType.IMG2VID: {"samplers": DEFAULT_VIDEO_SAMPLERS, "schedulers": DEFAULT_VIDEO_SCHEDULERS},
    },
}

# Engines that should reuse FLUX lists (fine-tuned variants share same controls).
ENGINE_ALIAS: Dict[str, str] = {
    "flux1_kontext": "flux1",
    "flux1_chroma": "flux1_chroma",
}


def _base_for_task(task: TaskType) -> Dict[str, List[str]]:
    if task in {TaskType.TXT2VID, TaskType.IMG2VID}:
        return {"samplers": DEFAULT_VIDEO_SAMPLERS, "schedulers": DEFAULT_VIDEO_SCHEDULERS}
    return {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS}


def allowed_samplers(engine_key: str, task: TaskType) -> List[str]:
    """Return the list of sampler display names allowed for the engine/task."""
    key = (engine_key or "").lower()
    key = ENGINE_ALIAS.get(key, key)
    overrides = ENGINE_OVERRIDES.get(key, {})
    entry = overrides.get(task)
    samplers = (entry or _base_for_task(task))["samplers"]
    # Return a copy to keep internal constants immutable
    return list(dict.fromkeys(samplers)) or [SAMPLER_NAME[SamplerKind.AUTOMATIC]]


def allowed_schedulers(engine_key: str, task: TaskType) -> List[str]:
    """Return the list of scheduler display names allowed for the engine/task."""
    key = (engine_key or "").lower()
    key = ENGINE_ALIAS.get(key, key)
    overrides = ENGINE_OVERRIDES.get(key, {})
    entry = overrides.get(task)
    schedulers = (entry or _base_for_task(task))["schedulers"]
    return list(dict.fromkeys(schedulers)) or ["Automatic"]


__all__ = ["allowed_samplers", "allowed_schedulers"]
