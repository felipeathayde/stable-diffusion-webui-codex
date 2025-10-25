"""Sampler/scheduler availability policy for engines and tasks."""
from __future__ import annotations

from typing import Dict, List

from .engine_interface import TaskType
from apps.server.backend.engines.util.schedulers import SamplerKind


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
]
DEFAULT_VIDEO_SCHEDULERS: List[str] = ["Automatic", "Karras"]

# Per-engine overrides (lowercase engine keys)
ENGINE_OVERRIDES: Dict[str, Dict[TaskType, Dict[str, List[str]]]] = {
    "flux": {
        TaskType.TXT2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
        TaskType.IMG2IMG: {"samplers": DEFAULT_IMAGE_SAMPLERS, "schedulers": DEFAULT_IMAGE_SCHEDULERS},
    },
    "chroma": {
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
    "flux.1": "flux",
    "flux-1": "flux",
    "flux_schnell": "flux",
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
