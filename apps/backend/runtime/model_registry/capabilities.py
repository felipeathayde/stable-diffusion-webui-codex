"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Semantic engine capability surfaces exposed to the UI layer.
Defines `SemanticEngine` tags and an `EngineParamSurface` describing which high-level UI sections and tasks are expected to be used for each engine.

Symbols (top-level; keep in sync; no ghosts):
- `SemanticEngine` (enum): UI-facing semantic engine tags used by API/frontend gating.
- `EngineParamSurface` (dataclass): Declared parameter surface for an engine (workflow flags + optional sampler/scheduler allow-lists).
- `ENGINE_SURFACES` (constant): Mapping of semantic engine tag to `EngineParamSurface`.
- `list_engine_capabilities` (function): Returns engine surfaces keyed by string tag for API responses.
- `serialize_engine_capabilities` (function): Returns engine capability surfaces as JSON-serializable dicts.
- `serialize_family_capabilities` (function): Returns model family capability surfaces as JSON-serializable dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Mapping


class SemanticEngine(str, Enum):
    """Semantic engine tags exposed to the UI layer.

    These align with `_detect_semantic_engine()` in `run_api` and describe
    high-level workflow families rather than individual checkpoints.
    """

    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux1"
    ZIMAGE = "zimage"
    CHROMA = "chroma"
    WAN22 = "wan22"
    HUNYUAN_VIDEO = "hunyuan_video"
    SVD = "svd"


@dataclass(frozen=True)
class EngineParamSurface:
    """Declared parameter surface for a semantic engine.

    This describes which high-level workflows and feature sections are intended
    to be used from the Codex UI. It is deliberately narrower than the full
    backend capabilities so the frontend can hide params that have no effect
    for a given engine.
    """

    supports_txt2img: bool
    supports_img2img: bool
    supports_txt2vid: bool
    supports_img2vid: bool
    supports_highres: bool
    supports_refiner: bool
    supports_lora: bool
    supports_controlnet: bool
    # Optional: restrict UI to only these samplers/schedulers. None = allow all.
    samplers: tuple[str, ...] | None = None
    schedulers: tuple[str, ...] | None = None
    # Optional: UI defaults for sampler/scheduler selection.
    default_sampler: str | None = None
    default_scheduler: str | None = None


ENGINE_SURFACES: Dict[SemanticEngine, EngineParamSurface] = {
    # Classic SD1.x-style image generation.
    SemanticEngine.SD15: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_highres=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        default_sampler="pndm",
        default_scheduler="ddim",
    ),
    # SDXL image workflows (base + hires + refiner).
    SemanticEngine.SDXL: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_highres=True,
        supports_refiner=True,
        supports_lora=True,
        supports_controlnet=False,
        default_sampler="euler",
        default_scheduler="euler_discrete",
    ),
    # Flux.1 (flow-based image diffusion).
    SemanticEngine.FLUX: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        samplers=("euler", "euler a", "ddim", "dpm++ 2m"),
        schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Z-Image (Turbo/Base variants; flow-based; tuned for simple predictor schedule).
    SemanticEngine.ZIMAGE: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        samplers=("euler", "dpm++ 2m"),
        schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Chroma (flow-based image generation).
    SemanticEngine.CHROMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        samplers=("euler", "dpm++ 2m", "ddim"),
        schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Wan 2.2 dual-stage video (txt2vid/img2vid).
    SemanticEngine.WAN22: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=True,  # high/low LoRA slots in WAN22 panel
        supports_controlnet=False,
        default_sampler="uni-pc",
        default_scheduler="simple",
    ),
    # Hunyuan Video: video-only workflows.
    SemanticEngine.HUNYUAN_VIDEO: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        default_sampler="ddpm",
        default_scheduler="beta",
    ),
    # SVD (Stable Video Diffusion): image-to-video only today.
    SemanticEngine.SVD: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_txt2vid=False,
        supports_img2vid=True,
        supports_highres=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
    ),
}


def list_engine_capabilities() -> Mapping[str, EngineParamSurface]:
    """Return engine capability surfaces keyed by semantic engine tag."""
    return {engine.value: surface for engine, surface in ENGINE_SURFACES.items()}


def serialize_engine_capabilities() -> Dict[str, Dict[str, object]]:
    """Return capabilities as plain dicts for JSON responses."""
    return {engine: asdict(surface) for engine, surface in list_engine_capabilities().items()}


def serialize_family_capabilities() -> Dict[str, Dict[str, object]]:
    """Return FamilyCapabilities for all model families as JSON-serializable dicts.

    Returns:
        Dict mapping family name (e.g. "FLUX", "SDXL") to capability dict.
    """
    from apps.backend.runtime.model_registry.family_runtime import FAMILY_RUNTIME_SPECS

    result = {}
    for family, spec in FAMILY_RUNTIME_SPECS.items():
        result[family.value] = spec.capabilities.to_dict()
    return result


__all__ = [
    "SemanticEngine",
    "EngineParamSurface",
    "ENGINE_SURFACES",
    "list_engine_capabilities",
    "serialize_engine_capabilities",
    "serialize_family_capabilities",
]
