"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Semantic engine capability surfaces exposed to the UI layer.
Defines `SemanticEngine` tags and an `EngineParamSurface` describing which high-level UI sections and tasks are expected to be used for each engine.
Includes Anima (`SemanticEngine.ANIMA`) as a flow-based image engine (txt2img/img2img) requiring sha-selected external assets and exposing
`er sde` in the sampler allowlist. FLUX.2 exposes the truthful Klein 4B/base-4B slice here: txt2img plus dedicated
image-conditioned img2img with hires enabled only after the real backend continuation path landed; LoRA remains off.
WAN semantic capabilities are bound to explicit WAN22 variant families via primary-family mapping.

Symbols (top-level; keep in sync; no ghosts):
- `SemanticEngine` (enum): UI-facing semantic engine tags used by API/frontend gating.
- `GuidanceAdvancedSurface` (dataclass): Optional per-engine support map for advanced CFG/APG controls (`extras.guidance` keys).
- `EngineParamSurface` (dataclass): Declared parameter surface for an engine (workflow flags + optional sampler/scheduler allow-lists).
- `ENGINE_SURFACES` (constant): Mapping of semantic engine tag to `EngineParamSurface`.
- `ENGINE_ID_TO_SEMANTIC_ENGINE` (constant): Canonical mapping from API engine ids to semantic engine tags.
- `list_engine_capabilities` (function): Returns engine surfaces keyed by string tag for API responses.
- `semantic_engine_for_engine_id` (function): Resolve a semantic engine tag from an API engine id (fail-loud on unknown ids).
- `engine_supports_cfg` (function): Return whether the engine family supports classic CFG (`cfg`) via family capabilities.
- `serialize_engine_capabilities` (function): Returns engine capability surfaces as JSON-serializable dicts.
- `serialize_family_capabilities` (function): Returns model family capability surfaces as JSON-serializable dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Mapping

from apps.backend.runtime.model_registry.specs import ModelFamily


class SemanticEngine(str, Enum):
    """Semantic engine tags exposed to the UI layer.

    These align with `_detect_semantic_engine()` in `run_api` and describe
    high-level workflow families rather than individual checkpoints.
    """

    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux1"
    FLUX2 = "flux2"
    ZIMAGE = "zimage"
    ANIMA = "anima"
    CHROMA = "chroma"
    WAN22 = "wan22"
    HUNYUAN_VIDEO = "hunyuan_video"
    SVD = "svd"


@dataclass(frozen=True)
class GuidanceAdvancedSurface:
    """Per-engine support map for advanced guidance controls exposed in `extras.guidance`."""

    apg_enabled: bool = False
    apg_start_step: bool = False
    apg_eta: bool = False
    apg_momentum: bool = False
    apg_norm_threshold: bool = False
    apg_rescale: bool = False
    guidance_rescale: bool = False
    cfg_trunc_ratio: bool = False
    renorm_cfg: bool = False


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
    supports_hires: bool
    supports_refiner: bool
    supports_lora: bool
    supports_controlnet: bool
    # Optional: restrict UI to only these samplers/schedulers. None = allow all.
    samplers: tuple[str, ...] | None = None
    schedulers: tuple[str, ...] | None = None
    # Optional: UI defaults for sampler/scheduler selection.
    default_sampler: str | None = None
    default_scheduler: str | None = None
    # Optional: support map for advanced guidance controls (`extras.guidance` keys).
    guidance_advanced: GuidanceAdvancedSurface | None = None


_GUIDANCE_ADVANCED_CLASSIC_CFG = GuidanceAdvancedSurface(
    apg_enabled=True,
    apg_start_step=True,
    apg_eta=True,
    apg_momentum=True,
    apg_norm_threshold=True,
    apg_rescale=True,
    guidance_rescale=True,
    cfg_trunc_ratio=True,
    renorm_cfg=True,
)


ENGINE_SURFACES: Dict[SemanticEngine, EngineParamSurface] = {
    # Classic SD1.x-style image generation.
    SemanticEngine.SD15: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        default_sampler="pndm",
        default_scheduler="ddim",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # SDXL image workflows (base + hires + refiner).
    SemanticEngine.SDXL: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=True,
        supports_lora=True,
        supports_controlnet=False,
        default_sampler="euler",
        default_scheduler="euler_discrete",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Flux.1 (flow-based image diffusion).
    SemanticEngine.FLUX: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        samplers=("euler", "euler a", "ddim", "dpm++ 2m"),
        schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # FLUX.2 Klein (single-Qwen; txt2img + image-conditioned img2img; hires enabled after dedicated continuation wiring).
    SemanticEngine.FLUX2: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        samplers=("euler", "dpm++ 2m"),
        schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Z-Image (Turbo/Base variants; flow-based; tuned for simple predictor schedule).
    SemanticEngine.ZIMAGE: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        samplers=("euler", "dpm++ 2m"),
        schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Anima (Cosmos Predict2; flow-based; Qwen3-0.6B conditioning; classic CFG).
    SemanticEngine.ANIMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        samplers=("euler", "euler a", "dpm++ 2m", "er sde"),
        schedulers=("simple", "beta", "normal", "exponential"),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Chroma (flow-based image generation).
    SemanticEngine.CHROMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
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
        supports_hires=False,
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
        supports_hires=False,
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
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
    ),
}

ENGINE_ID_TO_SEMANTIC_ENGINE: Dict[str, SemanticEngine] = {
    "sd15": SemanticEngine.SD15,
    "sd20": SemanticEngine.SD15,
    "sdxl": SemanticEngine.SDXL,
    "sdxl_refiner": SemanticEngine.SDXL,
    "sd35": SemanticEngine.SDXL,
    "flux1": SemanticEngine.FLUX,
    "flux1_kontext": SemanticEngine.FLUX,
    "flux1_fill": SemanticEngine.FLUX,
    "flux2": SemanticEngine.FLUX2,
    "flux1_chroma": SemanticEngine.CHROMA,
    "zimage": SemanticEngine.ZIMAGE,
    "anima": SemanticEngine.ANIMA,
    "wan22_5b": SemanticEngine.WAN22,
    "wan22_14b": SemanticEngine.WAN22,
    "wan22_14b_animate": SemanticEngine.WAN22,
    "svd": SemanticEngine.SVD,
    "hunyuan_video": SemanticEngine.HUNYUAN_VIDEO,
}

_ENGINE_ID_PRIMARY_FAMILY: Dict[str, ModelFamily] = {
    "sd15": ModelFamily.SD15,
    "sd20": ModelFamily.SD20,
    "sdxl": ModelFamily.SDXL,
    "sdxl_refiner": ModelFamily.SDXL_REFINER,
    "sd35": ModelFamily.SD35,
    "flux1": ModelFamily.FLUX,
    "flux1_kontext": ModelFamily.FLUX_KONTEXT,
    "flux1_fill": ModelFamily.FLUX,
    "flux2": ModelFamily.FLUX2,
    "flux1_chroma": ModelFamily.CHROMA,
    "zimage": ModelFamily.ZIMAGE,
    "anima": ModelFamily.ANIMA,
    "wan22_5b": ModelFamily.WAN22_5B,
    "wan22_14b": ModelFamily.WAN22_14B,
    "wan22_14b_animate": ModelFamily.WAN22_ANIMATE,
    "hunyuan_video": ModelFamily.HUNYUAN,
    "svd": ModelFamily.SVD,
}


def list_engine_capabilities() -> Mapping[str, EngineParamSurface]:
    """Return engine capability surfaces keyed by semantic engine tag."""
    return {engine.value: surface for engine, surface in ENGINE_SURFACES.items()}


def semantic_engine_for_engine_id(engine_id: str) -> SemanticEngine:
    normalized = str(engine_id or "").strip()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    if normalized not in ENGINE_ID_TO_SEMANTIC_ENGINE:
        raise KeyError(f"Unknown engine id for semantic mapping: {normalized!r}")
    return ENGINE_ID_TO_SEMANTIC_ENGINE[normalized]


def engine_supports_cfg(engine_id: str) -> bool:
    from apps.backend.runtime.model_registry.family_runtime import get_family_spec

    normalized = str(engine_id or "").strip()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    family = _ENGINE_ID_PRIMARY_FAMILY.get(normalized)
    if family is None:
        raise KeyError(f"No primary family mapping for engine id {normalized!r}.")
    spec = get_family_spec(family)
    return bool(spec.capabilities.supports_cfg)


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
    "GuidanceAdvancedSurface",
    "EngineParamSurface",
    "ENGINE_SURFACES",
    "ENGINE_ID_TO_SEMANTIC_ENGINE",
    "list_engine_capabilities",
    "semantic_engine_for_engine_id",
    "engine_supports_cfg",
    "serialize_engine_capabilities",
    "serialize_family_capabilities",
]
