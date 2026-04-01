"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Semantic engine capability surfaces exposed to the UI layer.
Defines `SemanticEngine` tags and an `EngineParamSurface` describing which high-level UI sections and tasks are expected to be used for each engine,
including explicit masked-img2img/inpaint support plus native IP-Adapter/SUPIR discoverability, with executable defaults and recommendation hints for the live surface (for example SD15
`ddim`/`ddim`, WAN22 `uni-pc bh2`/`simple`, and LTX2 `euler`/`simple` with no sampler fiction beyond the live runtime lane).
Includes Anima (`SemanticEngine.ANIMA`) as a flow-based image engine (txt2img/img2img) requiring sha-selected external assets and exposing
`er sde` in the recommended sampler surface. FLUX.2 exposes the truthful Klein 4B/base-4B slice here: txt2img plus dedicated
image-conditioned img2img with hires enabled only after the real backend continuation path landed; LoRA remains off.
WAN semantic capabilities are bound to explicit WAN22 variant families via primary-family mapping.

Symbols (top-level; keep in sync; no ghosts):
- `SemanticEngine` (enum): UI-facing semantic engine tags used by API/frontend gating.
- `GuidanceAdvancedSurface` (dataclass): Optional per-engine support map for advanced CFG/APG controls (`extras.guidance` keys).
- `EngineParamSurface` (dataclass): Declared parameter surface for an engine (workflow flags including masked img2img/inpaint + IP-Adapter/SUPIR support + optional sampler/scheduler recommendations).
- `ENGINE_SURFACES` (constant): Mapping of semantic engine tag to `EngineParamSurface`.
- `ENGINE_ID_TO_SEMANTIC_ENGINE` (constant): Canonical mapping from API engine ids to semantic engine tags.
- `ip_adapter_support_error` (function): Return the fail-loud exact-engine/semantic-engine support error for IP-Adapter, or `None` when supported.
- `supports_ip_adapter_engine_id` (function): Return whether the exact engine id is allowed to run IP-Adapter in tranche 1.
- `supir_support_error` (function): Return the fail-loud exact-engine/semantic-engine support error for SUPIR mode, or `None` when supported.
- `build_ltx2_capability_surface` (function): Build the truthful semantic capability surface for the live LTX2 lane.
- `list_engine_capabilities` (function): Returns engine surfaces keyed by string tag for API responses.
- `semantic_engine_for_engine_id` (function): Resolve a semantic engine tag from an API engine id (fail-loud on unknown ids).
- `primary_family_for_engine_id` (function): Resolve the exact primary `ModelFamily` authority for a runtime engine id (fail-loud on unknown ids).
- `engine_supports_cfg` (function): Return whether the engine family supports classic CFG (`cfg`) via family capabilities.
- `serialize_engine_capabilities` (function): Returns engine capability surfaces as JSON-serializable dicts.
- `serialize_family_capabilities` (function): Returns model family capability surfaces as JSON-serializable dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Mapping

from apps.backend.runtime.model_registry.ltx2_execution import (
    LTX2_EXECUTION_SURFACE_KEY,
    Ltx2ExecutionSurface,
    build_ltx2_execution_surface,
)
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
    LTX2 = "ltx2"
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
    supports_img2img_masking: bool
    supports_txt2vid: bool
    supports_img2vid: bool
    supports_hires: bool
    supports_refiner: bool
    supports_lora: bool
    supports_controlnet: bool
    supports_ip_adapter: bool
    supports_supir_mode: bool = False
    # Optional: recommended sampler/scheduler lists for UI hinting.
    recommended_samplers: tuple[str, ...] | None = None
    recommended_schedulers: tuple[str, ...] | None = None
    # Optional: UI defaults for sampler/scheduler selection.
    default_sampler: str | None = None
    default_scheduler: str | None = None
    # Optional: support map for advanced guidance controls (`extras.guidance` keys).
    guidance_advanced: GuidanceAdvancedSurface | None = None
    # Optional: nested LTX-only execution-profile/default surface.
    ltx_execution_surface: Ltx2ExecutionSurface | None = None


def build_ltx2_capability_surface() -> EngineParamSurface:
    """Build the truthful semantic capability surface for the live LTX2 lane."""

    return EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler",),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
        ltx_execution_surface=build_ltx2_execution_surface(),
    )


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
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=True,
        supports_supir_mode=False,
        default_sampler="ddim",
        default_scheduler="ddim",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # SDXL image workflows (base + hires + refiner).
    SemanticEngine.SDXL: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=True,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=True,
        supports_supir_mode=True,
        default_sampler="euler",
        default_scheduler="euler_discrete",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Flux.1 (flow-based image diffusion).
    SemanticEngine.FLUX: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "euler a", "dpm++ 2m"),
        recommended_schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # FLUX.2 Klein (single-Qwen; txt2img + image-conditioned img2img; hires enabled after dedicated continuation wiring).
    SemanticEngine.FLUX2: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Z-Image (Turbo/Base variants; flow-based; tuned for simple predictor schedule).
    SemanticEngine.ZIMAGE: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Anima (Cosmos Predict2; flow-based; Qwen3-0.6B conditioning; classic CFG).
    SemanticEngine.ANIMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "euler a", "dpm++ 2m", "er sde"),
        recommended_schedulers=("simple", "beta", "normal", "exponential"),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Chroma (flow-based image generation).
    SemanticEngine.CHROMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Wan 2.2 dual-stage video (txt2vid/img2vid).
    SemanticEngine.WAN22: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=True,  # high/low LoRA slots in WAN22 panel
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("uni-pc bh2", "uni-pc", "euler", "euler a"),
        recommended_schedulers=("simple",),
        default_sampler="uni-pc bh2",
        default_scheduler="simple",
    ),
    # LTX2 distilled/core-only video workflows (txt2vid/img2vid).
    SemanticEngine.LTX2: build_ltx2_capability_surface(),
    # Hunyuan Video: video-only workflows.
    SemanticEngine.HUNYUAN_VIDEO: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        default_sampler=None,
        default_scheduler=None,
    ),
    # SVD (Stable Video Diffusion): image-to-video only today.
    SemanticEngine.SVD: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
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
    "ltx2": SemanticEngine.LTX2,
    "svd": SemanticEngine.SVD,
    "hunyuan_video": SemanticEngine.HUNYUAN_VIDEO,
}

_IP_ADAPTER_EXACT_ENGINE_REJECTS: Dict[str, str] = {
    "sd20": "Engine 'sd20' is unsupported for IP-Adapter in tranche 1.",
    "sd35": "Engine 'sd35' is unsupported for IP-Adapter in tranche 1.",
    "sdxl_refiner": "Engine 'sdxl_refiner' is unsupported for IP-Adapter in tranche 1.",
}

_SUPIR_EXACT_ENGINE_REJECTS: Dict[str, str] = {
    "sd15": "Engine 'sd15' is unsupported for SUPIR mode in tranche 1.",
    "sd20": "Engine 'sd20' is unsupported for SUPIR mode in tranche 1.",
    "sd35": "Engine 'sd35' is unsupported for SUPIR mode in tranche 1.",
    "sdxl_refiner": "Engine 'sdxl_refiner' is unsupported for SUPIR mode in tranche 1.",
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
    "ltx2": ModelFamily.LTX2,
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


def primary_family_for_engine_id(engine_id: str) -> ModelFamily:
    normalized = str(engine_id or "").strip()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    family = _ENGINE_ID_PRIMARY_FAMILY.get(normalized)
    if family is None:
        raise KeyError(f"No primary family mapping for engine id {normalized!r}.")
    return family


def ip_adapter_support_error(engine_id: str) -> str | None:
    normalized = str(engine_id or "").strip().lower()
    if normalized == "":
        return "IP-Adapter requires a non-empty engine id."
    exact_reject = _IP_ADAPTER_EXACT_ENGINE_REJECTS.get(normalized)
    if exact_reject is not None:
        return exact_reject
    try:
        semantic_engine = semantic_engine_for_engine_id(normalized)
    except KeyError:
        return f"Engine '{normalized}' is unsupported for IP-Adapter in tranche 1."
    if semantic_engine not in {SemanticEngine.SD15, SemanticEngine.SDXL}:
        return (
            f"Engine '{normalized}' is unsupported for IP-Adapter in tranche 1. "
            "Supported semantic engines: sd15, sdxl."
        )
    return None


def supports_ip_adapter_engine_id(engine_id: str) -> bool:
    return ip_adapter_support_error(engine_id) is None


def supir_support_error(engine_id: str) -> str | None:
    normalized = str(engine_id or "").strip().lower()
    if normalized == "":
        return "SUPIR mode requires a non-empty engine id."
    exact_reject = _SUPIR_EXACT_ENGINE_REJECTS.get(normalized)
    if exact_reject is not None:
        return exact_reject
    if normalized == "sdxl":
        return None
    try:
        semantic_engine = semantic_engine_for_engine_id(normalized)
    except KeyError:
        return f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1."
    if semantic_engine is SemanticEngine.SDXL:
        return f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1."
    return (
        f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1. "
        "Supported semantic engine: sdxl (exact engine id 'sdxl' only)."
    )

def engine_supports_cfg(engine_id: str) -> bool:
    from apps.backend.runtime.model_registry.family_runtime import get_family_spec

    spec = get_family_spec(primary_family_for_engine_id(engine_id))
    return bool(spec.capabilities.supports_cfg)


def serialize_engine_capabilities() -> Dict[str, Dict[str, object]]:
    """Return capabilities as plain dicts for JSON responses."""
    result: Dict[str, Dict[str, object]] = {}
    for engine, surface in list_engine_capabilities().items():
        payload = asdict(surface)
        if payload.get(LTX2_EXECUTION_SURFACE_KEY) is None:
            payload.pop(LTX2_EXECUTION_SURFACE_KEY, None)
        result[engine] = payload
    return result


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
    "ip_adapter_support_error",
    "supports_ip_adapter_engine_id",
    "supir_support_error",
    "build_ltx2_capability_surface",
    "list_engine_capabilities",
    "semantic_engine_for_engine_id",
    "primary_family_for_engine_id",
    "engine_supports_cfg",
    "serialize_engine_capabilities",
    "serialize_family_capabilities",
]
