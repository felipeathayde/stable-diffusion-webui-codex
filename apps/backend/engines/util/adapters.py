"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request→processing adapters for txt2img/img2img (hires/swap-model/refiner/smart flags).
Builds Codex processing objects from API request DTOs, including Hires, first-pass swap-model, and Refiner configs (with hires tile config),
per-job smart runtime flags, and strict pass-through overrides like `extras.er_sde` for sampler runtime wiring.

Symbols (top-level; keep in sync; no ghosts):
- `_build_swap_model_config` (function): Builds a typed `SwapModelConfig` from request payload data.
- `_build_swap_stage_config` (function): Builds a typed `SwapStageConfig` for the global first-pass `swap_model` stage.
- `_build_hires_config` (function): Builds a `CodexHiresConfig` from request payload data (including hires tile config + nested hires refiner).
- `_build_refiner_config` (function): Builds a `RefinerConfig` from request payload data.
- `build_txt2img_processing` (function): Converts a `Txt2ImgRequest` into a fully-populated `CodexProcessingTxt2Img`.
- `build_img2img_processing` (function): Converts an `Img2ImgRequest` into a fully-populated `CodexProcessingImg2Img` (including inpaint mask wiring).
"""

from __future__ import annotations

import math
from typing import Any, Mapping
import logging

from apps.backend.core.requests import Img2ImgRequest, Txt2ImgRequest
from apps.backend.runtime.processing.models import (
    CodexHiresConfig,
    CodexProcessingImg2Img,
    CodexProcessingTxt2Img,
    RefinerConfig,
    SwapStageConfig,
    SwapModelConfig,
)
from apps.backend.runtime.vision.upscalers.specs import tile_config_from_payload

_log = logging.getLogger(__name__)


def _parse_batch_count(extras: Mapping[str, Any] | None) -> int:
    if not isinstance(extras, Mapping):
        return 1
    raw = extras.get("batch_count")
    if raw is None:
        return 1
    if isinstance(raw, bool):
        raise ValueError("Invalid 'extras.batch_count': expected integer >= 1, got boolean.")
    if isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {raw!r}.")
        parsed = int(raw)
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            raise ValueError("Invalid 'extras.batch_count': expected integer >= 1, got empty string.")
        try:
            parsed = int(token, 10)
        except ValueError as exc:
            raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {raw!r}.") from exc
    else:
        raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {type(raw).__name__}.")
    if parsed < 1:
        raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {parsed}.")
    return parsed


def _parse_optional_finite_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{field}': expected finite number, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid '{field}': expected finite number, got {value!r}.")
    return parsed


def _build_hires_config(data: Mapping[str, Any] | None, *, default_cfg: float, default_distilled: float, default_denoise: float) -> CodexHiresConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    if not enabled:
        return CodexHiresConfig(
            enabled=False,
            scale=1.0,
            denoise=0.0,
            upscaler=None,
            second_pass_steps=0,
            resize_x=0,
            resize_y=0,
            prompt="",
            negative_prompt="",
            cfg=default_cfg,
            distilled_cfg=default_distilled,
            sampler_name=None,
            scheduler=None,
            additional_modules=tuple(),
            swap_model=None,
            refiner=None,
        )

    prompt_value = payload.get("prompt", "")
    negative_value = payload.get("negative_prompt", "")
    cfg_value = payload.get("cfg", default_cfg)
    distilled_value = payload.get("distilled_cfg", default_distilled)
    sampler_name = payload.get("sampler_name")
    scheduler = payload.get("scheduler")

    modules_raw = payload.get("additional_modules")
    if modules_raw is None:
        modules_tuple: tuple[str, ...] = tuple()
    elif isinstance(modules_raw, (list, tuple)):
        modules_tuple = tuple(str(m) for m in modules_raw)
    else:
        modules_tuple = (str(modules_raw),)

    raw_swap_model = payload.get("swap_model")
    if raw_swap_model is not None and not isinstance(raw_swap_model, Mapping):
        raise ValueError("'hires.swap_model' must be an object when provided.")
    swap_model_cfg = _build_swap_model_config(raw_swap_model)
    if raw_swap_model is not None and swap_model_cfg is None:
        raise ValueError("'hires.swap_model' requires 'model' or 'model_sha' when provided.")
    refiner_cfg = _build_refiner_config(payload.get("refiner"), default_cfg=cfg_value, context="hires.refiner")
    tile_cfg = tile_config_from_payload(payload.get("tile"), context="hires.tile")

    return CodexHiresConfig(
        enabled=enabled,
        scale=float(payload.get("scale", 2.0)) if enabled else 1.0,
        denoise=float(payload.get("denoise", default_denoise)) if enabled else 0.0,
        upscaler=payload.get("upscaler") if enabled else None,
        tile=tile_cfg,
        second_pass_steps=int(payload.get("steps", 0)) if enabled else 0,
        resize_x=int(payload.get("resize_x", 0)) if enabled else 0,
        resize_y=int(payload.get("resize_y", 0)) if enabled else 0,
        prompt=str(prompt_value),
        negative_prompt=str(negative_value),
        cfg=float(cfg_value),
        distilled_cfg=float(distilled_value),
        sampler_name=sampler_name,
        scheduler=scheduler,
        additional_modules=modules_tuple,
        swap_model=swap_model_cfg,
        refiner=refiner_cfg if refiner_cfg.enabled else None,
    )


def _build_swap_model_config(data: Mapping[str, Any] | None) -> SwapModelConfig | None:
    payload = data or {}
    model_raw = payload.get("model")
    model_name = str(model_raw).strip() if model_raw is not None else ""
    model_sha_raw = payload.get("model_sha")
    model_sha = str(model_sha_raw).strip().lower() if model_sha_raw is not None else ""
    if not model_name and not model_sha:
        return None

    raw_tenc_path = payload.get("tenc_path")
    tenc_path: str | tuple[str, ...] | None = None
    if isinstance(raw_tenc_path, str):
        normalized = raw_tenc_path.strip()
        tenc_path = normalized or None
    elif isinstance(raw_tenc_path, (list, tuple)):
        normalized_paths = tuple(
            str(entry).strip()
            for entry in raw_tenc_path
            if isinstance(entry, str) and str(entry).strip()
        )
        tenc_path = normalized_paths or None

    text_encoder_override_raw = payload.get("text_encoder_override")
    text_encoder_override = (
        {str(key): value for key, value in text_encoder_override_raw.items()}
        if isinstance(text_encoder_override_raw, Mapping)
        else None
    )

    model_format_raw = payload.get("model_format")
    model_format = str(model_format_raw).strip().lower() if isinstance(model_format_raw, str) else None
    if model_format not in {None, "checkpoint", "diffusers", "gguf"}:
        raise ValueError(f"Invalid swap-model model_format: {model_format_raw!r}.")

    zimage_variant_raw = payload.get("zimage_variant")
    zimage_variant = str(zimage_variant_raw).strip().lower() if isinstance(zimage_variant_raw, str) else None
    if zimage_variant not in {None, "turbo", "base"}:
        raise ValueError(f"Invalid swap-model zimage_variant: {zimage_variant_raw!r}.")

    vae_source_raw = payload.get("vae_source")
    vae_source = str(vae_source_raw).strip().lower() if isinstance(vae_source_raw, str) else None
    if vae_source not in {None, "built_in", "external"}:
        raise ValueError(f"Invalid swap-model vae_source: {vae_source_raw!r}.")

    checkpoint_core_only = payload.get("checkpoint_core_only")
    if checkpoint_core_only is not None and not isinstance(checkpoint_core_only, bool):
        raise ValueError("swap_model.checkpoint_core_only must be a boolean when provided.")

    vae_path_raw = payload.get("vae_path")
    vae_path = str(vae_path_raw).strip() if isinstance(vae_path_raw, str) else None

    return SwapModelConfig(
        model=model_name or None,
        model_sha=model_sha or None,
        checkpoint_core_only=checkpoint_core_only,
        model_format=model_format,  # type: ignore[arg-type]
        zimage_variant=zimage_variant,  # type: ignore[arg-type]
        vae_source=vae_source,  # type: ignore[arg-type]
        vae_path=vae_path or None,
        tenc_path=tenc_path,
        text_encoder_override=text_encoder_override,
    )


def _build_refiner_config(data: Mapping[str, Any] | None, *, default_cfg: float, context: str) -> RefinerConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    swap_at_step = int(payload.get("switch_at_step", 0) or 0)
    cfg = float(payload.get("cfg", default_cfg))
    seed = int(payload.get("seed", -1))
    selection = _build_swap_model_config(payload)
    if selection is not None and selection.zimage_variant is not None:
        raise ValueError(f"'{context}.zimage_variant' is unsupported.")
    if enabled and swap_at_step <= 0:
        raise ValueError(f"'{context}.switch_at_step' must be >= 1 when '{context}.enable' is true.")
    if enabled and selection is None:
        raise ValueError(f"'{context}' requires 'model' or 'model_sha' when enabled.")

    return RefinerConfig(
        enabled=enabled,
        swap_at_step=swap_at_step,
        cfg=cfg,
        seed=seed,
        selection=selection or SwapModelConfig(),
    )


def _build_swap_stage_config(data: Mapping[str, Any] | None, *, default_cfg: float, context: str) -> SwapStageConfig | None:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    swap_at_step = int(payload.get("switch_at_step", 0) or 0)
    cfg = float(payload.get("cfg", default_cfg))
    seed = int(payload.get("seed", -1))
    selection = _build_swap_model_config(payload)
    if not enabled:
        return None
    if swap_at_step <= 0:
        raise ValueError(f"'{context}.switch_at_step' must be >= 1 when '{context}.enable' is true.")
    if selection is None:
        raise ValueError(f"'{context}' requires 'model' or 'model_sha' when enabled.")
    return SwapStageConfig(
        enabled=True,
        swap_at_step=swap_at_step,
        cfg=cfg,
        seed=seed,
        selection=selection,
    )


def build_txt2img_processing(req: Txt2ImgRequest) -> CodexProcessingTxt2Img:
    _log.debug(
        "build_txt2img_processing: size=%dx%d steps=%s sampler=%s scheduler=%s cfg=%s seed=%s hr=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        req.seed,
        bool(req.hires),
    )
    metadata = dict(req.metadata or {})
    extras = req.extras if isinstance(req.extras, Mapping) else {}
    iterations = _parse_batch_count(extras)
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))
    distilled_cfg = _parse_optional_finite_float(metadata.get("distilled_cfg_scale"), field="metadata.distilled_cfg_scale")
    if distilled_cfg is None:
        distilled_cfg = 3.5

    hires_cfg = _build_hires_config(
        req.hires if isinstance(req.hires, dict) else {},
        default_cfg=req.guidance_scale or 7.0,
        default_distilled=distilled_cfg,
        default_denoise=0.5,
    )
    processing = CodexProcessingTxt2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=iterations,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=distilled_cfg,
        width=req.width,
        height=req.height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        metadata=metadata,
        smart_offload=smart_offload,
        smart_fallback=smart_fallback,
        smart_cache=smart_cache,
    )
    processing.swap_model = _build_swap_stage_config(
        extras.get("swap_model"),
        default_cfg=processing.guidance_scale,
        context="extras.swap_model",
    )
    refiner_cfg = _build_refiner_config(
        extras.get("refiner"),
        default_cfg=processing.guidance_scale,
        context="extras.refiner",
    )
    processing.refiner = refiner_cfg if refiner_cfg.enabled else None
    if hires_cfg.enabled:
        processing.enable_hires(cfg=hires_cfg)
    for key, value in extras.items():
        if key == "er_sde" and isinstance(value, Mapping):
            processing.update_override(key, dict(value))
        else:
            processing.update_override(key, value)
        if key == "eta_noise_seed_delta":
            try:
                processing.eta_noise_seed_delta = int(value)
            except Exception:
                processing.eta_noise_seed_delta = value
    return processing


def build_img2img_processing(req: Img2ImgRequest) -> CodexProcessingImg2Img:
    _log.debug(
        "build_img2img_processing: size=%sx%s steps=%s sampler=%s scheduler=%s cfg=%s denoise=%s has_init=%s has_mask=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        getattr(req, "denoise_strength", None),
        bool(getattr(req, "init_image", None)),
        bool(getattr(req, "mask", None)),
    )
    width = req.width
    height = req.height
    if getattr(req, "init_image", None) is not None:
        try:
            w, h = req.init_image.size  # type: ignore[attr-defined]
            width = width or w
            height = height or h
        except Exception:
            pass

    metadata = dict(req.metadata or {})
    extras = req.extras if isinstance(req.extras, Mapping) else {}
    iterations = _parse_batch_count(extras)
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))

    distilled_cfg = _parse_optional_finite_float(metadata.get("distilled_cfg_scale"), field="metadata.distilled_cfg_scale")
    if distilled_cfg is None:
        distilled_cfg = 3.5

    image_cfg_scale = _parse_optional_finite_float(metadata.get("image_cfg_scale"), field="metadata.image_cfg_scale")
    per_step_blend_strength = _parse_optional_finite_float(
        req.per_step_blend_strength,
        field="Img2ImgRequest.per_step_blend_strength",
    )
    if per_step_blend_strength is None:
        raise ValueError("Img2ImgRequest.per_step_blend_strength must be provided explicitly")
    if per_step_blend_strength < 0.0 or per_step_blend_strength > 1.0:
        raise ValueError("Img2ImgRequest.per_step_blend_strength must be between 0.0 and 1.0")

    mask_round = bool(getattr(req, "mask_round", True))
    processing = CodexProcessingImg2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=iterations,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=distilled_cfg,
        width=width,
        height=height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        init_image=req.init_image,
        mask=req.mask,
        mask_enforcement=getattr(req, "mask_enforcement", None),
        per_step_blend_strength=per_step_blend_strength,
        mask_region_split=bool(getattr(req, "mask_region_split", False)),
        mask_blur=int(getattr(req, "mask_blur", 4) or 0),
        mask_blur_x=int(getattr(req, "mask_blur_x", getattr(req, "mask_blur", 4)) or 0),
        mask_blur_y=int(getattr(req, "mask_blur_y", getattr(req, "mask_blur", 4)) or 0),
        mask_round=mask_round,
        round_image_mask=mask_round,
        image_mask=req.mask,
        inpainting_fill=int(getattr(req, "inpainting_fill", 0) or 0),
        inpaint_full_res_padding=int(getattr(req, "inpaint_full_res_padding", 0) or 0),
        inpainting_mask_invert=int(getattr(req, "inpainting_mask_invert", 0) or 0),
        denoising_strength=float(req.denoise_strength),
        image_cfg_scale=image_cfg_scale,
        metadata=metadata,
        smart_offload=smart_offload,
        smart_fallback=smart_fallback,
        smart_cache=smart_cache,
    )
    if req.hires:
        hires_cfg = _build_hires_config(
            req.hires,
            default_cfg=processing.guidance_scale,
            default_distilled=processing.distilled_guidance_scale,
            default_denoise=processing.denoising_strength,
        )
        if hires_cfg.enabled:
            processing.enable_hires(hires_cfg)
    for key, value in extras.items():
        if key == "er_sde" and isinstance(value, Mapping):
            processing.update_override(key, dict(value))
        else:
            processing.update_override(key, value)
        if key == "eta_noise_seed_delta":
            try:
                processing.eta_noise_seed_delta = int(value)
            except Exception:
                processing.eta_noise_seed_delta = value
    return processing
