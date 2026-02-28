"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request→processing adapters for txt2img/img2img (hires/refiner/smart flags).
Builds Codex processing objects from API request DTOs, including Hires/Refiner configs (with hires tile config), per-job smart runtime
flags, and strict pass-through overrides like `extras.er_sde` for sampler runtime wiring.

Symbols (top-level; keep in sync; no ghosts):
- `_build_hires_config` (function): Builds a `CodexHiresConfig` from request payload data (including hires tile config + nested hires refiner).
- `_build_refiner_config` (function): Builds a `RefinerConfig` from request payload data.
- `build_txt2img_processing` (function): Converts a `Txt2ImgRequest` into a fully-populated `CodexProcessingTxt2Img`.
- `build_img2img_processing` (function): Converts an `Img2ImgRequest` into a fully-populated `CodexProcessingImg2Img` (including inpaint mask wiring).
"""

from __future__ import annotations

from typing import Any, Mapping
import logging

from apps.backend.core.requests import Img2ImgRequest, Txt2ImgRequest
from apps.backend.runtime.processing.models import (
    CodexHiresConfig,
    CodexProcessingImg2Img,
    CodexProcessingTxt2Img,
    RefinerConfig,
)
from apps.backend.runtime.vision.upscalers.specs import tile_config_from_payload

_log = logging.getLogger(__name__)


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
            checkpoint_name=None,
            refiner=None,
        )

    prompt_value = payload.get("hr_prompt")
    if prompt_value is None:
        prompt_value = payload.get("prompt", "")
    negative_value = payload.get("hr_negative_prompt")
    if negative_value is None:
        negative_value = payload.get("negative_prompt", "")

    cfg_value = payload.get("hr_cfg")
    if cfg_value is None:
        cfg_value = payload.get("cfg", default_cfg)

    distilled_value = payload.get("hr_distilled_cfg")
    if distilled_value is None:
        distilled_value = payload.get("distilled_cfg", default_distilled)

    sampler_name = payload.get("hr_sampler_name")
    if sampler_name is None:
        sampler_name = payload.get("sampler")

    scheduler = payload.get("hr_scheduler")
    if scheduler is None:
        scheduler = payload.get("scheduler")

    modules_raw = payload.get("hr_additional_modules")
    if modules_raw is None:
        modules_raw = payload.get("additional_modules")
    if modules_raw is None:
        modules_tuple: tuple[str, ...] = tuple()
    elif isinstance(modules_raw, (list, tuple)):
        modules_tuple = tuple(str(m) for m in modules_raw)
    else:
        modules_tuple = (str(modules_raw),)

    checkpoint_name = payload.get("hr_checkpoint_name")
    if checkpoint_name is None:
        checkpoint_name = payload.get("checkpoint")

    refiner_cfg = _build_refiner_config(payload.get("refiner"), default_cfg=cfg_value)
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
        checkpoint_name=checkpoint_name,
        refiner=refiner_cfg if refiner_cfg.enabled else None,
    )


def _build_refiner_config(data: Mapping[str, Any] | None, *, default_cfg: float) -> RefinerConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    swap_at_step = int(payload.get("switch_at_step", 0) or 0)
    cfg = float(payload.get("cfg", default_cfg))
    seed = int(payload.get("seed", -1))
    model_raw = payload.get("model")
    model_name = str(model_raw).strip() if model_raw is not None else ""
    vae_raw = payload.get("vae")
    vae_name = str(vae_raw).strip() if vae_raw else ""

    return RefinerConfig(
        enabled=enabled and swap_at_step > 0,
        swap_at_step=swap_at_step,
        cfg=cfg,
        seed=seed,
        model=model_name or None,
        vae=vae_name or None,
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
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))
    distilled_cfg = 3.5
    try:
        meta_distilled = metadata.get("distilled_cfg_scale")
        if meta_distilled is not None:
            distilled_cfg = float(meta_distilled)
    except Exception:
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
        iterations=1,
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
    refiner_cfg = _build_refiner_config(req.extras.get("refiner") if isinstance(req.extras, Mapping) else None, default_cfg=processing.guidance_scale)
    processing.refiner = refiner_cfg if refiner_cfg.enabled else None
    if hires_cfg.enabled:
        processing.enable_hires(cfg=hires_cfg)
    for key, value in (req.extras or {}).items():
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
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))

    distilled_cfg = 3.5
    try:
        meta_distilled = metadata.get("distilled_cfg_scale")
        if meta_distilled is not None:
            distilled_cfg = float(meta_distilled)
    except Exception:
        distilled_cfg = 3.5

    image_cfg_scale = None
    try:
        meta_img_cfg = metadata.get("image_cfg_scale")
        if meta_img_cfg is not None:
            image_cfg_scale = float(meta_img_cfg)
    except Exception:
        image_cfg_scale = None

    mask_round = bool(getattr(req, "mask_round", True))
    processing = CodexProcessingImg2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=1,
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
    for key, value in (req.extras or {}).items():
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
