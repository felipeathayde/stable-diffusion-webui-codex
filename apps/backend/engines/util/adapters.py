from __future__ import annotations

from typing import Any, Mapping, Sequence
import logging

from apps.backend.core.requests import Img2ImgRequest, Txt2ImgRequest
from apps.backend.runtime.processing.models import (
    CodexHighResConfig,
    CodexProcessingImg2Img,
    CodexProcessingTxt2Img,
)

_log = logging.getLogger(__name__)


def _build_hires_config(data: Mapping[str, Any] | None, *, default_cfg: float, default_distilled: float, default_denoise: float) -> CodexHighResConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    return CodexHighResConfig(
        enabled=enabled,
        scale=float(payload.get("scale", 2.0)) if enabled else 1.0,
        denoise=float(payload.get("denoise", default_denoise)) if enabled else 0.0,
        upscaler=payload.get("upscaler") if enabled else None,
        second_pass_steps=int(payload.get("steps", 0)) if enabled else 0,
        resize_x=int(payload.get("resize_x", 0)) if enabled else 0,
        resize_y=int(payload.get("resize_y", 0)) if enabled else 0,
        prompt=str(payload.get("hr_prompt", "")) if enabled else "",
        negative_prompt=str(payload.get("hr_negative_prompt", "")) if enabled else "",
        cfg=float(payload.get("hr_cfg", default_cfg)) if enabled else default_cfg,
        distilled_cfg=float(payload.get("hr_distilled_cfg", default_distilled)) if enabled else default_distilled,
        sampler_name=payload.get("hr_sampler_name") if enabled else None,
        scheduler=payload.get("hr_scheduler") if enabled else None,
        additional_modules=tuple(payload.get("hr_additional_modules", ())) if enabled else tuple(),
        checkpoint_name=payload.get("hr_checkpoint_name") if enabled else None,
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
        bool(req.highres_fix),
    )
    hires_cfg = _build_hires_config(
        req.highres_fix if isinstance(req.highres_fix, dict) else {},
        default_cfg=req.guidance_scale or 7.0,
        default_distilled=3.5,
        default_denoise=0.5,
    )
    processing = CodexProcessingTxt2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=1,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=3.5,
        width=req.width,
        height=req.height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        metadata=dict(req.metadata or {}),
    )
    if hires_cfg.enabled:
        processing.enable_hires(cfg=hires_cfg)
    for key, value in (req.extras or {}).items():
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

    processing = CodexProcessingImg2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=1,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=3.5,
        width=width,
        height=height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        init_image=req.init_image,
        mask=req.mask,
        denoising_strength=float(req.denoise_strength),
        metadata=dict(req.metadata or {}),
    )
    if req.highres_fix:
        hires_cfg = _build_hires_config(
            req.highres_fix,
            default_cfg=processing.guidance_scale,
            default_distilled=processing.distilled_guidance_scale,
            default_denoise=processing.denoising_strength,
        )
        if hires_cfg.enabled:
            processing.enable_hires(hires_cfg)
    for key, value in (req.extras or {}).items():
        processing.update_override(key, value)
        if key == "eta_noise_seed_delta":
            try:
                processing.eta_noise_seed_delta = int(value)
            except Exception:
                processing.eta_noise_seed_delta = value
    return processing
