"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux Kontext img2img use case (init image as conditioning tokens).
Implements the Kontext img2img flow where the init image becomes conditioning (`image_latents`) and sampling starts from pure noise (no
denoise-strength schedule), following the diffusers FluxKontext pipeline semantics.

Symbols (top-level; keep in sync; no ghosts):
- `_floor_multiple` (function): Floors a value to a positive multiple (used for resolution constraints).
- `_pick_preferred_resolution` (function): Picks a recommended Kontext resolution based on init image aspect ratio.
- `_compute_conditioning` (function): Builds conditional/unconditional conditioning using the engine's TE hooks.
- `generate_kontext_img2img` (function): Runs Kontext img2img sampling and returns latent samples.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from apps.backend.runtime.processing.datatypes import ConditioningPayload
from apps.backend.runtime.processing.models import CodexProcessingImg2Img
from apps.backend.runtime.workflows.image_init import prepare_init_bundle
from apps.backend.runtime.workflows.prompt_context import (
    apply_dimension_overrides,
    apply_prompt_context,
    build_prompt_context,
)
from apps.backend.runtime.workflows.sampling_execute import execute_sampling
from apps.backend.runtime.workflows.sampling_plan import (
    apply_sampling_overrides,
    build_sampling_plan,
    ensure_sampler_and_rng,
)
from apps.backend.runtime.workflows.scripts import run_process_scripts
from apps.backend.runtime.workflows.tiling import apply_tiling_if_requested, finalize_tiling

logger = logging.getLogger("backend.use_cases.kontext_img2img")

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

_MULTIPLE_OF = 16

# Recommended resolutions from upstream diffusers FluxKontextPipeline.
_PREFERRED_KONTEXT_RESOLUTIONS: list[tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def _floor_multiple(value: int, *, multiple_of: int) -> int:
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive")
    if value <= 0:
        raise ValueError("value must be positive")
    floored = (int(value) // multiple_of) * multiple_of
    return max(multiple_of, floored)


def _pick_preferred_resolution(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid init_image size")
    aspect = float(width) / float(height)
    _, best_w, best_h = min((abs(aspect - w / h), w, h) for w, h in _PREFERRED_KONTEXT_RESOLUTIONS)
    return int(best_w), int(best_h)


def _compute_conditioning(processing: CodexProcessingImg2Img, prompt_context) -> tuple[Any, Any]:
    sd_model = getattr(processing, "sd_model", None)
    if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
        raise RuntimeError("kontext img2img requires processing.sd_model.get_learned_conditioning")

    prompts = list(prompt_context.prompts or [getattr(processing, "prompt", "")])
    negative_prompts = list(prompt_context.negative_prompts or [getattr(processing, "negative_prompt", "")])

    uses_distilled_cfg = bool(getattr(sd_model, "use_distilled_cfg_scale", False))

    if hasattr(sd_model, "_prepare_prompt_wrappers"):
        wrapped = sd_model._prepare_prompt_wrappers(prompts, processing, is_negative=False)
        cond = sd_model.get_learned_conditioning(wrapped)
        if uses_distilled_cfg:
            uncond = None
        else:
            negative_wrapped = sd_model._prepare_prompt_wrappers(negative_prompts, processing, is_negative=True)
            uncond = sd_model.get_learned_conditioning(negative_wrapped)
    else:
        cond = sd_model.get_learned_conditioning(prompts)
        uncond = None if uses_distilled_cfg else sd_model.get_learned_conditioning(negative_prompts)

    if cond is None:
        raise RuntimeError("Failed to build conditioning for kontext img2img; got None.")

    return cond, uncond


def generate_kontext_img2img(
    processing: Any,
    conditioning: Any,
    unconditional_conditioning: Any,
    prompts: Sequence[str],
) -> torch.Tensor:
    """Run Flux Kontext img2img.

    Unlike classic img2img, Kontext treats the init image as conditioning tokens
    (`image_latents`) and starts sampling from pure noise (no denoise strength).
    """
    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("generate_kontext_img2img expects CodexProcessingImg2Img")

    if getattr(processing, "init_image", None) is None:
        raise ValueError("kontext img2img requires processing.init_image")

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)
    apply_dimension_overrides(processing, prompt_context.controls)

    overrides = getattr(processing, "override_settings", {})
    auto_resize = True
    if isinstance(overrides, dict) and "kontext_auto_resize" in overrides:
        auto_resize = bool(overrides.get("kontext_auto_resize"))

    init_image: Image.Image = processing.init_image.convert("RGB")
    if auto_resize:
        target_width, target_height = _pick_preferred_resolution(init_image)
    else:
        target_width, target_height = int(processing.width), int(processing.height)
    target_width = _floor_multiple(target_width, multiple_of=_MULTIPLE_OF)
    target_height = _floor_multiple(target_height, multiple_of=_MULTIPLE_OF)

    if init_image.size != (target_width, target_height):
        init_image = init_image.resize((target_width, target_height), _RESAMPLE_LANCZOS)
    processing.init_image = init_image
    processing.width = target_width
    processing.height = target_height

    # Context knobs: kontext does not use denoise strength / init_latent.
    if hasattr(processing, "denoising_strength"):
        try:
            denoise = float(getattr(processing, "denoising_strength", 0.0) or 0.0)
        except Exception:
            denoise = None
        if denoise not in (None, 0.0, 1.0):
            logger.warning("[kontext] denoising_strength is ignored (got=%s)", denoise)

    seeds = list(getattr(processing, "seeds", []) or [])
    if not seeds:
        seeds = [int(getattr(processing, "seed", -1) or -1)]
    subseeds = list(getattr(processing, "subseeds", []) or [])
    if not subseeds:
        subseeds = [int(getattr(processing, "subseed", -1) or -1)]
    subseed_strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)

    plan = build_sampling_plan(processing, seeds, subseeds, subseed_strength)
    plan = apply_sampling_overrides(processing, prompt_context.controls, plan)
    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)

    cond = conditioning
    uncond = unconditional_conditioning
    if cond is None:
        cond, uncond = _compute_conditioning(processing, prompt_context)

    # Encode init image latents and attach as conditioning tokens.
    bundle = prepare_init_bundle(processing)
    image_latents = bundle.latents
    if not isinstance(cond, dict):
        raise TypeError(
            "kontext requires dict conditioning (crossattn/vector) to pass image_latents; "
            f"got {type(cond).__name__}"
        )
    cond["image_latents"] = image_latents
    if isinstance(uncond, dict):
        uncond["image_latents"] = image_latents

    payload = ConditioningPayload(conditioning=cond, unconditional=uncond)

    tiling_applied, old_tiled = apply_tiling_if_requested(processing, prompt_context.controls)
    try:
        samples = execute_sampling(
            processing,
            plan,
            payload,
            prompt_context,
            prompt_context.loras,
            prompt_context.controls,
            rng=rng,
            init_latent=None,
            start_at_step=0,
        )
    finally:
        finalize_tiling(tiling_applied, old_tiled)

    return samples


__all__ = ["generate_kontext_img2img"]
