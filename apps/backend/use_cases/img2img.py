"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image-to-image use case orchestration (init image + optional hires pass).
Builds prompt/sampling plans from `CodexProcessingImg2Img`, prepares init-image bundles/latents, runs the sampler loop, and optionally performs a hires second pass.
When smart offload is enabled, keeps the CLIP patcher loaded across cond+uncond so the text encoder is not unloaded/reloaded mid-stage.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_img2img_variant` (function): Decide which img2img variant to run (classic vs Flux Kontext).
- `_build_hires_plan` (function): Builds a `HiResPlan` from the processing config (or returns `None` when disabled).
- `_build_hr_prompt_context` (function): Builds the prompt context used for the hires second pass (supports prompt overrides).
- `_run_hires_pass` (function): Runs the hires second pass by reconditioning and resampling from the base samples.
- `_compute_conditioning_payload` (function): Ensure (cond/uncond) conditioning exists for a prompt context.
- `_generate_kontext_img2img` (function): Flux Kontext img2img implementation (init image as `image_latents`, no denoise schedule).
- `_derive_seeds` (function): Normalizes seed/subseed inputs from processing config.
- `generate_img2img` (function): Canonical img2img implementation; selects the variant and executes sampling.
- `run_img2img` (function): Thin wrapper used by orchestrators to run img2img with an engine + prepared processing object.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import logging
import torch

from apps.backend.core.rng import ImageRNG
from apps.backend.runtime.diagnostics.pipeline_debug import log as pipeline_log
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
from apps.backend.runtime.memory.smart_offload_invariants import (
    enforce_smart_offload_pre_conditioning_residency,
)
from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    img2img_conditioning,
)
from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    GenerationResult,
    HiResPlan,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.models import CodexProcessingImg2Img
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.runtime.workflows.image_init import prepare_init_bundle
from apps.backend.runtime.workflows.image_io import latents_to_pil, pil_to_tensor
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
from PIL import Image

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

logger = logging.getLogger(__name__)

_KONTEXT_MULTIPLE_OF = 16

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


def _resolve_img2img_variant(processing: CodexProcessingImg2Img) -> str:
    engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "")
    return "kontext" if engine_id == "flux1_kontext" else "classic"


def _floor_multiple(value: int, *, multiple_of: int) -> int:
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive")
    if value <= 0:
        raise ValueError("value must be positive")
    floored = (int(value) // multiple_of) * multiple_of
    return max(multiple_of, floored)


def _pick_preferred_kontext_resolution(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid init_image size")
    aspect = float(width) / float(height)
    _, best_w, best_h = min((abs(aspect - w / h), w, h) for w, h in _PREFERRED_KONTEXT_RESOLUTIONS)
    return int(best_w), int(best_h)


def _compute_conditioning_payload(
    processing: CodexProcessingImg2Img,
    prompt_context: PromptContext,
    prompts: Sequence[str],
    conditioning: Any,
    unconditional_conditioning: Any,
) -> ConditioningPayload:
    cond = conditioning
    uncond = unconditional_conditioning

    sd_model = getattr(processing, "sd_model", None)
    if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
        raise RuntimeError("img2img requires processing.sd_model with get_learned_conditioning")

    enforce_smart_offload_pre_conditioning_residency(sd_model, stage="img2img.conditioning")

    uses_distilled_cfg = bool(getattr(sd_model, "use_distilled_cfg_scale", False))

    clip_patcher = None
    clip_loaded_here = False
    needs_conditioning = cond is None or (uncond is None and not uses_distilled_cfg)
    if needs_conditioning and smart_offload_enabled():
        try:
            clip_patcher = sd_model.codex_objects.text_encoders["clip"].patcher
        except Exception:
            clip_patcher = None
        if clip_patcher is not None:
            clip_loaded_here = not memory_management.manager.is_model_loaded(clip_patcher)
            if clip_loaded_here:
                pipeline_log("[img2img.conditioning] smart_offload: loading CLIP patcher for stage")
                memory_management.manager.load_model(clip_patcher)

    try:
        if cond is None:
            texts = list(prompt_context.prompts or [getattr(processing, "prompt", "")])
            if hasattr(sd_model, "_prepare_prompt_wrappers"):
                wrapped = sd_model._prepare_prompt_wrappers(texts, processing, is_negative=False)
                cond = sd_model.get_learned_conditioning(wrapped)
            else:
                cond = sd_model.get_learned_conditioning(texts)
            if cond is None:
                raise RuntimeError("Failed to build conditioning for img2img; get_learned_conditioning returned None.")

        if uncond is None and not uses_distilled_cfg:
            negatives = list(prompt_context.negative_prompts or [getattr(processing, "negative_prompt", "")])
            if hasattr(sd_model, "_prepare_prompt_wrappers"):
                wrapped = sd_model._prepare_prompt_wrappers(negatives, processing, is_negative=True)
                uncond = sd_model.get_learned_conditioning(wrapped)
            else:
                uncond = sd_model.get_learned_conditioning(negatives)
    finally:
        if clip_patcher is not None and clip_loaded_here:
            pipeline_log("[img2img.conditioning] smart_offload: unloading CLIP patcher after stage")
            memory_management.manager.unload_model(clip_patcher)

    return ConditioningPayload(conditioning=cond, unconditional=uncond)


def _generate_kontext_img2img(
    processing: CodexProcessingImg2Img,
    conditioning: Any,
    unconditional_conditioning: Any,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None,
    subseeds: Sequence[int] | None,
    subseed_strength: float | None,
) -> torch.Tensor:
    if getattr(processing, "init_image", None) is None:
        raise ValueError("img2img requires processing.init_image")

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)
    apply_dimension_overrides(processing, prompt_context.controls)

    overrides = getattr(processing, "override_settings", {})
    auto_resize = True
    if isinstance(overrides, dict) and "kontext_auto_resize" in overrides:
        auto_resize = bool(overrides.get("kontext_auto_resize"))

    init_image: Image.Image = processing.init_image.convert("RGB")
    if auto_resize:
        target_width, target_height = _pick_preferred_kontext_resolution(init_image)
    else:
        target_width, target_height = int(processing.width), int(processing.height)

    target_width = _floor_multiple(target_width, multiple_of=_KONTEXT_MULTIPLE_OF)
    target_height = _floor_multiple(target_height, multiple_of=_KONTEXT_MULTIPLE_OF)

    if init_image.size != (target_width, target_height):
        init_image = init_image.resize((target_width, target_height), _RESAMPLE_LANCZOS)
    processing.init_image = init_image
    processing.width = target_width
    processing.height = target_height

    # Kontext does not use denoise strength / init_latent.
    if hasattr(processing, "denoising_strength"):
        try:
            denoise = float(getattr(processing, "denoising_strength", 0.0) or 0.0)
        except Exception:
            denoise = None
        if denoise not in (None, 0.0, 1.0):
            logger.warning("[kontext] denoising_strength is ignored (got=%s)", denoise)

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)
    plan = apply_sampling_overrides(processing, prompt_context.controls, plan)
    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)

    payload = _compute_conditioning_payload(processing, prompt_context, prompts, conditioning, unconditional_conditioning)

    bundle = prepare_init_bundle(processing)
    image_latents = bundle.latents
    if not isinstance(payload.conditioning, dict):
        raise TypeError(
            "kontext requires dict conditioning (crossattn/vector) to pass image_latents; "
            f"got {type(payload.conditioning).__name__}"
        )
    payload.conditioning["image_latents"] = image_latents
    if isinstance(payload.unconditional, dict):
        payload.unconditional["image_latents"] = image_latents

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


def _build_hires_plan(processing: CodexProcessingImg2Img) -> HiResPlan | None:
    if not getattr(processing, "enable_hr", False):
        return None

    hi_cfg = processing.hires
    target_width = hi_cfg.resize_x or int(processing.width * hi_cfg.scale)
    target_height = hi_cfg.resize_y or int(processing.height * hi_cfg.scale)
    steps = hi_cfg.second_pass_steps or processing.steps
    denoise = float(hi_cfg.denoise)
    cfg_scale = hi_cfg.cfg
    return HiResPlan(
        enabled=True,
        target_width=target_width,
        target_height=target_height,
        steps=int(steps),
        denoise=denoise,
        cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
        checkpoint_name=getattr(processing, "hr_checkpoint_name", None),
        additional_modules=getattr(processing, "hr_additional_modules", None),
    )


def _build_hr_prompt_context(
    processing: CodexProcessingImg2Img, base_context: PromptContext
) -> PromptContext:
    hi_cfg = processing.hires
    prompt_seed = hi_cfg.prompt if hi_cfg.prompt else base_context.prompts
    if isinstance(prompt_seed, str):
        hr_prompts_source = [prompt_seed]
    else:
        hr_prompts_source = list(prompt_seed)
    if not hr_prompts_source:
        hr_prompts_source = list(base_context.prompts)
    hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(
        hr_prompts_source
    )
    negative = (
        [hi_cfg.negative_prompt]
        if hi_cfg.negative_prompt
        else list(base_context.negative_prompts)
    )
    return PromptContext(
        prompts=hr_cleaned_prompts,
        negative_prompts=negative,
        loras=hr_prompt_loras,
        controls=dict(hr_prompt_controls),
    )


def _run_hires_pass(
    processing: CodexProcessingImg2Img,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    base_samples: torch.Tensor,
    base_context: PromptContext,
) -> torch.Tensor:
    hi_cfg = processing.hires
    target_width = hi_cfg.resize_x or int(processing.width * hi_cfg.scale)
    target_height = hi_cfg.resize_y or int(processing.height * hi_cfg.scale)
    steps = hi_cfg.second_pass_steps or processing.steps
    denoise = float(hi_cfg.denoise)

    original = {
        "prompts": processing.prompts,
        "negative_prompts": getattr(processing, "negative_prompts", []),
        "width": processing.width,
        "height": processing.height,
        "guidance_scale": processing.guidance_scale,
        "steps": processing.steps,
        "denoising_strength": getattr(processing, "denoising_strength", 0.75),
    }

    hi_prompt_context = _build_hr_prompt_context(processing, base_context)

    processing.prompts = hi_prompt_context.prompts
    processing.negative_prompts = hi_prompt_context.negative_prompts
    processing.width = target_width
    processing.height = target_height
    processing.guidance_scale = float(hi_cfg.cfg or processing.guidance_scale)
    processing.cfg_scale = processing.guidance_scale
    processing.steps = int(steps)
    processing.denoising_strength = denoise
    processing.prepare_prompt_data()

    decoded = decode_latent_batch(processing.sd_model, base_samples)
    pil_images = latents_to_pil(decoded)
    upscaled = [img.resize((target_width, target_height), _RESAMPLE_LANCZOS) for img in pil_images]
    tensor = pil_to_tensor(upscaled)
    latents = processing.sd_model.encode_first_stage(tensor)
    image_conditioning = img2img_conditioning(
        processing.sd_model,
        tensor,
        latents,
        image_mask=getattr(processing, "image_mask", None),
        round_mask=getattr(processing, "round_image_mask", True),
    )

    hires_settings = plan.noise_settings
    rng = ImageRNG(
        (latents.shape[1], latents.shape[2], latents.shape[3]),
        plan.seeds,
        subseeds=plan.subseeds,
        subseed_strength=plan.subseed_strength,
        seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
        settings=hires_settings,
    )
    noise = rng.next().to(latents)

    start_index = max(0, min(int(round(denoise * processing.steps)), processing.steps - 1))

    hr_plan = replace(
        plan,
        steps=int(processing.steps),
        guidance_scale=float(processing.guidance_scale),
    )

    samples = execute_sampling(
        processing,
        hr_plan,
        payload,
        hi_prompt_context,
        hi_prompt_context.loras,
        hi_prompt_context.controls,
        rng=rng,
        noise=noise,
        image_conditioning=image_conditioning,
        init_latent=latents,
        start_at_step=start_index,
    )

    processing.prompts = original["prompts"]
    processing.negative_prompts = original["negative_prompts"]
    processing.width = original["width"]
    processing.height = original["height"]
    processing.guidance_scale = original["guidance_scale"]
    processing.cfg_scale = processing.guidance_scale
    processing.steps = original["steps"]
    processing.denoising_strength = original["denoising_strength"]
    processing.prepare_prompt_data()

    return samples


def _derive_seeds(processing: CodexProcessingImg2Img) -> tuple[list[int], list[int], float]:
    seeds = list(getattr(processing, "seeds", []) or [])
    if not seeds:
        seeds = [int(getattr(processing, "seed", -1) or -1)]
    subseeds = list(getattr(processing, "subseeds", []) or [])
    strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)
    return seeds, subseeds, strength


def generate_img2img(
    processing,
    conditioning,
    unconditional_conditioning,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None = None,
    subseeds: Sequence[int] | None = None,
    subseed_strength: float | None = None,
) -> torch.Tensor:
    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("generate_img2img expects CodexProcessingImg2Img")

    if _resolve_img2img_variant(processing) == "kontext":
        return _generate_kontext_img2img(
            processing,
            conditioning,
            unconditional_conditioning,
            prompts,
            seeds=seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
        )

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)
    apply_dimension_overrides(processing, prompt_context.controls)

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)
    plan = apply_sampling_overrides(processing, prompt_context.controls, plan)

    if "denoise" in prompt_context.controls:
        processing.denoising_strength = float(prompt_context.controls["denoise"])

    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)

    payload = _compute_conditioning_payload(
        processing,
        prompt_context,
        prompts,
        conditioning,
        unconditional_conditioning,
    )

    bundle = prepare_init_bundle(processing)
    processing.init_latent = bundle.latents

    image_conditioning = img2img_conditioning(
        processing.sd_model,
        bundle.tensor,
        bundle.latents,
        image_mask=bundle.mask,
        round_mask=getattr(processing, "round_image_mask", True),
    )
    processing.image_conditioning = image_conditioning

    noise = rng.next().to(bundle.latents)
    start_step = max(
        0,
        min(
            int(round(float(getattr(processing, "denoising_strength", 0.5) or 0.5) * plan.steps)),
            plan.steps - 1,
        ),
    )

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
            noise=noise,
            image_conditioning=image_conditioning,
            init_latent=bundle.latents,
            start_at_step=start_step,
        )
    finally:
        finalize_tiling(tiling_applied, old_tiled)

    hires_plan = _build_hires_plan(processing)
    if hires_plan is None:
        result = GenerationResult(samples=samples, decoded=None)
        return result.samples

    hires_samples = _run_hires_pass(
        processing,
        plan,
        payload,
        samples,
        prompt_context,
    )

    result = GenerationResult(samples=hires_samples, decoded=None)
    return result.samples


def run_img2img(
    *,
    engine,
    processing: CodexProcessingImg2Img,
    conditioning: Any,
    unconditional_conditioning: Any,
    prompts: Sequence[str],
) -> Any:
    return generate_img2img(
        processing,
        conditioning,
        unconditional_conditioning,
        prompts,
    )
