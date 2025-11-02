from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import torch

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.runtime import memory_management
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
from apps.backend.runtime.workflows import (
    apply_dimension_overrides,
    apply_prompt_context,
    apply_sampling_overrides,
    apply_tiling_if_requested,
    build_prompt_context,
    build_sampling_plan,
    ensure_sampler_and_rng,
    execute_sampling,
    finalize_tiling,
    latents_to_pil,
    pil_to_tensor,
    run_process_scripts,
    prepare_init_bundle,
)
from PIL import Image

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


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

    decoded = decode_latent_batch(processing.sd_model, base_samples, target_device=memory_management.cpu)
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

    payload = ConditioningPayload(conditioning=conditioning, unconditional=unconditional_conditioning)

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
