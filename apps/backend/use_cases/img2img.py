"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image-to-image use case orchestration and canonical streaming wrapper (init image + optional hires pass).
Builds prompt/sampling plans from `CodexProcessingImg2Img`, prepares init-image bundles/latents, runs the sampler loop, and optionally performs a hires second pass.
The hires pass init is prepared via the global hires-fix stage (`apps/backend/runtime/pipeline_stages/hires_fix.py`).
When configured, the hires second pass applies sampler/scheduler overrides (validated) by deriving a dedicated `SamplingPlan` for the hires pass.
When smart offload is enabled, keeps required text-encoder patchers loaded across cond+uncond and unloads them after conditioning.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_img2img_variant` (function): Decide which img2img variant to run (classic vs Flux Kontext).
- `_build_hires_plan` (function): Builds a `HiResPlan` from the processing config (or returns `None` when disabled).
- `_build_hr_prompt_context` (function): Builds the prompt context used for the hires second pass (supports prompt overrides).
- `_run_hires_pass` (function): Runs the hires second pass by reconditioning and resampling from the base samples (init prepared via global hires-fix stage).
- `_compute_conditioning_payload` (function): Ensure (cond/uncond) conditioning exists for a prompt context.
- `_generate_kontext_img2img` (function): Flux Kontext img2img implementation (init image as `image_latents`, no denoise schedule).
- `_derive_seeds` (function): Normalizes seed/subseed inputs from processing config.
- `generate_img2img` (function): Canonical img2img implementation; selects the variant and executes sampling.
- `run_img2img` (function): Canonical img2img mode wrapper (progress polling + decode + result events) used by engines/orchestrator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterator, Sequence

import logging
import torch

from apps.backend.core.rng import ImageRNG
from apps.backend.runtime.diagnostics.pipeline_debug import log as pipeline_log
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
from apps.backend.runtime.memory.smart_offload_invariants import (
    enforce_smart_offload_pre_conditioning_residency,
    enforce_smart_offload_text_encoders_off,
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
from apps.backend.runtime.pipeline_stages.image_init import prepare_init_bundle
from apps.backend.runtime.pipeline_stages.image_io import latents_to_pil, pil_to_tensor
from apps.backend.runtime.pipeline_stages.hires_fix import (
    prepare_hires_latents_and_conditioning,
    start_at_step_from_denoise,
)
from apps.backend.runtime.pipeline_stages.masked_img2img import (
    MASK_ENFORCEMENT_PER_STEP_CLAMP,
    MASK_ENFORCEMENT_POST_BLEND,
    apply_inpaint_full_res_composite,
    prepare_masked_img2img_bundle,
)
from apps.backend.runtime.pipeline_stages.prompt_context import (
    apply_dimension_overrides,
    apply_prompt_context,
    build_prompt_context,
)
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling
from apps.backend.runtime.pipeline_stages.sampling_plan import (
    apply_sampling_overrides,
    build_sampling_plan,
    ensure_sampler_and_rng,
    resolve_sampler_scheduler_override,
)
from apps.backend.runtime.pipeline_stages.scripts import run_process_scripts
from apps.backend.runtime.pipeline_stages.tiling import apply_tiling_if_requested, finalize_tiling
from apps.backend.runtime.sampling.driver import CodexSampler
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

    text_encoder_patchers: list[tuple[str, object]] = []
    needs_conditioning = cond is None or (uncond is None and not uses_distilled_cfg)
    if needs_conditioning and smart_offload_enabled():
        codex_objects = getattr(sd_model, "codex_objects", None)
        text_encoders = getattr(codex_objects, "text_encoders", None) if codex_objects is not None else None
        if isinstance(text_encoders, dict):
            for name, entry in text_encoders.items():
                if entry is None:
                    continue
                patcher = getattr(entry, "patcher", None)
                patcher_obj = patcher if patcher is not None else entry
                text_encoder_patchers.append((str(name), patcher_obj))
            for name, patcher in text_encoder_patchers:
                if memory_management.manager.is_model_loaded(patcher):
                    continue
                pipeline_log(f"[img2img.conditioning] smart_offload: loading text encoder '{name}' patcher for stage")
                memory_management.manager.load_model(patcher)

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
        if smart_offload_enabled():
            if text_encoder_patchers:
                pipeline_log("[img2img.conditioning] smart_offload: unloading text encoders after stage")
            enforce_smart_offload_text_encoders_off(sd_model, stage="img2img.conditioning(post)")

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

    bundle = prepare_init_bundle(processing)
    image_latents = bundle.latents
    payload = _compute_conditioning_payload(processing, prompt_context, prompts, conditioning, unconditional_conditioning)
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
    raw_upscaler = getattr(hi_cfg, "upscaler", None)
    if not isinstance(raw_upscaler, str) or not raw_upscaler.strip():
        raise ValueError(
            "Hires is enabled but 'hires.upscaler' is missing. "
            "Choose an upscaler id from GET /api/upscalers (e.g. 'latent:bicubic-aa')."
        )
    upscaler_id = raw_upscaler.strip()
    from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES

    if upscaler_id not in LATENT_UPSCALE_MODES and not upscaler_id.startswith("spandrel:"):
        raise ValueError(
            f"Invalid 'hires.upscaler': {upscaler_id!r}. "
            "Expected a 'latent:*' or 'spandrel:*' upscaler id from GET /api/upscalers."
        )
    if hi_cfg.resize_x is not None:
        target_width = int(hi_cfg.resize_x)
        if target_width <= 0:
            raise ValueError("Hires is enabled but 'hires.resize_x' must be > 0 when provided.")
    else:
        if hi_cfg.scale is None:
            raise ValueError(
                "Hires is enabled but neither 'hires.resize_x' nor 'hires.scale' is set. "
                "Provide explicit dimensions or a scale."
            )
        target_width = int(processing.width * hi_cfg.scale)
    if hi_cfg.resize_y is not None:
        target_height = int(hi_cfg.resize_y)
        if target_height <= 0:
            raise ValueError("Hires is enabled but 'hires.resize_y' must be > 0 when provided.")
    else:
        if hi_cfg.scale is None:
            raise ValueError(
                "Hires is enabled but neither 'hires.resize_y' nor 'hires.scale' is set. "
                "Provide explicit dimensions or a scale."
            )
        target_height = int(processing.height * hi_cfg.scale)
    if hi_cfg.second_pass_steps is not None:
        steps = int(hi_cfg.second_pass_steps)
        if steps <= 0:
            raise ValueError("Hires is enabled but 'hires.steps' must be > 0 when provided.")
    else:
        steps = int(processing.steps)
    denoise = float(hi_cfg.denoise)
    cfg_scale = hi_cfg.cfg
    return HiResPlan(
        enabled=True,
        target_width=target_width,
        target_height=target_height,
        upscaler_id=upscaler_id,
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
    hires_plan: HiResPlan,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    base_samples: torch.Tensor,
    base_context: PromptContext,
) -> torch.Tensor:
    hi_cfg = processing.hires
    target_width = int(hires_plan.target_width)
    target_height = int(hires_plan.target_height)
    steps = int(hires_plan.steps)
    denoise = float(hires_plan.denoise)

    original = {
        "prompts": processing.prompts,
        "negative_prompts": getattr(processing, "negative_prompts", []),
        "width": processing.width,
        "height": processing.height,
        "guidance_scale": processing.guidance_scale,
        "steps": processing.steps,
        "denoising_strength": getattr(processing, "denoising_strength", 0.75),
        "sampler_name": getattr(processing, "sampler_name", None),
        "scheduler": getattr(processing, "scheduler", None),
        "sampler": getattr(processing, "sampler", None),
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
    hires_sampler, hires_scheduler = resolve_sampler_scheduler_override(
        base_sampler=str(plan.sampler_name or ""),
        base_scheduler=str(plan.scheduler_name or ""),
        sampler_override=getattr(hi_cfg, "sampler_name", None),
        scheduler_override=getattr(hi_cfg, "scheduler", None),
    )
    processing.sampler_name = hires_sampler
    processing.scheduler = hires_scheduler
    processing.sampler = CodexSampler(processing.sd_model, algorithm=hires_sampler)
    processing.prepare_prompt_data()

    latents, image_conditioning = prepare_hires_latents_and_conditioning(
        processing,
        base_samples=base_samples,
        base_decoded=None,
        hires_plan=hires_plan,
        tile=getattr(hi_cfg, "tile", None),
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

    start_index = start_at_step_from_denoise(denoise=denoise, steps=int(processing.steps))

    hr_plan = replace(
        plan,
        sampler_name=hires_sampler,
        scheduler_name=hires_scheduler,
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
    processing.sampler_name = original["sampler_name"]
    processing.scheduler = original["scheduler"]
    processing.sampler = original["sampler"]
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
) -> GenerationResult:
    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("generate_img2img expects CodexProcessingImg2Img")

    if _resolve_img2img_variant(processing) == "kontext":
        if processing.has_mask():
            raise NotImplementedError("masking is not supported for flux1_kontext img2img yet")
        samples = _generate_kontext_img2img(
            processing,
            conditioning,
            unconditional_conditioning,
            prompts,
            seeds=seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
        )
        return GenerationResult(samples=samples, decoded=None)

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

    post_step_hook = None
    post_sample_hook = None
    full_res_plan = None
    if processing.has_mask():
        if bool(getattr(getattr(processing, "hires", None), "enabled", False)):
            raise NotImplementedError("HiRes is not supported for masked img2img yet")
        enforcement = getattr(processing, "mask_enforcement", None)
        masked_bundle, enforcer = prepare_masked_img2img_bundle(
            processing,
            plan,
            enforce_mode=enforcement,
        )
        processing.init_latent = masked_bundle.init_latent
        processing.image_conditioning = masked_bundle.image_conditioning
        full_res_plan = masked_bundle.full_res
        enforcement_value = str(enforcement).strip()
        if enforcement_value == MASK_ENFORCEMENT_PER_STEP_CLAMP:
            post_step_hook = enforcer.post_step
            post_sample_hook = enforcer.post_sample
        elif enforcement_value == MASK_ENFORCEMENT_POST_BLEND:
            post_sample_hook = enforcer.post_sample
        else:
            raise ValueError(
                f"Unknown mask enforcement '{enforcement_value}' (internal validation bug)"
            )
        init_latent = masked_bundle.init_latent
        image_conditioning = masked_bundle.image_conditioning
    else:
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
        init_latent = bundle.latents

    payload = _compute_conditioning_payload(
        processing,
        prompt_context,
        prompts,
        conditioning,
        unconditional_conditioning,
    )

    noise = rng.next().to(init_latent)
    start_step = start_at_step_from_denoise(
        denoise=float(getattr(processing, "denoising_strength", 0.5) or 0.5),
        steps=int(plan.steps),
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
            init_latent=init_latent,
            start_at_step=start_step,
            post_step_hook=post_step_hook,
            post_sample_hook=post_sample_hook,
        )
    finally:
        finalize_tiling(tiling_applied, old_tiled)

    if full_res_plan is not None:
        decoded = decode_latent_batch(processing.sd_model, samples)
        images = latents_to_pil(decoded)
        composited = apply_inpaint_full_res_composite(images, plan=full_res_plan)
        return GenerationResult(samples=samples, decoded=composited)

    hires_plan = _build_hires_plan(processing)
    if hires_plan is None:
        return GenerationResult(samples=samples, decoded=None)

    hires_samples = _run_hires_pass(
        processing,
        hires_plan,
        plan,
        payload,
        samples,
        prompt_context,
    )

    return GenerationResult(samples=hires_samples, decoded=None)


def run_img2img(
    *,
    engine: Any,
    request: Any,
) -> Iterator["InferenceEvent"]:
    """Run img2img as a canonical event stream.

    This wrapper owns the mode-level concerns (seed defaults, progress polling, decode + result packaging).
    Engines should delegate here rather than implementing per-mode pipelines.
    """

    import json

    from apps.backend.core.requests import Img2ImgRequest, ProgressEvent, ResultEvent
    from apps.backend.engines.util.adapters import build_img2img_processing
    from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides
    from apps.backend.runtime.text_processing import last_extra_generation_params

    from ._image_streaming import (
        _build_common_info,
        _decode_generation_output,
        _iter_sampling_progress,
        _resolve_seed_plan,
        _run_inference_worker,
    )

    if not isinstance(request, Img2ImgRequest):
        raise TypeError("run_img2img expects Img2ImgRequest")

    engine.ensure_loaded()

    proc = build_img2img_processing(request)
    proc.sd_model = engine

    base_seed, seeds, subseeds, subseed_strength = _resolve_seed_plan(
        seed=getattr(request, "seed", None),
        batch_total=proc.batch_total,
    )
    proc.seed = base_seed
    proc.seeds = list(seeds)
    proc.subseed = -1
    proc.subseeds = list(subseeds)

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]

    def _generate() -> GenerationResult:
        with smart_runtime_overrides(
            smart_offload=bool(getattr(proc, "smart_offload", False)),
            smart_fallback=bool(getattr(proc, "smart_fallback", False)),
            smart_cache=bool(getattr(proc, "smart_cache", False)),
        ):
            return generate_img2img(
                proc,
                conditioning=None,
                unconditional_conditioning=None,
                prompts=prompts,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
            )

    done, outcome = _run_inference_worker(name=f"{engine.engine_id}-img2img-worker", fn=_generate)

    for step, total, eta in _iter_sampling_progress(done=done):
        pct = max(5.0, min(99.0, (step / total) * 100.0))
        yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)

    if outcome.error is not None:
        raise outcome.error

    images, decode_ms = _decode_generation_output(engine=engine, output=outcome.output, task_label="img2img")

    all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
    seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

    extra_params: dict[str, object] = {}
    try:
        extra_params.update(last_extra_generation_params)
        extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
    except Exception:  # noqa: BLE001
        extra_params = getattr(proc, "extra_generation_params", {}) or {}

    timings: dict[str, float] = {"decode_ms": float(decode_ms)}
    try:
        if outcome.sampling_start is not None and outcome.sampling_end is not None:
            timings["sampling_ms"] = max(0.0, (outcome.sampling_end - outcome.sampling_start) * 1000.0)
    except Exception:  # noqa: BLE001
        pass

    mode_info: dict[str, object] = {
        "denoise_strength": float(getattr(proc, "denoising_strength", 0.0) or 0.0),
    }
    if bool(getattr(getattr(proc, "hires", None), "enabled", False)):
        try:
            mode_info["hires"] = getattr(proc, "hires", None).as_dict()
        except Exception:  # noqa: BLE001
            pass

    info = _build_common_info(
        engine_id=engine.engine_id,
        task="img2img",
        proc=proc,
        seed=seed_value,
        all_seeds=all_seeds,
        extra_params=extra_params,
        timings_ms=timings,
        mode_info=mode_info,
    )

    post_cleanup = getattr(engine, "_post_txt2img_cleanup", None)
    if callable(post_cleanup):
        post_cleanup()

    yield ResultEvent(payload={"images": images, "info": json.dumps(info)})
