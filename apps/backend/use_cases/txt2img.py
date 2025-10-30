from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import os

import numpy as np
import torch
from PIL import Image

from apps.backend.codex import main as codex_main
from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.patchers.token_merging import SkipWritingToConfig
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
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
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
    maybe_decode_for_hr,
    pil_to_tensor,
    run_process_scripts,
)
from loader import EngineLoadOptions, load_engine as _load_engine

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _prepare_first_pass_from_image(
    processing: CodexProcessingTxt2Img,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image = processing.firstpass_image
    if image is None or not processing.enable_hr:
        return None, None

    if processing.latent_scale_mode is None:
        array = np.array(image).astype(np.float32) / 255.0
        array = array * 2.0 - 1.0
        array = np.moveaxis(array, 2, 0)
        decoded_samples = torch.from_numpy(np.expand_dims(array, 0))
        return None, decoded_samples

    array = np.array(image).astype(np.float32) / 255.0
    array = np.moveaxis(array, 2, 0)
    tensor = torch.from_numpy(np.expand_dims(array, axis=0))
    tensor = tensor.to(devices.default_device(), dtype=torch.float32)

    samples = processing.sd_model.encode_first_stage(tensor)
    devices.torch_gc()
    return samples, None


def _reload_for_hires(processing: CodexProcessingTxt2Img) -> None:
    with SkipWritingToConfig():
        from apps.backend.codex import main as _codex

        checkpoint_before = getattr(_codex, "_SELECTIONS").checkpoint_name
        modules_before = list(getattr(_codex, "_SELECTIONS").additional_modules)

        reload_required = False
        if (
            getattr(processing, "hr_additional_modules", None) is not None
            and "Use same choices" not in processing.hr_additional_modules
        ):
            modules_changed = codex_main.modules_change(
                processing.hr_additional_modules, save=False, refresh=False
            )
            reload_required = reload_required or modules_changed

        if (
            processing.hr_checkpoint_name
            and processing.hr_checkpoint_name != "Use same checkpoint"
        ):
            checkpoint_changed = codex_main.checkpoint_change(
                processing.hr_checkpoint_name, save=False, refresh=False
            )
            if checkpoint_changed:
                processing.firstpass_use_distilled_cfg_scale = (
                    processing.sd_model.use_distilled_cfg_scale
                )
                reload_required = True

        if reload_required:
            try:
                codex_main.refresh_model_loading_parameters()
                load_opts = EngineLoadOptions(
                    device=None,
                    dtype=None,
                    attention_backend=os.getenv("CODEX_ATTENTION_BACKEND"),
                    accelerator=os.getenv("CODEX_ACCELERATOR"),
                    vae_path=None,
                )
                new_engine = _load_engine(processing.hr_checkpoint_name, options=load_opts)
                processing.sd_model = new_engine
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load hires checkpoint '{processing.hr_checkpoint_name}': {exc}"
                ) from exc
            finally:
                codex_main.modules_change(modules_before, save=False, refresh=False)
                codex_main.checkpoint_change(checkpoint_before, save=False, refresh=False)
                codex_main.refresh_model_loading_parameters()

        if processing.sd_model.use_distilled_cfg_scale:
            processing.extra_generation_params["Hires Distilled CFG Scale"] = (
                processing.hr_distilled_cfg
            )


def _build_hires_plan(processing: CodexProcessingTxt2Img) -> HiResPlan | None:
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


def _run_hires_pass(
    processing: CodexProcessingTxt2Img,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    base_samples: torch.Tensor,
    decoded_samples: torch.Tensor | None,
    prompt_context: PromptContext,
) -> torch.Tensor:
    hi_cfg = processing.hires
    processing.ensure_hires_prompts()

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
    }

    hr_prompts_source = processing.hr_prompts or processing.all_prompts or prompt_context.prompts
    hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(
        list(hr_prompts_source)
    )
    hr_negative = processing.hr_negative_prompts or original["negative_prompts"]
    hi_prompt_context = PromptContext(
        prompts=hr_cleaned_prompts,
        negative_prompts=hr_negative,
        loras=hr_prompt_loras,
        controls=dict(hr_prompt_controls),
    )

    processing.prompts = hi_prompt_context.prompts
    processing.negative_prompts = hi_prompt_context.negative_prompts
    processing.width = target_width
    processing.height = target_height
    processing.guidance_scale = float(hi_cfg.cfg or processing.guidance_scale)
    processing.cfg_scale = processing.guidance_scale
    processing.steps = int(steps)
    processing.prepare_prompt_data()

    if getattr(processing, "latent_scale_mode", None) is not None:
        mode = processing.latent_scale_mode.get("mode", "bilinear")
        antialias = bool(processing.latent_scale_mode.get("antialias", False))
        latents = torch.nn.functional.interpolate(
            base_samples,
            size=(target_height // 8, target_width // 8),
            mode=mode,
            antialias=antialias,
        )
        tensor = decode_latent_batch(processing.sd_model, latents, target_device=devices.cpu())
        image_conditioning = img2img_conditioning(
            processing.sd_model,
            tensor,
            latents,
            image_mask=getattr(processing, "image_mask", None),
            round_mask=getattr(processing, "round_image_mask", True),
        )
    else:
        if decoded_samples is None:
            decoded_samples = decode_latent_batch(
                processing.sd_model, base_samples, target_device=devices.cpu()
            )
        pil_images = latents_to_pil(decoded_samples)
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
    rng_hr = ImageRNG(
        (latents.shape[1], latents.shape[2], latents.shape[3]),
        plan.seeds,
        subseeds=plan.subseeds,
        subseed_strength=plan.subseed_strength,
        seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
        settings=hires_settings,
    )
    noise = rng_hr.next().to(latents)

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
        rng=rng_hr,
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
    processing.prepare_prompt_data()

    return samples


def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
):
    if not isinstance(processing, CodexProcessingTxt2Img):
        raise TypeError("generate_txt2img expects CodexProcessingTxt2Img")

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)
    apply_dimension_overrides(processing, prompt_context.controls)

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

    payload = ConditioningPayload(conditioning=conditioning, unconditional=unconditional_conditioning)

    init_latents, init_decoded = _prepare_first_pass_from_image(processing)
    base_samples = init_latents
    decoded_samples = init_decoded

    hires_plan = _build_hires_plan(processing)

    if base_samples is None and decoded_samples is None:
        tiling_applied, old_tiled = apply_tiling_if_requested(processing, prompt_context.controls)
        try:
            base_samples = execute_sampling(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                prompt_context.controls,
                rng=rng,
            )
            decoded_samples = maybe_decode_for_hr(processing, base_samples)
        finally:
            finalize_tiling(tiling_applied, old_tiled)
    elif base_samples is None and decoded_samples is not None:
        tensor = decoded_samples.to(devices.default_device(), dtype=torch.float32)
        base_samples = processing.sd_model.encode_first_stage(tensor)

    if base_samples is None:
        raise RuntimeError("txt2img failed to produce initial samples")

    if hires_plan is None:
        result = GenerationResult(samples=base_samples, decoded=decoded_samples)
        return result.samples

    _reload_for_hires(processing)

    hires_samples = _run_hires_pass(
        processing,
        plan,
        payload,
        base_samples,
        decoded_samples,
        prompt_context,
    )

    result = GenerationResult(samples=hires_samples, decoded=None)
    return result.samples
