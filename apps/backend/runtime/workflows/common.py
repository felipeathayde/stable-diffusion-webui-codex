"""Shared workflow primitives used across Codex generation tasks."""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from apps.backend.codex import lora as codex_lora
from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.core.state import state as backend_state
from apps.backend.patchers.lora_apply import apply_loras_to_engine
from apps.backend.patchers.token_merging import apply_token_merging
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.processing.conditioners import decode_latent_batch, txt2img_conditioning
from apps.backend.runtime.processing.datatypes import ConditioningPayload, PromptContext, SamplingPlan
from apps.backend.runtime.sampling.context import SchedulerName, build_sampling_context
from apps.backend.runtime.sampling.catalog import (
    AUTO_TOKENS,
    SAMPLER_DEFAULT_SCHEDULER,
    SCHEDULER_ALIAS_TO_CANONICAL,
    SUPPORTED_SCHEDULERS,
)
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.infra.config.repo_root import get_repo_root

logger = logging.getLogger(__name__)

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_dump_latents(
    latents: torch.Tensor,
    processing: Any,
    plan: SamplingPlan,
    prompt_context: PromptContext,
) -> None:
    if not _truthy(os.getenv("CODEX_DUMP_LATENTS")):
        return

    path_hint = os.getenv("CODEX_DUMP_LATENTS_PATH")
    if path_hint:
        target = Path(path_hint).expanduser()
    else:
        target = get_repo_root() / "logs" / "diagnostics"
    if not target.suffix:
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        target = target / f"latents-{timestamp}.pt"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "latents": latents.detach().cpu(),
            "metadata": {
                "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                "width": int(getattr(processing, "width", 0) or 0),
                "height": int(getattr(processing, "height", 0) or 0),
                "steps": int(plan.steps),
                "guidance_scale": float(plan.guidance_scale),
                "sampler": getattr(processing, "sampler_name", None) or plan.sampler_name,
                "scheduler": plan.scheduler_name,
                "prompts": prompt_context.prompts,
                "negative_prompts": prompt_context.negative_prompts,
                "seeds": plan.seeds,
                "subseeds": plan.subseeds,
            },
        }
        torch.save(payload, target)
        logger.info("[diagnostics] dumped latents to %s", target)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to dump latents to %s: %s", target, exc)

def _normalize_scheduler_name(sampler: str | None, scheduler: str | None) -> str:
    sampler_key = (sampler or "").strip().lower()
    raw = (scheduler or "").strip().lower()
    if raw in AUTO_TOKENS:
        raw = SAMPLER_DEFAULT_SCHEDULER.get(sampler_key, "automatic")
    canonical = SCHEDULER_ALIAS_TO_CANONICAL.get(raw, raw)
    if canonical not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Scheduler '{canonical}' is not supported")
    try:
        canonical_enum = SchedulerName.from_string(canonical)
    except ValueError as exc:
        raise ValueError(f"Unsupported scheduler '{scheduler}' for sampler '{sampler}'") from exc
    return canonical_enum.value


def build_prompt_context(processing: Any, prompts: Sequence[str]) -> PromptContext:
    """Parse prompts, negative prompts, and extra network descriptors."""
    cleaned_prompts, prompt_loras, prompt_controls = parse_prompts_with_extras(list(prompts))
    controls = dict(prompt_controls)
    if "clip_skip" not in controls:
        meta = getattr(processing, "metadata", None)
        if isinstance(meta, dict) and meta.get("clip_skip") is not None:
            try:
                controls["clip_skip"] = max(1, int(meta.get("clip_skip")))
            except Exception:
                pass
    negative_prompts = list(
        getattr(processing, "negative_prompts", [getattr(processing, "negative_prompt", "")])
    )
    return PromptContext(
        prompts=cleaned_prompts,
        negative_prompts=negative_prompts,
        loras=prompt_loras,
        controls=controls,
    )


def apply_prompt_context(processing: Any, context: PromptContext) -> None:
    """Mutate processing object with normalized prompt data."""
    processing.prompts = context.prompts
    processing.negative_prompts = context.negative_prompts
    processing.cfg_scale = getattr(processing, "guidance_scale", 7.0)

    if "clip_skip" in context.controls:
        try:
            clip_skip = int(context.controls["clip_skip"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid clip_skip: must be an integer") from exc
        if clip_skip < 1:
            raise ValueError("Invalid clip_skip: must be >= 1")
        model = getattr(processing, "sd_model", None)
        if model is not None and hasattr(model, "set_clip_skip"):
            model.set_clip_skip(clip_skip)


def apply_dimension_overrides(processing: Any, controls: Mapping[str, Any]) -> None:
    """Apply dimension overrides parsed from prompt tags."""
    if "width" in controls:
        width = int(controls["width"])
        if width % 8 != 0 or width < 8 or width > 8192:
            raise ValueError("Invalid <width>: must be multiple of 8 and in [8,8192]")
        processing.width = width
    if "height" in controls:
        height = int(controls["height"])
        if height % 8 != 0 or height < 8 or height > 8192:
            raise ValueError("Invalid <height>: must be multiple of 8 and in [8,8192]")
        processing.height = height


def resolve_noise_settings(processing: Any) -> NoiseSettings:
    """Inspect processing overrides/environment and return noise source settings."""
    source = None
    eta_delta = 0
    overrides = getattr(processing, "override_settings", {})
    if isinstance(overrides, dict):
        source = overrides.get("randn_source") or overrides.get("noise_source")
        eta_delta = overrides.get("eta_noise_seed_delta", eta_delta)
    metadata = getattr(processing, "metadata", {})
    if isinstance(metadata, dict):
        source = metadata.get("randn_source", source)
    if getattr(processing, "noise_source", None):
        source = processing.noise_source
    env_source = os.getenv("CODEX_NOISE_SOURCE")
    if source is None and env_source:
        source = env_source

    try:
        source_kind = NoiseSourceKind.from_string(source) if source else NoiseSourceKind.GPU
    except ValueError:
        source_kind = NoiseSourceKind.GPU

    delta = int(getattr(processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0)
    settings = NoiseSettings(source=source_kind, eta_noise_seed_delta=delta)
    processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
    return settings


def build_sampling_plan(
    processing: Any,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    noise_settings: NoiseSettings | None = None,
) -> SamplingPlan:
    """Create a sampling plan for the generation run."""
    if noise_settings is None:
        noise_settings = resolve_noise_settings(processing)
    guidance = float(getattr(processing, "guidance_scale", 7.0) or 7.0)
    steps = int(getattr(processing, "steps", 20) or 20)
    sampler_name = getattr(processing, "sampler_name", None)
    scheduler_name = getattr(processing, "scheduler", None)
    try:
        normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    except ValueError as exc:
        logger.warning("Invalid scheduler '%s' for sampler '%s'; falling back to automatic.", scheduler_name, sampler_name)
        normalized_scheduler = SchedulerName.AUTOMATIC.value
    processing.scheduler = normalized_scheduler
    return SamplingPlan(
        sampler_name=sampler_name,
        scheduler_name=normalized_scheduler,
        steps=steps,
        guidance_scale=guidance,
        seeds=list(seeds),
        subseeds=list(subseeds),
        subseed_strength=float(subseed_strength),
        noise_settings=noise_settings,
    )


def apply_sampling_overrides(
    processing: Any,
    controls: Mapping[str, Any],
    plan: SamplingPlan,
) -> SamplingPlan:
    """Apply prompt-derived overrides to the sampling plan."""
    sampler_name = controls.get("sampler")
    if sampler_name:
        processing.sampler_name = str(sampler_name)
        plan.sampler_name = str(sampler_name)

    try:
        if "cfg" in controls:
            cfg = float(controls["cfg"])
            processing.guidance_scale = cfg
            processing.cfg_scale = cfg
            plan.guidance_scale = cfg
        if "steps" in controls:
            steps = int(float(controls["steps"]))
            processing.steps = steps
            plan.steps = steps
        if "seed" in controls:
            seed = int(float(controls["seed"]))
            plan.seeds = [seed]
            processing.seeds = [seed]
    except Exception:
        logger.debug("Failed to apply sampling overrides", exc_info=True)
    return plan


def ensure_sampler_and_rng(
    processing: Any,
    plan: SamplingPlan,
    *,
    latent_channels: int | None = None,
) -> ImageRNG:
    """Ensure processing has a sampler + RNG configured for the current plan."""
    algo = plan.sampler_name or getattr(processing, "sampler_name", None)
    processing.sampler = CodexSampler(processing.sd_model, algorithm=algo)
    if latent_channels is None:
        latent_channels = getattr(
            processing.sd_model.codex_objects_after_applying_lora.vae,
            "latent_channels",
            4,
        )
    shape = (
        latent_channels,
        processing.height // 8,
        processing.width // 8,
    )
    rng = ImageRNG(
        shape,
        plan.seeds,
        subseeds=plan.subseeds,
        subseed_strength=plan.subseed_strength,
        seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
        settings=plan.noise_settings,
    )
    processing.rng = rng
    return rng


def run_process_scripts(processing: Any) -> None:
    """Execute legacy script hooks if present."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is not None and hasattr(script_runner, "process"):
        script_runner.process(processing)


def activate_extra_networks(processing: Any) -> None:
    """Apply globally selected extra networks to the current engine."""
    if getattr(processing, "disable_extra_networks", False):
        return
    try:
        selections = codex_lora.get_selections()
    except Exception:
        selections = []
    if not selections:
        return
    stats = apply_loras_to_engine(processing.sd_model, selections)
    logger.info("[native] Applied %d LoRA(s), %d params touched", stats.files, stats.params_touched)


def set_shared_job(processing: Any) -> None:
    """Update shared backend job metadata for batch runs."""
    if getattr(processing, "iterations", 1) <= 1:
        return
    backend_state.begin(job=f"Batch 1 out of {processing.iterations}")


def collect_lora_selections(prompt_loras: Sequence[Any]) -> list[Any]:
    """Merge global selections with prompt-local LoRA descriptors."""
    selections: list[Any] = []
    seen: set[str] = set()
    try:
        all_selections: Iterable[Any] = list(codex_lora.get_selections()) + list(prompt_loras)
    except Exception:
        all_selections = list(prompt_loras)
    for sel in all_selections:
        path = getattr(sel, "path", None)
        if not path or path in seen:
            continue
        seen.add(path)
        selections.append(sel)
    return selections


def run_before_sampling_hooks(
    processing: Any,
    prompt_context: PromptContext,
    seeds: Sequence[int],
    subseeds: Sequence[int],
) -> None:
    """Invoke before-sampling hooks on processing scripts."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is None:
        activate_extra_networks(processing)
        return

    hook_kwargs = {
        "batch_number": 0,
        "prompts": prompt_context.prompts,
        "seeds": list(seeds),
        "subseeds": list(subseeds),
        "negative_prompts": prompt_context.negative_prompts,
    }

    if hasattr(script_runner, "before_process_batch"):
        script_runner.before_process_batch(processing, **hook_kwargs)

    if hasattr(script_runner, "process_batch"):
        script_runner.process_batch(processing, **hook_kwargs)

    activate_extra_networks(processing)
    set_shared_job(processing)


def run_post_sample_hooks(processing: Any, samples: torch.Tensor) -> torch.Tensor:
    """Invoke post-sample hooks, returning the potentially modified samples."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is None or not hasattr(script_runner, "post_sample"):
        return samples

    class _Args:
        def __init__(self, value: torch.Tensor) -> None:
            self.samples = value

    args = _Args(samples)
    script_runner.post_sample(processing, args)
    return getattr(args, "samples", samples)


def latents_to_pil(decoded: torch.Tensor) -> list[Image.Image]:
    """Convert decoded latent tensor into RGB PIL images."""
    images: list[Image.Image] = []
    for sample in decoded:
        arr = sample.detach().float().cpu().clamp(-1, 1)
        arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def pil_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    """Convert a sequence of PIL images into a normalized tensor."""
    arrays = []
    for img in images:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.moveaxis(arr, 2, 0)
        arrays.append(arr)
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    return tensor.to(devices.default_device(), dtype=torch.float32)


def maybe_decode_for_hr(processing: Any, samples: torch.Tensor) -> torch.Tensor | None:
    """Decode samples to RGB when the hires pass requires pixel-space input."""
    if not getattr(processing, "enable_hr", False):
        return None

    devices.torch_gc()

    if getattr(processing, "latent_scale_mode", None) is None:
        decoded = decode_latent_batch(processing.sd_model, samples)
        return decoded.to(dtype=torch.float32)
    return None


def execute_sampling(
    processing: Any,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    prompt_context: PromptContext,
    prompt_loras: Sequence[Any],
    prompt_controls: Mapping[str, Any],
    *,
    rng: ImageRNG,
    noise: torch.Tensor | None = None,
    image_conditioning: torch.Tensor | None = None,
    init_latent: torch.Tensor | None = None,
    start_at_step: int | None = None,
) -> torch.Tensor:
    """Execute the sampler using the provided configuration."""
    if noise is None:
        noise = rng.next()

    model = processing.sd_model
    if hasattr(model, "codex_objects_original") and model.codex_objects_original is not None:
        model.codex_objects = model.codex_objects_original.shallow_copy()

    run_before_sampling_hooks(processing, prompt_context, plan.seeds, plan.subseeds)

    merged = collect_lora_selections(prompt_loras)
    if merged:
        stats = apply_loras_to_engine(model, merged)
        logger.info("[native] Applied %d LoRA(s), %d params touched", stats.files, stats.params_touched)
    model.codex_objects = model.codex_objects_after_applying_lora.shallow_copy()

    strategy = prompt_controls.get("token_merge_strategy") or getattr(
        processing,
        "get_token_merging_strategy",
        lambda: None,
    )()
    if not strategy:
        strategy = os.getenv("CODEX_TOKEN_MERGE_STRATEGY", "avg")
    ratio_override = prompt_controls.get("token_merge_ratio")
    ratio = float(ratio_override) if ratio_override is not None else float(
        processing.get_token_merging_ratio(for_hires=bool(init_latent is not None))
    )
    apply_token_merging(model, ratio, strategy=strategy)

    if processing.scripts is not None:
        processing.scripts.process_before_every_sampling(
            processing,
            x=noise,
            noise=noise,
            c=payload.conditioning,
            uc=payload.unconditional,
        )

    if getattr(processing, "modified_noise", None) is not None:
        noise = processing.modified_noise
        processing.modified_noise = None

    def _preview_cb(denoised_latent: torch.Tensor, step: int, total: int) -> None:
        # Skip preview decode on the final step; the engine will decode once
        # for the actual output, avoiding redundant VAE work at the tail.
        try:
            if total is not None and int(total) > 0 and int(step) >= int(total):
                return
        except Exception:
            # If step/total are malformed, fall back to best-effort preview.
            pass
        img = decode_latent_batch(processing.sd_model, denoised_latent)
        arr = img[0].detach().float().cpu().clamp(-1, 1)
        arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
        backend_state.set_current_image(Image.fromarray(arr, mode="RGB"))

    if image_conditioning is None:
        image_conditioning = txt2img_conditioning(
            processing.sd_model,
            noise,
            processing.width,
            processing.height,
        )

    sampler_name = plan.sampler_name or getattr(processing, "sampler_name", None) or processing.sampler.algorithm
    scheduler_name = plan.scheduler_name or getattr(processing, "scheduler", None)
    context = build_sampling_context(
        processing.sd_model,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        steps=int(plan.steps),
        noise_source=plan.noise_settings.source.value,
        eta_noise_seed_delta=plan.noise_settings.eta_noise_seed_delta,
        device=noise.device,
        dtype=noise.dtype,
    )

    samples = processing.sampler.sample(
        processing,
        noise,
        payload.conditioning,
        payload.unconditional,
        image_conditioning=image_conditioning,
        init_latent=init_latent,
        start_at_step=start_at_step,
        preview_callback=_preview_cb,
        context=context,
    )

    samples = run_post_sample_hooks(processing, samples)
    _maybe_dump_latents(samples, processing, plan, prompt_context)
    devices.torch_gc()
    return samples


def apply_tiling_if_requested(processing: Any, controls: Mapping[str, Any]) -> tuple[bool, bool]:
    """Enable VAE tiling temporarily when prompts request it."""
    old_value = memory_management.VAE_ALWAYS_TILED
    applied = False
    if controls.get("tiling") is True:
        memory_management.VAE_ALWAYS_TILED = True
        applied = True
    return applied, old_value


def finalize_tiling(applied: bool, previous: bool) -> None:
    """Restore VAE tiling flag."""
    if applied:
        memory_management.VAE_ALWAYS_TILED = previous


__all__ = [
    "build_prompt_context",
    "apply_prompt_context",
    "apply_dimension_overrides",
    "resolve_noise_settings",
    "build_sampling_plan",
    "apply_sampling_overrides",
    "ensure_sampler_and_rng",
    "run_process_scripts",
    "activate_extra_networks",
    "set_shared_job",
    "collect_lora_selections",
    "run_before_sampling_hooks",
    "run_post_sample_hooks",
    "latents_to_pil",
    "pil_to_tensor",
    "maybe_decode_for_hr",
    "execute_sampling",
    "apply_tiling_if_requested",
    "finalize_tiling",
]
