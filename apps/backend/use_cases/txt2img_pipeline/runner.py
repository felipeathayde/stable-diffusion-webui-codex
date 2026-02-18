"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stage-based txt2img pipeline orchestrator (sampling + hi-res + optional refiner).
Coordinates prompt parsing, conditioning, sampling execution, tiling/overrides, and optional refiner stages while producing images and metadata, with fail-loud conditioning guards that avoid embedding raw prompt text in raised errors.
Conditioning smart-cache entries are keyed by model/load identity plus wrapped prompt metadata and stored detached on CPU to avoid stale hits and cross-request GPU pinning.
The hires stage delegates init preparation and `denoise` semantics to the global hires-fix workflow stage (`apps/backend/runtime/pipeline_stages/hires_fix.py`).
When configured, the hires second pass applies sampler/scheduler overrides (validated) by deriving a dedicated `SamplingPlan` for the hires pass.
When smart offload is enabled, keeps required text-encoder patchers loaded across cond+uncond and unloads them after conditioning.

Symbols (top-level; keep in sync; no ghosts):
- `PrepareState` (dataclass): Prepared per-run state (resolved engine + plans + prompt context) used across stages.
- `SamplingOutput` (dataclass): Sampling result container (latents/images + metadata) passed between pipeline stages.
- `Txt2ImgPipelineRunner` (class): Main orchestrator; owns the stage pipeline (conditioning/sampling/hires/refiner) and calls the runtime helpers
  (hires stage uses the global hires-fix stage to route latent vs spandrel upscalers; integrates smart cache + pipeline tracing).
- `GenerationResult` (dataclass): Standardized output container for the runner (`samples` + optional `decoded`).
"""
# // tags: txt2img, pipeline, sdxl, hires, refiner

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, is_dataclass, replace
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.infra.config import args as backend_args
from apps.backend.runtime.diagnostics.pipeline_debug import log as pipeline_log, pipeline_trace
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import (
    smart_cache_enabled,
    smart_offload_enabled,
    record_smart_cache_hit,
    record_smart_cache_miss,
)
from apps.backend.runtime.memory.smart_offload_invariants import (
    enforce_smart_offload_pre_conditioning_residency,
    enforce_smart_offload_text_encoders_off,
)
from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    GenerationResult,
    HiResPlan,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img, RefinerConfig
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.runtime.pipeline_stages.image_io import maybe_decode_for_hr
from apps.backend.runtime.pipeline_stages.hires_fix import (
    prepare_hires_latents_and_conditioning,
    start_at_step_from_denoise,
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
from apps.backend.core.engine_loader import EngineLoadOptions, load_engine as _load_engine
from apps.backend.use_cases.txt2img_pipeline.refiner import GlobalRefinerStage, HiresRefinerStage, RefinerStage


@dataclass(slots=True)
class PrepareState:
    """State captured after the preparation stage."""

    prompt_context: PromptContext
    hires_prompt_context: PromptContext | None
    sampling_plan: SamplingPlan
    rng: ImageRNG
    payload: ConditioningPayload
    hires_plan: HiResPlan | None
    init_latents: torch.Tensor | None
    init_decoded: torch.Tensor | None
    cond: object | None = None
    uncond: object | None = None


@dataclass(slots=True)
class SamplingOutput:
    """Result of executing a sampling stage."""

    samples: torch.Tensor
    decoded: torch.Tensor | None


class Txt2ImgPipelineRunner:
    """Orchestrates txt2img generation through well-defined stages."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("backend.use_cases.txt2img.pipeline")
        # SDXL conditioning cache (prompt + dims → (cond, uncond))
        # shared across runs for this runner instance.
        self._conditioning_cache: dict[tuple, tuple[object, object | None]] = {}

    @classmethod
    def _freeze_cache_value(cls, value: Any, *, _depth: int = 0) -> object:
        """Convert arbitrary values to a deterministic, hashable cache token."""
        if _depth > 6:
            return ("depth_limit", type(value).__name__)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            if type(value) is str:
                return value
            attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
            if attrs:
                return (
                    "string_subclass",
                    f"{type(value).__module__}.{type(value).__qualname__}",
                    str(value),
                    attrs,
                )
            return ("string_subclass", f"{type(value).__module__}.{type(value).__qualname__}", str(value))
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                tuple(int(dim) for dim in value.shape),
                str(value.dtype),
                str(value.device),
            )
        if isinstance(value, Mapping):
            items: list[tuple[str, object]] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                items.append((str(key), cls._freeze_cache_value(value[key], _depth=_depth + 1)))
            return ("mapping", tuple(items))
        if isinstance(value, tuple):
            return ("tuple", tuple(cls._freeze_cache_value(item, _depth=_depth + 1) for item in value))
        if isinstance(value, list):
            attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
            return (
                "list",
                tuple(cls._freeze_cache_value(item, _depth=_depth + 1) for item in value),
                attrs,
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(
                    sorted(
                        (cls._freeze_cache_value(item, _depth=_depth + 1) for item in value),
                        key=repr,
                    )
                ),
            )
        if is_dataclass(value):
            try:
                return (
                    "dataclass",
                    f"{type(value).__module__}.{type(value).__qualname__}",
                    cls._freeze_cache_value(asdict(value), _depth=_depth + 1),
                )
            except Exception:
                return ("dataclass_repr", f"{type(value).__module__}.{type(value).__qualname__}", repr(value))

        attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
        if attrs:
            return (
                "object",
                f"{type(value).__module__}.{type(value).__qualname__}",
                attrs,
                str(value),
            )
        return ("repr", f"{type(value).__module__}.{type(value).__qualname__}", repr(value))

    @classmethod
    def _freeze_known_attrs(cls, value: Any, *, _depth: int) -> tuple[tuple[str, object], ...]:
        attr_names = (
            "is_negative_prompt",
            "smart_cache",
            "distilled_cfg_scale",
            "cfg_scale",
            "width",
            "height",
            "target_width",
            "target_height",
            "crop_left",
            "crop_top",
            "tenc_source",
            "tenc_path",
            "vae_source",
            "vae_path",
            "core_streaming_enabled",
            "extras",
            "label",
            "family",
        )
        attrs: list[tuple[str, object]] = []
        for attr_name in attr_names:
            if not hasattr(value, attr_name):
                continue
            try:
                attr_value = getattr(value, attr_name)
            except Exception:
                continue
            if callable(attr_value):
                continue
            attrs.append((attr_name, cls._freeze_cache_value(attr_value, _depth=_depth + 1)))
        return tuple(sorted(attrs))

    @classmethod
    def _conditioning_model_identity(cls, sd_model: Any) -> tuple[object, ...]:
        model_ref = getattr(sd_model, "model_ref", None)
        if model_ref in (None, ""):
            model_ref = getattr(sd_model, "_current_model_ref", None)
        load_options = getattr(sd_model, "_load_options", None)
        lora_hash = getattr(sd_model, "current_lora_hash", None)

        codex_objects = None
        try:
            codex_objects = getattr(sd_model, "codex_objects", None)
        except Exception:
            codex_objects = None

        denoiser_obj = getattr(codex_objects, "denoiser", None) if codex_objects is not None else None
        vae_obj = getattr(codex_objects, "vae", None) if codex_objects is not None else None
        vae_patcher = getattr(vae_obj, "patcher", vae_obj) if vae_obj is not None else None

        text_encoder_ids: tuple[tuple[str, int], ...] = ()
        if codex_objects is not None:
            text_encoders = getattr(codex_objects, "text_encoders", None)
            if isinstance(text_encoders, dict):
                pairs: list[tuple[str, int]] = []
                for name, entry in text_encoders.items():
                    if entry is None:
                        continue
                    patcher = getattr(entry, "patcher", entry)
                    pairs.append((str(name), int(id(patcher))))
                text_encoder_ids = tuple(sorted(pairs))

        return (
            f"{type(sd_model).__module__}.{type(sd_model).__qualname__}",
            str(getattr(sd_model, "engine_id", "") or ""),
            cls._freeze_cache_value(model_ref),
            cls._freeze_cache_value(load_options),
            cls._freeze_cache_value(lora_hash),
            int(id(denoiser_obj)),
            int(id(vae_patcher)),
            text_encoder_ids,
        )

    @staticmethod
    def _conditioning_target_device() -> torch.device | str:
        get_device = getattr(memory_management.manager, "get_device", None)
        if callable(get_device):
            try:
                return get_device(DeviceRole.TEXT_ENCODER)
            except Exception:
                pass
        return "cpu"

    @staticmethod
    def _log_tensor_stats(label: str, tensor: torch.Tensor | None) -> None:
        logger = logging.getLogger("backend.use_cases.txt2img.pipeline")
        if tensor is None:
            logger.info("[sampling] %s: <none>", label)
            return
        with torch.no_grad():
            data = tensor.detach()
            try:
                stats_tensor = data.float()
            except Exception:
                stats_tensor = data
            mean = float(stats_tensor.mean().item())
            std = float(stats_tensor.std(unbiased=False).item())
            min_value = float(stats_tensor.min().item())
            max_value = float(stats_tensor.max().item())
        logger.info(
            "[sampling] %s: shape=%s dtype=%s device=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            min_value,
            max_value,
            mean,
            std,
        )

    def _compute_conditioning(self, processing: CodexProcessingTxt2Img, context: PromptContext):
        """Build cond/uncond using the engine's SDXL-aware helpers after prompt parsing and dimension overrides.

        Smart Cache is resolved per job: when ``processing.smart_cache`` is present it
        takes precedence over the global options snapshot, so callers can flip cache
        on/off for individual requests without touching Quicksettings.
        """
        sd_model = getattr(processing, "sd_model", None)
        if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
            return None, None

        setattr(processing, "_codex_conditioning_cache_hit", False)

        prompts = list(context.prompts or [getattr(processing, "prompt", "")])
        negative_prompts = list(context.negative_prompts or [getattr(processing, "negative_prompt", "")])
        uses_distilled_cfg = bool(getattr(sd_model, "use_distilled_cfg_scale", False))
        if hasattr(sd_model, "_prepare_prompt_wrappers"):
            prompts_payload = sd_model._prepare_prompt_wrappers(prompts, processing, is_negative=False)
            negative_payload = (
                None
                if uses_distilled_cfg
                else sd_model._prepare_prompt_wrappers(negative_prompts, processing, is_negative=True)
            )
        else:
            prompts_payload = prompts
            negative_payload = None if uses_distilled_cfg else negative_prompts

        smart_flag = getattr(processing, "smart_cache", None)
        cache_enabled = bool(smart_flag) if smart_flag is not None else smart_cache_enabled()
        key = None
        if cache_enabled:
            try:
                clip_skip = int(context.controls.get("clip_skip")) if "clip_skip" in context.controls else None
                key = (
                    self._conditioning_model_identity(sd_model),
                    self._freeze_cache_value(prompts_payload),
                    self._freeze_cache_value(negative_payload),
                    clip_skip,
                )
            except Exception:
                key = None

        if cache_enabled and key is not None:
            cached = self._conditioning_cache.get(key)
            if cached is not None:
                record_smart_cache_hit("sdxl.runner.conditioning")
                setattr(processing, "_codex_conditioning_cache_hit", True)
                enforce_smart_offload_text_encoders_off(sd_model, stage="txt2img.conditioning(cache-hit)")
                target_device = self._conditioning_target_device()
                return move_to_device(cached, device=target_device)
            record_smart_cache_miss("sdxl.runner.conditioning")

        enforce_smart_offload_pre_conditioning_residency(sd_model, stage="txt2img.conditioning")

        text_encoder_patchers: list[tuple[str, object]] = []
        if smart_offload_enabled():
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
                    pipeline_log(f"[conditioning] smart_offload: loading text encoder '{name}' patcher for stage")
                    memory_management.manager.load_model(patcher)

        try:
            # Preserve spatial metadata via engine helper when available
            cond = sd_model.get_learned_conditioning(prompts_payload)
            if uses_distilled_cfg:
                uncond = None
            else:
                uncond = sd_model.get_learned_conditioning(negative_payload)

            # If uncond comes back zero for a non-empty negative prompt, fail fast instead of sampling with CFG degenerate
            non_empty_negative = any(str(p or "").strip() for p in negative_prompts)
            if non_empty_negative:
                uncond_cross = uncond.get("crossattn") if isinstance(uncond, dict) else None
                if isinstance(uncond_cross, torch.Tensor):
                    norm_uncond = float(uncond_cross.abs().sum().item())
                    if norm_uncond < 1e-6:
                        negative_count = sum(1 for item in negative_prompts if str(item or "").strip())
                        raise RuntimeError(
                            "Unconditional embedding returned all zeros for a non-empty negative prompt batch "
                            f"(count={negative_count}). Check CLIP encoders or prompt handling before sampling."
                        )
        finally:
            if smart_offload_enabled():
                if text_encoder_patchers:
                    pipeline_log("[conditioning] smart_offload: unloading text encoders after stage")
                enforce_smart_offload_text_encoders_off(sd_model, stage="txt2img.conditioning(post)")

        pair = (cond, uncond)
        if cache_enabled and key is not None:
            # Always keep only the most recent entry; older cache is discarded.
            self._conditioning_cache.clear()
            self._conditioning_cache[key] = detach_to_cpu(pair)
        return pair

    def _log_conditioning(self, cond: object, uncond: object) -> None:
        """Optional conditioning diagnostics controlled by --debug-conditioning / CODEX_DEBUG_COND."""
        try:
            if not getattr(backend_args.args, "debug_conditioning", False):
                return

            def _shape(t):
                return tuple(t.shape) if isinstance(t, torch.Tensor) else None

            def _dtype(t):
                return str(t.dtype) if isinstance(t, torch.Tensor) else None

            def _device(t):
                return str(t.device) if isinstance(t, torch.Tensor) else None

            def _norm(t):
                return float(t.detach().abs().mean().item()) if isinstance(t, torch.Tensor) else None

            def _split_vector(v):
                if not isinstance(v, torch.Tensor):
                    return None, None
                if int(v.shape[1]) <= 1280:
                    pooled = v
                    pooled_l2 = float(pooled.detach().float().norm().item())
                    return (
                        float(pooled.detach().abs().mean().item()),
                        0.0,
                        pooled_l2,
                        0.0,
                    )
                pooled = v[:, :1280]
                adm = v[:, 1280:]
                pooled_l2 = float(pooled.detach().float().norm().item())
                adm_l2 = float(adm.detach().float().norm().item())
                return (
                    float(pooled.detach().abs().mean().item()),
                    float(adm.detach().abs().mean().item()),
                    pooled_l2,
                    adm_l2,
                )

            ca = cond.get("crossattn") if isinstance(cond, dict) else None
            va = cond.get("vector") if isinstance(cond, dict) else None
            ga = cond.get("guidance") if isinstance(cond, dict) else None
            ua = uncond.get("crossattn") if isinstance(uncond, dict) else None
            uv = uncond.get("vector") if isinstance(uncond, dict) else None
            ug = uncond.get("guidance") if isinstance(uncond, dict) else None

            p_mean, adm_mean, p_l2, adm_l2 = _split_vector(va) if va is not None else (None, None, None, None)
            up_mean, uadm_mean, up_l2, uadm_l2 = _split_vector(uv) if uv is not None else (None, None, None, None)

            def _guidance_scalar(t):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    return None
                try:
                    return float(t.detach().float().view(-1)[0].item())
                except Exception:
                    return None

            self._logger.info(
                "[sdxl] cond: cross shape=%s dtype=%s dev=%s mean_abs=%.4f l2=%.4f | vector shape=%s dtype=%s dev=%s mean_abs=%.4f l2=%.4f pooled_mean=%.4f pooled_l2=%.4f adm_mean=%.4f adm_l2=%.4f guidance=%s",
                _shape(ca), _dtype(ca), _device(ca), (_norm(ca) or 0.0), float(ca.detach().float().norm().item()) if ca is not None else 0.0,
                _shape(va), _dtype(va), _device(va), (_norm(va) or 0.0), float(va.detach().float().norm().item()) if va is not None else 0.0,
                (p_mean or 0.0), (p_l2 or 0.0), (adm_mean or 0.0), (adm_l2 or 0.0),
                _guidance_scalar(ga),
            )
            # Only log uncond if it exists (distilled CFG models like Flux don't use uncond)
            if uncond is not None:
                self._logger.info(
                    "[sdxl] uncond: cross shape=%s dtype=%s dev=%s mean_abs=%.4f l2=%.4f | vector shape=%s dtype=%s dev=%s mean_abs=%.4f l2=%.4f pooled_mean=%.4f pooled_l2=%.4f adm_mean=%.4f adm_l2=%.4f guidance=%s",
                    _shape(ua), _dtype(ua), _device(ua), (_norm(ua) or 0.0), float(ua.detach().float().norm().item()) if ua is not None else 0.0,
                    _shape(uv), _dtype(uv), _device(uv), (_norm(uv) or 0.0), float(uv.detach().float().norm().item()) if uv is not None else 0.0,
                    (up_mean or 0.0), (up_l2 or 0.0), (uadm_mean or 0.0), (uadm_l2 or 0.0),
                    _guidance_scalar(ug),
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("[sdxl] conditioning diagnostics skipped: %s", exc)

    def _apply_refiner_stage(
        self,
        stage: RefinerStage,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        if not stage.is_enabled():
            return samples

        return stage.run(
            processing=processing,
            prompt_context=prompt_context,
            noise_settings=noise_settings,
            samples=samples,
            compute_conditioning=self._compute_conditioning,
            log_conditioning=self._log_conditioning,
            log_tensor_stats=self._log_tensor_stats,
        )

    # ------------------------------------------------------------------ public API
    @pipeline_trace
    def run(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> GenerationResult:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "StableDiffusionXL exige CUDA ativo (torch.cuda.is_available() retornou False). "
                "Instale os drivers/CUDA corretos ou execute o modo somente CPU conscientemente."
            )
        setattr(processing, "_codex_last_decode_engine", None)
        t_start = time.perf_counter()
        t_prepare_end: float | None = None
        t_base_end: float | None = None
        t_hires_end: float | None = None
        t_refiner_end: float | None = None

        model_device, model_dtype = self._sd_model_device_info(processing)
        if model_device is not None:
            self._logger.info("SDXL sd_model device=%s dtype=%s", model_device, model_dtype)
            if model_device.type != "cuda":
                raise RuntimeError(
                    "StableDiffusionXL carregou o modelo em %s; configure uma GPU CUDA antes de continuar."
                    % model_device
                )

        state = self._prepare_state(
            processing,
            conditioning_data,
            unconditional_data,
            seeds,
            subseeds,
            subseed_strength,
            prompts,
        )
        base_result = self._execute_base_sampling(processing, state)
        t_prepare_end = time.perf_counter()

        final_samples = base_result.samples
        t_base_end = time.perf_counter()

        if state.hires_plan is not None:
            self._reload_for_hires(processing, state)
            final_samples = self._run_hires_pass(processing, state, base_result)
            t_hires_end = time.perf_counter()
        else:
            t_hires_end = t_base_end

        final_samples = self._maybe_run_refiner_pass(processing, state, final_samples)
        t_refiner_end = time.perf_counter()

        try:
            timings: dict[str, float] = {}
            if t_prepare_end is not None:
                timings["prepare_ms"] = (t_prepare_end - t_start) * 1000.0
            if t_base_end is not None and t_prepare_end is not None:
                timings["base_sampling_ms"] = (t_base_end - t_prepare_end) * 1000.0
            if state.hires_plan is not None and t_hires_end is not None and t_base_end is not None:
                timings["hires_ms"] = (t_hires_end - t_base_end) * 1000.0
            if t_refiner_end is not None and t_hires_end is not None:
                timings["refiner_ms"] = max(0.0, (t_refiner_end - t_hires_end) * 1000.0)
            timings["total_pipeline_ms"] = max(0.0, (t_refiner_end or time.perf_counter()) - t_start) * 1000.0
            processing.update_extra_param("Timings (ms)", timings)
        except Exception:
            # Timings must never break generation; swallow errors defensively.
            pass

        # Auto-print and save timeline trace if enabled
        try:
            from apps.backend.runtime.diagnostics.timeline import auto_save_and_print
            trace_path = auto_save_and_print()
            if trace_path:
                processing.update_extra_param("Timeline Trace", trace_path)
        except Exception:
            pass  # Timeline should never break generation

        return GenerationResult(
            samples=final_samples,
            decoded=None,
            metadata={"conditioning_cache_hit": bool(getattr(processing, "_codex_conditioning_cache_hit", False))},
            decode_engine=getattr(processing, "_codex_last_decode_engine", None) or processing.sd_model,
        )

    # ------------------------------------------------------------------ stages
    @pipeline_trace
    def _prepare_state(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> PrepareState:
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

        payload = ConditioningPayload(conditioning=conditioning_data, unconditional=unconditional_data)

        init_latents, init_decoded = self._prepare_first_pass_from_image(processing)
        hires_plan = self._build_hires_plan(processing)

        # Compute conditioning if not provided (SDXL path); preserve metadata (width/height/targets) after overrides.
        cond = conditioning_data
        uncond = unconditional_data
        if cond is None or uncond is None:
            cond, uncond = self._compute_conditioning(processing, prompt_context)
            # For distilled CFG models (Flux), uncond is intentionally None - only cond is required
            if cond is None:
                raise RuntimeError("Failed to build conditioning for txt2img; get_learned_conditioning returned None.")
            payload = ConditioningPayload(conditioning=cond, unconditional=uncond)
            self._log_conditioning(cond, uncond)

        return PrepareState(
            prompt_context=prompt_context,
            hires_prompt_context=None,
            sampling_plan=plan,
            rng=rng,
            payload=payload,
            hires_plan=hires_plan,
            init_latents=init_latents,
            init_decoded=init_decoded,
            cond=cond,
            uncond=uncond,
        )

    @pipeline_trace
    def _execute_base_sampling(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> SamplingOutput:
        base_samples = state.init_latents
        decoded_samples = state.init_decoded

        if base_samples is None and decoded_samples is None:
            tiling_applied, previous_tiling = apply_tiling_if_requested(processing, state.prompt_context.controls)
            try:
                base_samples = execute_sampling(
                    processing,
                    state.sampling_plan,
                    state.payload,
                    state.prompt_context,
                    state.prompt_context.loras,
                    state.prompt_context.controls,
                    rng=state.rng,
                )
                decoded_samples = maybe_decode_for_hr(processing, base_samples)
            finally:
                finalize_tiling(tiling_applied, previous_tiling)
        elif base_samples is None and decoded_samples is not None:
            tensor = decoded_samples.to(devices.default_device(), dtype=torch.float32)
            base_samples = processing.sd_model.encode_first_stage(tensor)

        if base_samples is None:
            raise RuntimeError("txt2img failed to produce initial samples")

        self._log_tensor_stats("base_samples", base_samples)
        self._log_tensor_stats("base_decoded", decoded_samples)

        return SamplingOutput(samples=base_samples, decoded=decoded_samples)

    @pipeline_trace
    def _reload_for_hires(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> None:
        assert state.hires_plan is not None
        model_name = getattr(processing, "hr_checkpoint_name", None)
        if not model_name or model_name == "Use same checkpoint":
            return

        processing.firstpass_use_distilled_cfg_scale = processing.sd_model.use_distilled_cfg_scale
        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=None,
        )
        try:
            processing.sd_model = _load_engine(str(model_name), options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load hires checkpoint '{model_name}': {exc}") from exc

        if processing.sd_model.use_distilled_cfg_scale:
            processing.extra_generation_params["Hires Distilled CFG Scale"] = processing.hr_distilled_cfg

    @pipeline_trace
    def _run_hires_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        base_result: SamplingOutput,
    ) -> torch.Tensor:
        assert state.hires_plan is not None
        hires_plan_cfg = state.hires_plan
        hires_cfg = processing.hires
        processing.ensure_hires_prompts()

        target_width = int(hires_plan_cfg.target_width)
        target_height = int(hires_plan_cfg.target_height)
        steps = int(hires_plan_cfg.steps)
        denoise = float(hires_plan_cfg.denoise)

        original_attrs = {
            "prompts": processing.prompts,
            "negative_prompts": getattr(processing, "negative_prompts", []),
            "width": processing.width,
            "height": processing.height,
            "guidance_scale": processing.guidance_scale,
            "steps": processing.steps,
            "sampler_name": getattr(processing, "sampler_name", None),
            "scheduler": getattr(processing, "scheduler", None),
            "sampler": getattr(processing, "sampler", None),
        }

        hr_prompts_source = (
            processing.hr_prompts
            or processing.all_prompts
            or state.prompt_context.prompts
        )
        hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(list(hr_prompts_source))
        hr_negative = processing.hr_negative_prompts or original_attrs["negative_prompts"]
        hires_prompt_context = PromptContext(
            prompts=hr_cleaned_prompts,
            negative_prompts=hr_negative,
            loras=hr_prompt_loras,
            controls=dict(hr_prompt_controls),
        )
        state.hires_prompt_context = hires_prompt_context

        try:
            processing.prompts = hires_prompt_context.prompts
            processing.negative_prompts = hires_prompt_context.negative_prompts
            processing.width = target_width
            processing.height = target_height
            processing.guidance_scale = float(hires_cfg.cfg or processing.guidance_scale)
            processing.cfg_scale = processing.guidance_scale
            processing.steps = int(steps)
            hires_sampler, hires_scheduler = resolve_sampler_scheduler_override(
                base_sampler=str(state.sampling_plan.sampler_name or ""),
                base_scheduler=str(state.sampling_plan.scheduler_name or ""),
                sampler_override=getattr(hires_cfg, "sampler_name", None),
                scheduler_override=getattr(hires_cfg, "scheduler", None),
            )
            processing.sampler_name = hires_sampler
            processing.scheduler = hires_scheduler
            processing.sampler = CodexSampler(processing.sd_model, algorithm=hires_sampler)
            processing.prepare_prompt_data()

            latents, image_conditioning = prepare_hires_latents_and_conditioning(
                processing,
                base_samples=base_result.samples,
                base_decoded=base_result.decoded,
                hires_plan=hires_plan_cfg,
                tile=getattr(hires_cfg, "tile", None),
            )

            hires_settings = state.sampling_plan.noise_settings
            rng_hr = ImageRNG(
                (latents.shape[1], latents.shape[2], latents.shape[3]),
                state.sampling_plan.seeds,
                subseeds=state.sampling_plan.subseeds,
                subseed_strength=state.sampling_plan.subseed_strength,
                seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
                seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
                settings=hires_settings,
            )
            noise = rng_hr.next().to(latents)
            start_index = start_at_step_from_denoise(denoise=denoise, steps=int(processing.steps))

            hires_plan = replace(
                state.sampling_plan,
                sampler_name=hires_sampler,
                scheduler_name=hires_scheduler,
                steps=int(processing.steps),
                guidance_scale=float(processing.guidance_scale),
            )

            # Recompute conditioning for hires pass with updated width/height/targets (SDXL parity).
            cond_hr, uncond_hr = self._compute_conditioning(processing, hires_prompt_context)
            if cond_hr is not None:
                state.payload = ConditioningPayload(conditioning=cond_hr, unconditional=uncond_hr)
                self._log_conditioning(cond_hr, uncond_hr)

            samples = execute_sampling(
                processing,
                hires_plan,
                state.payload,
                hires_prompt_context,
                hires_prompt_context.loras,
                hires_prompt_context.controls,
                rng=rng_hr,
                noise=noise,
                image_conditioning=image_conditioning,
                init_latent=latents,
                start_at_step=start_index,
            )

            if processing.hires_refiner is not None:
                samples = self._apply_refiner_stage(
                    HiresRefinerStage(processing.hires_refiner),
                    processing,
                    hires_prompt_context,
                    hires_settings,
                    samples,
                )
            return samples
        finally:
            processing.prompts = original_attrs["prompts"]
            processing.negative_prompts = original_attrs["negative_prompts"]
            processing.width = original_attrs["width"]
            processing.height = original_attrs["height"]
            processing.guidance_scale = original_attrs["guidance_scale"]
            processing.cfg_scale = processing.guidance_scale
            processing.steps = original_attrs["steps"]
            processing.sampler_name = original_attrs["sampler_name"]
            processing.scheduler = original_attrs["scheduler"]
            processing.sampler = original_attrs["sampler"]
            processing.prepare_prompt_data()

    @pipeline_trace
    def _maybe_run_refiner_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        refiner_cfg = getattr(processing, "refiner", None)
        if refiner_cfg is None:
            overrides = getattr(processing, "override_settings", {}) or {}
            raw = overrides.get("refiner") if isinstance(overrides, dict) else None
            if isinstance(raw, dict):
                refiner_cfg = RefinerConfig(
                    enabled=bool(raw.get("enable", False)) and int(raw.get("switch_at_step", 0) or 0) > 0,
                    swap_at_step=int(raw.get("switch_at_step", 0) or 0),
                    cfg=float(raw.get("cfg", getattr(processing, "guidance_scale", 7.0))),
                    seed=int(raw.get("seed", -1)),
                    model=str(raw.get("model") or "").strip() or None,
                    vae=str(raw.get("vae") or "").strip() or None,
                )

        stage = GlobalRefinerStage(refiner_cfg)
        return self._apply_refiner_stage(
            stage,
            processing,
            state.prompt_context,
            state.sampling_plan.noise_settings,
            samples,
        )

    # ------------------------------------------------------------------ helpers
    @pipeline_trace
    def _prepare_first_pass_from_image(
        self, processing: CodexProcessingTxt2Img
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

    @pipeline_trace
    def _build_hires_plan(self, processing: CodexProcessingTxt2Img) -> HiResPlan | None:
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
        resize_x = int(hi_cfg.resize_x) if hi_cfg.resize_x is not None else 0
        resize_y = int(hi_cfg.resize_y) if hi_cfg.resize_y is not None else 0
        if resize_x < 0:
            raise ValueError("Hires is enabled but 'hires.resize_x' must be >= 0 (0 means fallback to scale).")
        if resize_y < 0:
            raise ValueError("Hires is enabled but 'hires.resize_y' must be >= 0 (0 means fallback to scale).")

        scale = float(hi_cfg.scale) if hi_cfg.scale is not None else None
        if resize_x > 0:
            target_width = resize_x
        else:
            if scale is None or scale <= 0.0:
                raise ValueError(
                    "Hires is enabled but neither 'hires.resize_x' nor a valid positive 'hires.scale' is set. "
                    "Provide explicit dimensions or a scale."
                )
            target_width = int(processing.width * scale)

        if resize_y > 0:
            target_height = resize_y
        else:
            if scale is None or scale <= 0.0:
                raise ValueError(
                    "Hires is enabled but neither 'hires.resize_y' nor a valid positive 'hires.scale' is set. "
                    "Provide explicit dimensions or a scale."
                )
            target_height = int(processing.height * scale)

        second_pass_steps = int(hi_cfg.second_pass_steps) if hi_cfg.second_pass_steps is not None else 0
        if second_pass_steps < 0:
            raise ValueError("Hires is enabled but 'hires.steps' must be >= 0 (0 means reuse first-pass steps).")
        steps = second_pass_steps if second_pass_steps > 0 else int(processing.steps)
        if steps <= 0:
            raise ValueError("Hires is enabled but resolved 'steps' must be > 0.")
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

    # ------------------------------------------------------------------ diagnostics
    def _sd_model_device_info(
        self, processing: CodexProcessingTxt2Img
    ) -> tuple[torch.device | None, torch.dtype | None]:
        model = getattr(processing, "sd_model", None)
        if model is None:
            return None, None
        tensor = None
        try:
            params = model.parameters()
            if params is not None:
                tensor = next(params)
        except StopIteration:
            tensor = None
        except Exception:
            tensor = None
        if tensor is None:
            candidate = getattr(model, "weight", None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.device, tensor.dtype
        device_attr = getattr(model, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr, None
        return None, None
