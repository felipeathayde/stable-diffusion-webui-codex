"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling driver (native + optional k-diffusion) for diffusion runtimes.
Selects sampler implementations from specs, compiles conditioning, handles cancellation/precision fallback, and runs the sampling loop
while emitting timeline/diagnostic hooks.

Symbols (top-level; keep in sync; no ghosts):
- `_SamplingCancelled` (exception): Raised when an in-flight sampling run is cancelled (checked via backend state).
- `_raise_if_cancelled` (function): Checks cancellation state and raises `_SamplingCancelled` when requested.
- `_PrecisionFallbackRequest` (exception): Signals the caller to retry sampling with a different precision policy.
- `_kd_sampler_callable` (function): Wraps a k-diffusion sampler call with the compiled cond/uncond and CFG scale plumbing.
- `_KDiffusionModel` (class): Adapter that exposes the model interface expected by k-diffusion samplers.
- `_run_kdiffusion_sampler` (function): Runs the selected k-diffusion sampler with the current sampling context and hooks.
- `CodexSampler` (class): Main sampler driver; builds `SamplingContext`, resolves sampler specs (native vs k-diffusion), runs the inner loop,
  and integrates memory-management/timeline diagnostics.
"""

from __future__ import annotations

# tags: sampling, diagnostics

from typing import Any, Optional, Callable, List
import math
import logging

import torch

from apps.backend.infra.config.env_flags import env_flag, env_int

from .inner_loop import sampling_function_inner, sampling_prepare, sampling_cleanup
from .condition import compile_conditions
from .context import SamplingContext, build_sampling_context
from .registry import get_sampler_spec
from ...core.state import state as backend_state
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.diagnostics.timeline import timeline


try:
    # Optional dependency; when missing, k-diffusion-backed samplers stay disabled
    import k_diffusion.sampling as _KD_SAMPLING  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - absence is expected on minimal installs
    _KD_SAMPLING = None


_KD_MAPPING = {
    SamplerKind.EULER: "sample_euler",
    SamplerKind.EULER_A: "sample_euler_ancestral",
    SamplerKind.EULER_CFG_PP: "sample_euler_cfg_pp",
    SamplerKind.EULER_A_CFG_PP: "sample_euler_ancestral_cfg_pp",
    SamplerKind.HEUN: "sample_heun",
    SamplerKind.HEUNPP2: "sample_heunpp2",
    SamplerKind.LMS: "sample_lms",
    SamplerKind.DDIM: "sample_ddim",
    SamplerKind.DDIM_CFG_PP: "sample_ddim_cfgpp",
    SamplerKind.PLMS: "sample_plms",
    SamplerKind.PNDM: "sample_pndm",
    SamplerKind.DPM2: "sample_dpm_2",
    SamplerKind.DPM2_ANCESTRAL: "sample_dpm_2_ancestral",
    SamplerKind.DPM_FAST: "sample_dpm_fast",
    SamplerKind.DPM_ADAPTIVE: "sample_dpm_adaptive",
    SamplerKind.DPM2S_ANCESTRAL: "sample_dpmpp_2s_ancestral",
    SamplerKind.DPM2S_ANCESTRAL_CFG_PP: "sample_dpmpp_2s_ancestral_cfg_pp",
    SamplerKind.DPM2M: "sample_dpmpp_2m",
    SamplerKind.DPM2M_CFG_PP: "sample_dpmpp_2m_cfg_pp",
    SamplerKind.DPM2M_SDE: "sample_dpmpp_2m_sde",
    SamplerKind.DPM2M_SDE_HEUN: "sample_dpmpp_2m_sde_heun",
    SamplerKind.DPM2M_SDE_GPU: "sample_dpmpp_2m_sde_gpu",
    SamplerKind.DPM2M_SDE_HEUN_GPU: "sample_dpmpp_2m_sde_heun_gpu",
    SamplerKind.DPM_SDE: "sample_dpmpp_sde",
    SamplerKind.DPM3M_SDE: "sample_dpmpp_3m_sde",
    SamplerKind.DPM3M_SDE_GPU: "sample_dpmpp_3m_sde_gpu",
    SamplerKind.DDPM: "extra:sample_ddpm",
    SamplerKind.LCM: "sample_lcm",
    SamplerKind.IPNDM: "sample_ipndm",
    SamplerKind.IPNDM_V: "sample_ipndm_v",
    SamplerKind.DEIS: "sample_deis",
    SamplerKind.UNI_PC: "extra:sample_unipc",
    SamplerKind.UNI_PC_BH2: "extra:sample_unipc_bh2",
    SamplerKind.RES_MULTISTEP: "sample_res_multistep",
    SamplerKind.RES_MULTISTEP_CFG_PP: "sample_res_multistep_cfg_pp",
    SamplerKind.RES_MULTISTEP_ANCESTRAL: "sample_res_multistep_ancestral",
    SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP: "sample_res_multistep_ancestral_cfg_pp",
    SamplerKind.GRADIENT_ESTIMATION: "sample_gradient_estimation",
    SamplerKind.GRADIENT_ESTIMATION_CFG_PP: "sample_gradient_estimation_cfg_pp",
    SamplerKind.ER_SDE: "sample_er_sde",
    SamplerKind.SEEDS_2: "sample_seeds_2",
    SamplerKind.SEEDS_3: "sample_seeds_3",
    SamplerKind.SA_SOLVER: "sample_sa_solver",
    SamplerKind.SA_SOLVER_PECE: "sample_sa_solver_pece",
    SamplerKind.RESTART: "extra:restart_sampler",
}


class _SamplingCancelled(Exception):
    """Signal that sampling was cancelled externally."""


def _raise_if_cancelled() -> None:
    if backend_state.should_stop:
        raise _SamplingCancelled("cancelled")



_LMS_COEFFS = {
    1: (1.0,),
    2: (3.0 / 2.0, -1.0 / 2.0),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


class _PrecisionFallbackRequest(Exception):
    """Internal control flow exception used to trigger a sampling retry."""


def _kd_sampler_callable(model, compiled_cond, compiled_uncond, cfg_scale):
    def _fn(x: torch.Tensor, sigma_hat: torch.Tensor, **extra_args):
        sigma_batch = sigma_hat if sigma_hat.ndim == 1 else sigma_hat.view(-1)
        return sampling_function_inner(
            model,
            x,
            sigma_batch,
            compiled_uncond,
            compiled_cond,
            cfg_scale,
            getattr(model, "model_options", {}),
            seed=None,
            return_full=False,
        )
    return _fn


class _KDiffusionModel:
    """Adapter exposing .inner_model for k-diffusion samplers."""

    def __init__(self, inner_model, fn):
        self.inner_model = inner_model
        self._fn = fn

    def __call__(self, x, sigma, **extra_args):
        return self._fn(x, sigma, **extra_args)


def _run_kdiffusion_sampler(
    sampler_kind: SamplerKind,
    sampler_fn_name: str,
    *,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    compiled_cond,
    compiled_uncond,
    cfg_scale: float,
    progress_bar,
    logger: logging.Logger,
    log_enabled: bool,
    preview_callback: Optional[Callable[[torch.Tensor, int, int], None]],
    total_steps: int,
    tick: Optional[Callable[[int], None]] = None,
    preview_interval: int = 0,
) -> torch.Tensor:
    sampler_fn = None
    if sampler_fn_name.startswith("extra:"):
        from apps.backend.runtime.k_diffusion import k_diffusion_extra as kd_extra

        extra_name = sampler_fn_name.split(":", 1)[1]
        sampler_fn = getattr(kd_extra, extra_name, None)
    else:
        if _KD_SAMPLING is None:
            raise RuntimeError("k-diffusion is required for sampler execution but is unavailable")
        sampler_fn = getattr(_KD_SAMPLING, sampler_fn_name, None)

    if sampler_fn is None:
        raise NotImplementedError(f"Sampler '{sampler_kind.value}' not yet ported (missing {sampler_fn_name})")
    kd_model = _KDiffusionModel(model, _kd_sampler_callable(model, compiled_cond, compiled_uncond, cfg_scale))

    step_counter = {"i": 0}

    def _callback(payload):
        idx = int(payload.get("i", step_counter["i"]))
        step_counter["i"] = idx
        timeline.exit("sampling", f"step[{idx}]")
        timeline.enter("sampling", f"step[{idx+1}]")
        if tick is not None:
            tick(idx + 1)
        if preview_callback is not None and preview_interval > 0:
            try:
                if ((idx + 1) % preview_interval == 0) or (idx + 1) == total_steps:
                    preview_callback(payload.get("denoised"), idx + 1, total_steps)
            except Exception:
                pass
        if progress_bar is not None:
            progress_bar.update(1)

    # Timeline: mark first step entry
    timeline.enter("sampling", "step[0]")
    
    out = sampler_fn(
        kd_model,
        x,
        sigmas,
        extra_args={"model_options": getattr(model, "model_options", {})},
        callback=_callback,
        disable=not log_enabled,
    )
    
    # Timeline: close final step
    timeline.exit("sampling", f"step[{step_counter['i']}]")

    if progress_bar is not None:
        progress_bar.close()

    if log_enabled:
        logger.info("k-diffusion sampler=%s finished steps=%d", sampler_kind.value, total_steps)
    return out



class CodexSampler:
    """Minimal native sampler for txt2img using an Euler ODE loop.

    - Uses the model's predictor to derive a sigma schedule from sigma_max→sigma_min.
    - Calls `sampling_function_inner` (CFG and condition assembly) each step.
    - Updates `backend_state` for progress reporting.
    - No ancestral noise yet; samplers other than Euler will be added iteratively.
    """

    def __init__(self, sd_model: Any, *, algorithm: str | None = None) -> None:
        self.sd_model = sd_model
        self.algorithm = (algorithm or "euler a").strip().lower()
        self._logger = logging.getLogger(__name__ + ".CodexSampler")
        self._log_enabled = env_flag("CODEX_LOG_SAMPLER", default=False)
        self._log_sigmas = env_flag("CODEX_LOG_SIGMAS", default=False)

    def _summarize_sigmas(self, sigmas: torch.Tensor, *, window: int = 6) -> str:
        try:
            values = [float(v) for v in sigmas.detach().cpu().tolist()]
        except Exception:
            return "<unavailable>"
        if len(values) <= window * 2:
            return ",".join(f"{v:.6g}" for v in values)
        head = ",".join(f"{v:.6g}" for v in values[:window])
        tail = ",".join(f"{v:.6g}" for v in values[-window:])
        return f"{head},...,{tail}"

    def _rebind_unet_precision(self, dtype: torch.dtype) -> None:
        denoiser = self.sd_model.codex_objects.denoiser
        model = getattr(denoiser, "model", None)
        if model is None:
            return
        previous = getattr(model, "computation_dtype", None)
        if hasattr(model, "computation_dtype"):
            model.computation_dtype = dtype
        diffusion_model = getattr(model, "diffusion_model", None)
        if diffusion_model is not None:
            diffusion_model.to(dtype=dtype)
        self._logger.info(
            "Diffusion core precision updated: %s -> %s",
            str(previous),
            str(dtype),
        )

    @torch.inference_mode()
    def sample(
        self,
        processing: Any,
        noise: torch.Tensor,
        cond: Any,
        uncond: Optional[Any],
        image_conditioning: Optional[torch.Tensor] = None,
        *,
        init_latent: Optional[torch.Tensor] = None,
        start_at_step: int | None = None,
        preview_callback: Optional[Callable[[torch.Tensor, int, int], None]] = None,
        context: SamplingContext | None = None,
    ) -> torch.Tensor:
        base_noise = noise.detach().clone()
        base_context = context

        spec = get_sampler_spec(self.algorithm)

        while True:
            denoiser = self.sd_model.codex_objects.denoiser
            model = denoiser.model

            steps = int(getattr(processing, "steps", 20))
            cfg_scale = float(getattr(processing, "cfg_scale", 7.0))
            
            # Flux Dev uses distilled CFG via guidance embedding - disable traditional CFG
            # to avoid double scaling which corrupts the output (sand texture artifact)
            if hasattr(self.sd_model, "use_distilled_cfg_scale") and self.sd_model.use_distilled_cfg_scale:
                if cfg_scale != 1.0:
                    self._logger.info("[flux] Distilled CFG active: forcing cfg_scale=1.0 (was %.2f)", cfg_scale)
                cfg_scale = 1.0

            if steps <= 0:
                raise ValueError("steps must be >= 1")
            if noise.ndim != 4:
                raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

            target_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            # Diffusers flow pipelines (Flux/Z-Image) keep latents in fp32 for the
            # scheduler integration even when the core runs in bf16/fp16. Keeping
            # x/eps in low precision can destabilize the tail and produce "noise soup".
            pred_type = getattr(getattr(model, "predictor", None), "prediction_type", None)
            if isinstance(pred_type, str) and pred_type.lower() == "const":
                if target_dtype in (torch.float16, torch.bfloat16):
                    self._logger.info("[flow] Forcing sampling latents to float32 (was %s)", str(target_dtype))
                target_dtype = torch.float32
            noise = base_noise.to(dtype=target_dtype)

            if init_latent is not None and init_latent.dtype != noise.dtype:
                init_latent = init_latent.to(dtype=noise.dtype)

            progress_bar = None
            retry = False
            prepared = False
            state_started = False
            active_context = base_context

            try:
                sampling_prepare(denoiser, noise)
                prepared = True

                scheduler_name = getattr(processing, "scheduler", None)
                if scheduler_name in (None, ""):
                    scheduler_name = spec.default_scheduler
                if not isinstance(scheduler_name, str) or not scheduler_name:
                    raise ValueError("Scheduler name must be a non-empty string")
                if not spec.is_supported_scheduler(scheduler_name):
                    raise ValueError(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{spec.name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    )
                if active_context is None:
                    active_context = build_sampling_context(
                        self.sd_model,
                        sampler_name=self.algorithm,
                        scheduler_name=scheduler_name,
                        steps=steps,
                        noise_source=getattr(processing, "noise_source", None),
                        eta_noise_seed_delta=int(getattr(processing, "eta_noise_seed_delta", 0) or 0),
                        height=(int(getattr(processing, "height", 0) or 0) or None),
                        width=(int(getattr(processing, "width", 0) or 0) or None),
                        device=noise.device,
                        dtype=noise.dtype,
                        predictor=model,
                        is_sdxl=bool(getattr(getattr(self.sd_model, "engine", None), "is_sdxl", False)),
                    )

                # Keep sigma ladder in fp32 for numeric stability; casting to bf16/fp16
                # quantizes the schedule and can produce severe quality regressions.
                sigmas = active_context.sigmas.to(device=noise.device, dtype=torch.float32)
                steps = active_context.steps

                if self._log_sigmas or self._log_enabled:
                    schedule_first = float(sigmas[0]) if len(sigmas) > 0 else float("nan")
                    schedule_last = float(sigmas[-1]) if len(sigmas) > 0 else float("nan")
                    schedule_summary = self._summarize_sigmas(sigmas)
                    sigma_min_val = float("nan") if active_context.sigma_min is None else float(active_context.sigma_min)
                    sigma_max_val = float("nan") if active_context.sigma_max is None else float(active_context.sigma_max)
                    self._logger.info(
                        "sigma schedule len=%d predict_min=%.6g predict_max=%.6g first=%.6g last=%.6g ladder=%s",
                        len(sigmas) - 1,
                        sigma_min_val,
                        sigma_max_val,
                        schedule_first,
                        schedule_last,
                        schedule_summary,
                    )

                start_idx = int(start_at_step or 0)
                start_idx = max(0, min(start_idx, steps - 1))
                sigmas_run = sigmas[start_idx:]
                if init_latent is not None:
                    x = init_latent + float(sigmas_run[0]) * noise
                else:
                    # Keep x in the core dtype (bf16/fp16) while preserving the sigma ladder precision.
                    sigma0 = sigmas_run[:1].to(dtype=noise.dtype)
                    x = model.predictor.noise_scaling(sigma0, noise, torch.zeros_like(noise))

                if self._log_enabled:
                    try:
                        smax = float(sigmas[0].item()) if hasattr(sigmas[0], "item") else float(sigmas[0])
                        smin = float(sigmas[-1].item()) if hasattr(sigmas[-1], "item") else float(sigmas[-1])
                        head = [float(v) for v in sigmas[: min(4, len(sigmas))].detach().cpu().tolist()]
                    except Exception:
                        smax = float("nan")
                        smin = float("nan")
                        head = []
                    pred_type = getattr(model.predictor, "prediction_type", None)
                    sigma_data = getattr(model.predictor, "sigma_data", None)
                    self._logger.info(
                        "sampler algorithm=%s scheduler=%s steps=%d cfg_scale=%.4g prediction=%s sigma_max=%.6g sigma_min=%.6g sigma_data=%s head=%s",
                        self.algorithm,
                        active_context.scheduler_name,
                        steps,
                        float(cfg_scale),
                        pred_type or getattr(active_context, "prediction_type", None) or "<unknown>",
                        smax,
                        smin,
                        f"{float(sigma_data):.4g}" if sigma_data is not None else "n/a",
                        head,
                    )

                compiled_cond = compile_conditions(cond)
                compiled_uncond = compile_conditions(uncond) if uncond is not None else None
                log_cfg_delta = False
                cfg_delta_steps = 0
                if self._log_enabled:
                    log_cfg_delta = env_flag("CODEX_LOG_CFG_DELTA", default=False)
                    if log_cfg_delta:
                        cfg_delta_steps = env_int("CODEX_LOG_CFG_DELTA_N", default=2, min_value=0)

                if isinstance(image_conditioning, torch.Tensor):
                    if (
                        image_conditioning.shape[0] == noise.shape[0]
                        and image_conditioning.shape[2] == noise.shape[2]
                        and image_conditioning.shape[3] == noise.shape[3]
                    ):
                        from .condition import Condition

                        for entry in compiled_cond:
                            entry["model_conds"]["c_concat"] = Condition(image_conditioning)
                        if compiled_uncond is not None:
                            for entry in compiled_uncond:
                                entry["model_conds"]["c_concat"] = Condition(image_conditioning)

                backend_state.start(job_count=1, sampling_steps=steps - start_idx)
                state_started = True

                strict = True
                import time as _time

                preview_interval = active_context.preview_interval
                t0 = _time.perf_counter()
                use_progress = active_context.enable_progress
                if use_progress:
                    from tqdm.auto import tqdm

                    progress_bar = tqdm(total=steps - start_idx, desc="sampling", leave=False)

                sampler_kind = active_context.sampler_kind

                # Default to native sampler; enable k-diffusion only when explicitly requested.
                use_kd = False
                if use_kd and sampler_kind in _KD_MAPPING and _KD_SAMPLING is not None:
                    kd_name = _KD_MAPPING[sampler_kind]

                    compiled_cond = compile_conditions(cond)
                    compiled_uncond = compile_conditions(uncond) if uncond is not None else None

                    backend_state.start(job_count=1, sampling_steps=len(sigmas_run) - 1)
                    state_started = True

                    if self._log_sigmas or self._log_enabled:
                        schedule_first = float(sigmas_run[0]) if len(sigmas_run) > 0 else float("nan")
                        schedule_last = float(sigmas_run[-1]) if len(sigmas_run) > 0 else float("nan")
                        schedule_summary = self._summarize_sigmas(sigmas_run)
                        sigma_min_val = float("nan") if active_context.sigma_min is None else float(active_context.sigma_min)
                        sigma_max_val = float("nan") if active_context.sigma_max is None else float(active_context.sigma_max)
                        self._logger.info(
                            "sigma schedule len=%d predict_min=%.6g predict_max=%.6g first=%.6g last=%.6g ladder=%s",
                            len(sigmas_run) - 1,
                            sigma_min_val,
                            sigma_max_val,
                            schedule_first,
                            schedule_last,
                            schedule_summary,
                        )

                    if self._log_enabled:
                        head = []
                        try:
                            head = [float(v) for v in sigmas_run[: min(4, len(sigmas_run))].detach().cpu().tolist()]
                        except Exception:
                            head = []
                        self._logger.info(
                            "sampler algorithm=%s scheduler=%s steps=%d cfg_scale=%.4g head=%s",
                            sampler_kind.value,
                            active_context.scheduler_name,
                            len(sigmas_run) - 1,
                            float(cfg_scale),
                            head,
                        )

                    progress_bar = None
                    use_progress = active_context.enable_progress
                    if use_progress:
                        from tqdm.auto import tqdm

                        progress_bar = tqdm(total=len(sigmas_run) - 1, desc="sampling", leave=False)

                    x = _run_kdiffusion_sampler(
                        sampler_kind,
                        kd_name,
                        model=model,
                        x=x,
                        sigmas=sigmas_run,
                        compiled_cond=compiled_cond,
                        compiled_uncond=compiled_uncond,
                        cfg_scale=cfg_scale,
                        progress_bar=progress_bar,
                        logger=self._logger,
                        log_enabled=self._log_enabled,
                        preview_callback=preview_callback,
                        total_steps=len(sigmas_run) - 1,
                        tick=lambda step: (_raise_if_cancelled() or backend_state.tick(sampling_step=step)),
                        preview_interval=active_context.preview_interval,
                    )

                    sampling_cleanup(denoiser)
                    prepared = False
                    backend_state.end()
                    state_started = False
                    return x

                if self._log_enabled:
                    head = []
                    try:
                        head = [float(v) for v in sigmas_run[: min(4, len(sigmas_run))].detach().cpu().tolist()]
                    except Exception:
                        head = []
                    self._logger.info(
                        "sampler algorithm=%s scheduler=%s steps=%d cfg_scale=%.4g head=%s",
                        sampler_kind.value,
                        active_context.scheduler_name,
                        len(sigmas_run) - 1,
                        float(cfg_scale),
                        head,
                    )

                old_denoised: Optional[torch.Tensor] = None
                t_prev: float | None = None
                h_prev: float | None = None
                eps_history: List[torch.Tensor] = []

                for i in range(start_idx, steps):
                    if backend_state.should_stop:
                        raise _SamplingCancelled("cancelled")
                    sigma = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    sigma_batch = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=torch.float32)

                    if log_cfg_delta and (i - start_idx) < cfg_delta_steps:
                        denoised, cond_pred, uncond_pred = sampling_function_inner(
                            model,
                            x,
                            sigma_batch,
                            compiled_uncond,
                            compiled_cond,
                            cfg_scale,
                            denoiser.model_options,
                            seed=None,
                            return_full=True,
                        )
                        cfg1_optimization = math.isclose(cfg_scale, 1.0) and not denoiser.model_options.get(
                            "disable_cfg1_optimization", False
                        )
                        if compiled_uncond is None or cfg1_optimization:
                            self._logger.info(
                                "cfg-delta step=%d/%d sigma=%.6g cfg_scale=%.4g uncond_used=%s",
                                i + 1,
                                steps,
                                float(sigma),
                                float(cfg_scale),
                                False,
                            )
                        else:
                            try:
                                delta_abs_mean = float((cond_pred - uncond_pred).detach().float().abs().mean().item())
                            except Exception:
                                delta_abs_mean = float("nan")
                            self._logger.info(
                                "cfg-delta step=%d/%d sigma=%.6g cfg_scale=%.4g delta_abs_mean=%.6g",
                                i + 1,
                                steps,
                                float(sigma),
                                float(cfg_scale),
                                delta_abs_mean,
                            )
                    else:
                        denoised = sampling_function_inner(
                            model,
                            x,
                            sigma_batch,
                            compiled_uncond,
                            compiled_cond,
                            cfg_scale,
                            denoiser.model_options,
                            seed=None,
                            return_full=False,
                        )

                    eps = (x - denoised) / max(float(sigma), 1e-8)
                    if strict and (torch.isnan(eps).any() or torch.isnan(denoised).any()):
                        reason = f"NaN detected at sampling step {i + 1}"
                        self._logger.warning(
                            "NaN encountered at step %d with dtype=%s; attempting precision fallback.",
                            i + 1,
                            str(getattr(model, "computation_dtype", x.dtype)),
                        )
                        next_dtype = memory_management.manager.report_precision_failure(
                            DeviceRole.CORE,
                            location=f"sampler.step_{i + 1}",
                            reason=reason,
                        )
                        if next_dtype is None:
                            hint = memory_management.manager.precision_hint(DeviceRole.CORE)
                            raise RuntimeError(
                                f"Diffusion core produced NaNs at step {i + 1} on {noise.device} with dtype {getattr(model, 'computation_dtype', x.dtype)}. {hint}"
                            )
                        self._rebind_unet_precision(next_dtype)
                        retry = True
                        raise _PrecisionFallbackRequest

                    eps_history.append(eps.detach())
                    if len(eps_history) > 4:
                        eps_history.pop(0)

                    if sampler_kind is SamplerKind.EULER:
                        x = x - (float(sigma) - float(sigma_next)) * eps
                    elif sampler_kind is SamplerKind.EULER_A:
                        sigma = float(sigma)
                        sigma_next = float(sigma_next)
                        if sigma_next <= 0.0:
                            x = denoised
                        else:
                            sigma_up_sq = max(sigma_next**2 * (sigma**2 - sigma_next**2) / max(sigma**2, 1e-8), 0.0)
                            sigma_up = sigma_up_sq ** 0.5
                            sigma_down = (max(sigma_next**2 - sigma_up_sq, 0.0)) ** 0.5
                            x = denoised + sigma_down * eps
                            noise = torch.randn_like(x)
                            x = x + sigma_up * noise
                    elif sampler_kind is SamplerKind.DPM2M:
                        # DPM-Solver++(2M) in log-sigma time (matches k-diffusion sample_dpmpp_2m).
                        sigma_f = float(sigma)
                        sigma_next_f = float(sigma_next)
                        if sigma_next_f <= 0.0:
                            x = denoised
                            old_denoised = denoised.detach()
                            t_prev = None
                        else:
                            t = -math.log(max(sigma_f, 1e-12))
                            t_next = -math.log(max(sigma_next_f, 1e-12))
                            h = t_next - t
                            if old_denoised is None or t_prev is None:
                                x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised
                            else:
                                h_last = t - t_prev
                                r = h_last / h if abs(h) > 1e-12 else 1.0
                                denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                                x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised_d
                            old_denoised = denoised.detach()
                            t_prev = t
                    elif sampler_kind is SamplerKind.DPM2M_SDE:
                        # DPM-Solver++(2M) SDE (midpoint) in log-sigma time.
                        # This is a conservative native approximation (no BrownianTree),
                        # but matches the core update form used by k-diffusion.
                        sigma_f = float(sigma)
                        sigma_next_f = float(sigma_next)
                        if sigma_next_f <= 0.0:
                            x = denoised
                            old_denoised = denoised.detach()
                            h_prev = None
                        else:
                            t = -math.log(max(sigma_f, 1e-12))
                            s = -math.log(max(sigma_next_f, 1e-12))
                            h = s - t
                            eta = 1.0
                            s_noise = 1.0
                            eta_h = eta * h
                            phi = -math.expm1(-h - eta_h)  # 1 - exp(-h-eta*h)
                            x = (sigma_next_f / sigma_f) * math.exp(-eta_h) * x + phi * denoised
                            if old_denoised is not None and h_prev is not None:
                                r = h_prev / h if abs(h) > 1e-12 else 1.0
                                x = x + 0.5 * phi * (1.0 / r) * (denoised - old_denoised)
                            if eta != 0.0:
                                noise_scale = sigma_next_f * math.sqrt(max(-math.expm1(-2.0 * eta_h), 0.0)) * s_noise
                                x = x + torch.randn_like(x) * noise_scale
                            old_denoised = denoised.detach()
                            h_prev = h
                    elif sampler_kind is SamplerKind.DDIM:
                        x = denoised + float(sigma_next) * eps
                    elif sampler_kind in (SamplerKind.PLMS, SamplerKind.PNDM):
                        order = min(len(eps_history), 4)
                        coeffs = _LMS_COEFFS[order]
                        derivative = torch.zeros_like(x)
                        for idx_coeff, coeff in enumerate(coeffs):
                            derivative = derivative + coeff * eps_history[-(idx_coeff + 1)]
                        delta = float(sigma) - float(sigma_next)
                        x = x - delta * derivative
                    elif sampler_kind is SamplerKind.UNI_PC:
                        delta = float(sigma) - float(sigma_next)
                        x_pred = x - delta * eps
                        sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=torch.float32)
                        denoised_next = sampling_function_inner(
                            model,
                            x_pred,
                            sigma_next_batch,
                            compiled_uncond,
                            compiled_cond,
                            cfg_scale,
                            denoiser.model_options,
                            seed=None,
                            return_full=False,
                        )
                        eps_next = (x_pred - denoised_next) / max(float(sigma_next), 1e-8)
                        x = x - delta * 0.5 * (eps + eps_next)
                    elif sampler_kind is SamplerKind.UNI_PC_BH2:
                        # Reuse the UniPC two-stage update as a BH2 variant placeholder.
                        delta = float(sigma) - float(sigma_next)
                        x_pred = x - delta * eps
                        sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=torch.float32)
                        denoised_next = sampling_function_inner(
                            model,
                            x_pred,
                            sigma_next_batch,
                            compiled_uncond,
                            compiled_cond,
                            cfg_scale,
                            denoiser.model_options,
                            seed=None,
                            return_full=False,
                        )
                        eps_next = (x_pred - denoised_next) / max(float(sigma_next), 1e-8)
                        x = x - delta * 0.5 * (eps + eps_next)
                    else:
                        raise NotImplementedError(f"Sampler '{sampler_kind.value}' is not implemented natively yet")

                    if preview_callback is not None and (preview_interval > 0 and ((i + 1) % preview_interval == 0) or (i + 1) == steps):
                        try:
                            preview_callback(denoised.detach(), i + 1, steps)
                        except Exception:
                            pass

                    if self._log_enabled and (i == 0 or (i + 1) == steps or (i + 1) % max(1, steps // 5) == 0):
                        eps_norm = float(eps.norm().item()) if hasattr(eps, "norm") else float("nan")
                        den_norm = float(denoised.norm().item()) if hasattr(denoised, "norm") else float("nan")
                        self._logger.info(
                            "step=%d/%d sigma=%.6g->%.6g norm(x)=%.4f norm(eps)=%.4f norm(den)=%.4f dt=%.2fms",
                            i + 1,
                            steps,
                            float(sigma),
                            float(sigma_next),
                            float(x.norm().item()),
                            eps_norm,
                            den_norm,
                            (_time.perf_counter() - t0) * 1000.0,
                        )
                        t0 = _time.perf_counter()

                    if progress_bar is not None:
                        progress_bar.update(1)

                    backend_state.tick(sampling_step=i + 1)

                if progress_bar is not None:
                    progress_bar.close()
                    progress_bar = None

                sampling_cleanup(denoiser)
                prepared = False

                backend_state.end()
                state_started = False

                return x
            except _PrecisionFallbackRequest:
                self._logger.warning("Precision fallback requested for diffusion core; retrying with next dtype.")
            except _SamplingCancelled:
                self._logger.info("Sampling cancelled by request; aborting current run.")
                raise RuntimeError("cancelled")
            finally:
                if progress_bar is not None:
                    progress_bar.close()
                if prepared:
                    sampling_cleanup(denoiser)
                if state_started:
                    backend_state.end()
                backend_state.clear_flags()

            if retry:
                memory_management.manager.soft_empty_cache(force=True)
                continue



__all__ = ["CodexSampler"]
