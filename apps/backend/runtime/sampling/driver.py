"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling driver (native-only) for diffusion runtimes.
Selects sampler implementations from specs, compiles conditioning, handles cancellation/precision fallback, and runs the sampling loop
while emitting timeline/diagnostic hooks (and optional global profiling sections via `CODEX_PROFILE`), including native ER-SDE stage updates
with strict runtime option validation (`solver_type`, `max_stage`, `eta`, `s_noise`).

Symbols (top-level; keep in sync; no ghosts):
- `_SamplingCancelled` (exception): Raised when an in-flight sampling run is cancelled (checked via backend state).
- `_raise_if_cancelled` (function): Checks cancellation state and raises `_SamplingCancelled` when requested.
- `_PrecisionFallbackRequest` (exception): Signals the caller to retry sampling with a different precision policy.
- `CodexSampler` (class): Main sampler driver; builds `SamplingContext`, resolves sampler specs, runs the native sampler loop, and integrates
  memory-management/timeline diagnostics.
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
from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_sampling_residency
from apps.backend.runtime.diagnostics.timeline import timeline
from apps.backend.runtime.diagnostics.profiler import profiler

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

    @staticmethod
    def _normalize_er_sde_solver_type(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ER-SDE option 'solver_type' must be a string")
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        solver_map = {
            "er sde": "er_sde",
            "reverse time sde": "reverse_time_sde",
            "ode": "ode",
        }
        solver_type = solver_map.get(normalized)
        if solver_type is None:
            raise ValueError(
                "ER-SDE option 'solver_type' must be one of: ER-SDE, Reverse-time SDE, ODE"
            )
        return solver_type

    @staticmethod
    def _resolve_er_sde_runtime_params(er_sde_options: Any) -> dict[str, Any]:
        allowed_keys = {"solver_type", "max_stage", "eta", "s_noise"}
        if er_sde_options is None:
            payload: dict[str, Any] = {}
        elif isinstance(er_sde_options, dict):
            raw_payload = dict(er_sde_options)
            unknown_keys = sorted(set(raw_payload.keys()) - allowed_keys)
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {key: value for key, value in raw_payload.items() if value is not None}
        else:
            attr_keys: set[str] = set()
            if hasattr(er_sde_options, "__dict__"):
                attr_keys.update(str(key) for key in vars(er_sde_options).keys())
            slots_value = getattr(type(er_sde_options), "__slots__", ())
            if isinstance(slots_value, str):
                slots_iterable = (slots_value,)
            else:
                slots_iterable = tuple(slots_value) if isinstance(slots_value, (tuple, list)) else ()
            attr_keys.update(str(key) for key in slots_iterable)
            unknown_keys = sorted(
                key for key in attr_keys if key and not key.startswith("_") and key not in allowed_keys
            )
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {}
            for key in allowed_keys:
                value = getattr(er_sde_options, key, None)
                if value is not None:
                    payload[key] = value

        solver_raw = payload.get("solver_type", "er_sde")
        solver_type = CodexSampler._normalize_er_sde_solver_type(solver_raw)

        max_stage_raw = payload.get("max_stage", 3)
        if isinstance(max_stage_raw, bool) or not isinstance(max_stage_raw, (int, float)):
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        if isinstance(max_stage_raw, float) and not max_stage_raw.is_integer():
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        max_stage = int(max_stage_raw)
        if max_stage < 1 or max_stage > 3:
            raise ValueError("ER-SDE option 'max_stage' must be in [1, 3]")

        eta_raw = payload.get("eta", 1.0)
        if isinstance(eta_raw, bool) or not isinstance(eta_raw, (int, float)):
            raise ValueError("ER-SDE option 'eta' must be numeric")
        eta = float(eta_raw)
        if not math.isfinite(eta):
            raise ValueError("ER-SDE option 'eta' must be finite")
        if eta < 0.0:
            raise ValueError("ER-SDE option 'eta' must be >= 0")

        s_noise_raw = payload.get("s_noise", 1.0)
        if isinstance(s_noise_raw, bool) or not isinstance(s_noise_raw, (int, float)):
            raise ValueError("ER-SDE option 's_noise' must be numeric")
        s_noise = float(s_noise_raw)
        if not math.isfinite(s_noise):
            raise ValueError("ER-SDE option 's_noise' must be finite")
        if s_noise < 0.0:
            raise ValueError("ER-SDE option 's_noise' must be >= 0")

        if solver_type == "ode" or (solver_type == "reverse_time_sde" and eta == 0.0):
            eta = 0.0
            s_noise = 0.0

        return {
            "solver_type": solver_type,
            "max_stage": max_stage,
            "eta": eta,
            "s_noise": s_noise,
        }

    @staticmethod
    def _er_sde_noise_scaler(
        values: torch.Tensor,
        *,
        solver_type: str,
        eta: float,
    ) -> torch.Tensor:
        if solver_type == "er_sde":
            return values * (torch.exp(values.pow(0.3)) + 10.0)
        return values.pow(eta + 1.0)

    @staticmethod
    def _compute_er_sde_lambdas(sigmas: torch.Tensor, *, prediction_type: str | None) -> torch.Tensor:
        if sigmas.ndim != 1:
            raise RuntimeError(f"ER-SDE expects a 1D sigma schedule, got shape={tuple(sigmas.shape)}")
        sigmas_fp32 = sigmas.to(dtype=torch.float32)
        if prediction_type == "const":
            sigma_safe = sigmas_fp32.clamp(min=1e-6, max=1.0 - 1e-6)
            half_log_snr = -torch.logit(sigma_safe)
        else:
            sigma_safe = sigmas_fp32.clamp(min=1e-12)
            half_log_snr = -torch.log(sigma_safe)
        return torch.exp(-half_log_snr)

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
        post_step_hook: Optional[Callable[[torch.Tensor, int, int], None]] = None,
        post_sample_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context: SamplingContext | None = None,
        er_sde_options: Any = None,
    ) -> torch.Tensor:
        base_noise = noise.detach().clone()
        base_context = context
        warned_full_preview = False

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
            noise = base_noise.to(dtype=target_dtype)

            if init_latent is not None and init_latent.dtype != noise.dtype:
                init_latent = init_latent.to(dtype=noise.dtype)

            progress_bar = None
            retry = False
            prepared = False
            state_started = False
            active_context = base_context

            try:
                allow_vae_resident = False
                preview_interval = 0
                try:
                    if base_context is not None:
                        preview_interval = max(0, int(getattr(base_context, "preview_interval", 0) or 0))
                    else:
                        from apps.backend.runtime.live_preview import preview_interval_steps

                        preview_interval = preview_interval_steps(default=0)
                except Exception:
                    preview_interval = 0

                if preview_callback is not None and preview_interval > 0:
                    try:
                        from apps.backend.runtime.live_preview import LivePreviewMethod, live_preview_method

                        method = live_preview_method(default=LivePreviewMethod.FULL)
                        allow_vae_resident = method == LivePreviewMethod.FULL
                        if allow_vae_resident and not warned_full_preview:
                            self._logger.warning(
                                "Live preview FULL uses VAE decode during sampling. "
                                "This increases VRAM usage and can reduce generation performance; "
                                "prefer 'Approx cheap' when possible."
                            )
                            warned_full_preview = True
                    except Exception:
                        allow_vae_resident = False

                enforce_smart_offload_pre_sampling_residency(
                    self.sd_model,
                    stage="sampling.prepare",
                    allow_vae_resident=allow_vae_resident,
                )

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
                profile_meta = {
                    "algorithm": self.algorithm,
                    "sampler_kind": sampler_kind.value,
                    "scheduler": active_context.scheduler_name,
                    "steps": steps - start_idx,
                    "cfg_scale": float(cfg_scale),
                    "device": str(noise.device),
                    "noise_dtype": str(noise.dtype),
                    "x_dtype": str(x.dtype),
                    "model_compute_dtype": str(getattr(model, "computation_dtype", None)),
                }
                profile_name = f"sampling-{sampler_kind.value}-{active_context.scheduler_name}"

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
                old_denoised_d: Optional[torch.Tensor] = None
                t_prev: float | None = None
                h_prev: float | None = None
                eps_history: List[torch.Tensor] = []
                er_sde_params: dict[str, Any] | None = None
                er_sde_lambdas: torch.Tensor | None = None
                er_sde_point_indices: torch.Tensor | None = None
                if sampler_kind is SamplerKind.ER_SDE:
                    er_sde_params = self._resolve_er_sde_runtime_params(er_sde_options)
                    er_sde_lambdas = self._compute_er_sde_lambdas(
                        sigmas_run,
                        prediction_type=getattr(active_context, "prediction_type", None),
                    )
                    er_sde_point_indices = torch.arange(
                        0.0,
                        200.0,
                        dtype=torch.float32,
                        device=x.device,
                    )

                with profiler.profile_run(profile_name, meta=profile_meta):
                    for i in range(start_idx, steps):
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")
                        step_index = i - start_idx
                        with profiler.section(f"sampling.step/{step_index + 1}"):
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
                                # DPM-Solver++(2M) in log-sigma time (reference update form).
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
                                # but matches the core update form.
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
                            elif sampler_kind is SamplerKind.ER_SDE:
                                if er_sde_params is None or er_sde_lambdas is None or er_sde_point_indices is None:
                                    raise RuntimeError("ER-SDE runtime state is not initialized")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                else:
                                    solver_type = str(er_sde_params["solver_type"])
                                    eta = float(er_sde_params["eta"])
                                    s_noise = float(er_sde_params["s_noise"])
                                    max_stage = int(er_sde_params["max_stage"])
                                    stage_used = min(max_stage, step_index + 1)

                                    er_lambda_s = er_sde_lambdas[step_index]
                                    er_lambda_t = er_sde_lambdas[step_index + 1]
                                    er_lambda_s_f = float(er_lambda_s)
                                    er_lambda_t_f = float(er_lambda_t)
                                    if (
                                        not math.isfinite(er_lambda_s_f)
                                        or not math.isfinite(er_lambda_t_f)
                                        or er_lambda_s_f <= 0.0
                                        or er_lambda_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid lambda values: {er_lambda_s_f}, {er_lambda_t_f}"
                                        )

                                    alpha_s = sigma_f / er_lambda_s_f
                                    alpha_t = sigma_next_f / er_lambda_t_f
                                    if (
                                        not math.isfinite(alpha_s)
                                        or not math.isfinite(alpha_t)
                                        or alpha_s <= 0.0
                                        or alpha_t <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid alpha values: {alpha_s}, {alpha_t}"
                                        )

                                    r_alpha = alpha_t / alpha_s
                                    noise_scale_s = self._er_sde_noise_scaler(
                                        er_lambda_s,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_t = self._er_sde_noise_scaler(
                                        er_lambda_t,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_s_f = float(noise_scale_s)
                                    noise_scale_t_f = float(noise_scale_t)
                                    if (
                                        not math.isfinite(noise_scale_s_f)
                                        or not math.isfinite(noise_scale_t_f)
                                        or noise_scale_s_f <= 0.0
                                        or noise_scale_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE noise scaler returned invalid values: {noise_scale_s_f}, {noise_scale_t_f}"
                                        )
                                    r = noise_scale_t_f / noise_scale_s_f
                                    if not math.isfinite(r):
                                        raise RuntimeError(f"ER-SDE produced non-finite ratio r={r}")

                                    # Stage 1 (Euler)
                                    x = r_alpha * r * x + alpha_t * (1.0 - r) * denoised

                                    if stage_used >= 2:
                                        if old_denoised is None:
                                            raise RuntimeError("ER-SDE stage-2 requires previous denoised state")
                                        dt = er_lambda_t_f - er_lambda_s_f
                                        lambda_step_size = -dt / 200.0
                                        lambda_pos = er_lambda_t + er_sde_point_indices * lambda_step_size
                                        scaled_pos = self._er_sde_noise_scaler(
                                            lambda_pos,
                                            solver_type=solver_type,
                                            eta=eta,
                                        )
                                        if not bool(torch.all(torch.isfinite(scaled_pos))):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite scaled positions at step={step_index + 1}"
                                            )
                                        if bool(torch.any(scaled_pos <= 0.0)):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-positive scaled positions at step={step_index + 1}"
                                            )
                                        s_term = float(torch.sum(1.0 / scaled_pos) * lambda_step_size)
                                        if not math.isfinite(s_term):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite integral term at step={step_index + 1}"
                                            )
                                        prev_gap = er_lambda_s_f - float(er_sde_lambdas[step_index - 1])
                                        if abs(prev_gap) <= 1e-12:
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 denominator collapsed at step={step_index + 1}"
                                            )
                                        denoised_d = (denoised - old_denoised) / prev_gap
                                        x = x + alpha_t * (dt + s_term * noise_scale_t_f) * denoised_d

                                        if stage_used >= 3:
                                            if old_denoised_d is None:
                                                raise RuntimeError("ER-SDE stage-3 requires previous stage-2 state")
                                            s_u = float(
                                                torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                                            )
                                            if not math.isfinite(s_u):
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 produced non-finite integral term at step={step_index + 1}"
                                                )
                                            stage3_gap = (er_lambda_s_f - float(er_sde_lambdas[step_index - 2])) / 2.0
                                            if abs(stage3_gap) <= 1e-12:
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 denominator collapsed at step={step_index + 1}"
                                                )
                                            denoised_u = (denoised_d - old_denoised_d) / stage3_gap
                                            x = x + alpha_t * ((dt**2) / 2.0 + s_u * noise_scale_t_f) * denoised_u
                                        old_denoised_d = denoised_d.detach()

                                    if s_noise > 0.0:
                                        noise_term = er_lambda_t_f**2 - er_lambda_s_f**2 * (r**2)
                                        if noise_term < -1e-8:
                                            raise RuntimeError(
                                                f"ER-SDE produced negative stochastic term at step={step_index + 1}: {noise_term}"
                                            )
                                        noise_scale = math.sqrt(max(noise_term, 0.0))
                                        if not math.isfinite(noise_scale):
                                            raise RuntimeError(
                                                f"ER-SDE produced non-finite stochastic scale at step={step_index + 1}"
                                            )
                                        x = x + (
                                            alpha_t
                                            * torch.randn_like(x)
                                            * s_noise
                                            * noise_scale
                                        )
                                    old_denoised = denoised.detach()
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

                            if post_step_hook is not None:
                                post_step_hook(x, i + 1, steps)

                            if preview_callback is not None and (
                                preview_interval > 0 and ((i + 1) % preview_interval == 0) or (i + 1) == steps
                            ):
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
                        profiler.step()

                if progress_bar is not None:
                    progress_bar.close()
                    progress_bar = None

                sampling_cleanup(denoiser)
                prepared = False

                backend_state.end()
                state_started = False

                if post_sample_hook is not None:
                    x = post_sample_hook(x)

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
