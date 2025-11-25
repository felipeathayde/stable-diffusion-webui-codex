from __future__ import annotations

# tags: sampling, diagnostics

from typing import Any, Optional, Callable, List
import os
import logging

import torch

from . import sampling_function_inner, sampling_prepare, sampling_cleanup
from .condition import compile_conditions
from .context import SamplingContext, build_sampling_context
from ...core.state import state as backend_state
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole


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
        self._log_enabled = str(os.getenv("CODEX_LOG_SAMPLER", "0")).lower() in ("1","true","yes","on")
        self._log_sigmas = str(os.getenv("CODEX_LOG_SIGMAS", "0")).lower() in ("1","true","yes","on")

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
        unet = self.sd_model.codex_objects.unet
        model = getattr(unet, "model", None)
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

        while True:
            unet = self.sd_model.codex_objects.unet
            model = unet.model

            steps = int(getattr(processing, "steps", 20))
            cfg_scale = float(getattr(processing, "cfg_scale", 7.0))

            if steps <= 0:
                raise ValueError("steps must be >= 1")
            if noise.ndim != 4:
                raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

            target_dtype = memory_management.core_dtype()
            noise = base_noise.to(dtype=target_dtype)

            if init_latent is not None and init_latent.dtype != noise.dtype:
                init_latent = init_latent.to(dtype=noise.dtype)

            progress_bar = None
            retry = False
            prepared = False
            state_started = False
            active_context = base_context

            try:
                sampling_prepare(unet, noise)
                prepared = True

                scheduler_name = getattr(processing, "scheduler", None)
                if active_context is None:
                    active_context = build_sampling_context(
                        self.sd_model,
                        sampler_name=self.algorithm,
                        scheduler_name=scheduler_name,
                        steps=steps,
                        noise_source=os.getenv("CODEX_NOISE_SOURCE"),
                        eta_noise_seed_delta=int(getattr(processing, "eta_noise_seed_delta", 0) or 0),
                        device=noise.device,
                        dtype=noise.dtype,
                    )

                sigmas = active_context.sigmas.to(device=noise.device, dtype=noise.dtype)
                steps = active_context.steps

                if self._log_sigmas or self._log_enabled:
                    schedule_first = float(sigmas[0]) if len(sigmas) > 0 else float("nan")
                    schedule_last = float(sigmas[-1]) if len(sigmas) > 0 else float("nan")
                    schedule_summary = self._summarize_sigmas(sigmas)
                    self._logger.info(
                        "sigma schedule len=%d predict_min=%.6g predict_max=%.6g first=%.6g last=%.6g ladder=%s",
                        len(sigmas) - 1,
                        float(active_context.sigma_min or float("nan")),
                        float(active_context.sigma_max or float("nan")),
                        schedule_first,
                        schedule_last,
                        schedule_summary,
                    )

                start_idx = int(start_at_step or 0)
                start_idx = max(0, min(start_idx, steps - 1))
                if init_latent is not None:
                    x = init_latent + float(sigmas[start_idx]) * noise
                else:
                    x = model.predictor.noise_scaling(sigmas[:1], noise, torch.zeros_like(noise))

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
                        "sampler algorithm=%s scheduler=%s steps=%d prediction=%s sigma_max=%.6g sigma_min=%.6g sigma_data=%s head=%s",
                        self.algorithm,
                        active_context.scheduler_name,
                        steps,
                        pred_type or getattr(active_context, "prediction_type", None) or "<unknown>",
                        smax,
                        smin,
                        f"{float(sigma_data):.4g}" if sigma_data is not None else "n/a",
                        head,
                    )

                compiled_cond = compile_conditions(cond)
                compiled_uncond = compile_conditions(uncond) if uncond is not None else None

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

                strict = str(os.getenv("CODEX_SAMPLER_STRICT", "1")).lower() in ("1","true","yes","on")
                import time as _time

                preview_interval = active_context.preview_interval
                t0 = _time.perf_counter()
                use_progress = active_context.enable_progress
                if use_progress:
                    from tqdm.auto import tqdm

                    progress_bar = tqdm(total=steps - start_idx, desc="sampling", leave=False)

                sampler_kind = active_context.sampler_kind
                if sampler_kind is SamplerKind.AUTOMATIC:
                    sampler_kind = SamplerKind.EULER_A

                eps_prev: Optional[torch.Tensor] = None
                sigma_prev: Optional[float] = None
                eps_history: List[torch.Tensor] = []

                for i in range(start_idx, steps):
                    sigma = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    sigma_batch = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)

                    denoised = sampling_function_inner(
                        model,
                        x,
                        sigma_batch,
                        compiled_uncond,
                        compiled_cond,
                        cfg_scale,
                        unet.model_options,
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
                        next_dtype = memory_management.report_precision_failure(
                            DeviceRole.CORE,
                            location=f"sampler.step_{i + 1}",
                            reason=reason,
                        )
                        if next_dtype is None:
                            hint = memory_management.precision_hint(DeviceRole.CORE)
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
                        h = float(sigma_next) - float(sigma)
                        if eps_prev is None or sigma_prev is None:
                            x_euler = denoised + float(sigma_next) * eps
                            sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=x.dtype)
                            denoised_next = sampling_function_inner(
                                model,
                                x_euler,
                                sigma_next_batch,
                                compiled_uncond,
                                compiled_cond,
                                cfg_scale,
                                unet.model_options,
                                seed=None,
                                return_full=False,
                            )
                            eps_next = (x_euler - denoised_next) / max(float(sigma_next), 1e-8)
                            f_n = -eps
                            f_next = -eps_next
                            x = x + h * 0.5 * (f_n + f_next)
                        else:
                            h_prev = float(sigma) - float(sigma_prev)
                            r = h / (h_prev if abs(h_prev) > 1e-12 else h)
                            f_n = -eps
                            f_prev = -eps_prev
                            x = x + h * (((1.0 + r) * 0.5) * f_n - (r * 0.5) * f_prev)
                        eps_prev = eps.detach()
                        sigma_prev = float(sigma)
                    elif sampler_kind is SamplerKind.DPM2M_SDE:
                        h = float(sigma_next) - float(sigma)
                        if eps_prev is None or sigma_prev is None:
                            x_euler = denoised + float(sigma_next) * eps
                            sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=x.dtype)
                            denoised_next = sampling_function_inner(
                                model,
                                x_euler,
                                sigma_next_batch,
                                compiled_uncond,
                                compiled_cond,
                                cfg_scale,
                                unet.model_options,
                                seed=None,
                                return_full=False,
                            )
                            eps_next = (x_euler - denoised_next) / max(float(sigma_next), 1e-8)
                            f_n = -eps
                            f_next = -eps_next
                            x = x + h * 0.5 * (f_n + f_next)
                        else:
                            h_prev = float(sigma) - float(sigma_prev)
                            r = h / (h_prev if abs(h_prev) > 1e-12 else h)
                            f_n = -eps
                            f_prev = -eps_prev
                            x = x + h * (((1.0 + r) * 0.5) * f_n - (r * 0.5) * f_prev)
                        s = float(sigma)
                        s_next = float(sigma_next)
                        if s_next > 0.0:
                            sigma_up_sq = max(s_next**2 * (s**2 - s_next**2) / max(s**2, 1e-8), 0.0)
                            sigma_up = sigma_up_sq ** 0.5
                            x = x + sigma_up * torch.randn_like(x)
                        eps_prev = eps.detach()
                        sigma_prev = float(sigma)
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
                        sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=x.dtype)
                        denoised_next = sampling_function_inner(
                            model,
                            x_pred,
                            sigma_next_batch,
                            compiled_uncond,
                            compiled_cond,
                            cfg_scale,
                            unet.model_options,
                            seed=None,
                            return_full=False,
                        )
                        eps_next = (x_pred - denoised_next) / max(float(sigma_next), 1e-8)
                        x = x - delta * 0.5 * (eps + eps_next)
                    else:
                        raise NotImplementedError(f"Sampler '{sampler_kind.value}' is not implemented natively yet")

                    if sampler_kind not in (SamplerKind.DPM2M, SamplerKind.DPM2M_SDE):
                        sigma_prev = float(sigma)
                        eps_prev = eps.detach()

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

                sampling_cleanup(unet)
                prepared = False

                backend_state.end()
                state_started = False

                return x
            except _PrecisionFallbackRequest:
                self._logger.warning("Precision fallback requested for diffusion core; retrying with next dtype.")
            finally:
                if progress_bar is not None:
                    progress_bar.close()
                if prepared:
                    sampling_cleanup(unet)
                if state_started:
                    backend_state.end()

            if retry:
                memory_management.soft_empty_cache(force=True)
                continue



__all__ = ["CodexSampler"]
