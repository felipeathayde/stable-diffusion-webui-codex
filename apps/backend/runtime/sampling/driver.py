from __future__ import annotations

from typing import Any, Optional, Callable, List
import os
import logging

import torch

from . import sampling_function_inner, sampling_prepare, sampling_cleanup
from .condition import compile_conditions
from .context import SamplingContext, build_sampling_context
from ...core.state import state as backend_state
from apps.backend.engines.util.schedulers import SamplerKind


_LMS_COEFFS = {
    1: (1.0,),
    2: (3.0 / 2.0, -1.0 / 2.0),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


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
        unet = self.sd_model.codex_objects.unet
        model = unet.model

        steps = int(getattr(processing, "steps", 20))
        cfg_scale = float(getattr(processing, "cfg_scale", 7.0))

        if steps <= 0:
            raise ValueError("steps must be >= 1")
        if noise.ndim != 4:
            raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

        # Prepare GPU residency and model options
        sampling_prepare(unet, noise)

        scheduler_name = getattr(processing, "scheduler", None)
        if context is None:
            context = build_sampling_context(
                self.sd_model,
                sampler_name=self.algorithm,
                scheduler_name=scheduler_name,
                steps=steps,
                noise_source=os.getenv("CODEX_NOISE_SOURCE"),
                eta_noise_seed_delta=int(getattr(processing, "eta_noise_seed_delta", 0) or 0),
                device=noise.device,
                dtype=noise.dtype,
            )
        sigmas = context.sigmas.to(device=noise.device, dtype=noise.dtype)
        steps = context.steps
        # Starting latent
        start_idx = int(start_at_step or 0)
        start_idx = max(0, min(start_idx, steps - 1))
        if init_latent is not None:
            # Start from provided latent with noise at sigma[start_idx]
            x = init_latent + float(sigmas[start_idx]) * noise
        else:
            x = model.predictor.noise_scaling(sigmas[:1], noise, torch.zeros_like(noise))

        if self._log_enabled:
            try:
                smax = float(sigmas[0].item()) if hasattr(sigmas[0], 'item') else float(sigmas[0])
                smin = float(sigmas[-1].item()) if hasattr(sigmas[-1], 'item') else float(sigmas[-1])
            except Exception:
                smax = float('nan'); smin = float('nan')
            self._logger.info("sampler algorithm=%s steps=%d sigma_max=%.6g sigma_min=%.6g", self.algorithm, steps, smax, smin)

        # Compile conditions
        compiled_cond = compile_conditions(cond)
        compiled_uncond = compile_conditions(uncond) if uncond is not None else None

        # Inject image conditioning (c_concat) when provided and shape-compatible
        if isinstance(image_conditioning, torch.Tensor):
            if (
                image_conditioning.shape[0] == noise.shape[0]
                and image_conditioning.shape[2] == noise.shape[2]
                and image_conditioning.shape[3] == noise.shape[3]
            ):
                from .condition import Condition
                for entry in compiled_cond:
                    entry['model_conds']['c_concat'] = Condition(image_conditioning)
                if compiled_uncond is not None:
                    for entry in compiled_uncond:
                        entry['model_conds']['c_concat'] = Condition(image_conditioning)

        # Progress state
        backend_state.start(job_count=1, sampling_steps=steps - start_idx)

        # Sampling loop
        eps_prev: Optional[torch.Tensor] = None
        sigma_prev: Optional[float] = None
        strict = str(os.getenv("CODEX_SAMPLER_STRICT", "1")).lower() in ("1","true","yes","on")
        import time as _time
        preview_interval = context.preview_interval
        t0 = _time.perf_counter()
        use_progress = context.enable_progress
        progress_bar = None
        if use_progress:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=steps - start_idx, desc="sampling", leave=False)

        sampler_kind = context.sampler_kind
        if sampler_kind is SamplerKind.AUTOMATIC:
            sampler_kind = SamplerKind.EULER_A

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

            # Convert denoised sample x0 to epsilon: eps = (x - x0) / sigma
            eps = (x - denoised) / max(float(sigma), 1e-8)
            if strict and (torch.isnan(eps).any() or torch.isnan(denoised).any()):
                raise RuntimeError(f"NaN encountered at step {i+1}")

            eps_history.append(eps.detach())
            if len(eps_history) > 4:
                eps_history.pop(0)

            if sampler_kind is SamplerKind.EULER:
                # Euler ODE update: x_{t+1} = x_t - (sigma - sigma_next) * eps
                x = x - (float(sigma) - float(sigma_next)) * eps
            elif sampler_kind is SamplerKind.EULER_A:
                # Euler ancestral: split into deterministic step + noise
                sigma = float(sigma); sigma_next = float(sigma_next)
                if sigma_next <= 0.0:
                    # final step goes deterministic
                    x = denoised
                else:
                    # variance preserving update in sigma space
                    sigma_up_sq = max(sigma_next**2 * (sigma**2 - sigma_next**2) / max(sigma**2, 1e-8), 0.0)
                    sigma_up = sigma_up_sq ** 0.5
                    sigma_down = (max(sigma_next**2 - sigma_up_sq, 0.0)) ** 0.5
                    # deterministic part to x0 + sigma_down*eps
                    x = denoised + sigma_down * eps
                    # add fresh noise
                    noise = torch.randn_like(x)
                    x = x + sigma_up * noise
            elif sampler_kind is SamplerKind.DPM2M:
                # DPM++ 2M (multistep, ODE) using single eval per step after a Heun bootstrap
                h = float(sigma_next) - float(sigma)
                if eps_prev is None or sigma_prev is None:
                    # Bootstrap with Heun (two evals)
                    x_euler = denoised + float(sigma_next) * eps  # DDIM-like prediction at next sigma
                    sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=x.dtype)
                    denoised_next = sampling_function_inner(
                        model, x_euler, sigma_next_batch, compiled_uncond, compiled_cond, cfg_scale, unet.model_options, seed=None, return_full=False
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
                # update history
                eps_prev = eps.detach()
                sigma_prev = float(sigma)
            elif sampler_kind is SamplerKind.DPM2M_SDE:
                # DPM++ 2M SDE: same multistep core plus an ancestral noise term
                h = float(sigma_next) - float(sigma)
                if eps_prev is None or sigma_prev is None:
                    # Heun bootstrap
                    x_euler = denoised + float(sigma_next) * eps
                    sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=x.dtype)
                    denoised_next = sampling_function_inner(
                        model, x_euler, sigma_next_batch, compiled_uncond, compiled_cond, cfg_scale, unet.model_options, seed=None, return_full=False
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
                # add SDE noise akin to Euler A variance split
                s = float(sigma); s_next = float(sigma_next)
                if s_next > 0.0:
                    sigma_up_sq = max(s_next**2 * (s**2 - s_next**2) / max(s**2, 1e-8), 0.0)
                    sigma_up = sigma_up_sq ** 0.5
                    x = x + sigma_up * torch.randn_like(x)
                eps_prev = eps.detach()
                sigma_prev = float(sigma)
            elif sampler_kind is SamplerKind.DDIM:
                # Deterministic DDIM-like step in sigma domain (eta=0)
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

            # Optional preview callback on denoised (x0)
            if preview_callback is not None and (preview_interval > 0 and ((i+1) % preview_interval == 0) or (i+1) == steps):
                try:
                    preview_callback(denoised.detach(), i + 1, steps)
                except Exception:
                    pass

            if self._log_enabled and (i == 0 or (i+1) == steps or (i+1) % max(1, steps//5) == 0):
                self._logger.info("step=%d/%d sigma=%.6g->%.6g norm(x)=%.4f dt=%.2fms", i+1, steps, float(sigma), float(sigma_next), float(x.norm().item()), (_time.perf_counter()-t0)*1000.0)
                t0 = _time.perf_counter()

            if progress_bar is not None:
                progress_bar.update(1)

            backend_state.tick(sampling_step=i + 1)

        if progress_bar is not None:
            progress_bar.close()

        sampling_cleanup(unet)
        return x


__all__ = ["CodexSampler"]
