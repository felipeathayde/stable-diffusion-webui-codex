from __future__ import annotations

from typing import Any, Optional, Callable
import os
import logging

import torch

from . import sampling_function_inner, sampling_prepare, sampling_cleanup
from .condition import compile_conditions
from ...core.state import state as backend_state


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

    def _sigma_bounds(self, model) -> tuple[float, float]:
        pred = getattr(model, "predictor", None)
        if pred is None:
            raise RuntimeError("Model predictor missing; cannot derive sigma schedule")
        smin = float(getattr(pred, "sigma_min").item() if hasattr(pred.sigma_min, "item") else pred.sigma_min)
        smax = float(getattr(pred, "sigma_max").item() if hasattr(pred.sigma_max, "item") else pred.sigma_max)
        # ensure smax >= smin
        if smax < smin:
            smin, smax = smax, smin
        return smin, smax

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
    ) -> torch.Tensor:
        unet = self.sd_model.forge_objects.unet
        model = unet.model

        steps = int(getattr(processing, "steps", 20))
        cfg_scale = float(getattr(processing, "cfg_scale", 7.0))

        if steps <= 0:
            raise ValueError("steps must be >= 1")
        if noise.ndim != 4:
            raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

        # Prepare GPU residency and model options
        sampling_prepare(unet, noise)

        # Initial latent at sigma_max
        smin, smax = self._sigma_bounds(model)
        sigmas = torch.linspace(smax, smin, steps + 1, device=noise.device, dtype=noise.dtype)
        # Starting latent
        start_idx = int(start_at_step or 0)
        start_idx = max(0, min(start_idx, steps - 1))
        if init_latent is not None:
            # Start from provided latent with noise at sigma[start_idx]
            x = init_latent + float(sigmas[start_idx]) * noise
        else:
            x = model.predictor.noise_scaling(sigmas[:1], noise, torch.zeros_like(noise))

        if self._log_enabled:
            self._logger.info("sampler algorithm=%s steps=%d sigma_max=%.6g sigma_min=%.6g", self.algorithm, steps, float(smax), float(smin))

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
        preview_interval = int(os.getenv("CODEX_PREVIEW_INTERVAL", "0") or 0)
        t0 = _time.perf_counter()
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

            if self.algorithm in ("euler", "euler ode", "ode"):
                # Euler ODE update: x_{t+1} = x_t - (sigma - sigma_next) * eps
                x = x - (float(sigma) - float(sigma_next)) * eps
            elif self.algorithm in ("euler a", "euler_ancestral", "ancestral"):
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
            elif self.algorithm in ("dpm++ 2m", "dpmpp 2m"):
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
            elif self.algorithm in ("dpm++ 2m sde", "dpmpp 2m sde"):
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
            elif self.algorithm in ("ddim",):
                # Deterministic DDIM-like step in sigma domain (eta=0)
                x = denoised + float(sigma_next) * eps
            else:
                # default to Euler A
                sigma = float(sigma); sigma_next = float(sigma_next)
                sigma_up_sq = max(sigma_next**2 * (sigma**2 - sigma_next**2) / max(sigma**2, 1e-8), 0.0)
                sigma_up = sigma_up_sq ** 0.5
                sigma_down = (max(sigma_next**2 - sigma_up_sq, 0.0)) ** 0.5
                x = denoised + sigma_down * eps + sigma_up * torch.randn_like(x)

            # Optional preview callback on denoised (x0)
            if preview_callback is not None and (preview_interval > 0 and ((i+1) % preview_interval == 0) or (i+1) == steps):
                try:
                    preview_callback(denoised.detach(), i + 1, steps)
                except Exception:
                    pass

            if self._log_enabled and (i == 0 or (i+1) == steps or (i+1) % max(1, steps//5) == 0):
                self._logger.info("step=%d/%d sigma=%.6g->%.6g norm(x)=%.4f dt=%.2fms", i+1, steps, float(sigma), float(sigma_next), float(x.norm().item()), (_time.perf_counter()-t0)*1000.0)
                t0 = _time.perf_counter()

            backend_state.tick(sampling_step=i + 1)

        sampling_cleanup(unet)
        return x


__all__ = ["CodexSampler"]
