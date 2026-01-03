"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Optional sampler extras (UniPC, Restart, DDPM) for the k-diffusion-enabled path.
Provides lightweight noise-schedule utilities and step samplers used by the runtime sampling driver when k-diffusion-backed samplers are
explicitly enabled and available.

Symbols (top-level; keep in sync; no ghosts):
- `NoiseScheduleVP` (class): VP noise schedule helper (discrete/linear/cosine) providing alpha/sigma/lambda accessors for samplers.
- `model_wrapper` (function): Wraps a model into a VP-space epsilon predictor with optional guidance modes.
- `sample_unipc` (function): Minimal UniPC sampler implementation operating over a sigma schedule.
- `sample_unipc_bh2` (function): UniPC BH2 placeholder (currently forwards to `sample_unipc`).
- `restart_sampler` (function): Restart sampling wrapper (requires k-diffusion; supports restart segments + noise injection).
- `default_noise_sampler` (function): Returns a default noise sampler closure for stochastic samplers.
- `generic_step_sampler` (function): Generic sampler driver that iterates sigmas and calls a provided step function.
- `DDPMSampler_step` (function): Single-step DDPM update function used by the generic step sampler.
- `sample_ddpm` (function): DDPM sampler wrapper using `generic_step_sampler`.
"""

import math

import torch
from tqdm import trange

try:
  # Optional: k-diffusion-backed helpers; when missing, these samplers stay unavailable.
  import k_diffusion.sampling as _KD_SAMPLING  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - absence is expected on minimal installs
  _KD_SAMPLING = None


class NoiseScheduleVP:
    def __init__(self, schedule="discrete", *, betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0):
        if schedule not in {"discrete", "linear", "cosine"}:
            raise ValueError(f"Unsupported noise schedule {schedule}")
        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                if alphas_cumprod is None:
                    raise ValueError("alphas_cumprod required for discrete schedule")
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            self.T = 0.9946 if schedule == "cosine" else 1.0

    def marginal_log_mean_coeff(self, t):
        if self.schedule == "discrete":
            return torch.interp(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        if self.schedule == "linear":
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
        log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
        return log_alpha_t

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def model_wrapper(model, noise_schedule: NoiseScheduleVP, model_type="noise", model_kwargs=None, guidance_type="uncond", condition=None, unconditional_condition=None, guidance_scale=1.0):
    model_kwargs = model_kwargs or {}
    def _fn(x, t_continuous):
        t = t_continuous
        if noise_schedule.schedule == "discrete":
            t = t * 999
        sigma = noise_schedule.marginal_std(t)
        model_sigma = sigma
        model_input = x / noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1)))
        eps = model(model_input, model_sigma, **model_kwargs)
        if model_type == "x_start":
            x0_pred = eps
            eps = (x - x0_pred * noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1)))) / noise_schedule.marginal_std(t).view(x.shape[0], *((1,) * (x.ndim - 1)))
        elif model_type == "v":
            eps = noise_schedule.marginal_std(t).view(x.shape[0], *((1,) * (x.ndim - 1))) * eps + noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1))) * x
        if guidance_type == "uncond" or guidance_type is None:
            return eps
        if guidance_type == "classifier":
            raise NotImplementedError
        if guidance_type == "classifier-free":
            cond = condition
            uncond = unconditional_condition
            if cond is None or uncond is None:
                return eps
            eps_uncond, eps_cond = eps.chunk(2)
            return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        raise ValueError(f"Unknown guidance type {guidance_type}")
    return _fn


def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = extra_args or {}
    # Convert sigmas (descending) to t in [0,1]
    # Use VP discrete schedule; len(sigmas)-1 steps.
    alphas_cumprod = 1.0 / (sigmas ** 2 + 1.0)
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod[:-1].flip(0))
    model_fn = model_wrapper(model, noise_schedule, model_kwargs=extra_args, guidance_type=None)

    # Basic UniPC (order=2) adapted from official implementation
    timesteps = noise_schedule.t_array.to(x.device).view(-1)
    # map sigmas length to timesteps length; pad to match
    if timesteps.numel() < sigmas.numel():
        timesteps = torch.linspace(0, 1, sigmas.numel(), device=x.device)

    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        h = t_next - t_cur
        eps = model_fn(x, t_cur)
        x = x + h * eps
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": x - eps})
    return x


def sample_unipc_bh2(model, x, sigmas, extra_args=None, callback=None, disable=None):
    # Placeholder: reuse UniPC core; BH2 variant could adopt a higher-order update.
    return sample_unipc(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)


@torch.no_grad()
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, restart_list=None):
    """Restart sampling (Restart Sampling for Improving Generative Processes, 2023).

    Optionally inserts restart segments built with Karras sigmas, applies Heun/Euler steps, and injects
    noise between segments. Parameter semantics match the runtime config surface while using the native
    k-diffusion utilities already imported in this module.
    """

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    def _require_kd():
        if _KD_SAMPLING is None:
            raise RuntimeError("k-diffusion is required for restart sampler but is not installed")
        return _KD_SAMPLING

    def _heun_step(x_in: torch.Tensor, sigma_cur: torch.Tensor, sigma_next: torch.Tensor, *, second_order: bool = True):
        nonlocal step_id
        denoised = model(x_in, sigma_cur * s_in, **extra_args)
        kd_sampling = _require_kd()
        d = kd_sampling.to_d(x_in, sigma_cur, denoised)
        if callback is not None:
            callback({"x": x_in, "i": step_id, "sigma": sigma_next, "sigma_hat": sigma_cur, "denoised": denoised})
        dt = sigma_next - sigma_cur
        if sigma_next == 0 or not second_order:
            x_out = x_in + d * dt
        else:
            x_euler = x_in + d * dt
            denoised_2 = model(x_euler, sigma_next * s_in, **extra_args)
            d_2 = kd_sampling.to_d(x_euler, sigma_next, denoised_2)
            x_out = x_in + 0.5 * (d + d_2) * dt
        step_id += 1
        return x_out

    steps = sigmas.shape[0] - 1
    # Auto restart plan mirrors the historical heuristic
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            base = k_diffusion.sampling.get_sigmas_karras(
                steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device
            )
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
            sigmas = base
        else:
            restart_list = {}

    restart_map = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_pairs = []
    for i in range(len(sigmas) - 1):
        step_pairs.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_map:
            restart_steps, restart_times, restart_max = restart_map[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                kd_sampling = _require_kd()
                sigma_restart = kd_sampling.get_sigmas_karras(
                    restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device
                )[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_pairs.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for idx in trange(len(step_pairs), disable=disable):
        cur, nxt = step_pairs[idx]
        if last_sigma is None:
            last_sigma = cur
        elif last_sigma < cur:
            x = x + torch.randn_like(x) * s_noise * max(cur**2 - last_sigma**2, 0.0) ** 0.5
        x = _heun_step(x, cur, nxt)
        last_sigma = nxt
    return x


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)
