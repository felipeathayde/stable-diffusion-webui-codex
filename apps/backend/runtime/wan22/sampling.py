"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF sampling helpers (geometry + scheduler + per-stage sampling loops).
Builds patch geometry, prepares per-stage latent tensors, and runs the stage sampling loop (generator yields progress events).

Symbols (top-level; keep in sync; no ghosts):
- `PatchGeometry` (dataclass): Patch/tile geometry configuration used to infer latent/video shapes.
- `latent_dimensions` (function): Computes latent tensor dimensions from a `PatchGeometry` description.
- `resize_latents_hw` (function): Resizes latents to a target H/W (used for compatibility across stages/sizes).
- `ensure_latent_shape` (function): Validates/reshapes latent tensors to the expected `PatchGeometry` layout.
- `infer_patch_geometry` (function): Infers patch geometry defaults from config and requested latent size.
- `make_scheduler` (function): Constructs the scheduler instance for a run (based on sampler/scheduler selection).
- `cfg_merge` (function): Classifier-free guidance merge helper (uncond/cond + scale).
- `time_snr_shift` (function): Time/SNR shift helper used in scheduler-time transformations.
- `prepare_stage_seed_latents` (function): Prepares seeded stage latents (for determinism across runs/stages).
- `assemble_i2v_input` (function): Builds img2vid latent inputs to match expected input channels/order.
- `sample_stage_latents` (function): Core latent sampling for a single WAN stage (high/low) using the selected scheduler/sampler.
- `sample_stage_latents_generator` (function): Generator version of stage sampling for streaming progress (yields intermediate states).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from .config import WAN_FLOW_MULTIPLIER, resolve_i2v_order
from .diagnostics import (
    get_logger,
    log_cuda_mem,
    log_sigmas_enabled,
    log_t_mapping,
    summarize_tensor,
)


@dataclass(frozen=True)
class PatchGeometry:
    grid: Tuple[int, int, int]
    token_count: int
    token_dim: int
    latent_channels: int
    patch_kernel: Tuple[int, int, int]


def latent_dimensions(geom: PatchGeometry) -> Tuple[int, int, int]:
    kT, kH, kW = geom.patch_kernel
    return (
        int(geom.grid[0] * kT),
        int(geom.grid[1] * kH),
        int(geom.grid[2] * kW),
    )


def resize_latents_hw(x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    import torch.nn.functional as F

    if x.ndim == 5:
        b, c, t, h, w = x.shape
        if h == height and w == width:
            return x
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        xt = F.interpolate(xt, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        xt = xt.view(b, t, c, height, width).permute(0, 2, 1, 3, 4).contiguous()
        return xt

    if x.ndim == 4:
        b, c, h, w = x.shape
        if h == height and w == width:
            return x
        return F.interpolate(x, size=(int(height), int(width)), mode="bilinear", align_corners=False)

    return x


def ensure_latent_shape(x: torch.Tensor, geom: PatchGeometry) -> torch.Tensor:
    t_target, h_target, w_target = latent_dimensions(geom)
    if x.ndim != 5:
        raise RuntimeError(f"WAN22: expected 5D latents [B,C,T,H,W], got shape={tuple(x.shape)}")
    if x.shape[2] == t_target and x.shape[3] == h_target and x.shape[4] == w_target:
        return x
    return resize_latents_hw(x, height=h_target, width=w_target)


def infer_patch_geometry(model: Any, *, t: int, h_lat: int, w_lat: int) -> PatchGeometry:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("WAN22: expected model with .config (WanTransformer2DModel)")
    kT, kH, kW = tuple(int(x) for x in getattr(cfg, "patch_size", (1, 2, 2)))
    if t < kT or h_lat < kH or w_lat < kW:
        raise RuntimeError(
            f"WAN22: invalid latent shape for patch_embed: T={t} H={h_lat} W={w_lat} kernel={(kT, kH, kW)}"
        )
    gT = int(t - kT + 1)
    gH = int(((h_lat - kH) // kH) + 1)
    gW = int(((w_lat - kW) // kW) + 1)
    token_count = int(gT * gH * gW)
    c_out = int(getattr(cfg, "d_model", 0) or 0)
    c_in = int(getattr(cfg, "in_channels", 0) or 0)
    return PatchGeometry(
        grid=(gT, gH, gW),
        token_count=token_count,
        token_dim=c_out,
        latent_channels=c_in,
        patch_kernel=(kT, kH, kW),
    )


def make_scheduler(steps: int, *, sampler: Optional[str] = None, scheduler: Optional[str] = None):
    """Instantiate a Diffusers scheduler based on requested sampler/scheduler names."""

    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )

    s = (sampler or "").strip().lower()
    sch = (scheduler or "").strip().lower()

    cls = EulerDiscreteScheduler  # default
    if s in ("euler",):
        cls = EulerDiscreteScheduler
    elif s in ("euler a", "euler_a", "euler-ancestral", "ancestral"):
        cls = EulerAncestralDiscreteScheduler
    elif s in ("ddim",):
        cls = DDIMScheduler
    elif s in ("dpm++ 2m", "dpm++ 2m sde", "dpm2m", "dpmpp2m", "dpmpp2m sde"):
        cls = DPMSolverMultistepScheduler
    elif s in ("plms", "lms"):
        cls = LMSDiscreteScheduler
    elif s in ("pndm",):
        cls = PNDMScheduler
    elif s in ("uni-pc",):
        cls = UniPCMultistepScheduler
    else:
        # Fall back to scheduler hint, if provided
        if "euler a" in sch or "ancestral" in sch:
            cls = EulerAncestralDiscreteScheduler
        elif "euler" in sch:
            cls = EulerDiscreteScheduler
        elif "ddim" in sch:
            cls = DDIMScheduler
        elif "dpm" in sch:
            cls = DPMSolverMultistepScheduler
        elif "lms" in sch:
            cls = LMSDiscreteScheduler
        elif "pndm" in sch:
            cls = PNDMScheduler

    sched = cls()
    sched.set_timesteps(max(1, int(steps)))
    return sched


def cfg_merge(uncond: torch.Tensor, cond: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return cond
    return uncond + (cond - uncond) * float(scale)


def time_snr_shift(alpha: float, t: float) -> float:
    # Same functional form as time_snr_shift used in reference implementations
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def prepare_stage_seed_latents(
    latents: torch.Tensor,
    target_geom: PatchGeometry,
    *,
    logger: logging.Logger | None,
) -> torch.Tensor:
    c_src = int(latents.shape[1])
    c_dst = int(target_geom.latent_channels)
    if c_src == c_dst:
        return ensure_latent_shape(latents, target_geom)
    if c_src >= 16 and c_dst == 16:
        sliced = latents[:, :16, ...] if resolve_i2v_order() == "lat_first" else latents[:, -16:, ...]
        return ensure_latent_shape(sliced, target_geom)
    if c_src == 16 and c_dst == 36:
        assembled = assemble_i2v_input(latents, expected_cin=c_dst, logger=logger)
        return ensure_latent_shape(assembled, target_geom)
    raise RuntimeError(f"Cannot adapt latent channels from {c_src} to {c_dst}; unsupported hand-off configuration")


def assemble_i2v_input(
    latents: torch.Tensor,
    *,
    expected_cin: int,
    logger: logging.Logger | None,
) -> torch.Tensor:
    """Assemble I2V input volume to match expected Cin for patch embedding."""

    if latents.ndim != 5:
        raise RuntimeError(f"assemble_i2v_input: expected 5D latents [B,C,T,H,W], got {tuple(latents.shape)}")
    b, c, t, h, w = latents.shape
    extra = expected_cin - c
    if extra <= 0:
        return latents

    if extra == 20:
        # Build mask (zeros if not provided): [B,4,T,H,W]
        mask = latents.new_zeros((b, 4, t, h, w))
        # Image features: reuse VAE latents as 16-ch features by default
        image_feats = latents[:, : min(16, c)]
        if image_feats.shape[1] < 16:
            pad = 16 - image_feats.shape[1]
            image_feats = torch.cat([image_feats, latents.new_zeros((b, pad, t, h, w))], dim=1)

        order = resolve_i2v_order()
        if order == "lat_first":
            assembled = torch.cat([latents, mask, image_feats], dim=1)
            layout = f"[lat{c} + mask4 + img16]"
        else:
            assembled = torch.cat([mask, image_feats, latents], dim=1)
            layout = f"[mask4 + img16 + lat{c}]"
        if assembled.shape[1] != expected_cin:
            raise RuntimeError(
                f"I2V assembly produced {assembled.shape[1]} channels, expected {expected_cin} (mask4 + img16 + lat{c})."
            )
        if logger is not None:
            logger.info("[wan22.gguf] i2v assemble: order=%s %s → C=%d", order, layout, assembled.shape[1])
        return assembled

    raise RuntimeError(
        f"WAN22 GGUF (img2vid): expected C_in={expected_cin} but VAE produced C={c}. "
        f"I2V assembly requires extra={extra} channels (mask+image). Unsupported combo."
    )


def sample_stage_latents(
    *,
    model: Any,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger | None,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    on_progress: Optional[Any] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = "stage",
) -> torch.Tensor:
    gen = sample_stage_latents_generator(
        model=model,
        geom=geom,
        steps=steps,
        cfg_scale=cfg_scale,
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=device,
        dtype=dtype,
        logger=logger,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        seed=seed,
        state_init=state_init,
        log_mem_interval=log_mem_interval,
        flow_shift=flow_shift,
        flow_multiplier=flow_multiplier,
        stage_name=stage_name,
        emit_logs=(on_progress is None),
    )

    while True:
        try:
            event = next(gen)
        except StopIteration as stop:
            return stop.value
        if on_progress:
            payload = {k: event[k] for k in ("step", "total", "percent", "eta_seconds", "step_seconds") if k in event}
            try:
                on_progress(**payload)
            except Exception:
                get_logger(logger).debug("[wan22.gguf] progress callback raised", exc_info=True)


def sample_stage_latents_generator(
    *,
    model: Any,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger | None,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = "stage",
    emit_logs: bool = True,
):
    log = get_logger(logger)
    t_lat, h_lat, w_lat = latent_dimensions(geom)
    steps = max(int(steps), 1)

    batch = int(state_init.shape[0]) if state_init is not None else 1
    shape = (batch, int(geom.latent_channels), t_lat, h_lat, w_lat)

    if state_init is not None:
        state = ensure_latent_shape(state_init.to(device=device, dtype=dtype), geom).clone()
    else:
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            state = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            state = torch.randn(shape, device=device, dtype=dtype)
        sigma_init = time_snr_shift(flow_shift, 1.0) * flow_multiplier
        state = state * float(sigma_init)

    scheduler = make_scheduler(steps, sampler=sampler_name, scheduler=scheduler_name)
    timesteps = scheduler.timesteps
    total = len(timesteps)

    flow_progress = (
        torch.linspace(1.0, 0.0, total, device=device, dtype=torch.float32)
        if total > 1
        else torch.ones(1, device=device, dtype=torch.float32)
    )
    parity_idxs = {0, max(0, total // 2 - 1), max(0, total - 1)}

    if log_sigmas_enabled():
        sigmas = getattr(scheduler, "sigmas", None)
        if isinstance(sigmas, torch.Tensor):
            log.info(
                "[wan22.gguf] %s schedule: scheduler=%s timesteps=%d sigmas=%s",
                stage_name,
                scheduler.__class__.__name__,
                int(total),
                summarize_tensor(sigmas),
            )
        log_t_mapping(scheduler, timesteps, label=stage_name, logger=logger)

    yield {"type": "progress", "stage": stage_name, "step": 0, "total": total, "percent": 0.0}

    import time

    t0 = time.perf_counter()
    last = t0

    for idx, timestep in enumerate(timesteps):
        percent = float(flow_progress[idx].item()) if total > 1 else 1.0
        sigma_value = time_snr_shift(flow_shift, percent)
        di_timestep = float(sigma_value * flow_multiplier)

        if log_sigmas_enabled() and idx in parity_idxs:
            sched_sigmas = getattr(scheduler, "sigmas", None)
            sched_sigma = None
            if isinstance(sched_sigmas, torch.Tensor) and sched_sigmas.numel() >= (idx + 1):
                sched_sigma = float(sched_sigmas[idx].item())
            log.info(
                "[wan22.gguf] %s t-in[%d/%d]: percent=%.4f sigma_shifted=%.6g flow_multiplier=%.1f di_timestep=%.6g sched_timestep=%s sched_sigma=%s",
                stage_name,
                idx + 1,
                total,
                percent,
                float(sigma_value),
                float(flow_multiplier),
                float(di_timestep),
                str(timestep),
                str(sched_sigma),
            )

        if cfg_scale is None:
            eps = model(state, di_timestep, prompt_embeds)
        else:
            x_in = torch.cat([state, state], dim=0)
            ctx_in = torch.cat([prompt_embeds, negative_embeds], dim=0)
            t_in = torch.full((x_in.shape[0],), float(di_timestep), device=device, dtype=torch.float32)
            v_pred = model(x_in, t_in, ctx_in)
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            eps = cfg_merge(v_uncond, v_cond, cfg_scale)

        if eps.shape != state.shape:
            raise RuntimeError(
                f"WAN22 GGUF: model output shape {tuple(eps.shape)} does not match latent state {tuple(state.shape)} "
                f"(patch_size={geom.patch_kernel} grid={geom.grid})"
            )

        out = scheduler.step(model_output=eps, timestep=timestep, sample=state)
        state = out.prev_sample

        pct = float(idx + 1) / float(max(1, total))
        if log_mem_interval is not None:
            n = int(log_mem_interval or 0)
            if n > 0 and ((idx + 1) % n) == 0:
                log_cuda_mem(logger, label=f"{stage_name}-step-{idx + 1}")

        now = time.perf_counter()
        step_dt = now - last
        elapsed = now - t0
        remain = max(0, total - (idx + 1))
        eta = (elapsed / max(1, idx + 1)) * remain
        last = now

        if emit_logs and ((idx + 1) % 5 == 0 or idx + 1 == total):
            log.info("[wan22.gguf] %s step %d/%d (%.1f%%)", stage_name.upper(), idx + 1, total, pct * 100.0)

        yield {
            "type": "progress",
            "stage": stage_name,
            "step": idx + 1,
            "total": total,
            "percent": pct,
            "eta_seconds": eta,
            "step_seconds": step_dt,
        }

    return state
