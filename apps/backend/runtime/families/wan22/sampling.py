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
- `make_scheduler` (function): Builds the WAN22 scheduler from vendored metadata (`scheduler_config.json`) and validates sampler strings (Diffusers-free).
- `resolve_init_noise_sigma` (function): Resolves the scheduler initial noise sigma (`init_noise_sigma`) for seeding parity with Diffusers.
- `_assert_finite_tensor` (function): Fail-loud finite check helper with stage/step context and numeric summaries.
- `cfg_merge` (function): Classifier-free guidance merge helper (uncond/cond + scale).
- `time_snr_shift` (function): Time/SNR shift helper used in scheduler-time transformations.
- `prepare_stage_seed_latents` (function): Prepares seeded stage latents (for determinism across runs/stages).
- `build_i2v_mask4` (function): Builds the 4-channel I2V first-frame mask (Diffusers-compatible; latent time scale=4).
- `assemble_i2v_state` (function): Assembles I2V model state `[lat16 + mask4 + img16]` (order-aware, strict).
- `sample_stage_latents` (function): Core latent sampling for a single WAN stage (high/low) using the selected scheduler/sampler.
- `sample_stage_latents_generator` (function): Generator version of stage sampling for streaming progress (yields intermediate states).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from .config import WAN_FLOW_MULTIPLIER, resolve_i2v_order
from .diagnostics import (
    get_logger,
    log_cuda_mem,
    log_numerics_enabled,
    log_sigmas_enabled,
    log_t_mapping,
    summarize_tensor,
    summarize_numerics,
)


@dataclass(frozen=True)
class PatchGeometry:
    grid: Tuple[int, int, int]
    token_count: int
    token_dim: int
    in_channels: int
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
        in_channels=c_in,
        patch_kernel=(kT, kH, kW),
    )


def make_scheduler(
    steps: int,
    *,
    metadata_dir: str,
    flow_shift: float,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
):
    """Instantiate the WAN22 scheduler from vendored metadata (Diffusers-free).

    Source of truth is `model_index.json` + `scheduler/scheduler_config.json` shipped with the official repos.
    We do **not** silently fall back to unrelated schedulers (e.g., SD-style Euler) because WAN uses flow prediction.
    """

    import json
    import os

    if not metadata_dir:
        raise RuntimeError("WAN22 GGUF: metadata_dir is required to build the scheduler (missing WAN metadata).")

    raw_sampler = str(sampler or "").strip().lower()
    raw_scheduler = str(scheduler or "").strip().lower()
    if raw_sampler in {"", "inherit", "auto", "default"}:
        raw_sampler = ""
    if raw_scheduler in {"", "inherit", "auto", "default"}:
        raw_scheduler = ""

    vendor_dir = os.path.expanduser(str(metadata_dir))
    scheduler_dir = os.path.join(vendor_dir, "scheduler")
    if not os.path.isdir(scheduler_dir):
        parent = os.path.dirname(vendor_dir)
        scheduler_dir = os.path.join(parent, "scheduler") if parent else ""
        if scheduler_dir and os.path.isdir(scheduler_dir):
            vendor_dir = parent
        else:
            raise RuntimeError(
                f"WAN22 GGUF: metadata_dir must be a diffusers repo dir (or a tokenizer dir whose parent is one): {metadata_dir!r}"
            )

    config_path = None
    for fname in ("scheduler_config.json", "config.json"):
        candidate = os.path.join(vendor_dir, "scheduler", fname)
        if os.path.isfile(candidate):
            config_path = candidate
            break
    if not config_path:
        raise RuntimeError(f"WAN22 GGUF: scheduler config not found under: {vendor_dir!r} (expected scheduler_config.json)")

    try:
        config_raw = json.loads(open(config_path, encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001 - strict decode
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler config JSON: {config_path}: {exc}") from exc
    if not isinstance(config_raw, dict):
        raise RuntimeError(f"WAN22 GGUF: scheduler config must be a JSON object: {config_path}")

    class_name = str(config_raw.get("_class_name") or "").strip()
    if not class_name:
        raise RuntimeError(f"WAN22 GGUF: scheduler config missing _class_name: {config_path}")

    # Parse sampler hints from payload. WAN22 scheduler semantics are metadata-driven; when
    # users pass a non-UniPC sampler name (e.g., "euler"), treat it as a legacy no-op and
    # keep scheduler behavior defined by `scheduler_config.json`.
    if raw_sampler:
        parts = raw_sampler.split()
        if len(parts) > 2:
            raise RuntimeError(
                f"WAN22 GGUF: invalid sampler={sampler!r} (expected e.g. 'uni-pc' or 'uni-pc bh2')."
            )
        sampler_name = parts[0]
        sampler_solver = parts[1] if len(parts) == 2 else None

        if class_name == "UniPCMultistepScheduler":
            if sampler_name == "uni-pc":
                config_solver = str(config_raw.get("solver_type") or "").strip().lower() or None
                if sampler_solver is not None:
                    if config_solver is None:
                        raise RuntimeError(
                            f"WAN22 GGUF: sampler={sampler!r} specifies solver_type={sampler_solver!r}, "
                            f"but scheduler_config has no solver_type: {config_path}"
                        )
                    if sampler_solver != config_solver:
                        raise RuntimeError(
                            f"WAN22 GGUF: sampler={sampler!r} solver_type mismatch "
                            f"(requested={sampler_solver!r} config={config_solver!r})."
                        )
        else:
            raise RuntimeError(
                f"WAN22 GGUF: sampler override is not supported for metadata scheduler {class_name!r}. "
                "Use the defaults from scheduler_config.json."
            )

    # Note: `scheduler` is a legacy UI knob (sigma ladder families) in other engines.
    # WAN22 stage sampling uses the diffusers `scheduler_config.json` as source of truth.

    from .scheduler import build_wan_unipc_flow_scheduler

    if class_name != "UniPCMultistepScheduler":
        raise RuntimeError(
            f"WAN22 GGUF: unsupported metadata scheduler {class_name!r} in {config_path}; expected UniPCMultistepScheduler."
        )

    return build_wan_unipc_flow_scheduler(
        steps=max(1, int(steps)),
        vendor_dir=vendor_dir,
        flow_shift=float(flow_shift),
    )


def resolve_init_noise_sigma(scheduler: Any) -> float:
    """Return the scheduler-defined initial noise sigma (Diffusers parity).

    Diffusers pipelines scale the initial Gaussian noise by `scheduler.init_noise_sigma`.
    WAN22 GGUF uses the same behavior; `WAN_FLOW_MULTIPLIER` is only for model timestep inputs.
    """

    raw = getattr(scheduler, "init_noise_sigma", None)
    if raw is None:
        return 1.0
    try:
        val = float(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler.init_noise_sigma={raw!r}") from exc
    if not math.isfinite(val) or val <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler.init_noise_sigma={raw!r} (expected finite > 0)")
    return val


def _assert_finite_tensor(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    stage_name: str,
    local_step: int,
    total_steps: int,
    global_idx: int,
    timestep: Any,
) -> None:
    if torch.isfinite(tensor).all():
        return
    bad = int((~torch.isfinite(tensor)).sum().item())
    try:
        timestep_repr = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    except Exception:
        timestep_repr = str(timestep)
    raise RuntimeError(
        "WAN22 GGUF: non-finite tensor in stage sampling "
        f"(stage={stage_name} step={int(local_step)}/{int(total_steps)} idx={int(global_idx)} "
        f"timestep={timestep_repr} tensor={tensor_name} bad={bad}; "
        f"{summarize_numerics(tensor, name=tensor_name)})."
    )


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
    c_dst = int(target_geom.in_channels)
    if c_src == c_dst:
        return ensure_latent_shape(latents, target_geom)
    if c_src >= 16 and c_dst == 16:
        sliced = latents[:, :16, ...] if resolve_i2v_order() == "lat_first" else latents[:, -16:, ...]
        return ensure_latent_shape(sliced, target_geom)
    if c_src == 16 and c_dst == 36:
        raise RuntimeError(
            "WAN22 GGUF: cannot assemble I2V (Cin=36) state from 16-channel latents alone. "
            "Build the full I2V state explicitly (noise latents + mask4 + image_latents) in the img2vid runner."
        )
    raise RuntimeError(f"Cannot adapt latent channels from {c_src} to {c_dst}; unsupported hand-off configuration")



def build_i2v_mask4(
    *,
    batch: int,
    num_frames: int,
    latent_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    scale_factor_temporal: int = 4,
) -> torch.Tensor:
    """Build the 4-channel I2V mask (Diffusers-compatible).

    Diffusers (WanImageToVideoPipeline.prepare_latents) builds a 4-channel mask by:
    - creating a 1-channel per-frame mask at *output* time resolution,
    - repeating the first frame mask by `scale_factor_temporal`,
    - reshaping into `[B, 4, T_lat, H_lat, W_lat]` where `T_lat=(T_out-1)//scale+1`.

    For the I2V case (no `last_image`), the mask is 1 on the first latent-time chunk and 0 elsewhere.
    """

    if num_frames <= 0:
        raise RuntimeError(f"build_i2v_mask4: num_frames must be > 0, got {num_frames}")
    if scale_factor_temporal <= 0:
        raise RuntimeError(f"build_i2v_mask4: scale_factor_temporal must be > 0, got {scale_factor_temporal}")
    if num_frames % scale_factor_temporal != 1:
        raise RuntimeError(
            "build_i2v_mask4: num_frames must satisfy num_frames % scale_factor_temporal == 1 "
            f"(num_frames={num_frames} scale={scale_factor_temporal})"
        )

    expected_latent = int((int(num_frames) - 1) // int(scale_factor_temporal) + 1)
    if int(latent_frames) != expected_latent:
        raise RuntimeError(
            "build_i2v_mask4: latent_frames mismatch "
            f"(latent_frames={int(latent_frames)} expected={expected_latent} from num_frames={num_frames} scale={scale_factor_temporal})"
        )

    # mask_lat_size: [B,1,T_out,H_lat,W_lat]
    mask_lat_size = torch.ones((int(batch), 1, int(num_frames), int(height), int(width)), device=device, dtype=dtype)
    # I2V: only first frame is conditioned
    if int(num_frames) > 1:
        mask_lat_size[:, :, 1:int(num_frames)] = 0

    # Expand first frame to cover the first latent-time chunk
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=int(scale_factor_temporal))
    mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, ...]], dim=2)

    expected_frames = int(latent_frames) * int(scale_factor_temporal)
    if int(mask_lat_size.shape[2]) != expected_frames:
        raise RuntimeError(
            "build_i2v_mask4: internal frame count mismatch after expansion "
            f"(got={int(mask_lat_size.shape[2])} expected={expected_frames})"
        )

    # Reshape and transpose to `[B,4,T_lat,H_lat,W_lat]`
    mask_lat_size = mask_lat_size.view(int(batch), -1, int(scale_factor_temporal), int(height), int(width))
    mask_lat_size = mask_lat_size.transpose(1, 2).contiguous()
    if tuple(mask_lat_size.shape) != (int(batch), int(scale_factor_temporal), int(latent_frames), int(height), int(width)):
        raise RuntimeError(
            "build_i2v_mask4: unexpected output shape "
            f"(got={tuple(mask_lat_size.shape)} expected={(int(batch), int(scale_factor_temporal), int(latent_frames), int(height), int(width))})"
        )
    return mask_lat_size


def assemble_i2v_state(
    latents: torch.Tensor,
    *,
    mask4: torch.Tensor,
    image_latents: torch.Tensor,
    expected_cin: int,
    logger: logging.Logger | None,
) -> torch.Tensor:
    """Assemble I2V model input state to match expected Cin for patch embedding.

    This is the strict, canonical assembly for WAN I2V:
    - `latents`: noise latents (16ch) at latent time resolution
    - `mask4`: 4-channel first-frame mask at latent time resolution
    - `image_latents`: VAE-encoded video_condition latents (16ch) at latent time resolution
    """

    if latents.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D latents [B,C,T,H,W], got {tuple(latents.shape)}")
    if mask4.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D mask4 [B,4,T,H,W], got {tuple(mask4.shape)}")
    if image_latents.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D image_latents [B,16,T,H,W], got {tuple(image_latents.shape)}")

    b, c_lat, t, h, w = latents.shape
    if mask4.shape[0] != b or mask4.shape[2:] != (t, h, w):
        raise RuntimeError(
            "assemble_i2v_state: mask4 shape mismatch "
            f"(mask4={tuple(mask4.shape)} latents={tuple(latents.shape)})"
        )
    if image_latents.shape[0] != b or image_latents.shape[2:] != (t, h, w):
        raise RuntimeError(
            "assemble_i2v_state: image_latents shape mismatch "
            f"(image_latents={tuple(image_latents.shape)} latents={tuple(latents.shape)})"
        )
    if int(mask4.shape[1]) != 4:
        raise RuntimeError(f"assemble_i2v_state: expected mask4 to have 4 channels, got {int(mask4.shape[1])}")
    if int(image_latents.shape[1]) != 16:
        raise RuntimeError(
            f"assemble_i2v_state: expected image_latents to have 16 channels, got {int(image_latents.shape[1])}"
        )

    expected = int(c_lat) + 4 + 16
    if int(expected_cin) != expected:
        raise RuntimeError(
            "assemble_i2v_state: expected_cin mismatch "
            f"(expected_cin={int(expected_cin)} expected={expected} from latents={int(c_lat)} + mask4 + img16)"
        )

    order = resolve_i2v_order()
    if order == "lat_first":
        assembled = torch.cat([latents, mask4, image_latents], dim=1)
        layout = f"[lat{int(c_lat)} + mask4 + img16]"
    else:
        assembled = torch.cat([mask4, image_latents, latents], dim=1)
        layout = f"[mask4 + img16 + lat{int(c_lat)}]"
    if int(assembled.shape[1]) != int(expected_cin):
        raise RuntimeError(
            f"assemble_i2v_state: produced C={int(assembled.shape[1])}, expected C_in={int(expected_cin)} ({layout})."
        )
    if logger is not None:
        logger.info("[wan22.gguf] i2v assemble: order=%s %s → C=%d", order, layout, int(assembled.shape[1]))
    return assembled


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
    metadata_dir: Optional[str] = None,
    scheduler_obj: Any | None = None,
    timestep_start: int = 0,
    timestep_end: Optional[int] = None,
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
        metadata_dir=metadata_dir,
        scheduler_obj=scheduler_obj,
        timestep_start=timestep_start,
        timestep_end=timestep_end,
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
    metadata_dir: Optional[str] = None,
    scheduler_obj: Any | None = None,
    timestep_start: int = 0,
    timestep_end: Optional[int] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = "stage",
    emit_logs: bool = True,
):
    log = get_logger(logger)
    scheduler_state_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    t_lat, h_lat, w_lat = latent_dimensions(geom)
    steps = max(int(steps), 1)

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("WAN22: expected model with .config (WanTransformer2DModel)")
    cin = int(getattr(cfg, "in_channels", 0) or 0)
    cout = int(getattr(cfg, "latent_channels", 0) or 0)
    if cin <= 0 or cout <= 0:
        raise RuntimeError(f"WAN22: invalid model channels (in_channels={cin}, latent_channels={cout})")

    if int(geom.in_channels) != cin:
        raise RuntimeError(
            f"WAN22: geometry/model mismatch (geom.cin={int(geom.in_channels)} vs model.in_channels={cin})."
        )

    scheduler = scheduler_obj
    if scheduler is None:
        if metadata_dir is None:
            raise RuntimeError("WAN22 GGUF: metadata_dir is required for stage sampling (missing WAN metadata).")
        scheduler = make_scheduler(
            steps,
            metadata_dir=metadata_dir,
            flow_shift=flow_shift,
            sampler=sampler_name,
            scheduler=scheduler_name,
        )

    timesteps = scheduler.timesteps
    total_all = len(timesteps)

    start = int(timestep_start or 0)
    end = int(timestep_end) if timestep_end is not None else int(total_all)
    if start < 0 or end < 0 or start > end or end > total_all:
        raise RuntimeError(
            f"WAN22 GGUF: invalid timestep slice start={start} end={end} (timesteps={total_all})."
        )
    total = int(end - start)
    if total <= 0:
        raise RuntimeError("WAN22 GGUF: timestep slice is empty (no steps to run).")
    if scheduler_obj is not None and int(steps) != total:
        raise RuntimeError(
            f"WAN22 GGUF: step count mismatch for stage {stage_name!r} (steps={int(steps)} slice={total})."
        )
    if state_init is None and start != 0:
        raise RuntimeError("WAN22 GGUF: state_init is required when starting from a non-zero timestep index.")

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) not in (total_all, total_all + 1):
        raise RuntimeError(
            f"WAN22 GGUF: scheduler {scheduler.__class__.__name__} is missing a usable sigma ladder "
            f"(sigmas_len={len(sigmas) if sigmas is not None else None} timesteps={total_all})."
        )

    batch = int(state_init.shape[0]) if state_init is not None else 1
    shape = (batch, int(geom.in_channels), t_lat, h_lat, w_lat)

    if state_init is not None:
        state = ensure_latent_shape(state_init.to(device=device, dtype=scheduler_state_dtype), geom).clone()
    else:
        if cin != cout:
            raise RuntimeError(
                "WAN22 GGUF: state_init is required when model.in_channels != model.latent_channels "
                f"(in_channels={cin}, latent_channels={cout})."
            )
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            state = torch.randn(shape, generator=generator, device=device, dtype=scheduler_state_dtype)
        else:
            state = torch.randn(shape, device=device, dtype=scheduler_state_dtype)
        init_noise_sigma = resolve_init_noise_sigma(scheduler)
        state = state * float(init_noise_sigma)

    if scheduler_state_dtype != dtype:
        log.info(
            "[wan22.gguf] %s scheduler-state dtype island: model_dtype=%s scheduler_dtype=%s",
            stage_name,
            str(dtype),
            str(scheduler_state_dtype),
        )

    parity_idxs = {start, max(start, start + total // 2 - 1), max(start, end - 1)}

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

    order = resolve_i2v_order()

    for local_idx, idx in enumerate(range(start, end)):
        timestep = timesteps[idx]
        sigma_value = float(sigmas[idx])
        di_timestep = float(sigma_value) * float(flow_multiplier)
        step_number = int(local_idx + 1)

        _assert_finite_tensor(
            state,
            tensor_name="state_in",
            stage_name=stage_name,
            local_step=step_number,
            total_steps=total,
            global_idx=idx,
            timestep=timestep,
        )

        if log_sigmas_enabled() and idx in parity_idxs:
            log.info(
                "[wan22.gguf] %s t-in[%d/%d]: idx=%d sigma=%.6g flow_multiplier=%.1f di_timestep=%.6g sched_timestep=%s",
                stage_name,
                step_number,
                total,
                idx,
                float(sigma_value),
                float(flow_multiplier),
                float(di_timestep),
                str(timestep),
            )

        with torch.no_grad():
            if int(state.shape[1]) == cout:
                state_lat = state
                state_cond = None
            else:
                if int(state.shape[1]) != cin:
                    raise RuntimeError(
                        f"WAN22 GGUF: latent state channels C={int(state.shape[1])} does not match expected in_channels={cin}."
                    )

                # Inpainting-style I2V: state is [latents + conditioning], while the model predicts only the latent channels.
                if order == "lat_first":
                    state_lat = state[:, :cout, ...]
                    state_cond = state[:, cout:, ...]
                else:
                    state_cond = state[:, :-cout, ...]
                    state_lat = state[:, -cout:, ...]

            state_lat_scaled = state_lat
            scaler = getattr(scheduler, "scale_model_input", None)
            if callable(scaler):
                state_lat_scaled = scaler(state_lat, timestep)
            _assert_finite_tensor(
                state_lat_scaled,
                tensor_name="state_lat_scaled",
                stage_name=stage_name,
                local_step=step_number,
                total_steps=total,
                global_idx=idx,
                timestep=timestep,
            )

            state_lat_scaled_model = state_lat_scaled if state_lat_scaled.dtype == dtype else state_lat_scaled.to(dtype=dtype)
            state_cond_model: torch.Tensor | None = None
            if state_cond is not None:
                state_cond_model = state_cond if state_cond.dtype == dtype else state_cond.to(dtype=dtype)

            if state_cond_model is None:
                model_state = state_lat_scaled_model
            else:
                model_state = (
                    torch.cat([state_lat_scaled_model, state_cond_model], dim=1)
                    if order == "lat_first"
                    else torch.cat([state_cond_model, state_lat_scaled_model], dim=1)
                )

            if cfg_scale is None:
                eps_model = model(model_state, di_timestep, prompt_embeds)
                _assert_finite_tensor(
                    eps_model,
                    tensor_name="model_output",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
            else:
                x_in = torch.cat([model_state, model_state], dim=0)
                ctx_in = torch.cat([prompt_embeds, negative_embeds], dim=0)
                t_in = torch.full((x_in.shape[0],), float(di_timestep), device=device, dtype=torch.float32)
                v_pred = model(x_in, t_in, ctx_in)
                _assert_finite_tensor(
                    v_pred,
                    tensor_name="model_output_cfg_pair",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                eps_model = cfg_merge(v_uncond, v_cond, cfg_scale)
                _assert_finite_tensor(
                    eps_model,
                    tensor_name="cfg_merge_output",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )

            if eps_model.ndim != 5 or eps_model.shape[0] != state.shape[0] or eps_model.shape[2:] != state.shape[2:]:
                raise RuntimeError(
                    f"WAN22 GGUF: model output shape {tuple(eps_model.shape)} does not match latent state {tuple(state.shape)} "
                    f"(patch_size={geom.patch_kernel} grid={geom.grid})"
                )

            if int(eps_model.shape[1]) != cout:
                raise RuntimeError(
                    f"WAN22 GGUF: model output channels C={int(eps_model.shape[1])} does not match expected latent_channels={cout}."
                )
            eps = eps_model if eps_model.dtype == scheduler_state_dtype else eps_model.to(dtype=scheduler_state_dtype)

            if state_cond is None:
                out = scheduler.step(model_output=eps, timestep=timestep, sample=state_lat)
                state = out.prev_sample
                _assert_finite_tensor(
                    state,
                    tensor_name="scheduler_prev_sample",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
            else:
                if state_lat.shape != eps.shape:
                    raise RuntimeError(
                        f"WAN22 GGUF: model output shape {tuple(eps.shape)} does not match latent slice {tuple(state_lat.shape)} "
                        f"(patch_size={geom.patch_kernel} grid={geom.grid})"
                    )

                out = scheduler.step(model_output=eps, timestep=timestep, sample=state_lat)
                lat_next = out.prev_sample
                _assert_finite_tensor(
                    lat_next,
                    tensor_name="scheduler_prev_sample",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
                if order == "lat_first":
                    state = torch.cat([lat_next, state_cond], dim=1)
                else:
                    state = torch.cat([state_cond, lat_next], dim=1)
                _assert_finite_tensor(
                    state,
                    tensor_name="state_out",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )

            if log_numerics_enabled() and idx in parity_idxs:
                log.info(
                    "[wan22.gguf] %s numerics[%d/%d]: %s | %s",
                    stage_name,
                    step_number,
                    total,
                    summarize_numerics(eps, name="eps_step"),
                    summarize_numerics(state, name="state_step"),
                )

        pct = float(local_idx + 1) / float(max(1, total))
        if log_mem_interval is not None:
            n = int(log_mem_interval or 0)
            if n > 0 and ((local_idx + 1) % n) == 0:
                log_cuda_mem(logger, label=f"{stage_name}-step-{local_idx + 1}")

        now = time.perf_counter()
        step_dt = now - last
        elapsed = now - t0
        remain = max(0, total - (local_idx + 1))
        eta = (elapsed / max(1, local_idx + 1)) * remain
        last = now

        if emit_logs and ((local_idx + 1) % 5 == 0 or local_idx + 1 == total):
            log.info("[wan22.gguf] %s step %d/%d (%.1f%%)", stage_name.upper(), local_idx + 1, total, pct * 100.0)

        yield {
            "type": "progress",
            "stage": stage_name,
            "step": local_idx + 1,
            "total": total,
            "percent": pct,
            "eta_seconds": eta,
            "step_seconds": step_dt,
        }

    return state
