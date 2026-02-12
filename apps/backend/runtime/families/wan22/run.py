"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF run entrypoints (txt2vid/img2vid; batch + streaming).
Orchestrates text context, per-stage sampling, and VAE encode/decode (including file-VAE metadata config forwarding) while keeping GGUF support anchored in the shared quantization/ops layer.

Symbols (top-level; keep in sync; no ghosts):
- `_MemoryManagedModule` (class): Small adapter integrating plain nn.Modules with the Codex memory manager.
- `_try_set_cache_policy` (function): Configure GGUF dequant cache policy + limit when supported.
- `_try_clear_cache` (function): Clear GGUF dequant cache when supported.
- `_resolve_offload_level` (function): Resolve the effective offload profile level from the run config.
- `_require_flow_shift` (function): Validate that a stage has a usable flow_shift value (strict).
- `_parse_sampler` (function): Parse canonical WAN sampler strings (e.g. 'uni-pc bh2').
- `_build_shared_scheduler` (function): Build a single shared scheduler instance for high/low stage continuity.
- `_resolve_frame_counts` (function): Resolve output vs latent frame counts for the WAN VAE temporal scale.
- `_build_i2v_seed_state` (function): Build the initial I2V state `[lat16 + mask4 + img16]` (RNG noise scaled by `init_noise_sigma` + deterministic condition).
- `_extract_i2v_decode_latents` (function): Extract pure latent channels from I2V model state before VAE decode (order-aware `lat_first`/`lat_last`).
- `run_txt2vid` (function): Batch txt2vid runner; orchestrates text context, stage sampling, and VAE decode.
- `stream_txt2vid` (function): Streaming txt2vid generator; yields progress while sampling/decoding.
- `run_img2vid` (function): Batch img2vid runner; builds I2V conditioning + seeded noise state, runs stages, decodes frames (with explicit VAE config-dir forwarding).
- `stream_img2vid` (function): Streaming img2vid generator; yields progress while sampling/decoding (I2V conditioning + seeded noise state, with explicit VAE config-dir forwarding).
"""

from __future__ import annotations

import gc
import os
from typing import Any, Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled

from .config import (
    RunConfig,
    as_torch_dtype,
    resolve_device_name,
    resolve_i2v_order,
    resolve_wan_flow_multiplier,
)
from .diagnostics import cuda_empty_cache, get_logger, log_cuda_mem
from .sampling import (
    assemble_i2v_state,
    build_i2v_mask4,
    infer_patch_geometry,
    make_scheduler,
    resolve_init_noise_sigma,
    prepare_stage_seed_latents,
    resize_latents_hw,
    sample_stage_latents,
    sample_stage_latents_generator,
)
from .sdpa import set_sdpa_settings
from .stage_loader import load_stage_model_from_gguf, pick_stage_gguf
from .text_context import get_text_context
from .vae_io import decode_latents_to_frames, vae_encode_video_condition


class _MemoryManagedModule:
    """Tiny wrapper to integrate plain nn.Modules with the Codex memory manager.

    We intentionally keep this minimal (no patch plumbing during device moves).
    Stage-level LoRAs (when configured) are applied at load time by `stage_loader.load_stage_model_from_gguf(...)`.
    """

    def __init__(self, model: torch.nn.Module, *, load_device: torch.device) -> None:
        self.model = model
        self.load_device = load_device

    def model_dtype(self):  # noqa: ANN001 - matches memory manager dynamic protocol
        # Keep the model's existing dtype (GGUF loader already created weights correctly).
        return None

    def codex_patch_model(self, target_device: torch.device | None = None):  # noqa: ANN001 - protocol
        if target_device is None:
            return self.model
        try:
            self.model.to(target_device, non_blocking=True)
        except TypeError:
            self.model.to(target_device)
        return self.model


def _try_set_cache_policy(policy: Optional[str], limit_mb: Optional[int]) -> None:
    if policy is None:
        return
    lim = int(limit_mb or 0)
    pol = str(policy).strip().lower()
    if pol in {"none", "", "off"} or lim <= 0:
        return

    try:
        from apps.backend.runtime.ops.operations_gguf import set_cache_policy
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "GGUF dequant cache requested but not available in this build (set_cache_policy missing)."
        ) from exc

    set_cache_policy(pol, lim)


def _try_clear_cache() -> None:
    try:
        from apps.backend.runtime.ops.operations_gguf import clear_cache
    except Exception:
        return

    try:
        clear_cache()
    except Exception:
        return


def _resolve_offload_level(cfg: RunConfig) -> int:
    if cfg.offload_level is not None:
        if isinstance(cfg.offload_level, bool) or not isinstance(cfg.offload_level, int):
            raise RuntimeError(
                f"WAN22 GGUF: offload_level must be an integer when provided, got {type(cfg.offload_level).__name__}."
            )
        if cfg.offload_level < 0:
            raise RuntimeError(f"WAN22 GGUF: offload_level must be >= 0, got {cfg.offload_level}.")
        return cfg.offload_level
    aggressive_offload = getattr(cfg, "aggressive_offload", True)
    if not isinstance(aggressive_offload, bool):
        raise RuntimeError(
            "WAN22 GGUF: aggressive_offload must be a boolean in RunConfig "
            f"(got {type(aggressive_offload).__name__})."
        )
    return 3 if aggressive_offload else 0


def _require_flow_shift(stage: str, value: object | None) -> float:
    if value is None:
        raise RuntimeError(
            f"WAN22 GGUF stage '{stage}' is missing flow_shift. "
            "Provide an explicit stage override (extras.wan_high/wan_low.flow_shift) "
            "or ensure the engine resolves the default from the model's scheduler_config.json."
        )
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - strict input validation
        raise RuntimeError(f"WAN22 GGUF stage '{stage}' has invalid flow_shift: {value!r}") from exc


def _parse_sampler(value: object | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise RuntimeError(f"WAN22 GGUF: sampler must be a string when provided, got {value!r}.")
    raw = value.strip().lower()
    if not raw:
        return None, None
    parts = raw.split()
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise RuntimeError(f"WAN22 GGUF: invalid sampler={value!r} (expected e.g. 'uni-pc' or 'uni-pc bh2').")


def _build_shared_scheduler(
    cfg: RunConfig,
    *,
    steps_hi: int,
    steps_lo: int,
    sampler_hi: object | None,
    sampler_lo: object | None,
    scheduler_hi: object | None,
    scheduler_lo: object | None,
    flow_shift_hi: float,
    flow_shift_lo: float,
):
    if float(flow_shift_hi) != float(flow_shift_lo):
        raise RuntimeError(
            "WAN22 GGUF: high/low flow_shift mismatch. "
            f"High={flow_shift_hi} Low={flow_shift_lo}. Schedule must be continuous."
        )

    hi_name, hi_solver = _parse_sampler(sampler_hi)
    lo_name, lo_solver = _parse_sampler(sampler_lo)
    if hi_name and lo_name and hi_name != lo_name:
        raise RuntimeError(
            f"WAN22 GGUF: high/low sampler mismatch (high={sampler_hi!r} low={sampler_lo!r})."
        )
    if hi_solver and lo_solver and hi_solver != lo_solver:
        raise RuntimeError(
            f"WAN22 GGUF: high/low UniPC solver_type mismatch (high={sampler_hi!r} low={sampler_lo!r})."
        )

    total_steps = int(steps_hi) + int(steps_lo)
    if total_steps < 2:
        raise RuntimeError(f"WAN22 GGUF requires total steps >=2, got: {total_steps} ({steps_hi}+{steps_lo}).")

    if sampler_hi is not None and not isinstance(sampler_hi, str):
        raise RuntimeError(f"WAN22 GGUF: high sampler must be a string when provided, got {sampler_hi!r}.")
    if sampler_lo is not None and not isinstance(sampler_lo, str):
        raise RuntimeError(f"WAN22 GGUF: low sampler must be a string when provided, got {sampler_lo!r}.")
    if scheduler_hi is not None and not isinstance(scheduler_hi, str):
        raise RuntimeError(f"WAN22 GGUF: high scheduler must be a string when provided, got {scheduler_hi!r}.")
    if scheduler_lo is not None and not isinstance(scheduler_lo, str):
        raise RuntimeError(f"WAN22 GGUF: low scheduler must be a string when provided, got {scheduler_lo!r}.")

    sampler_eff = sampler_hi.strip() if isinstance(sampler_hi, str) and sampler_hi.strip() else (
        sampler_lo.strip() if isinstance(sampler_lo, str) and sampler_lo.strip() else None
    )
    scheduler_eff = scheduler_hi.strip() if isinstance(scheduler_hi, str) and scheduler_hi.strip() else (
        scheduler_lo.strip() if isinstance(scheduler_lo, str) and scheduler_lo.strip() else None
    )

    return make_scheduler(
        total_steps,
        metadata_dir=str(cfg.metadata_dir or ""),
        flow_shift=float(flow_shift_hi),
        sampler=sampler_eff,
        scheduler=scheduler_eff,
    ), total_steps


def _resolve_frame_counts(num_frames: int, *, logger: Any) -> tuple[int, int]:
    """Resolve (T_out, T_lat) for WAN video.

    Diffusers WAN pipelines enforce:
      - `num_frames % vae_scale_factor_temporal == 1`
      - `num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1`

    WAN video VAEs use a temporal scale factor of 4.
    """
    log = get_logger(logger)
    scale = 4
    requested = max(1, int(num_frames))
    effective = requested
    if effective % scale != 1:
        rounded = int(effective // scale * scale + 1)
        log.warning(
            "[wan22.gguf] num_frames=%d is incompatible with VAE temporal scale=%d; rounding to %d.",
            requested,
            scale,
            rounded,
        )
        effective = max(1, rounded)
    latent_frames = int((effective - 1) // scale + 1)
    return effective, latent_frames


def _build_i2v_seed_state(
    *,
    cfg: RunConfig,
    scheduler: Any,
    geom_hi: Any,
    latent_condition: torch.Tensor,
    num_frames: int,
    latent_frames: int,
    h_lat: int,
    w_lat: int,
    flow_multiplier: float,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
) -> torch.Tensor:
    """Build the initial I2V state `[lat16 + mask4 + img16]` (Diffusers-compatible).

    This is the critical ownership seam for WAN22 GGUF img2vid:
    - Noise latents must be seeded from RNG and scaled by `scheduler.init_noise_sigma`
      (Diffusers parity; do **not** multiply by the WAN flow multiplier).
    - The conditioning channels are constant across timesteps (mask4 + VAE-encoded video_condition).
    """

    log = get_logger(logger)

    cin = int(getattr(geom_hi, "in_channels", 0) or 0)
    if cin <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid geom_hi.in_channels={cin}")

    img = latent_condition
    if img.ndim == 4:
        img = img.unsqueeze(2)
    if img.ndim != 5:
        raise RuntimeError(f"WAN22 GGUF: I2V latent_condition must be 4D/5D, got {tuple(img.shape)}")
    img = resize_latents_hw(img, height=h_lat, width=w_lat).to(device=device, dtype=dtype)
    if int(img.shape[2]) != int(latent_frames):
        raise RuntimeError(
            "WAN22 GGUF: I2V latent_condition temporal mismatch "
            f"(got_T={int(img.shape[2])} expected_T_lat={int(latent_frames)})"
        )

    mask4 = build_i2v_mask4(
        batch=int(img.shape[0]),
        num_frames=int(num_frames),
        latent_frames=int(latent_frames),
        height=int(h_lat),
        width=int(w_lat),
        device=device,
        dtype=dtype,
        scale_factor_temporal=4,
    )

    c_lat = int(cin) - 4 - 16
    if c_lat != 16:
        raise RuntimeError(
            "WAN22 GGUF: unexpected I2V channel split "
            f"(cin={cin} implies latents={c_lat}, expected 16 for [lat16+mask4+img16])."
        )

    shape = (int(img.shape[0]), int(c_lat), int(latent_frames), int(h_lat), int(w_lat))
    seed_val = getattr(cfg, "seed", None)
    if seed_val is not None and int(seed_val) >= 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed_val))
        latents = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    else:
        latents = torch.randn(shape, device=device, dtype=dtype)

    init_noise_sigma = resolve_init_noise_sigma(scheduler)
    latents = latents * float(init_noise_sigma)

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) < 1:
        raise RuntimeError("WAN22 GGUF: scheduler is missing sigmas; cannot seed latents correctly.")
    sigma0 = float(sigmas[0])

    state = assemble_i2v_state(latents, mask4=mask4, image_latents=img, expected_cin=cin, logger=log)
    state = prepare_stage_seed_latents(state, geom_hi, logger=log)
    log.info(
        "[wan22.gguf] i2v seed: seed=%s init_noise_sigma=%.6g sigma0=%.6g flow_multiplier=%.1f",
        str(seed_val),
        float(init_noise_sigma),
        float(sigma0),
        float(flow_multiplier),
    )
    return state


def _extract_i2v_decode_latents(
    *,
    state: torch.Tensor,
    latent_channels: int,
    logger: Any,
) -> torch.Tensor:
    """Extract pure VAE latents from an I2V state tensor before decode."""
    log = get_logger(logger)
    if state.ndim != 5:
        raise RuntimeError(
            "WAN22 GGUF: expected 5D I2V state [B,C,T,H,W] before decode, "
            f"got shape={tuple(state.shape)}."
        )
    c_state = int(state.shape[1])
    c_lat = int(latent_channels)
    if c_lat <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid latent_channels for I2V decode extraction: {c_lat}.")
    if c_state == c_lat:
        return state
    c_cond = c_state - c_lat
    if c_cond != 20:
        raise RuntimeError(
            "WAN22 GGUF: cannot extract I2V decode latents from state with unexpected channel split "
            f"(state_C={c_state} latent_C={c_lat} cond_C={c_cond}; expected cond_C=20 for mask4+img16)."
        )
    order = resolve_i2v_order()
    lat = state[:, :c_lat, ...] if order == "lat_first" else state[:, -c_lat:, ...]
    log.info(
        "[wan22.gguf] i2v decode slice: order=%s state_C=%d -> latent_C=%d",
        order,
        c_state,
        c_lat,
    )
    return lat


def run_txt2vid(cfg: RunConfig, *, logger: Any = None, on_progress: Any = None) -> list[object]:
    log = get_logger(logger)
    hi_path = pick_stage_gguf(getattr(cfg.high, "model_dir", None) if cfg.high else None, stage="high")
    lo_path = pick_stage_gguf(getattr(cfg.low, "model_dir", None) if cfg.low else None, stage="low")
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, "sdpa_policy", None), getattr(cfg, "attn_chunk_size", None))
    _try_set_cache_policy(getattr(cfg, "gguf_cache_policy", None), getattr(cfg, "gguf_cache_limit_mb", None))

    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = resolve_device_name(getattr(cfg, "device", "auto"))
    dev = torch.device(dev_name)
    dt = as_torch_dtype(cfg.dtype)
    flow_multiplier = resolve_wan_flow_multiplier(str(cfg.metadata_dir or ""))

    lvl = _resolve_offload_level(cfg)

    # Load GGUF weights on CPU first; the memory manager will move to GPU right before sampling.
    hi_model = load_stage_model_from_gguf(
        hi_path,
        stage="high",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.high, "lora_path", None) if cfg.high else None),
        lora_weight=(getattr(cfg.high, "lora_weight", None) if cfg.high else None),
        logger=log,
    )
    hi_mm = _MemoryManagedModule(hi_model, load_device=dev)
    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.05)
        except Exception:
            pass

    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_t2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or dev_name
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    te_kernel_required = getattr(cfg, "te_kernel_required", False)
    if te_kernel_required is not None and not isinstance(te_kernel_required, bool):
        raise RuntimeError(
            "WAN22 GGUF: te_kernel_required must be boolean when provided "
            f"(got {type(te_kernel_required).__name__})."
        )
    if bool(te_kernel_required):
        te_impl_val = "cuda_fp8"
    te_required = te_impl_val == "cuda_fp8"
    if te_required:
        te_dev_eff = "cuda"
    log.info(
        "[wan22.gguf] offload profile: level=%s te_device=%s te_impl=%s te_required=%s",
        lvl,
        te_dev_eff,
        te_impl_val,
        str(bool(te_required)).lower(),
    )

    t_out, t_lat = _resolve_frame_counts(int(cfg.num_frames), logger=log)
    log.info("[wan22.gguf] frames: requested=%d effective=%d latent=%d", int(cfg.num_frames), t_out, t_lat)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = int(t_lat)

    prompt_embeds, negative_embeds = get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=model_key,
        metadata_dir=cfg.metadata_dir,
        logger=log,
        offload_after=smart_offload_enabled(),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    log.info(
        "[wan22.gguf] HIGH geom: grid=%s kernel=%s cin=%d",
        geom_hi.grid,
        geom_hi.patch_kernel,
        geom_hi.in_channels,
    )
    log_cuda_mem(log, label="after-high-setup")
    if lvl >= 3:
        cuda_empty_cache(log, label="pre-high")
    if on_progress:
        try:
            on_progress(stage="prepare", step=1, total=1, percent=0.15)
        except Exception:
            pass

    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

    scheduler, total_steps = _build_shared_scheduler(
        cfg,
        steps_hi=steps_hi,
        steps_lo=steps_lo,
        sampler_hi=sampler_hi,
        sampler_lo=sampler_lo,
        scheduler_hi=sched_hi,
        scheduler_lo=sched_lo,
        flow_shift_hi=flow_shift_hi_value,
        flow_shift_lo=flow_shift_lo_value,
    )
    log.info("[wan22.gguf] schedule: steps_total=%d steps_high=%d steps_low=%d", total_steps, steps_hi, steps_lo)
    log.info(
        "[wan22.gguf] HIGH: steps=%s sampler=%s scheduler=%s cfg_scale=%s seed=%s",
        steps_hi,
        sampler_hi,
        sched_hi,
        (getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        cfg.seed,
    )

    memory_management.manager.load_model(hi_mm)
    latents_hi = sample_stage_latents(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=0,
        timestep_end=steps_hi,
        seed=cfg.seed,
        state_init=None,
        on_progress=(lambda **p: on_progress(stage="high", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=flow_multiplier,
        stage_name="high",
    )

    memory_management.manager.unload_model(hi_mm)
    del hi_mm
    del hi_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-high")

    lo_model = load_stage_model_from_gguf(
        lo_path,
        stage="low",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.low, "lora_path", None) if cfg.low else None),
        lora_weight=(getattr(cfg.low, "lora_weight", None) if cfg.low else None),
        logger=log,
    )
    latent_channels_lo = int(getattr(getattr(lo_model, "config", None), "latent_channels", 0) or 0)
    if latent_channels_lo <= 0:
        raise RuntimeError(
            "WAN22 GGUF: low-stage model is missing a valid latent_channels config for I2V decode "
            f"(got {latent_channels_lo})."
        )
    lo_mm = _MemoryManagedModule(lo_model, load_device=dev)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    log.info(
        "[wan22.gguf] LOW geom: grid=%s kernel=%s cin=%d",
        geom_lo.grid,
        geom_lo.patch_kernel,
        geom_lo.in_channels,
    )

    seed_latents = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)
    if tuple(seed_latents.shape) != tuple(latents_hi.shape):
        raise RuntimeError(
            "WAN22 GGUF: high/low latent shapes differ after hand-off; cannot maintain a continuous schedule. "
            f"high={tuple(latents_hi.shape)} low_init={tuple(seed_latents.shape)}"
        )
    log.info(
        "[wan22.gguf] LOW: steps=%s sampler=%s scheduler=%s cfg_scale=%s",
        steps_lo,
        sampler_lo,
        sched_lo,
        (getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
    )

    memory_management.manager.load_model(lo_mm)
    latents_lo = sample_stage_latents(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=steps_hi,
        timestep_end=total_steps,
        seed=None,
        state_init=seed_latents,
        on_progress=(lambda **p: on_progress(stage="low", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=flow_multiplier,
        stage_name="low",
    )

    # Free the LOW stage weights before VAE decode (decode can be very VRAM-hungry).
    memory_management.manager.unload_model(lo_mm)
    del lo_mm
    del lo_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-low")

    frames = decode_latents_to_frames(
        latents=latents_lo,
        model_dir=os.path.dirname(lo_path),
        cfg=cfg,
        logger=log,
        expected_frames=t_out,
    )
    if lvl >= 3:
        cuda_empty_cache(log, label="after-decode")
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames


def stream_txt2vid(cfg: RunConfig, *, logger: Any = None):
    log = get_logger(logger)
    hi_path = pick_stage_gguf(getattr(cfg.high, "model_dir", None) if cfg.high else None, stage="high")
    lo_path = pick_stage_gguf(getattr(cfg.low, "model_dir", None) if cfg.low else None, stage="low")
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, "sdpa_policy", None), getattr(cfg, "attn_chunk_size", None))
    _try_set_cache_policy(getattr(cfg, "gguf_cache_policy", None), getattr(cfg, "gguf_cache_limit_mb", None))

    dev_name = resolve_device_name(getattr(cfg, "device", "auto"))
    dev = torch.device(dev_name)
    dt = as_torch_dtype(cfg.dtype)
    flow_multiplier = resolve_wan_flow_multiplier(str(cfg.metadata_dir or ""))
    lvl = _resolve_offload_level(cfg)

    hi_model = load_stage_model_from_gguf(
        hi_path,
        stage="high",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.high, "lora_path", None) if cfg.high else None),
        lora_weight=(getattr(cfg.high, "lora_weight", None) if cfg.high else None),
        logger=log,
    )
    hi_mm = _MemoryManagedModule(hi_model, load_device=dev)
    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_t2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or dev_name
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    te_kernel_required = getattr(cfg, "te_kernel_required", False)
    if te_kernel_required is not None and not isinstance(te_kernel_required, bool):
        raise RuntimeError(
            "WAN22 GGUF: te_kernel_required must be boolean when provided "
            f"(got {type(te_kernel_required).__name__})."
        )
    if bool(te_kernel_required):
        te_impl_val = "cuda_fp8"
    te_required = te_impl_val == "cuda_fp8"
    if te_required:
        te_dev_eff = "cuda"

    prompt_embeds, negative_embeds = get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=model_key,
        metadata_dir=cfg.metadata_dir,
        logger=log,
        offload_after=smart_offload_enabled(),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=te_impl_val,
        te_kernel_required=te_required,
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    t_out, t_lat = _resolve_frame_counts(int(cfg.num_frames), logger=log)
    log.info("[wan22.gguf] frames: requested=%d effective=%d latent=%d", int(cfg.num_frames), t_out, t_lat)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = int(t_lat)
    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

    scheduler, total_steps = _build_shared_scheduler(
        cfg,
        steps_hi=steps_hi,
        steps_lo=steps_lo,
        sampler_hi=sampler_hi,
        sampler_lo=sampler_lo,
        scheduler_hi=sched_hi,
        scheduler_lo=sched_lo,
        flow_shift_hi=flow_shift_hi_value,
        flow_shift_lo=flow_shift_lo_value,
    )
    log.info("[wan22.gguf] schedule: steps_total=%d steps_high=%d steps_low=%d", total_steps, steps_hi, steps_lo)

    memory_management.manager.load_model(hi_mm)
    latents_hi = yield from sample_stage_latents_generator(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=0,
        timestep_end=steps_hi,
        seed=cfg.seed,
        state_init=None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=flow_multiplier,
        stage_name="high",
        emit_logs=False,
    )

    memory_management.manager.unload_model(hi_mm)
    del hi_mm
    del hi_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-high")

    lo_model = load_stage_model_from_gguf(
        lo_path,
        stage="low",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.low, "lora_path", None) if cfg.low else None),
        lora_weight=(getattr(cfg.low, "lora_weight", None) if cfg.low else None),
        logger=log,
    )
    latent_channels_lo = int(getattr(getattr(lo_model, "config", None), "latent_channels", 0) or 0)
    if latent_channels_lo <= 0:
        raise RuntimeError(
            "WAN22 GGUF: low-stage model is missing a valid latent_channels config for I2V decode "
            f"(got {latent_channels_lo})."
        )
    lo_mm = _MemoryManagedModule(lo_model, load_device=dev)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_latents = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)
    if tuple(seed_latents.shape) != tuple(latents_hi.shape):
        raise RuntimeError(
            "WAN22 GGUF: high/low latent shapes differ after hand-off; cannot maintain a continuous schedule. "
            f"high={tuple(latents_hi.shape)} low_init={tuple(seed_latents.shape)}"
        )

    memory_management.manager.load_model(lo_mm)
    latents_lo = yield from sample_stage_latents_generator(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=steps_hi,
        timestep_end=total_steps,
        seed=None,
        state_init=seed_latents,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=flow_multiplier,
        stage_name="low",
        emit_logs=False,
    )
    latents_lo_decode = _extract_i2v_decode_latents(
        state=latents_lo,
        latent_channels=latent_channels_lo,
        logger=log,
    )

    # Free the LOW stage weights before VAE decode.
    memory_management.manager.unload_model(lo_mm)
    del lo_mm
    del lo_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-low")

    frames = decode_latents_to_frames(
        latents=latents_lo_decode,
        model_dir=os.path.dirname(lo_path),
        cfg=cfg,
        logger=log,
        expected_frames=t_out,
    )
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames}


def run_img2vid(cfg: RunConfig, *, logger: Any = None, on_progress: Any = None) -> list[object]:
    log = get_logger(logger)
    hi_path = pick_stage_gguf(getattr(cfg.high, "model_dir", None) if cfg.high else None, stage="high")
    lo_path = pick_stage_gguf(getattr(cfg.low, "model_dir", None) if cfg.low else None, stage="low")
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, "sdpa_policy", None), getattr(cfg, "attn_chunk_size", None))
    _try_set_cache_policy(getattr(cfg, "gguf_cache_policy", None), getattr(cfg, "gguf_cache_limit_mb", None))

    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = resolve_device_name(getattr(cfg, "device", "auto"))
    dev = torch.device(dev_name)
    dt = as_torch_dtype(cfg.dtype)
    flow_multiplier = resolve_wan_flow_multiplier(str(cfg.metadata_dir or ""))
    lvl = _resolve_offload_level(cfg)

    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_i2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or dev_name
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    te_kernel_required = getattr(cfg, "te_kernel_required", False)
    if te_kernel_required is not None and not isinstance(te_kernel_required, bool):
        raise RuntimeError(
            "WAN22 GGUF: te_kernel_required must be boolean when provided "
            f"(got {type(te_kernel_required).__name__})."
        )
    if bool(te_kernel_required):
        te_impl_val = "cuda_fp8"
    te_required = te_impl_val == "cuda_fp8"
    if te_required:
        te_dev_eff = "cuda"
    log.info(
        "[wan22.gguf] offload profile: level=%s te_device=%s te_impl=%s te_required=%s",
        lvl,
        te_dev_eff,
        te_impl_val,
        str(bool(te_required)).lower(),
    )

    t_out, t_lat = _resolve_frame_counts(int(cfg.num_frames), logger=log)
    log.info("[wan22.gguf] frames: requested=%d effective=%d latent=%d", int(cfg.num_frames), t_out, t_lat)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = int(t_lat)

    # Encode conditioning video *before* loading the text encoder on CUDA to avoid allocator fragmentation
    # causing large conv3d workspace allocations to fail.
    #
    # Diffusers I2V condition is a video where:
    # - frame 0 = init image
    # - frames 1.. = 0 (0.5 gray in [0,1] space), then VAE-encoded deterministically (mode/argmax)
    latent_condition = vae_encode_video_condition(
        cfg.init_image,
        num_frames=t_out,
        height=int(cfg.height),
        width=int(cfg.width),
        device=dev_name,
        dtype=cfg.dtype,
        vae_dir=cfg.vae_dir,
        vae_config_dir=cfg.vae_config_dir,
        logger=log,
    )
    if latent_condition.ndim == 4:
        latent_condition = latent_condition.unsqueeze(2)
    latent_condition = resize_latents_hw(latent_condition, height=h_lat, width=w_lat)
    if int(latent_condition.shape[2]) != int(t):
        raise RuntimeError(
            "WAN22 GGUF: unexpected latent_condition temporal size after VAE encode "
            f"(got_T={int(latent_condition.shape[2])} expected_T_lat={int(t)})"
        )

    prompt_embeds, negative_embeds = get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=model_key,
        metadata_dir=cfg.metadata_dir,
        logger=log,
        offload_after=smart_offload_enabled(),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    hi_model = load_stage_model_from_gguf(
        hi_path,
        stage="high",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.high, "lora_path", None) if cfg.high else None),
        lora_weight=(getattr(cfg.high, "lora_weight", None) if cfg.high else None),
        logger=log,
    )
    hi_mm = _MemoryManagedModule(hi_model, load_device=dev)
    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.05)
        except Exception:
            pass

    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)

    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

    scheduler, total_steps = _build_shared_scheduler(
        cfg,
        steps_hi=steps_hi,
        steps_lo=steps_lo,
        sampler_hi=sampler_hi,
        sampler_lo=sampler_lo,
        scheduler_hi=sched_hi,
        scheduler_lo=sched_lo,
        flow_shift_hi=flow_shift_hi_value,
        flow_shift_lo=flow_shift_lo_value,
    )
    log.info("[wan22.gguf] schedule: steps_total=%d steps_high=%d steps_low=%d", total_steps, steps_hi, steps_lo)

    seed_hi = _build_i2v_seed_state(
        cfg=cfg,
        scheduler=scheduler,
        geom_hi=geom_hi,
        latent_condition=latent_condition,
        num_frames=t_out,
        latent_frames=t,
        h_lat=h_lat,
        w_lat=w_lat,
        flow_multiplier=flow_multiplier,
        device=dev,
        dtype=dt,
        logger=log,
    )

    memory_management.manager.load_model(hi_mm)
    latents_hi = sample_stage_latents(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=0,
        timestep_end=steps_hi,
        seed=None,
        state_init=seed_hi,
        on_progress=(lambda **p: on_progress(stage="high", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=flow_multiplier,
        stage_name="high",
    )

    memory_management.manager.unload_model(hi_mm)
    del hi_mm
    del hi_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(logger=log, label="after-high")

    lo_model = load_stage_model_from_gguf(
        lo_path,
        stage="low",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.low, "lora_path", None) if cfg.low else None),
        lora_weight=(getattr(cfg.low, "lora_weight", None) if cfg.low else None),
        logger=log,
    )
    lo_mm = _MemoryManagedModule(lo_model, load_device=dev)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_lo = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)
    if tuple(seed_lo.shape) != tuple(latents_hi.shape):
        raise RuntimeError(
            "WAN22 GGUF: high/low latent shapes differ after hand-off; cannot maintain a continuous schedule. "
            f"high={tuple(latents_hi.shape)} low_init={tuple(seed_lo.shape)}"
        )

    memory_management.manager.load_model(lo_mm)
    latents_lo = sample_stage_latents(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=steps_hi,
        timestep_end=total_steps,
        seed=None,
        state_init=seed_lo,
        on_progress=(lambda **p: on_progress(stage="low", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=flow_multiplier,
        stage_name="low",
    )
    latents_lo_decode = _extract_i2v_decode_latents(
        state=latents_lo,
        latent_channels=latent_channels_lo,
        logger=log,
    )

    # Free the LOW stage weights before VAE decode.
    memory_management.manager.unload_model(lo_mm)
    del lo_mm
    del lo_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-low")

    frames = decode_latents_to_frames(
        latents=latents_lo_decode,
        model_dir=os.path.dirname(lo_path),
        cfg=cfg,
        logger=log,
        expected_frames=t_out,
    )
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames


def stream_img2vid(cfg: RunConfig, *, logger: Any = None):
    log = get_logger(logger)
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")

    hi_path = pick_stage_gguf(getattr(cfg.high, "model_dir", None) if cfg.high else None, stage="high")
    lo_path = pick_stage_gguf(getattr(cfg.low, "model_dir", None) if cfg.low else None, stage="low")
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")

    set_sdpa_settings(getattr(cfg, "sdpa_policy", None), getattr(cfg, "attn_chunk_size", None))
    _try_set_cache_policy(getattr(cfg, "gguf_cache_policy", None), getattr(cfg, "gguf_cache_limit_mb", None))

    dev_name = resolve_device_name(getattr(cfg, "device", "auto"))
    dev = torch.device(dev_name)
    dt = as_torch_dtype(cfg.dtype)
    flow_multiplier = resolve_wan_flow_multiplier(str(cfg.metadata_dir or ""))
    lvl = _resolve_offload_level(cfg)
    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_i2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or dev_name
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    te_kernel_required = getattr(cfg, "te_kernel_required", False)
    if te_kernel_required is not None and not isinstance(te_kernel_required, bool):
        raise RuntimeError(
            "WAN22 GGUF: te_kernel_required must be boolean when provided "
            f"(got {type(te_kernel_required).__name__})."
        )
    if bool(te_kernel_required):
        te_impl_val = "cuda_fp8"
    te_required = te_impl_val == "cuda_fp8"
    if te_required:
        te_dev_eff = "cuda"

    t_out, t_lat = _resolve_frame_counts(int(cfg.num_frames), logger=log)
    log.info("[wan22.gguf] frames: requested=%d effective=%d latent=%d", int(cfg.num_frames), t_out, t_lat)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = int(t_lat)

    # Encode conditioning video before running the text encoder on CUDA to avoid allocator fragmentation
    # causing large conv3d workspace allocations to fail.
    latent_condition = vae_encode_video_condition(
        cfg.init_image,
        num_frames=t_out,
        height=int(cfg.height),
        width=int(cfg.width),
        device=dev_name,
        dtype=cfg.dtype,
        vae_dir=cfg.vae_dir,
        vae_config_dir=cfg.vae_config_dir,
        logger=log,
    )
    if latent_condition.ndim == 4:
        latent_condition = latent_condition.unsqueeze(2)
    latent_condition = resize_latents_hw(latent_condition, height=h_lat, width=w_lat)
    if int(latent_condition.shape[2]) != int(t):
        raise RuntimeError(
            "WAN22 GGUF: unexpected latent_condition temporal size after VAE encode "
            f"(got_T={int(latent_condition.shape[2])} expected_T_lat={int(t)})"
        )

    prompt_embeds, negative_embeds = get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=model_key,
        metadata_dir=cfg.metadata_dir,
        logger=log,
        offload_after=smart_offload_enabled(),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=te_impl_val,
        te_kernel_required=te_required,
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    hi_model = load_stage_model_from_gguf(
        hi_path,
        stage="high",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.high, "lora_path", None) if cfg.high else None),
        lora_weight=(getattr(cfg.high, "lora_weight", None) if cfg.high else None),
        logger=log,
    )
    hi_mm = _MemoryManagedModule(hi_model, load_device=dev)
    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

    scheduler, total_steps = _build_shared_scheduler(
        cfg,
        steps_hi=steps_hi,
        steps_lo=steps_lo,
        sampler_hi=sampler_hi,
        sampler_lo=sampler_lo,
        scheduler_hi=sched_hi,
        scheduler_lo=sched_lo,
        flow_shift_hi=flow_shift_hi_value,
        flow_shift_lo=flow_shift_lo_value,
    )
    log.info("[wan22.gguf] schedule: steps_total=%d steps_high=%d steps_low=%d", total_steps, steps_hi, steps_lo)

    seed_hi = _build_i2v_seed_state(
        cfg=cfg,
        scheduler=scheduler,
        geom_hi=geom_hi,
        latent_condition=latent_condition,
        num_frames=t_out,
        latent_frames=t,
        h_lat=h_lat,
        w_lat=w_lat,
        flow_multiplier=flow_multiplier,
        device=dev,
        dtype=dt,
        logger=log,
    )

    memory_management.manager.load_model(hi_mm)
    latents_hi = yield from sample_stage_latents_generator(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=0,
        timestep_end=steps_hi,
        seed=None,
        state_init=seed_hi,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=flow_multiplier,
        stage_name="high",
        emit_logs=False,
    )

    memory_management.manager.unload_model(hi_mm)
    del hi_mm
    del hi_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(logger=log, label="after-high")

    lo_model = load_stage_model_from_gguf(
        lo_path,
        stage="low",
        device=torch.device("cpu"),
        dtype=dt,
        lora_path=(getattr(cfg.low, "lora_path", None) if cfg.low else None),
        lora_weight=(getattr(cfg.low, "lora_weight", None) if cfg.low else None),
        logger=log,
    )
    lo_mm = _MemoryManagedModule(lo_model, load_device=dev)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_lo = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)
    if tuple(seed_lo.shape) != tuple(latents_hi.shape):
        raise RuntimeError(
            "WAN22 GGUF: high/low latent shapes differ after hand-off; cannot maintain a continuous schedule. "
            f"high={tuple(latents_hi.shape)} low_init={tuple(seed_lo.shape)}"
        )

    memory_management.manager.load_model(lo_mm)
    latents_lo = yield from sample_stage_latents_generator(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        metadata_dir=cfg.metadata_dir,
        scheduler_obj=scheduler,
        timestep_start=steps_hi,
        timestep_end=total_steps,
        seed=None,
        state_init=seed_lo,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=flow_multiplier,
        stage_name="low",
        emit_logs=False,
    )

    # Free the LOW stage weights before VAE decode.
    memory_management.manager.unload_model(lo_mm)
    del lo_mm
    del lo_model
    gc.collect()
    if lvl >= 2:
        _try_clear_cache()
        cuda_empty_cache(log, label="after-low")

    frames = decode_latents_to_frames(
        latents=latents_lo,
        model_dir=os.path.dirname(lo_path),
        cfg=cfg,
        logger=log,
        expected_frames=t_out,
    )
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames}
