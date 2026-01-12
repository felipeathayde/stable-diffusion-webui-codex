"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF run entrypoints (txt2vid/img2vid; batch + streaming).
Orchestrates text context, per-stage sampling, and VAE encode/decode while keeping GGUF support anchored in the shared quantization/ops layer.

Symbols (top-level; keep in sync; no ghosts):
- `_try_set_cache_policy` (function): Configure GGUF dequant cache policy + limit when supported.
- `_try_clear_cache` (function): Clear GGUF dequant cache when supported.
- `_resolve_offload_level` (function): Resolve the effective offload profile level from the run config.
- `_require_flow_shift` (function): Validate that a stage has a usable flow_shift value (strict).
- `run_txt2vid` (function): Batch txt2vid runner; orchestrates text context, stage sampling, and VAE decode.
- `stream_txt2vid` (function): Streaming txt2vid generator; yields progress while sampling/decoding.
- `run_img2vid` (function): Batch img2vid runner; encodes init image, runs stages, decodes frames.
- `stream_img2vid` (function): Streaming img2vid generator; yields progress while sampling/decoding.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from .config import (
    RunConfig,
    WAN_FLOW_MULTIPLIER,
    as_torch_dtype,
    resolve_device_name,
)
from .diagnostics import cuda_empty_cache, get_logger, log_cuda_mem
from .sampling import (
    infer_patch_geometry,
    prepare_stage_seed_latents,
    resize_latents_hw,
    sample_stage_latents,
    sample_stage_latents_generator,
)
from .sdpa import set_sdpa_settings
from .stage_loader import load_stage_model_from_gguf, pick_stage_gguf
from .text_context import get_text_context
from .vae_io import decode_latents_to_frames, vae_encode_init


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
        try:
            return max(0, int(cfg.offload_level))
        except Exception:
            return 0
    return 3 if bool(getattr(cfg, "aggressive_offload", True)) else 0


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

    lvl = _resolve_offload_level(cfg)

    hi_model = load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.05)
        except Exception:
            pass

    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_t2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or ("cuda" if lvl <= 1 else "cpu")
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    if bool(getattr(cfg, "te_kernel_required", False)):
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
        offload_after=(lvl >= 1),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = max(1, int(cfg.num_frames))
    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    log.info(
        "[wan22.gguf] HIGH geom: grid=%s kernel=%s cin=%d",
        geom_hi.grid,
        geom_hi.patch_kernel,
        geom_hi.latent_channels,
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
    log.info(
        "[wan22.gguf] HIGH: steps=%s sampler=%s scheduler=%s cfg_scale=%s seed=%s",
        steps_hi,
        sampler_hi,
        sched_hi,
        (getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        cfg.seed,
    )

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
        seed=cfg.seed,
        state_init=None,
        on_progress=(lambda **p: on_progress(stage="high", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="high",
    )

    if lvl >= 2:
        try:
            del hi_model
        except Exception:
            pass
        cuda_empty_cache(log, label="after-high")

    lo_model = load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    log.info(
        "[wan22.gguf] LOW geom: grid=%s kernel=%s cin=%d",
        geom_lo.grid,
        geom_lo.patch_kernel,
        geom_lo.latent_channels,
    )

    seed_latents = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)
    log.info(
        "[wan22.gguf] LOW: steps=%s sampler=%s scheduler=%s cfg_scale=%s",
        steps_lo,
        sampler_lo,
        sched_lo,
        (getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
    )

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
        seed=None,
        state_init=seed_latents,
        on_progress=(lambda **p: on_progress(stage="low", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="low",
    )

    frames = decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
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
    lvl = _resolve_offload_level(cfg)

    hi_model = load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_t2v_{variant}"

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
        offload_after=(lvl >= 1),
        te_device=getattr(cfg, "te_device", None),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = max(1, int(cfg.num_frames))
    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

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
        seed=cfg.seed,
        state_init=None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="high",
        emit_logs=False,
    )

    if lvl >= 2:
        try:
            del hi_model
        except Exception:
            pass
        cuda_empty_cache(log, label="after-high")

    lo_model = load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_latents = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

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
        seed=None,
        state_init=seed_latents,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="low",
        emit_logs=False,
    )

    frames = decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
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
    lvl = _resolve_offload_level(cfg)

    hi_model = load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    if on_progress:
        try:
            on_progress(stage="prepare", step=0, total=1, percent=0.05)
        except Exception:
            pass

    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_i2v_{variant}"

    te_dev_eff = getattr(cfg, "te_device", None) or ("cuda" if lvl <= 1 else "cpu")
    te_impl_val = (getattr(cfg, "te_impl", None) or "hf").strip().lower()
    if bool(getattr(cfg, "te_kernel_required", False)):
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
        offload_after=(lvl >= 1),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = max(1, int(cfg.num_frames))

    lat0 = vae_encode_init(cfg.init_image, device=dev_name, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat0.ndim == 4:
        lat0 = lat0.unsqueeze(2)
    lat0 = lat0.repeat(1, 1, t, 1, 1)
    lat0 = resize_latents_hw(lat0, height=h_lat, width=w_lat)

    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_hi = prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)

    steps_hi = int(getattr(cfg.high, "steps", 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, "sampler", None) if cfg.high else None
    sched_hi = getattr(cfg.high, "scheduler", None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

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
        seed=None,
        state_init=seed_hi,
        on_progress=(lambda **p: on_progress(stage="high", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="high",
    )

    if lvl >= 2:
        try:
            del hi_model
        except Exception:
            pass
        cuda_empty_cache(logger=log, label="after-high")

    lo_model = load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_lo = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, "steps", 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, "sampler", None) if cfg.low else None
    sched_lo = getattr(cfg.low, "scheduler", None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

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
        seed=None,
        state_init=seed_lo,
        on_progress=(lambda **p: on_progress(stage="low", **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="low",
    )

    frames = decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
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
    lvl = _resolve_offload_level(cfg)

    hi_model = load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    variant = "5b" if "5b" in os.path.basename(hi_path).lower() else "14b"
    model_key = f"wan_i2v_{variant}"

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
        offload_after=(lvl >= 1),
        te_device=getattr(cfg, "te_device", None),
        te_impl=getattr(cfg, "te_impl", None),
        te_kernel_required=getattr(cfg, "te_kernel_required", None),
    )
    prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    h_lat = max(8, int(cfg.height) // 8)
    w_lat = max(8, int(cfg.width) // 8)
    t = max(1, int(cfg.num_frames))

    lat0 = vae_encode_init(cfg.init_image, device=dev_name, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat0.ndim == 4:
        lat0 = lat0.unsqueeze(2)
    lat0 = lat0.repeat(1, 1, t, 1, 1)
    lat0 = resize_latents_hw(lat0, height=h_lat, width=w_lat)

    geom_hi = infer_patch_geometry(hi_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_hi = prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)
    flow_shift_hi = getattr(cfg.high, "flow_shift", None) if cfg.high else None
    flow_shift_hi_value = _require_flow_shift("high", flow_shift_hi)

    latents_hi = yield from sample_stage_latents_generator(
        model=hi_model,
        geom=geom_hi,
        steps=int(getattr(cfg.high, "steps", 12) if cfg.high else 12),
        cfg_scale=(getattr(cfg.high, "cfg_scale", None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=(getattr(cfg.high, "sampler", None) if cfg.high else None),
        scheduler_name=(getattr(cfg.high, "scheduler", None) if cfg.high else None),
        seed=None,
        state_init=seed_hi,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_hi_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="high",
        emit_logs=False,
    )

    if lvl >= 2:
        try:
            del hi_model
        except Exception:
            pass
        cuda_empty_cache(logger=log, label="after-high")

    lo_model = load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = infer_patch_geometry(lo_model, t=t, h_lat=h_lat, w_lat=w_lat)
    seed_lo = prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)
    flow_shift_lo = getattr(cfg.low, "flow_shift", None) if cfg.low else None
    flow_shift_lo_value = _require_flow_shift("low", flow_shift_lo)

    latents_lo = yield from sample_stage_latents_generator(
        model=lo_model,
        geom=geom_lo,
        steps=int(getattr(cfg.low, "steps", 12) if cfg.low else 12),
        cfg_scale=(getattr(cfg.low, "cfg_scale", None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=(getattr(cfg.low, "sampler", None) if cfg.low else None),
        scheduler_name=(getattr(cfg.low, "scheduler", None) if cfg.low else None),
        seed=None,
        state_init=seed_lo,
        log_mem_interval=getattr(cfg, "log_mem_interval", None),
        flow_shift=flow_shift_lo_value,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name="low",
        emit_logs=False,
    )

    frames = decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames}
