"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 runtime (GGUF path) for txt2vid/img2vid.
This module loads WAN 2.2 stage GGUF weights into `WanTransformer2DModel` via Codex operations, runs flow sampling,
and supports both “run” (batch) and “stream” (generator) execution with optional per-step progress callbacks.

Symbols (top-level; keep in sync; no ghosts):
- `_wan_log_sigmas_enabled` (function): Enables/disables sigma logging via config/env toggles.
- `_summarize_tensor` (function): Debug helper summarizing tensor-ish objects (shape/dtype/range sample).
- `_resolve_i2v_order` (function): Resolves the image-to-video conditioning order policy.
- `_log_enabled` (function): Returns whether a given log verbosity level is enabled.
- `_li` (function): Logger convenience wrapper (info).
- `_lw` (function): Logger convenience wrapper (warning).
- `_le` (function): Logger convenience wrapper (error).
- `_ld` (function): Logger convenience wrapper (debug).
- `_dbg` (function): Lightweight debug marker/logger helper for tracing execution points.
- `_io` (function): Decorator factory for entry/exit tracing when `_DEBUG_IO` is enabled.
- `_latent_dimensions` (function): Computes latent tensor dimensions from a `PatchGeometry` description.
- `_ensure_latent_shape` (function): Validates/reshapes latent tensors to the expected `PatchGeometry` layout.
- `_sample_stage_latents` (function): Core latent sampling for a single WAN stage (high/low) using the selected scheduler/sampler.
- `_sample_stage_latents_generator` (function): Generator version of stage sampling for streaming progress (yields intermediate states).
- `_decode_latents_to_frames` (function): Decodes sampled latents into video frames (uses VAE and postprocessing).
- `_prepare_stage_seed_latents` (function): Prepares seeded stage latents (for determinism across runs/stages).
- `_assemble_i2v_input` (function): Builds img2vid latent inputs to match expected input channels/order.
- `_get_text_context` (function): Builds text conditioning/context (prompt + negative prompt) for the WAN transformer.
- `_load_vae` (function): Loads the WAN VAE component (and moves it to device/dtype as needed).
- `_log_latent_norm` (function): Logs latent normalization statistics for debugging.
- `_vae_encode_init` (function): Encodes an init image into latents for img2vid.
- `_vae_decode_video` (function): Decodes video latents to frames and optionally offloads/cleans up after decode.
- `PatchGeometry` (class): Patch/tile geometry configuration used to infer latent/video shapes.
- `_try_set_cache_policy` (function): Best-effort GGUF cache policy configuration (limit + eviction strategy).
- `_try_clear_cache` (function): Best-effort cache clearing (used during long streaming runs or debug).
- `_normalize_win_path` (function): Normalizes Windows paths to reduce path-encoding edge cases.
- `_pick_stage_gguf` (function): Selects the GGUF file for a given stage from a directory (if not explicitly provided).
- `_load_stage_model_from_gguf` (function): Loads a stage model from GGUF into a runtime transformer (wraps ops + remapping).
- `StageConfig` (class): Stage-level configuration (steps/cfg/seed/sampler/scheduler + stage model selection).
- `RunConfig` (class): Full run configuration (geometry, prompts, devices/dtypes, assets, and both stages).
- `_as_dtype` (function): Parses dtype strings into torch dtypes (with validation).
- `_get_logger` (function): Normalizes a logger argument into a `logging.Logger` (or `None`).
- `_cuda_empty_cache` (function): Best-effort CUDA cache emptying with optional logging.
- `_infer_patch_geometry` (function): Infers patch geometry defaults from config and requested output size.
- `_make_scheduler` (function): Constructs the scheduler instance for a run (based on sampler/scheduler selection).
- `_cfg_merge` (function): Classifier-free guidance merge helper (uncond/cond + scale).
- `_log_cuda_mem` (function): Logs CUDA memory stats for debugging long video runs.
- `_log_t_mapping` (function): Logs timestep mapping/debug info for schedulers.
- `_time_snr_shift` (function): Time/SNR shift helper used in scheduler-time transformations.
- `_resolve_device_name` (function): Normalizes device names (`cuda`/`cpu`/etc) into runtime-compatible values.
- `run_txt2vid` (function): Batch txt2vid runner; orchestrates text context, stage sampling, and VAE decode (uses multiple helpers).
- `stream_txt2vid` (function): Streaming txt2vid generator; yields progress while sampling/decoding.
- `run_img2vid` (function): Batch img2vid runner; encodes init image, runs stages, decodes frames (uses multiple helpers).
- `stream_img2vid` (function): Streaming img2vid generator; yields progress while sampling/decoding.
- `_resize_latents_hw` (function): Resizes latents to a target H/W (used for compatibility across stages/sizes).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional, Tuple

import torch
from diffusers import AutoencoderKLWan  # type: ignore

from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.utils import _load_gguf_state_dict

from .model import load_wan_transformer_from_state_dict, remap_wan22_gguf_state_dict
from .sdpa import set_sdpa_settings
# Local latent normalization. Try relative first, then absolute for robustness.
try:
    from .wan_latent_norms import resolve_norm
except Exception as _ex_rel:
    try:
        from apps.backend.runtime.wan22.wan_latent_norms import resolve_norm  # type: ignore
    except Exception as _ex_abs:  # pragma: no cover
        raise ImportError(
            "WAN latent norms module not found. Ensure apps/backend/runtime/wan22/wan_latent_norms.py exists "
            "and your working copy is up-to-date (git pull)."
        ) from _ex_abs
# WAN DiT helpers are inlined here to keep the one-file-per-model convention,
# matching flux/chroma. No dependence on legacy wan_gguf_core/*.

# Debug helpers (lightweight)
_DEBUG_IO = False  # set True to enable entry/exit logs

WAN_FLOW_SHIFT_DEFAULT = 8.0
WAN_FLOW_MULTIPLIER = 1000.0

def _wan_log_sigmas_enabled() -> bool:
    """Return True when sigma/timestep parity logs should be emitted."""
    for key in ("WAN_LOG_SIGMAS", "CODEX_LOG_SIGMAS"):
        v = os.getenv(key)
        if v is None:
            continue
        if v.strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False

def _summarize_tensor(t: object, *, window: int = 6) -> str:
    try:
        if not isinstance(t, torch.Tensor):
            return "<not-a-tensor>"
        values = [float(v) for v in t.detach().cpu().reshape(-1).tolist()]
    except Exception:
        return "<unavailable>"
    if len(values) <= window * 2:
        return ",".join(f"{v:.6g}" for v in values)
    head = ",".join(f"{v:.6g}" for v in values[:window])
    tail = ",".join(f"{v:.6g}" for v in values[-window:])
    return f"{head},...,{tail}"

def _resolve_i2v_order() -> str:
    """Return channel order for I2V concatenation.
    - 'lat_first': latents(16) then cond extras (mask4+img16) → matches the reference concat order (xc + c_concat).
    - 'lat_last' : cond extras first then latents(16).
    Defaults to 'lat_first'. Controlled by env WAN_I2V_ORDER.
    """
    try:
        v = str(os.getenv('WAN_I2V_ORDER', 'lat_first')).strip().lower()
        return 'lat_last' if v in ('lat_last', 'last', 'cond_first') else 'lat_first'
    except Exception:
        return 'lat_first'

def _log_enabled(level: str) -> bool:
    key = {
        'info': 'WAN_LOG_INFO',
        'warn': 'WAN_LOG_WARN',
        'error': 'WAN_LOG_ERROR',
        'debug': 'WAN_LOG_DEBUG',
    }.get(level, 'WAN_LOG_INFO')
    val = os.getenv(key)
    if val is None:
        # Default: info/warn/error on, debug off
        return False if level == 'debug' else True
    s = val.strip().lower()
    return s in ('1', 'true', 'yes', 'on')

def _li(logger, msg, *args):
    if not _log_enabled('info'):
        return
    try:
        _get_logger(logger).info(msg, *args)
    except Exception:
        pass

def _lw(logger, msg, *args):
    if not _log_enabled('warn'):
        return
    try:
        _get_logger(logger).warning(msg, *args)
    except Exception:
        pass

def _le(logger, msg, *args):
    if not _log_enabled('error'):
        return
    try:
        _get_logger(logger).error(msg, *args)
    except Exception:
        pass

def _ld(logger, msg, *args):
    if not _log_enabled('debug'):
        return
    try:
        _get_logger(logger).debug(msg, *args)
    except Exception:
        pass

def _dbg(logger, name: str, where: str) -> None:
    if not _DEBUG_IO:
        return
    try:
        _ld(logger, "[wan22.gguf] DEBUG: %s de função %s", where, name)
    except Exception:
        pass
def _io(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        try:
            _dbg(kwargs.get('logger', None), fn.__name__, 'antes')
        except Exception:
            _dbg(None, fn.__name__, 'antes')
        out = fn(*args, **kwargs)
        try:
            _dbg(kwargs.get('logger', None), fn.__name__, 'depois')
        except Exception:
            _dbg(None, fn.__name__, 'depois')
        return out
    return _wrap

def _latent_dimensions(geom: PatchGeometry) -> Tuple[int, int, int]:
    kT, kH, kW = geom.patch_kernel
    return (
        int(geom.grid[0] * kT),
        int(geom.grid[1] * kH),
        int(geom.grid[2] * kW),
    )


def _ensure_latent_shape(x: torch.Tensor, geom: PatchGeometry) -> torch.Tensor:
    t_target, h_target, w_target = _latent_dimensions(geom)
    if x.shape[2] == t_target and x.shape[3] == h_target and x.shape[4] == w_target:
        return x
    return _resize_latents_hw(x, H=h_target, W=w_target)


@_io
def _sample_stage_latents(
    *,
    model,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    on_progress: Optional[Any] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float = WAN_FLOW_SHIFT_DEFAULT,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = 'stage',
) -> torch.Tensor:
    gen = _sample_stage_latents_generator(
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
            payload = {k: event[k] for k in ('step', 'total', 'percent', 'eta_seconds', 'step_seconds') if k in event}
            try:
                on_progress(**payload)
            except Exception:
                pass


@_io
def _sample_stage_latents_generator(
    *,
    model,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float = WAN_FLOW_SHIFT_DEFAULT,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = 'stage',
    emit_logs: bool = True,
):
    log = _get_logger(logger)
    t_lat, h_lat, w_lat = _latent_dimensions(geom)
    steps = max(int(steps), 1)

    batch = int(state_init.shape[0]) if state_init is not None else 1
    shape = (batch, int(geom.latent_channels), t_lat, h_lat, w_lat)

    if state_init is not None:
        state = _ensure_latent_shape(state_init.to(device=device, dtype=dtype), geom).clone()
    else:
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            state = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            state = torch.randn(shape, device=device, dtype=dtype)
        sigma_init = _time_snr_shift(flow_shift, 1.0) * flow_multiplier
        state = state * float(sigma_init)

    scheduler = _make_scheduler(steps, sampler=sampler_name, scheduler=scheduler_name)
    timesteps = scheduler.timesteps
    total = len(timesteps)

    flow_progress = torch.linspace(1.0, 0.0, total, device=device, dtype=torch.float32) if total > 1 else torch.ones(1, device=device, dtype=torch.float32)
    parity_idxs = {0, max(0, total // 2 - 1), max(0, total - 1)}

    if _wan_log_sigmas_enabled():
        try:
            sigmas = getattr(scheduler, 'sigmas', None)
            if isinstance(sigmas, torch.Tensor):
                log.info(
                    "[wan22.gguf] %s schedule: scheduler=%s timesteps=%d sigmas=%s",
                    stage_name,
                    scheduler.__class__.__name__,
                    int(total),
                    _summarize_tensor(sigmas),
                )
            _log_t_mapping(scheduler, timesteps, label=stage_name, logger=logger)
        except Exception:
            pass

    yield {
        "type": "progress",
        "stage": stage_name,
        "step": 0,
        "total": total,
        "percent": 0.0,
    }

    import time
    t0 = time.perf_counter()
    last = t0

    for idx, timestep in enumerate(timesteps):
        percent = float(flow_progress[idx].item()) if total > 1 else 1.0
        sigma_value = _time_snr_shift(flow_shift, percent)
        di_timestep = float(sigma_value * flow_multiplier)

        if _wan_log_sigmas_enabled() and idx in parity_idxs:
            try:
                sched_sigmas = getattr(scheduler, 'sigmas', None)
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
            except Exception:
                pass

        if cfg_scale is None:
            eps = model(state, di_timestep, prompt_embeds)
        else:
            x_in = torch.cat([state, state], dim=0)
            ctx_in = torch.cat([prompt_embeds, negative_embeds], dim=0)
            t_in = torch.full((x_in.shape[0],), float(di_timestep), device=device, dtype=torch.float32)
            v_pred = model(x_in, t_in, ctx_in)
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            eps = _cfg_merge(v_uncond, v_cond, cfg_scale)

        if eps.shape != state.shape:
            raise RuntimeError(
                f"WAN22 GGUF: model output shape {tuple(eps.shape)} does not match latent state {tuple(state.shape)} "
                f"(patch_size={geom.patch_kernel} grid={geom.grid})"
            )

        out = scheduler.step(model_output=eps, timestep=timestep, sample=state)
        state = out.prev_sample

        pct = float(idx + 1) / float(max(1, total))
        if log_mem_interval is not None:
            try:
                n = int(log_mem_interval or 0)
                if n > 0 and ((idx + 1) % n) == 0:
                    _log_cuda_mem(logger, label=f'{stage_name}-step-{idx + 1}')
            except Exception:
                pass

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

@_io
def _decode_latents_to_frames(
    *,
    latents: torch.Tensor,
    model_dir: str,
    cfg: RunConfig,
    logger=None,
    debug_preview: bool = False,
) -> list[object]:
    x = latents
    try:
        _li(logger, "[wan22.gguf] decode latents: shape=%s", tuple(x.shape))
    except Exception:
        pass
    if debug_preview:
        try:
            v = os.getenv('WAN_I2V_DEBUG_CLAMP', '').strip()
            if v:
                lim = float(v)
                if lim > 0:
                    x = torch.clamp(x, min=-lim, max=lim)
        except Exception:
            pass
    C = int(x.shape[1])
    if C != 16:
        if C >= 16:
            if _resolve_i2v_order() == 'lat_first':
                x = x[:, :16, ...]
            else:
                x = x[:, -16:, ...]
            try:
                _li(logger, "[wan22.gguf] decode latents: sliced to 16 channels from C=%d", C)
            except Exception:
                pass
        else:
            raise RuntimeError(f"WAN22 GGUF: expected ≥16 latent channels for decode, got {C}")
    return _vae_decode_video(x, model_dir=model_dir, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=logger)


@_io
def _prepare_stage_seed_latents(
    latents: torch.Tensor,
    target_geom: PatchGeometry,
    *,
    logger=None,
) -> torch.Tensor:
    c_src = int(latents.shape[1])
    c_dst = int(target_geom.latent_channels)
    if c_src == c_dst:
        return _ensure_latent_shape(latents, target_geom)
    if c_src >= 16 and c_dst == 16:
        if _resolve_i2v_order() == 'lat_first':
            sliced = latents[:, :16, ...]
        else:
            sliced = latents[:, -16:, ...]
        return _ensure_latent_shape(sliced, target_geom)
    if c_src == 16 and c_dst == 36:
        assembled = _assemble_i2v_input(latents, c_dst, logger=_get_logger(logger))
        return _ensure_latent_shape(assembled, target_geom)
    raise RuntimeError(
        f"Cannot adapt latent channels from {c_src} to {c_dst}; unsupported hand-off configuration"
    )

def _assemble_i2v_input(latents: torch.Tensor, expected_cin: int, logger: logging.Logger | None = None) -> torch.Tensor:
    """Assemble I2V input volume to match expected Cin for patch embedding.

    The WAN I2V pipeline design uses 36 channels composed as:
    4-channel temporal mask + 16-channel image features + 16 latent channels.
    We follow this layout when expected_cin - C == 20.

    - latents: [B, C, T, H, W] (typically C=16 from VAE 2.1)
    - expected_cin: target channels for patch_embedding.weight.shape[1]
    Returns: [B, expected_cin, T, H, W]
    """
    if latents.ndim != 5:
        raise RuntimeError(f"_assemble_i2v_input: expected 5D latents [B,C,T,H,W], got {tuple(latents.shape)}")
    B, C, T, H, W = latents.shape
    extra = expected_cin - C
    if extra <= 0:
        return latents
    # I2V composition: 36 = 16 (lat) + 4 (mask) + 16 (image)
    # Concat order depends on WAN_I2V_ORDER (default: lat_first to match xc + c_concat)
    if extra == 20:
        # Build mask (zeros if not provided): [B,4,T,H,W]
        mask = latents.new_zeros((B, 4, T, H, W))
        # Image features: reuse VAE latents as 16-ch features by default
        image_feats = latents[:, : min(16, C)]
        if image_feats.shape[1] < 16:
            # Pad features to 16 if VAE channels < 16 (unlikely for WAN 2.1), using zeros
            pad = 16 - image_feats.shape[1]
            image_feats = torch.cat([image_feats, latents.new_zeros((B, pad, T, H, W))], dim=1)
        order = _resolve_i2v_order()
        if order == 'lat_first':
            assembled = torch.cat([latents, mask, image_feats], dim=1)
            layout = f"[lat{C} + mask4 + img16]"
        else:
            assembled = torch.cat([mask, image_feats, latents], dim=1)
            layout = f"[mask4 + img16 + lat{C}]"
        if assembled.shape[1] != expected_cin:
            raise RuntimeError(
                f"I2V assembly produced {assembled.shape[1]} channels, expected {expected_cin} (mask4 + img16 + lat{C})."
            )
        if logger:
            logger.info("[wan22.gguf] i2v assemble: order=%s %s → C=%d", order, layout, assembled.shape[1])
        return assembled
    # Unsupported pattern — surface a clear error
    raise RuntimeError(
        f"WAN22 GGUF (img2vid): expected C_in={expected_cin} but VAE produced C={C}. "
        f"I2V assembly requires extra={extra} channels (mask+image). Unsupported combo."
    )


@_io
def _get_text_context(
    model_dir: str,
    prompt: str,
    negative: Optional[str],
    *,
    device: str,
    dtype: str,
    text_encoder_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    vae_dir: Optional[str] = None,
    model_key: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    offload_after: bool = True,
    te_device: Optional[str] = None,
    te_impl: Optional[str] = None,
    te_kernel_required: Optional[bool] = None,
):
    """GGUF path: use Transformers tokenizer + encoder only; do NOT fall back to Diffusers.

    - Searches explicit extras first (tokenizer_dir/text_encoder_dir), then common subfolders under model_dir.
    - Never downloads; never calls Diffusers. If not found, raises an explicit, actionable error.
    """
    import torch
    from transformers import AutoTokenizer, AutoConfig
    try:
        from transformers import UMT5EncoderModel as _Enc
    except Exception:
        from transformers import T5EncoderModel as _Enc

    # Resolve tokenizer dir: prefer explicit tokenizer_dir; else infer from metadata_dir/tokenizer*
    tk_dir = tokenizer_dir
    if (not tk_dir) and metadata_dir:
        cand = os.path.join(metadata_dir, 'tokenizer')
        cand2 = os.path.join(metadata_dir, 'tokenizer_2')
        if os.path.isdir(cand):
            tk_dir = cand
        elif os.path.isdir(cand2):
            tk_dir = cand2
    te_dir = text_encoder_dir
    te_file: Optional[str] = None
    if te_dir and os.path.isfile(te_dir) and te_dir.lower().endswith('.safetensors'):
        te_file = te_dir
        te_dir = os.path.dirname(te_dir)
    if tk_dir and os.path.isfile(tk_dir):
        tk_dir = os.path.dirname(tk_dir)

    # Strict: require tokenizer source
    if not tk_dir or not os.path.isdir(tk_dir):
        raise RuntimeError("WAN22 GGUF: tokenizer metadata missing or invalid; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.")

    # Load tokenizer from the single provided directory
    try:
        tok = AutoTokenizer.from_pretrained(tk_dir, use_fast=True, local_files_only=True)
    except Exception as ex:
        raise RuntimeError(f"WAN22 GGUF: failed to load tokenizer from '{tk_dir}': {ex}") from ex
    try:
        _li(None, "[wan22.gguf] tokenizer loaded: dir=%s vocab=%d model_max_len=%s",
            tk_dir, len(getattr(tok, 'vocab', {}) or {}), str(getattr(tok, 'model_max_length', None)))
    except Exception:
        pass

    # Effective TE preferences (extras > env > defaults)
    try:
        te_impl_eff = (te_impl or os.getenv('WAN_TE_IMPL', '') or 'hf').strip().lower()
    except Exception:
        te_impl_eff = (te_impl or 'hf') if te_impl else 'hf'
    # No fallbacks allowed: if impl=cuda_fp8, kernel is REQUIRED
    te_req_eff = (te_impl_eff == 'cuda_fp8')
    te_dev_eff = (te_device or os.getenv('CODEX_TE_DEVICE') or device or 'cpu').strip().lower()
    if te_dev_eff == 'gpu':
        te_dev_eff = 'cuda'
    if te_dev_eff == 'cpu' and dtype != 'fp32':
        dtype = 'fp32'

    # One-time log of effective selection
    try:
        _li(None, "[wan22.gguf] text-encoder: impl=%s required=%s device=%s", te_impl_eff, str(bool(te_req_eff)).lower(), te_dev_eff)
    except Exception:
        pass

    # CUDA TE kernel (FP8). Required if selected; do not fallback.
    if te_impl_eff == 'cuda_fp8':
        try:
            from apps.backend.runtime.nn import wan_te_cuda as _tecuda
        except Exception as ex:
            raise RuntimeError(f"WAN22 TE CUDA kernel required but module not importable: {ex}") from ex
        if not _tecuda.available():
            # Try to surface the last loader error for clarity
            try:
                last = _tecuda.last_error()
            except Exception:
                last = None
            if last:
                raise RuntimeError(f"WAN22 TE CUDA kernel required but not available ({last}).")
            raise RuntimeError("WAN22 TE CUDA kernel required but not available. Build wan_te_cuda.")
        # Tokenize normally, then run experimental FP8 encoder path
        if te_impl_eff == 'cuda_fp8':
            # 1) Tokenizer
            try:
                tok = AutoTokenizer.from_pretrained(tk_dir, use_fast=True, local_files_only=True)
            except Exception as ex:
                raise RuntimeError(f"WAN22 GGUF: failed to load tokenizer from '{tk_dir}': {ex}") from ex
            inputs = tok([prompt or "", negative or ""], padding='max_length', truncation=True, max_length=225, return_tensors='pt')
            try:
                ids = inputs.get('input_ids')
                if ids is not None:
                    _li(None, "[wan22.gguf] tokenized(fp8): batch=%d seqlen=%d", int(ids.shape[0]), int(ids.shape[1]))
            except Exception:
                pass
            input_ids = inputs['input_ids']  # [2,L]
            attn_mask = inputs.get('attention_mask', None)
            # 2) Encoder FP8 (CUDA): run per prompt/negative separately to keep memory minimal
            from transformers import AutoConfig as _AutoCfg
            enc_dir = os.path.join(metadata_dir, 'text_encoder')
            cfg_hf = _AutoCfg.from_pretrained(enc_dir, local_files_only=True)
            num_heads = int(getattr(cfg_hf, 'num_heads', getattr(cfg_hf, 'num_attention_heads', 32)))
            d_kv = int(getattr(cfg_hf, 'd_kv', getattr(cfg_hf, 'hidden_size', 4096) // num_heads))
            from apps.backend.runtime.wan22.wan_te_encoder import encode_fp8 as _encode_fp8
            dev = torch.device('cuda' if (te_dev_eff or device) == 'cuda' and torch.cuda.is_available() else 'cpu')
            if dev.type != 'cuda':
                raise RuntimeError("WAN22 TE CUDA path requested but selected device is not CUDA")
            if te_impl_eff == 'cuda_fp8':
                dt = _as_dtype(dtype)
                def _run_one(ids: torch.Tensor) -> torch.Tensor:
                    ids = ids.to(torch.long)
                    return _encode_fp8(
                        te_weights_path=te_file or '',
                        input_ids=ids.to(dev),
                        attention_mask=(attn_mask[0:1].to(dev) if attn_mask is not None else None),
                        device=dev,
                        dtype=dt,
                        num_heads=num_heads,
                        d_kv=d_kv,
                        log_metrics=True,
                    )
                p = _run_one(input_ids[0:1])
                n = _run_one(input_ids[1:2])
                try:
                    _li(None, "[wan22.gguf] TE(fp8) outputs: prompt=%s negative=%s dtype=%s device=%s",
                        tuple(p.shape), tuple(n.shape), str(p.dtype), str(p.device))
                except Exception:
                    pass
                # Offload aggressively after TE
                if offload_after:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return p, n

    # Strict: require text encoder weights (file) OR a directory with config; when a file is provided,
    # the config is resolved from metadata_dir/text_encoder (vendored repo), never from the weights folder.
    if te_file is not None:
        if not metadata_dir or not os.path.isdir(metadata_dir):
            raise RuntimeError("WAN22 GGUF: 'wan_metadata_dir' is required when providing 'wan_text_encoder_path'.")
        enc_dir = os.path.join(metadata_dir, 'text_encoder')
        if not os.path.isdir(enc_dir):
            raise RuntimeError(
                f"WAN22 GGUF: expected text encoder config under metadata repo: '{enc_dir}'"
            )
        try:
            cfg = AutoConfig.from_pretrained(enc_dir, local_files_only=True)
        except Exception as ex:
            raise RuntimeError(
                f"WAN22 GGUF: failed to read text encoder config from '{enc_dir}': {ex}"
            ) from ex
        enc = _Enc(cfg)
        from safetensors.torch import load_file as _load_st
        try:
            sd = _load_st(te_file)
            enc.load_state_dict(sd, strict=False)
        except Exception as ex:
            raise RuntimeError(f"WAN22 GGUF: failed to load text encoder weights '{te_file}': {ex}") from ex
    else:
        # Strict mode: require a TE weights file; directory-based TE loading is not supported in WAN22 GGUF.
        raise RuntimeError(
            "WAN22 GGUF: 'wan_text_encoder_path' (.safetensors file) is required. Directory-based text encoders are not supported."
        )

    # Device/dtype for TE: set both in a single call to avoid transient FP32 allocation on GPU
    use_dev_name = (te_dev_eff or device or 'cpu').lower().strip()
    dev = torch.device('cuda' if use_dev_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    try:
        enc = enc.to(device=dev, dtype=_as_dtype(dtype))
    except Exception:
        # If dtype cast fails (rare), at least ensure device move
        enc = enc.to(device=dev)

    def _do(txt: str):
        inputs = tok([txt], padding='max_length', truncation=True, max_length=225, return_tensors='pt')
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = enc(**inputs).last_hidden_state
            return out.to(_as_dtype(dtype))

    p = _do(prompt or '')
    n = _do(negative or '') if negative is not None else _do('')
    try:
        # Sanity: hidden size vs config
        cfg_hidden = int(getattr(getattr(enc, 'config', None), 'hidden_size', p.shape[-1]))
        if int(p.shape[-1]) != cfg_hidden:
            raise RuntimeError(f"WAN22 GGUF: TE hidden_size mismatch: output={int(p.shape[-1])} config={cfg_hidden}")
        _li(None, "[wan22.gguf] TE outputs: prompt=%s negative=%s dtype=%s device=%s",
            tuple(p.shape), tuple(n.shape), str(p.dtype), str(p.device))
    except Exception as _te_log_exc:
        _lw(None, "[wan22.gguf] TE log note: %s", _te_log_exc)
    # Aggressive offload: drop TE from VRAM immediately after use
    if offload_after:
        try:
            enc.to('cpu')
        except Exception:
            pass
        del enc
        _cuda_empty_cache(logger=None, label='after-te')
    return p, n


@_io
def _load_vae(vae_path: Optional[str], *, torch_dtype):
    import os
    if not vae_path:
        raise RuntimeError('wan_vae_dir is required when running WAN GGUF (VAE path missing)')
    path = os.path.expanduser(str(vae_path))
    if os.path.isdir(path):
        return AutoencoderKLWan.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if os.path.isfile(path):
        loader = getattr(AutoencoderKLWan, 'from_single_file', None)
        if loader is None:
            raise RuntimeError(f'AutoencoderKLWan.from_single_file not available; provide a directory instead of file: {path}')
        return loader(path, torch_dtype=torch_dtype)
    raise RuntimeError(f'VAE path not found: {path}')


@_io
def _log_latent_norm(logger, *, norm_name: str, channels: int) -> None:
    if logger:
        try:
            logger.info('[wan22.gguf] VAE latent norm=%s channels=%d', norm_name, channels)
        except Exception:
            pass


@_io
def _vae_encode_init(init_image: Any, *, device: str, dtype: str, vae_dir: str | None = None, logger=None, offload_after: bool = True):
    import torch
    torch_dtype = _as_dtype(dtype)
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    # Choose latent normalization (env WAN_VAE_NORM: wan21|none)
    import os as _os
    norm = resolve_norm(_os.getenv('WAN_VAE_NORM', 'wan21'), channels=16)
    _log_latent_norm(logger, norm_name=norm.name, channels=norm.channels)
    target = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    from apps.backend.runtime.memory import memory_management as _mm
    _old = getattr(_mm, 'VAE_ALWAYS_TILED', False)
    try:
        _mm.VAE_ALWAYS_TILED = True
        vae = vae.to(device=target, dtype=torch_dtype)
    finally:
        _mm.VAE_ALWAYS_TILED = _old
    if not hasattr(init_image, 'to'):
        from PIL import Image
        import numpy as np
        if isinstance(init_image, Image.Image):
            img = init_image.convert('RGB')
            arr = np.array(img).astype('float32') / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0
        else:
            arr = np.asarray(init_image).astype('float32')
            if arr.ndim == 3 and arr.shape[2] in (1, 3):
                arr = arr / 255.0 if arr.max() > 1.0 else arr
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                raise RuntimeError('unsupported init_image array shape')
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0
    # VAE expects video tensor [B,C,T,H,W]; expand a single frame to T=1
    if hasattr(init_image, 'ndim'):
        if init_image.ndim == 4:
            init_image = init_image.unsqueeze(2)
        elif init_image.ndim != 5:
            raise RuntimeError('init_image must be 4D (B,C,H,W) or 5D (B,C,T,H,W) after preprocessing')
    with torch.no_grad():
        latents = vae.encode(init_image).latent_dist.sample()
        # Apply latent normalization for diffusion
        latents = norm.process_in(latents)
    if offload_after:
        try:
            vae.to('cpu')
        except Exception:
            pass
        del vae
        _cuda_empty_cache(logger, label='after-vae-encode')
    return latents


@_io
def _vae_decode_video(video_latents: Any, *, model_dir: str, device: str, dtype: str, vae_dir: str | None = None, logger=None, offload_after: bool = True):
    import torch
    from PIL import Image
    torch_dtype = _as_dtype(dtype)
    dev = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    import os as _os
    norm = resolve_norm(_os.getenv('WAN_VAE_NORM', 'wan21'), channels=int(getattr(vae, 'latent_channels', 16) or 16))
    _log_latent_norm(logger, norm_name=norm.name, channels=norm.channels)
    from apps.backend.runtime.memory import memory_management as _mm
    _old = getattr(_mm, 'VAE_ALWAYS_TILED', False)
    try:
        _mm.VAE_ALWAYS_TILED = True
        vae = vae.to(device=dev, dtype=torch_dtype)
    finally:
        _mm.VAE_ALWAYS_TILED = _old
    # Expect [B,C,T,H,W]; expand to T=1 if a single frame [B,C,H,W] is provided
    try:
        vt = video_latents
        if torch.is_tensor(vt):
            n_bad = int((~torch.isfinite(vt)).sum().item())
            if n_bad > 0:
                raise RuntimeError(f"WAN22 GGUF: non-finite latents before VAE decode (count={n_bad}).")
    except Exception:
        pass
    if hasattr(video_latents, 'ndim'):
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(2)
        elif video_latents.ndim != 5:
            raise RuntimeError(f"VAE decode expects 4D or 5D latents; got shape={tuple(getattr(video_latents,'shape',()))}")
    B, C, T, H, W = video_latents.shape
    # Optional stats to debug latent distribution before decode
    try:
        if str(os.getenv('WAN_I2V_LAT_STATS','0')).strip().lower() in ('1','true','yes','on'):
            vt = video_latents
            if torch.is_tensor(vt):
                vt_cpu = vt.detach().to(device='cpu', dtype=torch.float32)
                finite = torch.isfinite(vt_cpu)
                n_total = int(vt_cpu.numel())
                n_bad = int((~finite).sum().item())
                if n_bad < n_total:
                    vals = vt_cpu[finite]
                    mn = float(vals.min().item())
                    mx = float(vals.max().item())
                    mean = float(vals.mean().item())
                    std = float(vals.std(unbiased=False).item())
                    _li(logger, "[wan22.gguf] latents stats: B=%d C=%d T=%d H=%d W=%d min=%.4f max=%.4f mean=%.4f std=%.4f bad=%d", B, C, T, H, W, mn, mx, mean, std, n_bad)
                else:
                    _li(logger, "[wan22.gguf] latents stats: B=%d C=%d T=%d H=%d W=%d (all non-finite: %d)", B, C, T, H, W, n_bad)
    except Exception:
        pass
    # Hard guard: WAN VAE expects 16-channel latents. If not, the caller likely passed
    # a concatenated I2V tensor (e.g., mask4+img16+lat16 → 36ch). Callers must slice
    # to the 16 latent channels before decode (use _decode_latents_to_frames).
    if int(C) != 16:
        raise RuntimeError(
            f"WAN22 VAE decode expects 16 channels but received C={C}. "
            "If using I2V checkpoints that embed mask+image+latents, unembed tokens "
            "and slice the last 16 channels before decode (handled by _decode_latents_to_frames)."
        )
    frames: list[Image.Image] = []
    with torch.no_grad():
        for t in range(T):
            # Keep 5D layout for WAN VAE decode: [B,C,1,H,W]
            lat = video_latents[:, :, t:t+1]
            # Reverse latent normalization before decoding to RGB
            lat = norm.process_out(lat)
            img = vae.decode(lat).sample
            if t == 0 and logger:
                try:
                    logger.info('[wan22.gguf] VAE decode output shape=%s', tuple(getattr(img, 'shape', ())))
                except Exception:
                    pass
            x = img[0].detach()
            # Some WAN VAEs return [C,1,H,W]; squeeze singleton time if present
            if x.ndim == 4:
                # Common case [C,1,H,W]
                if x.shape[1] == 1:
                    x = x[:, 0, ...]
                # Fallback: squeeze any singleton dims
                if x.ndim == 4:
                    x = x.squeeze()
            if x.ndim != 3:
                raise RuntimeError(f'VAE decode produced unexpected tensor rank: shape={tuple(x.shape)}; expected [C,H,W]')
            # No sanitization: non-finite outputs are hard errors
            if not torch.isfinite(x).all():
                try:
                    n_bad = int((~torch.isfinite(x)).sum().item())
                except Exception:
                    n_bad = -1
                raise RuntimeError(f"WAN22 GGUF: VAE decode produced non-finite outputs (count={n_bad}).")
            x = x.clamp(0, 1)
            arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frames.append(Image.fromarray(arr))
    if offload_after:
        try:
            vae.to('cpu')
        except Exception:
            pass
        del vae
        _cuda_empty_cache(logger, label='after-vae-decode')
    return frames


# ------------------------------ geometry

@dataclass(frozen=True)
class PatchGeometry:
    grid: Tuple[int, int, int]
    token_count: int
    token_dim: int
    latent_channels: int
    patch_kernel: Tuple[int, int, int]


def _try_set_cache_policy(policy: Optional[str], limit_mb: Optional[int]) -> None:
    # Fallback to env if not explicitly provided
    if policy is None:
        policy = os.getenv('CODEX_GGUF_CACHE_POLICY', 'none')
    if (limit_mb is None) or (int(limit_mb or 0) <= 0):
        try:
            limit_mb = int(os.getenv('CODEX_GGUF_CACHE_LIMIT_MB', '0') or 0)
        except Exception:
            limit_mb = 0
    pol = (policy or 'none').strip().lower()
    lim = int(limit_mb or 0)
    if pol in ('none', '', 'off') or lim <= 0:
        return
    try:
        from apps.backend.runtime.ops.operations_gguf import set_cache_policy as _scp
    except Exception as ex:  # pragma: no cover
        raise RuntimeError("GGUF dequant cache requested but not available in this build (set_cache_policy missing). Update backend.") from ex
    _scp(pol, lim)


def _try_clear_cache() -> None:
    try:
        from apps.backend.runtime.ops.operations_gguf import clear_cache as _cc
        _cc()
    except Exception:
        pass


# ------------------------------ helpers for stage files

def _normalize_win_path(p: str) -> str:
    if os.name == 'nt':
        return p
    if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip('\\/')
        return f"/mnt/{drive}/" + rest.replace('\\\\', '/').replace('\\', '/')
    return p


def _pick_stage_gguf(dir_path: Optional[str], stage: str) -> Optional[str]:
    if not dir_path:
        return None

    raw = _normalize_win_path(dir_path)
    abspath = raw if os.path.isabs(raw) else os.path.abspath(raw)
    if os.path.isfile(abspath) and abspath.lower().endswith(".gguf"):
        return abspath
    if os.path.isdir(abspath):
        raise RuntimeError(
            f"WAN22 GGUF stage '{stage}' requires an explicit .gguf file path (sha-selected); got directory: {abspath}"
        )
    return None


def _load_stage_model_from_gguf(
    gguf_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
):
    state = _load_gguf_state_dict(gguf_path)
    state = remap_wan22_gguf_state_dict(state)
    with using_codex_operations(device=device, dtype=dtype, bnb_dtype="gguf"):
        model = load_wan_transformer_from_state_dict(state, config=None)
    model.eval()
    try:
        _li(logger, "[wan22.gguf] loaded stage model: %s", os.path.basename(gguf_path))
    except Exception:
        pass
    return model


# ------------------------------ task entrypoints (skeletons)

@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    flow_shift: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    width: int
    height: int
    fps: int
    num_frames: int
    guidance_scale: Optional[float]
    dtype: str
    device: str
    seed: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[object] = None
    vae_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    metadata_dir: Optional[str] = None
    high: Optional[StageConfig] = None
    low: Optional[StageConfig] = None
    # Memory/attention controls (optional)
    sdpa_policy: Optional[str] = None            # 'mem_efficient' | 'flash' | 'math'
    attn_chunk_size: Optional[int] = None        # split attention along sequence if set (>0)
    gguf_cache_policy: Optional[str] = None      # 'none' | 'cpu_lru'
    gguf_cache_limit_mb: Optional[int] = None    # MB limit for cpu_lru cache
    log_mem_interval: Optional[int] = None       # log CUDA mem every N steps if >0
    # Aggressive offload controls
    aggressive_offload: bool = True              # legacy switch; see offload_level
    te_device: Optional[str] = None              # 'cuda' | 'cpu' (None = follow cfg.device)
    te_impl: Optional[str] = None                # 'cuda_fp8' | 'hf' (None = default)
    te_kernel_required: Optional[bool] = None    # if True, error if CUDA kernel unavailable
    # New: coarse-grained offload profile (takes precedence over aggressive_offload if provided)
    # 0 = off (keep resident), 1 = light (offload TE/VAE only), 2 = balanced (also clear between stages), 3 = aggressive (current behavior)
    offload_level: Optional[int] = None

def _as_dtype(dtype: str):
    return {
        'fp16': torch.float16,
        'bf16': getattr(torch, 'bfloat16', torch.float16),
        'fp32': torch.float32,
    }.get(str(dtype).lower(), torch.float16)


def _get_logger(logger: Any):
    import logging
    if logger is not None:
        return logger
    lg = logging.getLogger("wan22.gguf")
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter('[wan22.gguf] %(levelname)s: %(message)s')
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg

def _cuda_empty_cache(logger=None, label: str = "gc") -> None:
    try:
        import torch
        if not (getattr(torch, 'cuda', None) and torch.cuda.is_available()):
            return
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved_before = torch.cuda.memory_reserved() // (1024 * 1024)
        torch.cuda.empty_cache()
        alloc_after = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved_after = torch.cuda.memory_reserved() // (1024 * 1024)
        _li(logger, "[wan22.gguf] cuda.gc(%s): alloc %d→%d MB reserved %d→%d MB", label, alloc_before, alloc_after, reserved_before, reserved_after)
    except Exception:
        pass


def _infer_patch_geometry(
    model,
    *,
    T: int,
    H_lat: int,
    W_lat: int,
) -> PatchGeometry:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("WAN22: expected model with .config (WanTransformer2DModel)")
    kT, kH, kW = tuple(int(x) for x in getattr(cfg, "patch_size", (1, 2, 2)))
    if T < kT or H_lat < kH or W_lat < kW:
        raise RuntimeError(
            f"WAN22: invalid latent shape for patch_embed: T={T} H={H_lat} W={W_lat} kernel={(kT, kH, kW)}"
        )
    gT = int(T - kT + 1)
    gH = int(((H_lat - kH) // kH) + 1)
    gW = int(((W_lat - kW) // kW) + 1)
    L = int(gT * gH * gW)
    Cout = int(getattr(cfg, "d_model", 0) or 0)
    Cin = int(getattr(cfg, "in_channels", 0) or 0)
    return PatchGeometry(
        grid=(gT, gH, gW),
        token_count=L,
        token_dim=Cout,
        latent_channels=Cin,
        patch_kernel=(kT, kH, kW),
    )


@_io
def _make_scheduler(steps: int, *, sampler: Optional[str] = None, scheduler: Optional[str] = None):
    """Instantiate a Diffusers scheduler based on requested sampler/scheduler names.

    Defaults to Euler when unspecified. No hardcoded counts; `steps` is passed through.
    """
    from diffusers import (
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )

    s = (sampler or "").strip().lower()
    sch = (scheduler or "").strip().lower()

    cls = EulerDiscreteScheduler  # default
    # Try sampler first (explicit user intent)
    if s in ("euler",):
        cls = EulerDiscreteScheduler
    elif s in ("euler a", "euler_a", "euler-ancestral", "ancestral"):  # tolerant forms
        cls = EulerAncestralDiscreteScheduler
    elif s in ("ddim",):
        cls = DDIMScheduler
    elif s in ("dpm++ 2m", "dpm++ 2m sde", "dpm2m", "dpmpp2m", "dpmpp2m sde"):
        cls = DPMSolverMultistepScheduler
    elif s in ("plms", "lms"):
        cls = LMSDiscreteScheduler
    elif s in ("pndm",):
        cls = PNDMScheduler
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


@_io
def _cfg_merge(uncond: torch.Tensor, cond: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return cond
    return uncond + (cond - uncond) * float(scale)


@_io
def _log_cuda_mem(logger: Any, label: str = "mem") -> None:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = float(torch.cuda.memory_allocated()) / (1024**2)
            reserv = float(torch.cuda.memory_reserved()) / (1024**2)
            total = float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
            logger.info("[wan22.gguf] %s: cuda mem alloc=%.0fMB reserved=%.0fMB total=%.0fMB", label, alloc, reserv, total)
    except Exception:
        pass


@_io
def _log_t_mapping(scheduler, timesteps, label: str, logger: Any) -> None:
    try:
        log = _get_logger(logger)
        n = len(timesteps)
        idxs = [0, max(0, n // 2 - 1), n - 1]
        vals: list[float] = []
        sigmas = getattr(scheduler, 'sigmas', None)
        for i in idxs:
            sig_ok = bool(sigmas is not None and len(sigmas) in (n, n + 1))
            if sig_ok:
                s = float(sigmas[i])
                s_min = float(sigmas[-1])
                s_max = float(sigmas[0])
                t = max(0.0, min(1.0, (s - s_min) / (s_max - s_min))) if (s_max - s_min) > 0 else 0.0
            else:
                t = 1.0 - (float(i) / float(max(1, n - 1)))
            vals.append(t)
        log.info(
            "[wan22.gguf] t-map(%s): t0=%.4f tmid=%.4f tend=%.4f (sigmas=%s)",
            label,
            vals[0],
            vals[1],
            vals[2],
            bool(sigmas is not None and len(sigmas) in (n, n + 1)),
        )
    except Exception:
        pass


def _time_snr_shift(alpha: float, t: float) -> float:
    # Same functional form as time_snr_shift used in reference implementations
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)




@_io
def _resolve_device_name(name: str) -> str:
    s = (name or 'auto').lower().strip()
    if s == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if s == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@_io
def run_txt2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> list[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    hi_model = _load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.05)
        except Exception:
            pass

    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_t2v_{_variant}"

    lvl = cfg.offload_level if cfg.offload_level is not None else (3 if getattr(cfg, 'aggressive_offload', True) else 0)
    te_dev_eff = getattr(cfg, 'te_device', None)
    if te_dev_eff is None:
        te_dev_eff = 'cuda' if lvl <= 1 else 'cpu'
    te_impl_val = (getattr(cfg, 'te_impl', None) or os.getenv('WAN_TE_IMPL', 'hf')).lower()
    te_required_val = (te_impl_val == 'cuda_fp8')
    _li(logger, "[wan22.gguf] offload profile: level=%s te_device=%s te_impl=%s te_required=%s",
        lvl, te_dev_eff, te_impl_val, str(te_required_val).lower())

    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        offload_after=(lvl >= 1),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, 'te_impl', None),
        te_kernel_required=getattr(cfg, 'te_kernel_required', None),
    )
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))
    geom_hi = _infer_patch_geometry(hi_model, T=T, H_lat=H_lat, W_lat=W_lat)
    log.info(
        "[wan22.gguf] HIGH geom: grid=%s kernel=%s cin=%d",
        geom_hi.grid,
        geom_hi.patch_kernel,
        geom_hi.latent_channels,
    )
    _log_cuda_mem(log, label='after-high-setup')
    if getattr(cfg, 'aggressive_offload', True):
        _cuda_empty_cache(log, label='pre-high')
    if on_progress:
        try:
            on_progress(stage='prepare', step=1, total=1, percent=0.15)
        except Exception:
            pass

    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, 'flow_shift', None) if cfg.high else None
    log.info(
        "[wan22.gguf] HIGH: steps=%s sampler=%s scheduler=%s cfg_scale=%s seed=%s",
        steps_hi,
        sampler_hi,
        sched_hi,
        (getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        cfg.seed,
    )

    latents_hi = _sample_stage_latents(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        seed=cfg.seed,
        on_progress=(lambda **p: on_progress(stage='high', **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_hi) if flow_shift_hi is not None else WAN_FLOW_SHIFT_DEFAULT,
        stage_name='high',
    )

    if str(os.getenv('WAN_I2V_DEBUG_HI_DECODE', '0')).strip().lower() in ('1', 'true', 'yes', 'on'):
        try:
            _ = _decode_latents_to_frames(latents=latents_hi, model_dir=os.path.dirname(hi_path), cfg=cfg, logger=log, debug_preview=True)
        except Exception:
            _lw(log, "[wan22.gguf] debug high decode failed", exc_info=True)

    if getattr(cfg, 'aggressive_offload', True):
        try:
            del hi_model
        except Exception:
            pass
        _cuda_empty_cache(log, label='after-high')

    lo_model = _load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = _infer_patch_geometry(lo_model, T=T, H_lat=H_lat, W_lat=W_lat)
    log.info(
        "[wan22.gguf] LOW geom: grid=%s kernel=%s cin=%d",
        geom_lo.grid,
        geom_lo.patch_kernel,
        geom_lo.latent_channels,
    )

    seed_latents = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, 'flow_shift', None) if cfg.low else None
    log.info(
        "[wan22.gguf] LOW: steps=%s sampler=%s scheduler=%s cfg_scale=%s",
        steps_lo,
        sampler_lo,
        sched_lo,
        (getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
    )

    latents_lo = _sample_stage_latents(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        seed=None,
        state_init=seed_latents,
        on_progress=(lambda **p: on_progress(stage='low', **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_lo) if flow_shift_lo is not None else WAN_FLOW_SHIFT_DEFAULT,
        stage_name='low',
    )

    frames_lo = _decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
    if getattr(cfg, 'aggressive_offload', True):
        _cuda_empty_cache(log, label='after-decode')
    _try_clear_cache()
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames_lo


def stream_txt2vid(cfg: RunConfig, *, logger=None):
    """Generator that yields progress and final frames for txt2vid."""
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)

    hi_model = _load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_t2v_{_variant}"
    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=cfg.device,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        te_device=getattr(cfg, 'te_device', None),
        te_impl=getattr(cfg, 'te_impl', None),
        te_kernel_required=getattr(cfg, 'te_kernel_required', None),
    )
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))
    geom_hi = _infer_patch_geometry(hi_model, T=T, H_lat=H_lat, W_lat=W_lat)
    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, 'flow_shift', None) if cfg.high else None

    latents_hi = yield from _sample_stage_latents_generator(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        seed=cfg.seed,
        state_init=None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_hi) if flow_shift_hi is not None else WAN_FLOW_SHIFT_DEFAULT,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name='high',
        emit_logs=False,
    )

    if str(os.getenv('WAN_I2V_DEBUG_HI_DECODE', '0')).strip().lower() in ('1','true','yes','on'):
        try:
            _ = _decode_latents_to_frames(latents=latents_hi, model_dir=os.path.dirname(hi_path), cfg=cfg, logger=log, debug_preview=True)
        except Exception:
            _lw(log, "[wan22.gguf] debug high decode failed", exc_info=True)

    lo_model = _load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = _infer_patch_geometry(lo_model, T=T, H_lat=H_lat, W_lat=W_lat)
    seed_latents = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, 'flow_shift', None) if cfg.low else None

    latents_lo = yield from _sample_stage_latents_generator(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        seed=None,
        state_init=seed_latents,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_lo) if flow_shift_lo is not None else WAN_FLOW_SHIFT_DEFAULT,
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name='low',
        emit_logs=False,
    )

    frames_lo = _decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames_lo}


def run_img2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> list[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    hi_model = _load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.05)
        except Exception:
            pass

    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_i2v_{_variant}"
    lvl = cfg.offload_level if cfg.offload_level is not None else (3 if getattr(cfg, 'aggressive_offload', True) else 0)
    te_dev_eff = getattr(cfg, 'te_device', None) or ('cuda' if lvl <= 1 else 'cpu')
    te_impl_val = (getattr(cfg, 'te_impl', None) or os.getenv('WAN_TE_IMPL', 'hf')).lower()
    te_required_val = (te_impl_val == 'cuda_fp8')
    _li(logger, "[wan22.gguf] offload profile: level=%s te_device=%s te_impl=%s te_required=%s",
        lvl, te_dev_eff, te_impl_val, str(te_required_val).lower())

    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        offload_after=(lvl >= 1),
        te_device=(cfg.te_device or te_dev_eff),
        te_impl=getattr(cfg, 'te_impl', None),
        te_kernel_required=getattr(cfg, 'te_kernel_required', None),
    )
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))

    lat0 = _vae_encode_init(cfg.init_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat0.ndim == 4:
        lat0 = lat0.unsqueeze(2)
    lat0 = lat0.repeat(1, 1, T, 1, 1)
    lat0 = _resize_latents_hw(lat0, H=H_lat, W=W_lat)

    geom_hi = _infer_patch_geometry(hi_model, T=T, H_lat=H_lat, W_lat=W_lat)
    seed_hi = _prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)

    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, 'flow_shift', None) if cfg.high else None

    latents_hi = _sample_stage_latents(
        model=hi_model,
        geom=geom_hi,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        seed=None,
        state_init=seed_hi,
        on_progress=(lambda **p: on_progress(stage='high', **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_hi) if flow_shift_hi is not None else WAN_FLOW_SHIFT_DEFAULT,
        stage_name='high',
    )

    if str(os.getenv('WAN_I2V_DEBUG_HI_DECODE', '0')).strip().lower() in ('1','true','yes','on'):
        try:
            _ = _decode_latents_to_frames(latents=latents_hi, model_dir=os.path.dirname(hi_path), cfg=cfg, logger=log, debug_preview=True)
        except Exception:
            _lw(log, "[wan22.gguf] debug high decode failed", exc_info=True)

    if lvl >= 2:
        try:
            del hi_model
        except Exception:
            pass
        _cuda_empty_cache(logger=log, label='after-high')

    lo_model = _load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = _infer_patch_geometry(lo_model, T=T, H_lat=H_lat, W_lat=W_lat)
    seed_lo = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, 'flow_shift', None) if cfg.low else None

    latents_lo = _sample_stage_latents(
        model=lo_model,
        geom=geom_lo,
        steps=steps_lo,
        cfg_scale=(getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_lo,
        scheduler_name=sched_lo,
        seed=None,
        state_init=seed_lo,
        on_progress=(lambda **p: on_progress(stage='low', **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(flow_shift_lo) if flow_shift_lo is not None else WAN_FLOW_SHIFT_DEFAULT,
        stage_name='low',
    )

    frames = _decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
    _try_clear_cache()
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames


def stream_img2vid(cfg: RunConfig, *, logger=None):
    log = _get_logger(logger)
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")

    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")

    set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)

    hi_model = _load_stage_model_from_gguf(hi_path, device=dev, dtype=dt, logger=log)
    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_i2v_{_variant}"

    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=cfg.device,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        te_device=getattr(cfg, 'te_device', None),
        te_impl=getattr(cfg, 'te_impl', None),
        te_kernel_required=getattr(cfg, 'te_kernel_required', None),
    )
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))

    lat0 = _vae_encode_init(cfg.init_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat0.ndim == 4:
        lat0 = lat0.unsqueeze(2)
    lat0 = lat0.repeat(1, 1, T, 1, 1)
    lat0 = _resize_latents_hw(lat0, H=H_lat, W=W_lat)

    geom_hi = _infer_patch_geometry(hi_model, T=T, H_lat=H_lat, W_lat=W_lat)
    seed_hi = _prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)

    latents_hi = yield from _sample_stage_latents_generator(
        model=hi_model,
        geom=geom_hi,
        steps=int(getattr(cfg.high, 'steps', 12) if cfg.high else 12),
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=(getattr(cfg.high, 'sampler', None) if cfg.high else None),
        scheduler_name=(getattr(cfg.high, 'scheduler', None) if cfg.high else None),
        seed=None,
        state_init=seed_hi,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(getattr(cfg.high, 'flow_shift', WAN_FLOW_SHIFT_DEFAULT) if cfg.high else WAN_FLOW_SHIFT_DEFAULT),
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name='high',
        emit_logs=False,
    )

    lo_model = _load_stage_model_from_gguf(lo_path, device=dev, dtype=dt, logger=log)
    geom_lo = _infer_patch_geometry(lo_model, T=T, H_lat=H_lat, W_lat=W_lat)
    seed_lo = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    latents_lo = yield from _sample_stage_latents_generator(
        model=lo_model,
        geom=geom_lo,
        steps=int(getattr(cfg.low, 'steps', 12) if cfg.low else 12),
        cfg_scale=(getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=(getattr(cfg.low, 'sampler', None) if cfg.low else None),
        scheduler_name=(getattr(cfg.low, 'scheduler', None) if cfg.low else None),
        seed=None,
        state_init=seed_lo,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
        flow_shift=float(getattr(cfg.low, 'flow_shift', WAN_FLOW_SHIFT_DEFAULT) if cfg.low else WAN_FLOW_SHIFT_DEFAULT),
        flow_multiplier=WAN_FLOW_MULTIPLIER,
        stage_name='low',
        emit_logs=False,
    )

    frames = _decode_latents_to_frames(latents=latents_lo, model_dir=os.path.dirname(lo_path), cfg=cfg, logger=log)
    if not frames:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames}


__all__ = [
    "RunConfig",
    "StageConfig",
    "run_txt2vid",
    "run_img2vid",
    "stream_txt2vid",
    "stream_img2vid",
]

def _resize_latents_hw(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    import torch.nn.functional as F
    if x.ndim == 5:
        B, C, T, h, w = x.shape
        if h == H and w == W:
            return x
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, h, w)
        xt = F.interpolate(xt, size=(int(H), int(W)), mode='bilinear', align_corners=False)
        xt = xt.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return xt
    if x.ndim == 4:
        import torch.nn.functional as F  # type: ignore
        B, C, h, w = x.shape
        if h == H and w == W:
            return x
        return F.interpolate(x, size=(int(H), int(W)), mode='bilinear', align_corners=False)
    return x
