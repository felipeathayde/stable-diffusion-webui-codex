from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import os
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from apps.backend.runtime.utils import _load_gguf_state_dict, read_arbitrary_config
from apps.backend.runtime.ops.operations_gguf import dequantize_tensor
from apps.backend.runtime import memory_management
import logging
from apps.backend.engines.wan22.wan22_common import resolve_wan_repo_candidates
from diffusers import AutoencoderKLWan  # type: ignore

from functools import wraps
"""WAN 2.2 — GGUF path (generic, PyTorch‑first, no custom kernels).

This module ports useful pieces from the prior WAN GGUF code into the generic
runtime, using only:
- apps.backend.gguf (readers/quants)
- apps.backend.runtime.ops (dequantize, ops)
- PyTorch SDPA for attention

It provides:
- derive_spec_from_state(): parse GGUF state keys into a model spec
- WanDiTGGUF: minimal Diffusion Transformer (DiT) wrapper with forward over SA/CA/FFN stacks
- run_txt2vid/run_img2vid: skeletons that validate stages and prepare flow

Notes
- VAE encode/decode and full sampler loop are wired later; we keep errors
  explicit instead of faking outputs.
"""
# Local latent normalization (Comfy-inspired). Try relative first, then absolute for robustness.
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
_LOG_ONCE = {
    'patch_embed': False,
    'patch_unembed': False,
    'sdpa': False,
}
_SDPA_LOG_COUNT = 0

WAN_FLOW_SHIFT_DEFAULT = 8.0
WAN_FLOW_MULTIPLIER = 1000.0

def _get_logger_legacy(logger: Any):
    # Legacy duplicate; keep for compatibility if referenced elsewhere
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

def _resolve_i2v_order() -> str:
    """Return channel order for I2V concatenation.
    - 'lat_first': latents(16) then cond extras (mask4+img16) → matches Comfy (xc + c_concat).
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

@_io
def _patch_embed3d(video, w, b):
    import torch
    from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = video.device
    dtype = video.dtype
    use_fp32 = str(os.getenv('WAN_I2V_CONV32','0')).strip().lower() in ('1','true','yes','on')
    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    old_dtype = getattr(W, 'dtype', None)
    W = W.to(device=device, dtype=(torch.float32 if use_fp32 else dtype))
    if _log_enabled('debug'):
        _ld(None, "[wan22.gguf] dtype(cast): patch_embedding.weight from %s to %s", str(old_dtype), str(W.dtype))
    bias = None
    if b is not None:
        old_bd = getattr(b, 'dtype', None)
        bias = b.to(device=device, dtype=(torch.float32 if use_fp32 else dtype))
        if _log_enabled('debug'):
            _ld(None, "[wan22.gguf] dtype(cast): patch_embedding.bias from %s to %s", str(old_bd), str(bias.dtype))
    B, C, T, H, Wd = video.shape
    kCout, kCin, kT, kH, kW = W.shape
    if C != kCin:
        raise RuntimeError(f"patch_embed: C_in mismatch: video C={C} vs weight {kCin}")
    if use_fp32 and video.dtype != torch.float32:
        video = video.to(torch.float32)
    y = torch.nn.functional.conv3d(video, W, bias=bias, stride=(1, kH, kW), padding=(0, 0, 0))
    if use_fp32 and dtype != torch.float32:
        y = y.to(dtype)
    B2, Cout, T2, H2, W2 = y.shape
    tokens = y.permute(0, 2, 3, 4, 1).contiguous().view(B2, T2 * H2 * W2, Cout)
    # One-time shape log for debugging
    global _LOG_ONCE
    if not _LOG_ONCE.get('patch_embed', False):
        _LOG_ONCE['patch_embed'] = True
        try:
            from .nn import wan22 as _self  # self-module for _li
        except Exception:
            _self = None
        try:
            (_self or globals()).get('_li', lambda *a, **k: None)(None, "[wan22.gguf] patch_embed3d: video=%s W=%s tokens=%s grid=(%d,%d,%d)", tuple(video.shape), tuple(W.shape), tuple(tokens.shape), T2, H2, W2)
        except Exception:
            pass
    return tokens, (T2, H2, W2)


def _repeat_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    if x.shape[1] <= 0:
        raise RuntimeError("repeat_to_length: modulation tensor has zero length")
    repeats = math.ceil(target_len / x.shape[1])
    tiled = x.repeat(1, repeats, 1)
    return tiled[:, :target_len]


def _unpatchify_tokens(
    patch_tokens: torch.Tensor,
    grid: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    latent_channels: int,
) -> torch.Tensor:
    B, L, feat = patch_tokens.shape
    gT, gH, gW = (int(grid[0]), int(grid[1]), int(grid[2]))
    pT, pH, pW = patch_size
    patch_volume = int(pT * pH * pW)
    if L != gT * gH * gW:
        raise RuntimeError(
            f"unpatchify: token length {L} does not match grid ({gT},{gH},{gW})"
        )
    expected_feat = latent_channels * patch_volume
    if feat != expected_feat:
        raise RuntimeError(
            f"unpatchify: feature dim {feat} expected {expected_feat}"
        )
    x = patch_tokens.view(B, gT, gH, gW, pT, pH, pW, latent_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(B, latent_channels, gT * pT, gH * pH, gW * pW)


def _apply_head_and_unpatch(
    tokens: torch.Tensor,
    *,
    spec: ModelSpec,
    state: Mapping[str, Any],
    tproj: torch.Tensor,
    grid: Tuple[int, int, int],
) -> torch.Tensor:
    if spec.head_weight is None or spec.patch_kernel is None or spec.latent_channels is None:
        raise RuntimeError(
            "WAN22 GGUF: missing head weights or patch geometry; ensure the GGUF includes head.head.* tensors."
        )
    weight = state.get(spec.head_weight)
    if weight is None:
        raise RuntimeError(f"Missing head weight: {spec.head_weight}")
    bias = state.get(spec.head_bias) if spec.head_bias else None

    device = tokens.device
    dtype = tokens.dtype

    normed = _layer_norm(tokens)
    token_len = tokens.shape[1]

    shift = tokens.new_zeros(tokens.shape[0], token_len, tokens.shape[2])
    scale = tokens.new_zeros_like(shift)
    if spec.head_modulation and spec.head_modulation in state:
        mod_param = state[spec.head_modulation]
        mod_param = dequantize_tensor(mod_param)
        if not torch.is_tensor(mod_param):
            mod_param = torch.as_tensor(mod_param)
        mod_param = mod_param.to(device=device, dtype=dtype)
        tp = tproj.to(device=device, dtype=dtype)
        if tp.ndim != 3:
            tp = tp.view(tp.shape[0], -1, tp.shape[-1])
        combined = mod_param.unsqueeze(0) + tp.unsqueeze(2)
        shift6, scale6 = combined.unbind(dim=2)
        shift = _repeat_to_length(shift6, token_len)
        scale = _repeat_to_length(scale6, token_len)

    fused = shift + normed * (1.0 + scale)
    patches = _linear(fused, weight, bias, name=spec.head_weight)

    return _unpatchify_tokens(patches, grid, spec.patch_kernel, spec.latent_channels)


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
    dit: 'WanDiTGGUF',
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
        dit=dit,
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
    dit: 'WanDiTGGUF',
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
    if geom.latent_channels is None:
        raise RuntimeError("Patch geometry missing latent channel count")
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

    dtype_tag = {
        torch.float16: 'fp16',
        getattr(torch, 'bfloat16', torch.float16): 'bf16',
        torch.float32: 'fp32',
    }.get(dtype, 'fp32')

    patch_w, patch_b = _resolve_patch_weights(dit.state)

    flow_progress = torch.linspace(1.0, 0.0, total, device=device, dtype=torch.float32) if total > 1 else torch.ones(1, device=device, dtype=torch.float32)

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

        tokens, grid_cur = _patch_embed3d(state, patch_w, patch_b)
        if grid_cur != geom.grid:
            try:
                log.warning("[wan22.gguf] grid mismatch: expected %s got %s", geom.grid, grid_cur)
            except Exception:
                pass

        eps_cond_tokens, tproj = dit.forward(tokens, di_timestep, prompt_embeds, dtype=dtype_tag, return_time_proj=True)
        eps_uncond_tokens = dit.forward(tokens, di_timestep, negative_embeds, dtype=dtype_tag, return_time_proj=False)

        eps_cond_latents = dit.tokens_to_latents(
            eps_cond_tokens,
            geom.grid,
            timestep=di_timestep,
            device=device,
            dtype=dtype,
            tproj=tproj,
        )
        eps_uncond_latents = dit.tokens_to_latents(
            eps_uncond_tokens,
            geom.grid,
            timestep=di_timestep,
            device=device,
            dtype=dtype,
            tproj=tproj,
        )

        eps = _cfg_merge(eps_uncond_latents, eps_cond_latents, cfg_scale)

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
) -> List[object]:
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
    # Concat order depends on WAN_I2V_ORDER (default: lat_first to match Comfy xc + c_concat)
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
def _patch_unembed3d(tokens, w, out_shape):
    import torch
    from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = tokens.device
    dtype = tokens.dtype
    use_fp32 = str(os.getenv('WAN_I2V_CONV32','0')).strip().lower() in ('1','true','yes','on')
    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    W = W.to(device=device, dtype=(torch.float32 if use_fp32 else dtype))
    B, L, Cout = tokens.shape
    kCout, kCin, kT, kH, kW = W.shape
    if Cout != kCout:
        raise RuntimeError(f"patch_unembed: C_out mismatch: tokens C={Cout} vs weight {kCout}")
    T2, H2, W2 = out_shape
    # Guard: if L != T2*H2*W2 (e.g., when seed latents H/W differ from cfg, or T inferred),
    # try to recompute T from L and spatial grid; if still mismatched, raise with a clear hint.
    expected_L = int(T2) * int(H2) * int(W2)
    if L != expected_L and H2 > 0 and W2 > 0 and (L % (H2 * W2) == 0):
        T2 = int(L // (H2 * W2))
        expected_L = int(T2) * int(H2) * int(W2)
    if L != expected_L:
        raise RuntimeError(
            f"patch_unembed: token length L={L} does not match grid (T,H',W')={out_shape} → expected {expected_L}. "
            f"This usually means the seed latents spatial size didn't match cfg; ensure init_image latent H/W align to cfg height/width." )
    y = tokens.view(B, T2, H2, W2, Cout).permute(0, 4, 1, 2, 3).contiguous().to(device=device, dtype=(torch.float32 if use_fp32 else dtype))
    video = torch.nn.functional.conv_transpose3d(y, W, bias=None, stride=(1, kH, kW), padding=(0, 0, 0))
    if use_fp32 and dtype != torch.float32:
        video = video.to(dtype)
    global _LOG_ONCE
    if not _LOG_ONCE.get('patch_unembed', False):
        _LOG_ONCE['patch_unembed'] = True
        try:
            (_self or globals()).get('_li', lambda *a, **k: None)(None, "[wan22.gguf] patch_unembed3d: tokens=%s W=%s out=%s grid=%s", tuple(tokens.shape), tuple(W.shape), tuple(video.shape), out_shape)
        except Exception:
            pass
    return video


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

    # Effective TE preferences (extras > env > defaults)
    try:
        te_impl_eff = (te_impl or os.getenv('WAN_TE_IMPL', '') or 'hf').strip().lower()
    except Exception:
        te_impl_eff = (te_impl or 'hf') if te_impl else 'hf'
    # No fallbacks allowed: if impl=cuda_fp8, kernel is REQUIRED
    te_req_eff = (te_impl_eff == 'cuda_fp8')
    te_dev_eff = (te_device or os.getenv('WAN_TE_DEVICE') or device or 'cpu').strip().lower()

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
        # Apply latent normalization for diffusion (Comfy-like)
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

try:  # progress bar for long loops (non-fatal if unavailable)
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


# ------------------------------ spec/mapping

@dataclass
class CrossAttnWeights:
    q_w: str | None = None
    q_b: str | None = None
    k_w: str | None = None
    k_b: str | None = None
    v_w: str | None = None
    v_b: str | None = None
    o_w: str | None = None
    o_b: str | None = None
    norm_q_w: str | None = None
    norm_q_b: str | None = None
    norm_k_w: str | None = None
    norm_k_b: str | None = None


@dataclass
class BlockSpec:
    index: int
    cross_attn: CrossAttnWeights = field(default_factory=CrossAttnWeights)
    self_attn: CrossAttnWeights = field(default_factory=CrossAttnWeights)
    ffn_in_w: Optional[str] = None
    ffn_in_b: Optional[str] = None
    ffn_out_w: Optional[str] = None
    ffn_out_b: Optional[str] = None
    norm3_w: Optional[str] = None
    norm3_b: Optional[str] = None
    modulation: Optional[str] = None  # [1,6,C]

@dataclass(frozen=True)
class PatchGeometry:
    grid: Tuple[int, int, int]
    token_count: int
    token_dim: int
    latent_channels: int
    patch_kernel: Tuple[int, int, int]


@dataclass
class ModelSpec:
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_blocks: int = 0
    blocks: List[BlockSpec] = field(default_factory=list)
    time_emb_0_w: Optional[str] = None
    time_emb_0_b: Optional[str] = None
    time_emb_2_w: Optional[str] = None
    time_emb_2_b: Optional[str] = None
    time_proj_w: Optional[str] = None
    time_proj_b: Optional[str] = None
    head_modulation: Optional[str] = None  # [1,2,C]
    # Optional text embedding projection (text_dim -> d_model)
    text_emb_0_w: Optional[str] = None
    text_emb_0_b: Optional[str] = None
    text_emb_2_w: Optional[str] = None
    text_emb_2_b: Optional[str] = None
    # Patch/head geometry
    patch_in_channels: Optional[int] = None
    patch_out_channels: Optional[int] = None
    patch_kernel: Optional[Tuple[int, int, int]] = None
    patch_stride: Optional[Tuple[int, int, int]] = None
    latent_channels: Optional[int] = None
    head_weight: Optional[str] = None
    head_bias: Optional[str] = None


@_io
def _shape_of(state: Mapping[str, object], key: str) -> Optional[Tuple[int, ...]]:
    v = state.get(key)
    if v is None:
        return None
    try:
        shp = tuple(int(s) for s in getattr(v, 'shape', tuple()))
        return shp if shp else None
    except Exception:
        return None


@_io
def derive_spec_from_state(state: Mapping[str, object]) -> ModelSpec:
    by_block: Dict[int, Dict[str, str]] = {}
    for k in state.keys():
        ks = str(k)
        if not ks.startswith("blocks."):
            continue
        try:
            rest = ks.split(".", 2)
            bi = int(rest[1])
            tail = rest[2]
        except Exception:
            continue
        by_block.setdefault(bi, {})[tail] = ks

    d_model: Optional[int] = None
    heads: Optional[int] = None
    if by_block:
        bk = by_block[min(by_block.keys())]
        for cname in ("cross_attn.q.weight", "cross_attn.k.weight", "cross_attn.o.weight"):
            key = bk.get(cname)
            if key:
                shp = _shape_of(state, key)
                if shp and len(shp) == 2:
                    d_model = int(shp[0] if cname.endswith("o.weight") else shp[1])
                    break
        if d_model and d_model % 128 == 0:
            h = d_model // 128
            if 8 <= h <= 64:
                heads = h

    blocks: List[BlockSpec] = []
    for bi in sorted(by_block.keys()):
        entries = by_block[bi]
        ca = CrossAttnWeights(
            q_w=entries.get("cross_attn.q.weight"), q_b=entries.get("cross_attn.q.bias"),
            k_w=entries.get("cross_attn.k.weight"), k_b=entries.get("cross_attn.k.bias"),
            v_w=entries.get("cross_attn.v.weight"), v_b=entries.get("cross_attn.v.bias"),
            o_w=entries.get("cross_attn.o.weight"), o_b=entries.get("cross_attn.o.bias"),
            norm_q_w=entries.get("cross_attn.norm_q.weight"), norm_q_b=entries.get("cross_attn.norm_q.bias"),
            norm_k_w=entries.get("cross_attn.norm_k.weight"), norm_k_b=entries.get("cross_attn.norm_k.bias"),
        )
        sa = CrossAttnWeights(
            q_w=entries.get("self_attn.q.weight"), q_b=entries.get("self_attn.q.bias"),
            k_w=entries.get("self_attn.k.weight"), k_b=entries.get("self_attn.k.bias"),
            v_w=entries.get("self_attn.v.weight"), v_b=entries.get("self_attn.v.bias"),
            o_w=entries.get("self_attn.o.weight"), o_b=entries.get("self_attn.o.bias"),
            norm_q_w=entries.get("self_attn.norm_q.weight"), norm_q_b=entries.get("self_attn.norm_q.bias"),
            norm_k_w=entries.get("self_attn.norm_k.weight"), norm_k_b=entries.get("self_attn.norm_k.bias"),
        )
        bspec = BlockSpec(index=bi, cross_attn=ca, self_attn=sa)
        bspec.ffn_in_w = entries.get("ffn.0.weight")
        bspec.ffn_in_b = entries.get("ffn.0.bias")
        bspec.ffn_out_w = entries.get("ffn.2.weight")
        bspec.ffn_out_b = entries.get("ffn.2.bias")
        bspec.norm3_w = entries.get("norm3.weight")
        bspec.norm3_b = entries.get("norm3.bias")
        bspec.modulation = entries.get("modulation")
        blocks.append(bspec)

    time_emb_0_w = "time_embedding.0.weight" if "time_embedding.0.weight" in state else None
    time_emb_0_b = "time_embedding.0.bias" if "time_embedding.0.bias" in state else None
    time_emb_2_w = "time_embedding.2.weight" if "time_embedding.2.weight" in state else None
    time_emb_2_b = "time_embedding.2.bias" if "time_embedding.2.bias" in state else None
    time_proj_w = "time_projection.1.weight" if "time_projection.1.weight" in state else None
    time_proj_b = "time_projection.1.bias" if "time_projection.1.bias" in state else None
    head_mod = "head.modulation" if "head.modulation" in state else None
    # Text embedding projection layers are optional
    text_emb_0_w = "text_embedding.0.weight" if "text_embedding.0.weight" in state else None
    text_emb_0_b = "text_embedding.0.bias" if "text_embedding.0.bias" in state else None
    text_emb_2_w = "text_embedding.2.weight" if "text_embedding.2.weight" in state else None
    text_emb_2_b = "text_embedding.2.bias" if "text_embedding.2.bias" in state else None

    patch_shape = _shape_of(state, "patch_embedding.weight")
    patch_in: Optional[int] = None
    patch_out: Optional[int] = None
    patch_kernel: Optional[Tuple[int, int, int]] = None
    patch_stride: Optional[Tuple[int, int, int]] = None
    if patch_shape and len(patch_shape) == 5:
        patch_out = int(patch_shape[0])
        patch_in = int(patch_shape[1])
        patch_kernel = (int(patch_shape[2]), int(patch_shape[3]), int(patch_shape[4]))
        patch_stride = (1, patch_kernel[1], patch_kernel[2])

    head_weight = "head.head.weight" if "head.head.weight" in state else None
    head_bias = "head.head.bias" if "head.head.bias" in state else None
    latent_channels: Optional[int] = None
    if head_weight and patch_kernel:
        hw_shape = _shape_of(state, head_weight)
        if hw_shape and len(hw_shape) == 2:
            patch_volume = int(patch_kernel[0] * patch_kernel[1] * patch_kernel[2])
            if patch_volume > 0 and hw_shape[0] % patch_volume == 0:
                latent_channels = hw_shape[0] // patch_volume

    return ModelSpec(
        d_model=d_model, n_heads=heads, n_blocks=len(blocks), blocks=blocks,
        time_emb_0_w=time_emb_0_w, time_emb_0_b=time_emb_0_b,
        time_emb_2_w=time_emb_2_w, time_emb_2_b=time_emb_2_b,
        time_proj_w=time_proj_w, time_proj_b=time_proj_b,
        head_modulation=head_mod,
        text_emb_0_w=text_emb_0_w, text_emb_0_b=text_emb_0_b,
        text_emb_2_w=text_emb_2_w, text_emb_2_b=text_emb_2_b,
        patch_in_channels=patch_in, patch_out_channels=patch_out,
        patch_kernel=patch_kernel, patch_stride=patch_stride,
        latent_channels=latent_channels,
        head_weight=head_weight, head_bias=head_bias,
    )


# ------------------------------ ops

@_io
def _rms_norm(x: torch.Tensor, w: Any) -> torch.Tensor:
    w = dequantize_tensor(w)
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    # Ensure weight lives on the same device/dtype as the activation
    w = w.to(device=x.device, dtype=x.dtype)
    eps = 1e-6
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w


@_io
def _layer_norm(x: torch.Tensor, w: Any | None = None, b: Any | None = None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm with optional affine params.
    - If w/b are None: LN without affine (used for pre-norm of SA/FFN).
    - If provided: cast to x device/dtype and apply as affine (used for norm3 before CA).
    """
    weight = None
    bias = None
    if w is not None:
        w = dequantize_tensor(w)
        if not torch.is_tensor(w):
            w = torch.as_tensor(w)
        weight = w.to(device=x.device, dtype=x.dtype)
    if b is not None:
        b = dequantize_tensor(b)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b)
        bias = b.to(device=x.device, dtype=x.dtype)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps)


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


@_io
def _linear(x: torch.Tensor, w: Any, b: Any | None, *, name: Optional[str] = None) -> torch.Tensor:
    # Dequantize and dtypecast with optional debug logs
    w = dequantize_tensor(w)
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    if _log_enabled('debug'):
        _ld(None, "[wan22.gguf] dtype(create): %s dtype=%s device=%s shape=%s", str(name or '<weight>'), str(w.dtype), str(w.device), tuple(w.shape))
    if w.dtype != x.dtype or w.device != x.device:
        old = (str(w.dtype), str(w.device))
        w = w.to(device=x.device, dtype=x.dtype)
        if _log_enabled('debug'):
            _ld(None, "[wan22.gguf] dtype(cast): %s from dtype=%s@%s to dtype=%s@%s", str(name or '<weight>'), old[0], old[1], str(w.dtype), str(w.device))
    if b is not None:
        b = dequantize_tensor(b)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b)
        if _log_enabled('debug'):
            _ld(None, "[wan22.gguf] dtype(create): %s dtype=%s device=%s shape=%s", str((name + ".bias") if name else '<bias>'), str(b.dtype), str(b.device), tuple(b.shape))
        if b.dtype != x.dtype or b.device != x.device:
            oldb = (str(b.dtype), str(b.device))
            b = b.to(device=x.device, dtype=x.dtype)
            if _log_enabled('debug'):
                _ld(None, "[wan22.gguf] dtype(cast): %s from dtype=%s@%s to dtype=%s@%s", str((name + ".bias") if name else '<bias>'), oldb[0], oldb[1], str(b.dtype), str(b.device))
    return torch.nn.functional.linear(x, w, b)


try:
    from contextlib import nullcontext
except Exception:  # pragma: no cover
    class nullcontext:  # type: ignore
        def __init__(self, *a, **k):
            ...
        def __enter__(self):
            return None
        def __exit__(self, *a):
            return False

_SDPA_SETTINGS = {
    'policy': 'mem_efficient',
    'chunk': 0,
}


def _set_sdpa_settings(policy: Optional[str], chunk: Optional[int]) -> None:
    # Allow override via env WAN_SDPA_POLICY when explicit policy is None
    env_pol = os.getenv('WAN_SDPA_POLICY', '').strip().lower() if os.getenv('WAN_SDPA_POLICY') else None
    pol = (policy or env_pol or _SDPA_SETTINGS['policy']).strip().lower()
    if pol not in ('mem_efficient', 'flash', 'math'):
        pol = _SDPA_SETTINGS['policy']
    ch = int(chunk) if (chunk is not None and int(chunk) > 0) else 0
    _SDPA_SETTINGS['policy'] = pol
    _SDPA_SETTINGS['chunk'] = ch


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol = str(_SDPA_SETTINGS['policy']).strip().lower()
    ch = int(_SDPA_SETTINGS['chunk'])
    # Prefer new context manager torch.nn.attention.sdpa_kernel(backends=...) when available
    ctx = nullcontext()
    eff = 'unknown'
    try:
        if q.is_cuda:
            from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # type: ignore[attr-defined]
            from torch.nn.attention import SDPBackend  # type: ignore[attr-defined]
            backend = {
                'flash': SDPBackend.FLASH_ATTENTION,
                'mem_efficient': SDPBackend.EFFICIENT_ATTENTION,
                'math': SDPBackend.MATH,
                'cudnn': getattr(SDPBackend, 'CUDNN_ATTENTION', SDPBackend.EFFICIENT_ATTENTION),
            }.get(pol, SDPBackend.EFFICIENT_ATTENTION)
            ctx = _sdpa_kernel(backend)
            eff = {
                SDPBackend.FLASH_ATTENTION: 'flash',
                SDPBackend.EFFICIENT_ATTENTION: 'mem_efficient',
                SDPBackend.MATH: 'math',
                getattr(SDPBackend, 'CUDNN_ATTENTION', SDPBackend.EFFICIENT_ATTENTION): 'cudnn',
            }.get(backend, pol)
    except Exception:
        # Fallback to legacy API (deprecated) only for compatibility on older torch builds
        try:
            if q.is_cuda and hasattr(torch.backends, 'cuda'):
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=(pol == 'flash'),
                    enable_math=(pol == 'math'),
                    enable_mem_efficient=(pol == 'mem_efficient'),
                )
                # Infer effective selection from flags (approximate on legacy API)
                try:
                    _b = torch.backends.cuda
                    if _b.is_flash_sdp_enabled():
                        eff = 'flash'
                    elif _b.is_mem_efficient_sdp_enabled():
                        eff = 'mem_efficient'
                    elif _b.is_math_sdp_enabled():
                        eff = 'math'
                except Exception:
                    eff = pol
        except Exception:
            ctx = nullcontext()

    # SDPA backend decision logging (throttled)
    global _LOG_ONCE, _SDPA_LOG_COUNT
    try:
        verbose = str(os.getenv('WAN_SDPA_DEBUG', '0')).strip().lower() in ('1', 'true', 'yes')
    except Exception:
        verbose = False
    # Determine interval (default 5, minimum 1)
    try:
        every = max(1, int(os.getenv('WAN_SDPA_DEBUG_EVERY', '5')))
    except Exception:
        every = 5
    _SDPA_LOG_COUNT += 1
    should_log = False
    if verbose:
        # Log every Nth call in verbose mode
        should_log = (_SDPA_LOG_COUNT % every == 0)
    else:
        # Non-verbose: keep single one-time log (first call only)
        if not _LOG_ONCE.get('sdpa', False):
            should_log = True
            _LOG_ONCE['sdpa'] = True
    if should_log:
        try:
            _li(None, "[wan22.gguf] sdpa[n=%d]: policy=%s effective=%s chunk=%d device=%s dtype=%s qkv=%s", _SDPA_LOG_COUNT, pol, eff, ch, str(q.device), str(q.dtype), (tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        except Exception:
            pass

    if ch and ch > 0:
        with ctx:
            B, H, L, D = q.shape
            out_chunks = []
            for s in range(0, L, ch):
                e = min(L, s + ch)
                out_chunks.append(torch.nn.functional.scaled_dot_product_attention(q[:, :, s:e], k, v, is_causal=causal))
            return torch.cat(out_chunks, dim=2)
    else:
        with ctx:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


@_io
def _split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    B, L, C = x.shape
    D = C // h
    return x.view(B, L, h, D).permute(0, 2, 1, 3).contiguous()


@_io
def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)


@_io
def _ca(x: torch.Tensor, ctx: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int, scale=None, shift=None) -> torch.Tensor:
    q_in = _rms_norm(x, state[w.norm_q_w]) if w.norm_q_w else x
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = _linear(q_in, state[w.q_w], state.get(w.q_b), name=w.q_w)
    k = _linear(_rms_norm(ctx, state[w.norm_k_w]) if w.norm_k_w else ctx, state[w.k_w], state.get(w.k_b), name=w.k_w)
    v = _linear(ctx, state[w.v_w], state.get(w.v_b), name=w.v_w)
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    out = _linear(a, state[w.o_w], state.get(w.o_b))
    return x + out


@_io
def _sa(x: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int, scale=None, shift=None) -> torch.Tensor:
    q_in = _rms_norm(x, state[w.norm_q_w]) if w.norm_q_w else x
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = _linear(q_in, state[w.q_w], state.get(w.q_b), name=w.q_w)
    k = _linear(_rms_norm(x, state[w.norm_k_w]) if w.norm_k_w else x, state[w.k_w], state.get(w.k_b), name=w.k_w)
    v = _linear(x, state[w.v_w], state.get(w.v_b), name=w.v_w)
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    out = _linear(a, state[w.o_w], state.get(w.o_b))
    return x + out


@_io
def _sa_core(x_in: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int) -> torch.Tensor:
    """Self-attention core that returns only the attention output (no residual, no scale/shift)."""
    q = _linear(x_in, state[w.q_w], state.get(w.q_b), name=w.q_w)
    k = _linear(x_in, state[w.k_w], state.get(w.k_b), name=w.k_w)
    v = _linear(x_in, state[w.v_w], state.get(w.v_b), name=w.v_w)
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    return _linear(a, state[w.o_w], state.get(w.o_b))


@_io
def _ca_core(x_in: torch.Tensor, ctx: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int) -> torch.Tensor:
    """Cross-attention core that returns only the attention output (no residual, no scale/shift).
    x_in is expected to be pre-normalized (norm3 when available).
    """
    q = _linear(x_in, state[w.q_w], state.get(w.q_b), name=w.q_w)
    k_in = _rms_norm(ctx, state[w.norm_k_w]) if w.norm_k_w else ctx
    k = _linear(k_in, state[w.k_w], state.get(w.k_b), name=w.k_w)
    v = _linear(ctx, state[w.v_w], state.get(w.v_b), name=w.v_w)
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    return _linear(a, state[w.o_w], state.get(w.o_b))


class WanDiTGGUF:
    def __init__(self, stage_dir: str, *, logger=None) -> None:
        self._logger = logger
        self.stage_dir = stage_dir
        self.state: Dict[str, Any] = self._load_state(stage_dir)
        self.spec: ModelSpec = derive_spec_from_state(self.state)

    def _load_state(self, stage_dir: str) -> Dict[str, Any]:
        path = _pick_stage_gguf(stage_dir, 'high') or _pick_stage_gguf(stage_dir, 'low')
        if not path or not os.path.isfile(path):
            raise RuntimeError(f".gguf not found in {stage_dir}")
        state = _load_gguf_state_dict(path)
        # bake once for speed on first use
        class _D:
            def parameters(self_inner):
                for v in state.values():
                    if hasattr(v, 'gguf_cls'):
                        yield v
        try:
            memory_management.bake_gguf_model(_D())
        except Exception:
            pass
        if self._logger:
            keys = list(state.keys())
            self._logger.info("[wan22.gguf] tensors=%d sample=%s", len(keys), keys[:3])
        return state

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        cond: torch.Tensor,
        *,
        dtype: str = "bf16",
        return_time_proj: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        spec = self.spec
        if spec.d_model is None or spec.n_heads is None or not spec.blocks:
            raise RuntimeError("WAN22 spec incomplete (d_model/heads/blocks)")
        C = spec.d_model
        H = spec.n_heads

        # dtype cast
        tt = {
            'fp16': torch.float16,
            'bf16': getattr(torch, 'bfloat16', torch.float16),
            'fp32': torch.float32,
        }.get(dtype, torch.float16)

        device = x.device
        cond = cond.to(device=device, dtype=tt)
        x = x.to(device=device, dtype=tt)

        # Time embedding (sinusoidal -> 5120 -> proj -> [B,6,C])
        if isinstance(t, torch.Tensor):
            t_in = t.to(device=device, dtype=torch.float32).view(-1)
        elif isinstance(t, (int, float)):
            t_in = torch.tensor([float(t)], device=device, dtype=torch.float32)
        else:
            t_in = torch.as_tensor(t, device=device, dtype=torch.float32).view(-1)
        if t_in.numel() == 1 and x.shape[0] > 1:
            t_in = t_in.expand(x.shape[0])

        tproj = self._compute_time_proj(t_in, device=device, dtype=tt)

        # Text embedding projection (if weights are present)
        ctx = cond
        if spec.text_emb_0_w and spec.text_emb_2_w:
            te0_w = self.state.get(spec.text_emb_0_w); te0_b = self.state.get(spec.text_emb_0_b)
            te2_w = self.state.get(spec.text_emb_2_w); te2_b = self.state.get(spec.text_emb_2_b)
            if te0_w is None or te2_w is None:
                raise RuntimeError("Missing text_embedding weights")
            ctx = _linear(ctx, te0_w, te0_b, name=spec.text_emb_0_w or 'text_embedding.0.weight')
            # GELU (approximate OK)
            ctx = torch.nn.functional.gelu(ctx)
            ctx = _linear(ctx, te2_w, te2_b, name=spec.text_emb_2_w or 'text_embedding.2.weight')
        else:
            # If no projection weights exist, require ctx dim to match d_model
            if ctx.shape[-1] != C:
                raise RuntimeError(f"WAN22 GGUF: text embedding dim {ctx.shape[-1]} != model d_model {C} and no text_embedding.* weights found.")

        h = x
        for bs in spec.blocks:
            # per-block modulation slices from tproj (+ optional modulation sum)
            e = tproj
            if bs.modulation and bs.modulation in self.state:
                mod = self.state[bs.modulation]
                mod = dequantize_tensor(mod)
                if not torch.is_tensor(mod):
                    mod = torch.as_tensor(mod, device=device, dtype=tt)
                else:
                    mod = mod.to(device=device, dtype=tt)
                # additive composition (ComfyUI semantics): e = tproj + mod
                # broadcast along batch if needed
                if mod.dim() == 2:  # [6,C]
                    mod = mod.unsqueeze(0)
                e = tproj + mod

            # Unpack e: [sa_shift, sa_scale, sa_gate, ffn_shift, ffn_scale, ffn_gate]
            sa_shift = e[:, 0]
            sa_scale = e[:, 1]
            sa_gate  = e[:, 2]
            ffn_shift = e[:, 3]
            ffn_scale = e[:, 4]
            ffn_gate  = e[:, 5]

            # Self-attention with pre-norm LN (no affine) and gating
            if bs.self_attn.q_w and bs.self_attn.k_w and bs.self_attn.v_w and bs.self_attn.o_w:
                x_sa = _layer_norm(h)  # LN no affine
                x_sa = x_sa * (1 + sa_scale[:, None, :]) + sa_shift[:, None, :]
                sa_out = _sa_core(x_sa, w=bs.self_attn, state=self.state, heads=H)
                h = h + sa_out * sa_gate[:, None, :]

            # Cross-attention with norm3 (affine) when available; no scale/shift from e
            x_ca = h
            if bs.norm3_w:
                x_ca = _layer_norm(h, self.state[bs.norm3_w], self.state.get(bs.norm3_b))
            ca_out = _ca_core(x_ca, ctx, w=bs.cross_attn, state=self.state, heads=H)
            h = h + ca_out

            # FFN with pre-norm LN (no affine) and gating
            if bs.ffn_in_w and bs.ffn_out_w:
                x_ffn = _layer_norm(h)
                x_ffn = x_ffn * (1 + ffn_scale[:, None, :]) + ffn_shift[:, None, :]
                u = _linear(x_ffn, self.state[bs.ffn_in_w], self.state.get(bs.ffn_in_b), name=bs.ffn_in_w or f'ffn.{bs.index}.0.weight')
                u = u * torch.sigmoid(u)  # SiLU
                u = _linear(u, self.state[bs.ffn_out_w], self.state.get(bs.ffn_out_b), name=bs.ffn_out_w or f'ffn.{bs.index}.2.weight')
                h = h + u * ffn_gate[:, None, :]
        if return_time_proj:
            return h, tproj
        return h

    def _compute_time_proj(
        self,
        t_in: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        spec = self.spec
        if spec.d_model is None:
            raise RuntimeError("Model spec missing d_model for time projection")

        te0_w = self.state.get(spec.time_emb_0_w)
        te0_b = self.state.get(spec.time_emb_0_b)
        te2_w = self.state.get(spec.time_emb_2_w)
        te2_b = self.state.get(spec.time_emb_2_b)
        tp_w = self.state.get(spec.time_proj_w)
        tp_b = self.state.get(spec.time_proj_b)
        for name, tensor in {
            'time_embedding.0.weight': te0_w,
            'time_embedding.0.bias': te0_b,
            'time_embedding.2.weight': te2_w,
            'time_embedding.2.bias': te2_b,
            'time_projection.1.weight': tp_w,
            'time_projection.1.bias': tp_b,
        }.items():
            if tensor is None:
                raise RuntimeError(f"Missing weight: {name}")

        base_dim = int(_shape_of(self.state, spec.time_emb_0_w)[-1] if _shape_of(self.state, spec.time_emb_0_w) else 256)
        half = max(base_dim // 2, 1)
        freq = torch.arange(half, device=device, dtype=torch.float32)
        div_term = torch.exp(-math.log(10000.0) * freq / max(half - 1, 1))
        angles = t_in[:, None] * div_term[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] != base_dim:
            emb = torch.nn.functional.pad(emb, (0, base_dim - emb.shape[1]))
        emb = emb.to(dtype=dtype)

        t5120 = _linear(emb, te0_w, te0_b, name=spec.time_emb_0_w or 'time_embedding.0.weight')
        t5120 = t5120 * torch.sigmoid(t5120)
        t5120 = _linear(t5120, te2_w, te2_b, name=spec.time_emb_2_w or 'time_embedding.2.weight')
        tproj = _linear(t5120, tp_w, tp_b, name=spec.time_proj_w or 'time_projection.1.weight')
        return tproj.view(t5120.shape[0], 6, spec.d_model)

    def tokens_to_latents(
        self,
        tokens: torch.Tensor,
        grid: Tuple[int, int, int],
        *,
        timestep: float | None = None,
        device: torch.device,
        dtype: torch.dtype,
        tproj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.to(device=device, dtype=dtype)
        spec = self.spec
        if spec.head_weight is None or spec.patch_kernel is None or spec.latent_channels is None:
            raise RuntimeError(
                "WAN22 GGUF: head weights missing; cannot reconstruct latents without head.head.* tensors."
            )
        proj = tproj
        if proj is None:
            time_value = float(0.0 if timestep is None else timestep)
            t_in = torch.full((tokens.shape[0],), time_value, device=device, dtype=torch.float32)
            proj = self._compute_time_proj(t_in, device=device, dtype=dtype)
        return _apply_head_and_unpatch(tokens, spec=spec, state=self.state, tproj=proj, grid=grid)


# ------------------------------ helpers for stage files

def _normalize_win_path(p: str) -> str:
    if os.name == 'nt':
        return p
    if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip('\\/')
        return f"/mnt/{drive}/" + rest.replace('\\\\', '/').replace('\\', '/')
    return p


def _first_gguf_in(dir_path: Optional[str]) -> Optional[str]:
    if not dir_path:
        return None
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        for fn in os.listdir(abspath):
            if fn.lower().endswith('.gguf'):
                return os.path.join(abspath, fn)
    except Exception:
        return None
    return None


def _pick_stage_gguf(dir_path: Optional[str], stage: str) -> Optional[str]:
    p = _first_gguf_in(dir_path)
    if not dir_path:
        return p
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        stage_lc = stage.lower().strip()
        cands = [fn for fn in os.listdir(abspath) if fn.lower().endswith('.gguf')]
        for fn in cands:
            name = fn.lower()
            if stage_lc == 'high' and ('high' in name or 'highnoise' in name):
                return os.path.join(abspath, fn)
            if stage_lc == 'low' and ('low' in name or 'lownoise' in name):
                return os.path.join(abspath, fn)
    except Exception:
        pass
    return p


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


def _resolve_patch_weights(state: Mapping[str, Any]) -> Tuple[Any, Any]:
    w = state.get('patch_embedding.weight')
    b = state.get('patch_embedding.bias')
    if w is None:
        raise RuntimeError("GGUF missing 'patch_embedding.weight'")
    return w, b


def _infer_patch_geometry(
    dit: 'WanDiTGGUF',
    *,
    T: int,
    H_lat: int,
    W_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PatchGeometry:
    w, b = _resolve_patch_weights(dit.state)
    # Probe with zeros to avoid guessing shapes; cheap and reliable
    w_shape = getattr(w, 'shape', [None, None, None, None, None])
    Cin = int(w_shape[1]) if len(w_shape) >= 2 and w_shape[1] is not None else None
    if Cin is None:
        raise RuntimeError("failed to read patch_embedding weight shape")
    vid = torch.zeros(1, Cin, T, H_lat, W_lat, device=device, dtype=dtype)
    tokens, grid = _patch_embed3d(vid, w, b)
    L, Cout = int(tokens.shape[1]), int(tokens.shape[2])
    patch_kernel = (
        int(w_shape[2] or 1),
        int(w_shape[3] or 1),
        int(w_shape[4] or 1),
    )
    return PatchGeometry(
        grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        token_count=L,
        token_dim=Cout,
        latent_channels=Cin,
        patch_kernel=patch_kernel,
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
            if sigmas is not None and len(sigmas) == n:
                s = float(sigmas[i]); s_min = float(sigmas[-1]); s_max = float(sigmas[0])
                t = max(0.0, min(1.0, (s - s_min) / (s_max - s_min))) if (s_max - s_min) > 0 else 0.0
            else:
                t = 1.0 - (float(i) / float(max(1, n - 1)))
            vals.append(t)
        log.info("[wan22.gguf] t-map(%s): t0=%.4f tmid=%.4f tend=%.4f (sigmas=%s)", label, vals[0], vals[1], vals[2], bool(sigmas is not None and len(sigmas)==n))
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
def run_txt2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> List[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    _set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
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
    geom_hi = _infer_patch_geometry(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
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
        dit=hi_dit,
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
        _cuda_empty_cache(log, label='after-high')

    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    geom_lo = _infer_patch_geometry(lo_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
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
        dit=lo_dit,
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

    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)

    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
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
    geom_hi = _infer_patch_geometry(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, 'flow_shift', None) if cfg.high else None

    latents_hi = yield from _sample_stage_latents_generator(
        dit=hi_dit,
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

    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    geom_lo = _infer_patch_geometry(lo_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    seed_latents = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, 'flow_shift', None) if cfg.low else None

    latents_lo = yield from _sample_stage_latents_generator(
        dit=lo_dit,
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


def run_img2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> List[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    _set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))

    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
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

    geom_hi = _infer_patch_geometry(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    seed_hi = _prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)

    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    flow_shift_hi = getattr(cfg.high, 'flow_shift', None) if cfg.high else None

    latents_hi = _sample_stage_latents(
        dit=hi_dit,
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
        _cuda_empty_cache(logger=log, label='after-high')

    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    geom_lo = _infer_patch_geometry(lo_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    seed_lo = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    flow_shift_lo = getattr(cfg.low, 'flow_shift', None) if cfg.low else None

    latents_lo = _sample_stage_latents(
        dit=lo_dit,
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

    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)

    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
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

    geom_hi = _infer_patch_geometry(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    seed_hi = _prepare_stage_seed_latents(lat0.to(device=dev, dtype=dt), geom_hi, logger=log)

    latents_hi = yield from _sample_stage_latents_generator(
        dit=hi_dit,
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

    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    geom_lo = _infer_patch_geometry(lo_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    seed_lo = _prepare_stage_seed_latents(latents_hi, geom_lo, logger=log)

    latents_lo = yield from _sample_stage_latents_generator(
        dit=lo_dit,
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
    'ModelSpec', 'BlockSpec', 'CrossAttnWeights', 'derive_spec_from_state',
    'WanDiTGGUF', 'run_txt2vid', 'run_img2vid',
]
def _get_logger(logger: Any):
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
