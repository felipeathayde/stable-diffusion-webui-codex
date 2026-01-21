"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 GGUF runtime config types and small parsing helpers.
Defines the dataclasses used by the WAN22 GGUF runners (RunConfig/StageConfig) and small env-driven knobs.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_FLOW_MULTIPLIER` (constant): Multiplier applied to shifted sigma to build the model timestep input.
- `StageConfig` (dataclass): Stage-level configuration (stage model selection + sampler/scheduler/steps/cfg/flow_shift + optional LoRA).
- `RunConfig` (dataclass): Full run configuration (geometry, prompts, devices/dtypes, assets, and both stages).
- `_coerce_int` (function): Best-effort coercion of optional values to `int` (returns `None` on failure).
- `_coerce_float` (function): Best-effort coercion of optional values to `float` (returns `None` on failure).
- `as_torch_dtype` (function): Parses dtype strings into torch dtypes (with validation).
- `resolve_device_name` (function): Normalizes device names (`cuda`/`cpu`/etc) into runtime-compatible values.
- `resolve_i2v_order` (function): Resolves the image-to-video conditioning channel order policy.
- `build_wan22_gguf_run_config` (function): Builds a validated GGUF `RunConfig` from a request-like object and its extras mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping, Optional

import torch

from .paths import normalize_win_path

WAN_FLOW_MULTIPLIER = 1000.0


@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    flow_shift: float
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None


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
    sdpa_policy: Optional[str] = None  # 'mem_efficient' | 'flash' | 'math'
    attn_chunk_size: Optional[int] = None  # split attention along sequence if set (>0)
    gguf_cache_policy: Optional[str] = None  # 'none' | 'cpu_lru'
    gguf_cache_limit_mb: Optional[int] = None  # MB limit for cpu_lru cache
    log_mem_interval: Optional[int] = None  # log CUDA mem every N steps if >0
    # Aggressive offload controls
    aggressive_offload: bool = True  # legacy switch; see offload_level
    te_device: Optional[str] = None  # 'cuda' | 'cpu' (None = follow cfg.device)
    te_impl: Optional[str] = None  # 'cuda_fp8' | 'hf' (None = default)
    te_kernel_required: Optional[bool] = None  # if True, error if CUDA kernel unavailable
    # New: coarse-grained offload profile (takes precedence over aggressive_offload if provided)
    # 0 = off (keep resident), 1 = light (offload TE/VAE only), 2 = balanced (also clear between stages), 3 = aggressive (current behavior)
    offload_level: Optional[int] = None


def as_torch_dtype(dtype: str) -> torch.dtype:
    key = str(dtype or "").strip().lower()
    if key in {"fp16", "float16", "f16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return getattr(torch, "bfloat16", torch.float16)
    if key in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r} (expected fp16/bf16/fp32)")


def resolve_device_name(name: str) -> str:
    raw = str(name or "auto").strip()
    s = raw.lower()

    if s in {"cpu"}:
        return "cpu"

    if s in {"auto", ""}:
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("WAN22: CUDA is not available; set device='cpu' explicitly to force CPU.")

    # Accept explicit CUDA device strings (cuda, cuda:0, etc).
    if s == "gpu" or s.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda" if s == "gpu" else s
        raise RuntimeError(f"WAN22: device={raw!r} requested but CUDA is not available; set device='cpu' explicitly.")

    raise ValueError(f"Unsupported device: {raw!r} (expected 'auto', 'cpu', or 'cuda').")


def resolve_i2v_order() -> str:
    """Return channel order for I2V concatenation.

    - 'lat_first': latents(16) then cond extras (mask4+img16).
    - 'lat_last' : cond extras first then latents(16).
    Defaults to 'lat_first'. (Env overrides removed; payload-driven only.)
    """
    return "lat_first"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def build_wan22_gguf_run_config(
    *,
    request: Any,
    device: str,
    dtype: str,
    logger: Any = None,
) -> RunConfig:
    """Build a validated WAN22 GGUF RunConfig from a request-like object.

    Contract: this is a pure mapping layer (no implicit fallbacks, no filesystem guessing).

    Expected `request` attrs (via getattr):
    - prompt / negative_prompt
    - width / height / fps / num_frames / steps / guidance_scale / seed
    - sampler / scheduler
    - init_image (img2vid only)
    - extras: mapping that includes WAN GGUF asset paths and stage overrides
    """
    ex_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any] = dict(ex_raw) if isinstance(ex_raw, Mapping) else {}

    vae_path = str(extras.get("wan_vae_path") or "").strip() or None

    if extras.get("wan_text_encoder_dir"):
        raise ValueError("WAN22: 'wan_text_encoder_dir' is unsupported in sha-only mode; provide 'wan_text_encoder_path' instead.")

    te_path = str(extras.get("wan_text_encoder_path") or "").strip() or None

    meta_dir = None
    if extras.get("wan_metadata_dir"):
        meta_dir = str(extras.get("wan_metadata_dir") or "").strip() or None
    elif extras.get("wan_tokenizer_dir"):
        # Allow providing tokenizer dir; scheduler_config resolution supports parent fallback.
        meta_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    if not te_path:
        raise RuntimeError(
            "WAN22 GGUF requires a text encoder weights file; provide 'wan_text_encoder_path' (resolved from sha selection)."
        )
    if not vae_path:
        raise RuntimeError("WAN22 GGUF requires a VAE weights file; provide 'wan_vae_path' (resolved from sha selection).")
    if not meta_dir:
        raise RuntimeError("WAN22 GGUF requires tokenizer metadata; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.")

    te_path = os.path.expanduser(te_path)
    te_lower = te_path.lower()
    if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
        raise RuntimeError("WAN22 GGUF: 'wan_text_encoder_path' must be a '.safetensors' or '.gguf' file, got: %s" % te_path)
    if not os.path.isfile(te_path):
        raise RuntimeError(f"WAN22 GGUF: text encoder weights not found: {te_path}")

    vae_path = os.path.expanduser(vae_path)
    if not os.path.isfile(vae_path) and not os.path.isdir(vae_path):
        raise RuntimeError(f"WAN22 GGUF: VAE weights not found: {vae_path}")

    wh_raw = extras.get("wan_high") if isinstance(extras.get("wan_high"), dict) else None
    wl_raw = extras.get("wan_low") if isinstance(extras.get("wan_low"), dict) else None

    forbidden = ("lightning", "lora_path")
    for stage_name, stage_cfg in (("wan_high", wh_raw), ("wan_low", wl_raw)):
        if not isinstance(stage_cfg, dict):
            continue
        for key in forbidden:
            if stage_cfg.get(key) not in (None, ""):
                if key == "lora_path":
                    raise RuntimeError(
                        f"WAN22 GGUF: '{stage_name}.lora_path' is not supported (use '{stage_name}.lora_sha')."
                    )
                raise RuntimeError(f"WAN22 GGUF: '{stage_name}.{key}' is not supported (use Diffusers path).")

    default_steps = int(getattr(request, "steps", 12) or 12)
    default_cfg = getattr(request, "guidance_scale", None)

    def _stage_opts(
        raw: dict | None,
        *,
        stage: str,
    ) -> tuple[str, int, Optional[float], Optional[str], Optional[str], Optional[float], Optional[int], Optional[str], Optional[float]]:
        if not isinstance(raw, dict):
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = str(raw.get("model_dir") or "").strip()
        if not model_dir:
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = normalize_win_path(os.path.expanduser(model_dir))
        if not model_dir.lower().endswith(".gguf"):
            raise RuntimeError(f"WAN22 GGUF: {stage} model must be a .gguf file, got: {model_dir}")
        if not os.path.isfile(model_dir):
            raise RuntimeError(f"WAN22 GGUF: {stage} model not found: {model_dir}")

        steps = _coerce_int(raw.get("steps"))
        steps = int(steps) if steps is not None else default_steps

        cfg_scale = _coerce_float(raw.get("cfg_scale")) if raw.get("cfg_scale") is not None else default_cfg
        sampler = str(raw.get("sampler")).strip() if raw.get("sampler") else None
        scheduler = str(raw.get("scheduler")).strip() if raw.get("scheduler") else None
        flow_shift = _coerce_float(raw.get("flow_shift")) if raw.get("flow_shift") is not None else None
        seed = _coerce_int(raw.get("seed")) if raw.get("seed") is not None else None
        lora_sha = str(raw.get("lora_sha") or "").strip().lower() or None
        lora_weight = _coerce_float(raw.get("lora_weight")) if raw.get("lora_weight") is not None else None
        if lora_weight is not None and not lora_sha:
            raise RuntimeError(f"WAN22 GGUF: {stage}.lora_weight requires {stage}.lora_sha.")
        lora_path = None
        if lora_sha:
            if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha must be sha256 (64 lowercase hex).")
            from apps.backend.inventory.cache import resolve_asset_by_sha

            resolved = resolve_asset_by_sha(lora_sha)
            if not resolved:
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha not found in inventory: {lora_sha}")
            lora_path = normalize_win_path(os.path.expanduser(str(resolved)))
            if not lora_path.lower().endswith(".safetensors"):
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha must resolve to a .safetensors file: {lora_sha}")
            if not os.path.isfile(lora_path):
                raise RuntimeError(f"WAN22 GGUF: {stage} LoRA file not found: {lora_path}")
        return model_dir, steps, cfg_scale, sampler, scheduler, flow_shift, seed, lora_path, lora_weight

    hi_dir, hi_steps, hi_cfg, hi_sampler, hi_scheduler, hi_flow_shift, hi_seed, hi_lora_path, hi_lora_weight = _stage_opts(
        wh_raw, stage="wan_high"
    )
    lo_dir, lo_steps, lo_cfg, lo_sampler, lo_scheduler, lo_flow_shift, _lo_seed, lo_lora_path, lo_lora_weight = _stage_opts(
        wl_raw, stage="wan_low"
    )

    if hi_flow_shift is None or lo_flow_shift is None:
        from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir

        vendor_dir = str(meta_dir or "").strip()
        if not vendor_dir:
            raise RuntimeError("WAN22 GGUF requires flow_shift defaults from scheduler_config.json, but wan_metadata_dir is missing.")
        if not os.path.isdir(os.path.join(vendor_dir, "scheduler")):
            parent = os.path.dirname(vendor_dir)
            if parent and os.path.isdir(os.path.join(parent, "scheduler")):
                vendor_dir = parent
        spec = flow_shift_spec_from_repo_dir(vendor_dir)
        default_flow_shift = spec.resolve()
        if hi_flow_shift is None:
            hi_flow_shift = default_flow_shift
        if lo_flow_shift is None:
            lo_flow_shift = default_flow_shift

    seed = getattr(request, "seed", None)
    if hi_seed is not None:
        seed = hi_seed

    sampler_fallback = str(getattr(request, "sampler", "") or "").strip() or "uni-pc"
    scheduler_fallback = str(getattr(request, "scheduler", "") or "").strip() or "simple"

    tokenizer_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    offload_level = _coerce_int(extras.get("gguf_offload_level"))
    if offload_level is not None and offload_level < 0:
        offload_level = None

    if logger is not None:
        try:
            logger.info(
                "[wan22.gguf] assets: metadata=%s te=%s vae=%s",
                os.path.basename(str(meta_dir)) if meta_dir else None,
                os.path.basename(str(te_path)) if te_path else None,
                os.path.basename(str(vae_path)) if vae_path else None,
            )
        except Exception:
            pass

    return RunConfig(
        width=int(getattr(request, "width", 768) or 768),
        height=int(getattr(request, "height", 432) or 432),
        fps=int(getattr(request, "fps", 24) or 24),
        num_frames=int(getattr(request, "num_frames", 16) or 16),
        guidance_scale=getattr(request, "guidance_scale", None),
        dtype=str(dtype or "fp16"),
        device=str(device or "cuda"),
        seed=seed,
        prompt=getattr(request, "prompt", None),
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
        vae_dir=vae_path,
        text_encoder_dir=te_path,
        tokenizer_dir=tokenizer_dir,
        metadata_dir=meta_dir,
        sdpa_policy=(extras.get("gguf_sdpa_policy") if extras.get("gguf_sdpa_policy") is not None else None),
        attn_chunk_size=(int(extras.get("gguf_attn_chunk", 0)) if extras.get("gguf_attn_chunk") not in (None, "", 0) else None),
        gguf_cache_policy=(extras.get("gguf_cache_policy") if extras.get("gguf_cache_policy") is not None else None),
        gguf_cache_limit_mb=(
            int(extras.get("gguf_cache_limit_mb", 0)) if extras.get("gguf_cache_limit_mb") not in (None, "", 0) else None
        ),
        log_mem_interval=(
            int(extras.get("gguf_log_mem_interval", 0)) if extras.get("gguf_log_mem_interval") not in (None, "", 0) else None
        ),
        aggressive_offload=bool(extras.get("gguf_offload", True)),
        offload_level=offload_level,
        te_device=(str(extras.get("gguf_te_device")).lower() if extras.get("gguf_te_device") is not None else None),
        te_impl=(str(extras.get("gguf_te_impl")).lower() if extras.get("gguf_te_impl") is not None else None),
        te_kernel_required=bool(extras.get("gguf_te_kernel_required", False)),
        high=StageConfig(
            model_dir=hi_dir,
            sampler=str(hi_sampler or sampler_fallback),
            scheduler=str(hi_scheduler or scheduler_fallback),
            steps=max(1, int(hi_steps)),
            cfg_scale=hi_cfg,
            flow_shift=float(hi_flow_shift),
            lora_path=hi_lora_path,
            lora_weight=hi_lora_weight,
        ),
        low=StageConfig(
            model_dir=lo_dir,
            sampler=str(lo_sampler or sampler_fallback),
            scheduler=str(lo_scheduler or scheduler_fallback),
            steps=max(1, int(lo_steps)),
            cfg_scale=lo_cfg,
            flow_shift=float(lo_flow_shift),
            lora_path=lo_lora_path,
            lora_weight=lo_lora_weight,
        ),
    )
