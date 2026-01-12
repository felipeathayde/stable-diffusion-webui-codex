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
- `StageConfig` (dataclass): Stage-level configuration (steps/cfg/seed/sampler/scheduler + stage model selection).
- `RunConfig` (dataclass): Full run configuration (geometry, prompts, devices/dtypes, assets, and both stages).
- `as_torch_dtype` (function): Parses dtype strings into torch dtypes (with validation).
- `resolve_device_name` (function): Normalizes device names (`cuda`/`cpu`/etc) into runtime-compatible values.
- `resolve_i2v_order` (function): Resolves the image-to-video conditioning channel order policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

WAN_FLOW_MULTIPLIER = 1000.0


@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    flow_shift: float


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
