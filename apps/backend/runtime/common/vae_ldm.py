"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Family-agnostic native LDM VAE runtime lane for 2D image models.
Exposes the canonical `AutoencoderKL_LDM` class that consumes LDM keyspace weights (`encoder.down.*`, `decoder.up.*`, `mid.attn_1.*`) without key remap.

Symbols (top-level; keep in sync; no ghosts):
- `AutoencoderKL_LDM` (class): Native 2D LDM AutoencoderKL class shared by Flux/SDXL/ZImage/WAN lanes.
- `sanitize_ldm_vae_config` (function): Removes unsupported diffusers-only keys before constructing `AutoencoderKL_LDM`.
- `is_ldm_native_vae_instance` (function): Returns True when a model instance belongs to the native LDM VAE lane.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.runtime.families.wan22.vae import AutoencoderKL_LDM


def sanitize_ldm_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    # LDM native lane is always mid-attn capable; this diffusers flag is redundant
    # and unsupported by the native constructor.
    cleaned.pop("mid_block_add_attention", None)
    if "latent_channels" not in cleaned and "z_dim" in cleaned:
        try:
            cleaned["latent_channels"] = int(cleaned["z_dim"])
        except (TypeError, ValueError):
            pass
    if "block_out_channels" not in cleaned and "base_dim" in cleaned and "dim_mult" in cleaned:
        try:
            base_dim = int(cleaned["base_dim"])
            multipliers = tuple(int(value) for value in cleaned["dim_mult"])
            if base_dim > 0 and multipliers:
                cleaned["block_out_channels"] = tuple(base_dim * value for value in multipliers)
        except (TypeError, ValueError):
            pass
    if "layers_per_block" not in cleaned and "num_res_blocks" in cleaned:
        try:
            layers = int(cleaned["num_res_blocks"])
            if layers > 0:
                cleaned["layers_per_block"] = layers
        except (TypeError, ValueError):
            pass
    return cleaned


def is_ldm_native_vae_instance(model: object) -> bool:
    return isinstance(model, AutoencoderKL_LDM)


__all__ = ["AutoencoderKL_LDM", "is_ldm_native_vae_instance", "sanitize_ldm_vae_config"]
