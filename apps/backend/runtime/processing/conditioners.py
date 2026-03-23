"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preprocessing helpers for building conditioning tensors and decoding latents.
Implements request-driven txt2img/img2img conditioning helpers used before sampling, including deterministic img2img
VAE-posterior encode seed plumbing, while masked img2img semantics now live in latent-mask pipeline stages instead of
model-class inpaint conditioning.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_torch_manual_seed` (function): Validates torch manual-seed bounds and remaps negative seeds to their runtime `uint64` representation.
- `resolve_processing_encode_seed` (function): Resolves the canonical img2img VAE-posterior seed from a processing object.
- `decode_latent_batch` (function): Decode a batch of latents via `sd_model.decode_first_stage`, with optional smart-offload pre-VAE guard.
- `encode_image_batch` (function): Encode a BCHW image tensor via `sd_model.encode_first_stage`, with optional smart-offload pre-VAE guard
  and deterministic posterior seed.
- `txt2img_conditioning` (function): Build the zero image-conditioning tensor used by request-driven txt2img paths.
- `img2img_conditioning` (function): Build the zero image-conditioning tensor used by request-driven img2img paths.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_vae_residency

_MIN_TORCH_SEED = -(1 << 63)
_MAX_TORCH_SEED = (1 << 64) - 1


def normalize_torch_manual_seed(value: int) -> int:
    resolved = int(value)
    if resolved < _MIN_TORCH_SEED or resolved > _MAX_TORCH_SEED:
        raise ValueError(
            f"Seed {resolved} is outside torch.Generator.manual_seed range "
            f"[{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}]."
        )
    if resolved < 0:
        return (1 << 64) + resolved
    return resolved


def resolve_processing_encode_seed(processing: Any) -> int | None:
    raw_seeds = list(getattr(processing, "seeds", []) or [])
    candidate = raw_seeds[0] if raw_seeds else getattr(processing, "seed", None)
    if candidate is None:
        return None
    return normalize_torch_manual_seed(int(candidate))


def decode_latent_batch(
    sd_model: Any,
    latents: torch.Tensor,
    *,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    decoded = sd_model.decode_first_stage(latents)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


def encode_image_batch(
    sd_model: Any,
    images: torch.Tensor,
    *,
    encode_seed: int | None = None,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    latents = sd_model.encode_first_stage(images, encode_seed=encode_seed)
    if target_device is not None:
        latents = latents.to(target_device)
    return latents


def txt2img_conditioning(sd_model: Any, latents: torch.Tensor, width: int, height: int) -> torch.Tensor:
    del sd_model, width, height
    return latents.new_zeros(latents.shape[0], 5, 1, 1)


def img2img_conditioning(
    sd_model: Any,
    source_image: torch.Tensor,
    latent_image: torch.Tensor,
    *,
    image_mask: Optional[Any] = None,
    round_mask: bool = True,
    precomputed_conditioning_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del sd_model, source_image, image_mask, round_mask, precomputed_conditioning_latent
    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
