"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 compatibility shim for the shared native LDM 2D VAE lane.
Re-exports the canonical implementation from `apps.backend.runtime.common.vae_ldm`
so legacy imports under `runtime.families.wan22.vae` remain valid while ownership
stays in the shared runtime-common module.

Symbols (top-level; keep in sync; no ghosts):
- `nonlinearity` (function): Activation helper re-exported from shared native LDM VAE module.
- `Normalize` (function): GroupNorm helper re-exported from shared native LDM VAE module.
- `DiagonalGaussianDistribution` (class): Distribution wrapper re-exported from shared native LDM VAE module.
- `Upsample` (class): Decoder upsample block re-exported from shared native LDM VAE module.
- `Downsample` (class): Encoder downsample block re-exported from shared native LDM VAE module.
- `ResnetBlock` (class): Residual block re-exported from shared native LDM VAE module.
- `AttnBlock` (class): Attention block re-exported from shared native LDM VAE module.
- `Encoder` (class): Encoder module re-exported from shared native LDM VAE module.
- `Decoder` (class): Decoder module re-exported from shared native LDM VAE module.
- `AutoencoderKL_LDM` (class): Canonical native LDM 2D VAE class re-export.
- `sanitize_ldm_vae_config` (function): Shared config sanitizer re-export.
- `is_ldm_native_vae_instance` (function): Shared instance predicate re-export.
- `__all__` (constant): Explicit export list.
"""

from apps.backend.runtime.common.vae_ldm import (
    AttnBlock,
    AutoencoderKL_LDM,
    Decoder,
    DiagonalGaussianDistribution,
    Downsample,
    Encoder,
    Normalize,
    ResnetBlock,
    Upsample,
    is_ldm_native_vae_instance,
    nonlinearity,
    sanitize_ldm_vae_config,
)

__all__ = [
    "AttnBlock",
    "AutoencoderKL_LDM",
    "Decoder",
    "DiagonalGaussianDistribution",
    "Downsample",
    "Encoder",
    "Normalize",
    "ResnetBlock",
    "Upsample",
    "is_ldm_native_vae_instance",
    "nonlinearity",
    "sanitize_ldm_vae_config",
]
