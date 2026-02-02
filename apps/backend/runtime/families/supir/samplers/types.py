"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR sampler type definitions.
Defines the stable identifiers used by the SUPIR enhance runtime, independent of UI labels.

Symbols (top-level; keep in sync; no ghosts):
- `SupirSamplerId` (enum): Canonical sampler IDs (stable + dev variants).
- `SupirSamplerSpec` (dataclass): Registry record for one sampler option (label + stability + tiling support).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupirSamplerId(str, Enum):
    # Stable
    RESTORE_HEUN_EDM_STABLE = "restore_heun_edm_stable"
    RESTORE_EULER_EDM_STABLE = "restore_euler_edm_stable"
    RESTORE_DPMPP2M_STABLE = "restore_dpmpp_2m_stable"

    # Dev (kept for parity; may be hidden in UI by default)
    RESTORE_HEUN_EDM_DEV = "restore_heun_edm_dev"
    RESTORE_EULER_EDM_DEV = "restore_euler_edm_dev"
    RESTORE_DPMPP2M_DEV = "restore_dpmpp_2m_dev"


@dataclass(frozen=True)
class SupirSamplerSpec:
    sampler_id: SupirSamplerId
    label: str
    stability: str  # 'stable' | 'dev'
    supports_tiling: bool


__all__ = ["SupirSamplerId", "SupirSamplerSpec"]
