"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler and scheduler type definitions.
Defines the canonical `SamplerKind` enum and an `ApplyOutcome` result container, including alias resolution for sampler strings.

Symbols (top-level; keep in sync; no ghosts):
- `ApplyOutcome` (dataclass): Result of applying sampler/scheduler selection to a pipeline/request.
- `SamplerKind` (enum): Canonical sampler identifiers with alias resolution via `SAMPLER_ALIAS_TO_CANONICAL`.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from apps.backend.runtime.sampling.catalog import SAMPLER_ALIAS_TO_CANONICAL


@dataclass
class ApplyOutcome:
    """Result of applying sampler/scheduler to a pipeline."""
    sampler_in: str
    scheduler_in: str
    sampler_effective: str
    scheduler_effective: str
    warnings: List[str]


class SamplerKind(str, Enum):
    """Canonical sampler identifiers."""
    AUTOMATIC = "automatic"
    EULER = "euler"
    EULER_A = "euler a"
    EULER_CFG_PP = "euler cfg++"
    EULER_A_CFG_PP = "euler a cfg++"
    HEUN = "heun"
    HEUNPP2 = "heunpp2"
    LMS = "lms"
    DDIM = "ddim"
    DDIM_CFG_PP = "ddim cfg++"
    DPM2M = "dpm++ 2m"
    DPM2M_CFG_PP = "dpm++ 2m cfg++"
    DPM2M_SDE = "dpm++ 2m sde"
    DPM2M_SDE_HEUN = "dpm++ 2m sde heun"
    DPM2M_SDE_GPU = "dpm++ 2m sde gpu"
    DPM2M_SDE_HEUN_GPU = "dpm++ 2m sde heun gpu"
    DPM_SDE = "dpm++ sde"
    PLMS = "plms"
    PNDM = "pndm"
    UNI_PC = "uni-pc"
    UNI_PC_BH2 = "uni-pc bh2"
    DPM2S_ANCESTRAL = "dpm++ 2s ancestral"
    DPM2S_ANCESTRAL_CFG_PP = "dpm++ 2s ancestral cfg++"
    DPM3M_SDE = "dpm++ 3m sde"
    DPM3M_SDE_GPU = "dpm++ 3m sde gpu"
    DPM2 = "dpm 2"
    DPM2_ANCESTRAL = "dpm 2 ancestral"
    DPM_FAST = "dpm fast"
    DPM_ADAPTIVE = "dpm adaptive"
    DDPM = "ddpm"
    LCM = "lcm"
    IPNDM = "ipndm"
    IPNDM_V = "ipndm v"
    DEIS = "deis"
    RES_MULTISTEP = "res multistep"
    RES_MULTISTEP_CFG_PP = "res multistep cfg++"
    RES_MULTISTEP_ANCESTRAL = "res multistep ancestral"
    RES_MULTISTEP_ANCESTRAL_CFG_PP = "res multistep ancestral cfg++"
    GRADIENT_ESTIMATION = "gradient estimation"
    GRADIENT_ESTIMATION_CFG_PP = "gradient estimation cfg++"
    ER_SDE = "er sde"
    SEEDS_2 = "seeds 2"
    SEEDS_3 = "seeds 3"
    SA_SOLVER = "sa-solver"
    SA_SOLVER_PECE = "sa-solver pece"
    RESTART = "restart"

    @staticmethod
    def from_string(name: str) -> "SamplerKind":
        """Parse sampler name to enum, with alias resolution."""
        key = (name or "automatic").strip().lower()
        canonical = SAMPLER_ALIAS_TO_CANONICAL.get(key, key)
        for member in SamplerKind:
            if canonical == member.value:
                return member
        raise ValueError(f"Unsupported sampler '{name}'. Valid: {[m.value for m in SamplerKind]}")


__all__ = ["SamplerKind", "ApplyOutcome"]
