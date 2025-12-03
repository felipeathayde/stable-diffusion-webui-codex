from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from apps.backend.runtime.sampling.catalog import SAMPLER_ALIAS_TO_CANONICAL


@dataclass
class ApplyOutcome:
    sampler_in: str
    scheduler_in: str
    sampler_effective: str
    scheduler_effective: str
    warnings: List[str]


class SamplerKind(str, Enum):
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
        key = (name or "automatic").strip().lower()
        canonical = SAMPLER_ALIAS_TO_CANONICAL.get(key, key)
        for member in SamplerKind:
            if canonical == member.value:
                return member
        raise ValueError(f"Unsupported sampler '{name}'. Valid: {[m.value for m in SamplerKind]}")


def apply_sampler_scheduler(pipe, sampler: Union[str, SamplerKind], scheduler: Optional[str]) -> ApplyOutcome:
    """Strict mapping of sampler/scheduler to Diffusers pipeline.

    - Allowed: Euler a, Euler, DDIM, DPM++ 2M, DPM++ 2M SDE, PLMS, PNDM.
    - On invalid or failed application, raises with the root cause; no fallbacks.
    """
    wanted_sampler = sampler.value if isinstance(sampler, SamplerKind) else (sampler or "Automatic").strip()
    wanted_scheduler = (scheduler or "Automatic").strip() if scheduler is not None else "Automatic"
    eff_sampler = wanted_sampler
    eff_scheduler = wanted_scheduler
    warnings: List[str] = []

    from diffusers import (
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )

    allowed = {
        SamplerKind.EULER: EulerDiscreteScheduler,
        SamplerKind.EULER_A: EulerAncestralDiscreteScheduler,
        SamplerKind.DDIM: DDIMScheduler,
        SamplerKind.DPM2M: DPMSolverMultistepScheduler,
        SamplerKind.DPM2M_SDE: DPMSolverMultistepScheduler,
        SamplerKind.PLMS: LMSDiscreteScheduler,
        SamplerKind.PNDM: PNDMScheduler,
        SamplerKind.UNI_PC: UniPCMultistepScheduler,
    }

    kind = SamplerKind.from_string(wanted_sampler)
    if kind is SamplerKind.AUTOMATIC:
        sched = getattr(pipe, "scheduler", None)
        if sched is None:
            raise RuntimeError("Pipeline has no scheduler to keep for 'Automatic'")
        eff_sampler = "Automatic"
        eff_scheduler = type(sched).__name__
        return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)

    target_cls = allowed.get(kind)
    if target_cls is None:
        raise ValueError(f"Unsupported sampler '{wanted_sampler}'")

    # Rebuild scheduler from config to preserve sigmas/timesteps defaults
    conf = getattr(pipe, "scheduler", None)
    conf = getattr(conf, "config", None)
    if conf is not None:
        pipe.scheduler = target_cls.from_config(conf)
    else:
        pipe.scheduler = target_cls()

    # Heuristics for options
    if isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
        pipe.scheduler.config.setdefault("use_karras_sigmas", True)
    if isinstance(pipe.scheduler, (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)):
        # trailing spacing is more compatible with SD style
        pipe.scheduler.config.setdefault("timestep_spacing", "trailing")

    eff_sampler = wanted_sampler
    eff_scheduler = type(pipe.scheduler).__name__
    return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)


__all__ = ["apply_sampler_scheduler", "ApplyOutcome", "SamplerKind"]
