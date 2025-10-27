from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


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
    DDIM = "ddim"
    DPM2M = "dpm++ 2m"
    DPM2M_SDE = "dpm++ 2m sde"
    PLMS = "plms"
    PNDM = "pndm"

    @staticmethod
    def from_string(name: str) -> "SamplerKind":
        key = (name or "automatic").strip().lower()
        for member in SamplerKind:
            if key == member.value:
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
    )

    allowed = {
        SamplerKind.EULER: EulerDiscreteScheduler,
        SamplerKind.EULER_A: EulerAncestralDiscreteScheduler,
        SamplerKind.DDIM: DDIMScheduler,
        SamplerKind.DPM2M: DPMSolverMultistepScheduler,
        SamplerKind.DPM2M_SDE: DPMSolverMultistepScheduler,
        SamplerKind.PLMS: LMSDiscreteScheduler,
        SamplerKind.PNDM: PNDMScheduler,
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
