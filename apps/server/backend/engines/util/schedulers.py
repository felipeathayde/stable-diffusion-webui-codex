from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ApplyOutcome:
    sampler_in: str
    scheduler_in: str
    sampler_effective: str
    scheduler_effective: str
    warnings: List[str]


def apply_sampler_scheduler(pipe, sampler: str, scheduler: str) -> ApplyOutcome:
    """Best-effort mapping of sampler/scheduler strings to Diffusers pipeline.

    Notes
    - WAN engines allow: Euler a, Euler, DDIM, DPM++ 2M, DPM++ 2M SDE, PLMS.
    - We set relevant flags on the scheduler instance when applicable.
    - Returns warnings when a request is adjusted.
    """
    wanted_sampler = (sampler or "Automatic").strip()
    wanted_scheduler = (scheduler or "Automatic").strip()
    eff_sampler = wanted_sampler
    eff_scheduler = wanted_scheduler
    warnings: List[str] = []

    try:
        from diffusers import (
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            LMSDiscreteScheduler,
            PNDMScheduler,
        )

        allowed = {
            "euler": EulerDiscreteScheduler,
            "euler a": EulerAncestralDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm++ 2m": DPMSolverMultistepScheduler,
            "dpm++ 2m sde": DPMSolverMultistepScheduler,
            "plms": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
        }

        key = wanted_sampler.lower()
        if key in ("auto", "automatic", ""):
            # Keep existing scheduler; annotate only
            sched = getattr(pipe, "scheduler", None)
            eff_sampler = "Automatic"
            eff_scheduler = type(sched).__name__ if sched is not None else "<none>"
            return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)

        target_cls = allowed.get(key)
        if target_cls is None:
            warnings.append(f"Unsupported sampler '{wanted_sampler}', keeping pipeline scheduler")
            sched = getattr(pipe, "scheduler", None)
            eff_sampler = type(sched).__name__ if sched is not None else "<none>"
            eff_scheduler = eff_sampler
            return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)

        # Rebuild scheduler from config to preserve sigmas/timesteps defaults
        conf = getattr(pipe, "scheduler", None)
        conf = getattr(conf, "config", None)
        if conf is not None:
            pipe.scheduler = target_cls.from_config(conf)
        else:
            pipe.scheduler = target_cls()

        # Heuristics for options
        if isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            # DPM++ family defaults
            try:
                pipe.scheduler.config.setdefault("use_karras_sigmas", True)
            except Exception:
                pass
        if isinstance(pipe.scheduler, (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)):
            try:
                # trailing spacing is more compatible with SD style
                pipe.scheduler.config.setdefault("timestep_spacing", "trailing")
            except Exception:
                pass

        eff_sampler = wanted_sampler
        eff_scheduler = type(pipe.scheduler).__name__
        return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)
    except Exception as exc:
        warnings.append(f"Failed to apply sampler/scheduler: {exc}")
        sched = getattr(pipe, "scheduler", None)
        eff = type(sched).__name__ if sched is not None else "<none>"
        return ApplyOutcome(wanted_sampler, wanted_scheduler, eff, eff, warnings)


__all__ = ["apply_sampler_scheduler", "ApplyOutcome"]

