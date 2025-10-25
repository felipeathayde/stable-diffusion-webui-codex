from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EulerSimpleConfig:
    num_inference_steps: int
    guidance_scale: Optional[float] = None
    timestep_spacing: str = "trailing"  # Simple


class EulerSimpleStepper:
    """Thin wrapper around Diffusers EulerDiscreteScheduler set to Simple spacing.

    Produces timesteps and exposes a step() that mirrors the minimal interface
    we need for native execution. Keeps Diffusers as the source of truth to
    match UI behavior.
    """

    def __init__(self, cfg: EulerSimpleConfig):
        from diffusers import EulerDiscreteScheduler  # type: ignore

        self._cfg = cfg
        self._scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        )
        self._timesteps = None

    def set_timesteps(self, steps: Optional[int] = None) -> None:
        steps = int(steps or self._cfg.num_inference_steps)
        steps = max(1, steps)
        self._scheduler.set_timesteps(steps)
        self._timesteps = self._scheduler.timesteps

    @property
    def timesteps(self):
        if self._timesteps is None:
            self.set_timesteps(self._cfg.num_inference_steps)
        return self._timesteps

    def scale_model_input(self, sample, timestep):
        return self._scheduler.scale_model_input(sample, timestep)

    def step(self, model_output, timestep, sample):
        out = self._scheduler.step(model_output, timestep, sample, return_dict=True)
        return out.prev_sample
