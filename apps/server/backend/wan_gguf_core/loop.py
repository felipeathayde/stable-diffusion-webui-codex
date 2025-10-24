from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoopConfig:
    steps: int
    guidance_scale: Optional[float] = None
    dtype: str = "bf16"
    device: str = "cuda"


class DiffusionLoop:
    def __init__(self, stepper, *, logger=None) -> None:
        self.stepper = stepper
        self._logger = logger

    def run(self, unet, x, cond_pos, cond_neg=None, *, cfg: LoopConfig):
        """Run Euler(Simple) steps with a UNet-like forward.

        This is a skeleton. It assumes `unet.forward(x, t, cond, dtype=cfg.dtype)` and returns a tensor like x.
        """
        import torch
        self.stepper.set_timesteps(cfg.steps)
        timesteps = self.stepper.timesteps
        sample = x
        for i, t in enumerate(timesteps):
            if self._logger:
                try:
                    total = int(getattr(timesteps, 'numel', lambda: len(timesteps))())
                except Exception:
                    total = len(timesteps)
                self._logger.info("[wan-gguf-core] step %d/%d", i + 1, total)
            model_in = self.stepper.scale_model_input(sample, t)
            # UNet predicts noise epsilon. If cond_neg present, apply CFG.
            if cond_neg is not None and cfg.guidance_scale and cfg.guidance_scale > 1.0:
                eps_p = unet.forward(model_in, t, cond_pos, guidance_scale=None, dtype=cfg.dtype)
                eps_n = unet.forward(model_in, t, cond_neg, guidance_scale=None, dtype=cfg.dtype)
                eps = eps_n + cfg.guidance_scale * (eps_p - eps_n)
            else:
                eps = unet.forward(model_in, t, cond_pos, guidance_scale=None, dtype=cfg.dtype)
            sample = self.stepper.step(eps, t, sample)
        return sample
