from __future__ import annotations

from typing import Any


class CodexSampler:
    """Native sampler façade for engines.

    This class will call into runtime.sampling once the unified driver is wired.
    For now, any attempt to sample raises an explicit NotImplementedError to
    avoid falling back to legacy behavior.
    """

    def __init__(self, sd_model: Any) -> None:
        self.sd_model = sd_model

    def sample(self, processing, noise, cond, uncond, image_conditioning=None):  # noqa: D401
        raise NotImplementedError("Codex sampler is not implemented yet for txt2img")


__all__ = ["CodexSampler"]

