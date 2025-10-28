"""Detector registrations for the model registry."""

from __future__ import annotations

from .sd_v1 import StableDiffusionV1Detector  # re-export for typing completeness

__all__ = [
    "StableDiffusionV1Detector",
]
