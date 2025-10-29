"""Detector registrations for the model registry."""

from __future__ import annotations

from .sd_v1 import StableDiffusionV1Detector
from .sd3 import StableDiffusion3Detector
from .flux import FluxDetector, FluxSchnellDetector
from .aura import AuraFlowDetector
from .stable_cascade import StableCascadeStageBDetector, StableCascadeStageCDetector

__all__ = [
    "StableDiffusionV1Detector",
    "StableDiffusion3Detector",
    "FluxDetector",
    "FluxSchnellDetector",
    "AuraFlowDetector",
    "StableCascadeStageBDetector",
    "StableCascadeStageCDetector",
]
