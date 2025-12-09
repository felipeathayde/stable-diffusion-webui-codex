"""Detector registrations for the model registry."""

from __future__ import annotations

from .sd_v1 import StableDiffusionV1Detector
from .sdxl import StableDiffusionXLDetector, StableDiffusionXLRefinerDetector
from .sd3 import StableDiffusion3Detector
from .flux import FluxDetector, FluxSchnellDetector
from .aura import AuraFlowDetector
from .chroma import ChromaDetector
from .stable_cascade import StableCascadeStageBDetector, StableCascadeStageCDetector
from .wan22 import Wan22Detector
from .zimage import ZImageDetector

__all__ = [
    "StableDiffusionV1Detector",
    "StableDiffusionXLDetector",
    "StableDiffusionXLRefinerDetector",
    "StableDiffusion3Detector",
    "FluxDetector",
    "FluxSchnellDetector",
    "AuraFlowDetector",
    "ChromaDetector",
    "StableCascadeStageBDetector",
    "StableCascadeStageCDetector",
    "Wan22Detector",
    "ZImageDetector",
]

