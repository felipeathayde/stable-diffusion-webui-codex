"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Built-in detector registrations for the model registry.
Imports all detector modules so they can self-register into the shared `REGISTRY` at import time, and re-exports detector classes for tests/diagnostics.

Symbols (top-level; keep in sync; no ghosts):
- `StableDiffusionV1Detector` (class): Detector for SD1.x/SD1.5 UNet checkpoints.
- `StableDiffusionXLDetector` (class): Detector for SDXL base checkpoints.
- `StableDiffusionXLRefinerDetector` (class): Detector for SDXL refiner checkpoints.
- `StableDiffusion3Detector` (class): Detector for SD3/SD3.5 checkpoints.
- `FluxDetector` (class): Detector for Flux dev checkpoints.
- `FluxSchnellDetector` (class): Detector for Flux schnell checkpoints.
- `AuraFlowDetector` (class): Detector for AuraFlow checkpoints.
- `ChromaDetector` (class): Detector for Chroma checkpoints.
- `StableCascadeStageBDetector` (class): Detector for Stable Cascade stage B checkpoints.
- `StableCascadeStageCDetector` (class): Detector for Stable Cascade stage C checkpoints.
- `Wan22Detector` (class): Detector for Wan2.2 checkpoints.
- `ZImageDetector` (class): Detector for Z-Image checkpoints.
- `QwenImageDetector` (class): Detector for Qwen Image checkpoints.
"""

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
from .qwen_image import QwenImageDetector

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
    "QwenImageDetector",
]
