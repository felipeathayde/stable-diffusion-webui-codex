"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Diffusion engine facade for legacy/compatibility import paths.
Re-exports the primary diffusion engines (SD/SDXL/Flux/Chroma) under a single namespace for callers that haven't migrated to the engine registry.

Symbols (top-level; keep in sync; no ghosts):
- `StableDiffusion` (class): SD 1.5 engine (re-export).
- `StableDiffusion2` (class): SD 2.x engine (re-export).
- `StableDiffusion3` (class): SD 3.5 engine (re-export).
- `StableDiffusionXL` (class): SDXL base engine (re-export).
- `StableDiffusionXLRefiner` (class): SDXL refiner engine (re-export).
- `Flux` (class): Flux engine (re-export).
- `Chroma` (class): Chroma engine (re-export).
- `__all__` (constant): Explicit export list for the facade.
"""

from apps.backend.engines.sd.sd15 import StableDiffusion
from apps.backend.engines.sd.sd20 import StableDiffusion2
from apps.backend.engines.sd.sd35 import StableDiffusion3
from apps.backend.engines.sd.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
from apps.backend.engines.flux.flux import Flux
from apps.backend.engines.chroma.chroma import Chroma

__all__ = [
    "Chroma",
    "Flux",
    "StableDiffusion",
    "StableDiffusion2",
    "StableDiffusion3",
    "StableDiffusionXL",
    "StableDiffusionXLRefiner",
]
