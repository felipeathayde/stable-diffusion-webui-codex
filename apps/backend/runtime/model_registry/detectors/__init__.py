"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Built-in detector registrations for the model registry.
Imports all detector modules so they can self-register into the shared `REGISTRY` at import time.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Empty export list; import detector classes from their defining modules.
"""

from __future__ import annotations

from . import aura as _aura
from . import chroma as _chroma
from . import flux as _flux
from . import qwen_image as _qwen_image
from . import sd3 as _sd3
from . import sd_v1 as _sd_v1
from . import sdxl as _sdxl
from . import stable_cascade as _stable_cascade
from . import wan22 as _wan22
from . import zimage as _zimage

del _aura, _chroma, _flux, _qwen_image, _sd3, _sd_v1, _sdxl, _stable_cascade, _wan22, _zimage

__all__ = []
