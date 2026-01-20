"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL ControlNet architecture facade.
Currently reuses the SD implementations until SDXL-specific variants are ported.

Symbols (top-level; keep in sync; no ghosts):
- `ControlNet` (class): Re-export of the SD ControlNet implementation.
- `ControlNetLite` (class): Re-export of the SD ControlNet-Lite placeholder.
- `ControlLora` (class): Re-export of the SD ControlNet LoRA implementation.
- `T2IAdapter` (class): Re-export of the SD T2I-Adapter implementation.
- `__all__` (constant): Explicit export list.
"""

from ..sd.control import ControlNet  # reuse SD implementation until specialization lands
from ..sd.control_lite import ControlNetLite
from ..sd.lora import ControlLora
from ..sd.t2i_adapter import T2IAdapter

__all__ = ["ControlNet", "ControlNetLite", "ControlLora", "T2IAdapter"]
