"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2img staged pipeline runner facade.
Re-exports `Txt2ImgPipelineRunner`, the staged runner used by the txt2img use case to prepare prompts, sample base/hires passes, and optionally run refiner stages.

Symbols (top-level; keep in sync; no ghosts):
- `Txt2ImgPipelineRunner` (class): Staged txt2img runner (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .runner import Txt2ImgPipelineRunner

__all__ = ["Txt2ImgPipelineRunner"]
