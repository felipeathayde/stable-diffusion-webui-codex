"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2img entry point using the staged pipeline runner.
Delegates execution to `Txt2ImgPipelineRunner` for processing/conditioning inputs and integrates pipeline tracing for debugging.

Symbols (top-level; keep in sync; no ghosts):
- `_logger` (constant): Module logger for the txt2img use case.
- `_RUNNER` (constant): Singleton `Txt2ImgPipelineRunner` instance.
- `generate_txt2img` (function): Runs the txt2img pipeline runner for the provided processing context and prompts.
"""

from __future__ import annotations

import logging
from typing import Sequence

from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.diagnostics.pipeline_debug import pipeline_trace
from .txt2img_pipeline import Txt2ImgPipelineRunner


_logger = logging.getLogger("backend.use_cases.txt2img")
_RUNNER = Txt2ImgPipelineRunner()


@pipeline_trace
def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
):
    if not isinstance(processing, CodexProcessingTxt2Img):
        raise TypeError("generate_txt2img expects CodexProcessingTxt2Img")

    return _RUNNER.run(
        processing=processing,
        conditioning_data=conditioning,
        unconditional_data=unconditional_conditioning,
        seeds=seeds,
        subseeds=subseeds,
        subseed_strength=subseed_strength,
        prompts=prompts,
    )
