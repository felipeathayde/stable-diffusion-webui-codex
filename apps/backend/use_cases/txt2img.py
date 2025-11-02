"""Txt2Img entry point using the staged pipeline runner."""

from __future__ import annotations

import logging
from typing import Sequence

from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.pipeline_debug import pipeline_trace
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
