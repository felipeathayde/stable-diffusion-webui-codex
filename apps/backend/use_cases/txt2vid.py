from __future__ import annotations

import time
from typing import Any, Iterator

from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.workflows import (
    apply_engine_loras,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
)


def _run_pipeline(pipe: Any, plan: VideoPlan, request: Txt2VidRequest) -> list[Any]:
    output = pipe(
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        num_frames=plan.frames,
        num_inference_steps=plan.steps,
        height=plan.height,
        width=plan.width,
        guidance_scale=plan.guidance_scale,
    )
    if hasattr(output, "frames"):
        return list(output.frames[0])
    if hasattr(output, "images"):
        return list(output.images)
    raise RuntimeError("txt2vid pipeline returned no frames")


def run_txt2vid(
    *,
    engine,
    comp,
    request: Txt2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    plan = build_video_plan(request)
    start = time.perf_counter()

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid")

    pipe = getattr(comp, "pipeline", None)
    if pipe is None:
        raise RuntimeError("txt2vid requires a Diffusers pipeline; none found in components")

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    if wan_hi_opts and wan_hi_opts.lora_path and hasattr(pipe, "load_lora_weights"):
        if logger:
            logger.info("[wan] loading stage LoRA: %s", wan_hi_opts.lora_path)
        pipe.load_lora_weights(wan_hi_opts.lora_path)  # type: ignore[attr-defined]

    apply_engine_loras(engine, logger)

    sampler_outcome = configure_sampler(pipe, plan, logger)

    yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
    frames = _run_pipeline(pipe, plan, request)

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None))

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        task="txt2vid",
        extra=plan.extras,
        video_meta=video_meta,
    )

    payload = {
        "images": result.frames,
        "info": engine._to_json(result.metadata),  # type: ignore[attr-defined]
    }
    yield ResultEvent(payload=payload)
