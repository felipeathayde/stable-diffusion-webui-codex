from __future__ import annotations

import time
from typing import Any, Iterator

from apps.backend.core.params.video import VideoInterpolationOptions
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.workflows import (
    apply_engine_loras,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
)
from apps.backend.video.interpolation import maybe_interpolate


def _run_stage(
    pipe: Any,
    plan: VideoPlan,
    *,
    prompt: str,
    negative_prompt: str | None,
    init_image: Any | None,
) -> list[Any]:
    if pipe is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")
    output = pipe(
        image=init_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=plan.frames,
        num_inference_steps=plan.steps,
        height=plan.height,
        width=plan.width,
        guidance_scale=plan.guidance_scale,
    )
    if hasattr(output, "frames"):
        return list(output.frames[0])
    raise RuntimeError("img2vid pipeline returned no frames")


def run_img2vid(
    *,
    engine,
    comp,
    request: Img2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    if getattr(request, "init_image", None) is None:
        raise RuntimeError("img2vid requires 'init_image'")

    plan = build_video_plan(request)
    start = time.perf_counter()

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid")

    pipe = getattr(comp, "pipeline", None)
    high_model = getattr(comp, "pipeline_high", None)
    low_model = getattr(comp, "pipeline_low", None)

    apply_engine_loras(engine, logger)

    active_pipe_hi = high_model or pipe
    if active_pipe_hi is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    if wan_hi_opts and wan_hi_opts.lora_path and hasattr(active_pipe_hi, "load_lora_weights"):
        if logger:
            logger.info("[wan] loading high-stage LoRA: %s", wan_hi_opts.lora_path)
        active_pipe_hi.load_lora_weights(wan_hi_opts.lora_path)  # type: ignore[attr-defined]

    outcome_hi = configure_sampler(active_pipe_hi, plan, logger)

    yield ProgressEvent(stage="run_high", percent=5.0, message="Stage 1 (High Noise)")
    frames_hi = _run_stage(
        active_pipe_hi,
        plan,
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
    )

    active_pipe_lo = low_model or pipe
    outcome_lo = None
    frames = list(frames_hi)

    if active_pipe_lo is not None and frames_hi:
        wan_low_cfg = extras.get("wan_low")
        wan_opts = WanStageOptions.from_mapping(wan_low_cfg) if isinstance(wan_low_cfg, dict) else None
        if wan_opts and wan_opts.lora_path and hasattr(active_pipe_lo, "load_lora_weights"):
            if logger:
                logger.info("[wan] loading low-stage LoRA: %s", wan_opts.lora_path)
            active_pipe_lo.load_lora_weights(wan_opts.lora_path)  # type: ignore[attr-defined]

        outcome_lo = configure_sampler(active_pipe_lo, plan, logger)
        yield ProgressEvent(stage="run_low", percent=50.0, message="Stage 2 (Low Noise)")
        frames = _run_stage(
            active_pipe_lo,
            plan,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            init_image=frames_hi[-1],
        )

    vfi_opts = None
    vfi_cfg = extras.get("video_interpolation") if isinstance(extras, dict) else None
    if isinstance(vfi_cfg, dict):
        vio = VideoInterpolationOptions(
            enabled=bool(vfi_cfg.get("enabled", False)),
            model=str(vfi_cfg.get("model")) if vfi_cfg.get("model") is not None else None,
            times=int(vfi_cfg.get("times")) if vfi_cfg.get("times") is not None else None,
        )
        vfi_opts = vio.as_dict()
        if vio.enabled and (vio.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
            frames, vfi_meta = maybe_interpolate(frames, enabled=vio.enabled, model=vio.model, times=vio.times or 2, logger=logger)
            vfi_opts = {**vfi_opts, "result": vfi_meta}

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None))

    extra_meta: dict[str, Any] = dict(extras) if isinstance(extras, dict) else {}
    if vfi_opts is not None:
        extra_meta["video_interpolation"] = vfi_opts
    if outcome_lo is not None:
        extra_meta["sampler_low"] = {
            "sampler_in": getattr(outcome_lo, "sampler_in", None),
            "scheduler_in": getattr(outcome_lo, "scheduler_in", None),
            "sampler": getattr(outcome_lo, "sampler_effective", None),
            "scheduler": getattr(outcome_lo, "scheduler_effective", None),
        }

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        outcome_hi,
        elapsed=elapsed,
        task="img2vid",
        extra=extra_meta,
        video_meta=video_meta,
    )

    payload = {
        "images": result.frames,
        "info": engine._to_json(result.metadata),  # type: ignore[attr-defined]
    }
    yield ResultEvent(payload=payload)
