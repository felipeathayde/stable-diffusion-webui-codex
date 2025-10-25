from __future__ import annotations

import time
from typing import Iterator, List, Any, Optional

from apps.server.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.server.backend.codex import lora as codex_lora
from apps.server.backend.patchers.lora_apply import apply_loras_to_engine
from apps.server.backend.engines.util.schedulers import apply_sampler_scheduler, SamplerKind


def run_txt2vid(*, engine, comp, request: Txt2VidRequest) -> Iterator[InferenceEvent]:
    """Generic txt2vid flow using Diffusers when available; GGUF path todo.

    Expects `comp` to have fields: pipeline, device, dtype, model_dir, high_dir, low_dir.
    """
    logger = getattr(engine, "_logger", None)
    start = time.perf_counter()
    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid")

    pipe = getattr(comp, "pipeline", None)

    # Native LoRA application (error on failure)
    sels = codex_lora.get_selections()
    if sels and hasattr(engine, "forge_objects_after_applying_lora"):
        apply_loras_to_engine(engine, sels)
        if logger:
            logger.info("[native] txt2vid applied %d LoRA(s)", len(sels))

    if pipe is None:
        raise RuntimeError("txt2vid requires a Diffusers pipeline; none found in components")

    # Sampler/Scheduler mapping
    sampler = getattr(request, "sampler", "Automatic")
    scheduler = getattr(request, "scheduler", "Automatic")
    outcome = apply_sampler_scheduler(pipe, SamplerKind.from_string(sampler), scheduler)
    for w in outcome.warnings:
        if logger:
            logger.warning("txt2vid: %s", w)

    yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
    out = pipe(
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        num_frames=int(getattr(request, "num_frames", 16) or 16),
        num_inference_steps=max(1, int(getattr(request, "steps", 12) or 12)),
        height=int(getattr(request, "height", 432) or 432),
        width=int(getattr(request, "width", 768) or 768),
        guidance_scale=getattr(request, "cfg_scale", None),
    )
    frames: List[object] = list(out.frames[0]) if hasattr(out, "frames") else []

    fps = int(getattr(request, "fps", 24) or 24)
    # Delegate export to engine helper if present
    video_opts = getattr(request, "video_options", None)
    video_meta = engine._maybe_export_video(frames, fps=fps, options=video_opts)  # type: ignore[attr-defined]

    info = {
        "engine": getattr(engine, "engine_id", "unknown"),
        "task": "txt2vid",
        "elapsed": round(time.perf_counter() - start, 3),
        "frames": len(frames),
        "sampler_in": outcome.sampler_in,
        "scheduler_in": outcome.scheduler_in,
        "sampler": outcome.sampler_effective,
        "scheduler": outcome.scheduler_effective,
    }
    yield ResultEvent(payload={"images": frames, "info": engine._to_json(info)})  # type: ignore[attr-defined]
