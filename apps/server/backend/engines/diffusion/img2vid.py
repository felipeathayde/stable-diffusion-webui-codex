from __future__ import annotations

import time
from typing import Iterator, List, Any

from apps.server.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Img2VidRequest
from apps.server.backend.engines.util.schedulers import apply_sampler_scheduler, SamplerKind
from .wan22_common import WanStageOptions
from apps.server.backend.core.params.video import VideoInterpolationOptions
from apps.server.backend.video.interpolation import maybe_interpolate


def run_img2vid(*, engine, comp, request: Img2VidRequest) -> Iterator[InferenceEvent]:
    """Generic img2vid flow with optional two-stage High→Low and VFI.

    Expects `comp` to have: pipeline (or stage pipelines), model_dir, high_dir, low_dir, device, dtype.
    """
    logger = getattr(engine, "_logger", None)
    start = time.perf_counter()
    if getattr(request, "init_image", None) is None:
        raise RuntimeError("img2vid requires 'init_image'")

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid")

    pipe = getattr(comp, "pipeline", None)
    high_model = getattr(comp, "pipeline_high", None)
    low_model = getattr(comp, "pipeline_low", None)

    # Prefer explicit stage pipelines if provided (per-stage dirs)
    if high_model is None and low_model is None and pipe is None and (getattr(comp, "high_dir", None) or getattr(comp, "low_dir", None)):
        raise RuntimeError("GGUF forward for img2vid pending; supply Diffusers WAN pipeline for now")

    # Stage 1 (High)
    active_pipe_hi = high_model or pipe
    if active_pipe_hi is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")
    outcome = apply_sampler_scheduler(
        active_pipe_hi,
        SamplerKind.from_string(getattr(request, "sampler", "Automatic")),
        getattr(request, "scheduler", "Automatic"),
    )
    for w in outcome.warnings:
        if logger:
            logger.warning("img2vid: %s", w)
    yield ProgressEvent(stage="run_high", percent=5.0, message="Stage 1 (High Noise)")
    out_hi = active_pipe_hi(
        image=getattr(request, "init_image", None),
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        num_frames=int(getattr(request, "num_frames", 16) or 16),
        num_inference_steps=max(1, int(getattr(request, "steps", 12) or 12)),
        height=int(getattr(request, "height", 432) or 432),
        width=int(getattr(request, "width", 768) or 768),
        guidance_scale=getattr(request, "cfg_scale", None),
    )
    frames_hi: List[object] = list(out_hi.frames[0]) if hasattr(out_hi, "frames") else []

    # Stage 2 (Low) — seed with last frame of High
    frames: List[object]
    active_pipe_lo = low_model or pipe
    lo = None
    try:
        extras = getattr(request, "extras", {}) or {}
        lo = WanStageOptions.from_mapping(extras.get("wan_low")) if isinstance(extras, dict) else None
    except Exception:
        lo = None
    if active_pipe_lo is not None and frames_hi:
        # Apply LoRA per-stage if available (best-effort)
        try:
            if isinstance(lo, WanStageOptions) and lo.lora_path:
                if hasattr(active_pipe_lo, "load_lora_weights"):
                    active_pipe_lo.load_lora_weights(lo.lora_path)  # type: ignore[attr-defined]
        except Exception:
            pass
        outcome_lo = apply_sampler_scheduler(
            active_pipe_lo,
            SamplerKind.from_string(getattr(request, "sampler", "Automatic")),
            getattr(request, "scheduler", "Automatic"),
        )
        for w in outcome_lo.warnings:
            if logger:
                logger.warning("img2vid(low): %s", w)
        yield ProgressEvent(stage="run_low", percent=50.0, message="Stage 2 (Low Noise)")
        seed_image = frames_hi[-1]
        out_lo = active_pipe_lo(
            image=seed_image,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            num_frames=int(getattr(request, "num_frames", 16) or 16),
            num_inference_steps=max(1, int(getattr(request, "steps", 12) or 12)),
            height=int(getattr(request, "height", 432) or 432),
            width=int(getattr(request, "width", 768) or 768),
            guidance_scale=getattr(request, "cfg_scale", None),
        )
        frames = list(out_lo.frames[0]) if hasattr(out_lo, "frames") else frames_hi
    else:
        frames = frames_hi

    # Optional interpolation (VFI)
    vfi_opts = None
    try:
        vi = (getattr(request, 'extras', {}) or {}).get('video_interpolation')
        if isinstance(vi, dict):
            vfi = VideoInterpolationOptions(
                enabled=bool(vi.get('enabled', False)),
                model=str(vi.get('model')) if vi.get('model') is not None else None,
                times=int(vi.get('times')) if vi.get('times') is not None else None,
            )
            vfi_opts = vfi.as_dict()
            if vfi.enabled and (vfi.times or 0) > 1:
                yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
                frames, vfi_meta = maybe_interpolate(frames, enabled=vfi.enabled, model=vfi.model, times=vfi.times or 2, logger=logger)
                vfi_opts = {**vfi_opts, **{"result": vfi_meta}}
    except Exception:
        vfi_opts = None

    fps = int(getattr(request, "fps", 24) or 24)
    video_meta = None
    try:
        video_opts = getattr(request, "video_options", None)
        video_meta = engine._maybe_export_video(frames, fps=fps, options=video_opts)  # type: ignore[attr-defined]
    except Exception:
        video_meta = None

    info = {
        "engine": getattr(engine, "engine_id", "unknown"),
        "task": "img2vid",
        "elapsed": round(time.perf_counter() - start, 3),
        "frames": len(frames),
        "video_interpolation": vfi_opts,
    }
    yield ResultEvent(payload={"images": frames, "info": engine._to_json(info)})  # type: ignore[attr-defined]
