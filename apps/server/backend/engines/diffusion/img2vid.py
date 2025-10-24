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
        # GGUF path: delegate to wan22 runtime (img2vid)
        from apps.server.backend.runtime.nn.wan22 import RunConfig, StageConfig, run_img2vid as gguf_i2v
        ex = getattr(request, 'extras', {}) or {}
        hi_ex = ex.get('wan_high') if isinstance(ex, dict) else None
        lo_ex = ex.get('wan_low') if isinstance(ex, dict) else None
        def _s(d, k, fallback):
            try:
                return (d.get(k) if isinstance(d, dict) else None) or fallback
            except Exception:
                return fallback
        cfg = RunConfig(
            width=int(getattr(request, "width", 768) or 768),
            height=int(getattr(request, "height", 432) or 432),
            fps=int(getattr(request, "fps", 24) or 24),
            num_frames=int(getattr(request, "num_frames", 16) or 16),
            guidance_scale=getattr(request, "cfg_scale", None),
            dtype=str(getattr(comp, "dtype", "fp16")),
            device=str(getattr(comp, "device", "cuda")),
            seed=(int(getattr(request, "seed", -1)) if getattr(request, "seed", None) is not None else None),
            init_image=getattr(request, "init_image", None),
            vae_dir=_s(ex, 'wan_vae_dir', getattr(comp, 'model_dir', None)),
            text_encoder_dir=_s(ex, 'wan_text_encoder_dir', None),
            tokenizer_dir=_s(ex, 'wan_tokenizer_dir', None),
            high=StageConfig(
                model_dir=(getattr(comp, 'high_dir', None) or _s(hi_ex, 'model_dir', getattr(comp, 'model_dir', ''))),
                sampler=str(_s(hi_ex, 'sampler', getattr(request, 'sampler', 'Automatic'))),
                scheduler=str(_s(hi_ex, 'scheduler', getattr(request, 'scheduler', 'Automatic'))),
                steps=int(_s(hi_ex, 'steps', 12)),
                cfg_scale=_s(hi_ex, 'cfg_scale', getattr(request, 'guidance_scale', None)),
            ),
            low=StageConfig(
                model_dir=(getattr(comp, 'low_dir', None) or _s(lo_ex, 'model_dir', getattr(comp, 'model_dir', ''))),
                sampler=str(_s(lo_ex, 'sampler', getattr(request, 'sampler', 'Automatic'))),
                scheduler=str(_s(lo_ex, 'scheduler', getattr(request, 'scheduler', 'Automatic'))),
                steps=int(_s(lo_ex, 'steps', 12)),
                cfg_scale=_s(lo_ex, 'cfg_scale', getattr(request, 'guidance_scale', None)),
            ),
        )
        frames = gguf_i2v(cfg, logger=logger)
        # VFI/export
        vfi_opts = None
        try:
            vi = (getattr(request, 'extras', {}) or {}).get('video_interpolation')
            if isinstance(vi, dict):
                from apps.server.backend.core.params.video import VideoInterpolationOptions
                vfi = VideoInterpolationOptions(enabled=bool(vi.get('enabled', False)), model=str(vi.get('model')) if vi.get('model') else None, times=int(vi.get('times')) if vi.get('times') else None)
                vfi_opts = vfi.as_dict()
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
            "wan": {
                "high": {
                    "steps": int(getattr(request, "steps", 12) or 12),
                    "sampler": str(getattr(request, "sampler", "Automatic")),
                    "scheduler": str(getattr(request, "scheduler", "Automatic")),
                    "cfg": getattr(request, "guidance_scale", None),
                },
                "low": {
                    "steps": int(getattr(request, "steps", 12) or 12),
                    "sampler": str(getattr(request, "sampler", "Automatic")),
                    "scheduler": str(getattr(request, "scheduler", "Automatic")),
                    "cfg": getattr(request, "guidance_scale", None),
                },
            },
            "seed": getattr(request, "seed", None),
        }
        yield ResultEvent(payload={"images": frames, "info": engine._to_json(info)})  # type: ignore[attr-defined]
        return

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
