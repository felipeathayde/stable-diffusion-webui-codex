from __future__ import annotations

import time
from typing import Iterator, List, Any

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

    # Best-effort native LoRA application (only if engine exposes patchers)
    try:
        sels = codex_lora.get_selections()
        if sels and hasattr(engine, 'forge_objects_after_applying_lora'):
            apply_loras_to_engine(engine, sels)
            if logger:
                logger.info("[native] txt2vid applied %d LoRA(s)", len(sels))
    except Exception:
        pass
    if pipe is None and (getattr(comp, "high_dir", None) or getattr(comp, "low_dir", None)):
        # GGUF path via runtime/nn/wan22
        from apps.server.backend.runtime.nn.wan22 import RunConfig, StageConfig, run_txt2vid as gguf_t2v
        # Stage overrides from request.extras
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
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
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
        frames = gguf_t2v(cfg, logger=logger)
        fps = int(getattr(request, "fps", 24) or 24)
        video_meta = None
        try:
            video_opts = getattr(request, "video_options", None)
            video_meta = engine._maybe_export_video(frames, fps=fps, options=video_opts)  # type: ignore[attr-defined]
        except Exception:
            video_meta = None
        info = {
            "engine": getattr(engine, "engine_id", "unknown"),
            "task": "txt2vid",
            "elapsed": round(time.perf_counter() - start, 3),
            "frames": len(frames),
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
    if pipe is None:
        raise RuntimeError("Diffusers pipeline not available for txt2vid and no GGUF fallback found")

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
    video_meta = None
    try:
        video_opts = getattr(request, "video_options", None)
        video_meta = engine._maybe_export_video(frames, fps=fps, options=video_opts)  # type: ignore[attr-defined]
    except Exception:
        video_meta = None

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
