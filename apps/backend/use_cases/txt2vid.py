"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2vid orchestration for WAN22 (Diffusers pipeline or GGUF runtime).
Configures sampler settings, applies LoRAs, runs the selected execution path, exports the resulting video, and yields progress/result events.

Symbols (top-level; keep in sync; no ghosts):
- `_run_pipeline` (function): Runs a Diffusers txt2vid pipeline and returns generated frames.
- `_yield_wan22_gguf_progress` (function): Maps WAN22 GGUF stream dict events into backend `ProgressEvent`s.
- `run_txt2vid` (function): Orchestrates txt2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.workflows.video import (
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


def _yield_wan22_gguf_progress(ev: dict) -> Optional[ProgressEvent]:
    if ev.get("type") != "progress":
        return None
    stage = str(ev.get("stage", "") or "")
    step = int(ev.get("step", 0))
    total = int(ev.get("total", 0))
    pct = float(ev.get("percent", 0.0))
    pct_out = (pct * 100.0) if (0.0 <= pct <= 1.0) else pct
    eta_raw = ev.get("eta_seconds", None)
    eta = float(eta_raw) if eta_raw is not None else None
    return ProgressEvent(stage=stage, percent=pct_out, step=step, total_steps=total, eta_seconds=eta)


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
        from apps.backend.runtime.families.wan22.config import build_wan22_gguf_run_config
        from apps.backend.runtime.families.wan22 import wan22 as gguf

        cfg = build_wan22_gguf_run_config(
            request=request,
            device=getattr(comp, "device", "auto"),
            dtype=getattr(comp, "dtype", "fp16"),
            logger=logger,
        )

        frames: list[Any] | None = None
        for ev in gguf.stream_txt2vid(cfg, logger=logger):
            if not isinstance(ev, dict):
                raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
            if ev.get("type") == "progress":
                pe = _yield_wan22_gguf_progress(ev)
                if pe is not None:
                    yield pe
                continue
            if ev.get("type") == "result":
                frames = list(ev.get("frames", []) or [])
                break
            raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")

        if not frames:
            raise RuntimeError("WAN22 GGUF: produced no frames")

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None))

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        extra_meta: dict[str, Any] = dict(plan.extras)
        if cfg.low is not None:
            extra_meta["sampler_low"] = {
                "sampler_in": cfg.low.sampler,
                "scheduler_in": cfg.low.scheduler,
                "sampler": cfg.low.sampler,
                "scheduler": cfg.low.scheduler,
            }

        elapsed = time.perf_counter() - start
        result = build_video_result(
            engine,
            frames,
            plan,
            _SamplerOutcome(
                sampler_in=getattr(request, "sampler", None),
                scheduler_in=getattr(request, "scheduler", None),
                sampler_effective=(cfg.high.sampler if cfg.high is not None else getattr(request, "sampler", None)),
                scheduler_effective=(cfg.high.scheduler if cfg.high is not None else getattr(request, "scheduler", None)),
            ),
            elapsed=elapsed,
            task="txt2vid",
            extra=extra_meta,
            video_meta=video_meta,
        )

        payload = {
            "images": result.frames,
            "info": engine._to_json(result.metadata),  # type: ignore[attr-defined]
        }
        yield ResultEvent(payload=payload)
        return

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
