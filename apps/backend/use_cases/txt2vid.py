"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2vid orchestration for WAN22 (Diffusers pipeline or GGUF runtime).
Configures sampler settings, applies LoRAs, runs the selected execution path, applies shared SeedVR2 upscaling/interpolation stages when requested, exports the
resulting video, and yields progress/result events.
Diffusers stage execution requires `extras.wan_high.prompt` (non-empty); stage negative uses explicit value when provided and falls back to request negative only when missing.

Symbols (top-level; keep in sync; no ghosts):
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional frames) and attaches warnings.
- `_run_pipeline` (function): Runs a Diffusers txt2vid pipeline and returns generated frames.
- `_yield_wan22_gguf_progress` (function): Maps WAN22 GGUF stream dict events into backend `ProgressEvent`s.
- `run_txt2vid` (function): Orchestrates txt2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.pipeline_stages.video import (
    apply_engine_loras,
    apply_video_interpolation,
    apply_video_upscaling,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
    read_video_interpolation_options,
    read_video_upscaling_options,
    resolve_video_output_fps,
)


def _build_result_payload(
    *,
    engine: Any,
    result: Any,
    plan: VideoPlan,
    request: Txt2VidRequest,
    video_meta: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(getattr(result, "metadata", {}) or {})

    user_return_frames = parse_bool_value(
        plan.extras.get("video_return_frames"),
        field="extras.video_return_frames",
        default=False,
    )
    video_options = getattr(request, "video_options", None)
    save_output = parse_bool_value(
        video_options.get("save_output") if isinstance(video_options, Mapping) else None,
        field="video_options.save_output",
        default=False,
    )

    video_saved = parse_bool_value(
        video_meta.get("saved") if isinstance(video_meta, dict) else None,
        field="video_meta.saved",
        default=False,
    )
    export_failed = save_output and not video_saved

    effective_return_frames = user_return_frames or (not save_output) or export_failed

    warnings: list[str] = []
    if not save_output:
        warnings.append(
            "Save output is OFF: no video file was written. "
            "Frames are returned so you can download them from the Results viewer."
        )
    if export_failed:
        reason = video_meta.get("reason") if isinstance(video_meta, dict) else None
        warnings.append(
            f"Video export failed ({reason or 'unknown error'}). "
            "Frames are returned as a fallback."
        )

    if warnings:
        metadata["warnings"] = warnings

    payload: dict[str, Any] = {"info": engine._to_json(metadata)}  # type: ignore[attr-defined]
    if effective_return_frames:
        payload["images"] = getattr(result, "frames", [])
    if video_saved:
        payload["video"] = {
            "rel_path": video_meta.get("rel_path"),
            "mime": video_meta.get("mime"),
        }
    return payload


def _run_pipeline(
    pipe: Any,
    plan: VideoPlan,
    request: Txt2VidRequest,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> list[Any]:
    prompt_text = str(prompt if prompt is not None else request.prompt or "").strip()
    if not prompt_text:
        raise RuntimeError("txt2vid requires a non-empty prompt.")
    negative_prompt_text = (
        str(negative_prompt).strip()
        if negative_prompt is not None
        else str(getattr(request, "negative_prompt", None) or "").strip()
    )
    import torch

    with torch.inference_mode():
        output = pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
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
            device=getattr(comp, "device", None),
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
                raw_frames = ev.get("frames", [])
                if raw_frames is None:
                    frames = []
                elif isinstance(raw_frames, list):
                    frames = raw_frames
                elif isinstance(raw_frames, tuple):
                    frames = list(raw_frames)
                else:
                    raise RuntimeError(
                        "WAN22 GGUF: invalid result payload for 'frames' "
                        f"(expected sequence, got {type(raw_frames).__name__})"
                    )
                break
            raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")

        if not frames:
            raise RuntimeError("WAN22 GGUF: produced no frames")

        upscaling_options = read_video_upscaling_options(plan.extras)
        if upscaling_options is not None and upscaling_options.enabled:
            yield ProgressEvent(stage="upscale", percent=1.0, message="Upscaling frames (SeedVR2)")
        frames, upscaling_opts = apply_video_upscaling(
            frames,
            options=upscaling_options,
            logger_=logger,
            component_device=getattr(comp, "device", None),
        )
        if frames:
            first_size = getattr(frames[0], "size", None)
            if isinstance(first_size, tuple) and len(first_size) == 2:
                plan.width = int(first_size[0])
                plan.height = int(first_size[1])

        vfi_options = read_video_interpolation_options(plan.extras)
        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        extra_meta: dict[str, Any] = dict(plan.extras)
        if upscaling_opts is not None:
            extra_meta["video_upscaling"] = upscaling_opts
        if vfi_opts is not None:
            extra_meta["video_interpolation"] = vfi_opts
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

        yield ResultEvent(
            payload=_build_result_payload(
                engine=engine,
                result=result,
                plan=plan,
                request=request,
                video_meta=video_meta,
            )
        )
        return

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    if wan_hi_opts is None or wan_hi_opts.prompt is None:
        raise RuntimeError("txt2vid requires extras.wan_high.prompt to be set.")
    prompt_text = str(wan_hi_opts.prompt).strip()
    if not prompt_text:
        raise RuntimeError("txt2vid requires a non-empty high-stage prompt.")
    negative_prompt_text = (
        str(wan_hi_opts.negative_prompt).strip()
        if wan_hi_opts and wan_hi_opts.negative_prompt is not None
        else str(getattr(request, "negative_prompt", None) or "").strip()
    )
    if wan_hi_opts and wan_hi_opts.loras and hasattr(pipe, "load_lora_weights"):
        total_stage_loras = len(wan_hi_opts.loras)
        for index, (lora_path, lora_weight) in enumerate(wan_hi_opts.loras):
            if logger:
                logger.info(
                    "[wan] loading stage LoRA %d/%d: %s (weight=%s)",
                    index + 1,
                    total_stage_loras,
                    lora_path,
                    lora_weight,
                )
            pipe.load_lora_weights(lora_path)  # type: ignore[attr-defined]

    apply_engine_loras(engine, logger)

    sampler_outcome = configure_sampler(pipe, plan, logger)

    yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
    frames = _run_pipeline(
        pipe,
        plan,
        request,
        prompt=prompt_text,
        negative_prompt=negative_prompt_text,
    )

    upscaling_options = read_video_upscaling_options(plan.extras)
    if upscaling_options is not None and upscaling_options.enabled:
        yield ProgressEvent(stage="upscale", percent=1.0, message="Upscaling frames (SeedVR2)")
    frames, upscaling_opts = apply_video_upscaling(
        frames,
        options=upscaling_options,
        logger_=logger,
        component_device=getattr(comp, "device", None),
    )
    if frames:
        first_size = getattr(frames[0], "size", None)
        if isinstance(first_size, tuple) and len(first_size) == 2:
            plan.width = int(first_size[0])
            plan.height = int(first_size[1])

    vfi_options = read_video_interpolation_options(plan.extras)
    if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
        yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
    frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
    plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")

    extra_meta: dict[str, Any] = dict(plan.extras)
    if upscaling_opts is not None:
        extra_meta["video_upscaling"] = upscaling_opts
    if vfi_opts is not None:
        extra_meta["video_interpolation"] = vfi_opts

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        task="txt2vid",
        extra=extra_meta,
        video_meta=video_meta,
    )

    yield ResultEvent(
        payload=_build_result_payload(
            engine=engine,
            result=result,
            plan=plan,
            request=request,
            video_meta=video_meta,
        )
    )
