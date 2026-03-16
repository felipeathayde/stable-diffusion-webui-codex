"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical txt2vid orchestration for backend video engines.
Runs the selected video execution path (active WAN22 Diffusers/GGUF lanes plus the native backend-only LTX2 branch), applies
shared SeedVR2 upscaling/interpolation stages when requested, exports the resulting video, and yields
progress/result events.
WAN22 Diffusers stage execution requires `extras.wan_high.prompt` (non-empty); stage negative uses explicit value when
provided and falls back to request negative only when missing. The native LTX2 branch consumes a local
`Ltx2RunResult` (`frames + AudioExportAsset + metadata`) and owns cleanup of generated temp audio after export.

Symbols (top-level; keep in sync; no ghosts):
- `_build_pipeline_telemetry_scope` (function): Creates a mutable task-scoped telemetry context owner for txt2vid run/stage events.
- `_emit_pipeline_event` (function): Emits canonical structured pipeline telemetry events (`pipeline.*`) for txt2vid.
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional frames) and attaches warnings.
- `_cleanup_owned_audio_asset` (function): Deletes owned temporary generated-audio artifacts after LTX2 export completes or fails.
- `_run_ltx2_txt2vid` (function): Runs the native backend-only LTX2 txt2vid branch and threads generated audio through the shared export seam.
- `_run_pipeline` (function): Runs a Diffusers txt2vid pipeline and returns generated frames.
- `_apply_stage_loras_to_pipeline` (function): Loads and activates ordered stage LoRA adapters on a Diffusers WAN pipeline.
- `_yield_wan22_gguf_progress` (function): Maps WAN22 GGUF stream dict events into backend `ProgressEvent`s.
- `run_txt2vid` (function): Orchestrates txt2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.pipeline_stages.hires_fix import resolve_pipeline_telemetry_context
from apps.backend.runtime.pipeline_stages.video import (
    AudioExportAsset,
    apply_engine_loras,
    apply_video_interpolation,
    apply_video_upscaling,
    build_video_request_effective_snapshot,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
    prepare_base_snapshot_video_options,
    read_video_interpolation_options,
    read_video_upscaling_options,
    resolve_generated_audio_export_policy,
    resolve_video_output_fps,
)


def _build_pipeline_telemetry_scope(*, mode: str) -> SimpleNamespace:
    scope = SimpleNamespace()
    setattr(scope, "_codex_pipeline_mode", str(mode))
    task_context = str(threading.current_thread().name or "").strip() or "unknown-thread"
    marker = "-task-"
    if marker in task_context:
        candidate = task_context.split(marker, 1)[1].strip()
        if candidate:
            setattr(scope, "_codex_task_id", candidate)
            setattr(scope, "_codex_correlation_id", candidate)
            setattr(scope, "_codex_hires_correlation_id", candidate)
            setattr(scope, "_codex_correlation_source", "task_id")
    resolve_pipeline_telemetry_context(
        scope,
        default_mode=str(mode),
        require_mode=True,
    )
    return scope


def _emit_pipeline_event(
    scope: Any,
    event: str,
    *,
    stage: str,
    **fields: object,
) -> None:
    telemetry = resolve_pipeline_telemetry_context(
        scope,
        default_mode="txt2vid",
        require_mode=True,
    )
    emit_backend_event(
        event,
        logger="backend.use_cases.txt2vid",
        mode=telemetry.mode,
        stage=stage,
        correlation_id=telemetry.correlation_id,
        correlation_source=telemetry.correlation_source,
        task_id=telemetry.task_id,
        **fields,
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


def _cleanup_owned_audio_asset(audio_asset: AudioExportAsset | None, *, logger: Any, task: str) -> None:
    if audio_asset is None or not audio_asset.owned_temp:
        return
    path = str(audio_asset.path or "").strip()
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        if logger is not None:
            logger.warning("%s: failed to remove owned temp audio asset '%s': %s", task, path, exc)


def _run_ltx2_txt2vid(
    *,
    engine: Any,
    comp: Any,
    request: Txt2VidRequest,
    plan: VideoPlan,
    start: float,
    logger: Any,
    telemetry_scope: Any,
) -> Iterator[InferenceEvent]:
    from apps.backend.runtime.families.ltx2.runtime import Ltx2RunResult

    @dataclass(frozen=True)
    class _SamplerOutcome:
        sampler_in: str | None
        scheduler_in: str | None
        sampler_effective: str | None
        scheduler_effective: str | None
        warnings: tuple[str, ...] = ()

    audio_asset: AudioExportAsset | None = None
    try:
        generated_audio_export_policy = resolve_generated_audio_export_policy(
            getattr(request, "video_options", None),
            task="txt2vid",
        )
        apply_engine_loras(engine, logger)

        yield ProgressEvent(stage="run", percent=5.0, message="Running LTX2 txt2vid")
        runtime_result = comp.run_txt2vid(
            request=request,
            plan=plan,
            generated_audio_export_policy=generated_audio_export_policy,
        )
        if not isinstance(runtime_result, Ltx2RunResult):
            raise RuntimeError(
                "LTX2 txt2vid runtime must return `Ltx2RunResult`; "
                f"got {type(runtime_result).__name__}."
            )

        frames = list(runtime_result.frames)
        audio_asset = runtime_result.audio_asset
        runtime_meta = dict(runtime_result.metadata)
        audio_source_kind = "generated" if audio_asset is not None else "none"

        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="ltx2",
            frame_count=int(len(frames)),
            has_audio=bool(audio_asset is not None),
        )

        upscaling_options = read_video_upscaling_options(plan.extras)
        vfi_options = read_video_interpolation_options(plan.extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="txt2vid",
            upscaling_options=upscaling_options,
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(
                engine,
                frames,
                plan,
                base_video_options,
                task="txt2vid",
                audio_asset=audio_asset,
            )
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path and logger is not None:
                    logger.info(
                        "txt2vid: base snapshot exported before post-process: %s",
                        base_rel_path,
                    )

        if upscaling_options is not None and upscaling_options.enabled:
            yield ProgressEvent(stage="upscale", percent=1.0, message="Upscaling frames (SeedVR2)")
        frames, upscaling_opts = apply_video_upscaling(
            frames,
            options=upscaling_options,
            logger_=logger,
            component_device=getattr(comp, "device", None),
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="upscaling.complete",
            stage_name="upscaling",
            backend="ltx2",
            upscaling_enabled=bool(upscaling_options is not None and upscaling_options.enabled),
            frame_count=int(len(frames)),
        )
        if frames:
            first_size = getattr(frames[0], "size", None)
            if isinstance(first_size, tuple) and len(first_size) == 2:
                plan.width = int(first_size[0])
                plan.height = int(first_size[1])

        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="interpolation.complete",
            stage_name="interpolation",
            backend="ltx2",
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=int(len(frames)),
        )

        video_meta = export_video(
            engine,
            frames,
            plan,
            getattr(request, "video_options", None),
            task="txt2vid",
            audio_asset=audio_asset,
        )
        video_saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="export.complete",
            stage_name="export",
            backend="ltx2",
            video_saved=video_saved,
            final_frame_count=int(len(frames)),
            has_audio=bool(audio_asset is not None),
        )

        extra_meta: dict[str, Any] = dict(plan.extras)
        if runtime_meta:
            extra_meta["ltx2_runtime"] = runtime_meta
        if upscaling_opts is not None:
            extra_meta["video_upscaling"] = upscaling_opts
        if vfi_opts is not None:
            extra_meta["video_interpolation"] = vfi_opts
        if base_video_meta is not None:
            extra_meta["video_base_snapshot"] = base_video_meta
        extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
            request=request,
            plan=plan,
            video_meta=video_meta,
            upscaling_options=upscaling_options,
            upscaling_meta=upscaling_opts,
            interpolation_options=vfi_options,
            interpolation_meta=vfi_opts,
            base_video_meta=base_video_meta,
            audio_source_kind=audio_source_kind,
            final_frame_count=len(frames),
        )

        sampler_effective = str(
            runtime_meta.get("sampler_effective")
            or runtime_meta.get("sampler")
            or getattr(request, "sampler", None)
            or ""
        ).strip() or None
        scheduler_effective = str(
            runtime_meta.get("scheduler_effective")
            or runtime_meta.get("scheduler")
            or getattr(request, "scheduler", None)
            or ""
        ).strip() or None

        elapsed = time.perf_counter() - start
        result = build_video_result(
            engine,
            frames,
            plan,
            _SamplerOutcome(
                sampler_in=getattr(request, "sampler", None),
                scheduler_in=getattr(request, "scheduler", None),
                sampler_effective=sampler_effective,
                scheduler_effective=scheduler_effective,
            ),
            elapsed=elapsed,
            task="txt2vid",
            extra=extra_meta,
            video_meta=video_meta,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="ltx2",
            total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
            final_frame_count=int(len(frames)),
            video_saved=video_saved,
            has_audio=bool(audio_asset is not None),
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
    finally:
        _cleanup_owned_audio_asset(audio_asset, logger=logger, task="txt2vid")


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


def _apply_stage_loras_to_pipeline(*, pipe: Any, stage_loras: tuple[tuple[str, float], ...], logger: Any, stage_label: str) -> None:
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()  # type: ignore[attr-defined]

    if not stage_loras:
        return
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{stage_label} stage LoRA requires a pipeline with 'load_lora_weights'.")
    if not hasattr(pipe, "set_adapters"):
        raise RuntimeError(f"{stage_label} stage LoRA requires a pipeline with 'set_adapters' for multi-LoRA support.")

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    total_stage_loras = len(stage_loras)
    for index, (lora_path, lora_weight) in enumerate(stage_loras):
        adapter_name = f"wan_{stage_label}_stage_lora_{index}"
        if logger:
            logger.info(
                "[wan] loading %s-stage LoRA %d/%d: %s (weight=%s adapter=%s)",
                stage_label,
                index + 1,
                total_stage_loras,
                lora_path,
                lora_weight,
                adapter_name,
            )
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)  # type: ignore[attr-defined]
        adapter_names.append(adapter_name)
        adapter_weights.append(float(lora_weight))

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)  # type: ignore[attr-defined]


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
    message_raw = ev.get("message", None)
    message = str(message_raw) if message_raw is not None else None
    raw_data = ev.get("data", None)
    data_payload: dict[str, Any] = dict(raw_data) if isinstance(raw_data, Mapping) else {}
    for key in ("progress_adapter", "progress_granularity", "coarse_reason"):
        if key in ev and ev.get(key) is not None:
            data_payload[key] = ev.get(key)
    return ProgressEvent(
        stage=stage,
        percent=pct_out,
        step=step,
        total_steps=total,
        eta_seconds=eta,
        message=message,
        data=data_payload,
    )


def run_txt2vid(
    *,
    engine,
    comp,
    request: Txt2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    telemetry_scope = _build_pipeline_telemetry_scope(mode="txt2vid")
    plan = build_video_plan(request)
    start = time.perf_counter()
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    if engine_id == "ltx2":
        pipe = None
        backend_variant = "ltx2"
    else:
        pipe = getattr(comp, "pipeline", None)
        backend_variant = "gguf" if pipe is None else "diffusers"

    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.run.start",
        stage="run.start",
        backend=backend_variant,
        engine_id=str(getattr(engine, "engine_id", "") or "unknown"),
        requested_frames=int(plan.frames),
        requested_width=int(plan.width),
        requested_height=int(plan.height),
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="prepare.complete",
        stage_name="prepare",
        backend=backend_variant,
        frames=int(plan.frames),
        width=int(plan.width),
        height=int(plan.height),
        steps=int(plan.steps),
    )

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid")

    if engine_id == "ltx2":
        yield from _run_ltx2_txt2vid(
            engine=engine,
            comp=comp,
            request=request,
            plan=plan,
            start=start,
            logger=logger,
            telemetry_scope=telemetry_scope,
        )
        return

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
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="gguf",
            frame_count=int(len(frames)),
        )

        upscaling_options = read_video_upscaling_options(plan.extras)
        vfi_options = read_video_interpolation_options(plan.extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="txt2vid",
            upscaling_options=upscaling_options,
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(engine, frames, plan, base_video_options, task="txt2vid")
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path:
                    logger.info(
                        "txt2vid: base snapshot exported before post-process: %s",
                        base_rel_path,
                    )

        if upscaling_options is not None and upscaling_options.enabled:
            yield ProgressEvent(stage="upscale", percent=1.0, message="Upscaling frames (SeedVR2)")
        frames, upscaling_opts = apply_video_upscaling(
            frames,
            options=upscaling_options,
            logger_=logger,
            component_device=getattr(comp, "device", None),
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="upscaling.complete",
            stage_name="upscaling",
            backend="gguf",
            upscaling_enabled=bool(upscaling_options is not None and upscaling_options.enabled),
            frame_count=int(len(frames)),
        )
        if frames:
            first_size = getattr(frames[0], "size", None)
            if isinstance(first_size, tuple) and len(first_size) == 2:
                plan.width = int(first_size[0])
                plan.height = int(first_size[1])

        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="interpolation.complete",
            stage_name="interpolation",
            backend="gguf",
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=int(len(frames)),
        )

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")
        video_saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="export.complete",
            stage_name="export",
            backend="gguf",
            video_saved=video_saved,
            final_frame_count=int(len(frames)),
        )

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
        if base_video_meta is not None:
            extra_meta["video_base_snapshot"] = base_video_meta
        extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
            request=request,
            plan=plan,
            video_meta=video_meta,
            upscaling_options=upscaling_options,
            upscaling_meta=upscaling_opts,
            interpolation_options=vfi_options,
            interpolation_meta=vfi_opts,
            base_video_meta=base_video_meta,
            audio_source_kind="none",
            final_frame_count=len(frames),
        )
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
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="gguf",
            total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
            final_frame_count=int(len(frames)),
            video_saved=video_saved,
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
    if wan_hi_opts and wan_hi_opts.loras:
        _apply_stage_loras_to_pipeline(
            pipe=pipe,
            stage_loras=wan_hi_opts.loras,
            logger=logger,
            stage_label="high",
        )

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
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="generation.complete",
        stage_name="generation",
        backend="diffusers",
        frame_count=int(len(frames)),
    )

    upscaling_options = read_video_upscaling_options(plan.extras)
    vfi_options = read_video_interpolation_options(plan.extras)
    base_video_options = prepare_base_snapshot_video_options(
        getattr(request, "video_options", None),
        task="txt2vid",
        upscaling_options=upscaling_options,
        interpolation_options=vfi_options,
    )
    base_video_meta: Any = None
    if base_video_options is not None:
        base_video_meta = export_video(engine, frames, plan, base_video_options, task="txt2vid")
        if isinstance(base_video_meta, Mapping):
            base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
            if base_rel_path:
                logger.info(
                    "txt2vid: base snapshot exported before post-process: %s",
                    base_rel_path,
                )

    if upscaling_options is not None and upscaling_options.enabled:
        yield ProgressEvent(stage="upscale", percent=1.0, message="Upscaling frames (SeedVR2)")
    frames, upscaling_opts = apply_video_upscaling(
        frames,
        options=upscaling_options,
        logger_=logger,
        component_device=getattr(comp, "device", None),
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="upscaling.complete",
        stage_name="upscaling",
        backend="diffusers",
        upscaling_enabled=bool(upscaling_options is not None and upscaling_options.enabled),
        frame_count=int(len(frames)),
    )
    if frames:
        first_size = getattr(frames[0], "size", None)
        if isinstance(first_size, tuple) and len(first_size) == 2:
            plan.width = int(first_size[0])
            plan.height = int(first_size[1])

    if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
        yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
    frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
    plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="interpolation.complete",
        stage_name="interpolation",
        backend="diffusers",
        interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
        output_fps=int(plan.fps),
        frame_count=int(len(frames)),
    )

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")
    video_saved = parse_bool_value(
        video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
        field="video_meta.saved",
        default=False,
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="export.complete",
        stage_name="export",
        backend="diffusers",
        video_saved=video_saved,
        final_frame_count=int(len(frames)),
    )

    extra_meta: dict[str, Any] = dict(plan.extras)
    if upscaling_opts is not None:
        extra_meta["video_upscaling"] = upscaling_opts
    if vfi_opts is not None:
        extra_meta["video_interpolation"] = vfi_opts
    if base_video_meta is not None:
        extra_meta["video_base_snapshot"] = base_video_meta
    extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
        request=request,
        plan=plan,
        video_meta=video_meta,
        upscaling_options=upscaling_options,
        upscaling_meta=upscaling_opts,
        interpolation_options=vfi_options,
        interpolation_meta=vfi_opts,
        base_video_meta=base_video_meta,
        audio_source_kind="none",
        final_frame_count=len(frames),
    )

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
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.run.complete",
        stage="run.complete",
        backend="diffusers",
        total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
        final_frame_count=int(len(frames)),
        video_saved=video_saved,
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
