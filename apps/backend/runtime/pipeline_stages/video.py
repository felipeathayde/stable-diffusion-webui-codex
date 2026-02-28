"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for Codex video generation pipelines.
Builds `VideoPlan`/`VideoResult`, applies LoRAs, configures sampler/scheduler, and assembles export metadata.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger used by video pipeline helpers.
- `build_video_plan` (function): Normalizes request attributes into a `VideoPlan`.
- `apply_engine_loras` (function): Applies globally selected LoRAs to the engine (when supported).
- `configure_sampler` (function): Applies sampler/scheduler configuration to a component given a `VideoPlan`.
- `read_video_interpolation_options` (function): Parses `extras.video_interpolation` into typed interpolation options when present.
- `apply_video_interpolation` (function): Applies the shared interpolation stage and returns `(frames_out, interpolation_metadata)`.
- `read_video_upscaling_options` (function): Parses `extras.video_upscaling` into typed upscaling options when present.
- `apply_video_upscaling` (function): Applies the shared SeedVR2 upscaling stage and returns `(frames_out, upscaling_metadata)`.
- `resolve_video_output_fps` (function): Computes output fps from request/base fps and interpolation metadata.
- `export_video` (function): Exports a frame sequence to a video file according to request options and a task label (stable output dir).
- `assemble_video_metadata` (function): Builds a metadata dict describing the generated video.
- `build_video_result` (function): Returns a `VideoResult` bundle for API/UI consumers.
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from apps.backend.core.params.video import VideoInterpolationOptions, VideoUpscalingOptions
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.runtime.adapters.lora import selections as lora_selections
from apps.backend.engines.util.schedulers import apply_sampler_scheduler, SamplerKind
from apps.backend.runtime.processing.datatypes import VideoPlan, VideoResult
from apps.backend.video.interpolation import maybe_interpolate
from apps.backend.video.upscaling.seedvr2 import run_seedvr2_upscaling

logger = logging.getLogger(__name__)
_VIDEO_UPSCALING_COLOR_CORRECTIONS = {"lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"}


def build_video_plan(request: Any) -> VideoPlan:
    """Normalize request attributes into a ``VideoPlan``."""

    extras_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any]
    if isinstance(extras_raw, Mapping):
        extras = dict(extras_raw)
    else:
        extras = {}

    return VideoPlan(
        sampler_name=getattr(request, "sampler", None),
        scheduler_name=getattr(request, "scheduler", None),
        steps=int(getattr(request, "steps", 30) or 30),
        frames=int(getattr(request, "num_frames", 16) or 16),
        fps=int(getattr(request, "fps", 24) or 24),
        width=int(getattr(request, "width", 768) or 768),
        height=int(getattr(request, "height", 432) or 432),
        guidance_scale=getattr(request, "guidance_scale", None),
        extras=extras,
    )


def apply_engine_loras(engine: Any, logger_: logging.Logger | None = None) -> Any | None:
    """Apply globally selected LoRAs to the engine, returning stats when available."""

    # Lazy import to keep pipeline stage module dependency-light for non-LoRA users.
    from apps.backend.patchers.lora_apply import apply_loras_to_engine

    try:
        selections = list(lora_selections.get_selections())
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        raise RuntimeError(f"Failed to fetch LoRA selections: {exc}") from exc

    has_lora_capability = hasattr(engine, "codex_objects_after_applying_lora")
    if not selections:
        if not has_lora_capability:
            return None
        # Empty-selection runs must still clear any stale LoRA state from prior requests.
        return apply_loras_to_engine(engine, [])

    if not has_lora_capability:
        raise RuntimeError(
            "Video pipeline LoRA selections were provided, but the active engine does not expose "
            "`codex_objects_after_applying_lora`."
        )

    stats = apply_loras_to_engine(engine, selections)
    if logger_:
        logger_.info(
            "[native] applied %d LoRA(s), %d params touched",
            getattr(stats, "files", len(selections)),
            getattr(stats, "params_touched", 0),
        )
    return stats

def configure_sampler(component: Any, plan: VideoPlan, logger_: logging.Logger | None = None) -> Any:
    """Apply sampler/scheduler selection on a Diffusers pipeline component."""

    sampler_name = plan.sampler_name or "euler"
    scheduler_name = plan.scheduler_name or "simple"
    outcome = apply_sampler_scheduler(
        component,
        SamplerKind.from_string(sampler_name),
        scheduler_name,
    )
    for warning in outcome.warnings:
        if logger_:
            logger_.warning("video sampler: %s", warning)
        else:  # pragma: no cover - fallback logging
            logger.warning("video sampler: %s", warning)
    return outcome


def read_video_interpolation_options(extras: Mapping[str, Any] | None) -> VideoInterpolationOptions | None:
    if not isinstance(extras, Mapping):
        return None
    cfg = extras.get("video_interpolation")
    if not isinstance(cfg, Mapping):
        return None
    model_raw = cfg.get("model")
    if model_raw is None:
        model = None
    else:
        if not isinstance(model_raw, str):
            raise RuntimeError(
                "video_interpolation.model must be a string when provided "
                f"(got {type(model_raw).__name__})."
            )
        model = model_raw.strip() or None

    times_raw = cfg.get("times")
    if times_raw is None:
        times = None
    else:
        if isinstance(times_raw, bool) or not isinstance(times_raw, int):
            raise RuntimeError(
                "video_interpolation.times must be an integer when provided "
                f"(got {type(times_raw).__name__})."
            )
        times = int(times_raw)
        if times < 2:
            raise RuntimeError(f"video_interpolation.times must be >= 2 when provided (got {times}).")

    return VideoInterpolationOptions(
        enabled=parse_bool_value(cfg.get("enabled"), field="video_interpolation.enabled", default=False),
        model=model,
        times=times,
    )


def apply_video_interpolation(
    frames: Sequence[Any],
    *,
    options: VideoInterpolationOptions | None,
    logger_: logging.Logger | None = None,
) -> tuple[list[Any], dict[str, Any] | None]:
    frames_list = frames if isinstance(frames, list) else list(frames)
    if options is None:
        return frames_list, None

    opts = options.as_dict()
    if options.enabled and (options.times or 0) > 1:
        out_frames, meta = maybe_interpolate(
            frames_list,
            enabled=options.enabled,
            model=options.model,
            times=options.times or 2,
            logger=logger_ if logger_ is not None else logger,
        )
        out_list = out_frames if isinstance(out_frames, list) else list(out_frames)
        return out_list, {**opts, "result": meta}

    return frames_list, opts


def read_video_upscaling_options(extras: Mapping[str, Any] | None) -> VideoUpscalingOptions | None:
    if not isinstance(extras, Mapping):
        return None
    cfg = extras.get("video_upscaling")
    if not isinstance(cfg, Mapping):
        return None

    def _optional_int(field: str, minimum: int | None = None) -> int | None:
        value = cfg.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"video_upscaling.{field} must be an integer when provided (got {type(value).__name__}).")
        parsed = int(value)
        if minimum is not None and parsed < minimum:
            raise RuntimeError(f"video_upscaling.{field} must be >= {minimum} when provided (got {parsed}).")
        return parsed

    def _optional_float(field: str, minimum: float, maximum: float) -> float | None:
        value = cfg.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"video_upscaling.{field} must be a number when provided (got {type(value).__name__}).")
        parsed = float(value)
        if parsed < minimum or parsed > maximum:
            raise RuntimeError(
                f"video_upscaling.{field} must be within [{minimum}, {maximum}] when provided (got {parsed})."
            )
        return parsed

    enabled = parse_bool_value(cfg.get("enabled"), field="video_upscaling.enabled", default=False)

    dit_model_raw = cfg.get("dit_model")
    if dit_model_raw is None:
        dit_model = None
    else:
        if not isinstance(dit_model_raw, str):
            raise RuntimeError(
                "video_upscaling.dit_model must be a string when provided "
                f"(got {type(dit_model_raw).__name__})."
            )
        dit_model_text = dit_model_raw.strip()
        dit_model = dit_model_text if dit_model_text else None

    resolution = _optional_int("resolution", minimum=16)
    max_resolution = _optional_int("max_resolution", minimum=0)
    batch_size = _optional_int("batch_size", minimum=1)
    if batch_size is not None and (batch_size - 1) % 4 != 0:
        raise RuntimeError(f"video_upscaling.batch_size must satisfy 4n+1 when provided (got {batch_size}).")

    uniform_batch_size_raw = cfg.get("uniform_batch_size")
    if uniform_batch_size_raw is None:
        uniform_batch_size = None
    else:
        uniform_batch_size = parse_bool_value(
            uniform_batch_size_raw,
            field="video_upscaling.uniform_batch_size",
            default=False,
        )

    temporal_overlap = _optional_int("temporal_overlap", minimum=0)
    prepend_frames = _optional_int("prepend_frames", minimum=0)

    color_correction_raw = cfg.get("color_correction")
    if color_correction_raw is None:
        color_correction = None
    else:
        if not isinstance(color_correction_raw, str):
            raise RuntimeError(
                "video_upscaling.color_correction must be a string when provided "
                f"(got {type(color_correction_raw).__name__})."
            )
        normalized_color = color_correction_raw.strip().lower()
        if normalized_color not in _VIDEO_UPSCALING_COLOR_CORRECTIONS:
            allowed = ", ".join(sorted(_VIDEO_UPSCALING_COLOR_CORRECTIONS))
            raise RuntimeError(
                f"video_upscaling.color_correction must be one of {{{allowed}}} when provided "
                f"(got {color_correction_raw!r})."
            )
        color_correction = normalized_color

    input_noise_scale = _optional_float("input_noise_scale", 0.0, 1.0)
    latent_noise_scale = _optional_float("latent_noise_scale", 0.0, 1.0)

    return VideoUpscalingOptions(
        enabled=enabled,
        dit_model=dit_model,
        resolution=resolution,
        max_resolution=max_resolution,
        batch_size=batch_size,
        uniform_batch_size=uniform_batch_size,
        temporal_overlap=temporal_overlap,
        prepend_frames=prepend_frames,
        color_correction=color_correction,
        input_noise_scale=input_noise_scale,
        latent_noise_scale=latent_noise_scale,
    )


def apply_video_upscaling(
    frames: Sequence[Any],
    *,
    options: VideoUpscalingOptions | None,
    logger_: logging.Logger | None = None,
    component_device: str | None = None,
) -> tuple[list[Any], dict[str, Any] | None]:
    frames_list = frames if isinstance(frames, list) else list(frames)
    if options is None:
        return frames_list, None

    opts = options.as_dict()
    if not options.enabled:
        return frames_list, opts

    out_frames, run_meta = run_seedvr2_upscaling(
        frames_list,
        options=options,
        component_device=component_device,
        logger_=logger_ if logger_ is not None else logger,
    )
    out_list = out_frames if isinstance(out_frames, list) else list(out_frames)
    return out_list, {**opts, "result": run_meta}


def resolve_video_output_fps(base_fps: int, interpolation_meta: Mapping[str, Any] | None) -> int:
    fps_base = int(base_fps) if int(base_fps) > 0 else 1
    if not isinstance(interpolation_meta, Mapping):
        return fps_base

    result = interpolation_meta.get("result")
    if not isinstance(result, Mapping) or not bool(result.get("applied", False)):
        return fps_base

    raw_times = interpolation_meta.get("times")
    try:
        times = int(raw_times) if raw_times is not None else 1
    except Exception:
        times = 1
    if times <= 1:
        return fps_base
    return fps_base * times


def export_video(engine: Any, frames: Sequence[Any], plan: VideoPlan, video_options: Any, *, task: str) -> Any:
    save_output = parse_bool_value(
        video_options.get("save_output") if isinstance(video_options, Mapping) else None,
        field="video_options.save_output",
        default=False,
    )
    if not hasattr(engine, "_maybe_export_video"):
        if save_output:
            raise RuntimeError(
                f"{task}: video export requested (save_output=true), but engine does not implement _maybe_export_video."
            )
        return None

    video_meta = engine._maybe_export_video(frames, fps=plan.fps, options=video_options, task=task)  # type: ignore[attr-defined]
    if save_output:
        saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        if not saved:
            reason = ""
            if isinstance(video_meta, Mapping):
                reason = str(video_meta.get("reason") or "").strip()
            raise RuntimeError(
                f"{task}: video export failed with save_output=true"
                + (f" ({reason})" if reason else "")
            )
    return video_meta


def assemble_video_metadata(
    engine: Any,
    plan: VideoPlan,
    sampler_outcome: Any,
    *,
    elapsed: float,
    frame_count: int,
    task: str,
    extra: Mapping[str, Any] | None = None,
    video_meta: Any = None,
) -> dict[str, Any]:
    engine_dispatch = str(getattr(engine, "engine_id", "unknown"))
    engine_label = engine_dispatch
    if isinstance(extra, Mapping):
        raw_variant = extra.get("wan_engine_variant")
        if isinstance(raw_variant, str) and raw_variant.strip():
            engine_label = raw_variant.strip()

    metadata: dict[str, Any] = {
        "engine": engine_label,
        "task": task,
        "elapsed": round(elapsed, 3),
        "frames": frame_count,
        "fps": plan.fps,
        "width": plan.width,
        "height": plan.height,
        "steps": plan.steps,
        "sampler_in": getattr(sampler_outcome, "sampler_in", None),
        "scheduler_in": getattr(sampler_outcome, "scheduler_in", None),
        "sampler": getattr(sampler_outcome, "sampler_effective", None),
        "scheduler": getattr(sampler_outcome, "scheduler_effective", None),
    }
    if plan.guidance_scale is not None:
        metadata["guidance_scale"] = float(plan.guidance_scale)
    if extra:
        metadata.update(dict(extra))
    if engine_label != engine_dispatch:
        metadata["engine_dispatch"] = engine_dispatch
    if video_meta is not None:
        metadata["video_export"] = video_meta
    return metadata


def build_video_result(
    engine: Any,
    frames: Sequence[Any],
    plan: VideoPlan,
    sampler_outcome: Any,
    *,
    elapsed: float,
    task: str,
    extra: Mapping[str, Any] | None = None,
    video_meta: Any = None,
) -> VideoResult:
    frames_list = frames if isinstance(frames, list) else list(frames)
    metadata = assemble_video_metadata(
        engine,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        frame_count=len(frames_list),
        task=task,
        extra=extra,
        video_meta=video_meta,
    )
    return VideoResult(frames=frames_list, metadata=metadata, video_meta=video_meta)


__all__ = [
    "apply_engine_loras",
    "build_video_plan",
    "configure_sampler",
    "read_video_interpolation_options",
    "apply_video_interpolation",
    "read_video_upscaling_options",
    "apply_video_upscaling",
    "resolve_video_output_fps",
    "export_video",
    "assemble_video_metadata",
    "build_video_result",
]
