"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for Codex video generation pipelines.
Builds `VideoPlan`/`VideoResult`, applies LoRAs, configures sampler/scheduler, explicitly rejects native-only sampler variants unsupported
by the diffusers video bridge, and assembles export metadata.

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
- `prepare_base_snapshot_video_options` (function): Builds a fail-loud snapshot export options payload for base-video persistence before post-processing.
- `build_video_request_effective_snapshot` (function): Builds an immutable request-vs-effective execution snapshot for WAN video metadata.
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
    sampler_kind = SamplerKind.from_string(sampler_name)
    if sampler_kind is SamplerKind.UNI_PC_BH2:
        raise RuntimeError(
            "Video scheduler bridge does not implement sampler 'uni-pc bh2'; "
            "use 'uni-pc' for this execution path."
        )
    outcome = apply_sampler_scheduler(
        component,
        sampler_kind,
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


def prepare_base_snapshot_video_options(
    video_options: Any,
    *,
    task: str,
    upscaling_options: VideoUpscalingOptions | None,
    interpolation_options: VideoInterpolationOptions | None,
) -> dict[str, Any] | None:
    save_output = parse_bool_value(
        video_options.get("save_output") if isinstance(video_options, Mapping) else None,
        field="video_options.save_output",
        default=False,
    )
    if not save_output:
        return None

    upscaling_enabled = bool(upscaling_options is not None and upscaling_options.enabled)
    interpolation_enabled = bool(
        interpolation_options is not None
        and interpolation_options.enabled
        and int(interpolation_options.times or 0) > 1
    )
    if not (upscaling_enabled or interpolation_enabled):
        return None

    normalized_options: dict[str, Any] = dict(video_options) if isinstance(video_options, Mapping) else {}
    base_prefix = str(normalized_options.get("filename_prefix") or task or "video").strip() or "video"
    normalized_options["filename_prefix"] = f"{base_prefix}_base"
    normalized_options["save_output"] = True
    return normalized_options


def _snapshot_clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _snapshot_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_clone(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _export_meta_field(meta: Any, *, key: str) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None)


def _normalized_export_meta(meta: Any) -> dict[str, Any] | None:
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        payload: dict[str, Any] = dict(meta)
    else:
        payload = {
            "saved": getattr(meta, "saved", None),
            "rel_path": getattr(meta, "rel_path", None),
            "mime": getattr(meta, "mime", None),
            "reason": getattr(meta, "reason", None),
            "fps": getattr(meta, "fps", None),
            "frames": getattr(meta, "frame_count", None),
        }
    if payload.get("frames") is None and payload.get("frame_count") is not None:
        payload["frames"] = payload.get("frame_count")
    payload.pop("frame_count", None)
    return _snapshot_clone(payload)


def _container_supports_audio(format_value: Any) -> bool:
    normalized = str(format_value or "").strip().lower()
    if normalized in {"video/gif", "image/gif", "gif"}:
        return False
    return True


def build_video_request_effective_snapshot(
    *,
    request: Any,
    plan: VideoPlan,
    video_meta: Any,
    upscaling_options: VideoUpscalingOptions | None,
    upscaling_meta: Mapping[str, Any] | None,
    interpolation_options: VideoInterpolationOptions | None,
    interpolation_meta: Mapping[str, Any] | None,
    base_video_meta: Any = None,
    audio_input: bool = False,
    final_frame_count: int,
) -> dict[str, Any]:
    """Build an immutable snapshot of requested vs effective WAN video execution settings."""

    raw_video_options = getattr(request, "video_options", None)
    video_options: dict[str, Any] = dict(raw_video_options) if isinstance(raw_video_options, Mapping) else {}
    extras_raw = getattr(request, "extras", {})
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}

    requested_return_frames = parse_bool_value(
        extras.get("video_return_frames"),
        field="extras.video_return_frames",
        default=False,
    )
    requested_save_output = parse_bool_value(
        video_options.get("save_output"),
        field="video_options.save_output",
        default=False,
    )
    requested_save_metadata = parse_bool_value(
        video_options.get("save_metadata"),
        field="video_options.save_metadata",
        default=False,
    )
    requested_trim_to_audio = parse_bool_value(
        video_options.get("trim_to_audio"),
        field="video_options.trim_to_audio",
        default=False,
    )
    requested_pingpong = parse_bool_value(
        video_options.get("pingpong"),
        field="video_options.pingpong",
        default=False,
    )

    requested_interpolation_enabled = bool(interpolation_options is not None and interpolation_options.enabled)
    requested_interpolation_times = (
        int(interpolation_options.times)
        if interpolation_options is not None and interpolation_options.times is not None
        else None
    )
    requested_interpolation_toggle = bool(
        requested_interpolation_enabled and int(requested_interpolation_times or 0) > 1
    )
    requested_upscaling_toggle = bool(upscaling_options is not None and upscaling_options.enabled)
    requested_base_snapshot = bool(
        requested_save_output and (requested_upscaling_toggle or requested_interpolation_toggle)
    )

    video_saved = parse_bool_value(
        _export_meta_field(video_meta, key="saved"),
        field="video_meta.saved",
        default=False,
    )
    export_failed = bool(requested_save_output and not video_saved)
    effective_return_frames = bool(requested_return_frames or (not requested_save_output) or export_failed)

    upscaling_result_raw = (
        upscaling_meta.get("result")
        if isinstance(upscaling_meta, Mapping)
        else None
    )
    upscaling_applied = parse_bool_value(
        upscaling_result_raw.get("applied") if isinstance(upscaling_result_raw, Mapping) else None,
        field="video_upscaling.result.applied",
        default=False,
    )
    interpolation_result_raw = (
        interpolation_meta.get("result")
        if isinstance(interpolation_meta, Mapping)
        else None
    )
    interpolation_applied = parse_bool_value(
        interpolation_result_raw.get("applied") if isinstance(interpolation_result_raw, Mapping) else None,
        field="video_interpolation.result.applied",
        default=False,
    )

    base_snapshot_saved = parse_bool_value(
        _export_meta_field(base_video_meta, key="saved"),
        field="video_base_snapshot.saved",
        default=False,
    )

    format_effective = str(video_options.get("format") or "video/h264-mp4")
    pix_fmt_effective = str(video_options.get("pix_fmt") or "yuv420p")
    crf_effective = int(video_options.get("crf", 23) or 23)
    loop_count_effective = int(video_options.get("loop_count", 0) or 0)
    trim_to_audio_effective = bool(
        requested_trim_to_audio
        and requested_save_output
        and video_saved
        and audio_input
        and _container_supports_audio(format_effective)
    )
    save_metadata_effective = bool(
        requested_save_metadata
        and requested_save_output
        and video_saved
    )
    pingpong_effective = bool(
        requested_pingpong
        and requested_save_output
        and video_saved
    )

    requested_snapshot = {
        "video_options": {
            "format": video_options.get("format"),
            "pix_fmt": video_options.get("pix_fmt"),
            "crf": video_options.get("crf"),
            "loop_count": video_options.get("loop_count"),
            "pingpong": requested_pingpong,
            "save_output": requested_save_output,
            "save_metadata": requested_save_metadata,
            "trim_to_audio": requested_trim_to_audio,
        },
        "video_return_frames": requested_return_frames,
        "video_interpolation": (
            interpolation_options.as_dict() if interpolation_options is not None else {"enabled": False}
        ),
        "video_upscaling": (
            upscaling_options.as_dict() if upscaling_options is not None else {"enabled": False}
        ),
        "video_base_snapshot": {"requested": requested_base_snapshot},
        "input_geometry": {
            "width": int(getattr(request, "width", plan.width) or plan.width),
            "height": int(getattr(request, "height", plan.height) or plan.height),
            "fps": int(getattr(request, "fps", plan.fps) or plan.fps),
            "frames": int(getattr(request, "num_frames", final_frame_count) or final_frame_count),
        },
        "audio_input": bool(audio_input),
    }

    effective_snapshot = {
        "video_options": {
            "format": format_effective,
            "pix_fmt": pix_fmt_effective,
            "crf": crf_effective,
            "loop_count": loop_count_effective,
            "pingpong": pingpong_effective,
            "save_output": bool(video_saved),
            "save_metadata": save_metadata_effective,
            "trim_to_audio": trim_to_audio_effective,
        },
        "video_return_frames": effective_return_frames,
        "video_interpolation": (
            _snapshot_clone(interpolation_meta)
            if isinstance(interpolation_meta, Mapping)
            else {"enabled": requested_interpolation_enabled, "result": {"applied": interpolation_applied}}
        ),
        "video_upscaling": (
            _snapshot_clone(upscaling_meta)
            if isinstance(upscaling_meta, Mapping)
            else {"enabled": requested_upscaling_toggle, "result": {"applied": upscaling_applied}}
        ),
        "video_base_snapshot": _normalized_export_meta(base_video_meta),
        "video_export": _normalized_export_meta(video_meta),
        "output_geometry": {
            "width": int(plan.width),
            "height": int(plan.height),
            "fps": int(plan.fps),
            "frames": int(final_frame_count),
        },
    }

    toggle_effective_map = {
        "video_return_frames": {
            "requested": requested_return_frames,
            "effective": effective_return_frames,
        },
        "video_save_output": {
            "requested": requested_save_output,
            "effective": bool(video_saved),
        },
        "video_save_metadata": {
            "requested": requested_save_metadata,
            "effective": save_metadata_effective,
        },
        "video_trim_to_audio": {
            "requested": requested_trim_to_audio,
            "effective": trim_to_audio_effective,
        },
        "video_pingpong": {
            "requested": requested_pingpong,
            "effective": pingpong_effective,
        },
        "video_interpolation_enabled": {
            "requested": requested_interpolation_toggle,
            "effective": interpolation_applied,
        },
        "video_upscaling_enabled": {
            "requested": requested_upscaling_toggle,
            "effective": upscaling_applied,
        },
        "video_base_snapshot": {
            "requested": requested_base_snapshot,
            "effective": base_snapshot_saved,
        },
    }

    return {
        "request_snapshot": _snapshot_clone(requested_snapshot),
        "effective_snapshot": _snapshot_clone(effective_snapshot),
        "toggle_effective_map": _snapshot_clone(toggle_effective_map),
    }


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
    "prepare_base_snapshot_video_options",
    "build_video_request_effective_snapshot",
    "assemble_video_metadata",
    "build_video_result",
]
