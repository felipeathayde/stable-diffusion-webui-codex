"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for Codex video generation workflows.
Builds `VideoPlan`/`VideoResult`, applies LoRAs, configures sampler/scheduler, and assembles export metadata.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger used by video workflow helpers.
- `build_video_plan` (function): Normalizes request attributes into a `VideoPlan`.
- `apply_engine_loras` (function): Applies globally selected LoRAs to the engine (when supported).
- `configure_sampler` (function): Applies sampler/scheduler configuration to a component given a `VideoPlan`.
- `export_video` (function): Exports a frame sequence to a video file according to request options.
- `assemble_video_metadata` (function): Builds a metadata dict describing the generated video.
- `build_video_result` (function): Returns a `VideoResult` bundle for API/UI consumers.
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from apps.backend.runtime.adapters.lora import selections as lora_selections
from apps.backend.engines.util.schedulers import apply_sampler_scheduler, SamplerKind
from apps.backend.runtime.processing.datatypes import VideoPlan, VideoResult

logger = logging.getLogger(__name__)


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

    # Lazy import to keep workflow module dependency-light for non-LoRA users.
    from apps.backend.patchers.lora_apply import apply_loras_to_engine

    try:
        selections = lora_selections.get_selections()
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        raise RuntimeError(f"Failed to fetch LoRA selections: {exc}") from exc

    if not selections or not hasattr(engine, "codex_objects_after_applying_lora"):
        return None

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


def export_video(engine: Any, frames: Sequence[Any], plan: VideoPlan, video_options: Any) -> Any:
    if not hasattr(engine, "_maybe_export_video"):
        return None
    return engine._maybe_export_video(frames, fps=plan.fps, options=video_options)  # type: ignore[attr-defined]


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
    metadata: dict[str, Any] = {
        "engine": getattr(engine, "engine_id", "unknown"),
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
    metadata = assemble_video_metadata(
        engine,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        frame_count=len(frames),
        task=task,
        extra=extra,
        video_meta=video_meta,
    )
    return VideoResult(frames=list(frames), metadata=metadata, video_meta=video_meta)


__all__ = [
    "apply_engine_loras",
    "build_video_plan",
    "configure_sampler",
    "export_video",
    "assemble_video_metadata",
    "build_video_result",
]
