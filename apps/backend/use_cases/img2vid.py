"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Img2vid orchestration for WAN22 (Diffusers pipeline or GGUF runtime).
Runs high/low stages, configures sampler settings, applies LoRAs, runs the shared video interpolation stage when requested, exports video, and yields
progress/result events.

Symbols (top-level; keep in sync; no ghosts):
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional frames) and attaches warnings.
- `_run_stage` (function): Runs a single Diffusers stage and returns its generated frames.
- `_yield_wan22_gguf_progress` (function): Maps WAN22 GGUF stream dict events into backend `ProgressEvent`s.
- `_parse_img2vid_chunk_options` (function): Parses and validates optional img2vid chunk controls from request extras.
- `_resolve_chunk_seed` (function): Resolves per-chunk seed semantics from base seed + selected chunk mode.
- `_blend_anchor_frame` (function): Blends previous chunk tail frame with init image for chunk re-anchoring.
- `run_img2vid` (function): Orchestrates img2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from apps.backend.core.requests import Img2VidRequest, InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.pipeline_stages.video import (
    apply_engine_loras,
    apply_video_interpolation,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
    read_video_interpolation_options,
    resolve_video_output_fps,
)


def _build_result_payload(
    *,
    engine: Any,
    result: Any,
    plan: VideoPlan,
    request: Img2VidRequest,
    video_meta: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(getattr(result, "metadata", {}) or {})

    user_return_frames = bool(plan.extras.get("video_return_frames", False))
    video_options = getattr(request, "video_options", None)
    save_output = bool(isinstance(video_options, Mapping) and bool(video_options.get("save_output", False)))

    video_saved = bool(isinstance(video_meta, dict) and bool(video_meta.get("saved")))
    export_failed = bool(save_output and not video_saved)

    effective_return_frames = bool(user_return_frames or (not save_output) or export_failed)

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


def _run_stage(
    pipe: Any,
    plan: VideoPlan,
    *,
    prompt: str,
    negative_prompt: str | None,
    init_image: Any | None,
) -> list[Any]:
    if pipe is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")
    output = pipe(
        image=init_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=plan.frames,
        num_inference_steps=plan.steps,
        height=plan.height,
        width=plan.width,
        guidance_scale=plan.guidance_scale,
    )
    if hasattr(output, "frames"):
        return list(output.frames[0])
    raise RuntimeError("img2vid pipeline returned no frames")


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


@dataclass(frozen=True)
class _Img2VidChunkOptions:
    chunk_frames: int
    overlap_frames: int
    anchor_alpha: float
    chunk_seed_mode: str


def _parse_img2vid_chunk_options(extras: Mapping[str, Any], *, total_frames: int) -> Optional[_Img2VidChunkOptions]:
    raw_chunk = extras.get("img2vid_chunk_frames")
    if raw_chunk in (None, "", 0):
        return None
    try:
        chunk_frames = int(raw_chunk)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_chunk_frames must be an integer, got: {raw_chunk!r}") from exc
    if chunk_frames < 9 or chunk_frames > 401:
        raise RuntimeError(f"img2vid_chunk_frames must be within [9, 401], got: {chunk_frames}")
    if (chunk_frames - 1) % 4 != 0:
        raise RuntimeError(f"img2vid_chunk_frames must satisfy 4n+1, got: {chunk_frames}")
    if chunk_frames >= int(total_frames):
        return None

    raw_overlap = extras.get("img2vid_overlap_frames", max(1, chunk_frames // 4))
    try:
        overlap_frames = int(raw_overlap)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_overlap_frames must be an integer, got: {raw_overlap!r}") from exc
    if overlap_frames < 0:
        raise RuntimeError(f"img2vid_overlap_frames must be >= 0, got: {overlap_frames}")
    if overlap_frames >= chunk_frames:
        raise RuntimeError(
            "img2vid_overlap_frames must be smaller than img2vid_chunk_frames "
            f"(overlap={overlap_frames} chunk={chunk_frames})"
        )

    raw_anchor = extras.get("img2vid_anchor_alpha", 0.2)
    try:
        anchor_alpha = float(raw_anchor)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_anchor_alpha must be a float, got: {raw_anchor!r}") from exc
    if anchor_alpha < 0.0 or anchor_alpha > 1.0:
        raise RuntimeError(f"img2vid_anchor_alpha must be within [0, 1], got: {anchor_alpha}")

    raw_seed_mode = str(extras.get("img2vid_chunk_seed_mode", "increment") or "").strip().lower()
    if raw_seed_mode not in {"fixed", "increment", "random"}:
        raise RuntimeError(
            f"img2vid_chunk_seed_mode must be one of ('fixed','increment','random'), got: {raw_seed_mode!r}"
        )

    return _Img2VidChunkOptions(
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        anchor_alpha=anchor_alpha,
        chunk_seed_mode=raw_seed_mode,
    )


def _resolve_chunk_seed(base_seed: Any, *, chunk_index: int, mode: str) -> Optional[int]:
    if not isinstance(base_seed, int) or base_seed < 0:
        return None
    if mode == "fixed":
        return int(base_seed)
    if mode == "increment":
        return int(base_seed) + int(chunk_index)
    if mode == "random":
        return None
    raise RuntimeError(f"Unsupported img2vid_chunk_seed_mode: {mode!r}")


def _blend_anchor_frame(previous_frame: Any, init_image: Any, *, alpha: float) -> Any:
    from PIL import Image  # type: ignore

    if not isinstance(previous_frame, Image.Image):
        return init_image
    if not isinstance(init_image, Image.Image):
        return previous_frame

    a = previous_frame.convert("RGB")
    b = init_image.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    return Image.blend(a, b, max(0.0, min(1.0, float(alpha))))


def run_img2vid(
    *,
    engine,
    comp,
    request: Img2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    if getattr(request, "init_image", None) is None:
        raise RuntimeError("img2vid requires 'init_image'")

    plan = build_video_plan(request)
    start = time.perf_counter()

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid")

    pipe = getattr(comp, "pipeline", None)
    high_model = getattr(comp, "pipeline_high", None)
    low_model = getattr(comp, "pipeline_low", None)

    if pipe is None and high_model is None and low_model is None:
        from apps.backend.runtime.families.wan22.config import build_wan22_gguf_run_config
        from apps.backend.runtime.families.wan22 import wan22 as gguf

        extras = dict(plan.extras) if isinstance(plan.extras, dict) else {}
        chunk_opts = _parse_img2vid_chunk_options(extras, total_frames=plan.frames)

        cfg = None
        frames: list[Any] | None = None

        if chunk_opts is None:
            cfg = build_wan22_gguf_run_config(
                request=request,
                device=getattr(comp, "device", "auto"),
                dtype=getattr(comp, "dtype", "fp16"),
                logger=logger,
            )

            for ev in gguf.stream_img2vid(cfg, logger=logger):
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
        else:
            if logger:
                logger.info(
                    "[img2vid] chunk mode enabled: chunk_frames=%d overlap=%d anchor_alpha=%.3f seed_mode=%s",
                    chunk_opts.chunk_frames,
                    chunk_opts.overlap_frames,
                    chunk_opts.anchor_alpha,
                    chunk_opts.chunk_seed_mode,
                )

            chunk_starts = list(
                range(
                    0,
                    int(plan.frames),
                    max(1, int(chunk_opts.chunk_frames - chunk_opts.overlap_frames)),
                )
            )
            if not chunk_starts:
                raise RuntimeError("img2vid chunk plan produced no chunk starts.")

            stitched: list[Any] = []
            base_seed = getattr(request, "seed", None)
            for chunk_index, chunk_start in enumerate(chunk_starts):
                chunk_seed = _resolve_chunk_seed(base_seed, chunk_index=chunk_index, mode=chunk_opts.chunk_seed_mode)
                if chunk_index == 0:
                    chunk_init = getattr(request, "init_image", None)
                else:
                    if chunk_start <= 0 or chunk_start > len(stitched):
                        raise RuntimeError(
                            f"img2vid chunk stitching invariant failed at chunk {chunk_index + 1}: "
                            f"start={chunk_start} stitched={len(stitched)}"
                        )
                    chunk_init = _blend_anchor_frame(
                        stitched[chunk_start - 1],
                        getattr(request, "init_image", None),
                        alpha=chunk_opts.anchor_alpha,
                    )

                chunk_request = Img2VidRequest(
                    task=request.task,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    sampler=request.sampler,
                    scheduler=request.scheduler,
                    seed=chunk_seed,
                    guidance_scale=request.guidance_scale,
                    batch_size=request.batch_size,
                    loras=request.loras,
                    extra_networks=request.extra_networks,
                    clip_skip=request.clip_skip,
                    metadata=request.metadata,
                    settings_revision=request.settings_revision,
                    smart_offload=request.smart_offload,
                    smart_fallback=request.smart_fallback,
                    smart_cache=request.smart_cache,
                    init_image=chunk_init,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    num_frames=chunk_opts.chunk_frames,
                    fps=request.fps,
                    motion_strength=request.motion_strength,
                    video_options=request.video_options,
                    extras=request.extras,
                )
                cfg = build_wan22_gguf_run_config(
                    request=chunk_request,
                    device=getattr(comp, "device", "auto"),
                    dtype=getattr(comp, "dtype", "fp16"),
                    logger=logger,
                )

                frames_chunk: list[Any] | None = None
                for ev in gguf.stream_img2vid(cfg, logger=logger):
                    if not isinstance(ev, dict):
                        raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
                    if ev.get("type") == "progress":
                        pe = _yield_wan22_gguf_progress(ev)
                        if pe is not None:
                            local = (
                                (float(pe.percent) / 100.0)
                                if pe.percent is not None
                                else 0.0
                            )
                            overall = ((float(chunk_index) + local) / float(len(chunk_starts))) * 100.0
                            yield ProgressEvent(
                                stage=f"run_chunk_{chunk_index + 1}",
                                percent=overall,
                                step=pe.step,
                                total_steps=pe.total_steps,
                                eta_seconds=pe.eta_seconds,
                            )
                        continue
                    if ev.get("type") == "result":
                        frames_chunk = list(ev.get("frames", []) or [])
                        break
                    raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")

                if not frames_chunk:
                    raise RuntimeError(f"WAN22 GGUF: chunk {chunk_index + 1}/{len(chunk_starts)} produced no frames")

                overlap_count = min(
                    int(chunk_opts.overlap_frames),
                    len(frames_chunk),
                    max(0, len(stitched) - chunk_start),
                )
                for overlap_index in range(overlap_count):
                    blend_alpha = float(overlap_index + 1) / float(overlap_count)
                    stitched[chunk_start + overlap_index] = _blend_anchor_frame(
                        stitched[chunk_start + overlap_index],
                        frames_chunk[overlap_index],
                        alpha=blend_alpha,
                    )

                needed = int(plan.frames) - int(chunk_start)
                for frame_index in range(overlap_count, min(len(frames_chunk), needed)):
                    absolute_index = int(chunk_start) + int(frame_index)
                    if absolute_index < len(stitched):
                        stitched[absolute_index] = frames_chunk[frame_index]
                    else:
                        stitched.append(frames_chunk[frame_index])

                yield ProgressEvent(
                    stage="run_chunks",
                    percent=((float(chunk_index + 1) / float(len(chunk_starts))) * 100.0),
                    message=f"Chunk {chunk_index + 1}/{len(chunk_starts)} complete",
                )

            frames = stitched[: int(plan.frames)]

        if not frames:
            raise RuntimeError("WAN22 GGUF: produced no frames")
        if cfg is None:
            raise RuntimeError("WAN22 GGUF: runtime config resolution failed (cfg is None).")

        vfi_options = read_video_interpolation_options(plan.extras)
        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="img2vid")

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        extra_meta: dict[str, Any] = dict(plan.extras) if isinstance(plan.extras, dict) else {}
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
            task="img2vid",
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

    apply_engine_loras(engine, logger)

    active_pipe_hi = high_model or pipe
    if active_pipe_hi is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    if wan_hi_opts and wan_hi_opts.lora_path and hasattr(active_pipe_hi, "load_lora_weights"):
        if logger:
            logger.info("[wan] loading high-stage LoRA: %s", wan_hi_opts.lora_path)
        active_pipe_hi.load_lora_weights(wan_hi_opts.lora_path)  # type: ignore[attr-defined]

    outcome_hi = configure_sampler(active_pipe_hi, plan, logger)

    yield ProgressEvent(stage="run_high", percent=5.0, message="Stage 1 (High Noise)")
    frames_hi = _run_stage(
        active_pipe_hi,
        plan,
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
    )

    active_pipe_lo = low_model or pipe
    outcome_lo = None
    frames = list(frames_hi)

    if active_pipe_lo is not None and frames_hi:
        wan_low_cfg = extras.get("wan_low")
        wan_opts = WanStageOptions.from_mapping(wan_low_cfg) if isinstance(wan_low_cfg, dict) else None
        if wan_opts and wan_opts.lora_path and hasattr(active_pipe_lo, "load_lora_weights"):
            if logger:
                logger.info("[wan] loading low-stage LoRA: %s", wan_opts.lora_path)
            active_pipe_lo.load_lora_weights(wan_opts.lora_path)  # type: ignore[attr-defined]

        outcome_lo = configure_sampler(active_pipe_lo, plan, logger)
        yield ProgressEvent(stage="run_low", percent=50.0, message="Stage 2 (Low Noise)")
        frames = _run_stage(
            active_pipe_lo,
            plan,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            init_image=frames_hi[-1],
        )

    vfi_options = read_video_interpolation_options(extras)
    if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
        yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
    frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
    plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="img2vid")

    extra_meta: dict[str, Any] = dict(extras) if isinstance(extras, dict) else {}
    if vfi_opts is not None:
        extra_meta["video_interpolation"] = vfi_opts
    if outcome_lo is not None:
        extra_meta["sampler_low"] = {
            "sampler_in": getattr(outcome_lo, "sampler_in", None),
            "scheduler_in": getattr(outcome_lo, "scheduler_in", None),
            "sampler": getattr(outcome_lo, "sampler_effective", None),
            "scheduler": getattr(outcome_lo, "scheduler_effective", None),
        }

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        outcome_hi,
        elapsed=elapsed,
        task="img2vid",
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
