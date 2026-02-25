"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2img entry point using the staged pipeline runner.
Delegates latent generation to `Txt2ImgPipelineRunner` and provides a canonical event-emitting wrapper used by engines/orchestrator.
The wrapper executes sampling + decode + post-cleanup inside the same worker-thread envelope so model residency/offload policies remain single-owner per job.
Worker-thread smart runtime overrides are propagated through `_image_streaming._run_inference_worker(...)` and cleanup hooks always run in a `finally` block.

Symbols (top-level; keep in sync; no ghosts):
- `_logger` (constant): Module logger for the txt2img use case.
- `_RUNNER` (constant): Singleton `Txt2ImgPipelineRunner` instance.
- `generate_txt2img` (function): Runs the txt2img pipeline runner and returns a `GenerationResult` (samples + optional decoded output).
- `run_txt2img` (function): Canonical txt2img mode wrapper (progress polling + decode + result events).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from apps.backend.runtime.processing.datatypes import GenerationResult
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.diagnostics.pipeline_debug import pipeline_trace
from .txt2img_pipeline.runner import Txt2ImgPipelineRunner


_logger = logging.getLogger("backend.use_cases.txt2img")
_RUNNER = Txt2ImgPipelineRunner()


@pipeline_trace
def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
) -> GenerationResult:
    if not isinstance(processing, CodexProcessingTxt2Img):
        raise TypeError("generate_txt2img expects CodexProcessingTxt2Img")

    return _RUNNER.run(
        processing=processing,
        conditioning_data=conditioning,
        unconditional_data=unconditional_conditioning,
        seeds=seeds,
        subseeds=subseeds,
        subseed_strength=subseed_strength,
        prompts=prompts,
    )


def run_txt2img(*, engine, request) -> Iterator["InferenceEvent"]:
    """Run txt2img as a canonical event stream.

    This wrapper owns the mode-level concerns (seed defaults, progress polling, decode + result packaging).
    Engines should delegate here rather than implementing per-mode pipelines.
    """

    import json

    from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
    from apps.backend.engines.util.adapters import build_txt2img_processing
    from apps.backend.runtime.text_processing import last_extra_generation_params

    from ._image_streaming import (
        _build_common_info,
        _decode_generation_output,
        _iter_sampling_progress,
        _resolve_seed_plan,
        _run_inference_worker,
    )

    if not isinstance(request, Txt2ImgRequest):
        raise TypeError("run_txt2img expects Txt2ImgRequest")

    engine.ensure_loaded()

    proc = build_txt2img_processing(request)
    proc.sd_model = engine

    base_seed, seeds, subseeds, subseed_strength = _resolve_seed_plan(
        seed=getattr(request, "seed", None),
        batch_total=proc.batch_total,
    )
    proc.seed = base_seed
    proc.seeds = list(seeds)
    proc.subseed = -1
    proc.subseeds = list(subseeds)

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
    smart_flags = {
        "smart_offload": bool(getattr(proc, "smart_offload", False)),
        "smart_fallback": bool(getattr(proc, "smart_fallback", False)),
        "smart_cache": bool(getattr(proc, "smart_cache", False)),
    }

    def _generate() -> dict[str, object]:
        import time

        cleanup_targets: list[Any] = [engine]
        sampling_start = 0.0
        sampling_end = 0.0
        active_decode_engine: Any = engine

        try:
            sampling_start = time.perf_counter()
            output = generate_txt2img(
                processing=proc,
                conditioning=None,
                unconditional_conditioning=None,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
                prompts=prompts,
            )
            sampling_end = time.perf_counter()

            output_decode_engine = getattr(output, "decode_engine", None)
            active_decode_engine = output_decode_engine if output_decode_engine is not None else getattr(proc, "sd_model", None)
            if active_decode_engine is None:
                active_decode_engine = engine
            elif active_decode_engine is not engine:
                _logger.info("txt2img decode will use active pipeline model instance (swap/refiner path).")
            if active_decode_engine is not None and not any(existing is active_decode_engine for existing in cleanup_targets):
                cleanup_targets.append(active_decode_engine)

            images, decode_ms = _decode_generation_output(
                engine=active_decode_engine,
                output=output,
                task_label="txt2img",
            )

            all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
            seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

            extra_params: dict[str, object] = {}
            try:
                extra_params.update(last_extra_generation_params)
                extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
            except Exception:  # noqa: BLE001
                extra_params = getattr(proc, "extra_generation_params", {}) or {}

            timings: dict[str, float] = {
                "sampling_ms": max(0.0, (sampling_end - sampling_start) * 1000.0),
                "decode_ms": float(decode_ms),
            }

            mode_info: dict[str, object] = {}
            if bool(getattr(proc, "enable_hr", False)):
                try:
                    mode_info["hires"] = getattr(proc, "hires", None).as_dict()
                except Exception:  # noqa: BLE001
                    pass

            info = _build_common_info(
                engine_id=engine.engine_id,
                task="txt2img",
                proc=proc,
                seed=seed_value,
                all_seeds=all_seeds,
                extra_params=extra_params,
                timings_ms=timings,
                mode_info=mode_info,
            )
            return {"images": images, "info": json.dumps(info)}
        finally:
            processing_model = getattr(proc, "sd_model", None)
            if processing_model is not None and not any(existing is processing_model for existing in cleanup_targets):
                cleanup_targets.append(processing_model)
            for target in cleanup_targets:
                post_cleanup = getattr(target, "_post_txt2img_cleanup", None)
                if callable(post_cleanup):
                    post_cleanup()

    done, outcome = _run_inference_worker(
        name=f"{engine.engine_id}-txt2img-worker",
        fn=_generate,
        runtime_overrides=smart_flags,
    )

    for step, total, block_index, block_total, eta in _iter_sampling_progress(done=done):
        completed_units = float(step)
        if block_total > 0 and step < total:
            completed_units += float(block_index) / float(block_total)
        progress_percent = (min(float(total), completed_units) / float(total)) * 100.0
        pct = max(5.0, min(99.0, progress_percent))

        if block_total > 0 and step < total:
            message = (
                f"Sampling step {min(step + 1, total)}/{total} "
                f"(block {block_index}/{block_total})"
            )
        else:
            message = f"Sampling step {step}/{total}"

        yield ProgressEvent(
            stage="sampling",
            percent=pct,
            step=step,
            total_steps=total,
            eta_seconds=eta,
            message=message,
            data={"block_index": int(block_index), "block_total": int(block_total)},
        )

    if outcome.error is not None:
        raise outcome.error

    payload = outcome.output
    if not isinstance(payload, dict):
        raise RuntimeError(
            "txt2img worker returned invalid payload type; expected dict with 'images' and 'info'. "
            f"Got {type(payload).__name__}."
        )
    images = payload.get("images")
    info = payload.get("info")
    if not isinstance(images, list):
        raise RuntimeError("txt2img worker payload field 'images' must be list.")
    if not isinstance(info, str):
        raise RuntimeError("txt2img worker payload field 'info' must be JSON string.")
    yield ResultEvent(payload={"images": images, "info": info})
