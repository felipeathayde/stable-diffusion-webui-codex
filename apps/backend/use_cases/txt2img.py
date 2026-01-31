"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2img entry point using the staged pipeline runner.
Delegates latent generation to `Txt2ImgPipelineRunner` and provides a canonical event-emitting wrapper used by engines/orchestrator.

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
    from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides
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

    def _generate() -> GenerationResult:
        with smart_runtime_overrides(
            smart_offload=bool(getattr(proc, "smart_offload", False)),
            smart_fallback=bool(getattr(proc, "smart_fallback", False)),
            smart_cache=bool(getattr(proc, "smart_cache", False)),
        ):
            return generate_txt2img(
                processing=proc,
                conditioning=None,
                unconditional_conditioning=None,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
                prompts=prompts,
            )

    done, outcome = _run_inference_worker(name=f"{engine.engine_id}-txt2img-worker", fn=_generate)

    for step, total, eta in _iter_sampling_progress(done=done):
        pct = max(5.0, min(99.0, (step / total) * 100.0))
        yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)

    if outcome.error is not None:
        raise outcome.error

    images, decode_ms = _decode_generation_output(engine=engine, output=outcome.output, task_label="txt2img")

    all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
    seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

    extra_params: dict[str, object] = {}
    try:
        extra_params.update(last_extra_generation_params)
        extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
    except Exception:  # noqa: BLE001
        extra_params = getattr(proc, "extra_generation_params", {}) or {}

    timings: dict[str, float] = {"decode_ms": float(decode_ms)}
    try:
        if outcome.sampling_start is not None and outcome.sampling_end is not None:
            timings["sampling_ms"] = max(0.0, (outcome.sampling_end - outcome.sampling_start) * 1000.0)
    except Exception:  # noqa: BLE001
        pass

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

    post_cleanup = getattr(engine, "_post_txt2img_cleanup", None)
    if callable(post_cleanup):
        post_cleanup()

    yield ResultEvent(payload={"images": images, "info": json.dumps(info)})
