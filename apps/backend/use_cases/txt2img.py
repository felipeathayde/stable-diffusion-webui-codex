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
- `generate_txt2img` (function): Runs the txt2img pipeline runner for the provided processing context and prompts.
- `run_txt2img` (function): Canonical txt2img mode wrapper (progress + result events).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

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
):
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
    import secrets
    import threading
    import time

    import torch

    from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
    from apps.backend.core.state import state as backend_state
    from apps.backend.engines.util.adapters import build_txt2img_processing
    from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides
    from apps.backend.runtime.processing.conditioners import decode_latent_batch
    from apps.backend.runtime.workflows.image_io import latents_to_pil
    from apps.backend.runtime.text_processing import last_extra_generation_params

    if not isinstance(request, Txt2ImgRequest):
        raise TypeError("run_txt2img expects Txt2ImgRequest")

    engine.ensure_loaded()

    raw_seed = int(getattr(request, "seed", -1) or -1)
    if raw_seed < 0:
        raw_seed = secrets.randbits(32) & 0x7FFFFFFF

    proc = build_txt2img_processing(request)
    proc.sd_model = engine
    proc.seed = raw_seed
    proc.seeds = [raw_seed]
    proc.subseed = -1
    proc.subseeds = [-1]

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
    seeds = [raw_seed]
    subseeds = [-1]
    subseed_strength = 0.0

    result: dict[str, Any] = {"latents": None, "error": None}
    sampling_times: dict[str, float | None] = {"start": None, "end": None}
    done = threading.Event()

    def _worker() -> None:
        try:
            sampling_times["start"] = time.perf_counter()
            with smart_runtime_overrides(
                smart_offload=bool(getattr(proc, "smart_offload", False)),
                smart_fallback=bool(getattr(proc, "smart_fallback", False)),
                smart_cache=bool(getattr(proc, "smart_cache", False)),
            ):
                result["latents"] = generate_txt2img(
                    processing=proc,
                    conditioning=None,
                    unconditional_conditioning=None,
                    seeds=seeds,
                    subseeds=subseeds,
                    subseed_strength=subseed_strength,
                    prompts=prompts,
                )
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            sampling_times["end"] = time.perf_counter()
            done.set()

    threading.Thread(target=_worker, name=f"{engine.engine_id}-txt2img-worker", daemon=True).start()

    t0 = time.perf_counter()
    last_step = -1
    while not done.is_set():
        try:
            step = int(getattr(backend_state, "sampling_step", 0) or 0)
            total = int(getattr(backend_state, "sampling_steps", 0) or 0)
        except Exception:  # noqa: BLE001
            step, total = 0, 0

        if total > 0 and step != last_step:
            elapsed = time.perf_counter() - t0
            eta = (elapsed * (total - step) / max(step, 1)) if step > 0 else None
            pct = max(5.0, min(99.0, (step / total) * 100.0))
            yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)
            last_step = step

        time.sleep(0.12)

    if result["error"] is not None:
        raise result["error"]

    latents = result["latents"]
    if not isinstance(latents, torch.Tensor):
        raise RuntimeError(f"txt2img returned {type(latents).__name__}; expected torch.Tensor (latents)")

    decode_start = time.perf_counter()
    if getattr(latents, "_already_decoded", False):
        decoded = latents
    else:
        decoded = decode_latent_batch(engine, latents)
    images = latents_to_pil(decoded)
    decode_end = time.perf_counter()

    extra_params: dict[str, object] = {}
    try:
        extra_params.update(last_extra_generation_params)
        extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
    except Exception:  # noqa: BLE001
        extra_params = getattr(proc, "extra_generation_params", {}) or {}

    info: dict[str, object] = {
        "engine": engine.engine_id,
        "task": "txt2img",
        "width": int(proc.width),
        "height": int(proc.height),
        "steps": int(proc.steps),
        "guidance_scale": float(getattr(proc, "guidance_scale", 0.0) or 0.0),
        "sampler": (str(getattr(proc, "sampler_name", "")).strip() or None),
        "scheduler": (str(getattr(proc, "scheduler", "")).strip() or None),
    }
    if getattr(proc, "prompt", None):
        info["prompt"] = str(getattr(proc, "prompt", ""))
    if getattr(proc, "negative_prompt", None):
        info["negative_prompt"] = str(getattr(proc, "negative_prompt", ""))
    info["seed"] = int(raw_seed)
    if extra_params:
        info["extra"] = extra_params

    timings: dict[str, float] = {}
    try:
        if sampling_times["start"] is not None and sampling_times["end"] is not None:
            timings["sampling_ms"] = max(0.0, (sampling_times["end"] - sampling_times["start"]) * 1000.0)
        timings["decode_ms"] = max(0.0, (decode_end - decode_start) * 1000.0)
        info["timings_ms"] = timings
    except Exception:  # noqa: BLE001
        pass

    post_cleanup = getattr(engine, "_post_txt2img_cleanup", None)
    if callable(post_cleanup):
        post_cleanup()

    yield ResultEvent(payload={"images": images, "info": json.dumps(info)})
