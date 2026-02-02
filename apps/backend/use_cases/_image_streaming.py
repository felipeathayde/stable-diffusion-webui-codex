"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared streaming helpers for image mode wrappers (txt2img/img2img).
Provides seed normalization, worker-thread execution, sampling progress polling, decode normalization, and common `info` metadata building.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_seed_plan` (function): Normalize request seed + batch_total into (seed, all_seeds, subseeds, subseed_strength).
- `_run_inference_worker` (function): Run a callable in a daemon thread while capturing output/error and sampling timings.
- `_iter_sampling_progress` (function): Poll `backend_state` and yield `ProgressEvent` updates until a worker signals completion.
- `_decode_generation_output` (function): Normalize `GenerationResult`/tensor output into a list of PIL images and decode timing.
- `_build_common_info` (function): Build the shared `info` dict for image tasks (engine/task/dims/seed/sampler/scheduler/prompts/extra/timings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence


_SEED_MASK = 0x7FFFFFFF


def _normalize_seed(value: int) -> int:
    return int(value) & _SEED_MASK


def _resolve_seed_plan(
    *,
    seed: int | None,
    batch_total: int,
) -> tuple[int, list[int], list[int], float]:
    """Resolve a request seed into per-image seeds for a batch.

    Rules (Codex semantics):
    - If seed is missing or < 0: generate a random seed per image.
    - Else: use seed+i for each image.
    """
    import secrets

    total = max(1, int(batch_total))
    raw_seed = None if seed is None else int(seed)
    if raw_seed is None or raw_seed < 0:
        seeds = [_normalize_seed(secrets.randbits(32)) for _ in range(total)]
        base = seeds[0]
    else:
        base = _normalize_seed(raw_seed)
        seeds = [_normalize_seed(base + idx) for idx in range(total)]

    subseeds = [-1 for _ in range(total)]
    return base, seeds, subseeds, 0.0


@dataclass(slots=True)
class _WorkerOutcome:
    output: Any = None
    error: BaseException | None = None
    sampling_start: float | None = None
    sampling_end: float | None = None


def _run_inference_worker(
    *,
    name: str,
    fn: Callable[[], Any],
) -> tuple["threading.Event", _WorkerOutcome]:
    import threading
    import time

    outcome = _WorkerOutcome()
    done = threading.Event()

    def _worker() -> None:
        try:
            outcome.sampling_start = time.perf_counter()
            outcome.output = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome.error = exc
        finally:
            outcome.sampling_end = time.perf_counter()
            done.set()

    threading.Thread(target=_worker, name=name, daemon=True).start()
    return done, outcome


def _iter_sampling_progress(
    *,
    done: "threading.Event",
    poll_interval_s: float = 0.12,
) -> Iterator[tuple[int, int, float | None]]:
    import time

    from apps.backend.core.state import state as backend_state

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
            yield step, total, eta
            last_step = step

        time.sleep(float(poll_interval_s))


def _decode_generation_output(
    *,
    engine: Any,
    output: Any,
    task_label: str,
) -> tuple[list[object], float]:
    import time

    import torch

    from apps.backend.runtime.processing.conditioners import decode_latent_batch
    from apps.backend.runtime.processing.datatypes import GenerationResult
    from apps.backend.runtime.pipeline_stages.image_io import latents_to_pil

    decoded_images: Any | None = None
    latents: Any = None
    if isinstance(output, GenerationResult):
        latents = output.samples
        decoded_images = output.decoded
    else:
        latents = output
        decoded_images = None

    decode_start = time.perf_counter()
    if decoded_images is not None:
        if isinstance(decoded_images, torch.Tensor):
            images = latents_to_pil(decoded_images)
        elif isinstance(decoded_images, list):
            try:
                from PIL import Image as _PILImage

                if not all(isinstance(img, _PILImage.Image) for img in decoded_images):
                    raise TypeError("decoded images are not PIL.Image.Image")
            except Exception as exc:
                raise RuntimeError(
                    f"{task_label} pipeline returned decoded images, but they are not a PIL image list"
                ) from exc
            images = decoded_images
        else:
            raise RuntimeError(
                f"{task_label} pipeline returned decoded images, expected torch.Tensor or list[PIL.Image.Image]"
            )
    else:
        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(
                f"{task_label} pipeline returned {type(latents).__name__}; expected torch.Tensor (latents)"
            )
        decoded = decode_latent_batch(engine, latents)
        images = latents_to_pil(decoded)

    decode_end = time.perf_counter()
    decode_ms = max(0.0, (decode_end - decode_start) * 1000.0)
    return list(images), decode_ms


def _build_common_info(
    *,
    engine_id: str,
    task: str,
    proc: Any,
    seed: int,
    all_seeds: Sequence[int],
    extra_params: Mapping[str, object],
    timings_ms: Mapping[str, float],
    mode_info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    info: dict[str, object] = {
        "engine": str(engine_id),
        "task": str(task),
        "width": int(getattr(proc, "width", 0) or 0),
        "height": int(getattr(proc, "height", 0) or 0),
        "steps": int(getattr(proc, "steps", 0) or 0),
        "guidance_scale": float(getattr(proc, "guidance_scale", 0.0) or 0.0),
        "sampler": (str(getattr(proc, "sampler_name", "")).strip() or None),
        "scheduler": (str(getattr(proc, "scheduler", "")).strip() or None),
        "seed": int(seed),
        "all_seeds": [int(s) for s in (all_seeds or [])],
    }

    prompt = str(getattr(proc, "primary_prompt", "") or "").strip()
    negative = str(getattr(proc, "primary_negative_prompt", "") or "").strip()
    if prompt:
        info["prompt"] = prompt
    if negative:
        info["negative_prompt"] = negative
    if extra_params:
        info["extra"] = dict(extra_params)
    if timings_ms:
        info["timings_ms"] = dict(timings_ms)
    if mode_info:
        info.update(dict(mode_info))
    return info
