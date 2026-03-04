"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared streaming helpers for image mode wrappers (txt2img/img2img).
Provides seed normalization, worker-thread execution (with smart runtime override propagation), sampling progress polling, decode normalization (including pre-decode cache flush + CPU-target decode transfer), and common `info` metadata building.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_seed_plan` (function): Normalize request seed + batch_total into (seed, all_seeds, subseeds, subseed_strength).
- `_run_inference_worker` (function): Run a callable in a daemon thread while propagating smart runtime overrides and capturing output/error/timings.
- `_iter_sampling_progress` (function): Poll `backend_state` and yield phase-aware progress snapshots (sampling + VAE encode/decode blocks) until a worker signals completion.
- `_decode_generation_output` (function): Normalize `GenerationResult`/tensor output into a list of PIL images and decode timing (pre-decode cache flush + CPU-target decode transfer).
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
    success: bool = False
    sampling_start: float | None = None
    sampling_end: float | None = None


def _run_inference_worker(
    *,
    name: str,
    fn: Callable[[], Any],
    runtime_overrides: Mapping[str, bool | None] | None = None,
) -> tuple["threading.Event", _WorkerOutcome]:
    import contextvars
    import threading
    import time

    from apps.backend.runtime.live_preview import (
        live_preview_method,
        preview_interval_steps,
        preview_runtime_overrides,
    )
    from apps.backend.runtime.memory.smart_offload import (
        current_smart_runtime_overrides,
    )

    if runtime_overrides is None:
        effective_runtime_overrides = current_smart_runtime_overrides()
    else:
        if not isinstance(runtime_overrides, Mapping):
            raise TypeError(
                "runtime_overrides must be Mapping[str, bool | None] when provided "
                f"(got {type(runtime_overrides).__name__})."
            )
        allowed_keys = {"smart_offload", "smart_fallback", "smart_cache"}
        unknown_keys = sorted(str(key) for key in runtime_overrides.keys() if key not in allowed_keys)
        if unknown_keys:
            raise ValueError(
                "runtime_overrides received unknown keys; expected only "
                f"{sorted(allowed_keys)} (got {unknown_keys})."
            )
        effective_runtime_overrides: dict[str, bool | None] = {}
        for key in ("smart_offload", "smart_fallback", "smart_cache"):
            raw_value = runtime_overrides.get(key, None)
            if raw_value is not None and not isinstance(raw_value, bool):
                raise TypeError(
                    "runtime_overrides values must be bool | None "
                    f"(key={key!r}, got {type(raw_value).__name__})."
                )
            effective_runtime_overrides[key] = raw_value

    outcome = _WorkerOutcome()
    done = threading.Event()
    effective_preview_interval = int(preview_interval_steps(default=0))
    effective_preview_method = live_preview_method()
    worker_context = contextvars.copy_context()

    def _worker_body() -> None:
        from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

        try:
            with smart_runtime_overrides(**effective_runtime_overrides):
                with preview_runtime_overrides(
                    interval_steps=effective_preview_interval,
                    method=effective_preview_method,
                ):
                    outcome.sampling_start = time.perf_counter()
                    outcome.output = fn()
                    outcome.success = True
        except BaseException as exc:  # noqa: BLE001
            outcome.error = exc
        finally:
            outcome.sampling_end = time.perf_counter()
            done.set()

    def _worker() -> None:
        worker_context.run(_worker_body)

    threading.Thread(target=_worker, name=name, daemon=True).start()
    return done, outcome


def _iter_sampling_progress(
    *,
    done: "threading.Event",
    outcome: _WorkerOutcome | None = None,
    poll_interval_s: float = 0.12,
) -> Iterator[tuple[str, int, int, int, int, float | None, int, int, int, int]]:
    import time

    from apps.backend.core.state import state as backend_state

    def _has_block_progress(*, block_index: int, block_total: int) -> bool:
        return block_total > 0 and 0 < block_index < block_total

    def _has_vae_progress(*, phase: str, block_index: int, block_total: int) -> bool:
        return phase in {"encode", "decode"} and block_total > 0 and block_index > 0

    t0 = time.perf_counter()
    last_snapshot = (-1, -1, -1, -1)
    last_vae_snapshot = ("", -1, -1)
    vae_phase_started_at: dict[str, float] = {}
    while True:
        try:
            snapshot = getattr(backend_state, "sampling_snapshot", None)
            if callable(snapshot):
                step, total, block_index, block_total = snapshot()
            else:
                step = int(getattr(backend_state, "sampling_step", 0) or 0)
                total = int(getattr(backend_state, "sampling_steps", 0) or 0)
                block_index = int(getattr(backend_state, "sampling_block_index", 0) or 0)
                block_total = int(getattr(backend_state, "sampling_block_total", 0) or 0)
        except Exception:  # noqa: BLE001
            step, total, block_index, block_total = 0, 0, 0, 0

        try:
            vae_snapshot = getattr(backend_state, "vae_progress_snapshot", None)
            if callable(vae_snapshot):
                vae_phase, vae_block_index, vae_block_total = vae_snapshot()
            else:
                vae_phase = str(getattr(backend_state, "vae_phase", "") or "")
                vae_block_index = int(getattr(backend_state, "vae_block_index", 0) or 0)
                vae_block_total = int(getattr(backend_state, "vae_block_total", 0) or 0)
        except Exception:  # noqa: BLE001
            vae_phase, vae_block_index, vae_block_total = "", 0, 0

        vae_phase = str(vae_phase or "").strip().lower()
        if vae_phase not in {"encode", "decode"}:
            vae_phase = ""
            vae_block_index = 0
            vae_block_total = 0

        total = max(0, int(total))
        step = max(0, min(int(step), total if total > 0 else int(step)))
        block_total = max(0, int(block_total))
        block_index = max(0, int(block_index))
        if block_total > 0:
            block_index = min(block_index, block_total)
        vae_block_total = max(0, int(vae_block_total))
        vae_block_index = max(0, int(vae_block_index))
        if vae_block_total > 0:
            vae_block_index = min(vae_block_index, vae_block_total)

        done_now = done.is_set()
        at_full_block_boundary = block_total > 0 and block_index >= block_total and step < total
        worker_succeeded = bool(done_now and outcome is not None and outcome.success and outcome.error is None)

        emit_step = step
        emit_block_index = block_index
        emit_block_total = block_total
        promote_completed_step = at_full_block_boundary
        if promote_completed_step:
            # Full block-boundary snapshots represent a completed step before the
            # backend tick lands. Promote them to the corresponding completed-step
            # snapshot instead of suppressing progress.
            emit_step = min(total, step + 1)
            emit_block_index = 0
            emit_block_total = 0

        current_snapshot = (emit_step, total, emit_block_index, emit_block_total)
        should_emit = (
            total > 0
            and (emit_step > 0 or emit_block_index > 0 or done_now)
            and current_snapshot != last_snapshot
        )
        if should_emit:
            elapsed = time.perf_counter() - t0
            completed_units = float(emit_step)
            if _has_block_progress(block_index=emit_block_index, block_total=emit_block_total):
                completed_units += float(emit_block_index) / float(emit_block_total)
            completed_units = min(float(total), completed_units)
            eta = (
                (elapsed * (float(total) - completed_units) / completed_units)
                if completed_units > 0.0
                else None
            )
            yield (
                "sampling",
                emit_step,
                total,
                emit_block_index,
                emit_block_total,
                eta,
                emit_step,
                total,
                emit_block_index,
                emit_block_total,
            )
            last_snapshot = current_snapshot

        current_vae_snapshot = (vae_phase, vae_block_index, vae_block_total)
        should_emit_vae = (
            _has_vae_progress(phase=vae_phase, block_index=vae_block_index, block_total=vae_block_total)
            and current_vae_snapshot != last_vae_snapshot
        )
        if should_emit_vae:
            now = time.perf_counter()
            if vae_phase not in vae_phase_started_at:
                vae_phase_started_at[vae_phase] = now
            elapsed_phase = max(0.0, now - vae_phase_started_at[vae_phase])
            completed_blocks = float(min(vae_block_total, vae_block_index))
            vae_eta = (
                (elapsed_phase * (float(vae_block_total) - completed_blocks) / completed_blocks)
                if completed_blocks > 0.0
                else None
            )
            yield (
                vae_phase,
                vae_block_index,
                vae_block_total,
                vae_block_index,
                vae_block_total,
                vae_eta,
                emit_step,
                total,
                emit_block_index,
                emit_block_total,
            )
            last_vae_snapshot = current_vae_snapshot

        if done_now:
            break

        time.sleep(float(poll_interval_s))


def _decode_generation_output(
    *,
    engine: Any,
    output: Any,
    task_label: str,
) -> tuple[list[object], float]:
    import gc
    import time

    import torch

    from apps.backend.runtime.memory import memory_management
    from apps.backend.runtime.memory.smart_offload_invariants import (
        enforce_smart_offload_post_decode_residency,
    )
    from apps.backend.runtime.processing.conditioners import decode_latent_batch
    from apps.backend.runtime.processing.datatypes import GenerationResult
    from apps.backend.runtime.pipeline_stages.image_io import latents_to_pil

    decoded_images: Any | None = None
    latents: Any = None
    metadata: dict[str, Any] = {}
    metadata_error: RuntimeError | None = None
    decode_engine = engine
    if isinstance(output, GenerationResult):
        latents = output.samples
        decoded_images = output.decoded
        if getattr(output, "decode_engine", None) is not None:
            decode_engine = output.decode_engine
        if not isinstance(output.metadata, dict):
            metadata_error = RuntimeError(
                f"{task_label} pipeline returned metadata as {type(output.metadata).__name__}; expected dict."
            )
        else:
            metadata = output.metadata
    else:
        latents = output
        decoded_images = None

    raw_cache_hit = metadata.get("conditioning_cache_hit", False)
    if not isinstance(raw_cache_hit, bool):
        metadata_error = RuntimeError(
            f"{task_label} pipeline metadata['conditioning_cache_hit'] must be bool; got {type(raw_cache_hit).__name__}."
        )

    decode_start = time.perf_counter()
    try:
        if metadata_error is None:
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
                gc.collect()
                memory_management.manager.soft_empty_cache(force=True)
                # Intentional exception: this egress path materializes decoded tensors on CPU
                # for immediate PIL conversion (`latents_to_pil`), independent of model offload policy.
                cpu_decode_target = memory_management.manager.cpu_device
                decoded = decode_latent_batch(
                    decode_engine,
                    latents,
                    target_device=cpu_decode_target,
                    stage=f"{task_label}.decode(pre)",
                )
                images = latents_to_pil(decoded)
    finally:
        enforce_smart_offload_post_decode_residency(
            decode_engine,
            stage=f"{task_label}.decode",
        )

    if metadata_error is not None:
        raise metadata_error

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
