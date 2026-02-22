"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared task orchestration helpers for generation endpoints.
Centralizes image encoding, engine-options building, and the common task worker loop (status/progress/result/end/error) so routers stay thin.
Uses the shared inference gate when `CODEX_SINGLE_FLIGHT=1` and always marks tasks finished via `TaskEntry.mark_finished` (stream termination + cleanup).
Inference-gate wait cancellation is mode-agnostic (both `immediate` and `after_current` cancel before start); once running, only immediate mode interrupts in-flight orchestration.
When `CODEX_TRACE_CONTRACT=1`, emits prompt-redacted contract-trace JSONL events (`prompt_hash` only) for prepare/run/progress/result/error/end stages.

Symbols (top-level; keep in sync; no ghosts):
- `encode_images` (function): Encode PIL images to base64 PNG payloads, optionally injecting PNG text metadata.
- `build_engine_options` (function): Build `engine_options` dict from request extras + options snapshot (TE/VAE overrides, Z-Image variant, core streaming).
- `resolve_request_smart_flags` (function): Parse/validate per-request smart flags (`smart_offload`/`smart_fallback`/`smart_cache`) as strict booleans.
- `force_runtime_memory_cleanup` (function): Best-effort runtime cleanup used on worker error paths (orchestrator cache + memory manager + GGUF cache + CUDA cache).
- `_format_parameters_infotext` (function): Serializes generation `info` dicts into A1111-compatible infotext for PNG `parameters`.
- `_build_png_metadata` (function): Builds PNG text chunks (`parameters` + provenance) for saved/API-encoded images.
- `run_image_task` (function): Run a generic image task worker (txt2img/img2img) using a `prepare(payload)` callback and orchestrator event stream.
"""

from __future__ import annotations

import base64
import contextlib
import gc
import io
import json
import logging
import math
import threading
from typing import Any, Callable, Mapping, Optional

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import public_task_error_message
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry, unregister_task
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.runtime.diagnostics.contract_trace import error_meta
from apps.backend.runtime.diagnostics.contract_trace import emit_event as emit_contract_trace
from apps.backend.runtime.diagnostics.contract_trace import hash_request_prompt

logger = logging.getLogger("backend.api.tasks.generation")


def encode_images(images: Any, *, metadata: Optional[Mapping[str, str]] = None) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
    encoded: list[dict[str, str]] = []
    for img in images or []:
        if img is None:
            continue
        buf = io.BytesIO()
        pnginfo = None
        use_metadata = False
        try:
            from PIL import PngImagePlugin  # type: ignore

            def _add_text(key: object, value: object) -> None:
                nonlocal pnginfo, use_metadata
                if not isinstance(key, str) or not isinstance(value, str):
                    return
                if not value:
                    return
                if pnginfo is None:
                    pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text(key, value)
                use_metadata = True

            info_items = getattr(img, "info", None)
            if isinstance(info_items, dict):
                for key, value in info_items.items():
                    _add_text(key, value)
            if metadata:
                for key, value in metadata.items():
                    _add_text(key, value)
        except Exception:
            pnginfo = None
            use_metadata = False

        img.save(buf, format="PNG", pnginfo=(pnginfo if use_metadata else None))
        encoded.append(
            {
                "format": "png",
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        )
    return encoded


def build_engine_options(*, req: Any, opts_snapshot: Callable[[], Any]) -> dict[str, object]:
    engine_options: dict[str, object] = {}
    extras = getattr(req, "extras", {}) or {}

    te_override = extras.get("text_encoder_override")
    if isinstance(te_override, dict):
        engine_options["text_encoder_override"] = dict(te_override)

    vae_path_from_extras = extras.get("vae_path")
    if isinstance(vae_path_from_extras, str) and vae_path_from_extras.strip():
        engine_options["vae_path"] = vae_path_from_extras.strip()
    engine_options["vae_source"] = "external" if "vae_path" in engine_options else "built_in"

    tenc_path_from_extras = extras.get("tenc_path")
    if isinstance(tenc_path_from_extras, str) and tenc_path_from_extras.strip():
        engine_options["tenc_path"] = tenc_path_from_extras.strip()
    elif isinstance(tenc_path_from_extras, list):
        resolved: list[str] = []
        for item in tenc_path_from_extras:
            if isinstance(item, str) and item.strip():
                resolved.append(item.strip())
        if resolved:
            engine_options["tenc_path"] = resolved

    engine_options["tenc_source"] = (
        "external" if ("tenc_path" in engine_options or "text_encoder_override" in engine_options) else "built_in"
    )

    zimage_variant = extras.get("zimage_variant")
    if isinstance(zimage_variant, str) and zimage_variant.strip():
        engine_options["zimage_variant"] = zimage_variant.strip()

    # Pass streaming option from settings to engine (no model-part fallbacks).
    snap = opts_snapshot()
    core_streaming_enabled = parse_bool_value(
        getattr(snap, "codex_core_streaming", None),
        field="options.codex_core_streaming",
        default=False,
    )
    if core_streaming_enabled:
        engine_options["codex_core_streaming"] = True

    return engine_options


def resolve_request_smart_flags(req: Any) -> tuple[bool, bool, bool]:
    values: dict[str, bool] = {}
    for field_name in ("smart_offload", "smart_fallback", "smart_cache"):
        field_value = getattr(req, field_name, False)
        if not isinstance(field_value, bool):
            raise RuntimeError(
                f"Invalid request field '{field_name}': expected boolean, got {type(field_value).__name__}."
            )
        values[field_name] = field_value
    return values["smart_offload"], values["smart_fallback"], values["smart_cache"]


def force_runtime_memory_cleanup(*, reason: str, orch: Any | None = None) -> None:
    clear_cache = getattr(orch, "clear_cache", None)
    if callable(clear_cache):
        try:
            clear_cache()
        except Exception as exc:
            logger.warning(
                "Failed to clear orchestrator cache during runtime cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )

    try:
        from apps.backend.runtime.memory import memory_management as memory_state
    except Exception as exc:
        logger.warning(
            "Runtime memory-manager import failed during cleanup (%s): %s",
            reason,
            exc,
            exc_info=False,
        )
    else:
        try:
            memory_state.manager.unload_all_models()
        except Exception as exc:
            logger.warning(
                "Runtime unload_all_models failed during cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )
        try:
            memory_state.manager.soft_empty_cache(force=True)
        except Exception as exc:
            logger.warning(
                "Runtime soft_empty_cache failed during cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )

    gguf_clear_cache: Callable[[], None] | None = None
    try:
        from apps.backend.runtime.ops.operations_gguf import clear_cache as gguf_clear_cache
    except Exception:
        gguf_clear_cache = None

    if callable(gguf_clear_cache):
        try:
            gguf_clear_cache()
        except Exception as exc:
            logger.warning(
                "Failed to clear GGUF cache during runtime cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )

    with contextlib.suppress(Exception):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.warning(
            "Torch cache cleanup failed during runtime cleanup (%s): %s",
            reason,
            exc,
            exc_info=False,
        )

    logger.info("Runtime memory cleanup completed (%s).", reason)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("+"):
            text = text[1:]
        if text.startswith("-"):
            return None
        if not text.isdigit():
            return None
        try:
            return int(text)
        except Exception:
            return None
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out
    return None


def _format_number(value: object) -> str | None:
    integer = _as_int(value)
    if integer is not None:
        return str(integer)
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:g}"


def _format_parameters_infotext(info_obj: object) -> str:
    if not isinstance(info_obj, dict):
        return _as_text(info_obj)

    info = dict(info_obj)
    lines: list[str] = []
    kv_entries: list[str] = []
    seen_keys: set[str] = set()

    def add_kv(label: str, value: object) -> None:
        if not isinstance(label, str):
            return
        key = label.strip()
        if not key:
            return
        text = _as_text(value)
        if not text:
            return
        normalized = key.lower()
        if normalized in seen_keys:
            return
        seen_keys.add(normalized)
        kv_entries.append(f"{key}: {text}")

    prompt = _as_text(info.get("prompt", ""))
    if prompt:
        lines.append(prompt)
    negative_prompt = _as_text(info.get("negative_prompt", ""))
    if negative_prompt:
        lines.append(f"Negative prompt: {negative_prompt}")

    steps = _as_int(info.get("steps"))
    if steps is not None and steps >= 0:
        add_kv("Steps", steps)

    sampler = _as_text(info.get("sampler_name", "") or info.get("sampler", ""))
    if sampler:
        add_kv("Sampler", sampler)

    scheduler = _as_text(info.get("schedule_type", "") or info.get("scheduler", ""))
    if scheduler:
        add_kv("Schedule type", scheduler)

    cfg_scale = _format_number(info.get("cfg_scale", info.get("guidance_scale")))
    if cfg_scale is not None:
        add_kv("CFG scale", cfg_scale)

    seed = _as_int(info.get("seed"))
    if seed is not None:
        add_kv("Seed", seed)

    width = _as_int(info.get("width"))
    height = _as_int(info.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        add_kv("Size", f"{width}x{height}")

    model_hash = _as_text(info.get("model_hash", ""))
    if model_hash:
        add_kv("Model hash", model_hash)

    model_name = _as_text(info.get("model", "") or info.get("sd_model_checkpoint", ""))
    if model_name:
        add_kv("Model", model_name)

    vae_name = _as_text(info.get("vae", "") or info.get("vae_name", ""))
    if vae_name:
        add_kv("VAE", vae_name)

    clip_skip = _as_int(info.get("clip_skip"))
    if clip_skip is not None and clip_skip >= 0:
        add_kv("Clip skip", clip_skip)

    denoise = _format_number(
        info.get("denoising_strength", info.get("denoise_strength", info.get("denoiseStrength")))
    )
    if denoise is not None:
        add_kv("Denoising strength", denoise)

    rng = _as_text(info.get("rng", "") or info.get("rng_source", ""))
    if rng:
        add_kv("RNG", rng)

    extra = info.get("extra")
    if isinstance(extra, dict):
        for raw_key, raw_value in extra.items():
            if raw_value is None:
                continue
            key_text = _as_text(raw_key)
            if not key_text:
                continue
            if isinstance(raw_value, (dict, list)):
                try:
                    value_text = json.dumps(raw_value, ensure_ascii=False)
                except Exception:
                    value_text = _as_text(raw_value)
            else:
                value_text = _as_text(raw_value)
                if "," in value_text or "\n" in value_text or "\r" in value_text:
                    value_text = json.dumps(value_text, ensure_ascii=False)
            if not value_text:
                continue
            add_kv(key_text, value_text)

    if kv_entries:
        lines.append(", ".join(kv_entries))

    return "\n".join(lines).strip()


def _build_png_metadata(info_obj: object, *, generation_provenance: Mapping[str, str]) -> dict[str, str]:
    metadata: dict[str, str] = {}

    parameters = _format_parameters_infotext(info_obj)
    if parameters:
        metadata["parameters"] = parameters

    for key, value in generation_provenance.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text or not value_text:
            continue
        metadata.setdefault(key_text, value_text)
    return metadata


def run_image_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    entry: TaskEntry,
    device: str,
    task_type: Any,
    prepare: Callable[[dict[str, Any]], tuple[Any, str, Optional[str]]],
    orch: Any,
    ensure_default_engines_registered: Callable[[], None],
    live_preview: Any,
    opts_get: Callable[..., Any],
    opts_snapshot: Callable[[], Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})
    try:
        ensure_default_engines_registered()
        req, engine_key, model_ref = prepare(payload)
    except Exception as err:
        emit_contract_trace(
            task_id=task_id,
            mode=str(getattr(task_type, "value", "unknown")),
            stage="prepare",
            action="error",
            component="router",
            device=device,
            strict=True,
            fallback_enabled=False,
            fallback_used=False,
            prompt_hash_value="",
            meta=error_meta(err),
        )
        entry.error = public_task_error_message(err)
        entry.mark_finished(success=False)
        unregister_task(task_id)
        raise

    mode = str(getattr(task_type, "value", "unknown"))
    prompt_hash_value = hash_request_prompt(req)
    smart_offload, smart_fallback, smart_cache = resolve_request_smart_flags(req)
    fallback_enabled = smart_fallback
    storage_dtype = getattr(req, "core_dtype", None)
    compute_dtype = getattr(req, "core_compute_dtype", None)

    emit_contract_trace(
        task_id=task_id,
        mode=mode,
        stage="prepare",
        action="ready",
        component="router",
        device=device,
        storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
        compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
        strict=True,
        fallback_enabled=fallback_enabled,
        fallback_used=False,
        prompt_hash_value=prompt_hash_value,
        meta={"engine_key": engine_key},
    )

    def worker() -> None:
        acquired = False
        success = False
        try:
            if single_flight_enabled():
                push({"type": "status", "stage": "waiting_for_inference"})
                emit_contract_trace(
                    task_id=task_id,
                    mode=mode,
                    stage="waiting_for_inference",
                    action="wait",
                    component="inference_gate",
                    device=device,
                    storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                    compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                    strict=True,
                    fallback_enabled=fallback_enabled,
                    fallback_used=False,
                    prompt_hash_value=prompt_hash_value,
                )

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested),
            )
            if not acquired:
                entry.error = "cancelled"
                emit_contract_trace(
                    task_id=task_id,
                    mode=mode,
                    stage="inference_gate",
                    action="cancelled",
                    component="inference_gate",
                    device=device,
                    storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                    compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                    strict=True,
                    fallback_enabled=fallback_enabled,
                    fallback_used=False,
                    prompt_hash_value=prompt_hash_value,
                )
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="running",
                action="start",
                component="orchestrator",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=False,
                prompt_hash_value=prompt_hash_value,
            )

            preview_cfg = live_preview.build_task_config(opts_get)
            entry.last_preview_id_sent = 0

            engine_options = build_engine_options(req=req, opts_snapshot=opts_snapshot)

            from apps.backend.core.requests import ProgressEvent, ResultEvent
            from apps.backend.core.state import state as backend_state
            from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

            cancelled_immediate = False
            backend_state.clear_progress_snapshot()
            with preview_cfg.runtime_overrides(), smart_runtime_overrides(
                smart_offload=smart_offload,
                smart_fallback=smart_fallback,
                smart_cache=smart_cache,
            ):
                for ev in orch.run(
                    task_type,
                    engine_key,
                    req,
                    model_ref=model_ref,
                    engine_options=engine_options,
                ):
                    if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                        if not cancelled_immediate:
                            entry.error = "cancelled"
                        cancelled_immediate = True
                        # Keep draining orchestrator events so generator/use-case finalizers run
                        # before this worker marks the task done and releases the inference gate.
                        continue

                    if isinstance(ev, ProgressEvent):
                        evt: dict[str, Any] = {
                            "type": "progress",
                            "stage": ev.stage,
                            "percent": ev.percent,
                            "step": ev.step,
                            "total_steps": ev.total_steps,
                            "eta_seconds": ev.eta_seconds,
                        }
                        live_preview.maybe_attach_to_progress_event(evt, entry, config=preview_cfg)
                        push(evt)
                        emit_contract_trace(
                            task_id=task_id,
                            mode=mode,
                            stage=str(ev.stage or "progress"),
                            action="progress",
                            component="orchestrator",
                            device=device,
                            storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                            compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                            strict=True,
                            fallback_enabled=fallback_enabled,
                            fallback_used=False,
                            prompt_hash_value=prompt_hash_value,
                            meta={
                                "step": ev.step,
                                "total_steps": ev.total_steps,
                                "percent": ev.percent,
                            },
                        )
                        continue

                    if not isinstance(ev, ResultEvent):
                        continue

                    payload_obj = ev.payload or {}
                    info_raw = payload_obj.get("info", "{}")
                    try:
                        info_obj = json.loads(info_raw)
                    except Exception:
                        info_obj = info_raw
                    info_dict = info_obj if isinstance(info_obj, dict) else None
                    if isinstance(info_obj, dict):
                        for key, value in generation_provenance.items():
                            info_obj.setdefault(key, value)
                    png_metadata = _build_png_metadata(info_obj, generation_provenance=generation_provenance)

                    if parse_bool_value(
                        opts_get("samples_save", True),
                        field="options.samples_save",
                        default=True,
                    ):
                        save_generated_images(
                            payload_obj.get("images", []),
                            task=task_type,
                            info=info_dict,
                            metadata=png_metadata,
                        )

                    result = {
                        "images": encode_images(payload_obj.get("images", []), metadata=png_metadata),
                        "info": info_obj,
                    }
                    entry.result = {"status": "completed", "result": result}
                    emit_contract_trace(
                        task_id=task_id,
                        mode=mode,
                        stage="result",
                        action="emit",
                        component="orchestrator",
                        device=device,
                        storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                        compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                        strict=True,
                        fallback_enabled=fallback_enabled,
                        fallback_used=False,
                        prompt_hash_value=prompt_hash_value,
                        meta={"image_count": len(payload_obj.get("images", []) or [])},
                    )

            success = not cancelled_immediate
        except Exception as err:  # pragma: no cover - surfaces runtime errors
            try:
                from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exc

                _dump_exc(type(err), err, err.__traceback__, where="generation_image_worker", context={"task_id": task_id})
            except Exception:
                pass
            try:
                from apps.backend.core.exceptions import EngineExecutionError

                if isinstance(err, EngineExecutionError):
                    logger.error(
                        "EngineExecutionError in generation_image_worker "
                        "(task_id=%s mode=%s engine=%s): %s",
                        task_id,
                        mode,
                        engine_key,
                        err,
                    )
            except Exception:
                pass

            force_runtime_memory_cleanup(
                reason=f"{mode}:worker_error",
                orch=orch,
            )
            entry.error = public_task_error_message(err)
            fallback_used = fallback_enabled and ("fallback" in str(err).lower())
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="error",
                action="error",
                component="orchestrator",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=fallback_used,
                prompt_hash_value=prompt_hash_value,
                meta=error_meta(err),
            )
            success = False
        finally:
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="end",
                action="finish",
                component="task",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=False,
                prompt_hash_value=prompt_hash_value,
                meta={"success": success},
            )
            if acquired:
                try:
                    release_inference_gate()
                except Exception:
                    pass

    threading.Thread(target=worker, name=f"{task_type.value}-task-{task_id}", daemon=True).start()
