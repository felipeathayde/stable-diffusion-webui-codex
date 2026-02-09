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

Symbols (top-level; keep in sync; no ghosts):
- `encode_images` (function): Encode PIL images to base64 PNG payloads, optionally injecting PNG text metadata.
- `build_engine_options` (function): Build `engine_options` dict from request extras + options snapshot (TE/VAE overrides, Z-Image variant, core streaming).
- `run_image_task` (function): Run a generic image task worker (txt2img/img2img) using a `prepare(payload)` callback and orchestrator event stream.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from typing import Any, Callable, Mapping, Optional

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry, unregister_task

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
    if getattr(snap, "codex_core_streaming", False):
        engine_options["codex_core_streaming"] = True

    return engine_options


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
        entry.error = str(err)
        entry.mark_finished(success=False)
        unregister_task(task_id)
        raise

    def worker() -> None:
        acquired = False
        success = False
        try:
            if single_flight_enabled():
                push({"type": "status", "stage": "waiting_for_inference"})

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE),
            )
            if not acquired:
                entry.error = "cancelled"
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)

            preview_cfg = live_preview.build_task_config(opts_get)
            entry.last_preview_id_sent = 0

            engine_options = build_engine_options(req=req, opts_snapshot=opts_snapshot)

            from apps.backend.core.requests import ProgressEvent, ResultEvent

            with preview_cfg.runtime_overrides():
                for ev in orch.run(
                    task_type,
                    engine_key,
                    req,
                    model_ref=model_ref,
                    engine_options=engine_options,
                ):
                    if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                        entry.error = "cancelled"
                        return

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

                    if bool(opts_get("samples_save", True)):
                        save_generated_images(
                            payload_obj.get("images", []),
                            task=task_type,
                            info=info_dict,
                            metadata=generation_provenance,
                        )

                    result = {
                        "images": encode_images(payload_obj.get("images", []), metadata=generation_provenance),
                        "info": info_obj,
                    }
                    entry.result = {"status": "completed", "result": result}

            success = True
        except Exception as err:  # pragma: no cover - surfaces runtime errors
            try:
                from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exc

                _dump_exc(type(err), err, err.__traceback__, where="generation_image_worker", context={"task_id": task_id})
            except Exception:
                pass

            entry.error = str(err)
            success = False
        finally:
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            if acquired:
                try:
                    release_inference_gate()
                except Exception:
                    pass

    threading.Thread(target=worker, name=f"{task_type.value}-task-{task_id}", daemon=True).start()
