"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task orchestration helpers for SUPIR endpoints.
Keeps `/api/supir/*` routers thin by centralizing the worker boilerplate:
status/progress/result/end/error + cancellation checks.

Symbols (top-level; keep in sync; no ghosts):
- `run_supir_enhance_task` (function): Runs a SUPIR enhance task worker (single-image v1).
"""

from __future__ import annotations

import io
import threading
from typing import Any, Callable, Mapping

from apps.backend.interfaces.api.task_registry import TaskEntry, tasks, tasks_lock


def run_supir_enhance_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    image_bytes: bytes,
    base_model_path: str,
    variant_ckpt_path: str,
    entry: TaskEntry,
    require_explicit_device: Callable[[dict[str, Any]], str],
    opts_get: Callable[..., Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    """Run a SUPIR enhance task (single image)."""

    loop = entry.loop

    def push(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(entry.queue.put_nowait, event)

    def mark_done(success: bool) -> None:
        def _set() -> None:
            if not entry.done.done():
                entry.done.set_result(success)

        loop.call_soon_threadsafe(_set)

    push({"type": "status", "stage": "queued"})

    try:
        require_explicit_device(payload)
    except Exception as err:
        entry.error = str(err)
        push({"type": "error", "message": entry.error})
        push({"type": "end"})
        mark_done(False)
        entry.schedule_cleanup(task_id, delay=0.0)
        with tasks_lock:
            tasks.pop(task_id, None)
        raise

    def worker() -> None:
        try:
            push({"type": "status", "stage": "running"})
            push({"type": "progress", "stage": "enhance", "percent": None, "step": None, "total_steps": None, "eta_seconds": None})
            if entry.cancel_requested and entry.cancel_mode == "immediate":
                entry.error = "cancelled"
                push({"type": "error", "message": "cancelled"})
                push({"type": "end"})
                mark_done(False)
                return

            from PIL import Image  # type: ignore

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            from apps.backend.use_cases.supir_enhance import supir_enhance_pil_image

            _ = supir_enhance_pil_image(
                img,
                payload=payload,
                base_model_path=base_model_path,
                variant_ckpt_path=variant_ckpt_path,
            )
            raise RuntimeError("SUPIR enhance returned no result")
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)

    threading.Thread(target=worker, name=f"supir-enhance-task-{task_id}", daemon=True).start()


__all__ = ["run_supir_enhance_task"]
