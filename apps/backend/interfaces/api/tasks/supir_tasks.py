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
Uses the shared inference gate when `CODEX_SINGLE_FLIGHT=1` and always marks tasks finished via `TaskEntry.mark_finished`.
Any cancel mode may abort while waiting on the inference gate; once running, only `immediate` interrupts the active SUPIR job.

Symbols (top-level; keep in sync; no ghosts):
- `run_supir_enhance_task` (function): Runs a SUPIR enhance task worker (single-image v1).
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any, Callable, Mapping

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import build_cancelled_task_error, build_public_task_error
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry

logger = logging.getLogger("backend.api.tasks.supir")


def run_supir_enhance_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    image_bytes: bytes,
    base_model_path: str,
    variant_ckpt_path: str,
    entry: TaskEntry,
    device: str,
    opts_get: Callable[..., Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    """Run a SUPIR enhance task (single image)."""

    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})

    def worker() -> None:
        acquired = False
        success = False
        try:
            if single_flight_enabled():
                push({"type": "status", "stage": "waiting_for_inference"})

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested),
            )
            if not acquired:
                entry.error = build_cancelled_task_error()
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)

            push({"type": "progress", "stage": "enhance", "percent": None, "step": None, "total_steps": None, "eta_seconds": None})
            if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                entry.error = build_cancelled_task_error()
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
            entry.error = build_public_task_error(err)
            success = False
        finally:
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            if acquired:
                try:
                    release_inference_gate()
                except Exception as exc:
                    logger.warning(
                        "inference gate release failed in supir enhance worker (task_id=%s): %s",
                        task_id,
                        exc,
                        exc_info=False,
                    )

    threading.Thread(target=worker, name=f"supir-enhance-task-{task_id}", daemon=True).start()


__all__ = ["run_supir_enhance_task"]
