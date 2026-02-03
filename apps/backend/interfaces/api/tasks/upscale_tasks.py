"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task orchestration helpers for upscaler endpoints.
Keeps `/api/upscale` and `/api/upscalers/download` routers thin by centralizing the worker boilerplate:
status/progress/result/end/error + cancellation checks.
Uses the shared inference gate for the upscale worker when `CODEX_SINGLE_FLIGHT=1` and always marks tasks finished via `TaskEntry.mark_finished`.

Symbols (top-level; keep in sync; no ghosts):
- `run_upscale_task` (function): Runs an upscale task worker (single-image v1).
- `run_upscaler_download_task` (function): Runs an HF download task worker for curated upscaler weights.
"""

from __future__ import annotations

import io
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.task_registry import TaskEntry

logger = logging.getLogger("backend.api.tasks.upscale")


@dataclass(slots=True)
class _DownloadItem:
    hf_path: str
    dst_path: Path


def run_upscale_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    image_bytes: bytes,
    entry: TaskEntry,
    device: str,
    opts_get: Callable[..., Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    """Run a standalone upscaling task (single image)."""

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
                should_cancel=lambda: bool(entry.cancel_requested and entry.cancel_mode == "immediate"),
            )
            if not acquired:
                entry.error = "cancelled"
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)

            if entry.cancel_requested and entry.cancel_mode == "immediate":
                entry.error = "cancelled"
                return

            from PIL import Image  # type: ignore

            from apps.backend.use_cases.upscale import UpscaleParams, upscale_pil_image
            from apps.backend.interfaces.api.tasks.generation_tasks import encode_images
            from apps.backend.core.engine_interface import TaskType

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            params = UpscaleParams.from_payload(payload)

            # Tile progress callback.
            def on_tile(step: int, total: int) -> None:
                if entry.cancel_requested and entry.cancel_mode == "immediate":
                    raise RuntimeError("cancelled")
                percent = None
                try:
                    percent = (float(step) / float(total)) * 100.0 if total > 0 else None
                except Exception:
                    percent = None
                push(
                    {
                        "type": "progress",
                        "stage": "upscale",
                        "percent": percent,
                        "step": int(step),
                        "total_steps": int(total) if total else None,
                        "eta_seconds": None,
                    }
                )

            out = upscale_pil_image(img, params=params, progress_callback=on_tile)
            result_images = [out]

            info_obj: dict[str, Any] = {
                "upscale": params.to_dict(),
                **generation_provenance,
            }

            if bool(opts_get("samples_save", True)):
                save_generated_images(result_images, task=TaskType.UPSCALE, info=info_obj, metadata=generation_provenance)

            result = {
                "images": encode_images(result_images, metadata=generation_provenance),
                "info": info_obj,
            }
            entry.result = {"status": "completed", "result": result}
            success = True
        except Exception as err:  # pragma: no cover - surfaces runtime errors
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

    threading.Thread(target=worker, name=f"upscale-task-{task_id}", daemon=True).start()


def run_upscaler_download_task(
    *,
    task_id: str,
    items: Sequence[_DownloadItem],
    entry: TaskEntry,
    hf_repo_id: str,
    hf_revision: str | None,
) -> None:
    """Download allowlisted upscaler weights from a curated HF repo into local model roots."""

    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})

    def worker() -> None:
        success = False
        try:
            push({"type": "status", "stage": "running"})

            from huggingface_hub import hf_hub_download  # type: ignore

            total = len(items)
            completed = 0
            written: list[str] = []

            for item in items:
                if entry.cancel_requested and entry.cancel_mode == "immediate":
                    raise RuntimeError("cancelled")

                completed += 1
                push(
                    {
                        "type": "progress",
                        "stage": "download",
                        "percent": (completed / total) * 100.0 if total else None,
                        "step": completed,
                        "total_steps": total,
                        "eta_seconds": None,
                    }
                )

                local_tmp = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=item.hf_path,
                    revision=hf_revision,
                )
                item.dst_path.parent.mkdir(parents=True, exist_ok=True)
                if item.dst_path.exists():
                    raise RuntimeError(f"Destination exists: {item.dst_path}")
                shutil.copy2(local_tmp, item.dst_path)
                written.append(str(item.dst_path))

            try:
                from apps.backend.runtime.vision.upscalers.registry import invalidate_upscalers_cache

                invalidate_upscalers_cache()
            except Exception:
                pass

            result = {"files": written}
            entry.result = {"status": "completed", "result": {"images": [], "info": result}}
            success = True
        except Exception as err:  # pragma: no cover
            entry.error = str(err)
            success = False
        finally:
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)

    threading.Thread(target=worker, name=f"upscalers-download-task-{task_id}", daemon=True).start()


__all__ = ["run_upscale_task", "run_upscaler_download_task"]
