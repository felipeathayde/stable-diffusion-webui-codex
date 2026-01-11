"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task status and output file API routes.
Exposes SSE task events, cancellation, and output file access under CODEX_ROOT/output.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for task endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from apps.backend.interfaces.api.task_registry import get_task, request_task_cancel


def build_router(*, codex_root: Path, backend_state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/output/{rel_path:path}")
    async def get_output_file(rel_path: str) -> FileResponse:
        root = (codex_root / "output").resolve()
        raw = str(rel_path or "").lstrip("/").replace("\\", "/")
        target = (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid output path") from None
        if not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(target))

    @router.get("/api/tasks/{task_id}")
    async def task_status(task_id: str) -> Dict[str, Any]:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if entry.done.done():
            if entry.error:
                return {"status": "error", "error": entry.error}
            entry.schedule_cleanup(task_id)
            return entry.result or {"status": "completed", "result": {}}
        return {"status": "running"}

    @router.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str) -> StreamingResponse:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")

        async def event_stream():
            while True:
                payload = await entry.queue.get()
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("type") == "end":
                    entry.schedule_cleanup(task_id)
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/api/tasks/{task_id}/cancel")
    async def task_cancel(task_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
        mode_raw = str(payload.get("mode", "immediate")).strip().lower() if isinstance(payload, dict) else "immediate"
        mode = "after_current" if mode_raw == "after_current" else "immediate"
        ok = request_task_cancel(task_id, mode=mode)
        if not ok:
            raise HTTPException(status_code=404, detail="Task not found")
        if mode == "immediate":
            try:
                backend_state.stop_generating()
            except Exception:
                pass
        return {"status": "cancelling", "mode": mode}

    return router
