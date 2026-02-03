"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task status and output file API routes.
Exposes SSE task events (with bounded replay/resume via `after` / `Last-Event-ID` + monotonic `id:`), cancellation, and output file access under
CODEX_ROOT/output.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for task endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
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
                entry.schedule_cleanup(task_id)
                return {"status": "error", "error": entry.error, "last_event_id": entry.last_event_id()}
            entry.schedule_cleanup(task_id)
            out = entry.result or {"status": "completed", "result": {}}
            try:
                if isinstance(out, dict):
                    out.setdefault("last_event_id", entry.last_event_id())
            except Exception:
                pass
            return out
        snap = entry.snapshot_running()
        snap["task_id"] = task_id
        return snap

    @router.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request, after: int | None = None) -> StreamingResponse:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")

        def _sse(*, event_id: int, json_payload: str) -> str:
            return f"id: {event_id}\n" f"data: {json_payload}\n\n"

        async def event_stream():
            last_id = 0
            if isinstance(after, int) and after >= 0:
                last_id = int(after)
            else:
                try:
                    raw = request.headers.get("Last-Event-ID")
                    if raw is not None and str(raw).strip():
                        last_id = max(0, int(str(raw).strip(), 10))
                except Exception:
                    last_id = 0

            while True:
                gap, items = entry.iter_events_after(last_id)
                if gap:
                    oldest, newest = entry.buffer_window()
                    gap_id = max(int(last_id) + 1, (int(oldest) - 1) if int(oldest) > 0 else int(last_id) + 1)
                    gap_payload = {
                        "type": "gap",
                        "oldest_event_id": int(oldest),
                        "newest_event_id": int(newest),
                        "last_event_id": int(entry.last_event_id()),
                    }
                    yield _sse(event_id=gap_id, json_payload=json.dumps(gap_payload))
                    # After a gap, replay whatever we still have. If the buffer is empty, avoid looping gaps.
                    if int(oldest) > 0:
                        last_id = int(gap_id)
                    else:
                        base_last_id = int(entry.last_event_id()) - (2 if entry.done.done() else 0)
                        last_id = max(int(gap_id), max(0, base_last_id))
                    continue

                for ev in items:
                    last_id = int(ev.event_id)
                    yield _sse(event_id=int(ev.event_id), json_payload=ev.json_payload)

                if entry.done.done():
                    # Terminal emission: always end the stream (contract).
                    terminal_ids = entry.terminal_event_ids()
                    if terminal_ids is None:
                        # Defensive: should be impossible when done is set.
                        break

                    primary_id, end_id = terminal_ids
                    if int(last_id) < int(primary_id):
                        if entry.error:
                            err_payload = {"type": "error", "message": entry.error}
                            yield _sse(event_id=int(primary_id), json_payload=json.dumps(err_payload))
                        else:
                            result_payload: dict[str, Any] = {"type": "result", "images": [], "info": {}}
                            if isinstance(entry.result, dict):
                                result_obj = entry.result.get("result")
                                if isinstance(result_obj, dict):
                                    result_payload.update(result_obj)
                            yield _sse(event_id=int(primary_id), json_payload=json.dumps(result_payload))
                        last_id = int(primary_id)

                    if int(last_id) < int(end_id):
                        yield _sse(event_id=int(end_id), json_payload=json.dumps({"type": "end"}))
                        last_id = int(end_id)

                    entry.schedule_cleanup(task_id)
                    break

                await entry.wait_for_event_or_done(after_event_id=int(last_id))

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
