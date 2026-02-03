"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: In-process task registry for API jobs.
Tracks task status, bounded SSE replay buffers, and cancellation requests for API endpoints.

Symbols (top-level; keep in sync; no ghosts):
- `tasks` (constant): In-memory task registry mapping task_id -> TaskEntry.
- `tasks_lock` (constant): Lock protecting access to the in-memory task registry.
- `TaskEntry` (class): In-memory task registry entry (snapshot/result/error + bounded event buffer + cancellation flags).
- `get_task` (function): Reads a task entry by id from the in-process registry.
- `register_task` (function): Registers a new task entry in the in-process registry.
- `unregister_task` (function): Unregisters a task by id from the in-process registry.
- `request_task_cancel` (function): Marks a task as cancelled (`immediate` vs `after_current`) for worker/coordinator checks.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

tasks: Dict[str, "TaskEntry"] = {}
tasks_lock = threading.Lock()

_DEFAULT_TASK_EVENT_BUFFER_MAX_EVENTS = 5000
_DEFAULT_TASK_EVENT_BUFFER_MAX_MB = 64


def _parse_int_env(key: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(key)
    if raw is None:
        return int(default)
    s = str(raw).strip()
    if not s:
        return int(default)
    try:
        value = int(s, 10)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer (got {raw!r}).") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum} (got {value}).")
    return value


TASK_EVENT_BUFFER_MAX_EVENTS = _parse_int_env(
    "CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
    _DEFAULT_TASK_EVENT_BUFFER_MAX_EVENTS,
    minimum=1,
)
TASK_EVENT_BUFFER_MAX_BYTES = _parse_int_env(
    "CODEX_TASK_EVENT_BUFFER_MAX_MB",
    _DEFAULT_TASK_EVENT_BUFFER_MAX_MB,
    minimum=1,
) * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BufferedEvent:
    event_id: int
    event_type: str
    json_payload: str
    byte_len: int


class TaskEntry:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        max_buffered_events: int = TASK_EVENT_BUFFER_MAX_EVENTS,
        max_buffered_bytes: int = TASK_EVENT_BUFFER_MAX_BYTES,
    ) -> None:
        self.loop = loop
        self._max_buffered_events = int(max_buffered_events)
        self._max_buffered_bytes = int(max_buffered_bytes)
        self._signal: asyncio.Event = asyncio.Event()
        self._events: Deque[BufferedEvent] = deque()
        self._events_bytes: int = 0
        self._events_ever_dropped: bool = False
        self._last_event_id: int = 0
        self._status_stage: str = "queued"
        self._progress: dict[str, Any] | None = None
        self._preview_image: dict[str, str] | None = None
        self._preview_step: int | None = None
        self._created_at_ms: int = int(time.time() * 1000)
        self._started_at_ms: int | None = None
        self._finished_at_ms: int | None = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.done: asyncio.Future[bool] = loop.create_future()
        self.cleanup_handle: Optional[asyncio.TimerHandle] = None
        self.cancel_requested: bool = False
        self.cancel_mode: str = "immediate"  # immediate | after_current
        self.last_preview_id_sent: int = 0

    def schedule_cleanup(self, task_id: str, delay: float = 300.0) -> None:
        if self.cleanup_handle:
            self.cleanup_handle.cancel()
        self.cleanup_handle = self.loop.call_later(delay, lambda: unregister_task(task_id))

    # ------------------------------------------------------------------ events
    def push_event(self, event: Dict[str, Any]) -> None:
        """Push a non-terminal event into the bounded replay buffer (thread-safe)."""

        if not isinstance(event, dict):
            raise TypeError("event must be a dict")

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self.loop:
            self._push_event_nowait(event)
            return

        self.loop.call_soon_threadsafe(self._push_event_nowait, dict(event))

    def _push_event_nowait(self, event: Dict[str, Any]) -> None:
        if self.done.done():
            raise RuntimeError("Cannot push events to a finished task.")

        event_type = str(event.get("type", "") or "").strip().lower()
        if not event_type:
            raise ValueError("event.type is required")

        if self._started_at_ms is None and event_type == "status" and str(event.get("stage", "") or "") == "running":
            self._started_at_ms = int(time.time() * 1000)

        if event_type == "status":
            stage = str(event.get("stage", "") or "").strip()
            if stage:
                self._status_stage = stage
        elif event_type == "progress":
            stage = str(event.get("stage", "") or "").strip()
            self._progress = {
                "stage": stage,
                "percent": event.get("percent", None),
                "step": event.get("step", None),
                "total_steps": event.get("total_steps", None),
                "eta_seconds": event.get("eta_seconds", None),
            }
            preview_image = event.get("preview_image")
            if (
                isinstance(preview_image, dict)
                and isinstance(preview_image.get("format"), str)
                and isinstance(preview_image.get("data"), str)
            ):
                self._preview_image = {"format": str(preview_image["format"]), "data": str(preview_image["data"])}
            preview_step = event.get("preview_step")
            if isinstance(preview_step, (int, float)) and not isinstance(preview_step, bool):
                try:
                    self._preview_step = int(preview_step)
                except Exception:
                    pass

        payload = json.dumps(event)
        byte_len = len(payload.encode("utf-8", errors="replace"))

        self._last_event_id += 1
        rec = BufferedEvent(event_id=self._last_event_id, event_type=event_type, json_payload=payload, byte_len=byte_len)
        self._events.append(rec)
        self._events_bytes += byte_len

        # Enforce caps by dropping oldest buffered events.
        while (len(self._events) > self._max_buffered_events) or (self._events_bytes > self._max_buffered_bytes):
            old = self._events.popleft()
            self._events_bytes = max(0, self._events_bytes - int(old.byte_len))
            self._events_ever_dropped = True

        self._signal_waiters()

    def mark_finished(self, *, success: bool) -> None:
        """Mark the task as finished (thread-safe).

        This must be called exactly once on every terminal path (success/error/cancel).
        """

        def _set() -> None:
            if self._finished_at_ms is None:
                self._finished_at_ms = int(time.time() * 1000)
            if not self.done.done():
                self.done.set_result(bool(success))
            self._signal_waiters()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self.loop:
            _set()
            return

        self.loop.call_soon_threadsafe(_set)

    def last_event_id(self) -> int:
        base = int(self._last_event_id)
        # Reserve ids for terminal `{result|error}` and `end` emissions.
        return base + 2 if self.done.done() else base

    def buffer_window(self) -> Tuple[int, int]:
        """Return (oldest_event_id, newest_event_id). 0,0 when empty."""
        if not self._events:
            return 0, 0
        return int(self._events[0].event_id), int(self._events[-1].event_id)

    def snapshot_running(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot used by GET /api/tasks/{id} while running."""
        oldest, newest = self.buffer_window()
        out: Dict[str, Any] = {
            "status": "running",
            "stage": self._status_stage,
            "progress": dict(self._progress) if isinstance(self._progress, dict) else None,
            "last_event_id": int(self.last_event_id()),
            "buffer_oldest_event_id": int(oldest),
            "buffer_newest_event_id": int(newest),
            "created_at_ms": int(self._created_at_ms),
            "started_at_ms": self._started_at_ms,
        }
        if self._preview_image is not None:
            out["preview_image"] = dict(self._preview_image)
        if self._preview_step is not None:
            out["preview_step"] = int(self._preview_step)
        return out

    def terminal_event_ids(self) -> Tuple[int, int] | None:
        """Return (terminal_primary_id, terminal_end_id) when finished."""
        if not self.done.done():
            return None
        base = int(self._last_event_id)
        return base + 1, base + 2

    async def wait_for_event_or_done(self, *, after_event_id: int) -> None:
        """Wait until a new event arrives (id > after_event_id) or the task completes.

        This is multi-subscriber-safe (no lost wake-ups).
        """
        after = int(after_event_id)
        while True:
            if self.done.done():
                return
            if int(self._last_event_id) > after:
                return
            signal = self._signal
            # Re-check after capturing the signal to avoid missing a concurrent publish.
            if self.done.done() or int(self._last_event_id) > after:
                return
            await signal.wait()

    def iter_events_after(self, after_event_id: int) -> Tuple[bool, list[BufferedEvent]]:
        """Return (gap, events) for buffered events after the given id."""
        if not self._events:
            # If the task produced events but the buffer was fully truncated, force a snapshot resync.
            if self._events_ever_dropped and int(after_event_id) < int(self._last_event_id):
                return True, []
            return False, []
        oldest = int(self._events[0].event_id)
        newest = int(self._events[-1].event_id)
        if int(after_event_id) < (oldest - 1):
            # history is truncated; caller should emit a gap signal
            return True, list(self._events)
        if int(after_event_id) >= newest:
            return False, []
        return False, [ev for ev in self._events if int(ev.event_id) > int(after_event_id)]

    def _signal_waiters(self) -> None:
        """Wake any SSE subscribers waiting for new events (event-loop only)."""
        self._signal.set()
        self._signal = asyncio.Event()


def get_task(task_id: str) -> Optional["TaskEntry"]:
    with tasks_lock:
        return tasks.get(task_id)


def register_task(task_id: str, entry: "TaskEntry") -> None:
    with tasks_lock:
        tasks[task_id] = entry


def unregister_task(task_id: str) -> None:
    with tasks_lock:
        tasks.pop(task_id, None)


def request_task_cancel(task_id: str, *, mode: str = "immediate") -> bool:
    with tasks_lock:
        entry = tasks.get(task_id)
        if entry is None:
            return False
        entry.cancel_requested = True
        entry.cancel_mode = mode if mode in {"immediate", "after_current"} else "immediate"
        return True


__all__ = ["tasks", "tasks_lock", "TaskEntry", "BufferedEvent", "TASK_EVENT_BUFFER_MAX_EVENTS", "TASK_EVENT_BUFFER_MAX_BYTES", "get_task", "register_task", "unregister_task", "request_task_cancel"]
