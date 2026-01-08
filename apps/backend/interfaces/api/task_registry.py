"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: In-process task registry for API jobs.
Tracks task status, SSE queues, and cancellation requests for generation endpoints.

Symbols (top-level; keep in sync; no ghosts):
- `tasks` (constant): In-memory task registry mapping task_id -> TaskEntry.
- `tasks_lock` (constant): Lock protecting access to the in-memory task registry.
- `TaskEntry` (class): In-memory task registry entry (status/result/error + SSE event queue + cancellation flags).
- `get_task` (function): Reads a task entry by id from the in-process registry.
- `register_task` (function): Registers a new task entry in the in-process registry.
- `request_task_cancel` (function): Marks a task as cancelled (`immediate` vs `after_current`) for worker/coordinator checks.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

tasks: Dict[str, "TaskEntry"] = {}
tasks_lock = threading.Lock()


class TaskEntry:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
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
        self.cleanup_handle = self.loop.call_later(delay, lambda: tasks.pop(task_id, None))


def get_task(task_id: str) -> Optional["TaskEntry"]:
    with tasks_lock:
        return tasks.get(task_id)


def register_task(task_id: str, entry: "TaskEntry") -> None:
    with tasks_lock:
        tasks[task_id] = entry


def request_task_cancel(task_id: str, *, mode: str = "immediate") -> bool:
    with tasks_lock:
        entry = tasks.get(task_id)
        if entry is None:
            return False
        entry.cancel_requested = True
        entry.cancel_mode = mode if mode in {"immediate", "after_current"} else "immediate"
        return True
