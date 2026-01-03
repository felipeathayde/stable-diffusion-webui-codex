"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight progress/task tracker without legacy globals.
Tracks queued/active task IDs and retains a small history of finished tasks/results for `/api/tasks` status and UI polling.

Symbols (top-level; keep in sync; no ghosts):
- `_now` (function): Returns current monotonic wall time as seconds.
- `create_task_id` (function): Generates a short task ID token for queueing/polling.
- `TrackerSnapshot` (dataclass): Snapshot of current/pending/finished task IDs.
- `CodexProgressTracker` (class): Thread-safe tracker with queue/begin/complete lifecycle and last-result access.
"""

from __future__ import annotations

import random
import string
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional, Tuple


_ID_ALPHABET = string.ascii_uppercase + string.digits


def _now() -> float:
    return time.time()


def create_task_id(kind: str) -> str:
    token = "".join(random.choice(_ID_ALPHABET) for _ in range(7))
    return f"task({kind}-{token})"


@dataclass
class TrackerSnapshot:
    current: Optional[str]
    pending: Tuple[str, ...]
    finished: Tuple[str, ...]


class CodexProgressTracker:
    """Tracks queued/active generation tasks without legacy globals."""

    def __init__(self, *, max_finished: int = 16, max_results: int = 2) -> None:
        self._lock = threading.Lock()
        self._current: Optional[str] = None
        self._pending: "OrderedDict[str, float]" = OrderedDict()
        self._finished: Deque[str] = deque(maxlen=max_finished)
        self._results: Deque[Tuple[str, Any]] = deque(maxlen=max_results)

    # ----- queue lifecycle -------------------------------------------------
    def queue(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._pending:
                return
            self._pending[task_id] = _now()

    def begin(self, task_id: str) -> None:
        with self._lock:
            self._current = task_id
            self._pending.pop(task_id, None)

    def complete(self, task_id: str, *, result: Any = None) -> None:
        with self._lock:
            if task_id == self._current:
                self._current = None
            self._pending.pop(task_id, None)
            self._finished.append(task_id)
            if result is not None:
                self._results.append((task_id, result))

    def discard(self, task_id: str) -> None:
        with self._lock:
            if task_id == self._current:
                self._current = None
            self._pending.pop(task_id, None)

    # ----- inspection ------------------------------------------------------
    def pending(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._pending.keys())

    def finished(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._finished)

    def snapshot(self) -> TrackerSnapshot:
        with self._lock:
            return TrackerSnapshot(
                current=self._current,
                pending=tuple(self._pending.keys()),
                finished=tuple(self._finished),
            )

    def status(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            active = task_id == self._current
            queued = task_id in self._pending
            completed = task_id in self._finished
            position = None
            if queued:
                ordered = list(self._pending.keys())
                position = ordered.index(task_id) + 1
            return {
                "active": active,
                "queued": queued,
                "completed": completed,
                "queue_position": position,
            }

    def last_result(self, task_id: str) -> Optional[Any]:
        with self._lock:
            for key, payload in reversed(self._results):
                if key == task_id:
                    return payload
        return None

    def iter_results(self) -> Iterable[Tuple[str, Any]]:
        with self._lock:
            return tuple(self._results)
