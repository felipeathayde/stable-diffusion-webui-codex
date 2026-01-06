"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Thread-safe in-memory log ring buffer.
Captures the last N launcher log lines for UI display and diagnostics without relying on external logging sinks.

Symbols (top-level; keep in sync; no ghosts):
- `CodexLogBuffer` (dataclass): Thread-safe ring buffer for launcher log lines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List
import threading


@dataclass
class CodexLogBuffer:
    """Thread-safe ring buffer that stores the last N log lines."""

    capacity: int = 4000
    _lines: Deque[str] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            while len(self._lines) > self.capacity:
                self._lines.popleft()

    def snapshot(self) -> List[str]:
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
