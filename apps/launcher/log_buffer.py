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
- `CodexLogRecord` (dataclass): Structured log record (timestamp/source/message) with a monotonic id.
- `format_log_record` (function): Formats a log record into a human-readable single-line string.
- `CodexLogBuffer` (dataclass): Thread-safe ring buffer for structured log records.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, List, Tuple
import threading


@dataclass(frozen=True, slots=True)
class CodexLogRecord:
    """Structured log record stored by `CodexLogBuffer`."""

    id: int
    created_at: float
    source: str
    message: str
    stream: str = "stdout"


def format_log_record(record: CodexLogRecord) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(record.created_at))
    source = str(record.source)
    stream = str(record.stream or "").strip()
    stream_suffix = f":{stream}" if stream and stream != "stdout" else ""
    return f"[{ts}] [{source}{stream_suffix}] {record.message}"


@dataclass
class CodexLogBuffer:
    """Thread-safe ring buffer that stores the last N log records."""

    capacity: int = 4000
    _records: Deque[CodexLogRecord] = field(default_factory=deque, init=False)
    _next_id: int = field(default=1, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def log(self, source: str, message: str, *, stream: str = "stdout", created_at: float | None = None) -> CodexLogRecord:
        with self._lock:
            record = CodexLogRecord(
                id=self._next_id,
                created_at=time.time() if created_at is None else float(created_at),
                source=str(source),
                message=str(message),
                stream=str(stream),
            )
            self._records.append(record)
            self._next_id += 1
            while len(self._records) > self.capacity:
                self._records.popleft()
            return record

    def snapshot_with_ids(self) -> List[Tuple[int, CodexLogRecord]]:
        """Return buffered log records with their monotonically increasing ids."""
        with self._lock:
            return [(rec.id, rec) for rec in self._records]

    def snapshot(self) -> List[CodexLogRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
