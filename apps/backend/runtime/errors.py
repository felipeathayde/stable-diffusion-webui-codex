"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Structured error reporting utilities for the Codex backend.
Provides a thread-safe registry of recent exceptions plus helpers to capture, log, format, and display actionable diagnostics without relying on
legacy error globals.

Symbols (top-level; keep in sync; no ghosts):
- `ExceptionFrame` (dataclass): One captured stack frame (location + optional source line).
- `ExceptionRecord` (dataclass): Captured exception payload (type/message/context + frames) with dict conversion helper.
- `ErrorRegistry` (class): Thread-safe bounded registry (deque) of recent exception records.
- `_format_tb` (function): Converts a traceback into `ExceptionFrame` entries.
- `record_exception` (function): Records an exception into the global registry and returns an `ExceptionRecord`.
- `record_current_exception` (function): Records the current active exception (if any).
- `get_recent_exceptions` (function): Returns recent exception records (newest first).
- `clear_records` (function): Clears the global registry.
- `report_error` (function): Logs an error message and records the associated exception (optional `exc_info`).
- `print_error_explanation` (function): Prints a human-readable explanation block (used for CLI/UI-friendly diagnostics).
- `display_exception` (function): Records and logs an exception with optional full traceback.
- `display_exception_once` (function): Like `display_exception`, but only logs once per context string.
- `iter_exception_dicts` (function): Iterates exception records as JSON-serializable dicts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import logging
import sys
import threading
import traceback
from types import TracebackType
from typing import Iterable, List, Optional


_LOGGER = logging.getLogger("backend.errors")


@dataclass(frozen=True)
class ExceptionFrame:
    """Single stack frame from a captured traceback."""

    location: str
    line: str | None


@dataclass(frozen=True)
class ExceptionRecord:
    """Structured representation of a captured exception."""

    exception_type: str
    message: str
    context: str | None
    frames: List[ExceptionFrame]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frames"] = [asdict(frame) for frame in self.frames]
        return data


class ErrorRegistry:
    """Thread-safe bounded deque of :class:`ExceptionRecord`."""

    def __init__(self, *, max_records: int = 5) -> None:
        self._records: deque[ExceptionRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def push(self, record: ExceptionRecord) -> None:
        with self._lock:
            self._records.append(record)

    def recent(self) -> List[ExceptionRecord]:
        with self._lock:
            return list(reversed(self._records))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


_REGISTRY = ErrorRegistry()
_DISPLAYED_CONTEXTS: set[str] = set()


def _format_tb(tb: TracebackType | None) -> List[ExceptionFrame]:
    frames: List[ExceptionFrame] = []
    if tb is None:
        return frames
    for entry in traceback.extract_tb(tb):
        location = f"{entry.filename}:{entry.lineno} in {entry.name}"
        frames.append(ExceptionFrame(location=location, line=entry.line))
    return frames


def record_exception(exc: BaseException, *, context: str | None = None, tb: TracebackType | None = None) -> ExceptionRecord:
    """Capture *exc* and append it to the registry."""

    trace = tb if tb is not None else exc.__traceback__
    record = ExceptionRecord(
        exception_type=exc.__class__.__name__,
        message=str(exc),
        context=context,
        frames=_format_tb(trace),
    )
    _REGISTRY.push(record)
    return record


def record_current_exception(*, context: str | None = None) -> ExceptionRecord | None:
    """Capture the active exception, if any, and append it to the registry."""

    exc_type, exc, tb = sys.exc_info()
    if exc is None:
        return None
    assert exc_type is not None  # narrow for type checkers
    return record_exception(exc, context=context, tb=tb)


def get_recent_exceptions() -> List[ExceptionRecord]:
    """Return the most recent records (newest first)."""

    return _REGISTRY.recent()


def clear_records() -> None:
    """Reset the registry."""

    _REGISTRY.clear()


def report_error(message: str, *, exc_info: bool = False, context: str | None = None) -> None:
    """Log *message* and persist it in the registry.

    When ``exc_info`` is true, the current exception is recorded; otherwise a
    synthetic record with no frames is stored.
    """

    if exc_info:
        _LOGGER.exception(message)
        record_current_exception(context=context or "report_error")
        return

    _LOGGER.error(message)
    record = ExceptionRecord(
        exception_type="Message",
        message=message,
        context=context,
        frames=[],
    )
    _REGISTRY.push(record)


def print_error_explanation(message: str, *, header: str | None = None) -> None:
    """Emit a formatted block explaining *message* and store it as a record."""

    lines = [line.rstrip() for line in message.strip().splitlines() if line.strip()]
    if not lines:
        return
    width = max(len(line) for line in lines)
    banner = "=" * width
    title = f"{header}:" if header else None
    if title:
        _LOGGER.error(title)
    _LOGGER.error(banner)
    for line in lines:
        _LOGGER.error(line)
    _LOGGER.error(banner)
    # Store the explanation so API clients can read it back if needed
    record = ExceptionRecord(
        exception_type="Explanation",
        message="\n".join(lines),
        context=header,
        frames=[],
    )
    _REGISTRY.push(record)


def display_exception(exc: BaseException, *, context: str | None = None, full_traceback: bool = False) -> ExceptionRecord:
    """Log *exc* with formatted traceback and persist it."""

    if full_traceback:
        _LOGGER.exception("%s", context or exc)
    else:
        summary = traceback.TracebackException.from_exception(exc, capture_locals=False)
        formatted = "".join(summary.format()).rstrip()
        _LOGGER.error("%s: %s", context or exc.__class__.__name__, exc)
        _LOGGER.error("%s", formatted)
    return record_exception(exc, context=context)


def display_exception_once(exc: BaseException, *, context: str) -> ExceptionRecord | None:
    """Display *exc* only once for a given *context* label."""

    if context in _DISPLAYED_CONTEXTS:
        return None
    _DISPLAYED_CONTEXTS.add(context)
    return display_exception(exc, context=context)


def iter_exception_dicts() -> Iterable[dict[str, object]]:
    """Yield recent records as serialisable dictionaries."""

    for record in get_recent_exceptions():
        yield record.as_dict()


__all__ = [
    "ExceptionFrame",
    "ExceptionRecord",
    "ErrorRegistry",
    "clear_records",
    "display_exception",
    "display_exception_once",
    "get_recent_exceptions",
    "iter_exception_dicts",
    "print_error_explanation",
    "record_current_exception",
    "record_exception",
    "report_error",
]
