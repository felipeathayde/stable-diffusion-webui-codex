"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global Python function-call tracer for deep Codex debugging.
Implements a sys/thread profile hook that logs function calls under `apps.*` with indentation and per-function call caps, and provides env-driven
enable/disable helpers to avoid flooding logs by default.

Symbols (top-level; keep in sync; no ghosts):
- `_truthy` (function): Parses common truthy tokens for env-driven toggles.
- `_profiler` (function): Profile hook used by `sys.setprofile` / `threading.setprofile` (logs `call`/`return` events).
- `_set_max_per_func` (function): Configures the per-function call cap (0 disables the cap).
- `_reset_counters` (function): Clears per-function call counters and “muted” notifications.
- `enable` (function): Enables global call tracing (optionally configuring caps) and installs the profiler hooks.
- `disable` (function): Disables call tracing and restores prior profiler hooks.
- `_env_trace_limit` (function): Reads the max-per-func cap from environment variables.
- `enable_from_env` (function): Enables tracing when env flags request it (launcher/API entrypoint integration).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from types import FrameType
from typing import Any, Callable, Optional, Tuple

_logger = logging.getLogger("backend.calltrace")

_enabled: bool = False
_local = threading.local()
_prev_profile: Optional[Callable[..., Any]] = None

_DEFAULT_MAX_PER_FUNC = 10
_max_per_func: int = _DEFAULT_MAX_PER_FUNC
_call_counts: dict[Tuple[str, str], int] = {}
_muted_notified: set[Tuple[str, str]] = set()


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profiler(frame: FrameType, event: str, arg: Any):  # pragma: no cover - runtime hook
    # Guard: prevent recursion while we log
    if getattr(_local, "busy", False):
        return _profiler

    if event == "call":
        try:
            mod = frame.f_globals.get("__name__", "<unknown>")

            # Limit noise: only log modules inside the Codex apps package
            if not isinstance(mod, str) or not mod.startswith("apps."):
                return _profiler
            _local.busy = True
            depth = getattr(_local, "depth", 0) + 1
            _local.depth = depth
            func = frame.f_code.co_name or "<unknown>"

            # Best-effort class name enrichment
            qn = func
            try:
                if "self" in frame.f_locals:
                    qn = f"{type(frame.f_locals['self']).__name__}.{func}"
                elif "cls" in frame.f_locals and hasattr(frame.f_locals["cls"], "__name__"):
                    qn = f"{frame.f_locals['cls'].__name__}.{func}"
            except Exception:
                pass

            key = (mod, qn)
            indent = " " * (depth - 1)

            if _max_per_func > 0:
                count = _call_counts.get(key, 0) + 1
                _call_counts[key] = count
                if count > _max_per_func:
                    if key not in _muted_notified:
                        _logger.debug("%sCALL %s.%s (muted after %d calls)", indent, mod, qn, _max_per_func)
                        _muted_notified.add(key)
                    return _profiler

            # Indent for readability but keep message short
            _logger.debug("%sCALL %s.%s", indent, mod, qn)
        except Exception:
            # Never raise from the profiler; keep tracing alive
            pass
        finally:
            _local.busy = False
        return _profiler
    elif event == "return":
        # Track depth to keep indentation balanced
        try:
            _local.depth = max(0, getattr(_local, "depth", 0) - 1)
        except Exception:
            _local.depth = 0
        return _profiler

    # Ignore other events (c_call/c_return, exceptions) to reduce noise
    return _profiler


def _set_max_per_func(value: Optional[int]) -> None:
    global _max_per_func
    if value is None:
        _max_per_func = _DEFAULT_MAX_PER_FUNC
        return
    try:
        numeric = int(value)
    except Exception:
        numeric = _DEFAULT_MAX_PER_FUNC
    _max_per_func = max(0, numeric)


def _reset_counters() -> None:
    _call_counts.clear()
    _muted_notified.clear()


def enable(*, max_calls_per_func: Optional[int] = None) -> None:
    """Enable global function-call tracing.

    Logging level must allow DEBUG for messages to be visible.
    """
    global _enabled, _prev_profile

    # Env fallback even when caller passes None (ensures UI/env overrides stick)
    if max_calls_per_func is None:
        max_calls_per_func = _env_trace_limit()
    _set_max_per_func(max_calls_per_func)
    if _enabled:
        _reset_counters()
        _logger.debug(
            "call-trace limit set to %s per function",
            "unlimited" if _max_per_func == 0 else _max_per_func,
        )
        return

    # Avoid tracing our own tracing/logging internals by bumping this logger level
    # if the root level is very low. We still emit debug from this logger.
    try:
        # Some frameworks attach handlers to root only; ensure propagation is disabled
        _root = logging.getLogger("backend")
        _root.setLevel(min(_root.level, logging.DEBUG))
        _root.propagate = False
    except Exception:
        pass

    _reset_counters()

    _prev_profile = sys.getprofile()
    sys.setprofile(_profiler)
    threading.setprofile(_profiler)
    _enabled = True
    _logger.debug(
        "call-trace enabled (sys.setprofile, limit=%s per function)",
        "unlimited" if _max_per_func == 0 else _max_per_func,
    )


def disable() -> None:  # pragma: no cover - runtime hook
    global _enabled, _prev_profile
    if not _enabled:
        return
    # Restore previous profiler if present
    sys.setprofile(_prev_profile)
    threading.setprofile(_prev_profile)
    _prev_profile = None
    _enabled = False
    _reset_counters()
    _logger.debug("call-trace disabled")


def _env_trace_limit() -> Optional[int]:
    raw = os.getenv("CODEX_TRACE_DEBUG_MAX_PER_FUNC")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def enable_from_env() -> None:
    """Enable when CODEX_TRACE_DEBUG=1 (or truthy)."""
    if _truthy(os.getenv("CODEX_TRACE_DEBUG")):
        enable(max_calls_per_func=_env_trace_limit())


__all__ = ["enable", "disable", "enable_from_env"]
