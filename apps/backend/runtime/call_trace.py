from __future__ import annotations

"""Global function-call tracer for Codex backend.

Design goals:
- Log every Python function call via logger.debug once enabled.
- Minimal overhead and no recursion into logging/tracing internals.
- Safe across threads (installs both sys.setprofile and threading.setprofile).
- Activates/deactivates cleanly without leaving patched globals around.

This is intentionally separate from `runtime.trace` and `pipeline_debug`:
- `runtime.trace` focuses on selected torch/state-dict events.
- `pipeline_debug` is an opt-in decorator for specific pipelines.
- `call_trace` is a global hammer: it logs every call for deep debugging.
"""

import logging
import os
import sys
import threading
from types import FrameType
from typing import Any, Callable, Optional

_logger = logging.getLogger("backend.calltrace")

_enabled: bool = False
_local = threading.local()
_prev_profile: Optional[Callable[..., Any]] = None


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
            _local.busy = True
            depth = getattr(_local, "depth", 0) + 1
            _local.depth = depth

            mod = frame.f_globals.get("__name__", "<unknown>")
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

            # Indent for readability but keep message short
            indent = " " * (depth - 1)
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


def enable() -> None:
    """Enable global function-call tracing.

    Logging level must allow DEBUG for messages to be visible.
    """
    global _enabled, _prev_profile
    if _enabled:
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

    _prev_profile = sys.getprofile()
    sys.setprofile(_profiler)
    threading.setprofile(_profiler)
    _enabled = True
    _logger.debug("call-trace enabled (sys.setprofile)")


def disable() -> None:  # pragma: no cover - runtime hook
    global _enabled, _prev_profile
    if not _enabled:
        return
    # Restore previous profiler if present
    sys.setprofile(_prev_profile)
    threading.setprofile(_prev_profile)
    _prev_profile = None
    _enabled = False
    _logger.debug("call-trace disabled")


def enable_from_env() -> None:
    """Enable when CODEX_TRACE_DEBUG=1 (or truthy)."""
    if _truthy(os.getenv("CODEX_TRACE_DEBUG")):
        enable()


__all__ = ["enable", "disable", "enable_from_env"]

