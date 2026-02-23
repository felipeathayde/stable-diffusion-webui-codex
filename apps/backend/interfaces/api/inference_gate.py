"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Single-flight inference gate for API tasks (env-controlled).
Provides a process-wide mutex used by API task workers to serialize GPU-heavy execution (generation/video/upscale/SUPIR),
keeping `/api/tasks/*` responsive and avoiding cross-task global state races (device switching, VRAM cache, engine lifecycles).

Symbols (top-level; keep in sync; no ghosts):
- `single_flight_enabled` (function): Returns whether single-flight is enabled via `CODEX_SINGLE_FLIGHT`.
- `acquire_inference_gate` (function): Attempts to acquire the process-wide inference mutex with cancel-aware polling.
- `release_inference_gate` (function): Releases the process-wide inference mutex.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable


_INFERENCE_LOCK = threading.Lock()
_INFERENCE_GATE_LOCAL = threading.local()


def _env_truthy(key: str, *, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be boolean (got {raw!r}).")


def single_flight_enabled() -> bool:
    return _env_truthy("CODEX_SINGLE_FLIGHT", default=True)


def acquire_inference_gate(*, should_cancel: Callable[[], bool], poll_interval_s: float = 0.10) -> bool:
    """Acquire the global inference gate.

    Returns True when acquired. Returns False when cancelled while waiting.
    """

    setattr(_INFERENCE_GATE_LOCAL, "lock_held", False)

    if not single_flight_enabled():
        return True

    interval = float(poll_interval_s)
    if interval <= 0:
        interval = 0.10

    while True:
        if should_cancel():
            return False
        if _INFERENCE_LOCK.acquire(timeout=interval):
            setattr(_INFERENCE_GATE_LOCAL, "lock_held", True)
            return True
        # brief sleep to avoid a tight loop when timeout is 0 on some platforms
        time.sleep(0.0)


def release_inference_gate() -> None:
    if not bool(getattr(_INFERENCE_GATE_LOCAL, "lock_held", False)):
        return
    try:
        _INFERENCE_LOCK.release()
    finally:
        setattr(_INFERENCE_GATE_LOCAL, "lock_held", False)


__all__ = ["single_flight_enabled", "acquire_inference_gate", "release_inference_gate"]
