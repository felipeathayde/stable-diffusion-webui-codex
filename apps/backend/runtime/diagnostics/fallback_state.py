"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Thread-local fallback state tracker for runtime diagnostics/contract traces.
Provides a single per-worker source-of-truth that records whether an automatic runtime fallback was used.

Symbols (top-level; keep in sync; no ghosts):
- `reset_fallback_state` (function): Clear thread-local fallback usage state for a new worker/task boundary.
- `mark_fallback_used` (function): Mark fallback as used for the current worker/thread.
- `fallback_used` (function): Return whether fallback has been marked for the current worker/thread.
"""

from __future__ import annotations

import threading


_STATE = threading.local()


def reset_fallback_state() -> None:
    _STATE.used = False


def mark_fallback_used() -> None:
    _STATE.used = True


def fallback_used() -> bool:
    return bool(getattr(_STATE, "used", False))


__all__ = ["reset_fallback_state", "mark_fallback_used", "fallback_used"]
