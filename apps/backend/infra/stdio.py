"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared low-level stdout/stderr stream emission helpers for CLI prompts and crash-path notices.
Provides tiny helpers so explicit stream writes stay centralized while preserving exact text/stream semantics required by
interactive prompts and best-effort exception notices.

Symbols (top-level; keep in sync; no ghosts):
- `write_stdout` (function): Writes text to stdout and optionally flushes immediately.
- `write_stderr` (function): Writes text to stderr and optionally flushes immediately.
- `flush_stdout` (function): Flushes stdout without emitting additional text.
- `flush_stderr` (function): Flushes stderr without emitting additional text.
"""

from __future__ import annotations

import sys


def write_stdout(text: str, *, flush: bool = False) -> None:
    sys.stdout.write(text)
    if flush:
        sys.stdout.flush()


def write_stderr(text: str, *, flush: bool = False) -> None:
    sys.stderr.write(text)
    if flush:
        sys.stderr.flush()


def flush_stdout() -> None:
    sys.stdout.flush()


def flush_stderr() -> None:
    sys.stderr.flush()
