"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Centralized exception capture + full-traceback dumping for backend processes.
Installs sys/threading/asyncio exception hooks that dump full tracebacks to an append-only log file (by day + pid) under `CODEX_ROOT/logs/`,
and provides helpers to persist caught exceptions with a one-line stderr notice for TUI visibility.

Symbols (top-level; keep in sync; no ghosts):
- `_ensure_log_path` (function): Resolves and creates the exception log file path (stable by day+pid).
- `install_exception_hooks` (function): Installs sys/threading exception hooks (idempotent) and returns the log path.
- `attach_asyncio` (function): Hooks the asyncio loop exception handler to dump unhandled task exceptions.
- `dump_message` (function): Writes an arbitrary message record into the exception log.
- `dump_exception` (function): Dumps an exception triple (type/value/tb) to the log and emits a stderr notice.
- `dump_current_exception` (function): Convenience wrapper for dumping the current exception via `sys.exc_info()`.
- `_stderr_notice` (function): Emits a one-line stderr notice pointing at the log file path.
- `get_log_path` (function): Returns the resolved exception log path (or `None` if not installed).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from apps.backend.infra.config.repo_root import get_repo_root

_installed = False
_log_path: Optional[str] = None
_orig_excepthook = sys.excepthook
_orig_threading_hook = getattr(threading, "excepthook", None)


def _ensure_log_path(log_dir: Optional[str] = None, file_path: Optional[str] = None) -> str:
    global _log_path
    if _log_path:
        return _log_path
    if file_path:
        path = file_path
    else:
        root = log_dir or os.environ.get("CODEX_ERROR_LOG_DIR") or str(get_repo_root() / "logs")
        os.makedirs(root, exist_ok=True)
        ts = time.strftime("%Y%m%d")
        path = os.path.join(root, f"exceptions-{ts}-{os.getpid()}.log")
    # Touch the file to ensure it exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass
    _log_path = path
    return path


def install_exception_hooks(log_dir: Optional[str] = None, file_path: Optional[str] = None) -> str:
    """Install sys and threading exception hooks.

    Returns the resolved log file path.
    """
    global _installed
    if _installed:
        return _ensure_log_path()

    path = _ensure_log_path(log_dir, file_path)

    def _dump_and_forward(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        try:
            dump_exception(exc_type, exc, tb, where="sys.excepthook", context=None)
        finally:
            try:
                _orig_excepthook(exc_type, exc, tb)
            except Exception:
                # Avoid recursive failures
                pass
        # Optionally terminate the process after dump to surface 1-by-1 errors
        if (os.environ.get("CODEX_STRICT_EXIT", "1") or "").strip() not in ("0", "false", "False"):
            os._exit(1)

    def _thread_hook(args):  # type: ignore[no-untyped-def]
        try:
            dump_exception(args.exc_type, args.exc_value, args.exc_traceback, where=f"thread:{args.thread.name}", context=None)
        finally:
            if callable(_orig_threading_hook):
                try:
                    _orig_threading_hook(args)
                except Exception:
                    pass
        if (os.environ.get("CODEX_STRICT_EXIT", "1") or "").strip() not in ("0", "false", "False"):
            os._exit(1)

    sys.excepthook = _dump_and_forward
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook  # type: ignore[assignment]

    _installed = True
    return path


def attach_asyncio(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Attach an asyncio exception handler to capture unhandled task errors."""
    try:
        lp = loop or asyncio.get_event_loop()
    except Exception:
        return

    def _handler(l, context):  # type: ignore[no-untyped-def]
        exc = context.get("exception")
        if exc is not None:
            dump_exception(type(exc), exc, exc.__traceback__, where="asyncio", context={k: str(v) for k, v in context.items() if k != "exception"})
        else:
            # No explicit exception: synthesize a pseudo-trace from message
            msg = context.get("message") or "asyncio: unknown error"
            dump_message(msg, where="asyncio", context={k: str(v) for k, v in context.items()})

    try:
        lp.set_exception_handler(_handler)
    except Exception:
        pass


def dump_message(message: str, where: str = "manual", context: Optional[Dict[str, Any]] = None) -> str:
    """Dump a textual error-like message with context to the exceptions log."""
    path = _ensure_log_path()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"===== MESSAGE {ts} pid={os.getpid()} thread={threading.current_thread().name} where={where}\n"
    )
    ctx = "" if not context else "context=" + ", ".join(f"{k}={v}" for k, v in context.items())
    body = f"{message}\n{ctx}\n\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(body)
    finally:
        _stderr_notice(path)
    return path


def dump_exception(exc_type, exc, tb, where: str = "unhandled", context: Optional[Dict[str, Any]] = None):  # type: ignore[no-untyped-def]
    """Write full traceback to the exceptions log and emit a single-line stderr notice."""
    path = _ensure_log_path()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"===== EXCEPTION {ts} pid={os.getpid()} thread={threading.current_thread().name} where={where}\n"
    )
    tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
    ctx = "" if not context else "context=" + ", ".join(f"{k}={v}" for k, v in context.items())
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(tb_text)
            if ctx:
                f.write(ctx + "\n")
            f.write("\n")
    finally:
        _stderr_notice(path)
    return path


def dump_current_exception(where: str = "manual", context: Optional[Dict[str, Any]] = None) -> str:
    exc_type, exc, tb = sys.exc_info()
    if exc_type is None or exc is None:
        return dump_message("dump_current_exception called with no active exception", where=where, context=context)
    return dump_exception(exc_type, exc, tb, where=where, context=context)


def _stderr_notice(path: str) -> None:
    try:
        sys.stderr.write(f"[EXC] Dumped exception to {path}\n")
        sys.stderr.flush()
    except Exception:
        pass


def get_log_path() -> Optional[str]:
    return _log_path
