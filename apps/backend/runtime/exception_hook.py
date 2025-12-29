"""Centralized exception capture and full-traceback dumper.

Usage:
- Call `install_exception_hooks(log_dir=...)` as early as possible in process startup.
- Optionally call `attach_asyncio(loop)` to capture unhandled asyncio task exceptions.
- For caught exceptions you still want to persist, call
  `dump_exception(exc_type, exc, tb, where=..., context=...)` or
  `dump_current_exception(where=..., context=...)`.

Design notes:
- Writes to a stable file per (day, pid): logs/exceptions-YYYYmmdd-<pid>.log
  to keep append-only and avoid unbounded file count.
- Emits a one-line notice to stderr on every dump so the TUI will display a
  confirmation line with the output path.
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
