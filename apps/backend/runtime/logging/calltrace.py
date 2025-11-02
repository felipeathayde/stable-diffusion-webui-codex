from __future__ import annotations

"""Simple call tracing (entry/exit) controlled by CODEX_CALLTRACE.

When `CODEX_CALLTRACE=1`, we instrument the main pipeline modules
(`apps.backend.runtime`, `apps.backend.engines`, `apps.backend.use_cases`,
`apps.backend.codex`) and log entry/exit at DEBUG level.

Instrumentation installs a dedicated calltrace handler so DEBUG messages surface
even if the global logging level remains INFO.

No additional configuration knobs; goal is a one-flag on/off switch.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import threading
import time
from functools import wraps
from types import ModuleType
from typing import Any, Callable


WRAPPED_FLAG = "__codex_calltrace_wrapped__"
ROOT_PREFIXES = (
    "apps.backend.runtime",
    "apps.backend.engines",
    "apps.backend.use_cases",
    "apps.backend.codex",
)
SKIP_PREFIXES = (
    "apps.backend.runtime.logging.calltrace",
)
MAX_FUNCS_PER_MODULE = 5000
MAX_ARGS = 6
MAX_DEPTH = 64
INDENT_STEP = 2

_indent = 0
CALLTRACE_RECORD_ATTR = "_codex_calltrace_record"
LOGGER_PREPARED_ATTR = "__codex_calltrace_logger__"
_CALLTRACE_HANDLER: logging.Handler | None = None
_ROOT_FILTER: logging.Filter | None = None
_HANDLER_LOCK = threading.RLock()


class _CalltraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, CALLTRACE_RECORD_ATTR, False))


class _SuppressCalltraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, CALLTRACE_RECORD_ATTR, False)


def _calltrace_extra() -> dict[str, bool]:
    return {CALLTRACE_RECORD_ATTR: True}


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_CalltraceFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [calltrace] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


def _ensure_root_filter() -> None:
    global _ROOT_FILTER
    root = logging.getLogger()
    if _ROOT_FILTER is None:
        _ROOT_FILTER = _SuppressCalltraceFilter()
    for handler in root.handlers:
        if _ROOT_FILTER not in getattr(handler, "filters", []):
            handler.addFilter(_ROOT_FILTER)


def _ensure_handler() -> logging.Handler:
    global _CALLTRACE_HANDLER
    with _HANDLER_LOCK:
        if _CALLTRACE_HANDLER is None:
            _CALLTRACE_HANDLER = _build_handler()
        _ensure_root_filter()
        return _CALLTRACE_HANDLER


def _prepare_logger(logger: logging.Logger) -> None:
    if getattr(logger, LOGGER_PREPARED_ATTR, False):
        return
    handler = _ensure_handler()
    if handler not in logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    setattr(logger, LOGGER_PREPARED_ATTR, True)


def _enabled() -> bool:
    return str(os.getenv("CODEX_CALLTRACE", "0")).strip().lower() in ("1", "true", "yes", "on")


def _summarize(value: Any) -> str:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    try:
        if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[arg-type]
            return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[arg-type]
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        if isinstance(value, dict):
            keys = list(value.keys())
            preview = keys[:5]
            suffix = "…" if len(keys) > 5 else ""
            return f"dict(keys={preview}{suffix})"
        if isinstance(value, (int, float, str, bool)) or value is None:
            s = repr(value)
            return s if len(s) <= 80 else s[:77] + "..."
        return type(value).__name__
    except Exception as exc:  # pragma: no cover - defensive
        return f"{type(value).__name__}(!{exc})"


def _format_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    items: list[str] = []
    for i, arg in enumerate(args):
        if i >= MAX_ARGS:
            items.append("…")
            break
        items.append(_summarize(arg))
    for key, value in kwargs.items():
        if len(items) >= MAX_ARGS:
            items.append("…")
            break
        items.append(f"{key}={_summarize(value)}")
    return ", ".join(items)


def _wrap(logger: logging.Logger, fqname: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _indent
        indent = " " * min(_indent, MAX_DEPTH)
        logger.debug("%s[enter] %s(%s)", indent, fqname, _format_args(args, kwargs), extra=_calltrace_extra())
        _indent += INDENT_STEP
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000.0
            _indent -= INDENT_STEP
            logger.debug(
                "%s[exit ] %s -> %s (dt=%.2fms)", indent, fqname, _summarize(result), duration, extra=_calltrace_extra()
            )
            return result
        except Exception:
            _indent -= INDENT_STEP
            logger.debug("%s[exit!] %s raised", indent, fqname, extra=_calltrace_extra())
            raise

    setattr(wrapper, WRAPPED_FLAG, True)
    return wrapper


def _should_wrap(obj: Any) -> bool:
    return callable(obj) and not getattr(obj, WRAPPED_FLAG, False)


def _instrument_module(mod: ModuleType) -> int:
    wrapped = 0
    logger = logging.getLogger(mod.__name__)
    _prepare_logger(logger)
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and _should_wrap(obj):
            setattr(mod, name, _wrap(logger, f"{mod.__name__}.{name}", obj))
            wrapped += 1
        elif inspect.isclass(obj):
            for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                if not _should_wrap(meth):
                    continue
                fqname = f"{mod.__name__}.{obj.__name__}.{meth_name}"
                wrapped_method = _wrap(logger, fqname, meth)
                setattr(obj, meth_name, wrapped_method)
                wrapped += 1
        if wrapped >= MAX_FUNCS_PER_MODULE:
            break
    return wrapped


def _instrument_recursive(mod: ModuleType, visited: set[str]) -> int:
    name = getattr(mod, "__name__", "")
    if not name or name in visited:
        return 0
    if any(name.startswith(prefix) for prefix in SKIP_PREFIXES):
        return 0
    visited.add(name)
    total = _instrument_module(mod)
    if hasattr(mod, "__path__"):
        for info in pkgutil.iter_modules(mod.__path__, prefix=name + "."):
            try:
                sub = importlib.import_module(info.name)
            except Exception:
                continue
            total += _instrument_recursive(sub, visited)
    return total


def setup_from_env() -> None:
    if not _enabled():
        return
    visited: set[str] = set()
    total = 0
    for prefix in ROOT_PREFIXES:
        try:
            mod = importlib.import_module(prefix)
        except Exception as exc:
            logging.getLogger("backend.calltrace").warning("calltrace: failed import %s (%s)", prefix, exc)
            continue
        total += _instrument_recursive(mod, visited)
    logging.getLogger("backend.calltrace").info("calltrace enabled (wrapped=%d functions)", total)


__all__ = ["setup_from_env"]
