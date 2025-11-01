from __future__ import annotations

"""Debug call tracing (entry/exit) for pipeline functions.

Enable via environment variables (all optional):
  - CODEX_CALLTRACE=1               → master on/off
  - CODEX_CALLTRACE_INCLUDE=csv     → module prefixes to instrument (default covers engines/runtime)
  - CODEX_CALLTRACE_EXCLUDE=csv     → module prefixes to skip (comma-separated)
  - CODEX_CALLTRACE_LIMIT=int       → max functions to wrap per module (default 1000)
  - CODEX_CALLTRACE_ARGS=int        → max args to summarize (default 6)
  - CODEX_CALLTRACE_DEPTH=int       → indentation depth to show (default 32)

Logs use logger of the target module at DEBUG level and print:
  [enter] fqname(arg=summary, ...)
  [exit ] fqname -> return_summary (dt=ms)
"""

import importlib
import inspect
import logging
import os
import time
from types import ModuleType, FunctionType
from typing import Any, Callable
from functools import wraps


_WRAPPED_ATTR = "__codex_calltrace_wrapped__"
_INDENT = 0


def _is_enabled() -> bool:
    return str(os.getenv("CODEX_CALLTRACE", "0")).strip().lower() in ("1", "true", "yes", "on")


def _split_csv(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default or []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _summarize(value: Any) -> str:
    try:
        import torch
        import numpy as np  # type: ignore
    except Exception:
        torch = None  # type: ignore
        np = None  # type: ignore

    try:
        if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[arg-type]
            shape = tuple(value.shape)
            return f"Tensor(shape={shape}, dtype={getattr(value, 'dtype', '?')}, device={getattr(value, 'device', '?')})"
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[arg-type]
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        if isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:5]})"
        if isinstance(value, (int, float, str, bool)) or value is None:
            s = repr(value)
            return s if len(s) <= 80 else s[:77] + "..."
        return f"{type(value).__name__}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"{type(value).__name__}(!{exc})"


def _format_args(args: tuple[Any, ...], kwargs: dict[str, Any], max_items: int) -> str:
    items: list[str] = []
    for i, a in enumerate(args):
        if i >= max_items:
            items.append("…")
            break
        items.append(_summarize(a))
    for i, (k, v) in enumerate(kwargs.items()):
        if len(items) >= max_items:
            items.append("…")
            break
        items.append(f"{k}={_summarize(v)}")
    return ", ".join(items)


def _wrap_function(mod_logger: logging.Logger, fqname: str, fn: Callable[..., Any], *, max_args: int) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _INDENT
        indent = " " * min(_INDENT, _DEPTH)
        mod_logger.debug("%s[enter] %s(%s)", indent, fqname, _format_args(args, kwargs, max_args))
        _INDENT += 2
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000.0
            _INDENT -= 2
            mod_logger.debug("%s[exit ] %s -> %s (dt=%.2fms)", indent, fqname, _summarize(result), dt)
            return result
        except Exception:
            _INDENT -= 2
            mod_logger.debug("%s[exit!] %s raised", indent, fqname)
            raise

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def _should_wrap(obj: Any) -> bool:
    if getattr(obj, _WRAPPED_ATTR, False):
        return False
    return callable(obj)


def _iter_public_functions(mod: ModuleType) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj):
            out.append((name, obj))
        elif inspect.isclass(obj):
            for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                out.append((f"{obj.__name__}.{meth_name}", meth))
    return out


_DEFAULT_INCLUDE = (
    "apps.backend.runtime.workflows",
    "apps.backend.runtime.sampling",
    "apps.backend.runtime.text_processing",
    "apps.backend.engines",
)
_EXCLUDE: list[str] = []
_LIMIT = int(os.getenv("CODEX_CALLTRACE_LIMIT", "1000") or "1000")
_MAX_ARGS = int(os.getenv("CODEX_CALLTRACE_ARGS", "6") or "6")
_DEPTH = int(os.getenv("CODEX_CALLTRACE_DEPTH", "32") or "32")


def instrument_module(mod: ModuleType) -> int:
    wrapped = 0
    logger = logging.getLogger(getattr(mod, "__name__", "backend.calltrace"))
    for name, fn in _iter_public_functions(mod):
        try:
            if not _should_wrap(fn):
                continue
            fqname = f"{mod.__name__}.{name}"
            if inspect.isfunction(fn):
                setattr(mod, name, _wrap_function(logger, fqname, fn, max_args=_MAX_ARGS))
                wrapped += 1
            elif inspect.ismethod(fn):
                # Shouldn't reach because we collect functions; keep for completeness
                setattr(mod, name, _wrap_function(logger, fqname, fn, max_args=_MAX_ARGS))
                wrapped += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("calltrace: skip %s.%s (%s)", mod.__name__, name, exc)
        if wrapped >= _LIMIT:
            break
    return wrapped


def setup_from_env() -> None:
    if not _is_enabled():
        return
    includes = _split_csv("CODEX_CALLTRACE_INCLUDE", list(_DEFAULT_INCLUDE))
    excludes = _split_csv("CODEX_CALLTRACE_EXCLUDE", [])
    global _EXCLUDE
    _EXCLUDE = excludes
    total = 0
    for prefix in includes:
        try:
            mod = importlib.import_module(prefix)
        except Exception as exc:
            logging.getLogger("backend.calltrace").debug("calltrace: failed to import %s: %s", prefix, exc)
            continue
        # instrument this module and submodules one level deep
        total += instrument_module(mod)
        # Submodules
        if hasattr(mod, "__path__"):
            import pkgutil
            for info in pkgutil.iter_modules(mod.__path__, prefix + "."):
                # Skip excluded prefixes
                if any(info.name.startswith(ex) for ex in _EXCLUDE):
                    continue
                try:
                    sub = importlib.import_module(info.name)
                    total += instrument_module(sub)
                except Exception as exc:
                    logging.getLogger("backend.calltrace").debug("calltrace: failed to import %s: %s", info.name, exc)
                    continue
    logging.getLogger("backend.calltrace").info("calltrace: enabled, wrapped=%d functions", total)


__all__ = ["setup_from_env", "instrument_module"]

