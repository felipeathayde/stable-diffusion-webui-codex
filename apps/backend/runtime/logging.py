"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Centralized backend logging setup (env-driven level, optional rich/tqdm integration).
Configures root logging once per interpreter, supports console/file handlers, and can wrap stream handlers to cooperate with tqdm progress
bars. Level filtering can also be controlled per-level via `CODEX_LOG_*` env vars.

Symbols (top-level; keep in sync; no ghosts):
- `TqdmAwareHandler` (class): Proxy handler that cooperates with tqdm-managed progress bars.
- `_is_stream_handler` (function): Detects stream handlers including wrapped `TqdmAwareHandler`.
- `_parse_level` (function): Parses level names (including TRACE=5) and returns a logging level.
- `format_log_message` (function): Builds a consistent event-style log message with optional key/value context.
- `get_backend_logger` (function): Returns a normalized backend logger (`backend.*`) from module or relative names.
- `LevelFilter` (class): Env-driven log-level filter (CODEX_LOG_DEBUG/INFO/WARNING/ERROR).
- `setup_logging` (function): Idempotent root logger setup using env vars (level/format/file, optional Rich handler).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

try:  # Optional; fall back to plain logs when missing
    from colorama import init as _colorama_init  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _colorama_init = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from rich.console import Console  # type: ignore
    from rich.logging import RichHandler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Console = None  # type: ignore[assignment]
    RichHandler = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

if _colorama_init is not None:  # pragma: no cover - environment dependent
    _colorama_init(autoreset=True)

_CONFIGURED = False
_SAFE_LOG_TOKEN = re.compile(r"^[A-Za-z0-9._:/-]+$")


class TqdmAwareHandler(logging.Handler):
    """Proxy handler that cooperates with tqdm-managed progress bars."""

    def __init__(self, inner: logging.Handler) -> None:
        super().__init__()
        self.inner = inner

    def setFormatter(self, fmt: logging.Formatter) -> None:  # noqa: N802 (logging API)
        super().setFormatter(fmt)
        self.inner.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI integration
        if tqdm is not None and getattr(tqdm, "_instances", None):
            try:
                tqdm.write(self.format(record))
                return
            except Exception:
                # Fall back to the wrapped handler on unexpected errors
                pass
        self.inner.emit(record)


def _is_stream_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.StreamHandler):
        return True
    return isinstance(handler, TqdmAwareHandler) and isinstance(handler.inner, logging.StreamHandler)


def _parse_level(value: Optional[str]) -> int:
    if not value:
        return logging.DEBUG
    v = value.strip().upper()
    mapping = {
        "TRACE": 5,  # custom: lower than DEBUG if someone sets it
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    resolved = mapping.get(v, logging.DEBUG)
    logging.addLevelName(5, "TRACE")
    return resolved


def format_log_message(event: str, /, **fields: object) -> str:
    """Build a consistent event-style log message.

    Output format:
    - without fields: `event`
    - with fields: `event | key=value key2='text value'`
    """

    tokens: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        safe_key = key if _SAFE_LOG_TOKEN.fullmatch(key) else repr(key)
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        elif isinstance(value, str):
            rendered = value if _SAFE_LOG_TOKEN.fullmatch(value) else repr(value)
        else:
            rendered = repr(value)
        tokens.append(f"{safe_key}={rendered}")

    safe_event = event if _SAFE_LOG_TOKEN.fullmatch(event) else repr(event)
    if not tokens:
        return safe_event
    return f"{safe_event} | {' '.join(tokens)}"


def get_backend_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a backend logger with normalized namespace.

    Accepted forms:
    - `apps.backend`/`backend` -> `backend`
    - `apps.backend.<...>` -> `backend.<...>`
    - `backend.<...>`      -> unchanged
    - `<relative>`         -> `backend.<relative>`
    - empty/None           -> `backend`
    """

    if name is None:
        return logging.getLogger("backend")

    normalized = name.strip()
    if not normalized:
        return logging.getLogger("backend")
    normalized = normalized.strip(".")
    if not normalized:
        return logging.getLogger("backend")

    if normalized in {"backend", "apps.backend"}:
        normalized = "backend"
    elif normalized.startswith("apps.backend."):
        normalized = "backend." + normalized[len("apps.backend.") :]
    elif not normalized.startswith("backend."):
        normalized = "backend." + normalized
    normalized = normalized.rstrip(".")
    normalized = ".".join(part for part in normalized.split(".") if part)
    if not normalized:
        normalized = "backend"

    return logging.getLogger(normalized)


class LevelFilter(logging.Filter):
    """Filter that enables/disables log levels based on individual env vars.
    
    Checks CODEX_LOG_DEBUG, CODEX_LOG_INFO, CODEX_LOG_WARNING, CODEX_LOG_ERROR.
    If set to "1"/"true"/"yes"/"on", that level is allowed.
    If set to "0"/"false"/"no"/"off", that level is blocked.
    If not set, defaults apply (DEBUG=0, INFO=1, WARNING=1, ERROR=1).
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._level_flags = self._read_flags()
    
    def _read_flags(self) -> dict[int, bool]:
        """Read level flags from environment."""
        def _is_enabled(env_var: str, default: str) -> bool:
            val = os.environ.get(env_var, default).strip().lower()
            return val in ("1", "true", "yes", "on")
        
        return {
            logging.DEBUG: _is_enabled("CODEX_LOG_DEBUG", "0"),
            logging.INFO: _is_enabled("CODEX_LOG_INFO", "1"),
            logging.WARNING: _is_enabled("CODEX_LOG_WARNING", "1"),
            logging.ERROR: _is_enabled("CODEX_LOG_ERROR", "1"),
            logging.CRITICAL: True,  # Always show critical
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be logged."""
        return self._level_flags.get(record.levelno, True)


def setup_logging(level: Optional[str] = None, *, install_tqdm_bridge: bool = True) -> None:
    """Initialize root logger based on env vars, only once.

    - Sets root level to env-provided level (default DEBUG).
    - Adds a stderr StreamHandler with a concise, actionable format.
    - Optionally adds a file handler if CODEX_LOG_FILE is set.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Determine level
    level_name = level if level is not None else (
        os.environ.get("CODEX_LOG_LEVEL")
        or os.environ.get("SDWEBUI_LOG_LEVEL")
        or os.environ.get("WEBUI_LOG_LEVEL")
    )
    resolved_level = _parse_level(level_name)

    # Concise format: [MM/DD/YY HH:MM:SS] LEVEL     message
    fmt = os.environ.get(
        "CODEX_LOG_FORMAT",
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
    )
    datefmt = "%m/%d/%y %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Avoid duplicate handlers if upstream configured something already
    def _should_force_plain() -> bool:
        env = os.environ.get("SD_WEBUI_NO_RICH") or os.environ.get("CODEX_LOG_NO_RICH")
        if not env:
            return False
        return env.strip().lower() in {"1", "true", "yes", "on"}

    def _env_true(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _build_stream_handler() -> logging.Handler:
        level_filter = LevelFilter()
        formatter: logging.Formatter
        if not _should_force_plain() and RichHandler is not None and Console is not None:
            console = Console(color_system="auto", soft_wrap=True, highlight=False, emoji=False)
            inner: logging.Handler = RichHandler(
                console=console,
                show_time=True,
                show_path=_env_true("CODEX_LOG_RICH_SHOW_PATH", "0"),
                rich_tracebacks=_env_true("CODEX_LOG_RICH_TRACEBACKS", "1"),
                markup=False,
            )
            formatter = logging.Formatter("%(name)s | %(message)s")
        else:
            inner = logging.StreamHandler(stream=sys.stderr)
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        if install_tqdm_bridge and tqdm is not None:
            handler: logging.Handler = TqdmAwareHandler(inner)
        else:
            handler = inner
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        handler.addFilter(level_filter)
        return handler

    if not any(_is_stream_handler(h) for h in root.handlers):
        root.addHandler(_build_stream_handler())

    # Ensure a dedicated handler for the 'backend' logger hierarchy so DEBUG
    # logs are not filtered by third-party handlers (e.g., uvicorn/gradio)
    codex = logging.getLogger("backend")
    codex.setLevel(resolved_level)
    # mark our handler to avoid duplicates on re-entry
    has_codex = False
    for h in codex.handlers:
        if getattr(h, "_codex", False):
            has_codex = True
            break
    if not has_codex:
        h = _build_stream_handler()
        setattr(h, "_codex", True)
        codex.addHandler(h)
    # prevent double printing via root handlers
    codex.propagate = False

    log_file = os.environ.get("CODEX_LOG_FILE")
    file_handler: logging.FileHandler | None = None
    if log_file:
        abs_log_file = os.path.abspath(log_file)
        for h in root.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_file:
                file_handler = h
                break

        if file_handler is None:
            try:
                fh = logging.FileHandler(abs_log_file, encoding="utf-8")
                fh.setLevel(resolved_level)
                fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
                fh.addFilter(LevelFilter())
                root.addHandler(fh)
                file_handler = fh
            except Exception:
                # If file handler fails, keep stderr logging only; do not crash startup
                logging.getLogger(__name__).exception("Failed to attach file handler: %s", log_file)

    # `backend` logger has `propagate=False`, so it won't reach root's file handler.
    # Attach the file handler explicitly so launcher users actually get backend logs.
    if file_handler is not None and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == getattr(file_handler, "baseFilename", None)
        for h in codex.handlers
    ):
        codex.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "logging configured level=%s file=%s handlers=%d",
        logging.getLevelName(resolved_level),
        log_file or "<stderr-only>",
        len(root.handlers),
    )

    _CONFIGURED = True
