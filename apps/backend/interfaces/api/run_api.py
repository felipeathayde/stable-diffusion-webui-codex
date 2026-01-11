# // tags: api-entrypoint, fastapi, task-router
"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FastAPI entrypoint + uvicorn factory for the Codex WebUI backend.
This module builds the `/api/*` surface by assembling router modules (generation/tasks/models/options/tools/ui persistence),
and mounts the built UI as SPA static files after API routes.

Symbols (top-level; keep in sync; no ghosts):
- `_cli_arg_value` (function): Reads a CLI flag value from argv (supports `--flag value` and `--flag=value` forms).
- `_parse_trace_max` (function): Parses `--trace-debug-max-per-func` into a non-negative int (or `None`).
- `ensure_initialized` (function): Performs early runtime bootstrap (repo root/sys.path, optional tracing/logging hooks) before serving.
- `_SuppressUvicornAccessNoiseFilter` (class): Logging filter to reduce uvicorn access-log spam for noisy endpoints.
- `_install_uvicorn_access_noise_filter` (function): Installs `_SuppressUvicornAccessNoiseFilter` when configured.
- `port_free` (function): Checks whether a TCP port is free on IPv4/IPv6 loopback/wildcard.
- `scan_range` (function): Scans a port range to find a free port.
- `pick_api_port_simple` (function): Picks a free port near a base port (and reports whether it was the base).
- `banner` (function): Prints the startup banner with the selected port.
- `_DummyRequest` (class): Minimal request shim used where a request-like object is needed without FastAPI internals.
- `build_app` (function): Constructs the FastAPI app; wires router modules, configures middleware, and mounts the UI SPA.
- `_bootstrap_runtime` (function): Bootstraps runtime settings/env before app creation (used by the uvicorn factory path).
- `_enable_trace_debug` (function): Enables global tracing/debug logging when requested via argv/env.
- `create_api_app` (function): Canonical uvicorn `--factory` entrypoint; calls bootstrap and returns the built FastAPI app.
- `main` (function): CLI entrypoint used by launchers (selects port, builds app, runs uvicorn).
"""

import asyncio
import errno
import os
import socket
import sys
from contextlib import closing
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apps.backend.infra.config.provenance import generation_provenance as _generation_provenance
from apps.backend.services.output_service import save_generated_images as _save_generated_images
from apps.backend.services.media_service import MediaService
from apps.backend.services.live_preview_service import LivePreviewService
from apps.backend.interfaces.api.path_utils import CODEX_ROOT
from apps.backend.interfaces.api.json_store import _save_json
from apps.backend.interfaces.api.routers import generation, models, options, paths, settings, system, tasks, tools, ui
from apps.backend.services import options_store
from apps.backend.infra.config import args as config_args
from apps.backend.runtime.pipeline_debug import apply_env_flag as _apply_pipeline_debug_flag
from apps.backend.runtime.memory import memory_management as mem_management
from apps.backend.runtime.models import api as model_api
from apps.backend.core.state import state as backend_state

def _cli_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(argv):
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _parse_trace_max(argv: Sequence[str]) -> Optional[int]:
    value = _cli_arg_value(argv, "--trace-debug-max-per-func")
    if value is None:
        return None
    try:
        numeric = int(value)
    except Exception:
        return None
    return max(0, numeric)

# Early tracing hook: if --trace-debug is present (or env truthy), configure
# logging at DEBUG and enable global call tracing before importing FastAPI/uvicorn.
try:
    if ("--trace-debug" in sys.argv) or (os.getenv("CODEX_TRACE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}):
        from apps.backend.runtime import logging as runtime_logging  # type: ignore
        runtime_logging.setup_logging(level="DEBUG")
        from apps.backend.runtime import call_trace as _call_trace  # type: ignore
        max_per_func = _parse_trace_max(sys.argv[1:])
        _call_trace.enable(max_calls_per_func=max_per_func)
except Exception:
    # Never block startup because of tracing/logging issues
    pass

try:
    from colorama import Fore, Style  # type: ignore

    def color_cyan(s: str) -> str: return Fore.CYAN + s + Style.RESET_ALL
    def color_red(s: str) -> str: return Fore.RED + s + Style.RESET_ALL
except Exception:  # pragma: no cover - optional dependency missing

    def color_cyan(s: str) -> str: return s
    def color_red(s: str) -> str: return s

# Install global exception hooks as early as possible so any startup errors are dumped
try:
    from apps.backend.runtime.exception_hook import install_exception_hooks as _install_exc_hooks
    _EXC_LOG_PATH = _install_exc_hooks(log_dir=str(CODEX_ROOT / 'logs'))
except Exception:
    _EXC_LOG_PATH = None


_initialized = False
_RUNTIME_NAMESPACE: Optional[Any] = None
_APP: Optional[FastAPI] = None


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    # Configure root logging so engine/runtime INFO logs are visible in console
    try:
        from apps.backend.runtime import logging as runtime_logging

        runtime_logging.setup_logging(level="INFO")
    except Exception:
        # Fall back to a minimal configuration on failure.
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    _initialized = True

_UVICORN_ACCESS_NOISE_PREFIXES = ("/api/tools/convert-gguf/",)
_UVICORN_ACCESS_NOISE_FILTER_INSTALLED = False


class _SuppressUvicornAccessNoiseFilter(logging.Filter):
    def __init__(self, suppress_path_prefixes: Optional[List[str]] = None) -> None:
        super().__init__()
        self._prefixes = tuple(suppress_path_prefixes or [])

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - depends on uvicorn internals
        if not self._prefixes:
            return True

        try:
            path: Optional[str] = None
            args = getattr(record, "args", None)
            if isinstance(args, tuple) and len(args) >= 3:
                path = str(args[2])
            elif isinstance(args, dict):
                raw = args.get("path") or args.get("raw_path")
                if raw is not None:
                    path = str(raw)

            if path is None:
                request_line = getattr(record, "request_line", None)
                if isinstance(request_line, str):
                    parts = request_line.split(" ")
                    if len(parts) >= 2:
                        path = parts[1]

            if path is None:
                msg = record.getMessage()
                for prefix in self._prefixes:
                    if prefix in msg:
                        return False
                return True

            path_only = path.split("?", 1)[0]
            return not path_only.startswith(self._prefixes)
        except Exception:
            return True


def _install_uvicorn_access_noise_filter() -> None:
    """Suppress noisy uvicorn access logs for high-frequency polling endpoints.

    The UI polls some tool endpoints (e.g. GGUF conversion progress). Uvicorn logs
    every request at INFO, flooding the console during long conversions.
    """
    global _UVICORN_ACCESS_NOISE_FILTER_INSTALLED
    if _UVICORN_ACCESS_NOISE_FILTER_INSTALLED:
        return

    allow_tools = os.getenv("CODEX_UVICORN_ACCESS_LOG_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}
    if allow_tools:
        return

    logger = logging.getLogger("uvicorn.access")
    logger.addFilter(_SuppressUvicornAccessNoiseFilter(list(_UVICORN_ACCESS_NOISE_PREFIXES)))
    _UVICORN_ACCESS_NOISE_FILTER_INSTALLED = True


# This module is commonly loaded via `python -m uvicorn --factory ...:create_api_app`.
# Install the access-log filter at import time so it applies even when uvicorn is launched
# via CLI (no custom log_config passed).
_install_uvicorn_access_noise_filter()


def port_free(port: int, host: str = '0.0.0.0') -> bool:
    """Return True if a port looks free across common bind targets.

    This check is intentionally conservative to avoid split-brain situations
    where something is already bound on IPv6 loopback (::1) while we only test
    IPv4 (0.0.0.0). That exact setup can make `localhost` resolve to the wrong
    service with *no* obvious bind error.

    We treat the port as "busy" if any of these binds fail with EADDRINUSE:
    - IPv4 wildcard + loopback
    - IPv6 wildcard + loopback (when supported)
    """

    def _can_bind(family: int, bind_host: str) -> bool:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    addr = (bind_host, port, 0, 0)
                else:
                    addr = (bind_host, port)
                s.bind(addr)
                return True
        except OSError as exc:
            # Ignore unsupported address families on systems without IPv6.
            if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                return True
            return False

    bind_targets: list[tuple[int, str]] = []
    host_norm = str(host or "").strip().lower()
    if host_norm in {"", "0.0.0.0", "0", "*"}:
        bind_targets.extend([(socket.AF_INET, "0.0.0.0"), (socket.AF_INET, "127.0.0.1")])
    elif host_norm == "localhost":
        bind_targets.append((socket.AF_INET, "127.0.0.1"))
    else:
        bind_targets.append((socket.AF_INET, host))

    # Also check IPv6 loopback/wildcard (when available).
    bind_targets.extend([(socket.AF_INET6, "::"), (socket.AF_INET6, "::1")])

    return all(_can_bind(fam, h) for fam, h in bind_targets)


def scan_range(r: Tuple[int, int], host: str = '0.0.0.0') -> Optional[int]:
    start, end = int(r[0]), int(r[1])
    for p in range(start, end + 1):
        if port_free(p, host):
            return p
    return None


def pick_api_port_simple(base: int, host: str = '0.0.0.0') -> Tuple[int, bool]:
    # Try base -> base+10000 -> base+20000
    for i, candidate in enumerate((base, base + 10000, base + 20000)):
        if port_free(candidate, host):
            return candidate, (i != 0)
    raise RuntimeError(f'No free API port among {base}, {base+10000}, {base+20000}')


def banner(port: int) -> None:
    msg = (
        "\n"
        "==============================================\n"
        "  PORT GUARD ACTIVATED — API Fallback        \n"
        "==============================================\n"
        f" Using API port {port}.                       \n"
        " Tip: set API_PORT or free blocked range.     \n"
        "==============================================\n"
    )
    print(color_cyan(msg))


class _DummyRequest:
    def __init__(self, username: str = "api") -> None:
        self.username = username


def build_app() -> FastAPI:
    ensure_initialized()

    # Native parameter helpers (replace legacy _txt2img/_img2img parsers)
    from apps.backend.services import param_utils as _p


    app = FastAPI()

    # middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    media = MediaService()
    live_preview = LivePreviewService()
    # Native options facade (JSON-backed). Import early so helpers are available
    # to any route or startup function defined below.
    from apps.backend.services.options_store import (
        get_value as _opts_get,
        set_values as _opts_set_many,
        get_snapshot as _opts_snapshot,
        load_values as _opts_load_native,
    )
    _settings_values_path = str(CODEX_ROOT / 'apps' / 'settings_values.json')
    _ui_dist_dir = str(CODEX_ROOT / 'apps' / 'interface' / 'dist')

    # Exception hooks for asyncio + HTTP middleware to dump unhandled route exceptions
    try:
        from apps.backend.runtime.exception_hook import attach_asyncio as _attach_asyncio, dump_current_exception as _dump_current_exception

        @app.on_event('startup')
        async def _setup_exc_hooks() -> None:  # pragma: no cover
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            _attach_asyncio(loop)

        @app.middleware('http')
        async def _errors_middleware(request, call_next):  # type: ignore[no-untyped-def]
            try:
                return await call_next(request)
            except Exception:
                try:
                    _dump_current_exception(where='http', context={'path': str(getattr(request, 'url', '')), 'method': getattr(request, 'method', '')})
                finally:
                    pass
                raise
    except Exception:
        pass

    # Settings registry (hardcoded dataclasses/enums via codegen)
    try:
        # Generated from scripts/generate_settings_registry.py
        from apps.backend.interfaces.schemas.settings_registry import (  # type: ignore
            schema_to_json as _schema_hardcoded,
            field_index as _field_index,
            SettingType as _SettingType,
        )
        _settings_registry_ok = True
    except Exception as _e:  # pragma: no cover - optional during transition
        print(color_red(f"[settings] registry not available: {_e}"))
        _settings_registry_ok = False
        _schema_hardcoded = None

        def _field_index() -> dict[str, Any]:
            return {}

        _SettingType = None

    # Load saved settings on startup and apply to shared.opts with validation
    def _apply_saved_settings() -> None:
        if not _settings_registry_ok:
            return
        saved = _opts_load_native()
        if not isinstance(saved, dict) or not saved:
            return
        idx = _field_index()
        # Validate and normalize persisted values against schema, then re-save
        changed = False
        for k in list(saved.keys()):
            f = idx.get(k)
            if not f:
                saved.pop(k, None)
                changed = True
                continue
            try:
                if getattr(f, 'choices', None) and isinstance(f.choices, list) and saved[k] not in f.choices:
                    saved.pop(k, None)
                    changed = True
                    continue
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    v = float(saved[k])
                    lo = getattr(f, 'min', None)
                    hi = getattr(f, 'max', None)
                    if isinstance(lo, (int, float)) and v < lo:
                        saved[k] = lo
                        changed = True
                    if isinstance(hi, (int, float)) and v > hi:
                        saved[k] = hi
                        changed = True
                if getattr(f, 'type', None) == _SettingType.CHECKBOX and isinstance(saved[k], str):
                    saved[k] = saved[k].lower() in ('1','true','yes','on')
                    changed = True
            except Exception:
                continue
        if changed:
            _save_json(_settings_values_path, saved)

    # Apply saved settings early (after modules init) before serving
    try:
        _apply_saved_settings()
    except Exception as e:  # pragma: no cover
        print(color_red(f"[settings] failed to validate saved settings: {e}"))

    # Honour pipeline debug env flag
    _apply_pipeline_debug_flag()

    # Register routers
    app.include_router(system.build_router(app_version=options_store.get_snapshot().as_dict().get("app_version", "")))
    app.include_router(settings.build_router(
        codex_root=CODEX_ROOT,
        settings_registry_ok=_settings_registry_ok,
        schema_hardcoded=_schema_hardcoded,
        field_index=_field_index,
        opts_load_native=_opts_load_native,
    ))
    app.include_router(ui.build_router(
        codex_root=CODEX_ROOT,
        opts_load_native=_opts_load_native,
        opts_set_many=_opts_set_many,
        model_api=model_api,
    ))
    app.include_router(models.build_router(
        codex_root=CODEX_ROOT,
        opts_load_native=_opts_load_native,
        opts_get=_opts_get,
        model_api=model_api,
    ))
    app.include_router(paths.build_router(codex_root=CODEX_ROOT))
    app.include_router(options.build_router(
        opts_load_native=_opts_load_native,
        opts_snapshot=_opts_snapshot,
        opts_set_many=_opts_set_many,
        settings_registry_ok=_settings_registry_ok,
        field_index=_field_index,
        setting_type=_SettingType,
    ))
    app.include_router(tasks.build_router(codex_root=CODEX_ROOT, backend_state=backend_state))
    app.include_router(tools.build_router(codex_root=CODEX_ROOT))
    app.include_router(generation.build_router(
        codex_root=CODEX_ROOT,
        media=media,
        live_preview=live_preview,
        opts_get=_opts_get,
        opts_snapshot=_opts_snapshot,
        generation_provenance=_generation_provenance,
        save_generated_images=_save_generated_images,
        param_utils=_p,
    ))

    # Serve built UI (Vite build) if present, with SPA fallback
    class SPAStaticFiles(StaticFiles):  # type: ignore[misc]
        async def get_response(self, path: str, scope):  # type: ignore[override]
            try:
                return await super().get_response(path, scope)
            except Exception as exc:
                try:
                    from starlette.exceptions import HTTPException as StarletteHTTPException  # type: ignore

                    if isinstance(exc, StarletteHTTPException) and exc.status_code == 404:
                        return await super().get_response("index.html", scope)
                except Exception:
                    pass
                raise

    # Mount UI dist after API routes
    if os.path.isdir(_ui_dist_dir):
        app.mount('/', SPAStaticFiles(directory=_ui_dist_dir, html=True), name='ui')

    return app


def _bootstrap_runtime(argv: Sequence[str], env: Mapping[str, str], settings: Mapping[str, Any]) -> Any:
    global _RUNTIME_NAMESPACE
    if _RUNTIME_NAMESPACE is not None:
        return _RUNTIME_NAMESPACE
    ns, runtime_config = config_args.initialize(
        argv=argv,
        env=env,
        settings=settings,
        strict=True,
    )
    # Expose CLI debug flags as env vars for runtime modules that rely on os.getenv.
    try:
        if getattr(ns, "debug_preview_factors", False):
            os.environ["CODEX_DEBUG_PREVIEW_FACTORS"] = "1"
    except Exception:
        pass
    mem_management.reinitialize(runtime_config)
    # Pre-warm model inventory at process bootstrap so `/api/models/inventory`
    # is already hot when the UI first loads quicksettings. This avoids paying
    # the full filesystem scan cost on the first UI request.
    try:
        from apps.backend.inventory import cache as _inv_cache
        inv = _inv_cache.refresh()
        logging.getLogger("inventory").info(
            "inventory: initialized at startup (vaes=%d, text_encoders=%d, loras=%d, wan22.gguf=%d, metadata=%d)",
            len(inv.get("vaes", [])),
            len(inv.get("text_encoders", [])),
            len(inv.get("loras", [])),
            len(inv.get("wan22", [])),
            len(inv.get("metadata", [])),
        )
    except Exception as e:
        logging.getLogger("inventory").warning("inventory: failed to initialize at startup: %s", e)
    _RUNTIME_NAMESPACE = ns
    return ns


def _enable_trace_debug(ns: Any) -> None:
    try:
        if getattr(ns, "trace_debug", False):
            from apps.backend.runtime import logging as runtime_logging  # type: ignore

            runtime_logging.setup_logging(level="DEBUG")
            from apps.backend.runtime import call_trace as _call_trace  # type: ignore

            _call_trace.enable(max_calls_per_func=getattr(ns, "trace_debug_max_per_func", None))
    except Exception:
        pass


def create_api_app(*, argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> FastAPI:
    argv_seq = list(argv or [])
    snapshot = options_store.get_snapshot()
    ns = _bootstrap_runtime(argv_seq, env or os.environ, snapshot.as_dict())
    _enable_trace_debug(ns)
    ensure_initialized()
    # Build a fresh app each time to avoid stale/None globals under factory mode
    app = build_app()
    if app is None:
        raise RuntimeError("build_app() returned None")
    global _APP
    _APP = app
    return app

def main(argv: Optional[Sequence[str]] = None) -> None:
    host = '0.0.0.0'
    override = os.environ.get('API_PORT_OVERRIDE')
    used_fallback = False
    port: Optional[int] = None
    if override:
        try:
            candidate = int(override)
        except ValueError:
            candidate = None
        if candidate is not None:
            # If chosen override busy, hop by +10000
            for c in (candidate, candidate + 10000, candidate + 20000):
                if port_free(c, host):
                    port = c
                    used_fallback = (c != candidate)
                    break
        else:
            override = None  # force fallback logic
    if port is None:
        try:
            # default base 7850
            port, used_fallback = pick_api_port_simple(7850, host)
        except RuntimeError as e:
            print(color_red(f"[PORT GUARD] {e}"))
            raise SystemExit(1)

    if used_fallback:
        banner(port)

    try:
        argv_seq = list(argv) if argv is not None else sys.argv[1:]
        api_app = create_api_app(argv=argv_seq, env=os.environ)
    except Exception as exc:
        print(color_red(f"[INIT] {exc}"))
        raise SystemExit(1) from exc

    # Configure Uvicorn logging to match our unified format
    suppress_access_prefixes = list(_UVICORN_ACCESS_NOISE_PREFIXES)

    log_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "filters": {
            "codex_access_noise": {
                "()": "apps.backend.interfaces.api.run_api._SuppressUvicornAccessNoiseFilter",
                "suppress_path_prefixes": suppress_access_prefixes,
            }
        },
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)-8s %(message)s",
                "datefmt": "%m/%d/%y %H:%M:%S",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "format": "[%(asctime)s] %(levelname)-8s %(client_addr)s - %(request_line)s %(status_code)s",
                "datefmt": "%m/%d/%y %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "filters": ["codex_access_noise"],
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }
    uvicorn.run(api_app, host=host, port=port, log_level='info', log_config=log_config)


if __name__ == '__main__':
    main()
