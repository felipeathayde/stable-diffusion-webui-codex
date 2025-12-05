# // tags: api-entrypoint, fastapi, task-router
import asyncio
import time
from datetime import datetime
import base64
import io
import json
import os
import socket
import sys
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4
import logging

from apps.backend.runtime.sampling.catalog import SAMPLER_OPTIONS, SCHEDULER_OPTIONS

# Make sure our project is on sys.path before any heavy third-party imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apps.backend.codex import options as codex_options
from apps.backend.infra.config import args as config_args
from apps.backend.runtime.pipeline_debug import apply_env_flag as _apply_pipeline_debug_flag
from apps.backend.runtime.models import api as model_api
from apps.backend.runtime.memory import memory_management as mem_management
from apps.backend.core.state import state as backend_state

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
    _EXC_LOG_PATH = _install_exc_hooks(log_dir=str(PROJECT_ROOT / 'logs'))
except Exception:
    _EXC_LOG_PATH = None


_initialized = False
_RUNTIME_NAMESPACE: Optional[Any] = None
_APP: Optional[FastAPI] = None
_APP_DEPRECATION = (
    "Importing apps.backend.interfaces.api.run_api:app without factory is deprecated. "
    "Prefer `uvicorn --factory apps.backend.interfaces.api.run_api:create_api_app`, "
    "but a module-level app is provided for compatibility."
)


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    try:
        from apps.backend.codex.initialization import (
            initialize_codex,
        )
    except Exception:  # pragma: no cover - optional compatibility layer
        def initialize_codex() -> None:
            return None

    # Native initialization only (no legacy bootstrap)
    initialize_codex()

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


tasks: Dict[str, 'TaskEntry'] = {}
tasks_lock = threading.Lock()


class TaskEntry:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.done: asyncio.Future[bool] = loop.create_future()
        self.cleanup_handle: Optional[asyncio.TimerHandle] = None
        self.cancel_requested: bool = False
        self.cancel_mode: str = "immediate"  # immediate | after_current

    def schedule_cleanup(self, task_id: str, delay: float = 300.0) -> None:
        if self.cleanup_handle:
            self.cleanup_handle.cancel()
        self.cleanup_handle = self.loop.call_later(delay, lambda: tasks.pop(task_id, None))


def get_task(task_id: str) -> Optional['TaskEntry']:
    with tasks_lock:
        return tasks.get(task_id)


def register_task(task_id: str, entry: 'TaskEntry') -> None:
    with tasks_lock:
        tasks[task_id] = entry


def request_task_cancel(task_id: str, *, mode: str = "immediate") -> bool:
    with tasks_lock:
        entry = tasks.get(task_id)
        if entry is None:
            return False
        entry.cancel_requested = True
        entry.cancel_mode = mode if mode in {"immediate", "after_current"} else "immediate"
        return True


def _load_json(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def port_free(port: int, host: str = '0.0.0.0') -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


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


def _serialize_checkpoint(info) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "title": info.title,
        "name": info.name,
        "model_name": info.model_name,
        "hash": info.shorthash,
        "filename": info.filename,
        "metadata": info.metadata,
    }


def _serialize_sampler(sampler) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    options = {}
    if isinstance(sampler.options, dict):
        for key, value in sampler.options.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                options[key] = value
    return {
        "name": sampler.name,
        "aliases": list(sampler.aliases or []),
        "options": options,
    }


def _serialize_scheduler(scheduler) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "name": scheduler.name,
        "label": scheduler.label,
        "aliases": list(scheduler.aliases or []),
    }


class _DummyRequest:
    def __init__(self, username: str = "api") -> None:
        self.username = username


def build_app() -> FastAPI:
    ensure_initialized()

    # Native parameter helpers (replace legacy _txt2img/_img2img parsers)
    from apps.backend.services import param_utils as _p
    _script_callbacks = None  # legacy callbacks not used in native backend
    _shared = None
    _sd_models = None
    from apps.backend.core.engine_interface import TaskType
    from apps.backend.core.orchestrator import InferenceOrchestrator
    from apps.backend.core.requests import (
        ProgressEvent,
        ResultEvent,
        Txt2ImgRequest,
        Img2ImgRequest,
        Txt2VidRequest,
        Img2VidRequest,
    )
    from apps.backend.services.media_service import MediaService

    app = FastAPI(title='SD WebUI API', version='0.2.0')
    # Ensure engines are registered (sd15/sdxl/flux + video engines incl. WAN)
    try:
        from apps.backend.engines import register_default_engines as _register_default_engines  # type: ignore
        _register_default_engines(replace=False)
    except Exception as _e:  # pragma: no cover
        print(color_red(f"[engines] registration failed: {_e}"))

    # Defensive: ensure WAN22 engines are present even if registration import failed
    try:
        from apps.backend.core.registry import registry as _engine_registry
        need = {"wan22_14b", "wan22_5b"} - set(_engine_registry.list())
        if need:
            from apps.backend.engines.wan22.wan22_14b import Wan2214BEngine  # type: ignore
            from apps.backend.engines.wan22.wan22_5b import Wan225BEngine  # type: ignore
            for key, cls in (("wan22_14b", Wan2214BEngine), ("wan22_5b", Wan225BEngine)):
                if key in need:
                    _engine_registry.register(key, cls, aliases=(key.replace('_', '-'),), replace=False)  # type: ignore
            print(color_cyan("[engines] ensured WAN22 engines are registered"))
    except Exception as _e:  # pragma: no cover
        print(color_red(f"[engines] failed to ensure WAN22 engines: {_e}"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    media = MediaService()
    # Native options facade (JSON‑backed). Import early so helpers are available
    # to any route or startup function defined below.
    from apps.backend.codex.options import (
        get_value as _opts_get,
        set_values as _opts_set_many,
        get_snapshot as _opts_snapshot,
        _load as _opts_load_native,  # private, but safe here
    )
    def _opts_load() -> Dict[str, Any]:  # thin alias used by existing helpers
        return _opts_load_native()
    _embedding_db = None  # lazy init
    _settings_schema_cache: Optional[Dict[str, Any]] = None
    _settings_values_path = os.path.join(os.getcwd(), 'apps', 'settings_values.json')
    _ui_blocks_cache: Optional[Dict[str, Any]] = None
    _ui_blocks_mtime: Optional[float] = None
    _ui_presets_cache: Optional[Dict[str, Any]] = None
    _ui_presets_mtime: Optional[float] = None
    _tabs_cache: Optional[Dict[str, Any]] = None
    _tabs_mtime: Optional[float] = None
    _workflows_cache: Optional[Dict[str, Any]] = None
    _workflows_mtime: Optional[float] = None
    _ui_dist_dir = os.path.join(PROJECT_ROOT, 'apps', 'interface', 'dist')

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

    def _detect_wan_variant_from_dir(p: Optional[str]) -> Optional[str]:
        """Return '5b' or '14b' when a WAN dir hints a variant via config or path name."""
        if not p:
            return None
        try:
            pd = p
            if os.path.isfile(pd):
                pd = os.path.dirname(pd)
            cfg = _load_json(os.path.join(pd, 'config.json'))
            txts: list[str] = []
            if isinstance(cfg, dict):
                for k, v in cfg.items():
                    if isinstance(v, str):
                        txts.append(v.lower())
            blob = ' '.join(txts + [str(p).lower()])
            if '5b' in blob:
                return '5b'
            if '14b' in blob or 'a14b' in blob or '14-b' in blob:
                return '14b'
        except Exception:
            pass
        s = str(p).lower()
        if '5b' in s:
            return '5b'
        if '14b' in s or 'a14b' in s or '14-b' in s:
            return '14b'
        return None

    def _pick_wan_engine(extras: Dict[str, Any]) -> str:
        """Choose wan22_14b or wan22_5b from provided WAN extras; default to 14b."""
        wh = extras.get('wan_high') or {}
        wl = extras.get('wan_low') or {}
        cand = None
        try:
            if isinstance(wh, dict) and wh.get('model_dir'):
                cand = _detect_wan_variant_from_dir(str(wh.get('model_dir')))
            if cand is None and isinstance(wl, dict) and wl.get('model_dir'):
                cand = _detect_wan_variant_from_dir(str(wl.get('model_dir')))
        except Exception:
            cand = None
        if cand == '5b':
            return 'wan22_5b'
        return 'wan22_14b'

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

    # (call to _apply_saved_settings moved below after its definition)

    @app.get('/api/health')
    def health() -> Dict[str, bool]:
        return {'ok': True}

    @app.get('/api/version')
    def version_info() -> Dict[str, Any]:
        """Return backend version details for footer display.
        Includes app version, git commit (short), Python, Torch and CUDA.
        """
        app_version = getattr(app, 'version', '0')
        # Git commit
        git_commit: Optional[str] = None
        try:
            import subprocess
            git_commit = (
                subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
                .decode('utf-8')
                .strip()
            )
        except Exception:
            git_commit = os.environ.get('GIT_COMMIT') or os.environ.get('VITE_GIT_COMMIT') or None

        # Python
        import sys
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Torch/CUDA (optional)
        torch_ver: Optional[str] = None
        cuda_ver: Optional[str] = None
        try:
            import torch  # type: ignore
            torch_ver = getattr(torch, '__version__', None)
            cuda_ver = getattr(getattr(torch, 'version', None), 'cuda', None)
            # Some builds set torch.version.cuda None even with runtime available; keep None if not set
        except Exception:
            pass

        return {
            'app_version': app_version,
            'git_commit': git_commit,
            'python_version': py_ver,
            'torch_version': torch_ver,
            'cuda_version': cuda_ver,
        }

    # Serve built UI (Vite build) if present, with SPA fallback
    class SPAStaticFiles(StaticFiles):  # type: ignore[misc]
        async def get_response(self, path: str, scope):  # type: ignore[override]
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:  # type: ignore[assignment]
                if exc.status_code == 404:
                    # Fallback to index.html for SPA routes
                    return await super().get_response('index.html', scope)
                raise

    @app.get('/api/memory')
    def memory() -> Dict[str, Any]:
        try:
            from apps.backend import memory_management as _mm  # type: ignore
            total = int(getattr(_mm, 'total_vram', 0))
        except Exception:
            total = 0
        return {"total_vram_mb": total}

    @app.get('/api/settings/schema')
    def settings_schema() -> Dict[str, Any]:
        nonlocal _settings_schema_cache
        if _settings_schema_cache is None:
            if _settings_registry_ok:
                try:
                    _settings_schema_cache = _schema_hardcoded()
                except Exception as e:  # pragma: no cover
                    print(color_red(f"[settings] hardcoded schema failed: {e}"))
                    _settings_schema_cache = None
            if _settings_schema_cache is None:
                schema_path = os.path.join(os.getcwd(), 'apps', 'backend', 'interfaces', 'schemas', 'settings_schema.json')
                if not os.path.isfile(schema_path):
                    schema_path = os.path.join(os.getcwd(), 'apps', 'server', 'settings_schema.json')
                _settings_schema_cache = _load_json(schema_path)
                if not _settings_schema_cache:
                    raise HTTPException(status_code=500, detail='settings schema not found (registry and JSON)')

        # Hydrate dynamic choices on-the-fly
        try:
            from apps.backend.interfaces.schemas.settings_hydration import hydrate_schema  # type: ignore
            return hydrate_schema(_settings_schema_cache)
        except Exception:
            return _settings_schema_cache

    # ------------------------------------------------------------------
    # UI Blocks (server-driven parameter panels)
    def _load_ui_blocks() -> Dict[str, Any]:
        nonlocal _ui_blocks_cache, _ui_blocks_mtime
        blocks_path = os.path.join(os.getcwd(), 'apps', 'interface', 'blocks.json')
        # Simple mtime-based cache
        try:
            stat = os.stat(blocks_path)
            mtime = stat.st_mtime
        except Exception:
            raise HTTPException(status_code=500, detail='ui blocks not found')
        if _ui_blocks_cache is not None and _ui_blocks_mtime == mtime:
            return _ui_blocks_cache
        data = _load_json(blocks_path)
        if not data or 'blocks' not in data:
            raise HTTPException(status_code=500, detail='invalid ui blocks json')
        # Optional overrides in apps/interface/blocks.d/*.json (merged by id)
        overrides_root = os.path.join(os.getcwd(), 'apps', 'interface', 'blocks.d')
        merged = {b.get('id'): b for b in (data.get('blocks') or []) if isinstance(b, dict)}
        try:
            if os.path.isdir(overrides_root):
                for fn in os.listdir(overrides_root):
                    if not fn.endswith('.json'):
                        continue
                    ov = _load_json(os.path.join(overrides_root, fn))
                    if isinstance(ov, dict) and 'blocks' in ov and isinstance(ov['blocks'], list):
                        for blk in ov['blocks']:
                            if isinstance(blk, dict) and blk.get('id'):
                                merged[blk['id']] = blk
        except Exception:
            pass
        out = {"version": int(data.get('version', 1)), "blocks": list(merged.values())}
        _ui_blocks_cache, _ui_blocks_mtime = out, mtime
        return out

    def _detect_semantic_engine() -> str:
        """Infer a semantic engine tag from the currently selected checkpoint.

        Returns one of: 'wan22', 'hunyuan_video', 'svd', 'flux', 'sdxl', 'sd15'.
        This is a conservative, metadata-first detector with filename fallback.
        """
        # Heuristics based on current selected checkpoint title/path and registry metadata
        try:
            from apps.backend.codex import main as _codex
            current = getattr(_codex, "_SELECTIONS").checkpoint_name
            infos = model_api.list_checkpoints_as_dict(refresh=False)
            target = None
            for i in infos:
                if i.get('name') == current or i.get('title') == current:
                    target = i
                    break
            blob = ''
            if target:
                blob = ' '.join([
                    str(target.get('title') or ''), str(target.get('path') or ''), str(target.get('format') or ''),
                ]).lower()
                comps = target.get('components') or []
                if any('flux' in blob for blob in [blob]) or 'transformer' in comps:
                    if 'flux' in blob:
                        return 'flux'
                if 'text_encoder_2' in comps and 'tokenizer_2' in comps:
                    return 'sdxl'
                if 'hunyuan' in blob:
                    return 'hunyuan_video'
                if 'svd' in blob:
                    return 'svd'
                if 'wan' in blob:
                    return 'wan22'
            # Fallback on title string hints
            t = (current or '').lower()
            if 'flux' in t:
                return 'flux'
            if 'xl' in t:
                return 'sdxl'
            if 'wan' in t:
                return 'wan22'
        except Exception:
            pass
        return 'sd15'

    # ------------------------------------------------------------------
    # Tabs & Workflows Persistence (JSON files)
    def _tabs_path() -> str:
        return os.path.join(os.getcwd(), 'apps', 'interface', 'tabs.json')

    def _workflows_path() -> str:
        return os.path.join(os.getcwd(), 'apps', 'interface', 'workflows.json')

    def _ensure_dirs() -> None:
        root = os.path.join(os.getcwd(), 'apps', 'interface')
        os.makedirs(root, exist_ok=True)

    def _default_tabs() -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        def mk(t: str, title: str, order: int) -> Dict[str, Any]:
            return {
                'id': f'tab-{t}-{order}', 'type': t, 'title': title,
                'order': order, 'enabled': True,
                'params': {}, 'meta': {'createdAt': now, 'updatedAt': now}
            }
        return {'version': 1, 'tabs': [
            mk('sd15', 'SD 1.5', 0), mk('sdxl', 'SDXL', 1), mk('flux', 'FLUX', 2), mk('wan', 'WAN 2.2', 3)
        ]}

    def _load_tabs() -> Dict[str, Any]:
        nonlocal _tabs_cache, _tabs_mtime
        _ensure_dirs()
        p = _tabs_path()
        if not os.path.exists(p):
            data = _default_tabs()
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        stat = os.stat(p)
        if _tabs_cache is not None and _tabs_mtime == stat.st_mtime:
            return _tabs_cache
        data = _load_json(p)
        if not isinstance(data, dict) or 'tabs' not in data:
            data = _default_tabs()
        _tabs_cache, _tabs_mtime = data, stat.st_mtime
        return data

    def _save_tabs(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _tabs_path()
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        stat = os.stat(p)
        nonlocal _tabs_cache, _tabs_mtime
        _tabs_cache, _tabs_mtime = data, stat.st_mtime

    def _load_workflows() -> Dict[str, Any]:
        nonlocal _workflows_cache, _workflows_mtime
        _ensure_dirs()
        p = _workflows_path()
        if not os.path.exists(p):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'version': 1, 'workflows': []}, f, indent=2)
        stat = os.stat(p)
        if _workflows_cache is not None and _workflows_mtime == stat.st_mtime:
            return _workflows_cache
        data = _load_json(p)
        if not isinstance(data, dict) or 'workflows' not in data:
            data = {'version': 1, 'workflows': []}
        _workflows_cache, _workflows_mtime = data, stat.st_mtime
        return data

    def _save_workflows(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _workflows_path()
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        stat = os.stat(p)
        nonlocal _workflows_cache, _workflows_mtime
        _workflows_cache, _workflows_mtime = data, stat.st_mtime

    @app.get('/api/ui/tabs')
    def api_get_tabs() -> Dict[str, Any]:
        return _load_tabs()

    @app.post('/api/ui/tabs')
    def api_create_tab(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_tabs()
        tabs = list(data.get('tabs') or [])
        ttype = str(payload.get('type') or 'sd15')
        title = str(payload.get('title') or ttype.upper())
        params = payload.get('params') or {}
        new_id = f"tab-{int(time.time()*1000)}"
        order = max([int(t.get('order', 0)) for t in tabs], default=-1) + 1
        now = datetime.utcnow().isoformat()
        tab = {'id': new_id, 'type': ttype, 'title': title, 'order': order, 'enabled': True, 'params': params, 'meta': {'createdAt': now, 'updatedAt': now}}
        tabs.append(tab)
        out = {'version': int(data.get('version', 1)), 'tabs': tabs}
        _save_tabs(out)
        return {'id': new_id}

    @app.patch('/api/ui/tabs/{tab_id}')
    def api_update_tab(tab_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_tabs()
        updated = False
        now = datetime.utcnow().isoformat()
        for t in data['tabs']:
            if str(t.get('id')) == tab_id:
                if 'title' in payload:
                    t['title'] = str(payload['title'])
                if 'enabled' in payload:
                    t['enabled'] = bool(payload['enabled'])
                if 'params' in payload and isinstance(payload['params'], dict):
                    # shallow merge for now
                    t['params'] = payload['params']
                t['meta'] = t.get('meta') or {}
                t['meta']['updatedAt'] = now
                updated = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail='tab not found')
        _save_tabs(data)
        return {'updated': tab_id}

    @app.post('/api/ui/tabs/reorder')
    def api_reorder_tabs(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ids = list(payload.get('ids') or [])
        data = _load_tabs()
        idx = {tid: i for i, tid in enumerate(ids)}
        for t in data['tabs']:
            tid = str(t.get('id'))
            if tid in idx:
                t['order'] = idx[tid]
        data['tabs'].sort(key=lambda x: int(x.get('order', 0)))
        _save_tabs(data)
        return {'ok': True}

    @app.delete('/api/ui/tabs/{tab_id}')
    def api_delete_tab(tab_id: str) -> Dict[str, Any]:
        data = _load_tabs()
        tabs = [t for t in data['tabs'] if str(t.get('id')) != tab_id]
        if len(tabs) == len(data['tabs']):
            raise HTTPException(status_code=404, detail='tab not found')
        # normalize order
        for i, t in enumerate(tabs):
            t['order'] = i
        out = {'version': int(data.get('version', 1)), 'tabs': tabs}
        _save_tabs(out)
        return {'deleted': tab_id}

    @app.get('/api/ui/workflows')
    def api_get_workflows() -> Dict[str, Any]:
        return _load_workflows()

    @app.post('/api/ui/workflows')
    def api_create_workflow(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_workflows()
        wfs = list(data.get('workflows') or [])
        wf_id = f"wf-{int(time.time()*1000)}"
        name = str(payload.get('name') or wf_id)
        source_tab_id = str(payload.get('source_tab_id') or '')
        wtype = str(payload.get('type') or 'sd15')
        params_snapshot = payload.get('params_snapshot') or {}
        now = datetime.utcnow().isoformat()
        wf = {'id': wf_id, 'name': name, 'source_tab_id': source_tab_id, 'type': wtype, 'created_at': now, 'engine_semantics': payload.get('engine_semantics') or wtype, 'params_snapshot': params_snapshot}
        wfs.insert(0, wf)
        out = {'version': int(data.get('version', 1)), 'workflows': wfs}
        _save_workflows(out)
        return {'id': wf_id}

    @app.patch('/api/ui/workflows/{wf_id}')
    def api_update_workflow(wf_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_workflows()
        updated = False
        for w in data['workflows']:
            if str(w.get('id')) == wf_id:
                if 'name' in payload:
                    w['name'] = str(payload['name'])
                if 'params_snapshot' in payload and isinstance(payload['params_snapshot'], dict):
                    w['params_snapshot'] = payload['params_snapshot']
                updated = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail='workflow not found')
        _save_workflows(data)
        return {'updated': wf_id}

    @app.delete('/api/ui/workflows/{wf_id}')
    def api_delete_workflow(wf_id: str) -> Dict[str, Any]:
        data = _load_workflows()
        wfs = [w for w in data['workflows'] if str(w.get('id')) != wf_id]
        if len(wfs) == len(data['workflows']):
            raise HTTPException(status_code=404, detail='workflow not found')
        out = {'version': int(data.get('version', 1)), 'workflows': wfs}
        _save_workflows(out)
        return {'deleted': wf_id}

    # Models Load/Unload stubs (per tab)
    @app.post('/api/models/load')
    def api_models_load(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get('tab_id') or '')
        if not tab_id:
            raise HTTPException(status_code=400, detail='tab_id required')
        # TODO: implement actual preloading based on tab params
        print(color_green(f"[models] load requested for tab {tab_id}"))
        return {'ok': True}

    @app.post('/api/models/unload')
    def api_models_unload(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get('tab_id') or '')
        if not tab_id:
            raise HTTPException(status_code=400, detail='tab_id required')
        print(color_yellow(f"[models] unload requested for tab {tab_id}"))
        return {'ok': True}

    @app.get('/engines/capabilities')
    def list_engine_capabilities() -> Dict[str, Any]:
        """Expose semantic engine parameter surfaces for frontend gating.

        Shape: { engines: { <semantic_engine>: { supports_txt2img, supports_img2img, ... } } }
        """
        try:
            from apps.backend.runtime.model_registry.capabilities import serialize_engine_capabilities
            return {"engines": serialize_engine_capabilities()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read engine capabilities: {exc}")

    @app.get('/api/ui/blocks')
    def ui_blocks(tab: Optional[str] = None) -> Dict[str, Any]:
        """Return UI blocks filtered by tab and current semantic engine.

        - Source of truth: apps/interface/blocks.json (+ overrides in apps/interface/blocks.d).
        - Filters by `tab` (when provided) and by current detected engine if block declares `when.engines`.
        - Includes `semantic_engine` in the response for UI gating.
        """
        data = _load_ui_blocks()
        sem = _detect_semantic_engine()
        blocks_in = list(data.get('blocks') or [])
        out: list[dict] = []
        tab_norm = str(tab).strip().lower() if tab else None
        for b in blocks_in:
            if not isinstance(b, dict):
                continue
            when = b.get('when') or {}
            ok_tab = True
            ok_eng = True
            if tab_norm and isinstance(when, dict) and when.get('tabs'):
                ok_tab = tab_norm in [str(t).lower() for t in when.get('tabs')]
            if isinstance(when, dict) and when.get('engines'):
                ok_eng = sem in [str(e).lower() for e in when.get('engines')]
            if ok_tab and ok_eng:
                out.append(b)
        return {"version": data.get("version", 1), "blocks": out, "semantic_engine": sem}

    # ------------------------------------------------------------------
    # UI Presets (Model UI)
    def _load_ui_presets() -> Dict[str, Any]:
        nonlocal _ui_presets_cache, _ui_presets_mtime
        presets_path = os.path.join(os.getcwd(), 'apps', 'ui', 'presets.json')
        try:
            stat = os.stat(presets_path)
            mtime = stat.st_mtime
        except Exception:
            logging.getLogger("backend.api").warning("ui presets not found at %s; returning empty list", presets_path)
            return {"version": 1, "presets": []}
        if _ui_presets_cache is not None and _ui_presets_mtime == mtime:
            return _ui_presets_cache
        data = _load_json(presets_path)
        if not data or 'presets' not in data:
            logging.getLogger("backend.api").warning("ui presets invalid at %s; returning empty list", presets_path)
            return {"version": int(data.get('version', 1)) if isinstance(data, dict) else 1, "presets": []}  # type: ignore[arg-type]
        out = {"version": int(data.get('version', 1)), "presets": list(data.get('presets') or [])}
        _ui_presets_cache, _ui_presets_mtime = out, mtime
        return out

    @app.get('/api/ui/presets')
    def ui_presets(tab: Optional[str] = None) -> Dict[str, Any]:
        """Return Model UI presets, optionally filtered by tab."""
        data = _load_ui_presets()
        if not tab:
            return data
        tab_norm = str(tab).strip().lower()
        presets = [
            p for p in (data.get('presets') or [])
            if not isinstance(p, dict)
            or not p.get('tabs')
            or tab_norm in [str(t).lower() for t in (p.get('tabs') or [])]
        ]
        return {"version": data.get("version", 1), "presets": presets}

    @app.post('/api/ui/presets/apply')
    def ui_presets_apply(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Apply a Model UI preset: resolve checkpoint and set options atomically.

        Payload: { id: string, tab: 'txt2img'|'img2img'|'txt2vid'|'img2vid' }
        """
        try:
            preset_id = str(payload.get('id'))
            tab = str(payload.get('tab')) if payload.get('tab') else None
        except Exception:
            raise HTTPException(status_code=400, detail='invalid payload')
        if not preset_id:
            raise HTTPException(status_code=400, detail='id is required')
        data = _load_ui_presets()
        candidates = [p for p in (data.get('presets') or []) if isinstance(p, dict) and p.get('id') == preset_id]
        if tab:
            tab_norm = str(tab).strip().lower()
            candidates = [p for p in candidates if not p.get('tabs') or tab_norm in [str(t).lower() for t in p.get('tabs')]]
        if not candidates:
            raise HTTPException(status_code=404, detail=f'preset not found: {preset_id}')
        preset = candidates[0]

        # Resolve checkpoint by selector
        selector = preset.get('model_select') or {}
        sel_type = str(selector.get('type', 'exact')).lower()
        sel_value = str(selector.get('value', ''))
        if not sel_value:
            raise HTTPException(status_code=409, detail='preset has no model selector')

        infos = model_api.list_checkpoints_as_dict(refresh=False)
        titles = [str(i.get('title') or i.get('name') or '') for i in infos]
        target: Optional[str] = None
        if sel_type == 'exact':
            for t in titles:
                if t == sel_value:
                    target = t
                    break
        else:  # pattern: case-insensitive containment in title
            sval = sel_value.lower()
            for t in titles:
                if sval in str(t).lower():
                    target = t
                    break
        if not target:
            raise HTTPException(status_code=409, detail=f'checkpoint not found for selector: {sel_type}:{sel_value}')

        # Apply options atomically
        from apps.backend.codex import main as _codex
        try:
            _codex.checkpoint_change(str(target), save=True, refresh=False)
            updates = {'sd_model_checkpoint': str(target)}
            extra = preset.get('options') or {}
            if isinstance(extra, dict):
                updates.update(extra)
            _opts_set_many(updates)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'failed to apply preset: {e}')

        return {"applied": True, "model": target}

    @app.get('/api/settings/values')
    def settings_values() -> Dict[str, Any]:
        try:
            vals = _opts_load()
            idx = _field_index() if _settings_registry_ok else {}
            if idx:
                vals = {k: vals.get(k) for k in idx.keys()}
            return {"values": vals}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read values: {e}")

    # Load saved settings on startup and apply to shared.opts with validation
    def _apply_saved_settings() -> None:
        if not _settings_registry_ok:
            return
        saved = _opts_load()
        if not isinstance(saved, dict) or not saved:
            return
        idx = _field_index()
        # Validate and normalize persisted values against schema, then re-save
        changed = False
        for k in list(saved.keys()):
            f = idx.get(k)
            if not f:
                saved.pop(k, None); changed = True; continue
            try:
                if getattr(f, 'choices', None) and isinstance(f.choices, list) and saved[k] not in f.choices:
                    saved.pop(k, None); changed = True; continue
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    v = float(saved[k])
                    lo = getattr(f, 'min', None); hi = getattr(f, 'max', None)
                    if isinstance(lo, (int, float)) and v < lo:
                        saved[k] = lo; changed = True
                    if isinstance(hi, (int, float)) and v > hi:
                        saved[k] = hi; changed = True
                if getattr(f, 'type', None) == _SettingType.CHECKBOX and isinstance(saved[k], str):
                    saved[k] = saved[k].lower() in ('1','true','yes','on'); changed = True
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

    @app.get('/api/models')
    def list_models() -> Dict[str, Any]:
        """List checkpoints discovered by the native registry.

        Response shape remains compatible: fields include title/name/filename/metadata.
        """
        from apps.backend.runtime.models import api as _model_api

        def _serialize(entry) -> Dict[str, Any]:
            return {
                "title": entry.title,
                "name": entry.name,
                "model_name": entry.model_name,
                "hash": entry.short_hash,
                "filename": entry.filename,
                "metadata": entry.metadata,
            }

        entries = _model_api.list_checkpoints()
        models = [_serialize(e) for e in entries]
        models_info = _model_api.list_checkpoints_as_dict()
        # Current selection: track via codex options when available
        try:
            from apps.backend.codex import main as _codex
            current = getattr(_codex, "_SELECTIONS").checkpoint_name or (models[0]["name"] if models else None)
        except Exception:
            current = models[0]["name"] if models else None
        return {"models": models, "current": current, "models_info": models_info}

    @app.get('/api/models/inventory')
    def list_models_inventory(refresh: bool = Query(False, description="If true, re-scan the models/ and huggingface/ folders.")) -> Dict[str, Any]:
        """Inventory of model-related assets discovered at startup.

        - vaes: files under /models/VAE
        - text_encoders: files under /models/text-encoder plus per-engine roots from apps/paths.json (sd15_tenc, sdxl_tenc, flux_tenc, wan22_tenc)
        - loras: files under /models/Lora
        - wan22.gguf: files under /models/wan22 (or explicit roots from apps/paths.json/wan22_ckpt)
        - metadata: org/repo roots under backend/huggingface
        """
        from apps.backend.inventory import cache as _inv_cache
        if refresh:
            try:
                inv = _inv_cache.refresh()
                logging.getLogger("inventory").info(
                    "inventory: refreshed (vaes=%d, text_encoders=%d, loras=%d, wan22.gguf=%d, metadata=%d)",
                    len(inv.get("vaes", [])), len(inv.get("text_encoders", [])), len(inv.get("loras", [])), len(inv.get("wan22", [])), len(inv.get("metadata", []))
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"inventory refresh failed: {e}")
        else:
            inv = _inv_cache.get()
        return {
            "vaes": inv.get("vaes", []),
            "text_encoders": inv.get("text_encoders", []),
            "loras": inv.get("loras", []),
            "wan22": {"gguf": inv.get("wan22", [])},
            "metadata": inv.get("metadata", []),
        }

    @app.post('/api/models/inventory/refresh')
    def refresh_models_inventory() -> Dict[str, Any]:
        """Force re-scan of model folders and return the updated inventory.

        This is useful after copying new files into models/ without restarting the backend.
        """
        from apps.backend.inventory import cache as _inv_cache
        try:
            inv = _inv_cache.refresh()
            logging.getLogger("inventory").info(
                "inventory: refreshed (vaes=%d, text_encoders=%d, loras=%d, wan22.gguf=%d, metadata=%d)",
                len(inv.get("vaes", [])), len(inv.get("text_encoders", [])), len(inv.get("loras", [])), len(inv.get("wan22", [])), len(inv.get("metadata", []))
            )
            return {
                "vaes": inv.get("vaes", []),
                "text_encoders": inv.get("text_encoders", []),
                "loras": inv.get("loras", []),
                "wan22": {"gguf": inv.get("wan22", [])},
                "metadata": inv.get("metadata", []),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inventory refresh failed: {e}")

    @app.get('/api/samplers')
    def list_samplers() -> Dict[str, Any]:
        from apps.backend.runtime.sampling.registry import get_sampler_spec

        samplers = []
        for entry in SAMPLER_OPTIONS:
            if not entry.get("supported", True):
                continue
            spec = None
            try:
                spec = get_sampler_spec(entry["name"])
            except Exception:
                pass
            samplers.append(
                {
                    "name": entry["name"],
                    "label": entry.get("label", str(entry["name"]).title()),
                    "aliases": [alias.strip() for alias in entry.get("aliases", []) if isinstance(alias, str) and alias.strip()],
                    "supported": bool(entry.get("supported", True)),
                    "default_scheduler": spec.default_scheduler if spec else None,
                    "allowed_schedulers": sorted(spec.allowed_schedulers) if spec else [],
                }
            )
        return {"samplers": samplers}

    @app.get('/api/schedulers')
    def list_schedulers() -> Dict[str, Any]:
        schedulers = []
        for entry in SCHEDULER_OPTIONS:
            if not entry.get("supported", True):
                continue
            schedulers.append(
                {
                    "name": entry["name"],
                    "label": entry.get("label", entry["name"].title()),
                    "aliases": [alias.strip() for alias in entry.get("aliases", []) if isinstance(alias, str) and alias.strip()],
                    "supported": bool(entry.get("supported", True)),
                }
            )
        return {"schedulers": schedulers}

    @app.get('/api/vaes')
    def list_vaes() -> Dict[str, Any]:
        """Return available VAE modules and current selection (native registry)."""
        from apps.backend.codex import options as _codex_opts
        from apps.backend.infra.registry.vae import list_vaes as _list_vaes, describe_vaes as _describe_vaes
        current = _codex_opts.get_selected_vae('Automatic')
        try:
            choices = _list_vaes()
            info = _describe_vaes()
            return {"vaes": choices, "current": current, "vaes_info": info}
        except Exception:
            return {"vaes": ['Automatic', 'Built in', 'None'], "current": current, "vaes_info": []}

    @app.get('/api/text-encoders')
    def list_text_encoders() -> Dict[str, Any]:
        """Return available text encoder overrides and current selection list.

        - `text_encoders`: flat list of labels suitable for QuickSettings (e.g., 'flux/my-te.safetensors').
        - `current`: persisted selection from Codex options (labels as stored).
        - `text_encoders_info`: structured metadata for each override entry.

        Vendored Hugging Face text encoder metadata remains available via
        `apps.backend.infra.registry.text_encoders.describe_text_encoders`, but is not
        surfaced directly here for overrides.
        """
        from apps.backend.infra.registry.text_encoder_roots import list_text_encoder_roots_by_family
        from apps.backend.codex import options as _codex_opts
        try:
            roots_by_family = list_text_encoder_roots_by_family()
            entries: list[dict[str, object]] = []
            labels: list[str] = []
            for family, roots in roots_by_family.items():
                for root in roots:
                    # For now, expose roots themselves as selectable labels; loaders
                    # will map labels to concrete paths when overrides are implemented.
                    label = root.name
                    labels.append(label)
                    entries.append(
                        {
                            "name": label,
                            "path": root.path,
                            "family": family,
                            "kind": root.kind,
                            "tags": list(root.tags),
                            "meta": dict(root.meta),
                        }
                    )
            labels_sorted = sorted(set(labels))
            selected = _codex_opts.get_additional_modules()
            return {"text_encoders": labels_sorted, "current": selected, "text_encoders_info": entries}
        except Exception:
            return {"text_encoders": {}, "current": [], "text_encoders_info": {}}

    @app.get('/api/embeddings')
    def list_embeddings() -> Dict[str, Any]:
        """Native Textual Inversion registry.

        Returns a non-breaking shape including `loaded` and `skipped`, but all
        entries are considered loaded when parseable.
        """
        from apps.backend.infra.registry.embeddings import describe_embeddings as _describe
        info = [e.__dict__ for e in _describe()]
        loaded = {e["name"]: {"name": e["name"], "vectors": e.get("vectors"), "shape": e.get("dims"), "step": e.get("step") } for e in info if e.get("vectors")}
        skipped = {e["name"]: {"name": e["name"], "vectors": e.get("vectors"), "shape": e.get("dims"), "step": e.get("step") } for e in info if not e.get("vectors")}
        return {"loaded": loaded, "skipped": skipped, "embeddings_info": info}

    @app.get('/api/loras')
    def list_loras() -> Dict[str, Any]:
        """Return LoRA files via native registry plus metadata.

        Shape: { loras: [{name, path}], loras_info: [LoraEntry-like dicts] }
        """
        from apps.backend.infra.registry.lora import list_loras as _list_loras, describe_loras as _describe_loras
        items = _list_loras()
        info = [e.__dict__ for e in _describe_loras()]
        return {"loras": items, "loras_info": info}

    @app.get('/api/loras/selections')
    def get_lora_selections() -> Dict[str, Any]:
        from apps.backend.codex.lora import get_selections
        sels = get_selections()
        return {"selections": [s.__dict__ for s in sels]}

    @app.post('/api/loras/apply')
    def apply_lora_selections(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Set LoRA selections process-wide (used at generation time).

        Payload: { selections: [{ path, weight?, online? }] }
        """
        if not isinstance(payload, dict) or 'selections' not in payload or not isinstance(payload['selections'], list):
            raise HTTPException(status_code=400, detail='payload must be {"selections": [...]}')
        from apps.backend.codex.lora import set_selections, LoraSelection
        raw = payload['selections']
        # Normalize to dataclasses
        sels = []
        for it in raw:
            if not isinstance(it, dict) or 'path' not in it:
                continue
            sels.append(LoraSelection(path=str(it['path']), weight=float(it.get('weight', 1.0)), online=bool(it.get('online', False))))
        set_selections(sels)
        return {"ok": True, "count": len(sels)}

    # Simple paths config for frontend-managed search locations
    @app.get('/api/paths')
    def get_paths() -> Dict[str, Any]:
        """Expose paths.json for the UI.

        Returned structure (for Settings → Paths):
        {
          "paths": {
            "sd15_ckpt": [...], "sd15_vae": [...], "sd15_loras": [...], "sd15_tenc": [...],
            "sdxl_ckpt": [...], "sdxl_vae": [...], "sdxl_loras": [...], "sdxl_tenc": [...],
            "flux_ckpt": [...], "flux_vae": [...], "flux_loras": [...], "flux_tenc": [...],
            "wan22_ckpt": [...], "wan22_vae": [...], "wan22_loras": [...], "wan22_tenc": [...],
            ...
          }
        }

        For backward compatibility, aggregated buckets (`checkpoints`, `vae`, `lora`, `text_encoders`)
        are also exposed, derived from the engine-specific keys.
        """
        cfg_path = os.path.join(os.getcwd(), 'apps', 'paths.json')
        raw = _load_json(cfg_path) or {}
        if not isinstance(raw, dict):
            raw = {}

        def _list(key: str) -> list[str]:
            v = raw.get(key) or []
            return v if isinstance(v, list) else []

        # Start from the engine-specific mapping and then add aggregated helpers.
        paths: Dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                # Coerce to list[str]
                paths[key] = [str(item) for item in value if isinstance(item, str)]

        paths["checkpoints"] = _list("sd15_ckpt") + _list("sdxl_ckpt") + _list("flux_ckpt") + _list("wan22_ckpt")
        paths["vae"] = _list("sd15_vae") + _list("sdxl_vae") + _list("flux_vae") + _list("wan22_vae")
        paths["lora"] = _list("sd15_loras") + _list("sdxl_loras") + _list("flux_loras") + _list("wan22_loras")
        paths["text_encoders"] = _list("sd15_tenc") + _list("sdxl_tenc") + _list("flux_tenc") + _list("wan22_tenc")

        return {"paths": paths}

    @app.post('/api/paths')
    def set_paths(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Update paths.json from UI-managed buckets or engine-specific keys.

        Payload: { paths: { ... } }
        - Aggregated keys (`checkpoints`, `vae`, `lora`, `text_encoders`) are mapped to per-engine keys.
        - Engine-specific keys (e.g., `sd15_ckpt`, `sdxl_vae`) are applied directly.
        - Unknown keys are preserved from the existing file unless explicitly overwritten.
        """
        if not isinstance(payload, dict) or 'paths' not in payload or not isinstance(payload['paths'], dict):
            raise HTTPException(status_code=400, detail='payload must be {"paths": {...}}')

        cfg_path = os.path.join(os.getcwd(), 'apps', 'paths.json')
        current = _load_json(cfg_path) or {}
        if not isinstance(current, dict):
            current = {}

        incoming = payload["paths"] or {}
        new_paths: Dict[str, Any] = dict(current)

        # 1) Apply aggregated buckets when present (legacy behaviour).
        checkpoints = list(incoming.get("checkpoints") or [])
        vae = list(incoming.get("vae") or [])
        lora = list(incoming.get("lora") or [])
        text_encoders = list(incoming.get("text_encoders") or [])

        if checkpoints or vae or lora or text_encoders:
            new_paths["sd15_ckpt"] = checkpoints
            new_paths["sdxl_ckpt"] = checkpoints
            new_paths["flux_ckpt"] = checkpoints
            new_paths["wan22_ckpt"] = checkpoints

            new_paths["sd15_vae"] = vae
            new_paths["sdxl_vae"] = vae
            new_paths["flux_vae"] = vae
            new_paths["wan22_vae"] = vae

            new_paths["sd15_loras"] = lora
            new_paths["sdxl_loras"] = lora
            new_paths["flux_loras"] = lora
            new_paths["wan22_loras"] = lora

            new_paths["sd15_tenc"] = text_encoders
            new_paths["sdxl_tenc"] = text_encoders
            new_paths["flux_tenc"] = text_encoders
            new_paths["wan22_tenc"] = text_encoders

        # 2) Apply engine-specific keys (and any additional explicit keys) directly.
        for key, value in incoming.items():
            if key in {"checkpoints", "vae", "lora", "text_encoders"}:
                # Aggregated buckets are handled above and not stored verbatim in paths.json.
                continue
            if value is None:
                new_paths[key] = []
            elif isinstance(value, list):
                new_paths[key] = [str(item) for item in value if isinstance(item, str)]

        _save_json(cfg_path, new_paths)
        return {"ok": True}

    # Native options store via Codex options facade (aliases defined above)

    @app.get('/api/options')
    def get_options() -> Dict[str, Any]:
        return {"values": _opts_load()}

    @app.get('/api/options/keys')
    def get_options_keys() -> Dict[str, Any]:
        """List supported option keys and basic metadata from the settings registry.

        Fields: { keys: [...], types: {key: type_name}, choices: {key: [..]} }
        """
        if not _settings_registry_ok:
            return {"keys": [], "types": {}, "choices": {}}
        try:
            idx = _field_index()  # type: ignore[name-defined]
            keys = list(idx.keys())
            types = {}
            choices = {}
            for k, f in idx.items():
                t = getattr(getattr(f, 'type', None), 'name', None) or str(getattr(f, 'type', None))
                types[k] = t
                ch = getattr(f, 'choices', None)
                if isinstance(ch, list):
                    choices[k] = ch
            return {"keys": keys, "types": types, "choices": choices}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read registry: {e}")

    @app.get('/api/options/snapshot')
    def get_options_snapshot() -> Dict[str, Any]:
        """Return a typed snapshot of current options (for UI defaults)."""
        try:
            return {"snapshot": _opts_snapshot().as_dict()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read snapshot: {e}")

    @app.get('/api/options/defaults')
    def get_options_defaults() -> Dict[str, Any]:
        """Return default values from the settings registry and the current snapshot.

        Shape: { defaults: {key: value}, snapshot: {...} }
        """
        defaults: Dict[str, Any] = {}
        if _settings_registry_ok:
            try:
                idx = _field_index()  # type: ignore[name-defined]
                for k, f in idx.items():
                    defaults[k] = getattr(f, 'default', None)
            except Exception:
                defaults = {}
        try:
            from apps.backend.codex.options import get_snapshot as _snap  # type: ignore
            snap = _snap().as_dict()
        except Exception:
            snap = {}
        return {"defaults": defaults, "snapshot": snap}

    def _validate_options(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        # If a registry schema exists, validate/clamp/choice-filter values
        if not _settings_registry_ok:
            return dict(payload)
        try:
            idx = _field_index()  # type: ignore[name-defined]
        except Exception:
            return dict(payload)
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            f = idx.get(k)
            if not f:
                continue
            try:
                # choices
                if getattr(f, 'choices', None) and isinstance(f.choices, list):
                    if v not in f.choices:
                        continue
                # type normalization
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    num = float(v)
                    lo = getattr(f, 'min', None); hi = getattr(f, 'max', None)
                    if isinstance(lo, (int, float)) and num < lo:
                        num = lo
                    if isinstance(hi, (int, float)) and num > hi:
                        num = hi
                    out[k] = num
                elif getattr(f, 'type', None) == _SettingType.CHECKBOX:
                    if isinstance(v, str):
                        out[k] = v.strip().lower() in ('1','true','yes','on')
                    else:
                        out[k] = bool(v)
                else:
                    out[k] = v
            except Exception:
                continue
        return out

    @app.post('/api/options')
    def set_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail='invalid payload')
        # Validate against schema when available, then persist via Codex options
        updates = _validate_options(payload)
        updated = _opts_set_many(updates)

        # Apply memory manager overrides when present
        from apps.backend.runtime import memory_management as mem_management
        role_map = {
            "codex_core_device": ("core", "backend"),
            "codex_te_device": ("text_encoder", "backend"),
            "codex_vae_device": ("vae", "backend"),
            "codex_core_dtype": ("core", "dtype"),
            "codex_te_dtype": ("text_encoder", "dtype"),
            "codex_vae_dtype": ("vae", "dtype"),
        }
        for key, value in payload.items():
            if key not in role_map:
                continue
            role, kind = role_map[key]
            try:
                if kind == "backend":
                    mem_management.set_component_backend(role, str(value))
                else:
                    mem_management.set_component_dtype(role, str(value))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid memory setting for {key}: {exc}")

        return {"updated": updated}

    @app.post('/api/options/validate')
    def validate_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Dry-run options validation; returns accepted and rejected keys with reasons.

        Shape: { accepted: {k:v}, rejected: {k: reason} }
        """
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail='invalid payload')
        if not _settings_registry_ok:
            return {"accepted": dict(payload), "rejected": {}}
        try:
            idx = _field_index()  # type: ignore[name-defined]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"registry unavailable: {e}")
        accepted: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        for k, v in payload.items():
            f = idx.get(k)
            if not f:
                rejected[k] = 'unknown key'
                continue
            try:
                if getattr(f, 'choices', None) and isinstance(f.choices, list) and v not in f.choices:
                    rejected[k] = 'not in choices'
                    continue
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    num = float(v)
                    lo = getattr(f, 'min', None); hi = getattr(f, 'max', None)
                    if isinstance(lo, (int, float)) and num < lo:
                        rejected[k] = f'below min {lo}'
                        continue
                    if isinstance(hi, (int, float)) and num > hi:
                        rejected[k] = f'above max {hi}'
                        continue
                    accepted[k] = num
                elif getattr(f, 'type', None) == _SettingType.CHECKBOX:
                    if isinstance(v, str):
                        accepted[k] = v.strip().lower() in ('1','true','yes','on')
                    else:
                        accepted[k] = bool(v)
                else:
                    accepted[k] = v
            except Exception:
                rejected[k] = 'invalid value'
        return {"accepted": accepted, "rejected": rejected}

    _TXT2IMG_ALLOWED_KEYS = {
        "__strict_version",
        "codex_device",
        "device",
        "codex_diffusion_device",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "sampler",
        "scheduler",
        "seed",
        "batch_size",
        "batch_count",
        "styles",
        "metadata",
        "extras",
        "distilled_guidance_scale",
        "engine",
        "codex_engine",
        "model",
        "sd_model_checkpoint",
        "smart_offload",
        "smart_fallback",
        "smart_cache",
    }
    _TXT2IMG_EXTRAS_KEYS = {"highres", "randn_source", "eta_noise_seed_delta", "refiner", "text_encoder_override"}
    _TXT2IMG_HIGHRES_KEYS = {
        "enable",
        "denoise",
        "scale",
        "resize_x",
        "resize_y",
        "steps",
        "upscaler",
        "checkpoint",
        "modules",
        "sampler",
        "scheduler",
        "prompt",
        "negative_prompt",
        "cfg",
        "distilled_cfg",
        "refiner",
    }

    def _reject_unknown_keys(obj: Mapping[str, Any], allowed: set[str], context: str) -> None:
        unknown = sorted(set(obj.keys()) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unexpected {context} key(s): {', '.join(unknown)}")


    def _require_str_field(payload: Dict[str, Any], key: str, *, allow_empty: bool = False, trim: bool = True) -> str:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a string")
        result = value.strip() if trim else value
        if not allow_empty and result == "":
            raise HTTPException(status_code=400, detail=f"'{key}' must not be empty")
        return result if trim else value


    def _require_int_field(payload: Dict[str, Any], key: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
        if isinstance(value, float):
            if not value.is_integer():
                raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
            value = int(value)
        else:
            value = int(value)
        if minimum is not None and value < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return value


    def _require_float_field(payload: Dict[str, Any], key: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a number")
        result = float(value)
        if minimum is not None and result < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and result > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return result


    def _parse_styles(payload: Dict[str, Any]) -> List[str]:
        raw = payload.get('styles')
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
        out: List[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
            text = entry.strip()
            if text:
                out.append(text)
        return out

    def _parse_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = payload.get('metadata')
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'metadata' must be an object")
        return dict(raw)


    def _parse_txt2img_extras(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raw = payload.get('extras')
        if raw is None:
            return {}, None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'extras' must be an object")
        _reject_unknown_keys(raw, _TXT2IMG_EXTRAS_KEYS, "extras")
        extras: Dict[str, Any] = {}
        if 'randn_source' in raw:
            extras['randn_source'] = str(raw['randn_source'])
        if 'eta_noise_seed_delta' in raw:
            val = raw['eta_noise_seed_delta']
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise HTTPException(status_code=400, detail="'extras.eta_noise_seed_delta' must be numeric")
            extras['eta_noise_seed_delta'] = int(val)
        # Highres options
        highres = raw.get('highres')
        highres_cfg: Optional[Dict[str, Any]] = None
        if highres is not None:
            if not isinstance(highres, dict):
                raise HTTPException(status_code=400, detail="'extras.highres' must be an object")
            _reject_unknown_keys(highres, _TXT2IMG_HIGHRES_KEYS | {"enable"}, "extras.highres")
            if bool(highres.get('enable')):
                required = ['denoise', 'scale', 'resize_x', 'resize_y', 'steps', 'upscaler']
                for key in required:
                    if key not in highres:
                        raise HTTPException(status_code=400, detail=f"Missing 'extras.highres.{key}'")
                hr_modules = highres.get('modules')
                if hr_modules is not None:
                    if not isinstance(hr_modules, list) or any(not isinstance(entry, str) for entry in hr_modules):
                        raise HTTPException(status_code=400, detail="'extras.highres.modules' must be an array of strings")
                    modules_list = list(hr_modules)
                else:
                    modules_list = []
                refiner_raw = highres.get('refiner')
                refiner_cfg: Optional[Dict[str, Any]] = None
                if refiner_raw is not None:
                    if not isinstance(refiner_raw, dict):
                        raise HTTPException(status_code=400, detail="'extras.highres.refiner' must be an object")
                    _reject_unknown_keys(refiner_raw, {"enable", "steps", "cfg", "seed", "model", "vae"}, "extras.highres.refiner")
                    if bool(refiner_raw.get('enable')):
                        refiner_cfg = {
                            "steps": _require_int_field(refiner_raw, 'steps', minimum=0),
                            "cfg": _require_float_field(refiner_raw, 'cfg'),
                            "seed": _require_int_field(refiner_raw, 'seed'),
                        }
                        if 'model' in refiner_raw:
                            refiner_cfg['model'] = str(refiner_raw['model'])
                        if 'vae' in refiner_raw:
                            refiner_cfg['vae'] = str(refiner_raw['vae'])
                highres_cfg = {
                    "denoise": float(highres['denoise']),
                    "scale": float(highres['scale']),
                    "resize_x": _require_int_field(highres, 'resize_x'),
                    "resize_y": _require_int_field(highres, 'resize_y'),
                    "steps": _require_int_field(highres, 'steps', minimum=0),
                    "upscaler": _require_str_field(highres, 'upscaler', allow_empty=False, trim=True),
                    "checkpoint": highres.get('checkpoint'),
                    "modules": modules_list,
                    "sampler": highres.get('sampler'),
                    "scheduler": highres.get('scheduler'),
                    "prompt": highres.get('prompt') or '',
                    "negative_prompt": highres.get('negative_prompt') or '',
                    "cfg": float(highres.get('cfg')) if highres.get('cfg') is not None else None,
                    "distilled_cfg": float(highres.get('distilled_cfg')) if highres.get('distilled_cfg') is not None else None,
                    "refiner": refiner_cfg,
                }

        # Refiner options (metadata only for now)
        refiner = raw.get('refiner')
        if refiner is not None:
            if not isinstance(refiner, dict):
                raise HTTPException(status_code=400, detail="'extras.refiner' must be an object")
            _reject_unknown_keys(refiner, {"enable", "steps", "cfg", "seed", "model", "vae"}, "extras.refiner")
            if bool(refiner.get('enable')):
                ref_cfg: Dict[str, Any] = {
                    "steps": _require_int_field(refiner, 'steps', minimum=0),
                    "cfg": _require_float_field(refiner, 'cfg'),
                    "seed": _require_int_field(refiner, 'seed'),
                }
                if 'model' in refiner:
                    ref_cfg['model'] = str(refiner['model'])
                if 'vae' in refiner:
                    ref_cfg['vae'] = str(refiner['vae'])
                extras['refiner'] = ref_cfg

        # Text encoder override (family + label [+ optional components])
        te_override = raw.get('text_encoder_override')
        if te_override is not None:
            if not isinstance(te_override, dict):
                raise HTTPException(status_code=400, detail="'extras.text_encoder_override' must be an object")
            _reject_unknown_keys(te_override, {"family", "label", "components"}, "extras.text_encoder_override")
            family_raw = te_override.get("family")
            label_raw = te_override.get("label")
            if not isinstance(family_raw, str) or not family_raw.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'extras.text_encoder_override.family' must be a non-empty string",
                )
            if not isinstance(label_raw, str) or not label_raw.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'extras.text_encoder_override.label' must be a non-empty string",
                )
            family = family_raw.strip()
            label = label_raw.strip()
            # Cheap sanity: labels from /api/text-encoders use the pattern '<family>/<abs_path>'.
            if "/" in label and not label.startswith(f"{family}/"):
                raise HTTPException(
                    status_code=400,
                    detail="extras.text_encoder_override.label must start with '<family>/'",
                )
            components_val = te_override.get("components")
            components: list[str] | None = None
            if components_val is not None:
                if not isinstance(components_val, list) or any(not isinstance(c, str) for c in components_val):
                    raise HTTPException(
                        status_code=400,
                        detail="'extras.text_encoder_override.components' must be an array of strings",
                    )
                components = [c.strip() for c in components_val if isinstance(c, str) and c.strip()]
            te_cfg: Dict[str, Any] = {"family": family, "label": label}
            if components:
                te_cfg["components"] = components
            extras["text_encoder_override"] = te_cfg

        return extras, highres_cfg


    def _build_highres_fix(cfg: Optional[Dict[str, Any]], width: int, height: int, fallback_cfg: float, fallback_distilled: float = 3.5) -> Dict[str, Any]:
        if cfg is None:
            return {
                "enable": False,
                "denoise": 0.0,
                "scale": 1.0,
                "upscaler": "Use same upscaler",
                "steps": 0,
                "resize_x": width,
                "resize_y": height,
                "hr_checkpoint_name": "Use same checkpoint",
                "hr_additional_modules": [],
                "hr_sampler_name": "Use same sampler",
                "hr_scheduler": "Use same scheduler",
                "hr_prompt": "",
                "hr_negative_prompt": "",
                "hr_cfg": fallback_cfg,
                "hr_distilled_cfg": fallback_distilled,
                "refiner": None,
            }
        return {
            "enable": True,
            "denoise": cfg["denoise"],
            "scale": cfg["scale"],
            "upscaler": cfg["upscaler"],
            "steps": cfg["steps"],
            "resize_x": cfg["resize_x"],
            "resize_y": cfg["resize_y"],
            "hr_checkpoint_name": cfg.get("checkpoint") or "Use same checkpoint",
            "hr_additional_modules": cfg.get("modules") or [],
            "hr_sampler_name": cfg.get("sampler") or "Use same sampler",
            "hr_scheduler": cfg.get("scheduler") or "Use same scheduler",
            "hr_prompt": cfg.get("prompt") or "",
            "hr_negative_prompt": cfg.get("negative_prompt") or "",
            "hr_cfg": cfg.get("cfg") if cfg.get("cfg") is not None else fallback_cfg,
            "hr_distilled_cfg": cfg.get("distilled_cfg") if cfg.get("distilled_cfg") is not None else fallback_distilled,
            "refiner": cfg.get("refiner"),
        }

    def prepare_txt2img(payload: Dict[str, Any]) -> Tuple["Txt2ImgRequest", str, Optional[str]]:
        _reject_unknown_keys(payload, _TXT2IMG_ALLOWED_KEYS, "txt2img")
        prompt = _require_str_field(payload, 'prompt', allow_empty=True)
        negative_prompt = str(payload.get('negative_prompt') or '')
        width = _require_int_field(payload, 'width', minimum=8)
        height = _require_int_field(payload, 'height', minimum=8)
        steps_val = _require_int_field(payload, 'steps', minimum=1)
        cfg_scale = _require_float_field(payload, 'guidance_scale')
        if 'distilled_guidance_scale' in payload:
            distilled_cfg_scale = _require_float_field(payload, 'distilled_guidance_scale')
        else:
            distilled_cfg_scale = 3.5
        sampler_name = _require_str_field(payload, 'sampler', allow_empty=False)
        scheduler_name = _require_str_field(payload, 'scheduler', allow_empty=False)
        seed_val = _require_int_field(payload, 'seed')
        batch_size = _require_int_field(payload, 'batch_size', minimum=1)
        batch_count = _require_int_field(payload, 'batch_count', minimum=1)
        styles = _parse_styles(payload)
        metadata = _parse_metadata(payload)
        extras, highres_cfg = _parse_txt2img_extras(payload)

        metadata.setdefault("mode", _opts_snapshot().codex_mode)
        metadata["styles"] = styles
        metadata["n_iter"] = batch_count
        metadata["batch_count"] = batch_count
        metadata["batch_size"] = batch_size
        metadata["hr"] = bool(highres_cfg)
        metadata["distilled_cfg_scale"] = distilled_cfg_scale

        # Smart offload/fallback flags: prefer payload when present, otherwise fall back to options snapshot.
        snap = _opts_snapshot()
        smart_offload = bool(payload.get("smart_offload", getattr(snap, "codex_smart_offload", False)))
        smart_fallback = bool(payload.get("smart_fallback", getattr(snap, "codex_smart_fallback", False)))
        smart_cache = bool(payload.get("smart_cache", getattr(snap, "codex_smart_cache", True)))

        engine_override = payload.get('engine') or payload.get('codex_engine')
        model_override = payload.get('model') or payload.get('sd_model_checkpoint')

        req = Txt2ImgRequest(
            task=TaskType.TXT2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            guidance_scale=cfg_scale,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            batch_size=batch_size,
            metadata=metadata,
            highres_fix=_build_highres_fix(highres_cfg, width, height, cfg_scale, distilled_cfg_scale),
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
        )

        engine_key = engine_override or snap.codex_engine
        model_ref = model_override or snap.sd_model_checkpoint
        return req, str(engine_key), model_ref

    def encode_images(images: Any) -> list[Dict[str, str]]:  # type: ignore[no-untyped-def]
        encoded: list[Dict[str, str]] = []
        for img in images or []:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            encoded.append({
                "format": "png",
                "data": base64.b64encode(buf.getvalue()).decode('ascii'),
            })
        return encoded


    def _require_explicit_device(payload: Dict[str, Any]) -> str:
        # Accept explicit aliases from payload only (no options fallback)
        raw = (
            payload.get('codex_device')
            or payload.get('device')
            or payload.get('codex_diffusion_device')
            or ""
        )
        dev = str(raw).strip().lower()
        allowed = {"cpu", "cuda", "mps", "xpu", "directml"}
        if dev not in allowed:
            raise HTTPException(status_code=400, detail="Missing or invalid 'codex_device' (cpu|cuda|mps|xpu|directml)")
        try:
            mem_management.switch_primary_device(dev)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return dev

    _ORCH = InferenceOrchestrator()
    
    
    def run_txt2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry) -> None:
        loop = entry.loop
    
        def push(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(entry.queue.put_nowait, event)
    
        def mark_done(success: bool) -> None:
            def _set() -> None:
                if not entry.done.done():
                    entry.done.set_result(success)
    
            loop.call_soon_threadsafe(_set)
    
        push({"type": "status", "stage": "queued"})
        try:
            # Enforce explicit device selection without fallback
            _require_explicit_device(payload)
            req, engine_key, model_ref = prepare_txt2img(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            entry.schedule_cleanup(task_id, delay=0.0)
            with tasks_lock:
                tasks.pop(task_id, None)
            raise
    
        def worker() -> None:
            try:
                push({"type": "status", "stage": "running"})
                engine_options: Dict[str, object] = {}
                try:
                    extras = getattr(req, "extras", {}) or {}
                    te_override = extras.get("text_encoder_override")
                    if isinstance(te_override, dict):
                        engine_options["text_encoder_override"] = dict(te_override)
                except Exception:
                    engine_options = {}
                with tasks_lock:
                    for ev in _ORCH.run(
                        TaskType.TXT2IMG,
                        engine_key,
                        req,
                        model_ref=model_ref,
                        engine_options=engine_options,
                    ):
                        if entry.cancel_requested and entry.cancel_mode == "immediate":
                            entry.error = "cancelled"
                            push({"type": "error", "message": "cancelled"})
                            push({"type": "end"})
                            mark_done(False)
                            return
                        if isinstance(ev, ProgressEvent):
                            push({
                                "type": "progress",
                                "stage": ev.stage,
                                "percent": ev.percent,
                                "step": ev.step,
                                "total_steps": ev.total_steps,
                                "eta_seconds": ev.eta_seconds,
                            })
                        elif isinstance(ev, ResultEvent):
                            payload_obj = ev.payload or {}
                            info_raw = payload_obj.get("info", "{}")
                            try:
                                info_obj = json.loads(info_raw)
                            except Exception:
                                info_obj = info_raw
                            encoded = encode_images(payload_obj.get("images", []))
                            result = {
                                "images": encoded,
                                "info": info_obj,
                            }
                            entry.result = {
                                "status": "completed",
                                "result": result,
                            }
                            push({"type": "result", **result})
                push({"type": "end"})
                mark_done(True)
            except Exception as err:  # pragma: no cover - surfaces runtime errors
                try:
                    from apps.backend.runtime.exception_hook import dump_exception as _dump_exc
                    _dump_exc(type(err), err, err.__traceback__, where='txt2img_worker', context={'task_id': task_id})
                except Exception:
                    pass
                entry.error = str(err)
                push({"type": "error", "message": entry.error})
                push({"type": "end"})
                mark_done(False)
    
        thread = threading.Thread(target=worker, name=f"txt2img-task-{task_id}", daemon=True)
        thread.start()
    
    def prepare_img2img(payload: Dict[str, Any]) -> Tuple[Img2ImgRequest, str, Optional[str]]:
        init_image_data = _p.require(payload, 'img2img_init_image')
        init_image = media.decode_image(init_image_data)
        mask_data = payload.get('img2img_mask')
        mask_image = media.decode_image(mask_data) if mask_data else None
    
        prompt = _p.require(payload, 'img2img_prompt') or ''
        negative_prompt = _p.require(payload, 'img2img_neg_prompt') or ''
        styles = _p.as_list(payload, 'img2img_styles') if 'img2img_styles' in payload else []
        batch_count = _p.as_int(payload, 'img2img_batch_count') if 'img2img_batch_count' in payload else 1
        batch_size = _p.as_int(payload, 'img2img_batch_size') if 'img2img_batch_size' in payload else 1
        steps_val = _p.as_int(payload, 'img2img_steps')
        cfg_scale = _p.as_float(payload, 'img2img_cfg_scale')
        distilled_cfg_scale = _p.as_float_optional(payload, 'img2img_distilled_cfg_scale') if 'img2img_distilled_cfg_scale' in payload else None
        image_cfg_scale = _p.as_float_optional(payload, 'img2img_image_cfg_scale') if 'img2img_image_cfg_scale' in payload else None
        denoise = _p.as_float(payload, 'img2img_denoising_strength')
        width_val = _p.as_int(payload, 'img2img_width')
        height_val = _p.as_int(payload, 'img2img_height')
        sampler_name = _p.require(payload, 'img2img_sampling')
        scheduler_name = _p.require(payload, 'img2img_scheduler')
        seed_val = _p.as_int(payload, 'img2img_seed')
        noise_source = payload.get('img2img_randn_source') or payload.get('img2img_noise_source')
        ensd_raw = payload.get('img2img_eta_noise_seed_delta')
    
        enable_hr = _p.as_bool(payload, 'img2img_hr_enable') if 'img2img_hr_enable' in payload else False
        if enable_hr:
            hr_data = {
                "enable": True,
                "scale": _p.as_float(payload, 'img2img_hr_scale') if 'img2img_hr_scale' in payload else 1.0,
                "resize_x": _p.as_int(payload, 'img2img_hr_resize_x') if 'img2img_hr_resize_x' in payload else 0,
                "resize_y": _p.as_int(payload, 'img2img_hr_resize_y') if 'img2img_hr_resize_y' in payload else 0,
                "steps": _p.as_int(payload, 'img2img_hr_steps') if 'img2img_hr_steps' in payload else 0,
                "denoise": _p.as_float(payload, 'img2img_hr_denoise') if 'img2img_hr_denoise' in payload else denoise,
                "upscaler": payload.get('img2img_hr_upscaler', 'Latent'),
                "hr_prompt": payload.get('img2img_hr_prompt', ''),
                "hr_negative_prompt": payload.get('img2img_hr_neg_prompt', ''),
                "hr_cfg": _p.as_float(payload, 'img2img_hr_cfg') if 'img2img_hr_cfg' in payload else cfg_scale,
                "hr_distilled_cfg": _p.as_float(payload, 'img2img_hr_distilled_cfg') if 'img2img_hr_distilled_cfg' in payload else (distilled_cfg_scale or 3.5),
            }
        else:
            hr_data = {"enable": False}
    
        extras: Dict[str, Any] = {}
        if noise_source:
            extras['randn_source'] = str(noise_source)
        if ensd_raw is not None:
            try:
                extras['eta_noise_seed_delta'] = int(float(ensd_raw))
            except Exception:
                raise HTTPException(status_code=400, detail="img2img_eta_noise_seed_delta must be numeric")
    
        metadata = {
            "styles": styles,
            "distilled_cfg_scale": distilled_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "batch_count": batch_count,
        }
        if noise_source:
            metadata["randn_source"] = str(noise_source)
        if 'eta_noise_seed_delta' in extras:
            metadata["eta_noise_seed_delta"] = extras['eta_noise_seed_delta']
    
        req = Img2ImgRequest(
            task=TaskType.IMG2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            guidance_scale=cfg_scale,
            batch_size=batch_size,
            metadata=metadata,
            init_image=init_image,
            mask=mask_image,
            denoise_strength=denoise,
            width=width_val,
            height=height_val,
            steps=steps_val,
            extras=extras,
            highres_fix=hr_data if hr_data.get("enable") else None,
        )
    
        engine_override = payload.get('engine') or payload.get('codex_engine')
        model_override = payload.get('model') or payload.get('sd_model_checkpoint')
    
        snap = _opts_snapshot()
        engine_key = engine_override or snap.codex_engine
        model_ref = model_override or snap.sd_model_checkpoint
        return req, str(engine_key), model_ref
    
    def run_img2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry) -> None:
        loop = entry.loop
    
        def push(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(entry.queue.put_nowait, event)
    
        def mark_done(success: bool) -> None:
            def _set() -> None:
                if not entry.done.done():
                    entry.done.set_result(success)
    
            loop.call_soon_threadsafe(_set)
    
        push({"type": "status", "stage": "queued"})
        try:
            _require_explicit_device(payload)
            req, engine_key, model_ref = prepare_img2img(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            entry.schedule_cleanup(task_id, delay=0.0)
            with tasks_lock:
                tasks.pop(task_id, None)
            raise
    
        def worker() -> None:
            try:
                push({"type": "status", "stage": "running"})
                with tasks_lock:
                    for ev in _ORCH.run(TaskType.IMG2IMG, engine_key, req, model_ref=model_ref):
                        if entry.cancel_requested and entry.cancel_mode == "immediate":
                            entry.error = "cancelled"
                            push({"type": "error", "message": "cancelled"})
                            push({"type": "end"})
                            mark_done(False)
                            return
                        if isinstance(ev, ProgressEvent):
                            push({
                                "type": "progress",
                                "stage": ev.stage,
                                "percent": ev.percent,
                                "step": ev.step,
                                "total_steps": ev.total_steps,
                                "eta_seconds": ev.eta_seconds,
                            })
                        elif isinstance(ev, ResultEvent):
                            payload_obj = ev.payload or {}
                            info_raw = payload_obj.get("info", "{}")
                            try:
                                info_obj = json.loads(info_raw)
                            except Exception:
                                info_obj = info_raw
                            encoded = encode_images(payload_obj.get("images", []))
                            result = {"images": encoded, "info": info_obj}
                            entry.result = {"status": "completed", "result": result}
                            push({"type": "result", **result})
                push({"type": "end"})
                mark_done(True)
            except Exception as err:
                try:
                    from apps.backend.runtime.exception_hook import dump_exception as _dump_exc
                    _dump_exc(type(err), err, err.__traceback__, where='img2img_worker', context={'task_id': task_id})
                except Exception:
                    pass
                entry.error = str(err)
                push({"type": "error", "message": entry.error})
                push({"type": "end"})
                mark_done(False)
    
        thread = threading.Thread(target=worker, name=f"img2img-task-{task_id}", daemon=True)
        thread.start()
    
    def prepare_txt2vid(payload: Dict[str, Any]) -> Tuple[Txt2VidRequest, str, Optional[str]]:
        prompt = payload.get('txt2vid_prompt', '')
        negative_prompt = payload.get('txt2vid_neg_prompt', '')
        width_val = int(payload.get('txt2vid_width', 768))
        height_val = int(payload.get('txt2vid_height', 432))
        steps_val = int(payload.get('txt2vid_steps', 30))
        fps_val = int(payload.get('txt2vid_fps', 24))
        frames_val = int(payload.get('txt2vid_num_frames', 16))
        sampler_name = str(payload.get('txt2vid_sampler', payload.get('txt2vid_sampling', 'Euler')))
        scheduler_name = str(payload.get('txt2vid_scheduler', 'Automatic'))
        seed_val = int(payload.get('txt2vid_seed', -1))
        cfg_val = float(payload.get('txt2vid_cfg_scale', 7.0))
    
        extras: Dict[str, Any] = {}
        # Video export options (pass-through; engines may consume)
        video_opts_keys = [
            'video_filename_prefix','video_format','video_pix_fmt','video_crf','video_loop_count','video_pingpong','video_save_metadata','video_save_output','video_trim_to_audio'
        ]
        video_opts = {k: payload.get(k) for k in video_opts_keys if k in payload}
        if video_opts:
            extras['video'] = video_opts
        if isinstance(payload.get('video_interpolation'), dict):
            extras['video_interpolation'] = payload.get('video_interpolation')
        if isinstance(payload.get('wan_high'), dict):
            extras['wan_high'] = payload.get('wan_high')
        if isinstance(payload.get('wan_low'), dict):
            extras['wan_low'] = payload.get('wan_low')
        # Pass-through of explicit WAN GGUF complements (no guessing)
        for key in (
            'wan_format',
            'wan_vae_path',
            'wan_text_encoder_path',
            'wan_text_encoder_dir',
            'wan_metadata_dir',
            'wan_tokenizer_dir',
            # memory/attention/runtime controls
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            # TE kernel/device controls
            'gguf_te_device',
            'gguf_te_impl',
            'gguf_te_kernel_required',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)
    
        req = Txt2VidRequest(
            task=TaskType.TXT2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width_val,
            height=height_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            extras=extras,
            metadata={
                "styles": payload.get('txt2vid_styles', []),
                "mode": _opts_snapshot().codex_mode,
            },
        )
    
        # Select engine: prefer explicit WAN extras, then semantic detection, then WAN default
        engine_key = None
        if extras.get('wan_high') or extras.get('wan_low'):
            engine_key = _pick_wan_engine(extras)
        else:
            sem = _detect_semantic_engine()
            if sem == 'wan22':
                engine_key = 'wan22_14b'
            elif sem == 'hunyuan_video':
                engine_key = 'hunyuan_video'
            else:
                engine_key = 'wan22_14b'
        # Choose model_ref: if WAN extras provide model_dir, use it; else fallback to current checkpoint
        model_ref = _opts_snapshot().sd_model_checkpoint
        try:
            wh = extras.get('wan_high') or {}
            wl = extras.get('wan_low') or {}
            if isinstance(wh, dict) and wh.get('model_dir'):
                model_ref = str(wh.get('model_dir'))
            elif isinstance(wl, dict) and wl.get('model_dir'):
                model_ref = str(wl.get('model_dir'))
        except Exception:
            pass
        return req, str(engine_key), model_ref
    
    def prepare_img2vid(payload: Dict[str, Any]) -> Tuple[Img2VidRequest, str, Optional[str]]:
        logging.getLogger('backend.api').info('[api] DEBUG: enter prepare_img2vid')
        prompt = payload.get('img2vid_prompt', '')
        negative_prompt = payload.get('img2vid_neg_prompt', '')
        width_val = int(payload.get('img2vid_width', 768))
        height_val = int(payload.get('img2vid_height', 432))
        steps_val = int(payload.get('img2vid_steps', 30))
        fps_val = int(payload.get('img2vid_fps', 24))
        frames_val = int(payload.get('img2vid_num_frames', 16))
        sampler_name = str(payload.get('img2vid_sampler', payload.get('img2vid_sampling', 'Euler')))
        scheduler_name = str(payload.get('img2vid_scheduler', 'Automatic'))
        seed_val = int(payload.get('img2vid_seed', -1))
        cfg_val = float(payload.get('img2vid_cfg_scale', 7.0))
    
        init_image_data = payload.get('img2vid_init_image')
        init_image = media.decode_image(init_image_data) if init_image_data else None
    
        extras: Dict[str, Any] = {}
        video_opts_keys = [
            'video_filename_prefix','video_format','video_pix_fmt','video_crf','video_loop_count','video_pingpong','video_save_metadata','video_save_output','video_trim_to_audio'
        ]
        video_opts = {k: payload.get(k) for k in video_opts_keys if k in payload}
        if video_opts:
            extras['video'] = video_opts
        if isinstance(payload.get('video_interpolation'), dict):
            extras['video_interpolation'] = payload.get('video_interpolation')
        if isinstance(payload.get('wan_high'), dict):
            extras['wan_high'] = payload.get('wan_high')
        if isinstance(payload.get('wan_low'), dict):
            extras['wan_low'] = payload.get('wan_low')
        # Pass-through of explicit WAN GGUF complements (no guessing)
        for key in (
            'wan_format',
            'wan_vae_path',
            'wan_text_encoder_path',
            'wan_text_encoder_dir',
            'wan_metadata_dir',
            'wan_tokenizer_dir',
            # memory/attention/runtime controls
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            # TE kernel/device controls
            'gguf_te_device',
            'gguf_te_impl',
            'gguf_te_kernel_required',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)
    
        req = Img2VidRequest(
            task=TaskType.IMG2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            width=width_val,
            height=height_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            extras=extras,
            metadata={
                "styles": payload.get('img2vid_styles', []),
                "mode": _opts_snapshot().codex_mode,
            },
        )
    
        # Select engine: prefer explicit WAN extras, then semantic detection, then WAN default
        engine_key = None
        if extras.get('wan_high') or extras.get('wan_low'):
            engine_key = _pick_wan_engine(extras)
        else:
            sem = _detect_semantic_engine()
            if sem == 'wan22':
                engine_key = 'wan22_14b'
            elif sem == 'hunyuan_video':
                engine_key = 'hunyuan_video'
            else:
                engine_key = 'wan22_14b'
        # Choose model_ref from WAN extras when provided
        model_ref = _opts_snapshot().sd_model_checkpoint
        try:
            wh = extras.get('wan_high') or {}
            wl = extras.get('wan_low') or {}
            if isinstance(wh, dict) and wh.get('model_dir'):
                model_ref = str(wh.get('model_dir'))
            elif isinstance(wl, dict) and wl.get('model_dir'):
                model_ref = str(wl.get('model_dir'))
        except Exception:
            pass
        logging.getLogger('backend.api').info('[api] DEBUG: exit prepare_img2vid engine=%s model_ref=%s size=%dx%d frames=%d', engine_key, model_ref, width_val, height_val, frames_val)
        return req, str(engine_key), model_ref
    
    def run_video_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, task_type: TaskType) -> None:
        loop = entry.loop
    
        def push(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(entry.queue.put_nowait, event)
    
        def mark_done(success: bool) -> None:
            def _set() -> None:
                if not entry.done.done():
                    entry.done.set_result(success)
    
            loop.call_soon_threadsafe(_set)
    
        push({"type": "status", "stage": "queued"})
        try:
            _require_explicit_device(payload)
            if task_type == TaskType.TXT2VID:
                req, engine_key, model_ref = prepare_txt2vid(payload)
            else:
                req, engine_key, model_ref = prepare_img2vid(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            entry.schedule_cleanup(task_id, delay=0.0)
            with tasks_lock:
                tasks.pop(task_id, None)
            raise
    
        def worker() -> None:
            logging.getLogger('backend.api').info('[api] DEBUG: enter worker task_id=%s type=%s engine=%s model=%s', task_id, task_type.value, engine_key, model_ref)
            try:
                push({"type": "status", "stage": "running"})
                with tasks_lock:
                    orch = InferenceOrchestrator()
                    engine_opts = {"export_video": bool(_opts_snapshot().codex_export_video)}
                    for ev in orch.run(task_type, engine_key, req, model_ref=model_ref, engine_options=engine_opts):
                        if entry.cancel_requested and entry.cancel_mode == "immediate":
                            entry.error = "cancelled"
                            push({"type": "error", "message": "cancelled"})
                            push({"type": "end"})
                            mark_done(False)
                            return
                        if isinstance(ev, ProgressEvent):
                            push({
                                "type": "progress",
                                "stage": ev.stage,
                                "percent": ev.percent,
                                "step": ev.step,
                                "total_steps": ev.total_steps,
                                "eta_seconds": ev.eta_seconds,
                            })
                        elif isinstance(ev, ResultEvent):
                            payload_obj = ev.payload or {}
                            info_raw = payload_obj.get("info", "{}")
                            try:
                                info_obj = json.loads(info_raw)
                            except Exception:
                                info_obj = info_raw
                            encoded = encode_images(payload_obj.get("images", []))
                            result = {"images": encoded, "info": info_obj}
                            entry.result = {"status": "completed", "result": result}
                            push({"type": "result", **result})
                push({"type": "end"})
                mark_done(True)
                logging.getLogger('backend.api').info('[api] DEBUG: exit worker task_id=%s', task_id)
            except Exception as err:
                try:
                    from apps.backend.runtime.exception_hook import dump_exception as _dump_exc
                    _dump_exc(type(err), err, err.__traceback__, where=f'{label}_worker', context={'task_id': task_id})
                except Exception:
                    pass
                entry.error = str(err)
                push({"type": "error", "message": entry.error})
                push({"type": "end"})
                mark_done(False)
    
        label = 'txt2vid' if task_type == TaskType.TXT2VID else 'img2vid'
        thread = threading.Thread(target=worker, name=f"{label}-task-{task_id}", daemon=True)
        thread.start()
    
    @app.post('/api/txt2img')
    async def txt2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        if payload.get('__strict_version') != 1:
            raise HTTPException(status_code=400, detail="Missing __strict_version == 1")
    
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-{uuid4().hex})"
        register_task(task_id, entry)
        run_txt2img_task(task_id, payload, entry)
        return {"task_id": task_id}
    
    @app.post('/api/img2img')
    async def img2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        if payload.get('__strict_version') != 1:
            raise HTTPException(status_code=400, detail="Missing __strict_version == 1")
    
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2img-{uuid4().hex})"
        register_task(task_id, entry)
        run_img2img_task(task_id, payload, entry)
        return {"task_id": task_id}
    
    @app.post('/api/txt2vid')
    async def txt2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        if payload.get('__strict_version') != 1:
            raise HTTPException(status_code=400, detail="Missing __strict_version == 1")
    
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-txt2vid-{uuid4().hex})"
        register_task(task_id, entry)
        run_video_task(task_id, payload, entry, TaskType.TXT2VID)
        return {"task_id": task_id}
    
    @app.post('/api/img2vid')
    async def img2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        logging.getLogger('backend.api').info('[api] DEBUG: POST /api/img2vid received')
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        if payload.get('__strict_version') != 1:
            raise HTTPException(status_code=400, detail="Missing __strict_version == 1")
    
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2vid-{uuid4().hex})"
        register_task(task_id, entry)
        logging.getLogger('backend.api').info('[api] DEBUG: scheduling img2vid task_id=%s', task_id)
        run_video_task(task_id, payload, entry, TaskType.IMG2VID)
        return {"task_id": task_id}
    
    @app.get('/api/tasks/{task_id}')
    async def task_status(task_id: str) -> Dict[str, Any]:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if entry.done.done():
            if entry.error:
                return {"status": "error", "error": entry.error}
            entry.schedule_cleanup(task_id)
            return entry.result or {"status": "completed", "result": {}}
        return {"status": "running"}
    
    @app.get('/api/tasks/{task_id}/events')
    async def task_events(task_id: str) -> StreamingResponse:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")

        async def event_stream():
            while True:
                payload = await entry.queue.get()
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get('type') == 'end':
                    entry.schedule_cleanup(task_id)
                    break

        return StreamingResponse(event_stream(), media_type='text/event-stream')

    @app.post('/api/tasks/{task_id}/cancel')
    async def task_cancel(task_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
        mode_raw = str(payload.get('mode', 'immediate')).strip().lower() if isinstance(payload, dict) else 'immediate'
        mode = 'after_current' if mode_raw == 'after_current' else 'immediate'
        ok = request_task_cancel(task_id, mode=mode)
        if not ok:
            raise HTTPException(status_code=404, detail="Task not found")
        if mode == 'immediate':
            try:
                backend_state.stop_generating()
            except Exception:
                pass
        return {"status": "cancelling", "mode": mode}

    # Serve built UI after API routes so /api/* is matched before the SPA fallback
    if os.path.isdir(_ui_dist_dir):
        app.mount('/', SPAStaticFiles(directory=_ui_dist_dir, html=True), name='ui')
    logging.getLogger('backend.api').info('build_app finished')
    if not isinstance(app, FastAPI):
        raise RuntimeError(f"build_app invariant violated: expected FastAPI, got {type(app)}")
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
    mem_management.reinitialize(runtime_config)
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
    snapshot = codex_options.get_snapshot()
    ns = _bootstrap_runtime(argv_seq, env or os.environ, snapshot.as_dict())
    _enable_trace_debug(ns)
    ensure_initialized()
    # Build a fresh app each time to avoid stale/None globals under factory mode
    app = build_app()
    if app is None:
        logging.getLogger("backend.api").error(
            "build_app() returned None; constructing minimal API fallback (health/version only)."
        )
        fallback = FastAPI(title="SD WebUI API (fallback)", version="0.0.0-fallback")

        @fallback.get("/api/health")
        def _health() -> Dict[str, bool]:
            return {"ok": True}

        @fallback.get("/api/version")
        def _version() -> Dict[str, Any]:
            return {
                "app_version": fallback.version,
                "git_commit": None,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "torch_version": None,
                "cuda_version": None,
            }

        app = fallback
    global _APP
    _APP = app
    return app

# Provide module-level ASGI app for ASGI servers that import run_api:app directly
async def app(scope, receive, send):  # type: ignore[override]
    """ASGI entrypoint for non-factory launches (uvicorn run_api:app)."""
    global _APP
    if _APP is None:
        _APP = create_api_app(argv=sys.argv[1:], env=os.environ)
    await _APP(scope, receive, send)


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

    uvicorn.run(api_app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
