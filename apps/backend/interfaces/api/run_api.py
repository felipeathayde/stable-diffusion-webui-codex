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
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.interfaces.api import codex_api
from apps.backend.runtime.pipeline_debug import apply_env_flag as _apply_pipeline_debug_flag
from apps.backend.runtime.models import api as model_api

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

    if os.path.isdir(_ui_dist_dir):
        app.mount('/', SPAStaticFiles(directory=_ui_dist_dir, html=True), name='ui')

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
            raise HTTPException(status_code=500, detail='ui presets not found')
        if _ui_presets_cache is not None and _ui_presets_mtime == mtime:
            return _ui_presets_cache
        data = _load_json(presets_path)
        if not data or 'presets' not in data:
            raise HTTPException(status_code=500, detail='invalid ui presets json')
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
        """Inventory of model-related assets discovered at startup (strict dirs).

        - vaes: files under /models/VAE
        - text_encoders: files under /models/text-encoder
        - loras: files under /models/Lora
        - wan22.gguf: files under /models/wan22
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
        from apps.backend.engines.util.schedulers import SamplerKind
        kinds = [
            (SamplerKind.EULER.value, ["k_euler"]),
            (SamplerKind.EULER_A.value, ["k_euler_a", "euler_a"]),
            (SamplerKind.DDIM.value, ["ddim"]),
            (SamplerKind.DPM2M.value, ["dpmpp_2m", "dpm++ 2m"]),
            (SamplerKind.DPM2M_SDE.value, ["dpmpp_2m_sde", "dpm++ 2m sde"]),
            (SamplerKind.PLMS.value, ["lms"]),
            (SamplerKind.PNDM.value, ["pndm"]),
            (SamplerKind.UNI_PC.value, ["unipc", "uni_pc"]),
        ]
        samplers = [{"name": n, "aliases": a, "options": {}} for n, a in kinds]
        return {"samplers": samplers}

    @app.get('/api/schedulers')
    def list_schedulers() -> Dict[str, Any]:
        schedulers = [
            {"name": "EulerDiscreteScheduler", "label": "Euler", "aliases": ["euler"]},
            {"name": "EulerAncestralDiscreteScheduler", "label": "Euler a", "aliases": ["euler a"]},
            {"name": "DDIMScheduler", "label": "DDIM", "aliases": ["ddim"]},
            {"name": "DPMSolverMultistepScheduler", "label": "DPM++ 2M", "aliases": ["dpm++ 2m", "dpmpp_2m"]},
            {"name": "LMSDiscreteScheduler", "label": "PLMS", "aliases": ["plms"]},
            {"name": "PNDMScheduler", "label": "PNDM", "aliases": ["pndm"]},
        ]
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
        """Return available text encoder modules and current selection list (native registry)."""
        from apps.backend.infra.registry.text_encoders import list_text_encoders as _list_tes, describe_text_encoders as _describe_tes
        from apps.backend.codex import options as _codex_opts
        try:
            mapping = _list_tes()
            info = _describe_tes()
            # current selection from codex options (paths/names as configured)
            selected = _codex_opts.get_additional_modules()
            return {"text_encoders": mapping, "current": selected, "text_encoders_info": info}
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
        cfg_path = os.path.join(os.getcwd(), 'apps', 'paths.json')
        data = _load_json(cfg_path)
        # Expected structure: { "checkpoints": [], "vae": [], "lora": [], "text_encoders": [] }
        return {"paths": data}

    @app.post('/api/paths')
    def set_paths(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict) or 'paths' not in payload or not isinstance(payload['paths'], dict):
            raise HTTPException(status_code=400, detail='payload must be {"paths": {...}}')
        cfg_path = os.path.join(os.getcwd(), 'apps', 'paths.json')
        _save_json(cfg_path, payload['paths'])
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

    def prepare_txt2img(payload: Dict[str, Any]) -> Tuple[Txt2ImgRequest, str, Optional[str]]:
        print("[api] preparing txt2img request from payload")
        prompt = _p.require(payload, 'txt2img_prompt') or ''
        negative_prompt = _p.require(payload, 'txt2img_neg_prompt') or ''
        prompt_styles = _p.as_list(payload, 'txt2img_styles')
        n_iter = _p.as_int(payload, 'txt2img_batch_count')
        batch_size = _p.as_int(payload, 'txt2img_batch_size')
        cfg_scale = _p.as_float(payload, 'txt2img_cfg_scale')
        distilled_cfg_scale = _p.as_float_optional(payload, 'txt2img_distilled_cfg_scale', 3.5)
        height = _p.as_int(payload, 'txt2img_height')
        width = _p.as_int(payload, 'txt2img_width')
        enable_hr = _p.as_bool(payload, 'txt2img_hr_enable')
        steps_val = _p.as_int(payload, 'txt2img_steps')
        sampler_name = _p.require(payload, 'txt2img_sampling')
        scheduler_name = _p.require(payload, 'txt2img_scheduler')
        seed_val = _p.as_int(payload, 'txt2img_seed')
        noise_source = payload.get('txt2img_randn_source') or payload.get('txt2img_noise_source')
        ensd_raw = payload.get('txt2img_eta_noise_seed_delta')

        if enable_hr:
            denoising_strength = _p.as_float(payload, 'txt2img_denoising_strength')
            hr_scale = _p.as_float(payload, 'txt2img_hr_scale')
            hr_upscaler = _p.require(payload, 'txt2img_hr_upscaler')
            hr_second_pass_steps = _p.as_int(payload, 'txt2img_hires_steps')
            hr_resize_x = _p.as_int(payload, 'txt2img_hr_resize_x')
            hr_resize_y = _p.as_int(payload, 'txt2img_hr_resize_y')
            hr_checkpoint_name = payload.get('hr_checkpoint_name') or _p.require(payload, 'hr_checkpoint')
            hr_additional_modules = payload.get('hr_additional_modules') or _p.as_list(payload, 'hr_vae_te')
            hr_sampler_name = payload.get('hr_sampler_name') or _p.require(payload, 'hr_sampler')
            hr_scheduler = payload.get('hr_scheduler') or _p.require(payload, 'hr_scheduler')
            hr_prompt = payload.get('txt2img_hr_prompt') or ''
            hr_negative_prompt = payload.get('txt2img_hr_neg_prompt') or ''
            hr_cfg = _p.as_float(payload, 'txt2img_hr_cfg')
            hr_distilled_cfg = _p.as_float(payload, 'txt2img_hr_distilled_cfg')
        else:
            denoising_strength = 0.0
            hr_scale = 1.0
            hr_upscaler = 'Use same upscaler'
            hr_second_pass_steps = 0
            hr_resize_x = width
            hr_resize_y = height
            hr_checkpoint_name = 'Use same checkpoint'
            hr_additional_modules = ['Use same choices']
            hr_sampler_name = 'Use same sampler'
            hr_scheduler = 'Use same scheduler'
            hr_prompt = ''
            hr_negative_prompt = ''
            hr_cfg = cfg_scale
            hr_distilled_cfg = distilled_cfg_scale

        extras: Dict[str, Any] = {}
        if noise_source:
            extras['randn_source'] = str(noise_source)
        if ensd_raw is not None:
            try:
                extras['eta_noise_seed_delta'] = int(float(ensd_raw))
            except Exception:
                raise HTTPException(status_code=400, detail="txt2img_eta_noise_seed_delta must be numeric")

        metadata = {
            "mode": _opts_snapshot().codex_mode,
            "styles": prompt_styles,
            "distilled_cfg_scale": distilled_cfg_scale,
            "hr": bool(enable_hr),
            "n_iter": n_iter,
        }
        if noise_source:
            metadata["randn_source"] = str(noise_source)
        if 'eta_noise_seed_delta' in extras:
            metadata["eta_noise_seed_delta"] = extras['eta_noise_seed_delta']

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
            highres_fix={
                "enable": bool(enable_hr),
                "denoise": denoising_strength,
                "scale": hr_scale,
                "upscaler": hr_upscaler,
                "steps": hr_second_pass_steps,
                "resize_x": hr_resize_x,
                "resize_y": hr_resize_y,
                "hr_checkpoint_name": hr_checkpoint_name,
                "hr_additional_modules": hr_additional_modules,
                "hr_sampler_name": hr_sampler_name,
                "hr_scheduler": hr_scheduler,
                "hr_prompt": hr_prompt,
                "hr_negative_prompt": hr_negative_prompt,
                "hr_cfg": hr_cfg,
                "hr_distilled_cfg": hr_distilled_cfg,
            },
            extras=extras,
        )

        snap = _opts_snapshot()
        engine_key = engine_override or snap.codex_engine
        model_ref = model_override or snap.sd_model_checkpoint
        print(f"[api] txt2img prepared: engine={engine_key}, model={model_ref}, prompt_len={len(prompt)}")
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
            req, engine_key, model_ref = prepare_txt2img(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            return

        def worker() -> None:
            try:
                push({"type": "status", "stage": "running"})
                with tasks_lock:
                    orch = InferenceOrchestrator()
                    for ev in orch.run(TaskType.TXT2IMG, engine_key, req, model_ref=model_ref):
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

        snap = _opts_snapshot()
        engine_key = snap.codex_engine
        model_ref = snap.sd_model_checkpoint
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
            req, engine_key, model_ref = prepare_img2img(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            return

        def worker() -> None:
            try:
                push({"type": "status", "stage": "running"})
                with tasks_lock:
                    orch = InferenceOrchestrator()
                    for ev in orch.run(TaskType.IMG2IMG, engine_key, req, model_ref=model_ref):
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
            if task_type == TaskType.TXT2VID:
                req, engine_key, model_ref = prepare_txt2vid(payload)
            else:
                req, engine_key, model_ref = prepare_img2vid(payload)
        except Exception as err:
            entry.error = str(err)
            push({"type": "error", "message": entry.error})
            push({"type": "end"})
            mark_done(False)
            return

        def worker() -> None:
            logging.getLogger('backend.api').info('[api] DEBUG: enter worker task_id=%s type=%s engine=%s model=%s', task_id, task_type.value, engine_key, model_ref)
            try:
                push({"type": "status", "stage": "running"})
                with tasks_lock:
                    orch = InferenceOrchestrator()
                    engine_opts = {"export_video": bool(_opts_snapshot().codex_export_video)}
                    for ev in orch.run(task_type, engine_key, req, model_ref=model_ref, engine_options=engine_opts):
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

    # Legacy callbacks are not used in the native backend entrypoint
    app.include_router(codex_api.router)
    return app


def main() -> None:
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

    ensure_initialized()
    uvicorn.run(build_app(), host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
