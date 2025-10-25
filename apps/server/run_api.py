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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from colorama import Fore, Style  # type: ignore

    def color_cyan(s: str) -> str: return Fore.CYAN + s + Style.RESET_ALL
    def color_red(s: str) -> str: return Fore.RED + s + Style.RESET_ALL
except Exception:  # pragma: no cover - optional dependency missing

    def color_cyan(s: str) -> str: return s
    def color_red(s: str) -> str: return s

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Install global exception hooks as early as possible so any startup errors are dumped
try:
    from apps.server.backend.runtime.exception_hook import install_exception_hooks as _install_exc_hooks
    _EXC_LOG_PATH = _install_exc_hooks(log_dir=str(PROJECT_ROOT / 'logs'))
except Exception:
    _EXC_LOG_PATH = None


_initialized = False


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    try:
        from apps.server.backend.codex.initialization import (
            initialize_codex,
        )
    except Exception:  # pragma: no cover - optional compatibility layer
        def initialize_codex() -> None:
            return None

    from modules import initialize, script_callbacks

    initialize_codex()
    initialize.imports()
    initialize.check_versions()
    initialize.initialize()
    # Allow scripts to run their setup hooks (mirrors webui before UI launch)
    script_callbacks.before_ui_callback()

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

    from modules import script_callbacks as _script_callbacks
    from modules import shared as _shared
    from modules import sd_models as _sd_models
    from modules import sd_samplers as _sd_samplers
    from modules import sd_schedulers as _sd_schedulers
    from modules import call_queue as _call_queue
    import modules.txt2img as _txt2img
    import modules.img2img as _img2img
    import modules.txt2vid as _txt2vid
    import modules.img2vid as _img2vid
    from apps.server.backend.core.engine_interface import TaskType
    from apps.server.backend.core.orchestrator import InferenceOrchestrator
    from apps.server.backend.core.requests import (
        ProgressEvent,
        ResultEvent,
        Txt2ImgRequest,
        Img2ImgRequest,
        Txt2VidRequest,
        Img2VidRequest,
    )
    from apps.server.backend.services.media_service import MediaService

    app = FastAPI(title='SD WebUI API', version='0.2.0')
    # Ensure engines are registered (sd15/sdxl/flux + video engines incl. WAN)
    try:
        from apps.server.backend.engines import register_default_engines as _register_default_engines  # type: ignore
        _register_default_engines(replace=False)
    except Exception as _e:  # pragma: no cover
        print(color_red(f"[engines] registration failed: {_e}"))

    # Defensive: ensure WAN22 engines are present even if registration import failed
    try:
        from apps.server.backend.core.registry import registry as _engine_registry
        need = {"wan22_14b", "wan22_5b"} - set(_engine_registry.list())
        if need:
            from apps.server.backend.engines.diffusion.wan22_14b import Wan2214BEngine  # type: ignore
            from apps.server.backend.engines.diffusion.wan22_5b import Wan225BEngine  # type: ignore
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
        from apps.server.backend.runtime.exception_hook import attach_asyncio as _attach_asyncio, dump_current_exception as _dump_current_exception

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
        from apps.server.settings_registry import schema_to_json as _schema_hardcoded, field_index as _field_index, SettingType as _SettingType  # type: ignore
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
            from apps.server.backend import memory_management as _mm  # type: ignore
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
                schema_path = os.path.join(os.getcwd(), 'apps', 'server', 'settings_schema.json')
                _settings_schema_cache = _load_json(schema_path)
                if not _settings_schema_cache:
                    raise HTTPException(status_code=500, detail='settings schema not found (registry and JSON)')

        # Hydrate dynamic choices on-the-fly
        try:
            from apps.server.settings_hydration import hydrate_schema  # type: ignore
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
        try:
            from modules import shared as _shared, sd_models as _sd_models
        except Exception:
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
        title = getattr(_shared.opts, 'sd_model_checkpoint', '') or ''
        title_l = str(title).lower()
        meta = {}
        try:
            # Find checkpoint info by title
            info = None
            for _k, v in getattr(_sd_models, 'checkpoints_list', {}).items():
                if getattr(v, 'title', None) == title:
                    info = v
                    break
            if info is not None:
                meta = getattr(info, 'metadata', {}) or {}
        except Exception:
            meta = {}

        def has_meta(keys: list[str], substr: str) -> bool:
            s = str(substr).lower()
            for k in keys:
                v = str(meta.get(k, '')).lower()
                if s in v:
                    return True
            return False

        # WAN 2.2
        if has_meta(['modelspec.architecture', 'architecture', 'model_name'], 'wan') or 'wan' in title_l:
            return 'wan22'
        # Hunyuan Video
        if has_meta(['modelspec.architecture', 'architecture', 'model_name'], 'hunyuan') or 'hunyuan' in title_l:
            return 'hunyuan_video'
        # SVD
        if has_meta(['modelspec.architecture', 'architecture', 'model_name'], 'svd') or 'svd' in title_l:
            return 'svd'
        # FLUX
        if has_meta(['modelspec.architecture', 'architecture', 'model_name'], 'flux') or 'flux' in title_l:
            return 'flux'
        # SDXL
        if 'xl' in title_l or has_meta(['modelspec.architecture', 'architecture', 'model_name'], 'xl'):
            return 'sdxl'
        return 'sd15'

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

        try:
            from modules import sd_models as _sd_models
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'models not available: {e}')
        # Build list of checkpoint titles
        titles = []
        try:
            for info in _sd_models.checkpoints_list.values():
                titles.append(info.title)
        except Exception:
            pass
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
        try:
            from modules import shared as _shared
            # sd_model_checkpoint
            _shared.opts.set('sd_model_checkpoint', target, is_api=True)
            # extra options if any
            extra = preset.get('options') or {}
            if isinstance(extra, dict):
                for k, v in extra.items():
                    try:
                        casted = _shared.opts.cast_value(k, v)
                    except Exception:
                        casted = v
                    _shared.opts.set(k, casted, is_api=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'failed to apply preset: {e}')

        return {"applied": True, "model": target}

    @app.get('/api/settings/values')
    def settings_values() -> Dict[str, Any]:
        try:
            from modules import shared as _shared
            vals = {}
            idx = _field_index() if _settings_registry_ok else {}
            keys = set(idx.keys()) if idx else set(_shared.opts.data.keys())
            for k in keys:
                if k in _shared.opts.data:
                    vals[k] = _shared.opts.data.get(k)
            return {"values": vals}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read values: {e}")

    # Load saved settings on startup and apply to shared.opts with validation
    def _apply_saved_settings() -> None:
        if not _settings_registry_ok:
            return
        try:
            from modules import shared as _shared
        except Exception:
            return
        saved = _load_json(_settings_values_path)
        if not isinstance(saved, dict) or not saved:
            return
        idx = _field_index()
        applied, skipped = 0, 0
        for k, v in saved.items():
            f = idx.get(k)
            if not f:
                skipped += 1
                continue
            try:
                # Validate basic domain rules
                if getattr(f, 'choices', None) and isinstance(f.choices, list):
                    if v not in f.choices:
                        skipped += 1
                        continue
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    try:
                        num = float(v)
                        lo = f.min if isinstance(getattr(f, 'min', None), (int, float)) else None
                        hi = f.max if isinstance(getattr(f, 'max', None), (int, float)) else None
                        if lo is not None:
                            num = max(num, lo)
                        if hi is not None:
                            num = min(num, hi)
                        v = num
                    except Exception:
                        skipped += 1
                        continue
                if getattr(f, 'type', None) == _SettingType.CHECKBOX:
                    if isinstance(v, str):
                        v = v.lower() in ('1', 'true', 'yes', 'on')
                    else:
                        v = bool(v)
                # Cast and apply
                try:
                    casted = _shared.opts.cast_value(k, v)
                except Exception:
                    casted = v
                if _shared.opts.set(k, casted, is_api=True):
                    applied += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if applied:
            print(color_cyan(f"[settings] applied {applied} saved item(s); skipped {skipped}"))

    # Apply saved settings early (after modules init) before serving
    try:
        _apply_saved_settings()
    except Exception as e:  # pragma: no cover
        print(color_red(f"[settings] failed to apply saved settings: {e}"))

    @app.get('/api/models')
    def list_models() -> Dict[str, Any]:
        """List checkpoints discovered by the native registry.

        Response shape remains compatible: fields include title/name/filename/metadata.
        """
        from apps.server.backend.registry.checkpoints import list_checkpoints as _list_ckpt

        def _serialize(entry) -> Dict[str, Any]:
            return {
                "title": entry.title,
                "name": entry.name,
                "model_name": entry.model_name,
                "hash": entry.short_hash,
                "filename": entry.filename,
                "metadata": entry.metadata,
            }

        entries = _list_ckpt()
        models = [_serialize(e) for e in entries]
        # Current selection: track via codex options when available
        try:
            from apps.server.backend.codex import main as _codex
            current = getattr(_codex, "_SELECTIONS").checkpoint_name or (models[0]["name"] if models else None)
        except Exception:
            current = models[0]["name"] if models else None
        return {"models": models, "current": current}

    @app.get('/api/samplers')
    def list_samplers() -> Dict[str, Any]:
        _sd_samplers.set_samplers()
        samplers = [_serialize_sampler(s) for s in _sd_samplers.visible_samplers()]
        return {"samplers": samplers}

    @app.get('/api/schedulers')
    def list_schedulers() -> Dict[str, Any]:
        schedulers = [_serialize_scheduler(s) for s in _sd_schedulers.schedulers]
        return {"schedulers": schedulers}

    @app.get('/api/vaes')
    def list_vaes() -> Dict[str, Any]:
        """Return available VAE modules and current selection (native registry)."""
        from apps.server.backend.codex import options as _codex_opts
        from apps.server.backend.registry.vae import list_vaes as _list_vaes, describe_vaes as _describe_vaes
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
        from apps.server.backend.registry.text_encoders import list_text_encoders as _list_tes, describe_text_encoders as _describe_tes
        from apps.server.backend.codex import options as _codex_opts
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
        """Return Textual Inversion embeddings (loaded + skipped)."""
        nonlocal _embedding_db
        try:
            from modules import shared as _s
            from modules.textual_inversion.textual_inversion import EmbeddingDatabase  # type: ignore
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Embeddings not available: {e}")

        if _embedding_db is None:
            db = EmbeddingDatabase()
            # Prefer configured directory
            try:
                db.add_embedding_dir(_s.cmd_opts.embeddings_dir)
            except Exception:
                pass
            try:
                db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load embeddings: {e}")
            _embedding_db = db
        else:
            try:
                _embedding_db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)
            except Exception:
                pass

        def convert(emb) -> Dict[str, Any]:
            return {
                "name": emb.name,
                "step": emb.step,
                "vectors": emb.vectors,
                "shape": emb.shape,
                "sd_checkpoint": getattr(emb, 'sd_checkpoint', None),
                "sd_checkpoint_name": getattr(emb, 'sd_checkpoint_name', None),
            }

        loaded = {name: convert(emb) for name, emb in getattr(_embedding_db, 'word_embeddings', {}).items()}
        skipped = {name: convert(emb) for name, emb in getattr(_embedding_db, 'skipped_embeddings', {}).items()}
        return {"loaded": loaded, "skipped": skipped}

    @app.get('/api/loras')
    def list_loras() -> Dict[str, Any]:
        """Return LoRA files discovered under models/lora and --lora-dir.

        This is a simple directory scan (basenames) for quick UI listing.
        """
        roots: list[str] = []
        try:
            from modules import paths as _paths
            roots.append(os.path.join(_paths.models_path, 'lora'))
        except Exception:
            pass
        try:
            from modules import shared as _s
            if isinstance(_s.cmd_opts.lora_dir, str):
                roots.append(_s.cmd_opts.lora_dir)
        except Exception:
            pass
        exts = {'.safetensors', '.pt', '.ckpt'}
        seen = set()
        items: list[Dict[str, str]] = []
        for r in roots:
            if not isinstance(r, str) or not os.path.isdir(r):
                continue
            for dp, _dn, files in os.walk(r):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        name = os.path.splitext(fn)[0]
                        if name in seen:
                            continue
                        seen.add(name)
                        items.append({"name": name, "path": os.path.join(dp, fn)})
        items.sort(key=lambda x: x["name"].lower())
        return {"loras": items}

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

    @app.get('/api/options')
    def get_options() -> Dict[str, Any]:
        return {"values": _shared.opts.data}

    @app.post('/api/options')
    def set_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        updated: list[str] = []
        for key, value in payload.items():
            if key in _shared.opts.data_labels:
                # cast to expected type
                try:
                    casted = _shared.opts.cast_value(key, value)
                except Exception:
                    casted = value
                if _shared.opts.set(key, casted, is_api=True):
                    updated.append(key)

        # Persist updated keys into apps/settings_values.json
        try:
            if updated:
                existing = _load_json(_settings_values_path) if os.path.exists(_settings_values_path) else {}
                for k in updated:
                    existing[k] = _shared.opts.data.get(k)
                _save_json(_settings_values_path, existing)
        except Exception as e:  # pragma: no cover
            print(color_red(f"[settings] failed to persist: {e}"))
        return {"updated": updated}

    def prepare_txt2img(payload: Dict[str, Any]) -> Tuple[Txt2ImgRequest, str, Optional[str]]:
        prompt = _txt2img._require(payload, 'txt2img_prompt') or ''
        negative_prompt = _txt2img._require(payload, 'txt2img_neg_prompt') or ''
        prompt_styles = _txt2img._as_list(payload, 'txt2img_styles')
        n_iter = _txt2img._as_int(payload, 'txt2img_batch_count')
        batch_size = _txt2img._as_int(payload, 'txt2img_batch_size')
        cfg_scale = _txt2img._as_float(payload, 'txt2img_cfg_scale')
        distilled_cfg_scale = _txt2img._as_float_optional(payload, 'txt2img_distilled_cfg_scale', 3.5)
        height = _txt2img._as_int(payload, 'txt2img_height')
        width = _txt2img._as_int(payload, 'txt2img_width')
        enable_hr = _txt2img._as_bool(payload, 'txt2img_hr_enable')
        steps_val = _txt2img._as_int(payload, 'txt2img_steps')
        sampler_name = _txt2img._require(payload, 'txt2img_sampling')
        scheduler_name = _txt2img._require(payload, 'txt2img_scheduler')
        seed_val = _txt2img._as_int(payload, 'txt2img_seed')

        if enable_hr:
            denoising_strength = _txt2img._as_float(payload, 'txt2img_denoising_strength')
            hr_scale = _txt2img._as_float(payload, 'txt2img_hr_scale')
            hr_upscaler = _txt2img._require(payload, 'txt2img_hr_upscaler')
            hr_second_pass_steps = _txt2img._as_int(payload, 'txt2img_hires_steps')
            hr_resize_x = _txt2img._as_int(payload, 'txt2img_hr_resize_x')
            hr_resize_y = _txt2img._as_int(payload, 'txt2img_hr_resize_y')
            hr_checkpoint_name = payload.get('hr_checkpoint_name') or _txt2img._require(payload, 'hr_checkpoint')
            hr_additional_modules = payload.get('hr_additional_modules') or _txt2img._as_list(payload, 'hr_vae_te')
            hr_sampler_name = payload.get('hr_sampler_name') or _txt2img._require(payload, 'hr_sampler')
            hr_scheduler = payload.get('hr_scheduler') or _txt2img._require(payload, 'hr_scheduler')
            hr_prompt = payload.get('txt2img_hr_prompt') or ''
            hr_negative_prompt = payload.get('txt2img_hr_neg_prompt') or ''
            hr_cfg = _txt2img._as_float(payload, 'txt2img_hr_cfg')
            hr_distilled_cfg = _txt2img._as_float(payload, 'txt2img_hr_distilled_cfg')
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
            metadata={
                "mode": getattr(_shared.opts, 'codex_mode', 'Normal'),
                "styles": prompt_styles,
                "distilled_cfg_scale": distilled_cfg_scale,
                "hr": bool(enable_hr),
                "n_iter": n_iter,
            },
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
            extras={},
        )

        engine_key = getattr(_shared.opts, 'codex_engine', 'sd15')
        model_ref = getattr(_shared.opts, 'sd_model_checkpoint', None)
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
                with _call_queue.queue_lock:
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
                    from apps.server.backend.runtime.exception_hook import dump_exception as _dump_exc
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
        init_image_data = _img2img._require(payload, 'img2img_init_image')
        init_image = media.decode_image(init_image_data)
        mask_data = payload.get('img2img_mask')
        mask_image = media.decode_image(mask_data) if mask_data else None

        prompt = _img2img._require(payload, 'img2img_prompt') or ''
        negative_prompt = _img2img._require(payload, 'img2img_neg_prompt') or ''
        styles = _img2img._as_list(payload, 'img2img_styles') if 'img2img_styles' in payload else []
        batch_count = _img2img._as_int(payload, 'img2img_batch_count') if 'img2img_batch_count' in payload else 1
        batch_size = _img2img._as_int(payload, 'img2img_batch_size') if 'img2img_batch_size' in payload else 1
        steps_val = _img2img._as_int(payload, 'img2img_steps')
        cfg_scale = _img2img._as_float(payload, 'img2img_cfg_scale')
        distilled_cfg_scale = _img2img._as_float(payload, 'img2img_distilled_cfg_scale') if 'img2img_distilled_cfg_scale' in payload else None
        image_cfg_scale = _img2img._as_float(payload, 'img2img_image_cfg_scale') if 'img2img_image_cfg_scale' in payload else None
        denoise = _img2img._as_float(payload, 'img2img_denoising_strength')
        width_val = _img2img._as_int(payload, 'img2img_width')
        height_val = _img2img._as_int(payload, 'img2img_height')
        sampler_name = _img2img._require(payload, 'img2img_sampling')
        scheduler_name = _img2img._require(payload, 'img2img_scheduler')
        seed_val = _img2img._as_int(payload, 'img2img_seed')

        req = Img2ImgRequest(
            task=TaskType.IMG2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            guidance_scale=cfg_scale,
            batch_size=batch_size,
            metadata={
                "styles": styles,
                "distilled_cfg_scale": distilled_cfg_scale,
                "image_cfg_scale": image_cfg_scale,
                "batch_count": batch_count,
            },
            init_image=init_image,
            mask=mask_image,
            denoise_strength=denoise,
            width=width_val,
            height=height_val,
            steps=steps_val,
        )

        engine_key = getattr(_shared.opts, 'codex_engine', 'sd15')
        model_ref = getattr(_shared.opts, 'sd_model_checkpoint', None)
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
                with _call_queue.queue_lock:
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
                    from apps.server.backend.runtime.exception_hook import dump_exception as _dump_exc
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
                "mode": getattr(_shared.opts, 'codex_mode', 'Normal'),
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
        model_ref = getattr(_shared.opts, 'sd_model_checkpoint', None)
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
                "mode": getattr(_shared.opts, 'codex_mode', 'Normal'),
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
        model_ref = getattr(_shared.opts, 'sd_model_checkpoint', None)
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
            try:
                push({"type": "status", "stage": "running"})
                with _call_queue.queue_lock:
                    orch = InferenceOrchestrator()
                    engine_opts = {"export_video": bool(getattr(_shared.opts, 'codex_export_video', False))}
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
            except Exception as err:
                try:
                    from apps.server.backend.runtime.exception_hook import dump_exception as _dump_exc
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
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        if payload.get('__strict_version') != 1:
            raise HTTPException(status_code=400, detail="Missing __strict_version == 1")

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2vid-{uuid4().hex})"
        register_task(task_id, entry)
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

    _script_callbacks.app_started_callback(None, app)
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
