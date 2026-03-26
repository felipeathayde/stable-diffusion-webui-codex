"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: UI persistence and metadata API routes.
    Handles tabs/workflows JSON persistence, UI blocks filtering, and presets application, with fail-loud tab-type validation for `/api/ui/tabs`.
    Normalizes WAN tab aliases (`wan22`, `wan22_5b`, `wan22_14b`, `wan22_14b_animate`) into the canonical UI `wan` tab type and accepts
    a dedicated `ltx2` video workspace tab type.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for UI endpoints.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.json_store import _load_json, _save_json


def build_router(
    *,
    codex_root: Path,
    opts_set_many: Callable[[Dict[str, Any]], list[str]],
    model_api: Any,
) -> APIRouter:
    router = APIRouter()

    _ui_blocks_cache: Optional[Dict[str, Any]] = None
    _ui_blocks_mtime: Optional[str] = None
    _ui_presets_cache: Optional[Dict[str, Any]] = None
    _ui_presets_mtime: Optional[float] = None
    _tabs_cache: Optional[Dict[str, Any]] = None
    _tabs_mtime: Optional[float] = None
    _workflows_cache: Optional[Dict[str, Any]] = None
    _workflows_mtime: Optional[float] = None

    # ------------------------------------------------------------------
    # UI Blocks (server-driven parameter panels)
    def _load_ui_blocks() -> Dict[str, Any]:
        nonlocal _ui_blocks_cache, _ui_blocks_mtime
        blocks_path = str(codex_root / "apps" / "interface" / "blocks.json")
        overrides_root = str(codex_root / "apps" / "interface" / "blocks.d")

        def _blocks_cache_key() -> str:
            parts: list[str] = []
            try:
                parts.append(f"base:{os.stat(blocks_path).st_mtime_ns}")
            except Exception:
                raise HTTPException(status_code=500, detail="ui blocks not found")
            if os.path.isdir(overrides_root):
                for fn in sorted(os.listdir(overrides_root)):
                    if not fn.endswith(".json"):
                        continue
                    override_path = os.path.join(overrides_root, fn)
                    try:
                        parts.append(f"{fn}:{os.stat(override_path).st_mtime_ns}")
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"failed to stat ui blocks override '{override_path}': {exc}",
                        ) from exc
            return "|".join(parts)

        # Simple mtime-based cache
        cache_key = _blocks_cache_key()
        if _ui_blocks_cache is not None and _ui_blocks_mtime == cache_key:
            return _ui_blocks_cache
        try:
            data = _load_json(blocks_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load ui blocks json: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="invalid ui blocks json: expected object")
        if "blocks" not in data:
            raise HTTPException(status_code=500, detail="invalid ui blocks json")
        # Optional overrides in apps/interface/blocks.d/*.json (merged by id)
        merged = {b.get("id"): b for b in (data.get("blocks") or []) if isinstance(b, dict)}
        if os.path.isdir(overrides_root):
            for fn in sorted(os.listdir(overrides_root)):
                if not fn.endswith(".json"):
                    continue
                override_path = os.path.join(overrides_root, fn)
                try:
                    ov = _load_json(override_path)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"failed to load ui blocks override '{override_path}': {exc}") from exc
                if not isinstance(ov, dict):
                    raise HTTPException(status_code=500, detail=f"invalid ui blocks override json: {override_path}")
                blocks = ov.get("blocks")
                if blocks is None:
                    continue
                if not isinstance(blocks, list):
                    raise HTTPException(
                        status_code=500,
                        detail=f"ui blocks override '{override_path}' must define 'blocks' as array",
                    )
                for blk in blocks:
                    if isinstance(blk, dict) and blk.get("id"):
                        merged[blk["id"]] = blk
        out = {"version": int(data.get("version", 1)), "blocks": list(merged.values())}
        _ui_blocks_cache, _ui_blocks_mtime = out, cache_key
        return out

    @router.get("/api/ui/blocks")
    def ui_blocks(tab: Optional[str] = None, engine: Optional[str] = None) -> Dict[str, Any]:
        """Return UI blocks filtered by tab and semantic engine.

        The backend no longer infers a “current engine” from a global checkpoint selection.
        Callers should pass `engine=<semantic_engine>` when engine-scoped blocks are needed.
        """
        data = _load_ui_blocks()
        sem = str(engine or "").strip().lower() or "sd15"
        blocks_in = list(data.get("blocks") or [])
        out: list[dict] = []
        tab_norm = str(tab).strip().lower() if tab else None
        for b in blocks_in:
            if not isinstance(b, dict):
                continue
            when = b.get("when") or {}
            ok_tab = True
            ok_eng = True
            if tab_norm and isinstance(when, dict) and when.get("tabs"):
                ok_tab = tab_norm in [str(t).lower() for t in when.get("tabs")]
            if isinstance(when, dict) and when.get("engines"):
                ok_eng = sem in [str(e).lower() for e in when.get("engines")]
            if ok_tab and ok_eng:
                out.append(b)
        return {"version": data.get("version", 1), "blocks": out, "semantic_engine": sem}

    # ------------------------------------------------------------------
    # Tabs & Workflows Persistence (JSON files)
    _ALLOWED_TAB_TYPES = {"sd15", "sdxl", "flux1", "flux2", "chroma", "zimage", "wan", "anima", "ltx2"}
    _IMAGE_PARAM_TOP_LEVEL_KEYS = {
        "schemaVersion",
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "sampler",
        "scheduler",
        "steps",
        "cfgScale",
        "seed",
        "clipSkip",
        "batchSize",
        "batchCount",
        "img2imgResizeMode",
        "img2imgUpscaler",
        "guidanceAdvanced",
        "hires",
        "highres",  # legacy alias persisted by older clients
        "swapModel",
        "refiner",
        "checkpoint",
        "textEncoders",
        "useInitImage",
        "initImageData",
        "initImageName",
        "denoiseStrength",
        "useMask",
        "maskImageData",
        "maskImageName",
        "maskEnforcement",
        "perStepBlendStrength",
        "perStepBlendSteps",
        "inpaintFullResPadding",
        "inpaintingFill",
        "maskInvert",
        "maskBlur",
        "maskRound",
        "maskRegionSplit",
        "zimageTurbo",
    }
    _WAN_PARAM_TOP_LEVEL_KEYS = {
        "schemaVersion",
        "high",
        "low",
        "video",
        "assets",
        "lightx2v",
        "lowFollowsHigh",
    }
    _LTX_PARAM_TOP_LEVEL_KEYS = {
        "schemaVersion",
        "mode",
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "fps",
        "frames",
        "steps",
        "cfgScale",
        "sampler",
        "scheduler",
        "seed",
        "checkpoint",
        "vae",
        "textEncoder",
        "useInitImage",
        "initImageData",
        "initImageName",
        "videoReturnFrames",
    }
    _NESTED_OBJECT_PARAM_KEYS = {
        "guidanceAdvanced",
        "hires",
        "highres",
        "swapModel",
        "refiner",
        "high",
        "low",
        "video",
        "assets",
    }

    def _normalize_tab_type(value: object) -> str:
        raw = str(value or "").strip().lower()
        if raw in ("wan22", "wan22_14b", "wan22_5b", "wan22_14b_animate"):
            return "wan"
        if raw == "flux":
            return "flux1"
        if raw in ("flux1_chroma", "flux1-chroma"):
            return "chroma"
        if raw in _ALLOWED_TAB_TYPES:
            return raw
        raise ValueError(f"invalid tab type: {raw or '<empty>'}")

    def _parse_bool_payload(value: object, *, field: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value in (0, 1):
                return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid '{field}': expected bool or one of "
                f"('true','false','1','0','yes','no','on','off')."
            ),
        )

    def _assert_plain_object(value: object, *, field: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"{field} must be an object")
        for key in value.keys():
            if not isinstance(key, str):
                raise HTTPException(status_code=400, detail=f"{field} contains a non-string key")
        return value

    def _allowed_param_keys(tab_type: str) -> set[str]:
        if tab_type == "wan":
            return _WAN_PARAM_TOP_LEVEL_KEYS
        if tab_type == "ltx2":
            return _LTX_PARAM_TOP_LEVEL_KEYS
        return _IMAGE_PARAM_TOP_LEVEL_KEYS

    def _sanitize_tab_params_patch(
        *,
        tab_type: str,
        raw_params: object,
        field: str = "params",
        reject_unknown: bool = True,
    ) -> Dict[str, Any]:
        params_patch = _assert_plain_object(raw_params, field=field)
        allowed = _allowed_param_keys(tab_type)
        unknown_keys = sorted(key for key in params_patch.keys() if key not in allowed)
        if unknown_keys and reject_unknown:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unknown_tab_params_keys",
                    "message": f"Unexpected {field} key(s): {', '.join(unknown_keys)}",
                    "field": field,
                    "tab_type": tab_type,
                    "unknown_keys": unknown_keys,
                },
            )
        sanitized: Dict[str, Any] = (
            dict(params_patch)
            if not unknown_keys
            else {key: value for key, value in params_patch.items() if key in allowed}
        )
        for key in _NESTED_OBJECT_PARAM_KEYS:
            if key in sanitized and sanitized[key] is not None and not isinstance(sanitized[key], dict):
                raise HTTPException(status_code=400, detail=f"{field}.{key} must be an object")
        if "textEncoders" in sanitized and sanitized["textEncoders"] is not None and not isinstance(
            sanitized["textEncoders"], list
        ):
            raise HTTPException(status_code=400, detail=f"{field}.textEncoders must be an array")
        if tab_type != "wan" and "highres" in sanitized:
            if "hires" not in sanitized and isinstance(sanitized["highres"], dict):
                sanitized["hires"] = sanitized["highres"]
            del sanitized["highres"]
        return sanitized

    def _sanitize_stored_tab_params(*, tab_type: str, raw_params: object, tab_id: str) -> Dict[str, Any]:
        if raw_params is None:
            return {}
        try:
            return _sanitize_tab_params_patch(
                tab_type=tab_type,
                raw_params=raw_params,
                field="params",
                reject_unknown=False,
            )
        except HTTPException as exc:
            raise HTTPException(
                status_code=500,
                detail=f"tabs.json invalid: tab '{tab_id}' {exc.detail}",
            ) from exc

    def _deep_merge_params(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base)
        for key, value in patch.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = _deep_merge_params(current, value)
            else:
                merged[key] = value
        return merged

    def _tabs_path() -> str:
        return str(codex_root / "apps" / "interface" / "tabs.json")

    def _workflows_path() -> str:
        return str(codex_root / "apps" / "interface" / "workflows.json")

    def _ensure_dirs() -> None:
        root = str(codex_root / "apps" / "interface")
        os.makedirs(root, exist_ok=True)

    def _default_tabs() -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()

        def mk(t: str, title: str, order: int) -> Dict[str, Any]:
            return {
                "id": f"tab-{t}-{order}",
                "type": t,
                "title": title,
                "order": order,
                "enabled": True,
                "params": {},
                "meta": {"createdAt": now, "updatedAt": now},
            }

        return {
            "version": 1,
            "tabs": [
                mk("sd15", "SD 1.5", 0),
                mk("sdxl", "SDXL", 1),
                mk("flux1", "FLUX.1", 2),
                mk("flux2", "FLUX.2", 3),
                mk("chroma", "Chroma", 4),
                mk("zimage", "Z Image", 5),
                mk("wan", "WAN 2.2", 6),
            ],
        }

    def _load_tabs() -> Dict[str, Any]:
        nonlocal _tabs_cache, _tabs_mtime
        _ensure_dirs()
        p = _tabs_path()
        if not os.path.exists(p):
            data = _default_tabs()
            try:
                _save_json(p, data)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to initialize tabs.json: {exc}") from exc
        stat = os.stat(p)
        if _tabs_cache is not None and _tabs_mtime == stat.st_mtime:
            return _tabs_cache
        try:
            data = _load_json(p)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load tabs.json: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="tabs.json invalid: expected object")
        tabs_in = data.get("tabs")
        if not isinstance(tabs_in, list):
            raise HTTPException(status_code=500, detail="tabs.json invalid: missing 'tabs' array")

        # Normalize/migrate tab payloads so the API never returns legacy identifiers.
        changed = False
        for index, t in enumerate(tabs_in):
            if not isinstance(t, dict):
                raise HTTPException(status_code=500, detail=f"tabs.json invalid: entry #{index + 1} must be object")
            old_type = t.get("type")
            try:
                new_type = _normalize_tab_type(old_type)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=f"tabs.json contains {exc}") from exc
            if new_type != old_type:
                t["type"] = new_type
                changed = True
            tab_id = str(t.get("id") or f"#{index + 1}")
            params = t.get("params")
            sanitized_params = _sanitize_stored_tab_params(tab_type=new_type, raw_params=params, tab_id=tab_id)
            if sanitized_params != params:
                t["params"] = sanitized_params
                changed = True
            if new_type == "flux1":
                title = str(t.get("title") or "")
                if title.strip().lower() == "flux":
                    t["title"] = "FLUX.1"
                    changed = True
                params_for_flux = t.get("params")
                if isinstance(params_for_flux, dict):
                    raw_labels = params_for_flux.get("textEncoders")
                    if isinstance(raw_labels, list):
                        migrated: list[str] = []
                        for raw in raw_labels:
                            s = str(raw or "").strip()
                            if s.startswith("flux/"):
                                s = "flux1/" + s[len("flux/") :]
                                changed = True
                            if s:
                                migrated.append(s)
                        params_for_flux["textEncoders"] = migrated

        if changed:
            _save_tabs(data)
            return data
        _tabs_cache, _tabs_mtime = data, stat.st_mtime
        return data

    def _save_tabs(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _tabs_path()
        try:
            _save_json(p, data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to save tabs.json: {exc}") from exc
        stat = os.stat(p)
        nonlocal _tabs_cache, _tabs_mtime
        _tabs_cache, _tabs_mtime = data, stat.st_mtime

    def _load_workflows() -> Dict[str, Any]:
        nonlocal _workflows_cache, _workflows_mtime
        _ensure_dirs()
        p = _workflows_path()
        if not os.path.exists(p):
            try:
                _save_json(p, {"version": 1, "workflows": []})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to initialize workflows.json: {exc}") from exc
        stat = os.stat(p)
        if _workflows_cache is not None and _workflows_mtime == stat.st_mtime:
            return _workflows_cache
        try:
            data = _load_json(p)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load workflows.json: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="workflows.json invalid: expected object")
        workflows = data.get("workflows")
        if not isinstance(workflows, list):
            raise HTTPException(status_code=500, detail="workflows.json invalid: missing 'workflows' array")
        for index, workflow in enumerate(workflows):
            if not isinstance(workflow, dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"workflows.json invalid: entry #{index + 1} must be object",
                )
        _workflows_cache, _workflows_mtime = data, stat.st_mtime
        return data

    def _save_workflows(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _workflows_path()
        try:
            _save_json(p, data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to save workflows.json: {exc}") from exc
        stat = os.stat(p)
        nonlocal _workflows_cache, _workflows_mtime
        _workflows_cache, _workflows_mtime = data, stat.st_mtime

    @router.get("/api/ui/tabs")
    def api_get_tabs() -> Dict[str, Any]:
        return _load_tabs()

    @router.post("/api/ui/tabs")
    def api_create_tab(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_tabs()
        tabs = list(data.get("tabs") or [])
        raw_type = str(payload.get("type") or "").strip().lower()
        if not raw_type:
            raise HTTPException(status_code=400, detail="tab type is required")
        if raw_type == "flux":
            raise HTTPException(status_code=400, detail="invalid tab type: flux (use flux1)")
        if raw_type not in _ALLOWED_TAB_TYPES and raw_type not in ("wan22", "wan22_14b", "wan22_5b", "wan22_14b_animate"):
            raise HTTPException(status_code=400, detail=f"invalid tab type: {raw_type}")
        ttype = _normalize_tab_type(raw_type)
        title = str(
            payload.get("title")
            or ("FLUX.1" if ttype == "flux1" else "FLUX.2" if ttype == "flux2" else ttype.upper())
        )
        if "params" in payload:
            params = _sanitize_tab_params_patch(tab_type=ttype, raw_params=payload.get("params"), field="params")
        else:
            params = {}
        new_id = str(payload.get("id") or "").strip() or f"tab-{int(time.time()*1000)}"
        if any(str(t.get("id")) == new_id for t in tabs):
            raise HTTPException(status_code=409, detail="tab id already exists")
        order = max([int(t.get("order", 0)) for t in tabs], default=-1) + 1
        now = datetime.utcnow().isoformat()
        tab = {
            "id": new_id,
            "type": ttype,
            "title": title,
            "order": order,
            "enabled": True,
            "params": params,
            "meta": {"createdAt": now, "updatedAt": now},
        }
        tabs.append(tab)
        out = {"version": int(data.get("version", 1)), "tabs": tabs}
        _save_tabs(out)
        return {"id": new_id}

    @router.patch("/api/ui/tabs/{tab_id}")
    def api_update_tab(tab_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_tabs()
        updated = False
        now = datetime.utcnow().isoformat()
        for t in data["tabs"]:
            if str(t.get("id")) == tab_id:
                if "title" in payload:
                    t["title"] = str(payload["title"])
                if "enabled" in payload:
                    t["enabled"] = _parse_bool_payload(payload["enabled"], field="enabled")
                if "params" in payload:
                    try:
                        tab_type = _normalize_tab_type(t.get("type"))
                    except ValueError as exc:
                        raise HTTPException(status_code=500, detail=f"tabs.json contains {exc}") from exc
                    params_patch = _sanitize_tab_params_patch(tab_type=tab_type, raw_params=payload.get("params"), field="params")
                    sanitized_current_params = _sanitize_stored_tab_params(
                        tab_type=tab_type,
                        raw_params=t.get("params"),
                        tab_id=tab_id,
                    )
                    t["params"] = _deep_merge_params(sanitized_current_params, params_patch)
                t["meta"] = t.get("meta") or {}
                t["meta"]["updatedAt"] = now
                updated = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail="tab not found")
        _save_tabs(data)
        return {"updated": tab_id}

    @router.post("/api/ui/tabs/reorder")
    def api_reorder_tabs(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ids = list(payload.get("ids") or [])
        data = _load_tabs()
        idx = {tid: i for i, tid in enumerate(ids)}
        for t in data["tabs"]:
            tid = str(t.get("id"))
            if tid in idx:
                t["order"] = idx[tid]
        data["tabs"].sort(key=lambda x: int(x.get("order", 0)))
        _save_tabs(data)
        return {"ok": True}

    @router.delete("/api/ui/tabs/{tab_id}")
    def api_delete_tab(tab_id: str) -> Dict[str, Any]:
        data = _load_tabs()
        tabs = [t for t in data["tabs"] if str(t.get("id")) != tab_id]
        if len(tabs) == len(data["tabs"]):
            raise HTTPException(status_code=404, detail="tab not found")
        # normalize order
        for i, t in enumerate(tabs):
            t["order"] = i
        out = {"version": int(data.get("version", 1)), "tabs": tabs}
        _save_tabs(out)
        return {"deleted": tab_id}

    @router.get("/api/ui/workflows")
    def api_get_workflows() -> Dict[str, Any]:
        return _load_workflows()

    @router.post("/api/ui/workflows")
    def api_create_workflow(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_workflows()
        wfs = list(data.get("workflows") or [])
        wf_id = f"wf-{int(time.time()*1000)}"
        name = str(payload.get("name") or wf_id)
        source_tab_id = str(payload.get("source_tab_id") or "")
        raw_type = str(payload.get("type") or "").strip().lower()
        if not raw_type:
            raise HTTPException(status_code=400, detail="workflow type is required")
        if raw_type == "flux":
            raise HTTPException(status_code=400, detail="invalid workflow type: flux (use flux1)")
        if raw_type not in _ALLOWED_TAB_TYPES and raw_type not in ("wan22", "wan22_14b", "wan22_5b", "wan22_14b_animate"):
            raise HTTPException(status_code=400, detail=f"invalid workflow type: {raw_type}")
        wtype = _normalize_tab_type(raw_type)
        params_snapshot = payload.get("params_snapshot")
        if params_snapshot is None:
            params_snapshot = {}
        if not isinstance(params_snapshot, dict):
            raise HTTPException(status_code=400, detail="params_snapshot must be an object")
        now = datetime.utcnow().isoformat()
        wf = {
            "id": wf_id,
            "name": name,
            "source_tab_id": source_tab_id,
            "type": wtype,
            "created_at": now,
            "engine_semantics": payload.get("engine_semantics") or wtype,
            "params_snapshot": params_snapshot,
        }
        wfs.insert(0, wf)
        out = {"version": int(data.get("version", 1)), "workflows": wfs}
        _save_workflows(out)
        return {"id": wf_id}

    @router.patch("/api/ui/workflows/{wf_id}")
    def api_update_workflow(wf_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        data = _load_workflows()
        updated = False
        for w in data["workflows"]:
            if str(w.get("id")) == wf_id:
                if "name" in payload:
                    w["name"] = str(payload["name"])
                if "params_snapshot" in payload and isinstance(payload["params_snapshot"], dict):
                    w["params_snapshot"] = payload["params_snapshot"]
                updated = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail="workflow not found")
        _save_workflows(data)
        return {"updated": wf_id}

    @router.delete("/api/ui/workflows/{wf_id}")
    def api_delete_workflow(wf_id: str) -> Dict[str, Any]:
        data = _load_workflows()
        wfs = [w for w in data["workflows"] if str(w.get("id")) != wf_id]
        if len(wfs) == len(data["workflows"]):
            raise HTTPException(status_code=404, detail="workflow not found")
        out = {"version": int(data.get("version", 1)), "workflows": wfs}
        _save_workflows(out)
        return {"deleted": wf_id}

    # ------------------------------------------------------------------
    # UI Presets (Model UI)
    def _load_ui_presets() -> Dict[str, Any]:
        nonlocal _ui_presets_cache, _ui_presets_mtime
        presets_path = str(codex_root / "apps" / "interface" / "presets.json")
        try:
            stat = os.stat(presets_path)
            mtime = stat.st_mtime
        except Exception:
            raise HTTPException(status_code=500, detail="presets.json not found")
        if _ui_presets_cache is not None and _ui_presets_mtime == mtime:
            return _ui_presets_cache
        try:
            data = _load_json(presets_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load presets.json: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="presets.json invalid: expected object")
        presets = data.get("presets")
        if not isinstance(presets, list):
            raise HTTPException(status_code=500, detail="presets.json invalid: missing 'presets' array")
        for index, preset in enumerate(presets):
            if not isinstance(preset, dict):
                raise HTTPException(status_code=500, detail=f"presets.json invalid: entry #{index + 1} must be object")
        out = {"version": int(data.get("version", 1)), "presets": presets}
        _ui_presets_cache, _ui_presets_mtime = out, mtime
        return out

    @router.get("/api/ui/presets")
    def ui_presets(tab: Optional[str] = None) -> Dict[str, Any]:
        """Return Model UI presets, optionally filtered by tab."""
        data = _load_ui_presets()
        if not tab:
            return data
        tab_norm = str(tab).strip().lower()
        presets = [
            p
            for p in (data.get("presets") or [])
            if not p.get("tabs")
            or tab_norm in [str(t).lower() for t in (p.get("tabs") or [])]
        ]
        return {"version": data.get("version", 1), "presets": presets}

    @router.post("/api/ui/presets/apply")
    def ui_presets_apply(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Apply a UI preset: resolve checkpoint selector and apply options.

        Presets no longer mutate a global “current checkpoint” option; callers must
        apply the returned checkpoint to their per-tab state.
        """
        try:
            preset_id = str(payload.get("id"))
            tab = str(payload.get("tab")) if payload.get("tab") else None
        except Exception:
            raise HTTPException(status_code=400, detail="invalid payload")
        if not preset_id:
            raise HTTPException(status_code=400, detail="id is required")
        data = _load_ui_presets()
        candidates = [p for p in (data.get("presets") or []) if isinstance(p, dict) and p.get("id") == preset_id]
        if tab:
            tab_norm = str(tab).strip().lower()
            candidates = [p for p in candidates if not p.get("tabs") or tab_norm in [str(t).lower() for t in p.get("tabs")]]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"preset not found: {preset_id}")
        preset = candidates[0]

        # Resolve checkpoint by selector
        selector = preset.get("model_select") or {}
        sel_type = str(selector.get("type", "exact")).lower()
        sel_value = str(selector.get("value", ""))
        if not sel_value:
            raise HTTPException(status_code=409, detail="preset has no model selector")

        infos = model_api.list_checkpoints_as_dict(refresh=False)
        titles = [str(i.get("title") or i.get("name") or "") for i in infos]
        target: Optional[str] = None
        if sel_type == "exact":
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
            raise HTTPException(status_code=409, detail=f"checkpoint not found for selector: {sel_type}:{sel_value}")

        # Apply registry-backed options atomically (checkpoint selection is returned, not persisted).
        try:
            extra = preset.get("options") or {}
            updates = dict(extra) if isinstance(extra, dict) else {}
            updated = opts_set_many(updates) if updates else []
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to apply preset: {exc}")

        return {"applied": True, "checkpoint": target, "model": target, "updated": updated}

    return router
