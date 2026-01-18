"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: UI persistence and metadata API routes.
Handles tabs/workflows JSON persistence, UI blocks filtering, and presets application.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for UI endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.json_store import _load_json


def build_router(
    *,
    codex_root: Path,
    opts_load_native: Callable[[], Dict[str, Any]],
    opts_set_many: Callable[[Dict[str, Any]], Dict[str, Any]],
    model_api: Any,
) -> APIRouter:
    router = APIRouter()

    _ui_blocks_cache: Optional[Dict[str, Any]] = None
    _ui_blocks_mtime: Optional[float] = None
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
        # Simple mtime-based cache
        try:
            stat = os.stat(blocks_path)
            mtime = stat.st_mtime
        except Exception:
            raise HTTPException(status_code=500, detail="ui blocks not found")
        if _ui_blocks_cache is not None and _ui_blocks_mtime == mtime:
            return _ui_blocks_cache
        data = _load_json(blocks_path)
        if not data or "blocks" not in data:
            raise HTTPException(status_code=500, detail="invalid ui blocks json")
        # Optional overrides in apps/interface/blocks.d/*.json (merged by id)
        overrides_root = str(codex_root / "apps" / "interface" / "blocks.d")
        merged = {b.get("id"): b for b in (data.get("blocks") or []) if isinstance(b, dict)}
        try:
            if os.path.isdir(overrides_root):
                for fn in os.listdir(overrides_root):
                    if not fn.endswith(".json"):
                        continue
                    ov = _load_json(os.path.join(overrides_root, fn))
                    if isinstance(ov, dict) and "blocks" in ov and isinstance(ov["blocks"], list):
                        for blk in ov["blocks"]:
                            if isinstance(blk, dict) and blk.get("id"):
                                merged[blk["id"]] = blk
        except Exception:
            pass
        out = {"version": int(data.get("version", 1)), "blocks": list(merged.values())}
        _ui_blocks_cache, _ui_blocks_mtime = out, mtime
        return out

    def _detect_semantic_engine() -> str:
        """Infer a semantic engine tag from the currently selected checkpoint."""
        try:
            current = str((opts_load_native() or {}).get("sd_model_checkpoint") or "")
            infos = model_api.list_checkpoints_as_dict(refresh=False)
            target = None
            for i in infos:
                if i.get("name") == current or i.get("title") == current:
                    target = i
                    break
            blob = ""
            if target:
                blob = " ".join(
                    [str(target.get("title") or ""), str(target.get("path") or ""), str(target.get("format") or "")]
                ).lower()
                comps = target.get("components") or []
                if any("flux" in blob for blob in [blob]) or "transformer" in comps:
                    if "flux" in blob:
                        return "flux1"
                if "text_encoder_2" in comps and "tokenizer_2" in comps:
                    return "sdxl"
                if "hunyuan" in blob:
                    return "hunyuan_video"
                if "svd" in blob:
                    return "svd"
                if "wan" in blob:
                    return "wan22"
            # Fallback on title string hints
            t = (current or "").lower()
            if "flux" in t:
                return "flux1"
            if "xl" in t:
                return "sdxl"
            if "wan" in t:
                return "wan22"
        except Exception:
            pass
        return "sd15"

    @router.get("/api/ui/blocks")
    def ui_blocks(tab: Optional[str] = None) -> Dict[str, Any]:
        """Return UI blocks filtered by tab and current semantic engine."""
        data = _load_ui_blocks()
        sem = _detect_semantic_engine()
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
    _ALLOWED_TAB_TYPES = {"sd15", "sdxl", "flux1", "chroma", "zimage", "wan"}

    def _normalize_tab_type(value: object) -> str:
        raw = str(value or "").strip().lower()
        if raw in ("wan22", "wan22_14b", "wan22_5b"):
            return "wan"
        if raw == "flux":
            return "flux1"
        if raw in ("flux1_chroma", "flux1-chroma"):
            return "chroma"
        if raw in _ALLOWED_TAB_TYPES:
            return raw
        return "sd15"

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
                mk("chroma", "Chroma", 3),
                mk("zimage", "Z Image", 4),
                mk("wan", "WAN 2.2", 5),
            ],
        }

    def _load_tabs() -> Dict[str, Any]:
        nonlocal _tabs_cache, _tabs_mtime
        _ensure_dirs()
        p = _tabs_path()
        if not os.path.exists(p):
            data = _default_tabs()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        stat = os.stat(p)
        if _tabs_cache is not None and _tabs_mtime == stat.st_mtime:
            return _tabs_cache
        data = _load_json(p)
        if not isinstance(data, dict) or "tabs" not in data:
            data = _default_tabs()

        # Normalize/migrate tab payloads so the API never returns legacy identifiers.
        changed = False
        tabs_in = data.get("tabs")
        if isinstance(tabs_in, list):
            for t in tabs_in:
                if not isinstance(t, dict):
                    continue
                old_type = t.get("type")
                new_type = _normalize_tab_type(old_type)
                if new_type != old_type:
                    t["type"] = new_type
                    changed = True
                if new_type == "flux1":
                    title = str(t.get("title") or "")
                    if title.strip().lower() == "flux":
                        t["title"] = "FLUX.1"
                        changed = True
                    params = t.get("params")
                    if isinstance(params, dict):
                        raw_labels = params.get("textEncoders")
                        if isinstance(raw_labels, list):
                            migrated: list[str] = []
                            for raw in raw_labels:
                                s = str(raw or "").strip()
                                if s.startswith("flux/"):
                                    s = "flux1/" + s[len("flux/") :]
                                    changed = True
                                if s:
                                    migrated.append(s)
                            params["textEncoders"] = migrated

        if changed:
            _save_tabs(data)
            return data
        _tabs_cache, _tabs_mtime = data, stat.st_mtime
        return data

    def _save_tabs(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _tabs_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        stat = os.stat(p)
        nonlocal _tabs_cache, _tabs_mtime
        _tabs_cache, _tabs_mtime = data, stat.st_mtime

    def _load_workflows() -> Dict[str, Any]:
        nonlocal _workflows_cache, _workflows_mtime
        _ensure_dirs()
        p = _workflows_path()
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "workflows": []}, f, indent=2)
        stat = os.stat(p)
        if _workflows_cache is not None and _workflows_mtime == stat.st_mtime:
            return _workflows_cache
        data = _load_json(p)
        if not isinstance(data, dict) or "workflows" not in data:
            data = {"version": 1, "workflows": []}
        _workflows_cache, _workflows_mtime = data, stat.st_mtime
        return data

    def _save_workflows(data: Dict[str, Any]) -> None:
        _ensure_dirs()
        p = _workflows_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
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
        raw_type = str(payload.get("type") or "sd15").strip().lower()
        if raw_type == "flux":
            raise HTTPException(status_code=400, detail="invalid tab type: flux (use flux1)")
        if raw_type not in _ALLOWED_TAB_TYPES and raw_type not in ("wan22", "wan22_14b", "wan22_5b"):
            raise HTTPException(status_code=400, detail=f"invalid tab type: {raw_type}")
        ttype = _normalize_tab_type(raw_type)
        title = str(payload.get("title") or ("FLUX.1" if ttype == "flux1" else ttype.upper()))
        params = payload.get("params") or {}
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
                    t["enabled"] = bool(payload["enabled"])
                if "params" in payload and isinstance(payload["params"], dict):
                    # shallow merge for now
                    t["params"] = payload["params"]
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
        wtype = str(payload.get("type") or "sd15")
        params_snapshot = payload.get("params_snapshot") or {}
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
            logging.getLogger("backend.api").warning("ui presets not found at %s; returning empty list", presets_path)
            return {"version": 1, "presets": []}
        if _ui_presets_cache is not None and _ui_presets_mtime == mtime:
            return _ui_presets_cache
        data = _load_json(presets_path)
        if not data or "presets" not in data:
            logging.getLogger("backend.api").warning("ui presets invalid at %s; returning empty list", presets_path)
            return {
                "version": int(data.get("version", 1)) if isinstance(data, dict) else 1,
                "presets": [],
            }
        out = {"version": int(data.get("version", 1)), "presets": list(data.get("presets") or [])}
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
            if not isinstance(p, dict)
            or not p.get("tabs")
            or tab_norm in [str(t).lower() for t in (p.get("tabs") or [])]
        ]
        return {"version": data.get("version", 1), "presets": presets}

    @router.post("/api/ui/presets/apply")
    def ui_presets_apply(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Apply a Model UI preset: resolve checkpoint and set options atomically."""
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

        # Apply options atomically
        try:
            updates = {"sd_model_checkpoint": str(target)}
            extra = preset.get("options") or {}
            if isinstance(extra, dict):
                updates.update(extra)
            opts_set_many(updates)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to apply preset: {exc}")

        return {"applied": True, "model": target}

    return router
