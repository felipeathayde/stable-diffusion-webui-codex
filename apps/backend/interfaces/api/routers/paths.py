"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Paths configuration API routes.
Exposes apps/paths.json for the UI and accepts updates for engine-specific keys.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for paths endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.json_store import _load_json, _save_json


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/paths")
    def get_paths() -> Dict[str, Any]:
        cfg_path = str(codex_root / "apps" / "paths.json")
        raw = _load_json(cfg_path) or {}
        if not isinstance(raw, dict):
            raw = {}

        paths: Dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                paths[key] = [str(item) for item in value if isinstance(item, str)]

        return {"paths": paths}

    @router.post("/api/paths")
    def set_paths(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict) or "paths" not in payload or not isinstance(payload["paths"], dict):
            raise HTTPException(status_code=400, detail='payload must be {"paths": {...}}')

        cfg_path = str(codex_root / "apps" / "paths.json")
        current = _load_json(cfg_path) or {}
        if not isinstance(current, dict):
            current = {}

        incoming = payload["paths"] or {}
        new_paths: Dict[str, Any] = dict(current)

        for key, value in incoming.items():
            if value is None:
                new_paths[key] = []
            elif isinstance(value, list):
                new_paths[key] = [str(item) for item in value if isinstance(item, str)]

        _save_json(cfg_path, new_paths)
        return {"ok": True}

    return router
