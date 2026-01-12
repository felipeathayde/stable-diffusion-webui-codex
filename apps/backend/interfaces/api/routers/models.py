"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model and asset inventory API routes.
Exposes checkpoints, inventories, samplers/schedulers, embeddings, and engine capabilities.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for model/inventory endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from apps.backend.runtime.sampling.catalog import SAMPLER_OPTIONS, SCHEDULER_OPTIONS
from apps.backend.interfaces.api.path_utils import _normalize_inventory_for_api
from apps.backend.interfaces.api.serializers import _serialize_checkpoint
from apps.backend.interfaces.api.file_metadata import read_file_metadata


def build_router(
    *,
    codex_root: Path,
    opts_load_native: Callable[[], Dict[str, Any]],
    opts_get: Callable[[str, Any], Any],
    model_api: Any,
) -> APIRouter:
    router = APIRouter()
    log = logging.getLogger("backend.api")

    @router.get("/api/models")
    def list_models(refresh: bool = Query(False, description="If true, re-scan checkpoint roots before returning.")) -> Dict[str, Any]:
        entries = model_api.list_checkpoints(refresh=bool(refresh))
        models = [_serialize_checkpoint(entry) for entry in entries]
        models_info = [e.as_dict() for e in entries]
        try:
            current = (opts_load_native() or {}).get("sd_model_checkpoint") or (models[0]["name"] if models else None)
        except Exception:
            current = models[0]["name"] if models else None
        return {"models": models, "current": current, "models_info": models_info}

    @router.get("/api/models/inventory")
    def list_models_inventory(refresh: bool = Query(False, description="If true, re-scan the models/ and huggingface/ folders.")) -> Dict[str, Any]:
        from apps.backend.inventory import cache as _inv_cache
        if refresh:
            try:
                inv = _inv_cache.refresh()
                logging.getLogger("inventory").info(
                    "inventory: refreshed (vaes=%d, text_encoders=%d, loras=%d, wan22.gguf=%d, metadata=%d)",
                    len(inv.get("vaes", [])),
                    len(inv.get("text_encoders", [])),
                    len(inv.get("loras", [])),
                    len(inv.get("wan22", [])),
                    len(inv.get("metadata", [])),
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"inventory refresh failed: {exc}")
        else:
            inv = _inv_cache.get()
        return {
            "vaes": _normalize_inventory_for_api(inv.get("vaes", [])),
            "text_encoders": _normalize_inventory_for_api(inv.get("text_encoders", [])),
            "loras": _normalize_inventory_for_api(inv.get("loras", [])),
            "wan22": {"gguf": _normalize_inventory_for_api(inv.get("wan22", []))},
            "metadata": _normalize_inventory_for_api(inv.get("metadata", [])),
        }

    @router.get("/api/models/file-metadata")
    def get_file_metadata(path: str = Query(..., description="Repo-relative or absolute path to a weights file.")) -> Dict[str, Any]:
        try:
            result = read_file_metadata(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return result.as_dict()

    @router.post("/api/models/inventory/refresh")
    def refresh_models_inventory() -> Dict[str, Any]:
        from apps.backend.inventory import cache as _inv_cache
        try:
            inv = _inv_cache.refresh()
            logging.getLogger("inventory").info(
                "inventory: refreshed (vaes=%d, text_encoders=%d, loras=%d, wan22.gguf=%d, metadata=%d)",
                len(inv.get("vaes", [])),
                len(inv.get("text_encoders", [])),
                len(inv.get("loras", [])),
                len(inv.get("wan22", [])),
                len(inv.get("metadata", [])),
            )
            return {
                "vaes": _normalize_inventory_for_api(inv.get("vaes", [])),
                "text_encoders": _normalize_inventory_for_api(inv.get("text_encoders", [])),
                "loras": _normalize_inventory_for_api(inv.get("loras", [])),
                "wan22": {"gguf": _normalize_inventory_for_api(inv.get("wan22", []))},
                "metadata": _normalize_inventory_for_api(inv.get("metadata", [])),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inventory refresh failed: {exc}")

    @router.post("/api/models/load")
    def api_models_load(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get("tab_id") or "")
        if not tab_id:
            raise HTTPException(status_code=400, detail="tab_id required")
        log.info("[models] load requested for tab %s", tab_id)
        return {"ok": True}

    @router.post("/api/models/unload")
    def api_models_unload(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get("tab_id") or "")
        if not tab_id:
            raise HTTPException(status_code=400, detail="tab_id required")
        log.info("[models] unload requested for tab %s", tab_id)
        return {"ok": True}

    @router.get("/api/engines/capabilities")
    def list_engine_capabilities() -> Dict[str, Any]:
        try:
            from apps.backend.runtime.model_registry.capabilities import (
                serialize_engine_capabilities,
                serialize_family_capabilities,
            )
            try:
                from apps.backend.runtime.memory.smart_offload import get_smart_cache_stats

                cache_stats = get_smart_cache_stats()
            except Exception:
                cache_stats = {}
            return {
                "engines": serialize_engine_capabilities(),
                "families": serialize_family_capabilities(),
                "smart_cache": cache_stats,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read engine capabilities: {exc}")

    @router.get("/api/samplers")
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
                    "supported": bool(entry.get("supported", True)),
                    "default_scheduler": spec.default_scheduler if spec else None,
                    "allowed_schedulers": sorted(spec.allowed_schedulers) if spec else [],
                }
            )
        return {"samplers": samplers}

    @router.get("/api/schedulers")
    def list_schedulers() -> Dict[str, Any]:
        schedulers = []
        for entry in SCHEDULER_OPTIONS:
            if not entry.get("supported", True):
                continue
            schedulers.append(
                {
                    "name": entry["name"],
                    "supported": bool(entry.get("supported", True)),
                }
            )
        return {"schedulers": schedulers}

    @router.get("/api/embeddings")
    def list_embeddings() -> Dict[str, Any]:
        from apps.backend.infra.registry.embeddings import describe_embeddings as _describe

        info = [e.__dict__ for e in _describe()]
        loaded = {
            e["name"]: {
                "name": e["name"],
                "vectors": e.get("vectors"),
                "shape": e.get("dims"),
                "step": e.get("step"),
            }
            for e in info
            if e.get("vectors")
        }
        skipped = {
            e["name"]: {
                "name": e["name"],
                "vectors": e.get("vectors"),
                "shape": e.get("dims"),
                "step": e.get("step"),
            }
            for e in info
            if not e.get("vectors")
        }
        return {"loaded": loaded, "skipped": skipped, "embeddings_info": info}

    return router
