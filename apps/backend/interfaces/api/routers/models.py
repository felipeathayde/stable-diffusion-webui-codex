"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model and asset inventory API routes.
Exposes checkpoints, inventories, samplers/schedulers, VAEs, text encoders, embeddings, and LoRA selections.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for model/inventory endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from apps.backend.runtime.sampling.catalog import SAMPLER_OPTIONS, SCHEDULER_OPTIONS
from apps.backend.interfaces.api.path_utils import _normalize_inventory_for_api, _path_for_api
from apps.backend.interfaces.api.serializers import _serialize_checkpoint


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

    @router.get("/api/vaes")
    def list_vaes() -> Dict[str, Any]:
        from apps.backend.infra.registry.vae import list_vaes as _list_vaes, describe_vaes as _describe_vaes
        current = str(opts_get("sd_vae", "Automatic"))
        try:
            models_root = str(codex_root / "models")
            hf_root = str(codex_root / "apps" / "backend" / "huggingface")
            choices = _list_vaes(models_root=models_root, vendored_hf_root=hf_root)
            info = _describe_vaes(models_root=models_root, vendored_hf_root=hf_root)
            return {"vaes": choices, "current": current, "vaes_info": info}
        except Exception:
            return {"vaes": ["Automatic", "Built in", "None"], "current": current, "vaes_info": []}

    @router.get("/api/text-encoders")
    def list_text_encoders() -> Dict[str, Any]:
        from apps.backend.infra.registry.text_encoder_roots import list_text_encoder_roots_by_family
        try:
            roots_by_family = list_text_encoder_roots_by_family()
            entries: list[dict[str, object]] = []
            labels: list[str] = []
            for family, roots in roots_by_family.items():
                for root in roots:
                    label_path = _path_for_api(root.path)
                    label = f"{family}/{label_path}"
                    labels.append(label)
                    entries.append(
                        {
                            "family": family,
                            "name": root.name,
                            "path": _path_for_api(root.path),
                            "label": label,
                        }
                    )
            labels_sorted = sorted(set(labels))
            selected = opts_get("text_encoder_overrides", []) or []
            if not isinstance(selected, list):
                selected = []
            selected_norm: list[str] = []
            for raw in selected:
                s = str(raw or "").strip()
                if not s:
                    continue
                if "/" in s:
                    fam, rest = s.split("/", 1)
                    if fam and rest:
                        selected_norm.append(f"{fam}/{_path_for_api(rest)}")
                        continue
                selected_norm.append(s)
            return {"text_encoders": labels_sorted, "current": selected_norm, "text_encoders_info": entries}
        except Exception:
            return {"text_encoders": [], "current": [], "text_encoders_info": []}

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

    @router.get("/api/loras")
    def list_loras() -> Dict[str, Any]:
        from apps.backend.infra.registry.lora import list_loras as _list_loras, describe_loras as _describe_loras
        items = _list_loras()
        info = [e.__dict__ for e in _describe_loras()]
        return {"loras": items, "loras_info": info}

    @router.get("/api/loras/selections")
    def get_lora_selections() -> Dict[str, Any]:
        from apps.backend.runtime.adapters.lora.selections import get_selections
        sels = get_selections()
        return {"selections": [{"path": s.path, "weight": s.weight, "online": s.online} for s in sels]}

    @router.post("/api/loras/apply")
    def apply_lora_selections(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict) or "selections" not in payload or not isinstance(payload["selections"], list):
            raise HTTPException(status_code=400, detail='payload must be {"selections": [...]}')
        from apps.backend.runtime.adapters.lora.selections import get_selections, set_selections
        set_selections(payload["selections"])
        return {"ok": True, "count": len(get_selections())}

    return router
