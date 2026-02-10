"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model and asset inventory API routes.
Exposes checkpoints, inventories, samplers/schedulers, embeddings, and engine capabilities.
Capability surfaces include semantic-engine asset contracts (owner-resolved from canonical engine ids) plus backend-owned dependency checks
so the UI can enforce sha-only external asset selection and readiness gating deterministically. Also provides prompt token-counting
(`/api/models/prompt-token-count`) using vendored offline tokenizers.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for model/inventory endpoints.
- `_count_prompt_tokens` (function): Returns tokenizer-accurate prompt token counts for supported semantic engines.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from apps.backend.runtime.sampling import SAMPLER_OPTIONS, SCHEDULER_OPTIONS
from apps.backend.interfaces.api.path_utils import _normalize_inventory_for_api
from apps.backend.interfaces.api.serializers import _serialize_checkpoint
from apps.backend.interfaces.api.file_metadata import read_file_metadata

_REPO_ROOT = Path(__file__).resolve().parents[5]
_HF_ROOT = _REPO_ROOT / "apps/backend/huggingface"

_PROMPT_TOKENIZER_DIRS: Dict[str, Path] = {
    "sd15": _HF_ROOT / "runwayml/stable-diffusion-v1-5/tokenizer",
    "sdxl": _HF_ROOT / "stabilityai/stable-diffusion-xl-base-1.0/tokenizer",
    "flux1": _HF_ROOT / "black-forest-labs/FLUX.1-dev/tokenizer_2",
    "chroma": _HF_ROOT / "Chroma/tokenizer",
    "zimage": _HF_ROOT / "Tongyi-MAI/Z-Image/tokenizer",
    "wan": _HF_ROOT / "Wan-AI/Wan2.2-T2V-A14B-Diffusers/tokenizer",
    "anima_qwen": _HF_ROOT / "circlestone-labs/Anima/qwen25_tokenizer",
    "anima_t5": _HF_ROOT / "circlestone-labs/Anima/t5_tokenizer",
}

_ENGINE_TOKENIZER_KEY: Dict[str, str] = {
    "sd15": "sd15",
    "sd20": "sd15",
    "sdxl": "sdxl",
    "flux1": "flux1",
    "flux1_kontext": "flux1",
    "chroma": "chroma",
    "flux1_chroma": "chroma",
    "zimage": "zimage",
    "anima": "anima",
    "wan": "wan",
    "wan22": "wan",
    "wan22_5b": "wan",
    "wan22_14b": "wan",
}


@lru_cache(maxsize=32)
def _load_tokenizer(tokenizer_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)


def _tokenize_len(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer([prompt], truncation=False, add_special_tokens=False)
    ids = encoded.get("input_ids")
    if not (isinstance(ids, list) and ids and isinstance(ids[0], list)):
        raise RuntimeError("Prompt tokenizer returned invalid 'input_ids' payload.")
    return len(ids[0])


def _resolve_tokenizer_path(key: str) -> Path:
    candidate = _PROMPT_TOKENIZER_DIRS.get(key)
    if candidate is None:
        raise RuntimeError(f"Unsupported prompt tokenizer key '{key}'.")
    if not candidate.exists():
        raise RuntimeError(
            f"Prompt tokenizer directory missing for '{key}': {candidate}. "
            "Expected vendored Hugging Face assets under apps/backend/huggingface."
        )
    return candidate


def _count_prompt_tokens(engine: str, prompt: str) -> int:
    normalized = str(engine or "").strip().lower()
    if not normalized:
        raise RuntimeError("Prompt token count requires a non-empty engine id.")
    if not prompt:
        return 0
    tokenizer_key = _ENGINE_TOKENIZER_KEY.get(normalized)
    if tokenizer_key is None:
        raise RuntimeError(
            f"Unsupported engine '{engine}' for prompt token count. "
            f"Supported: {', '.join(sorted(_ENGINE_TOKENIZER_KEY.keys()))}."
        )

    if tokenizer_key == "anima":
        qwen_tok = _load_tokenizer(str(_resolve_tokenizer_path("anima_qwen")))
        t5_tok = _load_tokenizer(str(_resolve_tokenizer_path("anima_t5")))
        qwen_count = _tokenize_len(qwen_tok, prompt)
        t5_count = _tokenize_len(t5_tok, prompt)
        return max(qwen_count, t5_count)

    tokenizer = _load_tokenizer(str(_resolve_tokenizer_path(tokenizer_key)))
    return _tokenize_len(tokenizer, prompt)


def build_router(
    *,
    model_api: Any,
) -> APIRouter:
    router = APIRouter()
    log = logging.getLogger("backend.api")

    @router.get("/api/models")
    def list_models(refresh: bool = Query(False, description="If true, re-scan checkpoint roots before returning.")) -> Dict[str, Any]:
        entries = model_api.list_checkpoints(refresh=bool(refresh))
        models = [_serialize_checkpoint(entry) for entry in entries]
        models_info = [e.as_dict() for e in entries]
        current = models[0]["title"] if models else None
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

    @router.get("/api/models/checkpoint-metadata")
    def get_checkpoint_metadata(value: str = Query(..., description="Checkpoint title/name/path to resolve.")) -> Dict[str, Any]:
        record = None
        try:
            record = model_api.find_checkpoint(value)
        except Exception:
            record = None

        if record is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")

        raw_path = str(getattr(record, "filename", "") or getattr(record, "path", "") or "").strip()
        if not raw_path:
            raise HTTPException(status_code=500, detail="checkpoint record missing filename")

        weights_path = Path(raw_path).expanduser()
        resolved = weights_path.resolve(strict=False)
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail="checkpoint is not a file")

        try:
            meta = read_file_metadata(str(resolved))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        size_bytes = int(resolved.stat().st_size)
        short_hash = getattr(record, "short_hash", None) or getattr(record, "shorthash", None)
        return {
            "hash": short_hash,
            "file.name": resolved.stem,
            "file.path": str(resolved),
            "file.size.bytes": size_bytes,
            "file.size.megabytes": round(size_bytes / 1_000_000, 3),
            "file.size.gigabytes": round(size_bytes / 1_000_000_000, 3),
            "metadata": {"raw": dict(meta.flat), "nested": dict(meta.nested)},
        }

    @router.post("/api/models/prompt-token-count")
    def prompt_token_count(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        engine = str(payload.get("engine") or "").strip()
        if not engine:
            raise HTTPException(status_code=400, detail="'engine' is required.")
        prompt = str(payload.get("prompt") or "")
        try:
            count = _count_prompt_tokens(engine=engine, prompt=prompt)
        except RuntimeError as exc:
            message = str(exc)
            if "Unsupported engine" in message:
                raise HTTPException(status_code=400, detail=message)
            raise HTTPException(status_code=500, detail=message)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to count prompt tokens: {exc}")
        return {
            "engine": engine,
            "prompt_len": len(prompt),
            "count": max(0, math.trunc(count)),
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
                ENGINE_ID_TO_SEMANTIC_ENGINE,
                serialize_engine_capabilities,
                serialize_family_capabilities,
            )
            from apps.backend.core.contracts.asset_requirements import (
                contract_for_core_only,
                contract_for_engine,
                contract_owner_for_semantic_engine,
            )
            from apps.backend.interfaces.api.dependency_checks import build_engine_dependency_checks
            try:
                from apps.backend.runtime.memory.smart_offload import get_smart_cache_stats

                cache_stats = get_smart_cache_stats()
            except Exception:
                cache_stats = {}

            engine_id_to_semantic_engine: Dict[str, str] = {
                engine_id: semantic.value for engine_id, semantic in ENGINE_ID_TO_SEMANTIC_ENGINE.items()
            }
            engines = serialize_engine_capabilities()
            asset_contracts: Dict[str, Any] = {}
            for semantic_engine in sorted(engines.keys()):
                contract_owner_engine_id = contract_owner_for_semantic_engine(semantic_engine)
                asset_contracts[semantic_engine] = {
                    "base": contract_for_engine(contract_owner_engine_id).as_dict(),
                    "core_only": contract_for_core_only(contract_owner_engine_id).as_dict(),
                }
            dependency_checks = build_engine_dependency_checks(
                engine_capabilities=engines,
                model_api=model_api,
            )
            return {
                "engines": engines,
                "families": serialize_family_capabilities(),
                "smart_cache": cache_stats,
                "asset_contracts": asset_contracts,
                "engine_id_to_semantic_engine": engine_id_to_semantic_engine,
                "dependency_checks": dependency_checks,
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
