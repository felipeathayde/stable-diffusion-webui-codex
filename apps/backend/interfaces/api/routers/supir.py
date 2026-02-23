"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR enhance API routes.
Exposes:
- SUPIR weights diagnostics (`GET /api/supir/models`)
- SUPIR enhance tasks (`POST /api/supir/enhance`)
Validates explicit per-request device selection and dispatches a background task worker (single-flight-gated) for enhance runs.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for SUPIR endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.interfaces.api.device_selection import parse_device_from_payload
from apps.backend.interfaces.api.public_errors import public_http_error_detail
from apps.backend.interfaces.api.task_registry import TaskEntry, register_task
from apps.backend.runtime.families.supir.config import parse_supir_enhance_config
from apps.backend.runtime.families.supir.errors import SupirBaseModelError, SupirConfigError, SupirWeightsError
from apps.backend.runtime.families.supir.loader import resolve_supir_assets
from apps.backend.runtime.families.supir.samplers.registry import iter_supir_sampler_labels
from apps.backend.runtime.families.supir.weights import SupirVariant, supir_weights_diagnostics

_router_log = logging.getLogger("backend.api.routers.supir")


def _parse_explicit_device(payload: Dict[str, Any]) -> str:
    """Validate the per-request device selection (fail loud).

    Note: do not call `switch_primary_device()` here; apply it only when the task starts running (single-flight-safe).
    """
    try:
        return parse_device_from_payload(payload)
    except ValueError as exc:
        _router_log.warning("supir device selection validation failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=public_http_error_detail(exc, fallback="Invalid 'device' selection"),
        ) from None


def build_router(
    *,
    codex_root: Path,
    opts_get,
    generation_provenance,
    save_generated_images,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/supir/models")
    async def get_supir_models() -> Dict[str, Any]:
        roots = get_paths_for("supir_models")
        return {
            "supir_models": supir_weights_diagnostics(roots=[Path(p) for p in roots]),
            "variants": [{"key": v.value, "label": v.value} for v in SupirVariant],
            "samplers": list(iter_supir_sampler_labels(include_dev=True)),
            "note": "SUPIR enhance is not yet ported (API skeleton + validations only).",
        }

    @router.post("/api/supir/enhance")
    async def supir_enhance(
        image: UploadFile | None = File(default=None),
        payload: str = Form(default="{}"),
    ) -> Dict[str, Any]:
        if image is None:
            raise HTTPException(status_code=400, detail="Missing 'image' file")
        try:
            data = json.loads(payload) if payload else {}
        except Exception as exc:
            _router_log.warning("supir payload JSON parse failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="payload must be valid JSON"),
            ) from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        if not get_paths_for("supir_models"):
            raise HTTPException(status_code=400, detail="No 'supir_models' path configured in apps/paths.json")

        device = _parse_explicit_device(data)
        try:
            config = parse_supir_enhance_config(data, device=device)
        except SupirConfigError as exc:
            _router_log.warning("supir config validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid SUPIR payload configuration"),
            ) from None

        try:
            assets = resolve_supir_assets(
                base_model=config.base_model,
                variant=config.variant,
                supir_models_roots=[Path(p) for p in get_paths_for("supir_models")],
            )
        except (SupirBaseModelError, SupirWeightsError) as exc:
            _router_log.warning("supir assets resolution failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="SUPIR assets resolution failed"),
            ) from None

        try:
            image_bytes = await image.read()
        except Exception as exc:
            _router_log.warning("supir upload read failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="failed to read upload"),
            ) from None
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty image upload")

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-supir-enhance-{uuid4().hex})"
        register_task(task_id, entry)

        from apps.backend.interfaces.api.tasks.supir_tasks import run_supir_enhance_task

        run_supir_enhance_task(
            task_id=task_id,
            payload=data,
            image_bytes=image_bytes,
            base_model_path=str(assets.base_checkpoint),
            variant_ckpt_path=str(assets.variant_ckpt),
            entry=entry,
            device=device,
            opts_get=opts_get,
            generation_provenance=generation_provenance,
            save_generated_images=save_generated_images,
        )

        return {"task_id": task_id}

    return router
