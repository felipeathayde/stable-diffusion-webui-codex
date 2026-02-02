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

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for SUPIR endpoints.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.interfaces.api.task_registry import TaskEntry, register_task
from apps.backend.runtime.families.supir.config import parse_supir_enhance_config
from apps.backend.runtime.families.supir.errors import SupirBaseModelError, SupirConfigError, SupirWeightsError
from apps.backend.runtime.families.supir.loader import resolve_supir_assets
from apps.backend.runtime.families.supir.samplers.registry import iter_supir_sampler_labels
from apps.backend.runtime.families.supir.weights import SupirVariant, supir_weights_diagnostics


def _require_explicit_device(payload: Dict[str, Any]) -> str:
    """Validate and apply the per-request device selection (fail loud)."""

    from apps.backend.runtime.memory import memory_management as mem_management

    raw = payload.get("codex_device") or payload.get("device") or payload.get("codex_diffusion_device") or ""
    dev = str(raw).strip().lower()
    allowed = {"cpu", "cuda", "mps", "xpu", "directml"}
    if dev not in allowed:
        raise HTTPException(status_code=400, detail="Missing or invalid device (cpu|cuda|mps|xpu|directml)")
    try:
        mem_management.switch_primary_device(dev)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return dev


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
            raise HTTPException(status_code=400, detail=f"payload must be JSON: {exc}") from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        if not get_paths_for("supir_models"):
            raise HTTPException(status_code=400, detail="No 'supir_models' path configured in apps/paths.json")

        try:
            config = parse_supir_enhance_config(data)
        except SupirConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        # Apply device selection (fail loud; no silent fallbacks).
        _require_explicit_device(data)

        try:
            assets = resolve_supir_assets(
                base_model=config.base_model,
                variant=config.variant,
                supir_models_roots=[Path(p) for p in get_paths_for("supir_models")],
            )
        except (SupirBaseModelError, SupirWeightsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        try:
            image_bytes = await image.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to read upload: {exc}") from None
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
            require_explicit_device=_require_explicit_device,
            opts_get=opts_get,
            generation_provenance=generation_provenance,
            save_generated_images=save_generated_images,
        )

        return {"task_id": task_id}

    return router
