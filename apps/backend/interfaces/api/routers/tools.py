"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools API routes (GGUF conversion + file browser).
Provides long-running conversion job tracking and filesystem browsing for file picker dialogs.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for tools endpoints.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()
    _gguf_conversion_jobs: Dict[str, Dict[str, Any]] = {}

    @router.post("/api/tools/convert-gguf")
    async def convert_to_gguf(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Start a GGUF conversion job."""
        from apps.backend.runtime.tools.gguf_converter import (
            ConversionConfig,
            QuantizationType,
            convert_safetensors_to_gguf,
        )

        job_id = str(uuid.uuid4())[:8]

        # Validate paths
        config_path = payload.get("config_path", "")
        safetensors_path = payload.get("safetensors_path", "")
        output_path = payload.get("output_path", "")
        quant_str = payload.get("quantization", "F16")
        overrides_raw = payload.get("tensor_type_overrides", [])

        if not config_path or not safetensors_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required paths")

        if not os.path.exists(config_path) and not os.path.exists(os.path.join(config_path, "config.json")):
            raise HTTPException(status_code=400, detail=f"Config not found: {config_path}")

        if not os.path.exists(safetensors_path):
            raise HTTPException(status_code=400, detail=f"Safetensors not found: {safetensors_path}")

        try:
            quant = QuantizationType(quant_str)
        except ValueError:
            quant = QuantizationType.F16

        tensor_type_overrides: list[str] = []
        if isinstance(overrides_raw, str):
            tensor_type_overrides = [ln.strip() for ln in overrides_raw.splitlines() if ln.strip()]
        elif isinstance(overrides_raw, list):
            tensor_type_overrides = [str(x).strip() for x in overrides_raw if str(x).strip()]

        # Create job entry
        _gguf_conversion_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "current_tensor": "",
            "error": None,
        }

        def run_conversion() -> None:
            try:
                config = ConversionConfig(
                    config_path=config_path,
                    safetensors_path=safetensors_path,
                    output_path=output_path,
                    quantization=quant,
                    tensor_type_overrides=tensor_type_overrides,
                )

                def progress_cb(prog):
                    _gguf_conversion_jobs[job_id].update(
                        {
                            "status": prog.status,
                            "progress": prog.progress_percent,
                            "current_tensor": prog.current_tensor,
                        }
                    )

                convert_safetensors_to_gguf(config, progress_callback=progress_cb)
                _gguf_conversion_jobs[job_id]["status"] = "complete"
                _gguf_conversion_jobs[job_id]["progress"] = 100

            except Exception as exc:
                _gguf_conversion_jobs[job_id]["status"] = "error"
                _gguf_conversion_jobs[job_id]["error"] = str(exc)

        thread = threading.Thread(target=run_conversion, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started"}

    @router.get("/api/tools/convert-gguf/{job_id}")
    async def get_gguf_conversion_status(job_id: str) -> Dict[str, Any]:
        if job_id not in _gguf_conversion_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _gguf_conversion_jobs[job_id]

    @router.get("/api/tools/browse-files")
    async def browse_files(path: str = "", extensions: str = "") -> Dict[str, Any]:
        """Browse files/directories for file picker."""
        if not path:
            path = str(codex_root / "models")

        if not os.path.exists(path):
            return {"path": path, "exists": False, "items": []}

        if os.path.isfile(path):
            return {"path": path, "exists": True, "is_file": True, "items": []}

        ext_list = [e.strip().lower() for e in extensions.split(",") if e.strip()] if extensions else []

        items = []
        try:
            for entry in os.scandir(path):
                if entry.is_dir():
                    items.append({"name": entry.name, "type": "directory"})
                elif entry.is_file():
                    if not ext_list or any(entry.name.lower().endswith(ext) for ext in ext_list):
                        items.append(
                            {
                                "name": entry.name,
                                "type": "file",
                                "size": entry.stat().st_size,
                            }
                        )
        except PermissionError:
            pass

        items.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))

        return {
            "path": path,
            "exists": True,
            "is_file": False,
            "parent": str(Path(path).parent),
            "items": items,
        }

    return router
