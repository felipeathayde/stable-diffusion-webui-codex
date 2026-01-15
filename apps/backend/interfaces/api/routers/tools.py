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
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()
    _gguf_conversion_jobs: Dict[str, Dict[str, Any]] = {}
    _gguf_conversion_controls: Dict[str, Dict[str, Any]] = {}

    @router.get("/api/tools/gguf-converter/presets")
    async def list_gguf_converter_presets() -> Dict[str, Any]:
        from apps.backend.runtime.tools.gguf_converter_presets import list_vendored_gguf_converter_presets

        presets = list_vendored_gguf_converter_presets(codex_root=codex_root)
        return {
            "presets": [
                {
                    "id": p.id,
                    "label": p.label,
                    "config_dir": p.config_dir,
                    "kind": p.kind,
                    "profile_id": p.profile_id,
                    "profile_id_comfy": p.profile_id_comfy,
                    "profile_id_native": p.profile_id_native,
                }
                for p in presets
            ]
        }

    @router.post("/api/tools/convert-gguf")
    async def convert_to_gguf(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Start a GGUF conversion job."""
        from apps.backend.runtime.tools.gguf_converter import (
            ConversionConfig,
            GGUFConversionCancelled,
            QuantizationType,
            convert_safetensors_to_gguf,
        )

        job_id = str(uuid.uuid4())[:8]

        # Validate paths
        config_path = payload.get("config_path", "")
        safetensors_path = payload.get("safetensors_path", "")
        output_path = payload.get("output_path", "")
        overwrite = bool(payload.get("overwrite", False))
        comfy_layout_raw = payload.get("comfy_layout", True)
        quant_str = payload.get("quantization", "F16")
        overrides_raw = payload.get("tensor_type_overrides", [])
        profile_id_raw = payload.get("profile_id", None)
        flux_txt_in_weight_dtype_raw = payload.get("flux_txt_in_weight_dtype", "auto")
        flux_out_proj_weight_dtype_raw = payload.get("flux_out_proj_weight_dtype", "auto")
        flux_final_modulation_weight_dtype_raw = payload.get("flux_final_modulation_weight_dtype", "auto")

        if not config_path or not safetensors_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required paths")

        if not os.path.exists(config_path) and not os.path.exists(os.path.join(config_path, "config.json")):
            raise HTTPException(status_code=400, detail=f"Config not found: {config_path}")

        if not os.path.exists(safetensors_path):
            raise HTTPException(status_code=400, detail=f"Safetensors not found: {safetensors_path}")

        final_path = Path(os.path.expanduser(str(output_path))).resolve()
        if final_path.exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"Output file already exists: {final_path}")
        if final_path.exists() and final_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory: {final_path}")

        if not isinstance(comfy_layout_raw, bool):
            raise HTTPException(status_code=400, detail="comfy_layout must be a boolean when provided")
        comfy_layout = bool(comfy_layout_raw)

        try:
            quant = QuantizationType(quant_str)
        except ValueError:
            quant = QuantizationType.F16

        profile_id: str | None = None
        if profile_id_raw is not None:
            profile_id = str(profile_id_raw).strip() or None
            if profile_id is not None:
                from apps.backend.runtime.tools.gguf_converter_profiles import profile_by_id

                try:
                    profile_by_id(profile_id)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

        def _normalize_float_dtype_choice(value: Any) -> str:
            raw = str(value or "").strip().upper()
            if raw in {"", "AUTO"}:
                return "auto"
            if raw in {"F16", "FP16"}:
                return "F16"
            if raw in {"F32", "FP32"}:
                return "F32"
            raise HTTPException(
                status_code=400,
                detail=f"Invalid float dtype selection: {value!r} (expected auto|F16|F32)",
            )

        flux_txt_in_weight_dtype = _normalize_float_dtype_choice(flux_txt_in_weight_dtype_raw)
        flux_out_proj_weight_dtype = _normalize_float_dtype_choice(flux_out_proj_weight_dtype_raw)
        flux_final_modulation_weight_dtype = _normalize_float_dtype_choice(flux_final_modulation_weight_dtype_raw)

        tensor_type_overrides: list[str] = []
        if isinstance(overrides_raw, str):
            tensor_type_overrides = [ln.strip() for ln in overrides_raw.splitlines() if ln.strip()]
        elif isinstance(overrides_raw, list):
            tensor_type_overrides = [str(x).strip() for x in overrides_raw if str(x).strip()]

        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_handle = tempfile.NamedTemporaryFile(
            prefix=f"{final_path.stem}.",
            suffix=f".part-{job_id}{final_path.suffix or '.gguf'}",
            dir=str(final_path.parent),
            delete=False,
        )
        tmp_path = Path(tmp_handle.name)
        tmp_handle.close()

        # Create job entry
        _gguf_conversion_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "current_tensor": "",
            "error": None,
        }

        cancel_event = threading.Event()
        _gguf_conversion_controls[job_id] = {
            "cancel_event": cancel_event,
            "tmp_path": tmp_path,
            "final_path": final_path,
        }

        def run_conversion() -> None:
            try:
                job = _gguf_conversion_jobs[job_id]
                ctrl = _gguf_conversion_controls[job_id]

                config = ConversionConfig(
                    config_path=config_path,
                    safetensors_path=safetensors_path,
                    output_path=str(ctrl["tmp_path"]),
                    profile_id=profile_id,
                    quantization=quant,
                    comfy_layout=comfy_layout,
                    tensor_type_overrides=tensor_type_overrides,
                    flux_txt_in_weight_dtype=flux_txt_in_weight_dtype,
                    flux_out_proj_weight_dtype=flux_out_proj_weight_dtype,
                    flux_final_modulation_weight_dtype=flux_final_modulation_weight_dtype,
                )

                def progress_cb(prog):
                    status = prog.status
                    percent = prog.progress_percent
                    if status == "complete":
                        # The converter reports completion before we atomically move the temp
                        # file into place; keep polling until final rename finishes.
                        status = "finalizing"
                        percent = min(99.9, percent)
                    job.update(
                        {
                            "status": status,
                            "progress": percent,
                            "current_tensor": prog.current_tensor,
                        }
                    )

                convert_safetensors_to_gguf(
                    config,
                    progress_callback=progress_cb,
                    should_cancel=lambda: bool(ctrl["cancel_event"].is_set()),
                )

                job["status"] = "finalizing"
                job["progress"] = 99.9
                os.replace(str(ctrl["tmp_path"]), str(ctrl["final_path"]))
                job["status"] = "complete"
                job["progress"] = 100

            except GGUFConversionCancelled:
                job = _gguf_conversion_jobs[job_id]
                job["status"] = "cancelled"
                job["error"] = None
                try:
                    ctrl = _gguf_conversion_controls[job_id]
                    tmp = Path(ctrl["tmp_path"])
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
            except Exception as exc:
                job = _gguf_conversion_jobs[job_id]
                job["status"] = "error"
                job["error"] = str(exc)
                try:
                    ctrl = _gguf_conversion_controls[job_id]
                    tmp = Path(ctrl["tmp_path"])
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

        thread = threading.Thread(target=run_conversion, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started"}

    @router.get("/api/tools/convert-gguf/{job_id}")
    async def get_gguf_conversion_status(job_id: str) -> Dict[str, Any]:
        if job_id not in _gguf_conversion_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _gguf_conversion_jobs[job_id]

    @router.post("/api/tools/convert-gguf/{job_id}/cancel")
    async def cancel_gguf_conversion(job_id: str) -> Dict[str, Any]:
        job = _gguf_conversion_jobs.get(job_id)
        ctrl = _gguf_conversion_controls.get(job_id)
        if job is None or ctrl is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") in {"complete", "error", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Job is not cancellable (status={job.get('status')})")

        ctrl["cancel_event"].set()
        job["status"] = "cancelling"
        return {"ok": True}

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
