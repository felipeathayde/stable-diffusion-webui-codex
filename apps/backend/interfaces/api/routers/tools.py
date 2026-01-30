"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools API routes (GGUF conversion + CodexPack v1 packing + file browser + PNG metadata inspection).
Provides long-running conversion/packing job tracking, filesystem browsing for file picker dialogs, and small utility endpoints used by the UI.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for tools endpoints.
"""

from __future__ import annotations

import errno
from io import BytesIO
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, File, HTTPException, UploadFile


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()
    _gguf_conversion_jobs: Dict[str, Dict[str, Any]] = {}
    _gguf_conversion_controls: Dict[str, Dict[str, Any]] = {}
    _codexpack_pack_jobs: Dict[str, Dict[str, Any]] = {}
    _codexpack_pack_controls: Dict[str, Dict[str, Any]] = {}

    def _alloc_nonexistent_path(parent: Path, *, prefix: str, suffix: str) -> Path:
        for _ in range(64):
            token = uuid.uuid4().hex[:10]
            candidate = parent / f"{prefix}{token}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Failed to allocate a unique output path under: {str(parent)!r}")

    def _replace_or_copy(src: Path, dst: Path) -> None:
        try:
            os.replace(str(src), str(dst))
            return
        except OSError as exc:
            # Cross-device rename (EXDEV) can happen when temp dirs live on a different volume.
            if getattr(exc, "errno", None) != errno.EXDEV:
                raise
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(str(src), str(dst))
        try:
            src.unlink()
        except Exception:
            pass

    def _derive_base_failed_path(codexpack_path: Path, *, job_id: str) -> Path:
        name = codexpack_path.name
        suffix = ".codexpack.gguf"
        if name.lower().endswith(suffix):
            base = name[: -len(suffix)]
        else:
            base = codexpack_path.stem
        return codexpack_path.with_name(f"{base}.base.failed-{job_id}.gguf")

    @router.get("/api/tools/gguf-converter/presets")
    async def list_gguf_converter_presets() -> Dict[str, Any]:
        from apps.backend.runtime.tools.gguf_converter_float_groups import float_groups_for_profile_id
        from apps.backend.runtime.tools.gguf_converter_model_metadata import list_vendored_gguf_converter_model_metadata

        models = list_vendored_gguf_converter_model_metadata(codex_root=codex_root)
        profile_ids: set[str] = set()
        for model in models:
            for comp in model.components:
                for pid in (comp.profile_id, comp.profile_id_comfy, comp.profile_id_native):
                    if pid:
                        profile_ids.add(pid)

        float_groups: dict[str, Any] = {}
        for pid in sorted(profile_ids):
            float_groups[pid] = [
                {"id": g.id, "label": g.label, "patterns": list(g.patterns)}
                for g in float_groups_for_profile_id(pid)
            ]

        return {
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "org": m.org,
                    "repo": m.repo,
                    "components": [
                        {
                            "id": c.id,
                            "label": c.label,
                            "config_dir": c.config_dir,
                            "kind": c.kind,
                            "profile_id": c.profile_id,
                            "profile_id_comfy": c.profile_id_comfy,
                            "profile_id_native": c.profile_id_native,
                        }
                        for c in m.components
                    ],
                }
                for m in models
            ],
            "float_groups": float_groups,
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
        codexpack_v1_raw = payload.get("codexpack_v1", False)
        quant_str = payload.get("quantization", "F16")
        overrides_raw = payload.get("tensor_type_overrides", [])
        profile_id_raw = payload.get("profile_id", None)
        float_group_overrides_raw = payload.get("float_group_overrides", {})

        if not config_path or not safetensors_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required paths")

        if not os.path.exists(config_path) and not os.path.exists(os.path.join(config_path, "config.json")):
            raise HTTPException(status_code=400, detail=f"Config not found: {config_path}")

        if not os.path.exists(safetensors_path):
            raise HTTPException(status_code=400, detail=f"Safetensors not found: {safetensors_path}")

        final_path = Path(os.path.expanduser(str(output_path))).resolve()

        if not isinstance(comfy_layout_raw, bool):
            raise HTTPException(status_code=400, detail="comfy_layout must be a boolean when provided")
        comfy_layout = bool(comfy_layout_raw)

        if not isinstance(codexpack_v1_raw, bool):
            raise HTTPException(status_code=400, detail="codexpack_v1 must be a boolean when provided")
        codexpack_v1 = bool(codexpack_v1_raw)

        if codexpack_v1:
            if not final_path.name.lower().endswith(".codexpack.gguf"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "When codexpack_v1 is enabled, output_path must end with `.codexpack.gguf` "
                        "(CodexPack is the primary output; the intermediate base GGUF is temp-only)."
                    ),
                )
        else:
            if final_path.name.lower().endswith(".codexpack.gguf"):
                raise HTTPException(
                    status_code=400,
                    detail="Output path ends with `.codexpack.gguf`. Use codexpack_v1=true or choose a base `.gguf` output path.",
                )
            if not final_path.name.lower().endswith(".gguf"):
                raise HTTPException(status_code=400, detail="Output path must end with `.gguf`.")

        if final_path.exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"Output file already exists: {final_path}")
        if final_path.exists() and final_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory: {final_path}")

        try:
            quant = QuantizationType(quant_str)
        except ValueError:
            if codexpack_v1:
                raise HTTPException(status_code=400, detail=f"Invalid quantization: {quant_str!r}")
            quant = QuantizationType.F16

        codexpack_path: Path | None = None
        if codexpack_v1:
            if quant is not QuantizationType.Q4_K:
                raise HTTPException(
                    status_code=400,
                    detail="CodexPack v1 requires quantization=Q4_K (mixed presets are not supported).",
                )
            if not comfy_layout:
                raise HTTPException(
                    status_code=400,
                    detail="CodexPack v1 requires Comfy Layout on (Comfy/Codex key layout).",
                )

            codexpack_path = final_path

            cfg_dir = Path(os.path.expanduser(str(config_path))).resolve()
            cfg_json_path = cfg_dir
            if cfg_dir.is_dir():
                cfg_json_path = cfg_dir / "config.json"
            if not cfg_json_path.is_file():
                raise HTTPException(status_code=400, detail=f"config.json not found for CodexPack: {cfg_json_path}")

            try:
                cfg = json.loads(cfg_json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read config.json for CodexPack: {exc}") from exc

            class_name = str(cfg.get("_class_name") or "").strip()
            if class_name != "ZImageTransformer2DModel":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "CodexPack v1 is only supported for Z-Image denoisers "
                        f"(expected _class_name='ZImageTransformer2DModel', got {class_name!r})."
                    ),
                )

            # v1 keymap id is locked to Z-Image Base (Turbo uses a different id).
            shift: float | None = None
            cfg_root = cfg_json_path.parent
            for cand in (
                cfg_root / "scheduler" / "scheduler_config.json",
                cfg_root.parent / "scheduler" / "scheduler_config.json",
            ):
                if not cand.is_file():
                    continue
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    continue
                raw_shift = data.get("shift")
                if raw_shift is None:
                    continue
                try:
                    shift = float(raw_shift)
                except Exception:
                    continue
                break

            if shift is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "CodexPack v1 requires Z-Image Base. Could not detect `shift` from scheduler_config.json "
                        f"near {cfg_root}. Ensure the source includes `scheduler/scheduler_config.json`."
                    ),
                )
            if abs(shift - 6.0) >= 1e-3:
                variant = "turbo" if abs(shift - 3.0) < 1e-3 else f"shift={shift:g}"
                raise HTTPException(
                    status_code=400,
                    detail=f"CodexPack v1 keymap id is Base-only (expected shift=6.0). Detected: {variant}.",
                )

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

        float_group_overrides: dict[str, str] = {}
        if float_group_overrides_raw is None:
            float_group_overrides = {}
        elif isinstance(float_group_overrides_raw, dict):
            for k, v in float_group_overrides_raw.items():
                group_id = str(k or "").strip()
                if not group_id:
                    raise HTTPException(status_code=400, detail="float_group_overrides contains an empty group id")
                float_group_overrides[group_id] = _normalize_float_dtype_choice(v)
        else:
            raise HTTPException(status_code=400, detail="float_group_overrides must be an object/dict when provided")

        if any(v != "auto" for v in float_group_overrides.values()):
            if profile_id is None:
                raise HTTPException(status_code=400, detail="float_group_overrides requires profile_id")

            from apps.backend.runtime.tools.gguf_converter_float_groups import float_groups_for_profile_id

            allowed = {g.id for g in float_groups_for_profile_id(profile_id)}
            for gid, choice in float_group_overrides.items():
                if choice == "auto":
                    continue
                if gid not in allowed:
                    allowed_msg = ", ".join(sorted(allowed)) if allowed else "(none)"
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown float dtype group for profile {profile_id!r}: {gid!r} (allowed: {allowed_msg})",
                    )

        tensor_type_overrides: list[str] = []
        if isinstance(overrides_raw, str):
            tensor_type_overrides = [ln.strip() for ln in overrides_raw.splitlines() if ln.strip()]
        elif isinstance(overrides_raw, list):
            tensor_type_overrides = [str(x).strip() for x in overrides_raw if str(x).strip()]

        final_path.parent.mkdir(parents=True, exist_ok=True)
        base_tmp_path: Path | None = None
        if codexpack_v1:
            base_tmp_handle = tempfile.NamedTemporaryFile(
                prefix=f"codexpack_base_{job_id}.",
                suffix=".gguf",
                delete=False,
            )
            base_tmp_path = Path(base_tmp_handle.name)
            base_tmp_handle.close()
            tmp_path = base_tmp_path
        else:
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
            "output_path": str(final_path),
        }
        if codexpack_path is not None:
            _gguf_conversion_jobs[job_id]["codexpack_output_path"] = str(codexpack_path)

        cancel_event = threading.Event()
        _gguf_conversion_controls[job_id] = {
            "cancel_event": cancel_event,
            "tmp_path": tmp_path,
            "final_path": final_path,
            "codexpack_path": codexpack_path,
            "base_tmp_path": base_tmp_path,
        }

        def run_conversion() -> None:
            try:
                job = _gguf_conversion_jobs[job_id]
                ctrl = _gguf_conversion_controls[job_id]
                converted_ok = False

                config = ConversionConfig(
                    config_path=config_path,
                    safetensors_path=safetensors_path,
                    output_path=str(ctrl["tmp_path"]),
                    profile_id=profile_id,
                    quantization=quant,
                    comfy_layout=comfy_layout,
                    tensor_type_overrides=tensor_type_overrides,
                    float_group_overrides=float_group_overrides,
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
                converted_ok = True

                job["status"] = "finalizing"
                job["progress"] = 99.9
                if ctrl.get("codexpack_path") is None:
                    os.replace(str(ctrl["tmp_path"]), str(ctrl["final_path"]))

                if ctrl.get("codexpack_path") is not None:
                    from apps.backend.quantization.codexpack_keymaps import ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1
                    from apps.backend.runtime.tools.codexpack_packer import pack_gguf_to_codexpack_v1

                    codexpack_final_path = Path(ctrl["codexpack_path"])
                    base_tmp = Path(ctrl["tmp_path"])
                    job["status"] = "packing_codexpack"
                    job["progress"] = 99.95

                    pack_tmp_path = _alloc_nonexistent_path(
                        codexpack_final_path.parent,
                        prefix=f"{codexpack_final_path.stem}.part-{job_id}-",
                        suffix=codexpack_final_path.suffix or ".gguf",
                    )
                    try:
                        pack_gguf_to_codexpack_v1(
                            str(base_tmp),
                            str(pack_tmp_path),
                            keymap_id=ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1,
                        )
                        os.replace(str(pack_tmp_path), str(codexpack_final_path))
                    finally:
                        try:
                            if pack_tmp_path.exists():
                                pack_tmp_path.unlink()
                        except Exception:
                            pass
                    if ctrl.get("base_tmp_path") is not None:
                        try:
                            base_tmp.unlink()
                        except Exception:
                            pass

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
                msg = str(exc)
                try:
                    ctrl = _gguf_conversion_controls[job_id]
                    tmp = Path(ctrl["tmp_path"])
                    if ctrl.get("codexpack_path") is not None and converted_ok and tmp.exists():
                        base_failed_path = _derive_base_failed_path(Path(ctrl["codexpack_path"]), job_id=job_id)
                        try:
                            _replace_or_copy(tmp, base_failed_path)
                            msg = f"{msg}\nPreserved base GGUF (packing failed): {str(base_failed_path)}"
                        except Exception:
                            try:
                                tmp.unlink()
                            except Exception:
                                pass
                    else:
                        if tmp.exists():
                            tmp.unlink()
                except Exception:
                    pass
                job["error"] = msg
                job["status"] = "error"

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

    @router.post("/api/tools/codexpack/pack-v1")
    async def pack_codexpack_v1(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Start a CodexPack v1 packing job from an existing base GGUF."""
        from apps.backend.quantization.codexpack_keymaps import ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1
        from apps.backend.runtime.checkpoint.io import read_gguf_metadata

        src_gguf_path = str(payload.get("src_gguf_path") or "").strip()
        output_path = str(payload.get("output_path") or "").strip()
        overwrite = bool(payload.get("overwrite", False))
        if not src_gguf_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required paths")

        src_path = Path(os.path.expanduser(src_gguf_path)).resolve()
        if not src_path.is_file():
            raise HTTPException(status_code=400, detail=f"GGUF not found: {src_path}")
        if src_path.name.lower().endswith(".codexpack.gguf"):
            raise HTTPException(status_code=400, detail="Refusing to pack an existing `*.codexpack.gguf` file.")

        final_path = Path(os.path.expanduser(output_path)).resolve()
        if not final_path.name.lower().endswith(".codexpack.gguf"):
            raise HTTPException(status_code=400, detail="output_path must end with `.codexpack.gguf` for CodexPack v1.")
        if final_path.exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"Output file already exists: {final_path}")
        if final_path.exists() and final_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory: {final_path}")

        meta: dict[str, Any] = {}
        try:
            meta = read_gguf_metadata(str(src_path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read GGUF metadata: {exc}") from exc

        comfy_layout = bool(meta.get("codex.converter.comfy_layout", False))
        variant = str(meta.get("codex.zimage.variant") or "").strip().lower()
        arch = str(meta.get("model.architecture") or "").strip().lower()
        quant = str(meta.get("gguf.quantization") or "").strip().upper()

        if arch != "zimage":
            raise HTTPException(status_code=400, detail=f"CodexPack v1 packer expects a Z-Image GGUF; got model.architecture={arch!r}")
        if not comfy_layout:
            raise HTTPException(
                status_code=400,
                detail="CodexPack v1 requires a base GGUF created with Comfy Layout on (`codex.converter.comfy_layout=true`).",
            )
        if variant != "base":
            raise HTTPException(
                status_code=400,
                detail=f"CodexPack v1 keymap id is Base-only; expected codex.zimage.variant='base', got {variant!r}.",
            )
        if quant != "Q4_K":
            raise HTTPException(status_code=400, detail=f"CodexPack v1 requires gguf.quantization='Q4_K'; got {quant!r}.")

        job_id = str(uuid.uuid4())[:8]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        _codexpack_pack_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "current_tensor": "",
            "error": None,
            "output_path": str(final_path),
        }
        _codexpack_pack_controls[job_id] = {
            "src_path": src_path,
            "final_path": final_path,
        }

        def run_pack() -> None:
            try:
                job = _codexpack_pack_jobs[job_id]
                ctrl = _codexpack_pack_controls[job_id]
                from apps.backend.runtime.tools.codexpack_packer import pack_gguf_to_codexpack_v1

                job["status"] = "packing_codexpack"
                job["progress"] = 1.0

                out_final = Path(ctrl["final_path"])
                out_tmp = _alloc_nonexistent_path(out_final.parent, prefix=f"{out_final.stem}.part-{job_id}-", suffix=out_final.suffix or ".gguf")
                try:
                    pack_gguf_to_codexpack_v1(
                        str(ctrl["src_path"]),
                        str(out_tmp),
                        keymap_id=ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1,
                    )
                    os.replace(str(out_tmp), str(out_final))
                finally:
                    try:
                        if out_tmp.exists():
                            out_tmp.unlink()
                    except Exception:
                        pass

                job["status"] = "complete"
                job["progress"] = 100.0
            except Exception as exc:
                job = _codexpack_pack_jobs[job_id]
                job["status"] = "error"
                job["error"] = str(exc)

        thread = threading.Thread(target=run_pack, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started"}

    @router.get("/api/tools/codexpack/pack-v1/{job_id}")
    async def get_codexpack_pack_status(job_id: str) -> Dict[str, Any]:
        if job_id not in _codexpack_pack_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _codexpack_pack_jobs[job_id]

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

    @router.post("/api/tools/pnginfo/analyze")
    async def analyze_pnginfo(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Extract PNG text metadata for the PNG Info UI."""
        try:
            raw = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        max_bytes = 50 * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_bytes} bytes)")

        try:
            from PIL import Image  # type: ignore

            img = Image.open(BytesIO(raw))
            img.load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        fmt = str(getattr(img, "format", "") or "").upper()
        if fmt != "PNG":
            raise HTTPException(status_code=415, detail="Only PNG is supported")

        metadata: dict[str, str] = {}
        info = getattr(img, "info", None)
        if isinstance(info, dict):
            for k, v in info.items():
                if isinstance(k, str) and isinstance(v, str):
                    text = v.strip()
                    if text:
                        metadata[k] = text

        # Some PIL versions expose textual chunks on `.text` as well.
        text_map = getattr(img, "text", None)
        if isinstance(text_map, dict):
            for k, v in text_map.items():
                if isinstance(k, str) and isinstance(v, str):
                    text = v.strip()
                    if text:
                        metadata.setdefault(k, text)

        return {
            "width": int(getattr(img, "width", 0) or 0),
            "height": int(getattr(img, "height", 0) or 0),
            "metadata": metadata,
        }

    return router
