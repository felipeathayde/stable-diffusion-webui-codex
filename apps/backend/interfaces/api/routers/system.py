"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: System/diagnostic API routes (health, version, memory).
Exposes lightweight endpoints used by the UI footer and diagnostics overlays.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for system endpoints.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException


def build_router(*, app_version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health() -> Dict[str, bool]:
        return {"ok": True}

    @router.get("/api/version")
    def version_info() -> Dict[str, Any]:
        """Return backend version details for footer display."""
        # Git commit
        git_commit: Optional[str] = None
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = os.environ.get("GIT_COMMIT") or os.environ.get("VITE_GIT_COMMIT") or None

        # Python
        import sys

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Torch/CUDA (optional)
        torch_ver: Optional[str] = None
        cuda_ver: Optional[str] = None
        try:
            import torch  # type: ignore

            torch_ver = getattr(torch, "__version__", None)
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        except Exception:
            pass

        return {
            "app_version": app_version,
            "git_commit": git_commit,
            "python_version": py_ver,
            "torch_version": torch_ver,
            "cuda_version": cuda_ver,
        }

    @router.get("/api/memory")
    def memory() -> Dict[str, Any]:
        """Return a snapshot of current VRAM/CPU memory state."""
        try:
            from apps.backend.runtime import memory_management as _mm  # type: ignore

            snap = _mm.memory_snapshot()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"memory snapshot failed: {exc}")

        probe = snap.get("probe", {}) or {}
        totals = snap.get("totals", {}) or {}
        torch_stats = snap.get("torch", {}) or {}

        total_vram_mb = probe.get("total_vram_mb") or 0
        try:
            total_vram_mb = int(total_vram_mb)
        except Exception:
            total_vram_mb = 0

        return {
            "device_backend": snap.get("device_backend"),
            "primary_device": snap.get("primary_device"),
            "total_vram_mb": total_vram_mb,
            "probe": probe,
            "budgets": snap.get("budgets", {}),
            "torch": torch_stats,
            "models": snap.get("models", []),
            "totals": totals,
        }

    return router
