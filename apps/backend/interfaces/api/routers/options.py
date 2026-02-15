"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Options API routes for reading, updating, and validating settings.
Exposes the JSON-backed options store and registry-driven validation helpers, applies supported runtime memory overrides immediately
(device backend + storage/compute dtype) via the memory manager, and emits apply metadata on `POST /api/options`.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for options endpoints.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Body, HTTPException


def build_router(
    *,
    opts_load_native: Callable[[], Dict[str, Any]],
    opts_snapshot,
    opts_set_many: Callable[[Dict[str, Any]], list[str]],
    settings_registry_ok: bool,
    field_index: Callable[[], Dict[str, Any]],
    setting_type,
) -> APIRouter:
    router = APIRouter()
    hot_apply_reasons: Dict[str, str] = {
        "codex_attention_backend": "hot-applied immediately (runtime attention backend reconfigured).",
        "codex_core_device": "hot-applied immediately (runtime memory manager backend updated).",
        "codex_te_device": "hot-applied immediately (runtime memory manager backend updated).",
        "codex_vae_device": "hot-applied immediately (runtime memory manager backend updated).",
        "codex_core_dtype": "hot-applied immediately (runtime memory manager storage dtype updated).",
        "codex_te_dtype": "hot-applied immediately (runtime memory manager storage dtype updated).",
        "codex_vae_dtype": "hot-applied immediately (runtime memory manager storage dtype updated).",
        "codex_core_compute_dtype": "hot-applied immediately (runtime memory manager compute dtype updated).",
        "codex_te_compute_dtype": "hot-applied immediately (runtime memory manager compute dtype updated).",
        "codex_vae_compute_dtype": "hot-applied immediately (runtime memory manager compute dtype updated).",
        "codex_smart_offload": "hot-applied immediately (effective for the next generation request).",
        "codex_smart_fallback": "hot-applied immediately (effective for the next generation request).",
        "codex_smart_cache": "hot-applied immediately (effective for the next generation request).",
        "codex_core_streaming": "hot-applied immediately (effective for the next generation request).",
        "codex_export_video": "hot-applied immediately (effective for the next generation request).",
    }

    @router.get("/api/options")
    def get_options() -> Dict[str, Any]:
        revision = 0
        try:
            revision = int(getattr(opts_snapshot(), "codex_options_revision", 0) or 0)
        except Exception:
            revision = 0
        return {"values": opts_load_native(), "revision": max(0, revision)}

    @router.get("/api/options/keys")
    def get_options_keys() -> Dict[str, Any]:
        """List supported option keys and basic metadata from the settings registry."""
        if not settings_registry_ok:
            return {"keys": [], "types": {}, "choices": {}}
        try:
            idx = field_index()
            keys = list(idx.keys())
            types = {}
            choices = {}
            for k, f in idx.items():
                t = getattr(getattr(f, "type", None), "name", None) or str(getattr(f, "type", None))
                types[k] = t
                ch = getattr(f, "choices", None)
                if isinstance(ch, list):
                    choices[k] = ch
            return {"keys": keys, "types": types, "choices": choices}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read registry: {exc}")

    @router.get("/api/options/snapshot")
    def get_options_snapshot() -> Dict[str, Any]:
        """Return a typed snapshot of current options (for UI defaults)."""
        try:
            return {"snapshot": opts_snapshot().as_dict()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read snapshot: {exc}")

    @router.get("/api/options/defaults")
    def get_options_defaults() -> Dict[str, Any]:
        """Return default values from the settings registry and the current snapshot."""
        defaults: Dict[str, Any] = {}
        if settings_registry_ok:
            try:
                idx = field_index()
                for k, f in idx.items():
                    defaults[k] = getattr(f, "default", None)
            except Exception:
                defaults = {}
        try:
            snap = opts_snapshot().as_dict()
        except Exception:
            snap = {}
        return {"defaults": defaults, "snapshot": snap}

    def _validate_options(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        if not settings_registry_ok:
            return dict(payload)
        try:
            idx = field_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"registry unavailable: {exc}")

        unknown = sorted(k for k in payload.keys() if k not in idx)
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown option key(s): {', '.join(unknown)}")

        out: Dict[str, Any] = {}
        for k, v in payload.items():
            f = idx[k]
            try:
                if getattr(f, "choices", None) and isinstance(f.choices, list) and v not in f.choices:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {k}: not in choices")
                if getattr(f, "type", None) in (setting_type.SLIDER, setting_type.NUMBER):
                    num = float(v)
                    lo = getattr(f, "min", None)
                    hi = getattr(f, "max", None)
                    if isinstance(lo, (int, float)) and num < lo:
                        num = lo
                    if isinstance(hi, (int, float)) and num > hi:
                        num = hi
                    out[k] = num
                elif getattr(f, "type", None) == setting_type.CHECKBOX:
                    if isinstance(v, str):
                        out[k] = v.strip().lower() in ("1", "true", "yes", "on")
                    else:
                        out[k] = bool(v)
                else:
                    out[k] = v
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}: {exc}") from exc
        return out

    @router.post("/api/options")
    def set_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        updates = _validate_options(payload)
        hot_applied_keys: set[str] = set()

        # Apply memory manager overrides when present
        from apps.backend.runtime import memory_management as mem_management

        role_map = {
            "codex_core_device": ("core", "backend"),
            "codex_te_device": ("text_encoder", "backend"),
            "codex_vae_device": ("vae", "backend"),
            "codex_core_dtype": ("core", "dtype"),
            "codex_te_dtype": ("text_encoder", "dtype"),
            "codex_vae_dtype": ("vae", "dtype"),
            "codex_core_compute_dtype": ("core", "compute_dtype"),
            "codex_te_compute_dtype": ("text_encoder", "compute_dtype"),
            "codex_vae_compute_dtype": ("vae", "compute_dtype"),
        }
        for key, value in updates.items():
            if key == "codex_attention_backend":
                try:
                    mem_management.set_attention_backend(str(value))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid memory setting for {key}: {exc}")
                hot_applied_keys.add(key)
                continue
            if key not in role_map:
                continue
            role, kind = role_map[key]
            try:
                if kind == "backend":
                    mem_management.set_component_backend(role, str(value))
                elif kind == "dtype":
                    mem_management.set_component_dtype(role, str(value))
                else:
                    mem_management.set_component_compute_dtype(role, str(value))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid memory setting for {key}: {exc}")
            hot_applied_keys.add(key)

        updated = opts_set_many(updates)
        applied_now: List[str] = []
        restart_required: List[str] = []
        for key in updated:
            if key == "codex_options_revision":
                continue
            reason = hot_apply_reasons.get(key)
            if reason is None and key in hot_applied_keys:
                reason = "hot-applied immediately."
            if reason is not None:
                applied_now.append(f"{key}: {reason}")
                continue
            restart_required.append(f"{key}: not hot-applied; restart required.")
        try:
            revision = int(getattr(opts_snapshot(), "codex_options_revision", 0) or 0)
        except Exception:
            revision = 0
        return {
            "updated": updated,
            "revision": max(0, revision),
            "applied_now": applied_now,
            "restart_required": restart_required,
        }

    @router.post("/api/options/validate")
    def validate_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Dry-run options validation; returns accepted and rejected keys with reasons."""
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        if not settings_registry_ok:
            return {"accepted": dict(payload), "rejected": {}}
        try:
            idx = field_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"registry unavailable: {exc}")
        accepted: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        for k, v in payload.items():
            f = idx.get(k)
            if not f:
                rejected[k] = "unknown key"
                continue
            try:
                if getattr(f, "choices", None) and isinstance(f.choices, list) and v not in f.choices:
                    rejected[k] = "not in choices"
                    continue
                if getattr(f, "type", None) in (setting_type.SLIDER, setting_type.NUMBER):
                    num = float(v)
                    lo = getattr(f, "min", None)
                    hi = getattr(f, "max", None)
                    if isinstance(lo, (int, float)) and num < lo:
                        rejected[k] = f"below min {lo}"
                        continue
                    if isinstance(hi, (int, float)) and num > hi:
                        rejected[k] = f"above max {hi}"
                        continue
                    accepted[k] = num
                elif getattr(f, "type", None) == setting_type.CHECKBOX:
                    if isinstance(v, str):
                        accepted[k] = v.strip().lower() in ("1", "true", "yes", "on")
                    else:
                        accepted[k] = bool(v)
                else:
                    accepted[k] = v
            except Exception:
                rejected[k] = "invalid value"
        return {"accepted": accepted, "rejected": rejected}

    return router
