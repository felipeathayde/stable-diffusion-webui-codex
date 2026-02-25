"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Options API routes for reading, updating, and validating settings.
Exposes the JSON-backed options store and registry-driven validation helpers, applies supported runtime memory overrides immediately
(device backend + storage/compute dtype) via the memory manager, enforces finite numeric values for number/slider settings,
and emits apply metadata on `POST /api/options`.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for options endpoints.
"""

from __future__ import annotations

import math
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
        try:
            revision = int(getattr(opts_snapshot(), "codex_options_revision", 0) or 0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read options revision: {exc}") from exc
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
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to read registry defaults: {exc}") from exc
        try:
            snap = opts_snapshot().as_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read options snapshot: {exc}") from exc
        return {"defaults": defaults, "snapshot": snap}

    def _parse_checkbox_value(key: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value in (0, 1):
                return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off"):
                return False
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid value for {key}: expected bool or one of "
                f"('true','false','1','0','yes','no','on','off')."
            ),
        )

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
                    if isinstance(v, bool):
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: boolean is not a numeric value")
                    num = float(v)
                    if not math.isfinite(num):
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: must be finite")
                    lo = getattr(f, "min", None)
                    hi = getattr(f, "max", None)
                    if isinstance(lo, (int, float)) and num < lo:
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: below min {lo}")
                    if isinstance(hi, (int, float)) and num > hi:
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: above max {hi}")
                    out[k] = num
                elif getattr(f, "type", None) == setting_type.CHECKBOX:
                    out[k] = _parse_checkbox_value(k, v)
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
        previous_values = opts_load_native()
        if not isinstance(previous_values, dict):
            raise HTTPException(status_code=500, detail="failed to read current options before update")
        hot_applied_keys: set[str] = set()
        device_keys = ("codex_core_device", "codex_te_device", "codex_vae_device")

        # Apply memory manager overrides when present
        from apps.backend.runtime import memory_management as mem_management

        requested_devices = {
            key: str(updates[key]).strip().lower()
            for key in device_keys
            if key in updates
        }
        if requested_devices:
            distinct = {value for value in requested_devices.values()}
            if len(distinct) != 1:
                joined = ", ".join(f"{key}={value!r}" for key, value in requested_devices.items())
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Main-device invariant violation: core/TE/VAE device updates must match exactly. "
                        f"Received: {joined}"
                    ),
                )
            resolved = next(iter(distinct))
            for key in device_keys:
                updates[key] = resolved

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

        def _apply_runtime_override(key: str, value: Any) -> bool:
            if key == "codex_attention_backend":
                mem_management.set_attention_backend(str(value))
                return True
            if key not in role_map:
                return False
            role, kind = role_map[key]
            if kind == "backend":
                mem_management.set_component_backend(role, str(value))
            elif kind == "dtype":
                mem_management.set_component_dtype(role, str(value))
            else:
                mem_management.set_component_compute_dtype(role, str(value))
            return True

        for key, value in updates.items():
            try:
                applied = _apply_runtime_override(key, value)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid memory setting for {key}: {exc}")
            if applied:
                hot_applied_keys.add(key)

        try:
            updated = opts_set_many(updates)
        except Exception as exc:
            rollback_failures: list[str] = []
            for key in sorted(hot_applied_keys):
                if key not in previous_values:
                    continue
                try:
                    _apply_runtime_override(key, previous_values[key])
                except Exception as rollback_exc:
                    rollback_failures.append(f"{key}: {rollback_exc}")
            if rollback_failures:
                detail = (
                    "Failed to persist options and failed to rollback runtime hot-applies: "
                    + "; ".join(rollback_failures)
                )
                raise HTTPException(status_code=500, detail=detail) from exc
            raise HTTPException(
                status_code=500,
                detail="Failed to persist options; runtime hot-applies were rolled back.",
            ) from exc
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read updated options revision: {exc}") from exc
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
                    if isinstance(v, bool):
                        rejected[k] = "boolean is not a numeric value"
                        continue
                    num = float(v)
                    if not math.isfinite(num):
                        rejected[k] = "must be finite"
                        continue
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
                    accepted[k] = _parse_checkbox_value(k, v)
                else:
                    accepted[k] = v
            except HTTPException as exc:
                rejected[k] = str(exc.detail)
            except Exception:
                rejected[k] = "invalid value"
        return {"accepted": accepted, "rejected": rejected}

    return router
