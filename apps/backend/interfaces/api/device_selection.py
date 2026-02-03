"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared explicit device selection helpers for API payloads (fail loud).
Parses and validates the request device without relying on options fallback, and applies the device switch only when the
task actually starts running (single-flight-safe).

Symbols (top-level; keep in sync; no ghosts):
- `parse_device_from_payload` (function): Extracts and validates a device string from a JSON payload.
- `apply_primary_device` (function): Applies the validated device via `memory_management.switch_primary_device`.
"""

from __future__ import annotations

from typing import Any, Mapping


_ALLOWED_DEVICES = {"cpu", "cuda", "mps", "xpu", "directml"}


def parse_device_from_payload(payload: Mapping[str, Any]) -> str:
    raw = (
        payload.get("codex_device")
        or payload.get("device")
        or payload.get("codex_diffusion_device")
        or ""
    )
    dev = str(raw).strip().lower()
    if dev not in _ALLOWED_DEVICES:
        allowed = "|".join(sorted(_ALLOWED_DEVICES))
        raise ValueError(f"Missing or invalid device (allowed: {allowed})")
    return dev


def apply_primary_device(device: str) -> None:
    from apps.backend.runtime.memory import memory_management as mem_management

    mem_management.switch_primary_device(str(device).strip().lower())


__all__ = ["parse_device_from_payload", "apply_primary_device"]

