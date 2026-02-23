"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared explicit device selection helpers for API payloads (fail loud).
Resolves the configured runtime main device, applies payload validation against that invariant, and applies the
device switch only when the task actually starts running (single-flight-safe).

Symbols (top-level; keep in sync; no ghosts):
- `_cuda_available_for_fallback` (function): Returns whether CUDA is available for default main-device fallback.
- `configured_main_device` (function): Resolves the active configured main device (`args`/env/fallback).
- `parse_device_from_payload` (function): Validates payload device and enforces main-device invariant.
- `apply_primary_device` (function): Applies the validated device via `memory_management.switch_primary_device`.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


_ALLOWED_DEVICES = {"cpu", "cuda", "mps", "xpu", "directml"}


def _cuda_available_for_fallback() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False


def configured_main_device() -> str:
    from apps.backend.infra.config import args as runtime_args

    candidates: list[str | None] = [
        getattr(runtime_args.args, "codex_main_device", None),
        getattr(runtime_args.args, "codex_core_device", None),
        os.getenv("CODEX_MAIN_DEVICE"),
        os.getenv("CODEX_CORE_DEVICE"),
    ]
    for raw in candidates:
        normalized = str(raw or "").strip().lower()
        if not normalized:
            continue
        if normalized == "auto":
            return "cuda" if _cuda_available_for_fallback() else "cpu"
        if normalized in _ALLOWED_DEVICES:
            return normalized
    return "cuda" if _cuda_available_for_fallback() else "cpu"


def parse_device_from_payload(payload: Mapping[str, Any]) -> str:
    main = configured_main_device()
    raw = (
        payload.get("codex_device")
        or payload.get("device")
        or payload.get("codex_diffusion_device")
        or ""
    )
    dev = str(raw).strip().lower()
    if not dev:
        return main
    if dev not in _ALLOWED_DEVICES:
        allowed = "|".join(sorted(_ALLOWED_DEVICES))
        raise ValueError(f"Invalid device (allowed: {allowed})")
    if dev != main:
        raise ValueError(
            f"Device override '{dev}' diverges from configured main device '{main}'. "
            "Set launcher main device and keep payload device aligned."
        )
    return dev


def apply_primary_device(device: str) -> None:
    from apps.backend.runtime.memory import memory_management as mem_management

    mem_management.switch_primary_device(str(device).strip().lower())


__all__ = ["configured_main_device", "parse_device_from_payload", "apply_primary_device"]
