"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Truthful LTX 2.3 execution-profile and checkpoint-default resolution.
Classifies discoverable LTX checkpoints from explicit local signals, emits the
checkpoint-scoped metadata forwarded by `/api/models`, and defines the single
engine-scoped execution surface exposed by `/api/engines/capabilities`.

Symbols (top-level; keep in sync; no ghosts):
- `LTX2_KIND_DEV` (constant): Classified generatable dev/full checkpoint kind.
- `LTX2_KIND_DISTILLED` (constant): Classified generatable distilled checkpoint kind.
- `LTX2_KIND_UNKNOWN` (constant): Unclassified or non-generatable checkpoint kind.
- `LTX2_PROFILE_ONE_STAGE` (constant): Execution profile id for the current one-stage dev lane.
- `LTX2_PROFILE_DISTILLED` (constant): Execution profile id for the current distilled lane.
- `LTX2_EXECUTION_SURFACE_KEY` (constant): `/api/engines/capabilities` key for nested LTX execution metadata.
- `Ltx2CheckpointExecutionDefaults` (dataclass): Checkpoint-scoped classification + defaults.
- `Ltx2ExecutionSurface` (dataclass): Engine-scoped LTX execution-profile/default surface.
- `resolve_ltx2_checkpoint_execution_defaults` (function): Classify one checkpoint and return its executable/default profile contract.
- `build_ltx2_checkpoint_metadata` (function): Build namespaced LTX metadata forwarded by `/api/models`.
- `build_ltx2_execution_surface` (function): Build the engine-scoped LTX execution metadata for `/api/engines/capabilities`.
- `debug_probe` (function): Return a compact classification/default table for dry validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from apps.backend.runtime.models.types import CheckpointRecord

LTX2_KIND_DEV = "dev"
LTX2_KIND_DISTILLED = "distilled"
LTX2_KIND_UNKNOWN = "unknown"

LTX2_PROFILE_ONE_STAGE = "one_stage"
LTX2_PROFILE_DISTILLED = "distilled"

LTX2_EXECUTION_SURFACE_KEY = "ltx_execution_surface"

_LTX2_METADATA_KIND_KEY = "ltx_checkpoint_kind"
_LTX2_METADATA_ALLOWED_PROFILES_KEY = "ltx_allowed_execution_profiles"
_LTX2_METADATA_DEFAULT_PROFILE_KEY = "ltx_default_execution_profile"
_LTX2_METADATA_DEFAULT_STEPS_KEY = "ltx_default_steps"
_LTX2_METADATA_DEFAULT_GUIDANCE_KEY = "ltx_default_guidance_scale"

_BLOCKED_MARKERS = (
    "distilled-lora",
    "distilled_lora",
    "spatial-upscaler",
    "spatial_upscaler",
    "temporal-upscaler",
    "temporal_upscaler",
    "upscaler",
    "lora",
)
_DISTILLED_MARKERS = ("distilled",)
_LTX_IDENTITY_MARKERS = ("ltx-2.3", "ltx2.3", "ltx-2")


@dataclass(frozen=True)
class Ltx2CheckpointExecutionDefaults:
    checkpoint_kind: str
    allowed_execution_profiles: tuple[str, ...]
    default_execution_profile: str | None
    default_steps: int | None
    default_guidance_scale: float | None


@dataclass(frozen=True)
class Ltx2ExecutionSurface:
    allowed_execution_profiles: tuple[str, ...]
    default_execution_profile: str
    default_steps_by_profile: Mapping[str, int]
    default_guidance_scale_by_profile: Mapping[str, float]


def _normalize_marker_input(raw_value: object) -> str:
    return str(raw_value or "").strip().lower().replace("\\", "/")


def _basename_or_self(raw_value: object) -> str:
    normalized = _normalize_marker_input(raw_value)
    if not normalized:
        return ""
    return PurePosixPath(normalized).name or normalized


def _candidate_strings(record: CheckpointRecord) -> tuple[str, ...]:
    candidates: list[str] = []
    raw_values = (
        record.title,
        record.name,
        record.model_name,
        record.family_hint,
    )
    for raw_value in raw_values:
        normalized = _normalize_marker_input(raw_value)
        if normalized:
            candidates.append(normalized)
    filename_candidate = _basename_or_self(record.filename)
    if filename_candidate:
        candidates.append(filename_candidate)
    for raw_key in (
        "repo_hint",
        "repo_id",
        "source_checkpoint_repo_id",
        "_name_or_path",
    ):
        normalized = _basename_or_self(record.metadata.get(raw_key))
        if normalized:
            candidates.append(normalized)
    return tuple(dict.fromkeys(candidates))


def _blocked_detection_candidates(record: CheckpointRecord) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_value in (
        record.title,
        record.name,
        record.model_name,
        record.filename,
    ):
        normalized = _basename_or_self(raw_value)
        if normalized:
            candidates.append(normalized)
    for raw_key in (
        "repo_hint",
        "repo_id",
        "source_checkpoint_repo_id",
        "_name_or_path",
    ):
        normalized = _basename_or_self(record.metadata.get(raw_key))
        if normalized:
            candidates.append(normalized)
    return tuple(dict.fromkeys(candidates))


def _path_component_markers(record: CheckpointRecord) -> tuple[str, ...]:
    components: list[str] = []
    for raw_value in (record.filename, record.path):
        normalized = _normalize_marker_input(raw_value)
        if not normalized:
            continue
        for part in PurePosixPath(normalized).parts:
            candidate = str(part or "").strip().lower()
            if candidate and candidate not in {"/", "."}:
                components.append(candidate)
    return tuple(dict.fromkeys(components))


def resolve_ltx2_checkpoint_execution_defaults(record: CheckpointRecord) -> Ltx2CheckpointExecutionDefaults:
    candidates = _candidate_strings(record)
    has_ltx_identity = any(marker in candidate for candidate in candidates for marker in _LTX_IDENTITY_MARKERS)
    blocked_candidates = _blocked_detection_candidates(record)
    blocked_path_components = _path_component_markers(record)

    if (
        any(marker in candidate for candidate in blocked_candidates for marker in _BLOCKED_MARKERS)
        or any(component in _BLOCKED_MARKERS for component in blocked_path_components)
    ):
        return Ltx2CheckpointExecutionDefaults(
            checkpoint_kind=LTX2_KIND_UNKNOWN,
            allowed_execution_profiles=(),
            default_execution_profile=None,
            default_steps=None,
            default_guidance_scale=None,
        )

    if has_ltx_identity and any(marker in candidate for candidate in candidates for marker in _DISTILLED_MARKERS):
        return Ltx2CheckpointExecutionDefaults(
            checkpoint_kind=LTX2_KIND_DISTILLED,
            allowed_execution_profiles=(LTX2_PROFILE_DISTILLED,),
            default_execution_profile=LTX2_PROFILE_DISTILLED,
            default_steps=8,
            default_guidance_scale=1.0,
        )

    if has_ltx_identity:
        return Ltx2CheckpointExecutionDefaults(
            checkpoint_kind=LTX2_KIND_DEV,
            allowed_execution_profiles=(LTX2_PROFILE_ONE_STAGE,),
            default_execution_profile=LTX2_PROFILE_ONE_STAGE,
            default_steps=30,
            default_guidance_scale=4.0,
        )

    return Ltx2CheckpointExecutionDefaults(
        checkpoint_kind=LTX2_KIND_UNKNOWN,
        allowed_execution_profiles=(),
        default_execution_profile=None,
        default_steps=None,
        default_guidance_scale=None,
    )


def build_ltx2_checkpoint_metadata(record: CheckpointRecord) -> dict[str, object]:
    defaults = resolve_ltx2_checkpoint_execution_defaults(record)
    return {
        _LTX2_METADATA_KIND_KEY: defaults.checkpoint_kind,
        _LTX2_METADATA_ALLOWED_PROFILES_KEY: list(defaults.allowed_execution_profiles),
        _LTX2_METADATA_DEFAULT_PROFILE_KEY: defaults.default_execution_profile,
        _LTX2_METADATA_DEFAULT_STEPS_KEY: defaults.default_steps,
        _LTX2_METADATA_DEFAULT_GUIDANCE_KEY: defaults.default_guidance_scale,
    }


def build_ltx2_execution_surface() -> Ltx2ExecutionSurface:
    return Ltx2ExecutionSurface(
        allowed_execution_profiles=(LTX2_PROFILE_ONE_STAGE, LTX2_PROFILE_DISTILLED),
        default_execution_profile=LTX2_PROFILE_ONE_STAGE,
        default_steps_by_profile={
            LTX2_PROFILE_ONE_STAGE: 30,
            LTX2_PROFILE_DISTILLED: 8,
        },
        default_guidance_scale_by_profile={
            LTX2_PROFILE_ONE_STAGE: 4.0,
            LTX2_PROFILE_DISTILLED: 1.0,
        },
    )


def debug_probe() -> dict[str, Any]:
    cases = (
        CheckpointRecord(
            name="ltx-2.3-22b-dev",
            title="ltx-2.3-22b-dev.safetensors",
            filename="/models/ltx2/ltx-2.3-22b-dev.safetensors",
            path="/models/ltx2",
            model_name="ltx-2.3-22b-dev",
            format="checkpoint",  # type: ignore[arg-type]
            family_hint="ltx2",
        ),
        CheckpointRecord(
            name="ltx-2.3-22b-distilled",
            title="ltx-2.3-22b-distilled.safetensors",
            filename="/models/ltx2/ltx-2.3-22b-distilled.safetensors",
            path="/models/ltx2",
            model_name="ltx-2.3-22b-distilled",
            format="checkpoint",  # type: ignore[arg-type]
            family_hint="ltx2",
        ),
        CheckpointRecord(
            name="ltx-2.3-spatial-upscaler-x2-1.1",
            title="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            filename="/models/ltx2/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            path="/models/ltx2",
            model_name="ltx-2.3-spatial-upscaler-x2-1.1",
            format="checkpoint",  # type: ignore[arg-type]
            family_hint="ltx2",
        ),
    )
    payload: dict[str, Any] = {}
    for record in cases:
        defaults = resolve_ltx2_checkpoint_execution_defaults(record)
        payload[record.name] = {
            "checkpoint_kind": defaults.checkpoint_kind,
            "allowed_execution_profiles": list(defaults.allowed_execution_profiles),
            "default_execution_profile": defaults.default_execution_profile,
            "default_steps": defaults.default_steps,
            "default_guidance_scale": defaults.default_guidance_scale,
        }
    return payload


__all__ = [
    "LTX2_EXECUTION_SURFACE_KEY",
    "LTX2_KIND_DEV",
    "LTX2_KIND_DISTILLED",
    "LTX2_KIND_UNKNOWN",
    "LTX2_PROFILE_DISTILLED",
    "LTX2_PROFILE_ONE_STAGE",
    "Ltx2CheckpointExecutionDefaults",
    "Ltx2ExecutionSurface",
    "build_ltx2_checkpoint_metadata",
    "build_ltx2_execution_surface",
    "debug_probe",
    "resolve_ltx2_checkpoint_execution_defaults",
]
