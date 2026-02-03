"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: HF upscalers manifest parsing and validation (import-light).
Defines the canonical schema for `upscalers/manifest.json` and provides a strict-but-non-blocking validator:
invalid manifest entries are skipped (raw listing still works), while validation issues are returned explicitly so the UI can surface them.

Symbols (top-level; keep in sync; no ghosts):
- `UpscalersManifestValidationResult` (dataclass): Result container (normalized manifest, index, errors).
- `validate_upscalers_manifest` (function): Validate/normalize a decoded manifest JSON object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse


_SCHEMA_VERSION_V1 = 1
_MANIFEST_SUPPORTED_SUFFIXES = (".safetensors", ".pt", ".pth")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_hf_path(raw: object) -> str:
    if not _is_non_empty_str(raw):
        raise ValueError("must be a non-empty string")
    s = str(raw).replace("\\", "/").lstrip("/")
    if ".." in s.split("/"):
        raise ValueError("must not contain '..' path segments")
    if not s.startswith("upscalers/"):
        raise ValueError("must start with 'upscalers/'")
    return s


def _validate_http_url(raw: object) -> str:
    if not _is_non_empty_str(raw):
        raise ValueError("must be a non-empty string")
    url = str(raw).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("must be an http(s) URL")
    return url


def _validate_sha256(raw: object) -> str:
    if not _is_non_empty_str(raw):
        raise ValueError("must be a non-empty string")
    s = str(raw).strip()
    if not _SHA256_RE.match(s):
        raise ValueError("must be a 64-hex sha256 string")
    return s.lower()


def _validate_scale(raw: object) -> int:
    if not isinstance(raw, int):
        raise ValueError("must be an integer")
    value = int(raw)
    if value < 1:
        raise ValueError("must be >= 1")
    return value


def _validate_tags(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("must be an array of strings")
    tags: list[str] = []
    for item in raw:
        if not _is_non_empty_str(item):
            raise ValueError("must be an array of non-empty strings")
        tags.append(str(item).strip())
    # Stable unique ordering.
    uniq = sorted(set(tags), key=lambda s: s.lower())
    return tuple(uniq)


@dataclass(frozen=True, slots=True)
class UpscalersManifestValidationResult:
    schema_version: int | None
    manifest: Dict[str, Any] | None
    weights_by_hf_path: Dict[str, Dict[str, Any]]
    errors: List[str]


def validate_upscalers_manifest(raw: object) -> UpscalersManifestValidationResult:
    """Validate and normalize a decoded `upscalers/manifest.json` object.

    Returns a normalized `manifest` object (or `None` when the schema is not usable) plus:
    - `weights_by_hf_path`: index of valid entries by `hf_path`
    - `errors`: validation issues (non-empty means the UI should warn)
    """

    errors: list[str] = []

    if not isinstance(raw, dict):
        return UpscalersManifestValidationResult(
            schema_version=None,
            manifest=None,
            weights_by_hf_path={},
            errors=[f"manifest must be a JSON object; got: {type(raw).__name__}"],
        )

    allowed_top_keys = {"schema_version", "weights"}
    unknown_top_keys = sorted([k for k in raw.keys() if k not in allowed_top_keys], key=str)
    for key in unknown_top_keys:
        errors.append(f"manifest has unknown top-level key: {key!r}")

    schema_version = raw.get("schema_version")
    if schema_version != _SCHEMA_VERSION_V1:
        return UpscalersManifestValidationResult(
            schema_version=schema_version if isinstance(schema_version, int) else None,
            manifest=None,
            weights_by_hf_path={},
            errors=errors + [f"manifest.schema_version must be {_SCHEMA_VERSION_V1}; got: {schema_version!r}"],
        )

    weights = raw.get("weights")
    if not isinstance(weights, list):
        return UpscalersManifestValidationResult(
            schema_version=_SCHEMA_VERSION_V1,
            manifest=None,
            weights_by_hf_path={},
            errors=errors + [f"manifest.weights must be an array; got: {type(weights).__name__}"],
        )

    allowed_entry_keys = {
        "id",
        "hf_path",
        "label",
        "arch",
        "scale",
        "license_name",
        "license_url",
        "license_spdx",
        "sha256",
        "tags",
        "notes",
    }

    seen_ids: set[str] = set()
    seen_hf_paths: set[str] = set()
    out_weights: list[dict[str, Any]] = []
    weights_by_hf_path: dict[str, dict[str, Any]] = {}

    for idx, entry in enumerate(weights):
        prefix = f"manifest.weights[{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object; got: {type(entry).__name__}")
            continue

        unknown_entry_keys = sorted([k for k in entry.keys() if k not in allowed_entry_keys], key=str)
        for key in unknown_entry_keys:
            errors.append(f"{prefix} has unknown key: {key!r}")

        try:
            weight_id = entry.get("id")
            if not _is_non_empty_str(weight_id):
                raise ValueError(f"{prefix}.id must be a non-empty string; got: {weight_id!r}")
            weight_id = str(weight_id).strip()
            if weight_id in seen_ids:
                raise ValueError(f"{prefix}.id must be unique; duplicate: {weight_id!r}")

            try:
                hf_path = _safe_hf_path(entry.get("hf_path"))
            except Exception as exc:
                raise ValueError(f"{prefix}.hf_path {exc}") from exc
            if hf_path in seen_hf_paths:
                raise ValueError(f"{prefix}.hf_path must be unique; duplicate: {hf_path!r}")
            suffix = "." + hf_path.rsplit(".", 1)[-1].lower() if "." in hf_path else ""
            if suffix not in _MANIFEST_SUPPORTED_SUFFIXES:
                allowed = "|".join(_MANIFEST_SUPPORTED_SUFFIXES)
                raise ValueError(f"{prefix}.hf_path must end with {allowed}; got: {hf_path!r}")

            label = entry.get("label")
            if not _is_non_empty_str(label):
                raise ValueError(f"{prefix}.label must be a non-empty string; got: {label!r}")

            arch = entry.get("arch")
            if not _is_non_empty_str(arch):
                raise ValueError(f"{prefix}.arch must be a non-empty string; got: {arch!r}")

            try:
                scale = _validate_scale(entry.get("scale"))
            except Exception as exc:
                raise ValueError(f"{prefix}.scale {exc}") from exc

            license_name = entry.get("license_name")
            if not _is_non_empty_str(license_name):
                raise ValueError(f"{prefix}.license_name must be a non-empty string; got: {license_name!r}")
            try:
                license_url = _validate_http_url(entry.get("license_url"))
            except Exception as exc:
                raise ValueError(f"{prefix}.license_url {exc}") from exc

            try:
                sha256 = _validate_sha256(entry.get("sha256"))
            except Exception as exc:
                raise ValueError(f"{prefix}.sha256 {exc}") from exc

            license_spdx_raw = entry.get("license_spdx")
            license_spdx = str(license_spdx_raw).strip() if _is_non_empty_str(license_spdx_raw) else None

            notes_raw = entry.get("notes")
            notes = str(notes_raw).strip() if _is_non_empty_str(notes_raw) else None

            try:
                tags = _validate_tags(entry.get("tags"))
            except Exception as exc:
                raise ValueError(f"{prefix}.tags {exc}") from exc

            normalized = {
                "id": weight_id,
                "hf_path": hf_path,
                "label": str(label).strip(),
                "arch": str(arch).strip(),
                "scale": scale,
                "license_name": str(license_name).strip(),
                "license_url": license_url,
                "license_spdx": license_spdx,
                "sha256": sha256,
                "tags": list(tags),
                "notes": notes,
            }

            seen_ids.add(weight_id)
            seen_hf_paths.add(hf_path)
            out_weights.append(normalized)
            weights_by_hf_path[hf_path] = dict(normalized)
        except Exception as exc:
            errors.append(str(exc))
            continue

    normalized_manifest: dict[str, Any] = {"schema_version": _SCHEMA_VERSION_V1, "weights": out_weights}
    return UpscalersManifestValidationResult(
        schema_version=_SCHEMA_VERSION_V1,
        manifest=normalized_manifest,
        weights_by_hf_path=weights_by_hf_path,
        errors=errors,
    )


__all__ = ["UpscalersManifestValidationResult", "validate_upscalers_manifest"]
