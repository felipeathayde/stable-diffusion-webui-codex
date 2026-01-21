"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Path normalization helpers for API payloads and responses.
Keeps API-facing paths stable (repo-relative when possible) and resolves payload paths back to absolute paths under CODEX_ROOT.

Symbols (top-level; keep in sync; no ghosts):
- `CODEX_ROOT` (constant): Repo root used for path normalization.
- `_path_for_api` (function): Normalizes filesystem paths for API responses (prefer repo-relative under CODEX_ROOT).
- `_normalize_inventory_for_api` (function): Applies `_path_for_api` to inventory items before returning them to the UI.
- `_path_from_api` (function): Resolves repo-relative paths from API payloads back to absolute paths under CODEX_ROOT.
- `_normalize_wan_stage_payload` (function): Normalizes WAN stage override payload fields (`model_dir`) to absolute paths; rejects stage `lora_path`.
"""

from __future__ import annotations

import os
from pathlib import Path

from apps.backend.infra.config.repo_root import get_repo_root

CODEX_ROOT = get_repo_root()


def _path_for_api(raw: object) -> str:
    """Return a stable, repo-relative path for API responses when possible.

    - Paths under CODEX_ROOT are returned relative (POSIX separators).
    - External absolute paths are returned as-is (POSIX separators).
    """
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        p = Path(os.path.expanduser(value))
    except Exception:
        return value.replace("\\", "/")

    # Keep relative paths relative (normalize slashes).
    if not p.is_absolute():
        return p.as_posix()

    try:
        root = CODEX_ROOT.resolve()
    except Exception:
        root = CODEX_ROOT
    try:
        resolved = p.resolve(strict=False)
    except Exception:
        resolved = p
    try:
        rel = resolved.relative_to(root)
    except Exception:
        return resolved.as_posix()
    return rel.as_posix()


def _normalize_inventory_for_api(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, object]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        entry: dict[str, object] = dict(it)
        if "path" in entry:
            entry["path"] = _path_for_api(entry.get("path"))
        out.append(entry)
    return out


def _path_from_api(raw: object) -> str:
    """Resolve a user-supplied path relative to CODEX_ROOT when not absolute."""
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        p = Path(os.path.expanduser(value))
    except Exception:
        return value
    if p.is_absolute():
        return str(p)
    return str(CODEX_ROOT / p)


def _normalize_wan_stage_payload(raw: object) -> object:
    """Normalize WAN stage override payloads.

    - `model_dir`: repo-relative path → absolute path under CODEX_ROOT.
    - `lora_path`: rejected (use `lora_sha`).
    """
    if not isinstance(raw, dict):
        return raw
    out: dict[str, object] = dict(raw)
    if isinstance(out.get("model_dir"), str):
        out["model_dir"] = _path_from_api(out.get("model_dir"))
    if isinstance(out.get("lora_path"), str) and str(out.get("lora_path")).strip():
        raise ValueError("WAN stage 'lora_path' is unsupported; use 'lora_sha' instead.")
    return out
