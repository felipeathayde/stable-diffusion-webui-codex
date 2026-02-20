"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Load and normalize the `apps/paths.json` backend paths config.
Provides cached accessors for model asset roots (checkpoints/text encoders/VAEs/LoRAs) and expands repo-relative paths into absolute paths.
Also provides roots for global modules such as upscalers.
Known per-family model keys include SD1.5, SDXL, Flux, Anima, WAN22, and ZImage roots.

Symbols (top-level; keep in sync; no ghosts):
- `_MODEL_DIR_KEYS` (constant): Keys in `apps/paths.json` whose missing repo-relative directories are created best-effort.
- `_repo_root` (function): Returns repo root (delegates to `get_repo_root()`).
- `_paths_json_path` (function): Returns the absolute path to `apps/paths.json` under the repo root.
- `_load_paths_config` (function): Loads and caches the `paths.json` mapping (and triggers best-effort directory provisioning).
- `_resolve_repo_relative_path` (function): Resolves and validates repo-relative entries with root-containment checks.
- `_ensure_model_dirs` (function): Creates missing model directories for known keys when entries are repo-relative.
- `get_paths_config` (function): Returns a shallow copy of the raw `paths.json` mapping.
- `get_paths_for` (function): Returns a normalized list of filesystem paths for a given logical key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List
import logging

from .repo_root import get_repo_root

_PATHS_CACHE: Dict[str, List[str]] | None = None
_PATHS_MTIME: float | None = None
_LOG = logging.getLogger("backend.infra.config.paths")


_MODEL_DIR_KEYS: tuple[str, ...] = (
    # SD 1.5
    "sd15_ckpt",
    "sd15_tenc",
    "sd15_vae",
    "sd15_loras",
    # SDXL
    "sdxl_ckpt",
    "sdxl_tenc",
    "sdxl_vae",
    "sdxl_loras",
    # Flux
    "flux1_ckpt",
    "flux1_tenc",
    "flux1_vae",
    "flux1_loras",
    # Anima
    "anima_ckpt",
    "anima_tenc",
    "anima_vae",
    "anima_loras",
    # WAN22
    "wan22_ckpt",
    "wan22_tenc",
    "wan22_vae",
    "wan22_loras",
    # Z Image
    "zimage_ckpt",
    "zimage_tenc",
    "zimage_vae",
    "zimage_loras",
    # SUPIR
    "supir_models",
    # Upscalers (standalone + hires-fix)
    "upscale_models",
    "latent_upscale_models",
)


def _repo_root() -> Path:
    return get_repo_root()


def _paths_json_path() -> Path:
    return _repo_root() / "apps" / "paths.json"


def _resolve_repo_relative_path(*, root: Path, key: str, entry: str) -> Path:
    value = entry.strip()
    if not value:
        raise RuntimeError(f"paths.json entry for key {key!r} must not be empty.")

    normalized = value.replace("\\", "/")
    normalized_parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if ".." in normalized_parts:
        raise RuntimeError(
            f"paths.json entry for key {key!r} escapes repository root via parent traversal: {entry!r}."
        )

    candidate = (root / value).resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"paths.json entry for key {key!r} resolves outside repository root: {entry!r} -> {candidate}."
        ) from exc
    return candidate


def _load_paths_config() -> Dict[str, List[str]]:
    global _PATHS_CACHE, _PATHS_MTIME

    path = _paths_json_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        _PATHS_CACHE = {}
        _PATHS_MTIME = None
        return {}

    if _PATHS_CACHE is not None and _PATHS_MTIME == stat.st_mtime:
        return _PATHS_CACHE

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"paths.json is invalid JSON at {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read paths.json at {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load paths.json at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"paths.json root must be an object at {path} (got {type(raw).__name__}).")

    cfg: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        items: List[str] = []
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    v = entry.strip()
                    if v:
                        items.append(v)
        cfg[key] = items

    _ensure_model_dirs(cfg)

    _PATHS_CACHE = cfg
    _PATHS_MTIME = stat.st_mtime
    return cfg


def _ensure_model_dirs(cfg: Dict[str, List[str]]) -> None:
    """Create default model directories for known keys when missing.

    This is intentionally conservative:
      - only keys in _MODEL_DIR_KEYS are considered;
      - only relative entries are created (joined against the repo root);
      - failures raise with key/path context.
    """
    root = _repo_root()
    for key in _MODEL_DIR_KEYS:
        values = cfg.get(key) or []
        for entry in values:
            v = entry.strip()
            if not v:
                continue
            # Only provision directories for repo-relative paths; absolute
            # paths may live on separate volumes or require manual setup.
            if os.path.isabs(v):
                continue
            path = _resolve_repo_relative_path(root=root, key=key, entry=v)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(f"Failed to create model directory for key {key!r} at {path}: {exc}") from exc


def get_paths_config() -> Dict[str, List[str]]:
    """Return a shallow copy of the current paths.json mapping.

    The mapping is {key: [relative_or_absolute_paths...]}. Callers should
    treat the result as read-only.
    """
    cfg = _load_paths_config()
    return {k: list(v) for k, v in cfg.items()}


def get_paths_for(key: str) -> List[str]:
    """Return normalized filesystem paths for a given logical key.

    Semantics:
    - Values from paths.json are treated as repo-root-relative when not absolute.
    - Non-existent paths are not filtered here; callers are expected to check.
    - Keys are treated literally (e.g. 'sd15_ckpt', 'wan22_vae'); no implicit aliasing.
    """
    cfg = _load_paths_config()
    values: List[str] = list(cfg.get(key) or [])
    root = _repo_root()
    out: List[str] = []
    for entry in values:
        v = entry.strip()
        if not v:
            continue
        if os.path.isabs(v):
            norm = Path(os.path.expanduser(v)).resolve(strict=False)
        else:
            norm = _resolve_repo_relative_path(root=root, key=key, entry=v)
        norm_str = str(norm)
        if norm_str not in out:
            out.append(norm_str)
    return out


__all__ = ["get_paths_config", "get_paths_for"]
