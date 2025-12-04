from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

_PATHS_CACHE: Dict[str, List[str]] | None = None
_PATHS_MTIME: float | None = None


def _repo_root() -> Path:
    # apps/backend/infra/config/paths.py -> repo_root = parents[4]
    return Path(__file__).resolve().parents[4]


def _paths_json_path() -> Path:
    return _repo_root() / "apps" / "paths.json"


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
    except Exception:
        raw = {}

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

    _PATHS_CACHE = cfg
    _PATHS_MTIME = stat.st_mtime
    return cfg


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
            norm = os.path.expanduser(v)
        else:
            norm = os.path.join(str(root), v)
        if norm not in out:
            out.append(norm)
    return out


__all__ = ["get_paths_config", "get_paths_for"]
