"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared primitives for asset registries (dataclasses + directory helpers).
Defines the base `AssetEntry` record plus small filesystem helpers used across registry modules.

Symbols (top-level; keep in sync; no ghosts):
- `AssetEntry` (dataclass): Generic asset record (name/path/kind/tags/meta) used by registry inventory endpoints.
- `_is_dir_with_any` (function): Returns True if a root directory contains any of the given subdirectories.
- `_iter_dirs` (function): Iterates direct child directories of a root path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List


@dataclass
class AssetEntry:
    name: str
    path: str
    kind: str
    tags: List[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _is_dir_with_any(root: str, subdirs: Iterable[str]) -> bool:
    try:
        for sd in subdirs:
            if os.path.isdir(os.path.join(root, sd)):
                return True
    except Exception:
        return False
    return False


def _iter_dirs(root: str) -> Iterable[str]:
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                yield full
    except Exception:
        return []


__all__ = ["AssetEntry", "_is_dir_with_any", "_iter_dirs"]
