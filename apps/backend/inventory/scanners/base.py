"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for filesystem-based asset scanners.
Provides CODEX_ROOT-anchored defaults, safe directory walking, and stable-order deduplication for inventory scanners.

Symbols (top-level; keep in sync; no ghosts):
- `default_models_root` (function): Returns the default `models/` directory under `CODEX_ROOT`.
- `iter_files` (function): Recursively yields files under roots matching a set of extensions (stable order).
- `dedupe_keep_order` (function): Deduplicates a list while preserving first-seen order.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Sequence, TypeVar

from apps.backend.infra.config.repo_root import get_repo_root

T = TypeVar("T")


def default_models_root() -> str:
    return str(get_repo_root() / "models")


def iter_files(roots: Sequence[str], *, exts: Iterable[str]) -> Iterator[str]:
    exts_lc = tuple(str(e).lower() for e in exts)
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort(key=lambda s: s.lower())
                for fn in sorted(filenames, key=lambda s: s.lower()):
                    if not fn.lower().endswith(exts_lc):
                        continue
                    yield os.path.join(dirpath, fn)
        except Exception:
            continue


def dedupe_keep_order(items: Sequence[T]) -> list[T]:
    out: list[T] = []
    seen: set[T] = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


__all__ = ["default_models_root", "dedupe_keep_order", "iter_files"]
