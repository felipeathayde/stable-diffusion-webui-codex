"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF stage discovery policy used by inventories and registries.
Defines strict roots for WAN22 GGUF stage weights (paths.json `wan22_ckpt` or `models/wan22`) and provides a shared stage classifier.

Symbols (top-level; keep in sync; no ghosts):
- `infer_wan22_stage` (function): Heuristically classify a GGUF filename as `high`/`low`/`unknown`.
- `iter_wan22_gguf_files` (function): Yields `.gguf` stage file paths under resolved roots (non-recursive, stable order).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from apps.backend.infra.config.paths import get_paths_for

from .base import default_models_root, dedupe_keep_order


def infer_wan22_stage(filename: str) -> str:
    n = str(filename or "").lower()
    if any(k in n for k in ("high", "highnoise", "high_noise")):
        return "high"
    if any(k in n for k in ("low", "lownoise", "low_noise")):
        return "low"
    return "unknown"


def _ggufs_in_dir(dir_path: str) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    out: list[str] = []
    try:
        for name in sorted(os.listdir(dir_path), key=lambda s: s.lower()):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) and name.lower().endswith(".gguf"):
                out.append(full)
    except Exception:
        return []
    return out


def iter_wan22_gguf_files(models_root: str | None = None, *, roots: Sequence[str] | None = None) -> Iterable[str]:
    mr = models_root or default_models_root()
    use_roots = list(roots) if roots is not None else (get_paths_for("wan22_ckpt") or [os.path.join(mr, "wan22")])
    out: list[str] = []
    for root in use_roots:
        if os.path.isfile(root) and root.lower().endswith(".gguf"):
            out.append(root)
        elif os.path.isdir(root):
            out.extend(_ggufs_in_dir(root))
    return dedupe_keep_order(out)


__all__ = ["infer_wan22_stage", "iter_wan22_gguf_files"]

