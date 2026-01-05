"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA file discovery policy used by backend inventories and registries.
Defines the default roots and per-family overrides (paths.json keys) and yields LoRA weight files in stable order.

Symbols (top-level; keep in sync; no ghosts):
- `LORA_EXTS` (constant): Recognized LoRA weight file extensions.
- `list_lora_roots` (function): Resolves LoRA search roots (models defaults + `get_paths_for("*_loras")` overrides).
- `iter_lora_files` (function): Yields LoRA file paths under the resolved roots (recursive, stable order).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from apps.backend.infra.config.paths import get_paths_for

from .base import default_models_root, dedupe_keep_order, iter_files

LORA_EXTS: tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".bin")


def list_lora_roots(models_root: str | None = None) -> list[str]:
    mr = models_root or default_models_root()
    roots: list[str] = []

    # Built-in conventions (common in SD deployments).
    for sub in ("Lora", "sd15-loras", "sdxl-loras", "flux-loras", "wan22-loras", "zimage-loras"):
        p = os.path.join(mr, sub)
        if os.path.isdir(p):
            roots.append(p)

    # Explicit overrides from apps/paths.json
    for key in ("sd15_loras", "sdxl_loras", "flux1_loras", "wan22_loras", "zimage_loras"):
        for p in get_paths_for(key):
            if os.path.isdir(p):
                roots.append(p)

    return dedupe_keep_order(roots)


def iter_lora_files(models_root: str | None = None, *, roots: Sequence[str] | None = None) -> Iterable[str]:
    use_roots = list(roots) if roots is not None else list_lora_roots(models_root=models_root)
    return dedupe_keep_order(list(iter_files(use_roots, exts=LORA_EXTS)))


__all__ = ["LORA_EXTS", "iter_lora_files", "list_lora_roots"]
