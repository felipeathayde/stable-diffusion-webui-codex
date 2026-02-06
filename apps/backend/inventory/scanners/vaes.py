"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: VAE file discovery policy used by backend inventories and registries.
Defines per-family VAE roots via `apps/paths.json` keys (`*_vae`, including Anima) and yields weight file paths in stable order.

Symbols (top-level; keep in sync; no ghosts):
- `VAE_EXTS` (constant): Recognized VAE weight file extensions.
- `iter_vae_files` (function): Yields VAE file paths under per-family `*_vae` roots (stable order).
"""

from __future__ import annotations

import os
from typing import Iterable

from apps.backend.infra.config.paths import get_paths_for

from .base import dedupe_keep_order, iter_files

VAE_EXTS: tuple[str, ...] = (".safetensors", ".pt", ".bin")


def iter_vae_files(models_root: str | None = None) -> Iterable[str]:
    out: list[str] = []

    # Engine-specific overrides (roots may be files or dirs).
    for key in ("sd15_vae", "sdxl_vae", "flux1_vae", "anima_vae", "zimage_vae", "wan22_vae"):
        for root in get_paths_for(key):
            if os.path.isfile(root) and root.lower().endswith(VAE_EXTS):
                out.append(root)
            elif os.path.isdir(root):
                out.extend(list(iter_files([root], exts=VAE_EXTS)))

    return dedupe_keep_order(out)


__all__ = ["VAE_EXTS", "iter_vae_files"]
