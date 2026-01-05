"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22-specific registry helpers for GGUF weights under `models/`.
Discovers `.gguf` candidates under the configured WAN22 roots and classifies them as high/low stage based on filename heuristics
for use in selection UIs and tooling.

Symbols (top-level; keep in sync; no ghosts):
- `GGUFEntry` (dataclass): Discovered GGUF file record (name/path/stage).
- `list_wan22_gguf` (function): Returns sorted GGUF candidates from the canonical WAN22 roots (`paths.json:wan22_ckpt` or `models/wan22`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.inventory.scanners.wan22_gguf import infer_wan22_stage, iter_wan22_gguf_files

@dataclass(frozen=True)
class GGUFEntry:
    name: str
    path: str
    stage: str  # 'high' | 'low' | 'unknown'

def list_wan22_gguf(models_root: str = "models") -> List[GGUFEntry]:
    out: List[GGUFEntry] = []
    mr = models_root
    if not os.path.isabs(mr):
        mr = os.path.join(str(get_repo_root()), mr)

    for full in iter_wan22_gguf_files(models_root=mr):
        name = os.path.basename(full)
        out.append(GGUFEntry(name=name, path=full, stage=infer_wan22_stage(name)))
    return sorted(out, key=lambda e: e.name.lower())


__all__ = ["GGUFEntry", "list_wan22_gguf"]
