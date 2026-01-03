"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22-specific registry helpers for GGUF weights under `models/`.
Discovers `.gguf` candidates and classifies them as high/low stage based on filename heuristics for use in selection UIs and tooling.

Symbols (top-level; keep in sync; no ghosts):
- `GGUFEntry` (dataclass): Discovered GGUF file record (name/path/stage).
- `_detect_stage` (function): Heuristically classifies a filename as `high`/`low`/`unknown`.
- `list_wan22_gguf` (function): Returns sorted GGUF candidates from `models/` and common WAN subfolders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class GGUFEntry:
    name: str
    path: str
    stage: str  # 'high' | 'low' | 'unknown'


def _detect_stage(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ("high", "highnoise", "high_noise")):
        return "high"
    if any(k in n for k in ("low", "lownoise", "low_noise")):
        return "low"
    return "unknown"


def list_wan22_gguf(models_root: str = "models") -> List[GGUFEntry]:
    out: List[GGUFEntry] = []
    if not os.path.isdir(models_root):
        return out
    # look for .gguf files at root and in common subfolders
    candidates: List[str] = []
    try:
        for name in os.listdir(models_root):
            p = os.path.join(models_root, name)
            if os.path.isfile(p) and name.lower().endswith(".gguf"):
                candidates.append(p)
    except Exception:
        pass
    for sub in ("Wan", "wan", "codex", "wan22", "WAN", "WAN22"):
        d = os.path.join(models_root, sub)
        try:
            if os.path.isdir(d):
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    if os.path.isfile(p) and name.lower().endswith(".gguf"):
                        candidates.append(p)
        except Exception:
            pass

    seen = set()
    for full in candidates:
        if full in seen:
            continue
        seen.add(full)
        stage = _detect_stage(os.path.basename(full))
        out.append(GGUFEntry(name=os.path.basename(full), path=full, stage=stage))
    return sorted(out, key=lambda e: e.name.lower())


__all__ = ["GGUFEntry", "list_wan22_gguf"]
