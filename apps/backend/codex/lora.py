"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Transitional LoRA selection state shared by legacy surfaces.
Stores the currently selected LoRAs so compatibility code paths can query/apply them while the native APIs are adopted.

Symbols (top-level; keep in sync; no ghosts):
- `LoraSelection` (dataclass): Selected LoRA record (path/weight/online flag).
- `set_selections` (function): Replaces the global selection list (tolerates dict-like inputs from API plumbing).
- `get_selections` (function): Returns a copy of the current selection list.
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraSelection:
    path: str
    weight: float = 1.0
    online: bool = False  # keep for future live-merge modes


_SELECTIONS: List[LoraSelection] = []


def set_selections(selections: List[LoraSelection]) -> None:
    global _SELECTIONS
    # normalize and copy
    out: List[LoraSelection] = []
    for s in selections:
        if not isinstance(s, LoraSelection):
            # tolerate dict inputs from API plumbing
            path = str(getattr(s, "path", None) or s.get("path"))  # type: ignore[attr-defined]
            weight = float(getattr(s, "weight", None) if hasattr(s, "weight") else s.get("weight", 1.0))  # type: ignore[attr-defined]
            online = bool(getattr(s, "online", None) if hasattr(s, "online") else s.get("online", False))  # type: ignore[attr-defined]
            out.append(LoraSelection(path=path, weight=weight, online=online))
        else:
            out.append(s)
    _SELECTIONS = out


def get_selections() -> List[LoraSelection]:
    return list(_SELECTIONS)


__all__ = ["LoraSelection", "set_selections", "get_selections"]
