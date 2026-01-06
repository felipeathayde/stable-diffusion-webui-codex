"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA adapter discovery and lightweight type sniffing.
Lists LoRA files from canonical inventory roots (models defaults + `get_paths_for("*_loras")` overrides), and optionally inspects safetensors keys to classify adapter types.

Symbols (top-level; keep in sync; no ghosts):
- `LoraEntry` (dataclass): Described LoRA metadata (path/format/size/types).
- `list_loras` (function): Returns stable `{name,path}` entries for discovered LoRAs.
- `_detect_types_safetensors` (function): Best-effort type detection from safetensors key names (`lora/loha/lokr/glora/diff`).
- `describe_loras` (function): Returns `LoraEntry` objects including file size and detected types when available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

from apps.backend.inventory.scanners.loras import iter_lora_files

try:
    import safetensors
    import safetensors.torch as sf
except Exception:  # pragma: no cover - optional at import time
    sf = None  # type: ignore


@dataclass
class LoraEntry:
    name: str
    path: str
    size_bytes: int | None
    format: str  # safetensors|pt|ckpt|unknown
    types: List[str]  # [lora|loha|lokr|glora|diff]


def list_loras(roots: List[str] | None = None) -> List[Dict[str, str]]:
    """Lightweight discovery for LoRA adapters.

    Returns a list of {name, path} objects in stable order.
    """
    items: Dict[str, str] = {}
    for full in iter_lora_files(roots=roots):
        name = os.path.splitext(os.path.basename(full))[0]
        items.setdefault(name, full)
    return [{"name": k, "path": items[k]} for k in sorted(items.keys(), key=lambda s: s.lower())]


def _detect_types_safetensors(path: str) -> List[str]:
    if sf is None:
        return []
    try:
        data = sf.safe_open(path, framework="pt")
    except Exception:
        return []
    keys = list(data.keys())
    types: set[str] = set()
    for k in keys:
        if ".lora_" in k or k.endswith(".lora.up.weight") or k.endswith(".lora_down.weight"):
            types.add("lora")
        if ".hada_w1_" in k or ".hada_w2_" in k:
            types.add("loha")
        if ".lokr_" in k:
            types.add("lokr")
        if ".a1.weight" in k and ".b2.weight" in k:
            types.add("glora")
        if k.endswith(".diff") or k.endswith(".diff_b") or k.endswith(".set_weight"):
            types.add("diff")
    return sorted(types)


def describe_loras(roots: List[str] | None = None) -> List[LoraEntry]:
    entries: List[LoraEntry] = []
    for item in list_loras(roots):
        path = item["path"]
        fmt = "safetensors" if path.lower().endswith(".safetensors") else ("pt" if path.lower().endswith(".pt") else ("ckpt" if path.lower().endswith(".ckpt") else "unknown"))
        try:
            size = os.path.getsize(path)
        except Exception:
            size = None
        types = _detect_types_safetensors(path) if fmt == "safetensors" else []
        entries.append(LoraEntry(name=item["name"], path=path, size_bytes=size, format=fmt, types=types))
    return entries


__all__ = ["LoraEntry", "list_loras", "describe_loras"]
