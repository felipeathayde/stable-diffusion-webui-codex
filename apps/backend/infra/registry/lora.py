from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

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


def _iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    try:
        for dp, _dn, files in os.walk(root):
            for fn in files:
                if any(fn.lower().endswith(ext) for ext in exts):
                    yield os.path.join(dp, fn)
    except Exception:
        return []


def _default_search_roots(models_root: str = "models") -> List[str]:
    roots = []
    # Built-in defaults: prefer per-model LoRA folders only.
    for sub in ("sd15-loras", "sdxl-loras", "flux-loras", "wan22-loras"):
        p = os.path.join(models_root, sub)
        if os.path.isdir(p):
            roots.append(p)
    # Optional user-configurable paths
    cfg = os.path.join("apps", "paths.json")
    try:
        import json

        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        # Per-model overrides
        for key in ("sd15_loras", "sdxl_loras", "flux_loras", "wan22_loras"):
            for p in (data.get(key) or []):
                if isinstance(p, str) and os.path.isdir(p):
                    roots.append(p)
    except Exception:
        pass
    # Deduplicate while keeping order
    out: List[str] = []
    seen = set()
    for r in roots:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def list_loras(roots: List[str] | None = None) -> List[Dict[str, str]]:
    """Lightweight discovery for LoRA adapters.

    Returns a list of {name, path} objects in stable order.
    """
    if roots is None:
        roots = _default_search_roots()
    exts = (".safetensors", ".ckpt", ".pt")
    items: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for full in _iter_files(root, exts):
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
