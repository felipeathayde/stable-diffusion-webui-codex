from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import json
import torch
import safetensors.torch as sf
from PIL import Image

from apps.backend.runtime.text_processing.textual_inversion import create_embedding_from_data


@dataclass
class EmbeddingEntry:
    name: str
    path: str
    format: str  # safetensors|pt|bin|png|webp|unknown
    vectors: int | None
    dims: int | None
    step: int | None


def _default_roots(models_root: str = "models") -> List[str]:
    roots = []
    for sub in ("embeddings", "Embeddings", "textual_inversion", "ti"):
        p = os.path.join(models_root, sub)
        if os.path.isdir(p):
            roots.append(p)
    # apps paths override
    cfg = os.path.join("apps", "paths.json")
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        for p in (data.get("embeddings") or []):
            if isinstance(p, str) and os.path.isdir(p):
                roots.append(p)
    except Exception:
        pass
    seen = set(); out: List[str] = []
    for r in roots:
        if r not in seen:
            seen.add(r); out.append(r)
    return out


def _iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    try:
        for dp, _dn, files in os.walk(root):
            for fn in files:
                if any(fn.lower().endswith(ext) for ext in exts):
                    yield os.path.join(dp, fn)
    except Exception:
        return []


def list_embeddings(roots: List[str] | None = None) -> List[Dict[str, str]]:
    if roots is None:
        roots = _default_roots()
    exts = (".safetensors", ".pt", ".bin", ".png", ".webp", ".jxl", ".avif")
    items: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for full in _iter_files(root, exts):
            base = os.path.splitext(os.path.basename(full))[0]
            # Drop .preview suffix
            if base.lower().endswith(".preview"):
                base = base[: -len(".preview")]
            items.setdefault(base, full)
    return [{"name": k, "path": items[k]} for k in sorted(items.keys(), key=lambda s: s.lower())]


def _load_meta(path: str) -> tuple[int | None, int | None, int | None]:
    """Return (vectors, dims, step) best-effort from file contents."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".safetensors",):
            data = sf.load_file(path, device="cpu")
        elif ext in (".pt", ".bin"):
            data = torch.load(path, map_location="cpu")
        elif ext in (".png", ".webp", ".jxl", ".avif"):
            img = Image.open(path)
            if hasattr(img, 'text') and 'sd-ti-embedding' in img.text:
                import base64, json
                from apps.backend.runtime.text_processing.textual_inversion import EmbeddingDecoder
                payload = base64.b64decode(img.text['sd-ti-embedding'])
                data = json.loads(payload, cls=EmbeddingDecoder)
            else:
                # unsupported embedded mode here
                return None, None, None
        else:
            return None, None, None
        emb = create_embedding_from_data(data, os.path.splitext(os.path.basename(path))[0], filename=os.path.basename(path), filepath=path)
        return int(getattr(emb, 'vectors', 0) or 0) or None, int(getattr(emb, 'shape', 0) or 0) or None, getattr(emb, 'step', None)
    except Exception:
        return None, None, None


def describe_embeddings(roots: List[str] | None = None) -> List[EmbeddingEntry]:
    out: List[EmbeddingEntry] = []
    for item in list_embeddings(roots):
        path = item["path"]
        fmt = os.path.splitext(path)[1].lower().lstrip(".")
        vecs, dims, step = _load_meta(path)
        out.append(EmbeddingEntry(name=item["name"], path=path, format=fmt or "unknown", vectors=vecs, dims=dims, step=step))
    return out


__all__ = ["EmbeddingEntry", "list_embeddings", "describe_embeddings"]

