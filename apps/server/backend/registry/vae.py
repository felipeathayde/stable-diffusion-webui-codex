from __future__ import annotations

import os
from typing import List

from .base import AssetEntry, _iter_dirs


DEFAULT_BASELINES = ["Automatic", "Built in", "None"]


def _collect_from_models_root(models_root: str) -> List[AssetEntry]:
    out: List[AssetEntry] = []
    if not models_root:
        return out
    # common locations: models/VAE/*.safetensors or any file/folder with 'vae' keyword
    vae_dir = os.path.join(models_root, "VAE")
    try:
        for base in (vae_dir, models_root):
            if not os.path.isdir(base):
                continue
            for name in os.listdir(base):
                full = os.path.join(base, name)
                if os.path.isdir(full):
                    if name.lower().startswith("vae"):
                        out.append(AssetEntry(name=name, path=full, kind="vae"))
                elif os.path.isfile(full) and name.lower().endswith((".safetensors", ".pt", ".bin")):
                    if "vae" in name.lower():
                        out.append(AssetEntry(name=name, path=full, kind="vae"))
    except Exception:
        return out
    return out


def _collect_from_vendored_hf(hf_root: str) -> List[AssetEntry]:
    out: List[AssetEntry] = []
    if not hf_root or not os.path.isdir(hf_root):
        return out
    try:
        for repo in _iter_dirs(hf_root):
            vae_dir = os.path.join(repo, "vae")
            if os.path.isdir(vae_dir):
                name = os.path.basename(repo)
                out.append(AssetEntry(name=f"{name}/vae", path=vae_dir, kind="vae"))
    except Exception:
        return out
    return out


def list_vaes(models_root: str = "models", vendored_hf_root: str = "apps/server/backend/huggingface") -> List[str]:
    """Return an ordered list of VAE choices (names).

    Baselines always first; discovered entries are appended in stable order.
    """
    entries: List[AssetEntry] = []
    entries.extend(_collect_from_models_root(models_root))
    # scan one level deep in vendored hf
    if os.path.isdir(vendored_hf_root):
        try:
            for org in _iter_dirs(vendored_hf_root):
                entries.extend(_collect_from_vendored_hf(org))
        except Exception:
            pass

    names = [e.name for e in entries]
    ordered = DEFAULT_BASELINES + [n for n in sorted(names) if n not in DEFAULT_BASELINES]
    return ordered


__all__ = ["list_vaes"]

