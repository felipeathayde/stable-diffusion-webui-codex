from __future__ import annotations

import os
from typing import List

from .base import AssetEntry, _iter_dirs
import json


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


def list_vaes(models_root: str = "models", vendored_hf_root: str = "apps/backend/huggingface") -> List[str]:
    """Return an ordered list of VAE choices (names).

    Baselines always first; discovered entries are appended in stable order.
    Includes VAEs from engine-specific paths in paths.json (flux_vae, sd15_vae, etc.).
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

    # Collect VAEs from engine-specific paths in paths.json
    try:
        from apps.backend.infra.config.paths import get_paths_for
        seen_paths = {e.path for e in entries}
        for key in ("sd15_vae", "sdxl_vae", "flux_vae", "wan22_vae"):
            for root in get_paths_for(key):
                if os.path.isdir(root):
                    for name in os.listdir(root):
                        full = os.path.join(root, name)
                        if os.path.isfile(full) and full not in seen_paths:
                            entries.append(AssetEntry(name=name, path=full, kind="vae"))
                            seen_paths.add(full)
                elif os.path.isfile(root) and root not in seen_paths:
                    entries.append(AssetEntry(name=os.path.basename(root), path=root, kind="vae"))
                    seen_paths.add(root)
    except Exception:
        pass

    names = [e.name for e in entries]
    ordered = DEFAULT_BASELINES + [n for n in sorted(names) if n not in DEFAULT_BASELINES]
    return ordered


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def describe_vaes(models_root: str = "models", vendored_hf_root: str = "apps/backend/huggingface") -> List[dict]:
    """Return metadata for discovered VAEs.

    Each entry contains: name, path, format (diffusers|file|dir), latent_channels, scaling_factor.
    Includes VAEs from engine-specific paths in paths.json (flux_vae, sd15_vae, etc.).
    """
    info = []
    seen_paths: set[str] = set()
    # Scan models_root
    for e in _collect_from_models_root(models_root):
        if e.path in seen_paths:
            continue
        seen_paths.add(e.path)
        fmt = "file" if os.path.isfile(e.path) or e.path.lower().endswith(('.safetensors','.pt','.bin')) else "dir"
        meta = {"name": e.name, "path": e.path, "format": fmt, "latent_channels": None, "scaling_factor": None}
        # Try to find adjacent config.json
        cfg_path = os.path.join(os.path.dirname(e.path) if fmt=="file" else e.path, "config.json")
        if os.path.isfile(cfg_path):
            cfg = _read_json(cfg_path)
            meta["latent_channels"] = cfg.get("latent_channels") or cfg.get("vae_latent_channels")
            meta["scaling_factor"] = cfg.get("scaling_factor")
        info.append(meta)
    # Scan vendored HF repos (vae subfolder)
    if os.path.isdir(vendored_hf_root):
        for org in _iter_dirs(vendored_hf_root):
            for repo in _iter_dirs(org):
                vae_dir = os.path.join(repo, "vae")
                if os.path.isdir(vae_dir) and vae_dir not in seen_paths:
                    seen_paths.add(vae_dir)
                    cfg_path = os.path.join(vae_dir, "config.json")
                    meta = {"name": f"{os.path.basename(repo)}/vae", "path": vae_dir, "format": "diffusers", "latent_channels": None, "scaling_factor": None}
                    if os.path.isfile(cfg_path):
                        cfg = _read_json(cfg_path)
                        meta["latent_channels"] = cfg.get("latent_channels")
                        meta["scaling_factor"] = cfg.get("scaling_factor")
                    info.append(meta)
    # Collect VAEs from engine-specific paths in paths.json
    try:
        from apps.backend.infra.config.paths import get_paths_for
        for key in ("sd15_vae", "sdxl_vae", "flux_vae", "wan22_vae"):
            for root in get_paths_for(key):
                if os.path.isdir(root):
                    for name in os.listdir(root):
                        full = os.path.join(root, name)
                        if os.path.isfile(full) and full not in seen_paths:
                            seen_paths.add(full)
                            meta = {"name": name, "path": full, "format": "file", "latent_channels": None, "scaling_factor": None}
                            cfg_path = os.path.join(os.path.dirname(full), "config.json")
                            if os.path.isfile(cfg_path):
                                cfg = _read_json(cfg_path)
                                meta["latent_channels"] = cfg.get("latent_channels") or cfg.get("vae_latent_channels")
                                meta["scaling_factor"] = cfg.get("scaling_factor")
                            info.append(meta)
                elif os.path.isfile(root) and root not in seen_paths:
                    seen_paths.add(root)
                    meta = {"name": os.path.basename(root), "path": root, "format": "file", "latent_channels": None, "scaling_factor": None}
                    cfg_path = os.path.join(os.path.dirname(root), "config.json")
                    if os.path.isfile(cfg_path):
                        cfg = _read_json(cfg_path)
                        meta["latent_channels"] = cfg.get("latent_channels") or cfg.get("vae_latent_channels")
                        meta["scaling_factor"] = cfg.get("scaling_factor")
                    info.append(meta)
    except Exception:
        pass
    return sorted(info, key=lambda m: m["name"].lower())


__all__ = ["list_vaes", "describe_vaes"]
