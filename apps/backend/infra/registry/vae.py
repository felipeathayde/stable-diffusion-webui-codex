"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: VAE asset discovery registry for UI dropdowns and metadata panels.
Scans per-family VAE roots configured in `apps/paths.json` and vendored Hugging Face repos under `apps/backend/huggingface`, returning stable
name lists and basic per-VAE metadata (format/latent_channels/scaling_factor) when available.

Symbols (top-level; keep in sync; no ghosts):
- `DEFAULT_BASELINES` (constant): Baseline VAE entries shown first (Automatic/Built in/None).
- `_collect_from_paths_roots` (function): Collects VAE entries from per-family `*_vae` roots declared in `apps/paths.json`.
- `_collect_from_vendored_hf` (function): Collects `vae/` directories from vendored HF repos.
- `list_vaes` (function): Returns ordered VAE choice names (baselines + discovered entries).
- `_read_json` (function): Best-effort JSON reader used for VAE config probing.
- `describe_vaes` (function): Returns metadata dicts for discovered VAEs (name/path/format + optional config hints).
"""

from __future__ import annotations

import os
from typing import List

from .base import AssetEntry, _iter_dirs
import json
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.inventory.scanners.vaes import iter_vae_files


DEFAULT_BASELINES = ["Automatic", "Built in", "None"]


def _collect_from_paths_roots(models_root: str) -> List[AssetEntry]:
    out: List[AssetEntry] = []
    for full in iter_vae_files(models_root=models_root):
        out.append(AssetEntry(name=os.path.basename(full), path=full, kind="vae"))
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
    Includes VAEs from engine-specific paths in paths.json (flux1_vae, zimage_vae, sd15_vae, etc.).
    """
    if not os.path.isabs(models_root):
        models_root = str(get_repo_root() / models_root)
    if not os.path.isabs(vendored_hf_root):
        vendored_hf_root = str(get_repo_root() / vendored_hf_root)

    entries: List[AssetEntry] = []
    entries.extend(_collect_from_paths_roots(models_root))
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


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def describe_vaes(models_root: str = "models", vendored_hf_root: str = "apps/backend/huggingface") -> List[dict]:
    """Return metadata for discovered VAEs.

    Each entry contains: name, path, format (diffusers|file|dir), latent_channels, scaling_factor.
    Includes VAEs from engine-specific paths in paths.json (flux1_vae, sd15_vae, etc.).
    """
    if not os.path.isabs(models_root):
        models_root = str(get_repo_root() / models_root)
    if not os.path.isabs(vendored_hf_root):
        vendored_hf_root = str(get_repo_root() / vendored_hf_root)

    info = []
    seen_paths: set[str] = set()
    # Scan paths.json roots
    for e in _collect_from_paths_roots(models_root):
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
    return sorted(info, key=lambda m: m["name"].lower())


__all__ = ["list_vaes", "describe_vaes"]
