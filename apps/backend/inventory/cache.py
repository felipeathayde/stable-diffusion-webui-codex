"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model-asset inventory scanning and caching.
Builds a snapshot of local model files (VAEs, text encoders, LoRAs, WAN22 GGUF) and exposes cached helpers used by backend inventory endpoints and asset resolution.

Symbols (top-level; keep in sync; no ghosts):
- `Inventory` (dataclass): Container for scanned model inventories (vaes/text_encoders/loras/wan22/metadata).
- `_CACHE` (constant): Process-local cached `Inventory` instance.
- `_repo_root` (function): Resolves the repository root used for scan defaults.
- `_models_root` (function): Returns the default `models/` root path under the repo.
- `_hf_root` (function): Returns the default Hugging Face metadata root path under `apps/backend/huggingface`.
- `_get_file_sha256` (function): Computes/loads SHA256 for a file via the model registry cache.
- `resolve_asset_by_sha` (function): Resolves a SHA256 hash to a file path using the current inventory snapshot.
- `_SHA_TO_PATH` (constant): Lazy cache mapping `sha256 -> path` populated from the current inventory.
- `scan_all` (function): Scans configured roots and returns an `Inventory` snapshot.
- `init` (function): Initializes the process-local inventory cache.
- `get` (function): Returns the cached inventory as a JSON-friendly dict.
- `refresh` (function): Rebuilds the inventory and replaces the process-local cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.inventory.scanners.loras import iter_lora_files
from apps.backend.inventory.scanners.text_encoders import iter_text_encoder_files
from apps.backend.inventory.scanners.vendored_hf import iter_vendored_hf_repos
from apps.backend.inventory.scanners.vaes import iter_vae_files
from apps.backend.inventory.scanners.wan22_gguf import infer_wan22_stage, iter_wan22_gguf_files


@dataclass(frozen=True)
class Inventory:
    vaes: List[Dict[str, str]]
    text_encoders: List[Dict[str, str]]
    loras: List[Dict[str, str]]
    wan22: List[Dict[str, str]]  # .gguf files under WAN22 roots
    metadata: List[Dict[str, str]]  # org/repo roots under backend/huggingface


_CACHE: Inventory | None = None


def _repo_root() -> Path:
    return get_repo_root()


def _models_root() -> str:
    return str(_repo_root() / "models")


def _hf_root() -> str:
    return str(_repo_root() / "apps" / "backend" / "huggingface")


def _get_file_sha256(path: str) -> str | None:
    """Get SHA256 for a file via the model registry hash cache."""
    try:
        from apps.backend.runtime.models.registry import get_registry
        reg = get_registry()
        sha256, _ = reg.hash_for(Path(path))
        return sha256
    except Exception:
        return None


# SHA256 -> Path resolution cache (populated during scan)
_SHA_TO_PATH: Dict[str, str] = {}


def resolve_asset_by_sha(sha256: str) -> str | None:
    """Resolve a SHA256 hash to its file path from the inventory cache.
    
    Searches all model types: text encoders, VAEs, LoRAs, and WAN22 GGUF models.
    """
    global _SHA_TO_PATH
    if not _SHA_TO_PATH:
        # Populate cache from current inventory (all model types)
        inv = get()
        for item in inv["text_encoders"] + inv["vaes"] + inv["loras"] + inv["wan22"]:
            sha = item.get("sha256")
            path = item.get("path")
            if sha and path:
                _SHA_TO_PATH[sha] = path
    return _SHA_TO_PATH.get(sha256)


def scan_all(models_root: str | None = None, hf_root: str | None = None) -> Inventory:
    mr = models_root or _models_root()
    hr = hf_root or _hf_root()

    vaes: List[Dict[str, str]] = []
    for full in iter_vae_files(models_root=mr):
        name = os.path.basename(full)
        entry: Dict[str, str] = {"name": name, "path": full}
        sha = _get_file_sha256(full)
        if sha:
            entry["sha256"] = sha
        vaes.append(entry)
    vaes.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))

    # Text encoders: models/text-encoder plus per-engine roots from apps/paths.json.
    text_encoders: List[Dict[str, str]] = []
    for full in iter_text_encoder_files(models_root=mr):
        name = os.path.basename(full)
        entry: Dict[str, str] = {"name": name, "path": full}
        sha = _get_file_sha256(full)
        if sha:
            entry["sha256"] = sha
        text_encoders.append(entry)
    text_encoders.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))
    loras: List[Dict[str, str]] = []
    for full in iter_lora_files(models_root=mr):
        name = os.path.basename(full)
        entry = {"name": name, "path": full}
        sha = _get_file_sha256(full)
        if sha:
            entry["sha256"] = sha
        loras.append(entry)
    loras.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))

    wan22: List[Dict[str, str]] = []
    for full in iter_wan22_gguf_files(models_root=mr):
        name = os.path.basename(full)
        stage = infer_wan22_stage(name)
        entry: Dict[str, str] = {"name": name, "path": full, "stage": stage}
        sha = _get_file_sha256(full)
        if sha:
            entry["sha256"] = sha
        wan22.append(entry)
    if wan22:
        wan22.sort(key=lambda d: d["name"].lower())

    # Metadata folders: org/repo roots under hf_root
    metadata: List[Dict[str, str]] = []
    for org, repo, repo_dir in iter_vendored_hf_repos(hr):
        metadata.append({"name": f"{org}/{repo}", "path": repo_dir})
    metadata.sort(key=lambda d: d["name"].lower())

    # Persist any newly computed hashes to disk
    try:
        from apps.backend.runtime.models.registry import get_registry
        get_registry().flush_hash_cache()
    except Exception:
        pass

    return Inventory(vaes=vaes, text_encoders=text_encoders, loras=loras, wan22=wan22, metadata=metadata)


def init(models_root: str | None = None, hf_root: str | None = None) -> None:
    global _CACHE
    _CACHE = scan_all(models_root=models_root, hf_root=hf_root)


def get() -> Dict[str, List[Dict[str, str]]]:
    global _CACHE
    if _CACHE is None:
        init()
    assert _CACHE is not None
    return asdict(_CACHE)


def refresh(models_root: str | None = None, hf_root: str | None = None) -> Dict[str, List[Dict[str, str]]]:
    """Re-scan models and HF metadata roots and replace the in-memory cache.

    Returns the refreshed inventory as a plain dict suitable for JSON responses.
    """
    global _CACHE
    _CACHE = scan_all(models_root=models_root, hf_root=hf_root)
    return asdict(_CACHE)
