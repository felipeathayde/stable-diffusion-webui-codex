"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model-asset inventory scanning and caching.
Builds a snapshot of local model files (VAEs, text encoders, LoRAs, WAN22 GGUF) and exposes cached helpers used by backend inventory endpoints and asset resolution.
Text encoder entries also include an optional `slot` field (e.g. `clip_l`, `clip_g`) derived via header-only inspection.

Symbols (top-level; keep in sync; no ghosts):
- `Inventory` (dataclass): Container for scanned model inventories (vaes/text_encoders/loras/wan22/metadata).
- `_CACHE` (constant): Process-local cached `Inventory` instance.
- `_repo_root` (function): Resolves the repository root used for scan defaults.
- `_models_root` (function): Returns the default `models/` root path under the repo.
- `_hf_root` (function): Returns the default Hugging Face metadata root path under `apps/backend/huggingface`.
- `_hash_file_sha256` (function): Computes sha256 by directly reading the file (fallback when the registry cache fails).
- `_get_file_sha256` (function): Computes/loads SHA256 for a file via the model registry cache.
- `resolve_asset_by_sha` (function): Resolves a SHA256 hash to a file path using the current inventory snapshot.
- `resolve_vae_path_by_sha` (function): Resolves a SHA256 hash to a VAE file path only.
- `resolve_text_encoder_slot_by_sha` (function): Resolves a SHA256 hash to a cached text-encoder slot (when available).
- `_SHA_TO_PATH` (constant): Lazy cache mapping `sha256 -> path` populated from the current inventory.
- `_SHA_TO_VAE_PATH` (constant): Lazy cache mapping `sha256 -> vae_path` populated from the current inventory.
- `scan_all` (function): Scans configured roots and returns an `Inventory` snapshot.
- `invalidate` (function): Clears process-local inventory and SHA maps so the next read rescans from current roots.
- `init` (function): Initializes the process-local inventory cache.
- `get` (function): Returns the cached inventory as a JSON-friendly dict.
- `refresh` (function): Rebuilds the inventory and replaces the process-local cache.
"""

from __future__ import annotations

import hashlib
import logging
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


def _hash_file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_file_sha256(path: str) -> str:
    """Get SHA256 for a file via the model registry hash cache.

    Falls back to direct hashing when the registry cache is unavailable.
    """
    try:
        from apps.backend.runtime.models.registry import get_registry

        reg = get_registry()
        sha256, _ = reg.hash_for(Path(path))
        if not sha256:
            raise RuntimeError("hash cache returned empty sha256")
        return sha256
    except Exception as exc:
        logging.getLogger("backend.inventory").warning(
            "inventory sha cache failed for %s (%s); falling back to direct hashing",
            path,
            exc.__class__.__name__,
        )
        return _hash_file_sha256(path)


# SHA256 -> Path resolution cache (populated during scan)
_SHA_TO_PATH: Dict[str, str] = {}
_SHA_TO_VAE_PATH: Dict[str, str] = {}
_SHA_TO_TEXT_ENCODER_SLOT: Dict[str, str] = {}


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


def resolve_vae_path_by_sha(sha256: str) -> str | None:
    """Resolve a SHA256 hash to a VAE file path from the inventory cache."""
    global _SHA_TO_VAE_PATH
    if not _SHA_TO_VAE_PATH:
        inv = get()
        for item in inv.get("vaes", []):
            if not isinstance(item, dict):
                continue
            sha = item.get("sha256")
            path = item.get("path")
            if isinstance(sha, str) and sha and isinstance(path, str) and path:
                _SHA_TO_VAE_PATH[sha] = path
    return _SHA_TO_VAE_PATH.get(str(sha256 or "").strip())

def resolve_text_encoder_slot_by_sha(sha256: str) -> str | None:
    """Resolve a text encoder sha256 to its cached slot label (if known)."""
    global _SHA_TO_TEXT_ENCODER_SLOT
    if not _SHA_TO_TEXT_ENCODER_SLOT:
        inv = get()
        for item in inv.get("text_encoders", []):
            if not isinstance(item, dict):
                continue
            sha = item.get("sha256")
            slot = item.get("slot")
            if isinstance(sha, str) and sha and isinstance(slot, str) and slot:
                _SHA_TO_TEXT_ENCODER_SLOT[sha] = slot
    return _SHA_TO_TEXT_ENCODER_SLOT.get(str(sha256 or "").strip())


def scan_all(models_root: str | None = None, hf_root: str | None = None) -> Inventory:
    mr = models_root or _models_root()
    hr = hf_root or _hf_root()

    vaes: List[Dict[str, str]] = []
    for full in iter_vae_files(models_root=mr):
        name = os.path.basename(full)
        entry: Dict[str, str] = {"name": name, "path": full}
        entry["sha256"] = _get_file_sha256(full)
        vaes.append(entry)
    vaes.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))

    # Text encoders: per-engine roots from apps/paths.json.
    text_encoders: List[Dict[str, str]] = []
    try:
        from apps.backend.core.contracts.text_encoder_slots import TextEncoderSlotError, classify_text_encoder_slot
    except Exception:
        TextEncoderSlotError = Exception  # type: ignore[assignment,misc]
        classify_text_encoder_slot = None  # type: ignore[assignment]
    for full in iter_text_encoder_files(models_root=mr):
        name = os.path.basename(full)
        entry: Dict[str, str] = {"name": name, "path": full}
        entry["sha256"] = _get_file_sha256(full)
        if callable(classify_text_encoder_slot):
            try:
                slot = classify_text_encoder_slot(full)
                entry["slot"] = str(slot)
            except TextEncoderSlotError:
                pass
            except Exception as exc:
                logging.getLogger("backend.inventory").debug(
                    "text encoder slot classification failed for %s (%s)",
                    full,
                    exc.__class__.__name__,
                )
        text_encoders.append(entry)
    text_encoders.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))
    loras: List[Dict[str, str]] = []
    for full in iter_lora_files(models_root=mr):
        name = os.path.basename(full)
        entry = {"name": name, "path": full}
        entry["sha256"] = _get_file_sha256(full)
        loras.append(entry)
    loras.sort(key=lambda d: (d["name"].lower(), d["path"].lower()))

    wan22: List[Dict[str, str]] = []
    for full in iter_wan22_gguf_files(models_root=mr):
        name = os.path.basename(full)
        stage = infer_wan22_stage(name)
        entry: Dict[str, str] = {"name": name, "path": full, "stage": stage}
        entry["sha256"] = _get_file_sha256(full)
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
    except Exception as exc:
        logging.getLogger("backend.inventory").warning(
            "inventory hash cache flush failed (%s)",
            exc.__class__.__name__,
        )

    return Inventory(vaes=vaes, text_encoders=text_encoders, loras=loras, wan22=wan22, metadata=metadata)


def invalidate() -> None:
    """Clear process-local inventory + SHA resolution caches.

    Use this when roots/config changed and callers want lazy rebuild on next `get()`.
    """

    global _CACHE
    _SHA_TO_PATH.clear()
    _SHA_TO_VAE_PATH.clear()
    _SHA_TO_TEXT_ENCODER_SLOT.clear()
    _CACHE = None


def init(models_root: str | None = None, hf_root: str | None = None) -> None:
    global _CACHE
    _SHA_TO_PATH.clear()
    _SHA_TO_VAE_PATH.clear()
    _SHA_TO_TEXT_ENCODER_SLOT.clear()
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
    _SHA_TO_PATH.clear()
    _SHA_TO_VAE_PATH.clear()
    _SHA_TO_TEXT_ENCODER_SLOT.clear()
    _CACHE = scan_all(models_root=models_root, hf_root=hf_root)
    return asdict(_CACHE)
