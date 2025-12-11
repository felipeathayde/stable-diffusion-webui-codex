from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from apps.backend.infra.config.paths import get_paths_for


@dataclass(frozen=True)
class Inventory:
    vaes: List[Dict[str, str]]
    text_encoders: List[Dict[str, str]]
    loras: List[Dict[str, str]]
    wan22: List[Dict[str, str]]  # .gguf files under WAN22 roots
    metadata: List[Dict[str, str]]  # org/repo roots under backend/huggingface


_CACHE: Inventory | None = None


def _repo_root() -> Path:
    # apps/backend/inventory/cache.py -> repo_root = parents[3]
    return Path(__file__).resolve().parents[3]


def _models_root() -> str:
    return str(_repo_root() / "models")


def _hf_root() -> str:
    return str(_repo_root() / "apps" / "backend" / "huggingface")


def _list_files(dir_path: str, exts: tuple[str, ...], *, include_sha: bool = False) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not os.path.isdir(dir_path):
        return out
    try:
        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) and name.lower().endswith(exts):
                item: Dict[str, str] = {"name": name, "path": full}
                if include_sha:
                    sha = _get_file_sha256(full)
                    if sha:
                        item["sha256"] = sha
                out.append(item)
    except Exception:
        return out
    return sorted(out, key=lambda d: d["name"].lower())


def _get_file_sha256(path: str) -> str | None:
    """Get SHA256 for a file, using registry cache with persistence.
    
    Uses registry._hash_for which updates and persists the cache to disk.
    """
    try:
        from pathlib import Path
        from apps.backend.runtime.models.registry import get_registry
        reg = get_registry()
        # Use registry's _hash_for which updates cache and marks dirty for persistence
        sha256, _ = reg._hash_for(Path(path))
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


def _list_dirs(dir_path: str) -> List[str]:
    try:
        return [os.path.join(dir_path, n) for n in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, n))]
    except Exception:
        return []


def scan_all(models_root: str | None = None, hf_root: str | None = None) -> Inventory:
    mr = models_root or _models_root()
    hr = hf_root or _hf_root()

    # Exact subfolder policy (no guessing) for legacy locations:
    vae_dir = os.path.join(mr, "VAE")
    lora_dir = os.path.join(mr, "Lora")

    vaes = _list_files(vae_dir, (".safetensors", ".bin", ".pt"), include_sha=True)

    # Text encoders: strict models/text-encoder plus per-engine roots from apps/paths.json.
    text_encoders: List[Dict[str, str]] = []
    te_exts = (".safetensors", ".bin", ".pt", ".gguf")
    seen_te_paths: set[str] = set()

    def _append_text_encoders(root: str) -> None:
        nonlocal text_encoders, seen_te_paths
        for item in _list_files(root, te_exts, include_sha=True):
            path = item.get("path")
            if not path or path in seen_te_paths:
                continue
            seen_te_paths.add(path)
            text_encoders.append(item)

    # Legacy root
    te_dir = os.path.join(mr, "text-encoder")
    _append_text_encoders(te_dir)

    # Engine-specific roots from apps/paths.json (sd15_tenc, sdxl_tenc, flux_tenc, wan22_tenc, zimage_tenc)
    try:
        for key in ("sd15_tenc", "sdxl_tenc", "flux_tenc", "wan22_tenc", "zimage_tenc"):
            for root in get_paths_for(key):
                if os.path.isdir(root):
                    _append_text_encoders(root)
    except Exception:
        # Do not break inventory on misconfigured paths.json; legacy roots still apply.
        pass
    # Engine-specific VAEs from apps/paths.json (flux_vae, zimage_vae): scan roots directly without
    # relying on filename heuristics. Any weight-like file under these roots is exposed
    # so the UI can list concrete Flux/ZImage VAEs based purely on paths.json.
    try:
        engine_vaes: List[Dict[str, str]] = []
        for key in ("flux_vae", "zimage_vae"):
            for root in get_paths_for(key):
                if os.path.isdir(root):
                    for name in os.listdir(root):
                        full = os.path.join(root, name)
                        if os.path.isfile(full):
                            sha = _get_file_sha256(full)
                            entry: Dict[str, str] = {"name": name, "path": full}
                            if sha:
                                entry["sha256"] = sha
                            engine_vaes.append(entry)
                elif os.path.isfile(root):
                    sha = _get_file_sha256(root)
                    entry = {"name": os.path.basename(root), "path": root}
                    if sha:
                        entry["sha256"] = sha
                    engine_vaes.append(entry)
        if engine_vaes:
            engine_vaes.sort(key=lambda d: d["name"].lower())
            seen_paths = {v["path"] for v in vaes}
            for item in engine_vaes:
                if item["path"] not in seen_paths:
                    vaes.append(item)
                    seen_paths.add(item["path"])
    except Exception:
        # Do not break inventory on misconfigured VAE roots.
        pass

    loras = _list_files(lora_dir, (".safetensors", ".bin", ".pt", ".ckpt"), include_sha=True)

    # WAN22 GGUF with stage detection (high/low/unknown).
    # Prefer explicit WAN22 roots from apps/paths.json ("wan22_ckpt"); fall back to models/wan22.
    wan22: List[Dict[str, str]] = []
    wan_roots = get_paths_for("wan22_ckpt") or [os.path.join(mr, "wan22")]
    for root in wan_roots:
        if not os.path.isdir(root):
            continue
        try:
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if os.path.isfile(full) and name.lower().endswith(".gguf"):
                    lower = name.lower()
                    stage = "unknown"
                    if any(k in lower for k in ("high", "highnoise", "high_noise")):
                        stage = "high"
                    elif any(k in lower for k in ("low", "lownoise", "low_noise")):
                        stage = "low"
                    sha = _get_file_sha256(full)
                    entry: Dict[str, str] = {"name": name, "path": full, "stage": stage}
                    if sha:
                        entry["sha256"] = sha
                    wan22.append(entry)
        except Exception:
            continue
    if wan22:
        wan22.sort(key=lambda d: d["name"].lower())

    # Metadata folders: org/repo roots under hf_root
    metadata: List[Dict[str, str]] = []
    for org in _list_dirs(hr):
        for repo in _list_dirs(org):
            metadata.append({
                "name": f"{os.path.basename(org)}/{os.path.basename(repo)}",
                "path": repo,
            })
    metadata.sort(key=lambda d: d["name"].lower())

    # Persist any newly computed hashes to disk
    try:
        from apps.backend.runtime.models.registry import get_registry, _save_hash_cache
        reg = get_registry()
        if reg._hash_cache_dirty:
            _save_hash_cache(reg._hash_cache)
            reg._hash_cache_dirty = False
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
