"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Checkpoint/VAE discovery with sha256 caching and telemetry.
Scans configured model roots (via `apps/paths.json` accessors) for checkpoint and VAE weight files, computes sha256 hashes, and maintains a
persistent cache in `models/.hashes.json` to support fast UI inventory and backend SHA-based resolution.

Symbols (top-level; keep in sync; no ghosts):
- `_repo_root` (function): Resolves repo root from `CODEX_ROOT` (fails fast if unset).
- `_default_models_root` (function): Returns the default `models/` directory under `CODEX_ROOT`.
- `_default_hf_root` (function): Returns the default Hugging Face vendor cache root under `CODEX_ROOT` (when used).
- `_sha256` (function): Computes sha256 digest for a file path.
- `_HashCacheEntry` (dataclass): Cache entry for one file (sha + mtime + size) used to avoid re-hashing unchanged files.
- `_load_hash_cache` (function): Loads `.hashes.json` cache from disk.
- `_save_hash_cache` (function): Writes `.hashes.json` cache to disk.
- `ModelRegistry` (class): Registry service; scans paths, maintains caches, and produces `CheckpointRecord`/`VAERecord` lists for UI/API (also provides public hash-cache helpers).
- `get_registry` (function): Returns the singleton `ModelRegistry` instance.
- `list_checkpoints` (function): Returns checkpoint records (optional refresh).
- `list_vaes` (function): Returns VAE records (optional refresh).
- `refresh` (function): Forces a rescan + cache update for checkpoints/VAEs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime import trace as _trace

from .types import (
    CheckpointFormat,
    CheckpointRecord,
    VAERecord,
)

_LOGGER = logging.getLogger("backend.registry")

_ALLOWED_CHECKPOINT_EXTS = {".ckpt", ".safetensor", ".safetensors", ".pt", ".bin", ".gguf"}
_CHECKPOINT_BLACKLIST_SUFFIXES = {".vae.ckpt", ".vae.safetensor", ".vae.safetensors", ".vae.pt"}
_VAE_EXTS = {".safetensor", ".safetensors", ".ckpt", ".pt"}


def _repo_root() -> Path:
    """Get repo root from CODEX_ROOT environment variable.
    
    Raises EnvironmentError if CODEX_ROOT is not set.
    """
    env_root = os.environ.get("CODEX_ROOT")
    if not env_root:
        raise EnvironmentError(
            "CODEX_ROOT environment variable is not set. "
            "Please use run-webui.bat or run-tui.bat to launch the application."
        )
    return Path(env_root).resolve()


def _default_models_root() -> Path:
    return _repo_root() / "models"


def _default_hf_root() -> Path:
    return _repo_root() / "apps" / "backend" / "huggingface"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _HashCacheEntry:
    mtime: float
    size: int  # file size for extra validation
    sha256: str
    short_hash: str


# Persistent hash cache file location (under models/)
_HASH_CACHE_FILE = _default_models_root() / ".hashes.json"


def _load_hash_cache() -> Dict[str, _HashCacheEntry]:
    """Load persistent hash cache from disk."""
    cache: Dict[str, _HashCacheEntry] = {}
    _LOGGER.info("loading hash cache from %s", _HASH_CACHE_FILE)
    try:
        if _HASH_CACHE_FILE.is_file():
            with _HASH_CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for path, entry in data.items():
                if isinstance(entry, dict):
                    cache[path] = _HashCacheEntry(
                        mtime=float(entry.get("mtime", 0)),
                        size=int(entry.get("size", 0)),
                        sha256=str(entry.get("sha256", "")),
                        short_hash=str(entry.get("short_hash", "")),
                    )
            _LOGGER.info("hash cache loaded: %d entries", len(cache))
        else:
            _LOGGER.info("hash cache not found, will compute hashes on first scan")
    except Exception as e:
        _LOGGER.warning("hash cache load failed: %s", e)
    return cache


def _save_hash_cache(cache: Dict[str, _HashCacheEntry]) -> None:
    """Persist hash cache to disk."""
    try:
        data = {
            path: {
                "mtime": entry.mtime,
                "size": entry.size,
                "sha256": entry.sha256,
                "short_hash": entry.short_hash,
            }
            for path, entry in cache.items()
        }
        _HASH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _HASH_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        _LOGGER.debug("hash cache save failed: %s", e)


class ModelRegistry:
    """Discover checkpoint/VAE assets and expose cached views."""

    def __init__(self, *, models_root: Path | None = None, hf_root: Path | None = None) -> None:
        self._models_root = Path(models_root or _default_models_root()).resolve()
        self._hf_root = Path(hf_root or _default_hf_root()).resolve()
        self._lock = threading.Lock()
        self._checkpoints: Dict[str, CheckpointRecord] = {}
        self._vaes: Dict[str, VAERecord] = {}
        self._hash_cache: Dict[str, _HashCacheEntry] | None = None  # Lazy load
        self._hash_cache_dirty = False  # Track if we need to save
        self._last_scan: float | None = None

    def _ensure_hash_cache(self) -> Dict[str, _HashCacheEntry]:
        """Lazy load hash cache on first access."""
        if self._hash_cache is None:
            self._hash_cache = _load_hash_cache()
        return self._hash_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hash_for(self, path: Path) -> Tuple[str | None, str | None]:
        """Return (sha256, short_hash) for a file path, using the persistent hash cache.

        This is safe to call outside registry scans (it is lock-protected) and is the
        supported way for other subsystems (e.g. inventory) to request hashes without
        reaching into private internals.
        """

        with self._lock:
            return self._hash_for(path)

    def flush_hash_cache(self) -> None:
        """Persist the hash cache to disk if any new hashes were computed."""

        with self._lock:
            if self._hash_cache_dirty:
                _save_hash_cache(self._ensure_hash_cache())
                self._hash_cache_dirty = False

    def list_checkpoints(self, *, refresh: bool = False) -> List[CheckpointRecord]:
        if refresh:
            self.refresh()
        with self._lock:
            if not self._checkpoints:
                self._scan_locked()
            return list(self._checkpoints.values())

    def list_vaes(self, *, refresh: bool = False) -> List[VAERecord]:
        if refresh:
            self.refresh()
        with self._lock:
            if not self._vaes:
                self._scan_locked()
            return list(self._vaes.values())

    def refresh(self) -> None:
        with self._lock:
            self._scan_locked()

    def get_checkpoint(self, name: str) -> CheckpointRecord | None:
        with self._lock:
            if not self._checkpoints:
                self._scan_locked()
            return self._checkpoints.get(name)

    def get_vae(self, name: str) -> VAERecord | None:
        with self._lock:
            if not self._vaes:
                self._scan_locked()
            return self._vaes.get(name)

    # ------------------------------------------------------------------
    # Internal scanning helpers
    # ------------------------------------------------------------------
    def _scan_locked(self) -> None:
        start = time.perf_counter()
        checkpoints = {rec.name: rec for rec in self._scan_checkpoints()}
        vaes = {rec.name: rec for rec in self._scan_vaes()}
        duration_ms = (time.perf_counter() - start) * 1000.0
        self._checkpoints = checkpoints
        self._vaes = vaes
        self._last_scan = time.time()
        # Persist hash cache if we computed any new hashes
        if self._hash_cache_dirty:
            _save_hash_cache(self._ensure_hash_cache())
            self._hash_cache_dirty = False
        _LOGGER.info(
            "model_registry: scan complete checkpoints=%d vaes=%d ms=%.1f",
            len(checkpoints),
            len(vaes),
            duration_ms,
        )
        _trace.event("model_registry_scan", checkpoints=len(checkpoints), vaes=len(vaes), ms=f"{duration_ms:.2f}")

    def _scan_checkpoints(self) -> Iterable[CheckpointRecord]:
        seen: set[str] = set()
        for file in self._iter_checkpoint_files():
            path_str = str(file.resolve())
            if path_str in seen:
                continue
            seen.add(path_str)
            suffix = file.suffix.lower()
            if suffix == ".gguf":
                fmt = CheckpointFormat.GGUF
            else:
                fmt = CheckpointFormat.CHECKPOINT
            sha256, short_hash = self._hash_for(file)
            stat = file.stat()
            record = CheckpointRecord(
                name=file.stem,
                title=file.name,
                filename=str(file),
                path=str(file.parent),
                model_name=file.stem,
                format=fmt,
                sha256=sha256,
                short_hash=short_hash,
                file_size=stat.st_size,
                updated_at=stat.st_mtime,
            )
            yield record

    def _scan_vaes(self) -> Iterable[VAERecord]:
        candidates: List[Path] = []
        for key in ("sd15_vae", "sdxl_vae", "flux1_vae", "wan22_vae", "zimage_vae"):
            for raw in get_paths_for(key):
                p = Path(raw)
                if p not in candidates:
                    candidates.append(p)

        files: List[Path] = []
        for root in candidates:
            if root.is_file() and root.suffix.lower() in _VAE_EXTS:
                files.append(root)
            elif root.is_dir():
                try:
                    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
                        if path.is_file() and path.suffix.lower() in _VAE_EXTS:
                            files.append(path)
                except Exception:
                    continue

        seen: set[str] = set()
        for path in files:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            sha256, short_hash = self._hash_for(path)
            stat = path.stat()
            yield VAERecord(
                name=path.name,
                filename=str(path),
                source=str(path.parent),
                sha256=sha256,
                short_hash=short_hash,
                updated_at=stat.st_mtime,
            )

    def _iter_checkpoint_files(self) -> Iterable[Path]:
        """Iterate over checkpoint files using paths.json overrides + curated defaults.

        Resolution order:
        1) Explicit roots from apps/paths.json per engine (sd15_ckpt, sdxl_ckpt, flux1_ckpt, wan22_ckpt).
        2) Built-in defaults under models/: root, sd15, sdxl, flux.

        This replaces the legacy scatter of ad-hoc checkpoint folders ('stable-diffusion', 'sd', 'checkpoints').
        """
        candidates: List[Path] = []

        # 1) User overrides from apps/paths.json per engine
        try:
            for key in ("sd15_ckpt", "sdxl_ckpt", "flux1_ckpt", "wan22_ckpt", "zimage_ckpt"):
                for raw in get_paths_for(key):
                    p = Path(raw)
                    if p not in candidates:
                        candidates.append(p)
        except Exception:
            # Do not break discovery if paths.json is invalid; fall back to defaults.
            candidates = []

        # 2) Curated built-in defaults quando não há overrides configurados.
        if not candidates:
            defaults = [
                self._models_root,
                self._models_root / "sd15",
                self._models_root / "sdxl",
                self._models_root / "flux",
                self._models_root / "zimage",
            ]
            for p in defaults:
                if p not in candidates:
                    candidates.append(p)

        for directory in candidates:
            if not directory.is_dir():
                continue
            for entry in directory.rglob("*"):
                if not entry.is_file():
                    continue
                suffix = entry.suffix.lower()
                if suffix not in _ALLOWED_CHECKPOINT_EXTS:
                    continue
                lower = entry.name.lower()
                if any(lower.endswith(suf) for suf in _CHECKPOINT_BLACKLIST_SUFFIXES):
                    continue
                yield entry

    def _hash_for(self, path: Path) -> Tuple[str | None, str | None]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None, None
        key = str(path)
        cache = self._ensure_hash_cache()
        entry = cache.get(key)
        # Cache hit: validate by mtime AND size (both must match)
        if entry and entry.mtime == stat.st_mtime and entry.size == stat.st_size:
            sha256 = entry.sha256
            short_hash = entry.short_hash or None
            return sha256, short_hash
        # Cache miss: compute hash (slow path, but only happens once per file)
        try:
            _LOGGER.debug("computing sha256 for %s (%.1f MB)", path.name, stat.st_size / 1e6)
            sha256 = _sha256(path)
            short_hash = sha256[:10]
        except Exception:
            sha256 = None
            short_hash = None
        if sha256:
            cache[key] = _HashCacheEntry(stat.st_mtime, stat.st_size, sha256, short_hash or "")
            self._hash_cache_dirty = True  # Mark for persistence
        return sha256, short_hash


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _DEFAULT_REGISTRY


def list_checkpoints(*, refresh: bool = False) -> List[CheckpointRecord]:
    return _DEFAULT_REGISTRY.list_checkpoints(refresh=refresh)


def list_vaes(*, refresh: bool = False) -> List[VAERecord]:
    return _DEFAULT_REGISTRY.list_vaes(refresh=refresh)


def refresh() -> None:
    _DEFAULT_REGISTRY.refresh()


__all__ = [
    "ModelRegistry",
    "get_registry",
    "list_checkpoints",
    "list_vaes",
    "refresh",
]
