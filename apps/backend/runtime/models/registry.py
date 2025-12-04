"""Checkpoint / VAE discovery with caching and telemetry."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime import trace as _trace

from .types import (
    CheckpointFormat,
    CheckpointPrediction,
    CheckpointRecord,
    VAERecord,
)

_LOGGER = logging.getLogger("backend.registry")

_ALLOWED_CHECKPOINT_EXTS = {".ckpt", ".safetensors", ".pt", ".bin", ".gguf"}
_CHECKPOINT_BLACKLIST_SUFFIXES = {".vae.ckpt", ".vae.safetensors", ".vae.pt"}
_VAE_EXTS = {".safetensors", ".ckpt", ".pt"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


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


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


@dataclass
class _HashCacheEntry:
    mtime: float
    sha256: str
    short_hash: str


class ModelRegistry:
    """Discover checkpoint/VAE assets and expose cached views."""

    def __init__(self, *, models_root: Path | None = None, hf_root: Path | None = None) -> None:
        self._models_root = Path(models_root or _default_models_root()).resolve()
        self._hf_root = Path(hf_root or _default_hf_root()).resolve()
        self._lock = threading.Lock()
        self._checkpoints: Dict[str, CheckpointRecord] = {}
        self._vaes: Dict[str, VAERecord] = {}
        self._hash_cache: Dict[str, _HashCacheEntry] = {}
        self._last_scan: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

        if self._hf_root.is_dir():
            for record in self._scan_diffusers():
                yield record

    def _scan_vaes(self) -> Iterable[VAERecord]:
        vae_root = self._models_root / "VAE"
        candidates: List[Path] = []
        if vae_root.is_dir():
            candidates.extend([p for p in vae_root.rglob("*") if p.is_file() and p.suffix.lower() in _VAE_EXTS])
        # Also consider checkpoints beside model files (suffix .vae.*)
        for file in self._iter_checkpoint_files():
            lower = file.name.lower()
            if any(lower.endswith(suf) for suf in _CHECKPOINT_BLACKLIST_SUFFIXES):
                candidates.append(file)

        seen: set[str] = set()
        for path in candidates:
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
        1) Explicit roots from apps/paths.json per engine (sd15_ckpt, sdxl_ckpt, flux_ckpt, wan22_ckpt).
        2) Built-in defaults under models/: root, sd15, sdxl, flux.

        This replaces the legacy scatter of A1111-style folders ('stable-diffusion', 'sd', 'checkpoints').
        """
        candidates: List[Path] = []

        # 1) User overrides from apps/paths.json per engine
        try:
            for key in ("sd15_ckpt", "sdxl_ckpt", "flux_ckpt", "wan22_ckpt"):
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

    def _scan_diffusers(self) -> Iterable[CheckpointRecord]:
        try:
            orgs = [d for d in self._hf_root.iterdir() if d.is_dir()]
        except FileNotFoundError:
            return []
        output: List[CheckpointRecord] = []
        for org in orgs:
            for repo in org.iterdir():
                if not repo.is_dir():
                    continue
                model_index = repo / "model_index.json"
                unet_dir = repo / "unet"
                transformer_dir = repo / "transformer"
                if not (model_index.is_file() or unet_dir.is_dir() or transformer_dir.is_dir()):
                    continue
                name = f"{org.name}/{repo.name}"
                metadata: Dict[str, object] = {"format": CheckpointFormat.DIFFUSERS.value}
                components: List[str] = []
                for sub in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
                    if (repo / sub).is_dir():
                        components.append(sub)
                default_dtype = None
                base_resolution = None
                prediction = CheckpointPrediction.UNKNOWN
                if model_index.is_file():
                    cfg = _read_json(model_index)
                    if isinstance(cfg.get("torch_dtype"), str):
                        default_dtype = str(cfg["torch_dtype"])
                unet_cfg = repo / "unet" / "config.json"
                if unet_cfg.is_file():
                    cfg = _read_json(unet_cfg)
                    sample_size = cfg.get("sample_size")
                    if isinstance(sample_size, int):
                        base_resolution = int(sample_size)
                scheduler_cfg = repo / "scheduler" / "config.json"
                if scheduler_cfg.is_file():
                    cfg = _read_json(scheduler_cfg)
                    pt = cfg.get("prediction_type")
                    if isinstance(pt, str):
                        pt_lower = pt.lower()
                        if pt_lower == "epsilon":
                            prediction = CheckpointPrediction.EPSILON
                        elif pt_lower == "v_prediction" or pt_lower == "vprediction":
                            prediction = CheckpointPrediction.V_PREDICTION
                        elif pt_lower == "edm":
                            prediction = CheckpointPrediction.EDM
                stat = model_index.stat() if model_index.is_file() else repo.stat()
                output.append(
                    CheckpointRecord(
                        name=name,
                        title=name,
                        filename=str(model_index if model_index.is_file() else repo),
                        path=str(repo),
                        model_name=repo.name,
                        format=CheckpointFormat.DIFFUSERS,
                        default_dtype=default_dtype,
                        base_resolution=base_resolution,
                        prediction_type=prediction,
                        components=tuple(components),
                        metadata=metadata,
                        updated_at=stat.st_mtime,
                    )
                )
        return output

    def _hash_for(self, path: Path) -> Tuple[str | None, str | None]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None, None
        key = str(path)
        entry = self._hash_cache.get(key)
        if entry and entry.mtime == stat.st_mtime:
            sha256 = entry.sha256
            short_hash = entry.short_hash or None
            return sha256, short_hash
        try:
            sha256 = _sha256(path)
            short_hash = sha256[:10]
        except Exception:
            sha256 = None
            short_hash = None
        if sha256:
            self._hash_cache[key] = _HashCacheEntry(stat.st_mtime, sha256, short_hash or "")
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
