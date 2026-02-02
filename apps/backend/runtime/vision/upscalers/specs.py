"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed upscaler specs (global, backend-facing).
Defines stable identifiers and config dataclasses for builtin latent upscalers and file-backed Spandrel SR models.

Symbols (top-level; keep in sync; no ghosts):
- `UpscalerKind` (enum): Upscaler kind (`latent`, `spandrel`).
- `TileConfig` (dataclass): Tile config (tile/overlap + fallback-on-OOM).
- `UpscalerDefinition` (dataclass): Resolved upscaler entry (id/label/kind/meta).
- `LatentUpscaleMode` (dataclass): Latent upscale mode (torch interpolate mode + antialias).
- `LATENT_UPSCALE_MODES` (constant): Built-in latent upscale modes (A1111-aligned labels).
- `tile_config_from_payload` (function): Parses a tile config payload dict into a validated `TileConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class UpscalerKind(str, Enum):
    LATENT = "latent"
    SPANDREL = "spandrel"


@dataclass(frozen=True, slots=True)
class TileConfig:
    """Tiling configuration for SR models.

    tile: tile size in pixels (square tile for v1).
    overlap: overlap in pixels (feathered blend in the stitcher).
    fallback_on_oom: when True, automatically halves tile size on OOM until a minimum threshold.
    """

    tile: int = 256
    overlap: int = 16
    fallback_on_oom: bool = True
    min_tile: int = 128


@dataclass(frozen=True, slots=True)
class UpscalerDefinition:
    """A stable upscaler entry exposed to the UI and to workflow stages."""

    id: str
    label: str
    kind: UpscalerKind
    meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class LatentUpscaleMode:
    mode: str
    antialias: bool


LATENT_UPSCALE_MODES: Mapping[str, LatentUpscaleMode] = {
    # Keep ids stable; labels can be changed separately by the UI.
    "latent:nearest": LatentUpscaleMode(mode="nearest", antialias=False),
    "latent:bilinear": LatentUpscaleMode(mode="bilinear", antialias=False),
    "latent:bilinear-aa": LatentUpscaleMode(mode="bilinear", antialias=True),
    "latent:bicubic": LatentUpscaleMode(mode="bicubic", antialias=False),
    "latent:bicubic-aa": LatentUpscaleMode(mode="bicubic", antialias=True),
}


def default_tile_config() -> TileConfig:
    return TileConfig()

def tile_config_from_payload(tile_raw: Any, *, context: str = "tile") -> TileConfig:
    """Parse a tile config payload dict into a `TileConfig`.

    This parser is shared by standalone `/upscale` and hires-fix, keeping the contract consistent:
    `{tile, overlap, fallback_on_oom, min_tile}`.
    """

    if tile_raw is None:
        return default_tile_config()
    if not isinstance(tile_raw, dict):
        raise ValueError(f"Invalid {context} (must be object)")

    def _int(name: str, default: int) -> int:
        if name not in tile_raw:
            return int(default)
        value = tile_raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid {context}.{name} (must be int)")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Invalid {context}.{name} (must be int)")
        return int(value)

    def _bool(name: str, default: bool) -> bool:
        if name not in tile_raw:
            return bool(default)
        value = tile_raw.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    cfg = TileConfig(
        tile=_int("tile", 256),
        overlap=_int("overlap", 16),
        fallback_on_oom=_bool("fallback_on_oom", True),
        min_tile=_int("min_tile", 128),
    )

    # Keep invariants aligned with the runtime upscalers registry.
    tile_size = int(cfg.tile)
    overlap = int(cfg.overlap)
    min_tile = int(cfg.min_tile)
    fallback = bool(cfg.fallback_on_oom)
    if tile_size < 0:
        raise ValueError(f"Invalid {context}.tile (must be >= 0)")
    if overlap < 0:
        raise ValueError(f"Invalid {context}.overlap (must be >= 0)")
    if min_tile <= 0:
        raise ValueError(f"Invalid {context}.min_tile (must be > 0)")
    if tile_size > 0 and overlap >= tile_size:
        raise ValueError(f"Invalid {context}.overlap (must be < {context}.tile)")
    if fallback and tile_size > 0 and min_tile > tile_size:
        raise ValueError(f"Invalid {context}.min_tile (must be <= {context}.tile when fallback_on_oom is enabled)")

    return cfg


__all__ = [
    "UpscalerKind",
    "TileConfig",
    "UpscalerDefinition",
    "LatentUpscaleMode",
    "LATENT_UPSCALE_MODES",
    "default_tile_config",
    "tile_config_from_payload",
]
