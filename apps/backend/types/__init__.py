"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend type facade and shared constants.
Re-exports sampler enums, payload key sets, and lazy export group definitions used across backend modules.

Symbols (top-level; keep in sync; no ghosts):
- `SamplerKind` (enum): Canonical sampler identifiers (re-export).
- `ApplyOutcome` (dataclass): Outcome container for applying sampler/scheduler selection (re-export).
- `ShaKeys` (dataclass): Frozen key sets for SHA-related payload fields (re-export).
- `Txt2ImgKeys` (dataclass): Frozen key sets for txt2img payload fields (re-export).
- `ExtrasKeys` (dataclass): Frozen key sets for extras payload fields (re-export).
- `SHA_KEYS` (constant): Singleton instance of `ShaKeys` (re-export).
- `TXT2IMG_KEYS` (constant): Singleton instance of `Txt2ImgKeys` (re-export).
- `EXTRAS_KEYS` (constant): Singleton instance of `ExtrasKeys` (re-export).
- `LazyExports` (dataclass): Frozen export groups loaded lazily (re-export).
- `LAZY_EXPORTS` (constant): Singleton instance of `LazyExports` (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .samplers import SamplerKind, ApplyOutcome
from .payloads import (
    ShaKeys,
    Txt2ImgKeys,
    ExtrasKeys,
    SHA_KEYS,
    TXT2IMG_KEYS,
    EXTRAS_KEYS,
)
from .exports import LazyExports, LAZY_EXPORTS

__all__ = [
    "SamplerKind",
    "ApplyOutcome",
    "ShaKeys",
    "Txt2ImgKeys",
    "ExtrasKeys",
    "SHA_KEYS",
    "TXT2IMG_KEYS",
    "EXTRAS_KEYS",
    "LazyExports",
    "LAZY_EXPORTS",
]
