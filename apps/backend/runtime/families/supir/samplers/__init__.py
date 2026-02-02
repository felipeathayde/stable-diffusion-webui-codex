"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR sampler registry package.
Defines SUPIR sampler IDs/specs and a small registry API used by the SUPIR enhance runner.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Public exports for sampler IDs/specs.
"""

from __future__ import annotations

from .registry import list_supir_samplers, resolve_supir_sampler
from .types import SupirSamplerId, SupirSamplerSpec

__all__ = [
    "SupirSamplerId",
    "SupirSamplerSpec",
    "list_supir_samplers",
    "resolve_supir_sampler",
]
