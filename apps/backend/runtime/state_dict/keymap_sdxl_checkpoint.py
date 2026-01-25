"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL checkpoint wrapper/prefix key normalization (Comfy/original SDXL layout).
Provides a strict, fail-loud remap that canonicalizes common wrapper prefixes (DDP `module.`, duplicated `model.model.*`,
`diffusion_model.*`, `vae.*`) into canonical prefixes used by Codex detectors/parser plans.

Symbols (top-level; keep in sync; no ghosts):
- `remap_sdxl_checkpoint_state_dict` (function): Returns (detected_style, remapped_view) for SDXL checkpoint wrapper/prefix normalization.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    SentinelKind,
    remap_state_dict_view,
)

_T = TypeVar("_T")


_DETECTOR = KeyStyleDetector(
    name="sdxl_checkpoint_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "model.diffusion_model."),
                KeySentinel(SentinelKind.PREFIX, "conditioner.embedders."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def remap_sdxl_checkpoint_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    """Normalize common SDXL checkpoint wrapper prefixes into canonical keys.

    This is intentionally string-only and import-light. It does **not** attempt to convert
    diffusers-style component weights into original SDXL keys; it only normalizes wrapper
    prefixes so Codex detectors and parser plans can match reliably.
    """

    def _normalize(key: str) -> str:
        k = str(key)

        # DDP wrappers.
        while k.startswith("module."):
            k = k[len("module.") :]

        # UNet wrappers.
        while k.startswith("model.model.diffusion_model."):
            k = "model.diffusion_model." + k[len("model.model.diffusion_model.") :]
        if k.startswith("diffusion_model."):
            k = "model.diffusion_model." + k[len("diffusion_model.") :]

        # Conditioner wrappers.
        while k.startswith("model.model.conditioner."):
            k = "conditioner." + k[len("model.model.conditioner.") :]
        if k.startswith("model.conditioner."):
            k = "conditioner." + k[len("model.conditioner.") :]

        # VAE wrappers.
        while k.startswith("model.model.first_stage_model."):
            k = "first_stage_model." + k[len("model.model.first_stage_model.") :]
        if k.startswith("model.first_stage_model."):
            k = "first_stage_model." + k[len("model.first_stage_model.") :]
        while k.startswith("model.model.vae."):
            k = "vae." + k[len("model.model.vae.") :]
        if k.startswith("model.vae."):
            k = "vae." + k[len("model.vae.") :]

        # Canonicalize VAE prefix to the detector/parser default.
        if k.startswith("vae."):
            k = "first_stage_model." + k[len("vae.") :]

        return k

    mappers = {
        KeyStyle.CODEX: lambda k: k,
    }

    return remap_state_dict_view(
        state_dict,
        detector=_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
    )
