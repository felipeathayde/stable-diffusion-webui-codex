"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL checkpoint wrapper/prefix keyspace resolver (checkpoint-wrapper / original SDXL layout).
Provides a strict, fail-loud resolver that canonicalizes common wrapper prefixes (DDP `module.`, duplicated `model.model.*`,
`diffusion_model.*`, `vae.*`) and SDXL UNet nested label-embedding keys into canonical prefixes used by Codex detectors/parser plans.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_sdxl_checkpoint_keyspace` (function): Resolves SDXL checkpoint wrapper/prefix normalization into canonical keyspace.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
    resolve_state_dict_keyspace,
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


def resolve_sdxl_checkpoint_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    """Normalize common SDXL checkpoint wrapper prefixes into canonical keys.

    This is intentionally string-only and import-light. It does **not** attempt to convert
    diffusers-style component weights into original SDXL keys; it only normalizes wrapper
    prefixes (and known nested label-embedding key layouts) so Codex detectors and parser
    plans can match reliably.
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
        if k.startswith("model.diffusion_model.label_emb.0."):
            suffix = k[len("model.diffusion_model.label_emb.0.") :]
            suffix_parts = suffix.split(".")
            if len(suffix_parts) >= 2 and suffix_parts[0].isdigit():
                k = "model.diffusion_model.label_emb." + ".".join([suffix_parts[0], *suffix_parts[1:]])

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

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_DETECTOR,
        normalize=_normalize,
        mappers={
            KeyStyle.CODEX: lambda k: k,
        },
    )
    resolved.metadata.setdefault("resolver", "sdxl_checkpoint")
    return resolved


__all__ = ["resolve_sdxl_checkpoint_keyspace"]
