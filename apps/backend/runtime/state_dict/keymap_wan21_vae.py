"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN2.1 VAE key-style detection + strict canonical remap.
Normalizes wrapper prefixes for WAN2.1 VAE checkpoints and validates canonical
key ownership fail-loud before model load.

Symbols (top-level; keep in sync; no ghosts):
- `remap_wan21_vae_state_dict` (function): Returns `(detected_style, remapped_view)` for WAN2.1 VAE keys.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    SentinelKind,
    remap_state_dict_view,
    strip_repeated_prefixes,
)

_T = TypeVar("_T")

_WAN21_VAE_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)
_WAN21_VAE_REQUIRED = (
    "decoder.head.0.gamma",
    "encoder.conv1.weight",
    "decoder.conv1.weight",
    "conv1.weight",
    "conv2.weight",
)
_WAN21_VAE_FORBIDDEN_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)

_WAN21_VAE_DETECTOR = KeyStyleDetector(
    name="wan21_vae_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder."),
                KeySentinel(SentinelKind.PREFIX, "decoder."),
                KeySentinel(SentinelKind.EXACT, "decoder.head.0.gamma"),
                KeySentinel(SentinelKind.EXACT, "conv1.weight"),
                KeySentinel(SentinelKind.EXACT, "conv2.weight"),
            ),
            min_sentinel_hits=2,
        ),
    ),
)


def _validate_required_keys(*, keys: Sequence[str], required: Sequence[str], detector_name: str) -> None:
    keys_set = frozenset(keys)
    missing = [key for key in required if key not in keys_set]
    if missing:
        raise KeyMappingError(
            f"{detector_name}: remap output is missing required canonical keys. "
            f"missing_sample={missing[:10]}"
        )


def _validate_forbidden_prefixes(*, keys: Sequence[str], prefixes: Sequence[str], detector_name: str) -> None:
    offenders = [key for key in keys if key.startswith(tuple(prefixes))]
    if offenders:
        raise KeyMappingError(
            f"{detector_name}: remap produced non-canonical keys with forbidden prefixes. "
            f"offenders_sample={sorted(offenders)[:10]}"
        )


def remap_wan21_vae_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _WAN21_VAE_PREFIXES)

    def _validate_output(keys: Sequence[str]) -> None:
        _validate_forbidden_prefixes(
            keys=keys,
            prefixes=_WAN21_VAE_FORBIDDEN_PREFIXES,
            detector_name=_WAN21_VAE_DETECTOR.name,
        )
        _validate_required_keys(
            keys=keys,
            required=_WAN21_VAE_REQUIRED,
            detector_name=_WAN21_VAE_DETECTOR.name,
        )

    mappers = {KeyStyle.CODEX: lambda key: key}
    return remap_state_dict_view(
        state_dict,
        detector=_WAN21_VAE_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        output_validator=_validate_output,
    )


__all__ = ["remap_wan21_vae_state_dict"]
