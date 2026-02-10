"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical key-style detection + remap for T5 text-encoder state_dict keys.
Provides strict, fail-loud mapping from HF-style T5 keys (`encoder.*`, `shared.weight`, `embed_tokens*`)
into Codex IntegratedT5 layout (`transformer.*`) so loader paths do not perform ad-hoc remap logic.

Symbols (top-level; keep in sync; no ghosts):
- `remap_t5_text_encoder_state_dict` (function): Remaps a T5 encoder state_dict into canonical IntegratedT5 keys.
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
    name="t5_text_encoder_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder."),
                KeySentinel(SentinelKind.EXACT, "shared.weight"),
                KeySentinel(SentinelKind.PREFIX, "embed_tokens"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def remap_t5_text_encoder_state_dict(
    state_dict: MutableMapping[str, _T],
) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    """Return a view that remaps T5 text-encoder keys into canonical IntegratedT5 keys.

    - CODEX style (`transformer.*`) is a no-op.
    - HF style (`encoder.*`, `shared.weight`, `embed_tokens*`) is remapped to `transformer.*`.
    - Unknown/ambiguous styles fail loud via key-style detection.
    """

    def _normalize(key: str) -> str:
        return str(key)

    def _map_hf(key: str) -> str:
        if key.startswith("encoder.") or key == "shared.weight" or key.startswith("embed_tokens"):
            return f"transformer.{key}"
        return key

    mappers = {
        KeyStyle.CODEX: lambda k: k,
        KeyStyle.HF: _map_hf,
    }

    return remap_state_dict_view(
        state_dict,
        detector=_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
    )

