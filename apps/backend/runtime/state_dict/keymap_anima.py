"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Strict key-style detection + remapping for Anima core transformer, WAN VAE, and Qwen3-0.6B text encoder.
Normalizes explicit wrapper prefixes only, preserves canonical keyspaces, and fails loud on unsupported/collision output.
Current detectors are intentionally single-style per component; ambiguous-style errors become reachable only if additional styles are added later.

Symbols (top-level; keep in sync; no ghosts):
- `remap_anima_transformer_state_dict` (function): Returns (detected_style, remapped_view) for Anima core transformer keys.
- `remap_anima_wan_vae_state_dict` (function): Returns (detected_style, remapped_view) for Anima WAN VAE keys.
- `remap_anima_qwen3_06b_state_dict` (function): Returns (detected_style, remapped_view) for Anima Qwen3-0.6B text-encoder keys.
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

_TRANSFORMER_PREFIXES = (
    "module.",
)
_TRANSFORMER_REQUIRED = (
    "x_embedder.proj.1.weight",
    "t_embedder.1.linear_1.weight",
    "blocks.0.self_attn.q_proj.weight",
    "blocks.0.cross_attn.k_proj.weight",
    "final_layer.linear.weight",
)
_TRANSFORMER_FORBIDDEN_PREFIXES = (
    "net.",
    "model.",
    "module.",
)

_WAN_VAE_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)
_WAN_VAE_REQUIRED = (
    "decoder.head.0.gamma",
    "encoder.conv1.weight",
    "decoder.conv1.weight",
    "conv1.weight",
    "conv2.weight",
)
_WAN_VAE_FORBIDDEN_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)

_QWEN_PREFIXES = (
    "module.",
    "text_encoder.",
)
_QWEN_REQUIRED = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.norm.weight",
)

_TRANSFORMER_DETECTOR = KeyStyleDetector(
    name="anima_transformer_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "x_embedder."),
                KeySentinel(SentinelKind.PREFIX, "blocks."),
                KeySentinel(SentinelKind.PREFIX, "final_layer."),
                KeySentinel(SentinelKind.PREFIX, "llm_adapter."),
            ),
            min_sentinel_hits=2,
        ),
    ),
)

_WAN_VAE_DETECTOR = KeyStyleDetector(
    name="anima_wan_vae_key_style",
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

_QWEN_DETECTOR = KeyStyleDetector(
    name="anima_qwen3_06b_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "model.embed_tokens.weight"),
                KeySentinel(SentinelKind.PREFIX, "model.layers."),
                KeySentinel(SentinelKind.EXACT, "model.norm.weight"),
            ),
            min_sentinel_hits=2,
        ),
    ),
)


def _validate_required_keys(*, keys: Sequence[str], required: Sequence[str], detector_name: str) -> None:
    keys_set = frozenset(keys)
    missing = [k for k in required if k not in keys_set]
    if missing:
        raise KeyMappingError(
            f"{detector_name}: remap output is missing required canonical keys. "
            f"missing_sample={missing[:10]}"
        )


def _validate_forbidden_prefixes(*, keys: Sequence[str], prefixes: Sequence[str], detector_name: str) -> None:
    offenders = [k for k in keys if k.startswith(tuple(prefixes))]
    if offenders:
        raise KeyMappingError(
            f"{detector_name}: remap produced non-canonical keys with forbidden prefixes. "
            f"offenders_sample={sorted(offenders)[:10]}"
        )


def remap_anima_transformer_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _TRANSFORMER_PREFIXES)

    def _validate_output(keys: Sequence[str]) -> None:
        _validate_forbidden_prefixes(
            keys=keys,
            prefixes=_TRANSFORMER_FORBIDDEN_PREFIXES,
            detector_name=_TRANSFORMER_DETECTOR.name,
        )
        _validate_required_keys(
            keys=keys,
            required=_TRANSFORMER_REQUIRED,
            detector_name=_TRANSFORMER_DETECTOR.name,
        )

    mappers = {KeyStyle.CODEX: lambda key: key}
    return remap_state_dict_view(
        state_dict,
        detector=_TRANSFORMER_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        output_validator=_validate_output,
    )


def remap_anima_wan_vae_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _WAN_VAE_PREFIXES)

    def _validate_output(keys: Sequence[str]) -> None:
        _validate_forbidden_prefixes(
            keys=keys,
            prefixes=_WAN_VAE_FORBIDDEN_PREFIXES,
            detector_name=_WAN_VAE_DETECTOR.name,
        )
        _validate_required_keys(
            keys=keys,
            required=_WAN_VAE_REQUIRED,
            detector_name=_WAN_VAE_DETECTOR.name,
        )

    mappers = {KeyStyle.CODEX: lambda key: key}
    return remap_state_dict_view(
        state_dict,
        detector=_WAN_VAE_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        output_validator=_validate_output,
    )


def remap_anima_qwen3_06b_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _QWEN_PREFIXES)

    def _validate_output(keys: Sequence[str]) -> None:
        _validate_required_keys(
            keys=keys,
            required=_QWEN_REQUIRED,
            detector_name=_QWEN_DETECTOR.name,
        )
        offenders = [k for k in keys if not str(k).startswith("model.")]
        if offenders:
            raise KeyMappingError(
                f"{_QWEN_DETECTOR.name}: remap produced non-canonical keys outside model.* keyspace. "
                f"offenders_sample={sorted(offenders)[:10]}"
            )

    mappers = {KeyStyle.HF: lambda key: key}
    return remap_state_dict_view(
        state_dict,
        detector=_QWEN_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        output_validator=_validate_output,
    )


__all__ = [
    "remap_anima_qwen3_06b_state_dict",
    "remap_anima_transformer_state_dict",
    "remap_anima_wan_vae_state_dict",
]
