"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Declarative key-style detection + key remapping utilities for checkpoint state_dicts.
Provides a strict, fail-loud mapping core used by model-family loaders to normalize multiple upstream key layouts into one canonical runtime layout
without ad-hoc string-replace chains or silent fallbacks.

Symbols (top-level; keep in sync; no ghosts):
- `KeyMappingError` (exception): Raised when key-style detection or remapping fails (unknown style, ambiguous style, collisions).
- `KeyStyleDetectionError` (exception): Raised when key-style detection fails (unknown/ambiguous layout).
- `KeyStyle` (enum): Stable identifiers for common key layouts (Codex, Diffusers, LDM, OpenCLIP, WAN export, llama.cpp GGUF, HF).
- `SentinelKind` (enum): Sentinel matching strategy (exact/prefix/substring/regex).
- `KeySentinel` (dataclass): A single “style signal” used to detect a key layout.
- `KeyStyleSpec` (dataclass): A named key-style + its sentinel set and matching threshold.
- `KeyStyleDetector` (dataclass): Detects a style from a key list with strict ambiguity handling.
- `strip_repeated_prefixes` (function): Removes known wrapper prefixes repeatedly (e.g. `model.diffusion_model.`).
- `remap_key` (function): Detects style for a single key and returns the remapped key.
- `remap_state_dict_view` (function): Detects style for a state_dict and returns a lazy `RemapKeysView` over it (with collision checks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, TypeVar

from apps.backend.runtime.state_dict.views import RemapKeysView

_T = TypeVar("_T")


class KeyMappingError(RuntimeError):
    pass


class KeyStyleDetectionError(KeyMappingError):
    pass


class KeyStyle(str, Enum):
    CODEX = "codex"
    DIFFUSERS = "diffusers"
    LDM = "ldm"
    OPENCLIP = "openclip"
    WAN_EXPORT = "wan_export"
    LLAMA_GGUF = "llama_gguf"
    HF = "hf"


class SentinelKind(str, Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    SUBSTRING = "substring"
    REGEX = "regex"


@dataclass(frozen=True, slots=True)
class KeySentinel:
    kind: SentinelKind
    pattern: str
    description: str = ""
    _compiled: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        if self.kind is SentinelKind.REGEX:
            object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def matches(self, key: str, *, keys_set: frozenset[str] | None = None) -> bool:
        if self.kind is SentinelKind.EXACT:
            if keys_set is None:
                return key == self.pattern
            return self.pattern in keys_set
        if self.kind is SentinelKind.PREFIX:
            return key.startswith(self.pattern)
        if self.kind is SentinelKind.SUBSTRING:
            return self.pattern in key
        if self.kind is SentinelKind.REGEX:
            assert self._compiled is not None
            return self._compiled.search(key) is not None
        raise KeyMappingError(f"Unknown sentinel kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class KeyStyleSpec:
    style: KeyStyle
    sentinels: tuple[KeySentinel, ...]
    min_sentinel_hits: int = 1


@dataclass(frozen=True, slots=True)
class KeyStyleDetector:
    name: str
    styles: tuple[KeyStyleSpec, ...]

    def detect(self, keys: Sequence[str]) -> KeyStyle:
        if not self.styles:
            raise KeyStyleDetectionError(f"{self.name}: no styles configured")
        if not keys:
            raise KeyStyleDetectionError(f"{self.name}: empty key list; cannot detect key style")

        keys_set: frozenset[str] = frozenset(keys)

        hits: dict[KeyStyle, int] = {}
        for spec in self.styles:
            hit_count = 0
            for sentinel in spec.sentinels:
                if sentinel.kind is SentinelKind.EXACT:
                    if sentinel.matches("", keys_set=keys_set):
                        hit_count += 1
                    continue
                if any(sentinel.matches(k, keys_set=keys_set) for k in keys):
                    hit_count += 1
            hits[spec.style] = hit_count

        matched: list[tuple[KeyStyle, int]] = []
        for spec in self.styles:
            score = hits.get(spec.style, 0)
            if score >= spec.min_sentinel_hits:
                matched.append((spec.style, score))

        preview = ", ".join(sorted(keys)[:10])
        if not matched:
            expected = "; ".join(
                f"{spec.style.value}: [{', '.join(s.pattern for s in spec.sentinels)}]"
                for spec in self.styles
            )
            raise KeyStyleDetectionError(
                f"{self.name}: could not detect key style (no sentinels matched). "
                f"expected one of: {expected}. sample_keys=[{preview}]"
            )

        if len(matched) == 1:
            return matched[0][0]

        max_score = max(score for _, score in matched)
        winners = [style for style, score in matched if score == max_score]
        if len(winners) == 1:
            return winners[0]

        scored = ", ".join(f"{style.value}={score}" for style, score in matched)
        raise KeyStyleDetectionError(
            f"{self.name}: ambiguous key style detection (matched multiple styles: {scored}). "
            f"sample_keys=[{preview}]"
        )


def strip_repeated_prefixes(name: str, prefixes: tuple[str, ...]) -> str:
    out = name
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
                break
    return out


def remap_key(
    key: str,
    *,
    detector: KeyStyleDetector,
    normalize: Callable[[str], str],
    mappers: Mapping[KeyStyle, Callable[[str], str]],
) -> str:
    normalized = normalize(key)
    style = detector.detect([normalized])
    mapper = mappers.get(style)
    if mapper is None:
        raise KeyMappingError(f"{detector.name}: no mapper registered for style={style.value!r}")
    return mapper(normalized)


def remap_state_dict_view(
    state_dict: MutableMapping[str, _T],
    *,
    detector: KeyStyleDetector,
    normalize: Callable[[str], str],
    mappers: Mapping[KeyStyle, Callable[[str], str]],
    view_factory: Callable[[MutableMapping[str, _T], dict[str, str]], MutableMapping[str, _T]] | None = None,
    output_validator: Callable[[Sequence[str]], None] | None = None,
) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    keys = list(state_dict.keys())
    normalized_keys = [normalize(k) for k in keys]
    style = detector.detect(normalized_keys)

    mapper = mappers.get(style)
    if mapper is None:
        raise KeyMappingError(f"{detector.name}: no mapper registered for style={style.value!r}")

    mapping: dict[str, str] = {}
    for original_key, normalized_key in zip(keys, normalized_keys, strict=True):
        remapped_key = mapper(normalized_key)
        previous = mapping.get(remapped_key)
        if previous is not None and previous != original_key:
            raise KeyMappingError(
                f"{detector.name}: multiple source keys map to the same destination key: "
                f"dst={remapped_key!r} srcs={previous!r},{original_key!r}"
            )
        mapping[remapped_key] = original_key

    if output_validator is not None:
        output_validator(list(mapping.keys()))

    factory = view_factory or (lambda base, mapping: RemapKeysView(base, mapping))
    return style, factory(state_dict, mapping)
