"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL base CLIP key-style detection + remapping (CLIP-L + CLIP-G) into Codex IntegratedCLIP state_dict layout.
Supports HF `text_model.*`, OpenCLIP legacy `transformer.resblocks.*`, and Codex-canonical `transformer.text_model.*` keys.
Strips wrapper prefixes and normalizes known buffers/weights, failing loud on unknown non-weight keys.

Symbols (top-level; keep in sync; no ghosts):
- `remap_sdxl_clip_l_state_dict` (function): Keymap for SDXL base CLIP-L (`text_encoder`) into Codex IntegratedCLIP keys.
- `remap_sdxl_clip_g_state_dict` (function): Keymap for SDXL base CLIP-G (`text_encoder_2`) into Codex IntegratedCLIP keys.

Notes: Target keyspace matches `apps/backend/runtime/common/nn/clip.py:IntegratedCLIP` (and related Codex CLIP wrappers).
Key policies (non-exhaustive): strip known wrapper prefixes; drop HF-only buffers (`*.position_ids`) and refuse other unknown non-weight keys;
canonicalize `logit_scale` (default `ln(100)`); canonicalize optional projection weights into `transformer.text_projection.weight` (CLIP-G; lazy transpose);
for OpenCLIP-style fused attention weights (`attn.in_proj_{weight,bias}`), expose Q/K/V projections as lazy slices.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from math import log
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    KeySentinel,
    SentinelKind,
    strip_repeated_prefixes,
)

_T = TypeVar("_T")

_WRAPPER_PREFIXES: tuple[str, ...] = (
    "conditioner.embedders.0.transformer.",
    "conditioner.embedders.0.model.",
    "conditioner.embedders.0.",
    "conditioner.embedders.1.transformer.",
    "conditioner.embedders.1.model.",
    "conditioner.embedders.1.",
    "cond_stage_model.model.",
    "cond_stage_model.",
    "text_encoders.clip_l.",
    "text_encoders.clip_g.",
    "clip_l.",
    "clip_g.",
    "model.text_model.",
    "model.",
)

_LOGIT_KEYS: tuple[str, ...] = (
    "logit_scale",
    "transformer.logit_scale",
    "transformer.text_model.logit_scale",
)

_PROJ_KEYS: tuple[str, ...] = (
    "transformer.text_projection.weight",
    "transformer.text_projection",
    "text_projection.weight",
    "text_projection",
)

_DETECTOR = KeyStyleDetector(
    name="sdxl_clip_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer.text_model.embeddings."),
                KeySentinel(SentinelKind.PREFIX, "transformer.text_model.encoder.layers."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "text_model.embeddings."),
                KeySentinel(SentinelKind.PREFIX, "text_model.encoder.layers."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.OPENCLIP,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer.resblocks."),
                KeySentinel(SentinelKind.EXACT, "token_embedding.weight"),
                KeySentinel(SentinelKind.EXACT, "positional_embedding"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)

_ESSENTIAL_KEYS: tuple[str, ...] = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.embeddings.position_embedding.weight",
    "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "transformer.text_model.final_layer_norm.weight",
)


@dataclass(frozen=True, slots=True)
class _Direct:
    key: str


@dataclass(frozen=True, slots=True)
class _SliceQKV:
    key: str
    index: int  # 0=q,1=k,2=v


@dataclass(frozen=True, slots=True)
class _Transpose:
    key: str


@dataclass(frozen=True, slots=True)
class _DefaultLogitScale:
    value: float


_Spec = _Direct | _SliceQKV | _Transpose | _DefaultLogitScale


class _SDXLCLIPKeymapView(MutableMapping[str, _T]):
    """Lazy key mapping view supporting QKV slicing + projection transpose."""

    def __init__(self, base: MutableMapping[str, _T], mapping: dict[str, _Spec]):
        self._base = base
        self._map = dict(mapping)
        self._keys = tuple(mapping.keys())
        # Cache only derived tensors (slices/defaults) so we don't keep the whole
        # text encoder resident when streaming from SafeTensors.
        self._derived_cache: dict[str, _T] = {}
        self._source_cache: dict[str, _T] = {}

    @staticmethod
    def _slice_in_proj(value: _T, index: int) -> _T:
        try:
            shape = getattr(value, "shape", None)
            if not shape:
                return value
            if len(shape) < 1:
                return value
            total = int(shape[0])
            if total % 3 != 0:
                raise KeyMappingError(f"OpenCLIP in_proj first dim is not divisible by 3 (shape={shape!r})")
            chunk = total // 3
            start = index * chunk
            end = (index + 1) * chunk
            return value[start:end]
        except Exception as exc:
            raise KeyMappingError(f"Failed to slice OpenCLIP in_proj tensor (index={index})") from exc

    @staticmethod
    def _transpose_2d(value: _T) -> _T:
        try:
            ndim = getattr(value, "ndim", None)
            if ndim != 2:
                return value
            return value.transpose(0, 1)
        except Exception as exc:
            raise KeyMappingError("Failed to transpose projection tensor") from exc

    def __getitem__(self, k: str) -> _T:
        spec = self._map[k]
        if isinstance(spec, _Direct):
            return self._base[spec.key]
        elif isinstance(spec, _SliceQKV):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            base_tensor = self._source_cache.get(spec.key)
            if base_tensor is None:
                base_tensor = self._base[spec.key]
                self._source_cache[spec.key] = base_tensor
            v = self._slice_in_proj(base_tensor, spec.index)
        elif isinstance(spec, _Transpose):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            v = self._transpose_2d(self._base[spec.key])
        elif isinstance(spec, _DefaultLogitScale):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            import torch

            v = torch.tensor(float(spec.value))  # type: ignore[assignment]
        else:  # pragma: no cover - defensive
            raise KeyError(k)

        self._derived_cache[k] = v
        return v

    def __setitem__(self, k: str, v: _T) -> None:
        self._derived_cache.pop(k, None)
        self._map[k] = _Direct(k)
        self._base[k] = v
        if k not in self._keys:
            self._keys = (*self._keys, k)

    def __delitem__(self, k: str) -> None:
        self._derived_cache.pop(k, None)
        spec = self._map.pop(k, None)
        if isinstance(spec, _Direct) and spec.key in self._base:
            del self._base[spec.key]
        if k in self._keys:
            self._keys = tuple(x for x in self._keys if x != k)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self[k]


def _normalize(key: str) -> str:
    return strip_repeated_prefixes(str(key), _WRAPPER_PREFIXES)


def _remap_clip_state_dict(
    state_dict: MutableMapping[str, _T],
    *,
    num_layers: int,
    keep_projection: bool,
    transpose_projection: bool,
) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    if len(state_dict) == 1 and "state_dict" in state_dict:
        inner = state_dict.get("state_dict")
        if isinstance(inner, MutableMapping):
            state_dict = inner

    raw_keys = list(state_dict.keys())
    normalized = [(raw, _normalize(raw)) for raw in raw_keys]
    keys_for_style = [k for _, k in normalized if not k.endswith(".position_ids")]
    style = _DETECTOR.detect(keys_for_style)

    mapping: dict[str, _Spec] = {}
    seen_logit: str | None = None
    seen_proj: str | None = None

    def _put(dst: str, spec: _Spec) -> None:
        prev = mapping.get(dst)
        if prev is not None:
            raise KeyMappingError(f"sdxl_clip: duplicate destination key {dst!r} (collision)")
        mapping[dst] = spec

    for raw_key, key in normalized:
        if key.endswith(".position_ids"):
            continue

        if key in _LOGIT_KEYS:
            if seen_logit is not None:
                raise KeyMappingError(f"sdxl_clip: multiple logit_scale sources: {seen_logit!r},{key!r}")
            seen_logit = key
            _put("logit_scale", _Direct(raw_key))
            continue

        if key in _PROJ_KEYS:
            if not keep_projection:
                continue
            if seen_proj is not None:
                raise KeyMappingError(f"sdxl_clip: multiple text_projection sources: {seen_proj!r},{key!r}")
            seen_proj = key
            if transpose_projection:
                _put("transformer.text_projection.weight", _Transpose(raw_key))
            else:
                _put("transformer.text_projection.weight", _Direct(raw_key))
            continue

        if style is KeyStyle.CODEX:
            if key.startswith("transformer.text_model."):
                _put(key, _Direct(raw_key))
                continue
            raise KeyMappingError(f"sdxl_clip: unsupported CODEX key {key!r}")

        if style is KeyStyle.HF:
            if key.startswith("text_model."):
                _put(f"transformer.{key}", _Direct(raw_key))
                continue
            raise KeyMappingError(f"sdxl_clip: unsupported HF key {key!r}")

        if style is KeyStyle.OPENCLIP:
            if key == "positional_embedding":
                _put("transformer.text_model.embeddings.position_embedding.weight", _Direct(raw_key))
                continue
            if key == "token_embedding.weight":
                _put("transformer.text_model.embeddings.token_embedding.weight", _Direct(raw_key))
                continue
            if key in {"ln_final.weight", "ln_final.bias"}:
                suffix = "weight" if key.endswith(".weight") else "bias"
                _put(f"transformer.text_model.final_layer_norm.{suffix}", _Direct(raw_key))
                continue
            if not key.startswith("transformer.resblocks."):
                raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP key {key!r}")

            parts = key.split(".")
            if len(parts) < 5 or parts[0] != "transformer" or parts[1] != "resblocks" or not parts[2].isdigit():
                raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP resblock key {key!r}")
            layer = int(parts[2])
            if layer < 0 or layer >= int(num_layers):
                raise KeyMappingError(
                    f"sdxl_clip: OpenCLIP resblock index out of range (layer={layer}, num_layers={num_layers}) for key={key!r}"
                )

            tail = parts[3:]
            base = f"transformer.text_model.encoder.layers.{layer}."

            # OpenCLIP layout:
            # - ln_1/ln_2: transformer.resblocks.{i}.ln_1.{weight,bias}
            # - mlp: transformer.resblocks.{i}.mlp.c_fc.{weight,bias} / c_proj.{weight,bias}
            # - attn: transformer.resblocks.{i}.attn.{in_proj_weight,in_proj_bias,out_proj.{weight,bias}}
            if len(tail) == 2 and tail[0] in {"ln_1", "ln_2"} and tail[1] in {"weight", "bias"}:
                mapped = "layer_norm1" if tail[0] == "ln_1" else "layer_norm2"
                _put(base + f"{mapped}.{tail[1]}", _Direct(raw_key))
                continue

            if len(tail) == 3 and tail[0] == "mlp" and tail[1] in {"c_fc", "c_proj"} and tail[2] in {"weight", "bias"}:
                mapped = "mlp.fc1" if tail[1] == "c_fc" else "mlp.fc2"
                _put(base + f"{mapped}.{tail[2]}", _Direct(raw_key))
                continue

            if len(tail) == 2 and tail[0] == "attn" and tail[1] in {"in_proj_weight", "in_proj_bias"}:
                suffix = "weight" if tail[1].endswith("_weight") else "bias"
                _put(base + f"self_attn.q_proj.{suffix}", _SliceQKV(raw_key, 0))
                _put(base + f"self_attn.k_proj.{suffix}", _SliceQKV(raw_key, 1))
                _put(base + f"self_attn.v_proj.{suffix}", _SliceQKV(raw_key, 2))
                continue

            if len(tail) == 3 and tail[0] == "attn" and tail[1] == "out_proj" and tail[2] in {"weight", "bias"}:
                _put(base + f"self_attn.out_proj.{tail[2]}", _Direct(raw_key))
                continue

            raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP resblock key {key!r}")

        raise KeyMappingError(f"sdxl_clip: unsupported detected style={style.value!r}")

    if "logit_scale" not in mapping:
        _put("logit_scale", _DefaultLogitScale(log(100.0)))

    if keep_projection and "transformer.text_projection.weight" not in mapping:
        raise KeyMappingError(
            "sdxl_clip: projection weights are required for this encoder but were not found "
            "(expected one of: %s)" % (", ".join(_PROJ_KEYS),)
        )

    missing_essentials = [key for key in _ESSENTIAL_KEYS if key not in mapping]
    if missing_essentials:
        sample = ", ".join(missing_essentials[:3])
        raise KeyMappingError(
            "sdxl_clip: key mapping failed (missing essential tensors). "
            f"missing_sample=[{sample}] style={style.value}"
        )

    # Ensure output is canonical (no source-keyspace remnants).
    forbidden = []
    for out_key in mapping.keys():
        if out_key.startswith("text_model.") or out_key.startswith("transformer.resblocks."):
            forbidden.append(out_key)
    if forbidden:
        raise KeyMappingError(f"sdxl_clip: produced non-canonical keys (sample={sorted(forbidden)[:10]})")

    return style, _SDXLCLIPKeymapView(state_dict, mapping)


def remap_sdxl_clip_l_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    """Keymap SDXL base CLIP-L weights (text_encoder) into Codex IntegratedCLIP keys."""

    return _remap_clip_state_dict(
        state_dict,
        num_layers=12,
        keep_projection=False,
        transpose_projection=False,
    )


def remap_sdxl_clip_g_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    """Keymap SDXL base CLIP-G weights (text_encoder_2) into Codex IntegratedCLIP keys."""

    return _remap_clip_state_dict(
        state_dict,
        num_layers=32,
        keep_projection=True,
        transpose_projection=True,
    )


__all__ = [
    "remap_sdxl_clip_g_state_dict",
    "remap_sdxl_clip_l_state_dict",
]
