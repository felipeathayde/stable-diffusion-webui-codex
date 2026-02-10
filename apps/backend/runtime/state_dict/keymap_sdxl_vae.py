"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL VAE key-style detection + remapping (LDM-style → diffusers AutoencoderKL).
Normalizes common LDM layouts into diffusers keyspace, strips wrapper prefixes, and flattens 1×1 Conv projection weights lazily.
Drops only known training metadata keys (`model_ema.decay` / `model_ema.num_updates`) and fails loud on other unknown keys.
Flattening conversion is globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto`=forbid, `convert`=allow).

Symbols (top-level; keep in sync; no ghosts):
- `remap_sdxl_vae_state_dict` (function): Returns (detected_style, remapped_view) for SDXL/Flow16 VAE keys.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
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

_PREFIXES = (
    "module.",
    "model.",
    "vae.",
    "first_stage_model.",
)

_WEIGHT_PREFIXES = (
    "encoder.",
    "decoder.",
    "quant_conv.",
    "post_quant_conv.",
)

# Some SDXL VAE weights files include training metadata tensors (e.g., EMA decay).
# These keys are not part of diffusers `AutoencoderKL.state_dict()`.
#
# Policy: allow (and drop) only the known metadata keys; anything else is an error.
_DROPPED_KEYS = (
    "model_ema.decay",
    "model_ema.num_updates",
)

_DETECTOR = KeyStyleDetector(
    name="sdxl_vae_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder.down_blocks."),
                KeySentinel(SentinelKind.PREFIX, "decoder.up_blocks."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid_block.attentions.0."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.LDM,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder.down."),
                KeySentinel(SentinelKind.PREFIX, "decoder.up."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid.attn_1."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid.block_"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


class _SDXLVAERemapView(MutableMapping[str, _T]):
    """Lazy remap view with SDXL VAE projection weight normalization.

    `diffusers.AutoencoderKL` expects attention projection weights in mid-block attentions as Linear weights
    `[C_out, C_in]`, but some LDM exports store them as 1×1 Conv2d weights `[C_out, C_in, 1, 1]`.
    This view flattens those tensors on access (and caches the flattened copy) to avoid eager materialization.
    """

    def __init__(
        self,
        base: MutableMapping[str, _T],
        mapping: dict[str, str],
        *,
        allow_structural_conversion: bool,
    ):
        self._base = base
        self._map = dict(mapping)
        self._cache: dict[str, _T] = {}
        self._allow_structural_conversion = bool(allow_structural_conversion)

    @staticmethod
    def _should_flatten(key: str) -> bool:
        if ".mid_block.attentions.0." not in key:
            return False
        return key.endswith(
            (
                ".to_q.weight",
                ".to_k.weight",
                ".to_v.weight",
                ".to_out.0.weight",
            )
        )

    @staticmethod
    def _flatten_conv_to_linear(value: _T) -> _T:
        try:
            ndim = getattr(value, "ndim", None)
            shape = getattr(value, "shape", None)
            if ndim != 4 or not shape:
                return value
            if tuple(shape[-2:]) != (1, 1):
                return value
            flattened = value[:, :, 0, 0]
            contiguous = getattr(flattened, "contiguous", None)
            if callable(contiguous):
                flattened = contiguous()
            return flattened
        except Exception:
            return value

    @staticmethod
    def _requires_flatten(value: _T) -> bool:
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)
        if ndim != 4 or not shape:
            return False
        return tuple(shape[-2:]) == (1, 1)

    def __getitem__(self, k: str) -> _T:
        cached = self._cache.get(k)
        if cached is not None:
            return cached

        v = self._base[self._map[k]]
        if self._should_flatten(k):
            if not self._allow_structural_conversion and self._requires_flatten(v):
                raise KeyMappingError(
                    "SDXL VAE key remap requires structural conversion (flatten 1x1 conv -> linear), "
                    f"but {ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto forbids it. "
                    f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
                )
            v = self._flatten_conv_to_linear(v)
            self._cache[k] = v
        return v

    def __setitem__(self, k: str, v: _T) -> None:
        self._cache.pop(k, None)
        self._map[k] = k
        self._base[k] = v

    def __delitem__(self, k: str) -> None:
        self._cache.pop(k, None)
        old = self._map.pop(k, None)
        if old is not None and old in self._base:
            del self._base[old]

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._map.keys())

    def items(self):
        for k in self._map.keys():
            yield k, self[k]


class _FilteredKeysView(MutableMapping[str, _T]):
    """Restrict a mutable mapping to an explicit key set (no eager tensor reads)."""

    def __init__(self, base: MutableMapping[str, _T], keys: Sequence[str]):
        self._base = base
        self._keys = tuple(keys)
        self._keys_set = set(self._keys)

    def __getitem__(self, k: str) -> _T:
        return self._base[k]

    def __setitem__(self, k: str, v: _T) -> None:
        self._base[k] = v
        if k not in self._keys_set:
            self._keys_set.add(k)
            self._keys = (*self._keys, k)

    def __delitem__(self, k: str) -> None:
        del self._base[k]
        if k in self._keys_set:
            self._keys_set.remove(k)
            self._keys = tuple(x for x in self._keys if x != k)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k: object) -> bool:
        return k in self._keys_set

    def keys(self):
        return list(self._keys)


def remap_sdxl_vae_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    """Normalize SDXL/Flow16 VAE keys into diffusers AutoencoderKL layout (fail loud).

    Accepted inputs:
    - Diffusers-style VAE keys (optionally wrapped): `encoder.down_blocks.*`, `decoder.up_blocks.*`, `*.mid_block.*`
    - LDM-style SDXL VAE keys (optionally wrapped): `encoder.down.*`, `decoder.up.*`, `*.mid.attn_1.*`

    Output:
    - Canonical diffusers AutoencoderKL keys (no wrapper prefixes).
    - Mid-attention projection weights are flattened on access when stored as 1×1 conv weights.
    """

    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _PREFIXES)

    raw_keys = list(state_dict.keys())
    kept_raw_keys: list[str] = []
    unknown_non_weight: list[str] = []

    for raw_key in raw_keys:
        normalized = _normalize(raw_key)
        if normalized.startswith(_WEIGHT_PREFIXES):
            kept_raw_keys.append(raw_key)
            continue
        if normalized in _DROPPED_KEYS:
            continue
        unknown_non_weight.append(normalized)

    if unknown_non_weight:
        sample = sorted(set(unknown_non_weight))[:10]
        raise KeyMappingError(
            "SDXL VAE key remap refuses unknown non-weight keys. "
            f"unknown_sample={sample}"
        )

    filtered = _FilteredKeysView(state_dict, kept_raw_keys)

    def _ldm_to_diffusers(key: str) -> str:
        new_key = key

        if key.startswith("encoder.down."):
            parts = key.split(".")
            # encoder.down.{i}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                i = int(parts[2])
                j = int(parts[4])
                rest = ".".join(parts[5:])
                new_key = f"encoder.down_blocks.{i}.resnets.{j}.{rest}"
            # encoder.down.{i}.downsample.rest...
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "downsample":
                i = int(parts[2])
                rest = ".".join(parts[4:])
                new_key = f"encoder.down_blocks.{i}.downsamplers.0.{rest}"

        elif key.startswith("decoder.up."):
            parts = key.split(".")
            # decoder.up.{k}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                k = int(parts[2])
                j = int(parts[4])
                # SDXL VAE indexes up_blocks in reverse order vs. LDM-style up.{k}
                i = 3 - k
                rest = ".".join(parts[5:])
                new_key = f"decoder.up_blocks.{i}.resnets.{j}.{rest}"
            # decoder.up.{k}.upsample.rest...
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "upsample":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[4:])
                new_key = f"decoder.up_blocks.{i}.upsamplers.0.{rest}"

        elif key.startswith("encoder.mid.block_") or key.startswith("decoder.mid.block_"):
            parts = key.split(".")
            if len(parts) >= 4 and parts[2].startswith("block_"):
                try:
                    block_index = int(parts[2].split("_", 1)[1]) - 1
                except (IndexError, ValueError):
                    block_index = 0
                rest = ".".join(parts[3:])
                prefix = "encoder" if parts[0] == "encoder" else "decoder"
                new_key = f"{prefix}.mid_block.resnets.{block_index}.{rest}"

        elif key.startswith("encoder.mid.attn_1.") or key.startswith("decoder.mid.attn_1."):
            is_encoder = key.startswith("encoder.")
            base = "encoder.mid.attn_1." if is_encoder else "decoder.mid.attn_1."
            suffix = key[len(base) :]
            prefix = "encoder" if is_encoder else "decoder"

            try:
                head, rest = suffix.split(".", 1)
            except ValueError:
                head, rest = suffix, ""

            table = {
                "q": "to_q",
                "k": "to_k",
                "v": "to_v",
                "proj_out": "to_out.0",
                "norm": "group_norm",
                # older / alternative naming
                "query": "to_q",
                "key": "to_k",
                "value": "to_v",
                "proj_attn": "to_out.0",
            }

            mapped = table.get(head)
            if mapped is not None and rest:
                new_key = f"{prefix}.mid_block.attentions.0.{mapped}.{rest}"

        if "nin_shortcut." in key:
            parts = key.split(".")
            # encoder.down.{i}.block.0.nin_shortcut.{weight,bias}
            if key.startswith("encoder.down.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                i = int(parts[2])
                rest = ".".join(parts[6:])
                new_key = f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.{rest}"
            # decoder.up.{k}.block.0.nin_shortcut.{weight,bias}
            elif key.startswith("decoder.up.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[6:])
                new_key = f"decoder.up_blocks.{i}.resnets.0.conv_shortcut.{rest}"

        if key.startswith("encoder.norm_out."):
            rest = key[len("encoder.norm_out.") :]
            new_key = f"encoder.conv_norm_out.{rest}"
        elif key.startswith("decoder.norm_out."):
            rest = key[len("decoder.norm_out.") :]
            new_key = f"decoder.conv_norm_out.{rest}"

        return new_key

    def _validate_output(keys: Sequence[str]) -> None:
        offenders: list[str] = []

        def _is_forbidden(k: str) -> bool:
            return (
                k.startswith("encoder.down.")
                or k.startswith("decoder.up.")
                or k.startswith("encoder.mid.attn_1.")
                or k.startswith("decoder.mid.attn_1.")
                or k.startswith("encoder.mid.block_")
                or k.startswith("decoder.mid.block_")
                or k.startswith("encoder.norm_out.")
                or k.startswith("decoder.norm_out.")
                or ".nin_shortcut." in k
            )

        for k in keys:
            if _is_forbidden(k):
                offenders.append(k)

        if offenders:
            sample = sorted(offenders)[:10]
            raise KeyMappingError(
                "SDXL VAE key remap produced non-canonical keys (mapping incomplete). "
                f"offenders_sample={sample}"
            )

        # Full VAE files always include down_blocks; require it when the tensor set is large enough.
        if len(keys) > 64 and not any(k.startswith("encoder.down_blocks.") for k in keys):
            preview = ", ".join(sorted(keys)[:10])
            raise KeyMappingError(
                "SDXL VAE key remap output is missing required encoder.down_blocks.* keys. "
                f"sample_keys=[{preview}]"
            )

    mappers = {
        KeyStyle.DIFFUSERS: lambda k: k,
        KeyStyle.LDM: _ldm_to_diffusers,
    }
    allow_structural_conversion = is_structural_weight_conversion_enabled()

    return remap_state_dict_view(
        filtered,
        detector=_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        view_factory=lambda base, mapping: _SDXLVAERemapView(
            base,
            mapping,
            allow_structural_conversion=allow_structural_conversion,
        ),
        output_validator=_validate_output,
    )


__all__ = ["remap_sdxl_vae_state_dict"]
