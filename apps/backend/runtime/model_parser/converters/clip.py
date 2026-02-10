"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP state-dict conversion helpers for model parsing (SD1.5/SD2.x/SDXL).
Normalizes CLIP checkpoints into the expected Codex/HF-like key layout using `transformers_convert`, fixes position-id dtype when requested,
and canonicalizes text-projection keys (handling alias/prefix variants).
Structural conversion operations (projection transpose and fused in_proj -> split Q/K/V converter paths) are globally policy-gated by
`CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto`=forbid, `convert`=allow).

Symbols (top-level; keep in sync; no ghosts):
- `_ensure_position_ids_long` (function): Ensures `position_ids` tensors are `torch.long` (rounding when needed).
- `_with_prefix` (function): Adds a prefix to every key in a state dict mapping.
- `_strip_prefix` (function): Removes a key prefix when present (leaves non-matching keys unchanged).
- `_normalize_text_projection` (function): Normalizes text-projection weights into `*.text_projection.weight` (optional transpose).
- `convert_clip` (function): Generic CLIP converter (alias-aware; OpenCLIP/diffusers key normalization via `transformers_convert`).
- `convert_sd15_clip` (function): SD1.5 CLIP-L converter (drops heads reconstructed at runtime).
- `convert_sd20_clip` (function): SD2.x CLIP-H converter (transposes projection weights).
- `convert_sdxl_clip_l` (function): SDXL CLIP-L converter (drops runtime-reconstructed projection weights).
- `convert_sdxl_clip_g` (function): SDXL CLIP-G converter (transposes projection; keeps logit_scale).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
from apps.backend.runtime.models.state_dict import transformers_convert


def _ensure_position_ids_long(sd: Dict[str, Any], key: str) -> None:
    value = sd.get(key)
    if isinstance(value, torch.Tensor) and value.dtype != torch.long:
        sd[key] = value.round().to(torch.long)


def _with_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out


def _normalize_text_projection(sd: Dict[str, Any], alias: str, *, transpose: bool = False) -> None:
    key_plain = f"{alias}.text_projection"
    if key_plain in sd:
        tensor = sd.pop(key_plain)
        if isinstance(tensor, torch.Tensor) and transpose:
            if not is_structural_weight_conversion_enabled():
                raise RuntimeError(
                    "CLIP converter requires structural conversion (projection transpose), "
                    f"but {ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto forbids it. "
                    f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
                )
            tensor = tensor.transpose(0, 1).contiguous()
        sd[f"{alias}.text_projection.weight"] = tensor

    key_plain_weight = f"{alias}.text_projection.weight"
    if key_plain_weight in sd:
        sd[f"{alias}.text_projection.weight"] = sd.pop(key_plain_weight)

    key_transform = f"{alias}.transformer.text_projection"
    if key_transform in sd:
        tensor = sd.pop(key_transform)
        if isinstance(tensor, torch.Tensor) and transpose:
            if not is_structural_weight_conversion_enabled():
                raise RuntimeError(
                    "CLIP converter requires structural conversion (projection transpose), "
                    f"but {ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto forbids it. "
                    f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
                )
            tensor = tensor.transpose(0, 1).contiguous()
        sd[f"{alias}.text_projection.weight"] = tensor
        sd[f"{alias}.transformer.text_projection.weight"] = tensor


def convert_clip(
    sd: Dict[str, Any],
    *,
    alias: str,
    layers: int,
    ensure_position_ids: bool = False,
    drop_logit_scale: bool = False,
    transpose_projection: bool = False,
) -> Dict[str, Any]:
    work = _with_prefix(dict(sd), f"{alias}.")
    # Accept OpenCLIP-style keys under "<alias>.transformer.resblocks.*" and normalize to
    # diffusers-style "<alias>.transformer.text_model.encoder.layers.*".
    transformers_convert(work, f"{alias}.", f"{alias}.transformer.text_model.", layers)
    if ensure_position_ids:
        _ensure_position_ids_long(work, f"{alias}.transformer.text_model.embeddings.position_ids")
    _normalize_text_projection(work, alias, transpose=transpose_projection)
    if drop_logit_scale:
        work.pop(f"{alias}.logit_scale", None)
    return _strip_prefix(work, f"{alias}.")


def convert_sd15_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=12,
        ensure_position_ids=True,
        drop_logit_scale=True,
        transpose_projection=False,
    )
    # Remove heads reconstructed at runtime.
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sd20_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_h",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
        transpose_projection=True,
    )


def convert_sdxl_clip_l(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sdxl_clip_g(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_g",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=False,
        transpose_projection=True,
    )
