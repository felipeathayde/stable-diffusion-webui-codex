"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: State-dict conversion and filtering helpers for CLIP vision encoders.
Normalizes OpenCLIP-style layouts into HF naming, rekeys diffusers/HF outputs, and provides diagnostics for loading.
Structural conversion operations (projection transpose and fused in_proj -> split Q/K/V converter paths) are globally policy-gated by
`CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto`=forbid, `convert`=allow).

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for state-dict conversion and filtering diagnostics.
- `_projection_weight_for_orientation` (function): Returns projection weights for a target orientation (`linear` or `matmul`).
- `convert_openclip_checkpoint` (function): In-place OpenCLIP→HF key conversion for vision checkpoints.
- `rekey_vision_state_dict` (function): Re-prefix helper for state dicts produced by diffusers/HF.
- `cleaned_state_dict` (function): Filters a state dict to keep only keys under the allowed prefixes.
- `summarize_state_dict` (function): Returns (#tensors, total parameters) for diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Dict, Iterable, Literal, Tuple

import torch

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
from apps.backend.runtime.models.state_dict import (
    state_dict_prefix_replace,
    transformers_convert,
)

from .errors import ClipVisionLoadError
from .specs import ClipVisionVariantSpec

logger = logging.getLogger("backend.runtime.vision.clip.state_dict")


def _projection_weight_for_orientation(
    weight: torch.Tensor,
    *,
    source_orientation: Literal["linear", "matmul"],
    target_orientation: Literal["linear", "matmul"],
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ClipVisionLoadError(
            f"Clip vision projection weight must be 2-D; received shape {tuple(weight.shape)}."
        )
    if source_orientation == target_orientation:
        return weight
    if not is_structural_weight_conversion_enabled():
        raise ClipVisionLoadError(
            "Clip vision projection conversion requires structural conversion (transpose), "
            f"but {ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto forbids it. "
            f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
        )
    return weight.transpose(0, 1).contiguous()


def convert_openclip_checkpoint(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    prefix: str,
    spec: ClipVisionVariantSpec | None = None,
    projection_orientation: Literal["auto", "linear", "matmul"] = "auto",
) -> None:
    """Mutate an OpenCLIP-style state dict so it matches HF naming."""
    if spec is None:
        from .registry import detect_variant_from_state_dict, get_variant_spec

        variant = detect_variant_from_state_dict(state_dict)
        spec = get_variant_spec(variant)
    prefix = prefix.rstrip(".")
    remap_keys: Dict[str, str] = {
        f"{prefix}.class_embedding": "vision_model.embeddings.class_embedding",
        f"{prefix}.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
        f"{prefix}.positional_embedding": "vision_model.embeddings.position_embedding.weight",
        f"{prefix}.ln_post.bias": "vision_model.post_layernorm.bias",
        f"{prefix}.ln_post.weight": "vision_model.post_layernorm.weight",
        f"{prefix}.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        f"{prefix}.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    }
    for source, target in remap_keys.items():
        if source in state_dict:
            state_dict[target] = state_dict.pop(source)
    proj_key = f"{prefix}.proj"
    if proj_key in state_dict:
        source_orientation: Literal["linear", "matmul"] = "matmul"
        if projection_orientation == "auto":
            target_orientation: Literal["linear", "matmul"] = source_orientation
        else:
            target_orientation = projection_orientation
        state_dict["visual_projection.weight"] = _projection_weight_for_orientation(
            state_dict.pop(proj_key),
            source_orientation=source_orientation,
            target_orientation=target_orientation,
        )
    transformers_convert(state_dict, f"{prefix}.", "vision_model.", spec.num_hidden_layers)


def rekey_vision_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    prefix: str,
) -> None:
    """Re-prefix a vision state dict produced by diffusers/HF."""
    state_dict_prefix_replace(state_dict, {prefix: ""})


def cleaned_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    keep_prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Return a copy containing only keys with allowed prefixes."""
    keep_prefixes = tuple(keep_prefixes)
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in keep_prefixes):
            filtered[key] = value
        else:
            logger.debug("Dropping unused clip vision key: %s", key)
    return filtered


def summarize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, int]:
    """Return (#tensors, total parameters) for diagnostics."""
    total_params = 0
    tensor_count = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            tensor_count += 1
            total_params += tensor.numel()
    return tensor_count, total_params
