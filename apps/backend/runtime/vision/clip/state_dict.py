from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Dict, Iterable, Tuple

import torch

from apps.backend.runtime.models.state_dict import (
    state_dict_prefix_replace,
    transformers_convert,
)

from .errors import ClipVisionLoadError
from .specs import ClipVisionVariantSpec

logger = logging.getLogger("backend.runtime.vision.clip.state_dict")


def _transpose_projection(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ClipVisionLoadError(
            f"Clip vision projection weight must be 2-D; received shape {tuple(weight.shape)}."
        )
    return weight.transpose(0, 1).contiguous()


def convert_openclip_checkpoint(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    prefix: str,
    spec: ClipVisionVariantSpec | None = None,
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
        state_dict["visual_projection.weight"] = _transpose_projection(state_dict.pop(proj_key))
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
