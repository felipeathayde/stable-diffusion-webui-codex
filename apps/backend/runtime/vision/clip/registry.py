from __future__ import annotations

import logging
from typing import Mapping, Sequence

import torch

from .errors import ClipVisionConfigError, ClipVisionLoadError
from .specs import ClipVisionVariant, ClipVisionVariantSpec, get_variant_spec

logger = logging.getLogger("backend.runtime.vision.clip.registry")


def _detect_openclip_variant(state_dict: Mapping[str, torch.Tensor]) -> ClipVisionVariant:
    prefix = "visual.transformer.resblocks."
    keys = state_dict.keys()
    if f"{prefix}47.layer_norm1.weight" in keys:
        return ClipVisionVariant.G
    if f"{prefix}30.layer_norm1.weight" in keys:
        return ClipVisionVariant.H
    if f"{prefix}22.layer_norm1.weight" in keys:
        return ClipVisionVariant.VIT_L
    raise ClipVisionLoadError(
        "Unable to detect clip vision variant from OpenCLIP layout; unsupported or truncated checkpoint."
    )


def detect_variant_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> ClipVisionVariant:
    """Infer the clip vision variant by inspecting hallmark parameter names."""
    keys = state_dict.keys()
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in keys:
        return _detect_openclip_variant(state_dict)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in keys:
        return ClipVisionVariant.G
    if "vision_model.encoder.layers.30.layer_norm1.weight" in keys:
        return ClipVisionVariant.H
    if "vision_model.encoder.layers.22.layer_norm1.weight" in keys:
        embed_key = "vision_model.embeddings.position_embedding.weight"
        if embed_key not in keys:
            raise ClipVisionLoadError(
                "Unable to detect clip vision variant: missing position embedding for VIT-L family."
            )
        embed_shape = state_dict[embed_key].shape[0]
        if embed_shape == 577:
            return ClipVisionVariant.VIT_L
        if embed_shape in (729, 1024):
            raise ClipVisionConfigError(
                "SigLIP-style clip vision checkpoints are not yet supported; "
                "please convert or downsample to a supported Codex variant."
            )
        raise ClipVisionLoadError(
            f"Unrecognised position embedding shape ({embed_shape}) for clip vision encoder."
        )
    raise ClipVisionLoadError(
        "Unable to detect clip vision variant: expected layer_norm markers for known variants."
    )


def get_spec_for_state_dict(state_dict: Mapping[str, torch.Tensor]) -> ClipVisionVariantSpec:
    variant = detect_variant_from_state_dict(state_dict)
    return get_variant_spec(variant)


def validate_state_dict(state_dict: Mapping[str, torch.Tensor], spec: ClipVisionVariantSpec) -> None:
    """Perform cheap validations to surface mismatches early."""
    expected_prefix = "vision_model.encoder.layers."
    candidate_layers: Sequence[int] = []
    for key in state_dict.keys():
        if key.startswith(expected_prefix) and key.endswith(".layer_norm1.weight"):
            try:
                layer_index = int(key[len(expected_prefix) :].split(".", 1)[0])
            except ValueError:
                continue
            candidate_layers.append(layer_index)
    if not candidate_layers:
        raise ClipVisionLoadError("State dict does not contain encoder layer norm weights.")
    max_layer_index = max(candidate_layers)
    if max_layer_index + 1 != spec.num_hidden_layers:
        raise ClipVisionLoadError(
            f"State dict encoder layer count mismatch: expected {spec.num_hidden_layers}, "
            f"detected {max_layer_index + 1}."
        )
    projection_key = "vision_model.post_layernorm.weight"
    if projection_key not in state_dict:
        raise ClipVisionLoadError("State dict missing post_layernorm weights required for projection head.")
    logger.debug(
        "Validated clip vision state dict against variant %s (%d layers).",
        spec.variant.value,
        spec.num_hidden_layers,
    )
