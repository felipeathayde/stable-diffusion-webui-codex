"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed specs for supported CLIP vision encoder variants.
Defines dataclass-driven configs used to instantiate HF `CLIPVisionConfig` and to validate checkpoint compatibility.

Symbols (top-level; keep in sync; no ghosts):
- `ClipVisionPreprocessSpec` (dataclass): Image preprocessing spec (image size + mean/std normalization).
- `ClipVisionVariantSpec` (dataclass): Full model spec for a supported CLIP vision variant.
- `ClipVisionVariant` (enum): Supported CLIP vision variant identifiers.
- `_VARIANT_SPECS` (constant): Canonical mapping of variant → spec used by the registry.
- `get_variant_spec` (function): Returns the canonical spec for a given variant (raises on unknown variants).
- `list_supported_variants` (function): Returns all supported variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from .errors import ClipVisionConfigError


@dataclass(frozen=True)
class ClipVisionPreprocessSpec:
    image_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclass(frozen=True)
class ClipVisionVariantSpec:
    variant: "ClipVisionVariant"
    model_type: str
    hidden_act: str
    hidden_size: int
    image_size: int
    initializer_factor: float
    initializer_range: float
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    torch_dtype: str
    attention_dropout: float = 0.0
    dropout: float = 0.0
    preprocess: ClipVisionPreprocessSpec = ClipVisionPreprocessSpec(
        image_size=224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    def to_huggingface_kwargs(self) -> Dict[str, object]:
        """Return kwargs compatible with ``transformers.CLIPVisionConfig``."""
        return {
            "attention_dropout": self.attention_dropout,
            "dropout": self.dropout,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "image_size": self.image_size,
            "initializer_factor": self.initializer_factor,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "layer_norm_eps": self.layer_norm_eps,
            "model_type": self.model_type,
            "num_attention_heads": self.num_attention_heads,
            "num_channels": self.num_channels,
            "num_hidden_layers": self.num_hidden_layers,
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
            "torch_dtype": self.torch_dtype,
        }


class ClipVisionVariant(str, Enum):
    """Enumerates supported clip vision encoder variants."""

    G = "clip-vision-g"
    H = "clip-vision-h"
    VIT_L = "clip-vision-vit-l"


_VARIANT_SPECS: Dict[ClipVisionVariant, ClipVisionVariantSpec] = {
    ClipVisionVariant.G: ClipVisionVariantSpec(
        variant=ClipVisionVariant.G,
        model_type="clip_vision_model",
        hidden_act="gelu",
        hidden_size=1664,
        image_size=224,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=8192,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=48,
        patch_size=14,
        projection_dim=1280,
        torch_dtype="float32",
    ),
    ClipVisionVariant.H: ClipVisionVariantSpec(
        variant=ClipVisionVariant.H,
        model_type="clip_vision_model",
        hidden_act="gelu",
        hidden_size=1280,
        image_size=224,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=5120,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=32,
        patch_size=14,
        projection_dim=1024,
        torch_dtype="float32",
    ),
    ClipVisionVariant.VIT_L: ClipVisionVariantSpec(
        variant=ClipVisionVariant.VIT_L,
        model_type="clip_vision_model",
        hidden_act="quick_gelu",
        hidden_size=1024,
        image_size=224,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=24,
        patch_size=14,
        projection_dim=768,
        torch_dtype="float32",
    ),
}


def get_variant_spec(variant: ClipVisionVariant) -> ClipVisionVariantSpec:
    try:
        return _VARIANT_SPECS[variant]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ClipVisionConfigError(f"Unsupported clip vision variant: {variant}") from exc


def list_supported_variants() -> Tuple[ClipVisionVariant, ...]:
    return tuple(_VARIANT_SPECS.keys())
