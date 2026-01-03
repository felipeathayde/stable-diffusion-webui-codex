"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public facade for Codex-native CLIP vision runtime helpers.
Re-exports specs, checkpoint/state-dict tooling, preprocessing helpers, and the `ClipVisionEncoder` wrapper.

Symbols (top-level; keep in sync; no ghosts):
- `ClipVisionEncoder` (class): Runtime wrapper for CLIP vision encoders (load + encode with device/dtype management).
- `ClipVisionError` (class): Base exception for CLIP vision runtime failures.
- `ClipVisionConfigError` (class): Raised when a vision spec/config is invalid or unsupported.
- `ClipVisionInputError` (class): Raised when caller-provided tensors/inputs are malformed.
- `ClipVisionLoadError` (class): Raised when loading a vision checkpoint/state dict fails.
- `ClipVisionOutput` (class): Structured encoder output bundle (hidden states + embeddings).
- `ClipVisionPreprocessSpec` (dataclass): Preprocessing parameters (image size + normalization stats).
- `ClipVisionVariant` (enum): Supported CLIP vision variant identifiers.
- `ClipVisionVariantSpec` (dataclass): Full model/spec config used to construct HF `CLIPVisionConfig`.
- `cleaned_state_dict` (function): Filters a state dict to keep only the expected key prefixes.
- `convert_openclip_checkpoint` (function): In-place key remap for OpenCLIP-style checkpoints into HF naming.
- `detect_variant_from_state_dict` (function): Detects a supported vision variant by inspecting hallmark keys.
- `get_spec_for_state_dict` (function): Resolves the matching `ClipVisionVariantSpec` for a state dict.
- `get_variant_spec` (function): Returns the canonical spec for a given variant enum.
- `list_supported_variants` (function): Lists supported vision variants.
- `preprocess_image` (function): Converts an input image tensor to the normalized format expected by the encoder.
- `rekey_vision_state_dict` (function): Re-prefix helper for state dicts produced by HF/diffusers.
- `summarize_state_dict` (function): Returns cheap diagnostics (#tensors, total parameters) for a state dict.
- `validate_state_dict` (function): Validates a state dict matches a given variant spec (layer count, required keys).
"""

from .encoder import ClipVisionEncoder
from .errors import (
    ClipVisionConfigError,
    ClipVisionError,
    ClipVisionInputError,
    ClipVisionLoadError,
)
from .preprocess import preprocess_image
from .registry import (
    detect_variant_from_state_dict,
    get_spec_for_state_dict,
    validate_state_dict,
)
from .specs import (
    ClipVisionPreprocessSpec,
    ClipVisionVariant,
    ClipVisionVariantSpec,
    get_variant_spec,
    list_supported_variants,
)
from .state_dict import (
    cleaned_state_dict,
    convert_openclip_checkpoint,
    rekey_vision_state_dict,
    summarize_state_dict,
)
from .types import ClipVisionOutput

__all__ = [
    "ClipVisionEncoder",
    "ClipVisionError",
    "ClipVisionConfigError",
    "ClipVisionInputError",
    "ClipVisionLoadError",
    "ClipVisionOutput",
    "ClipVisionPreprocessSpec",
    "ClipVisionVariant",
    "ClipVisionVariantSpec",
    "cleaned_state_dict",
    "convert_openclip_checkpoint",
    "detect_variant_from_state_dict",
    "get_spec_for_state_dict",
    "get_variant_spec",
    "list_supported_variants",
    "preprocess_image",
    "rekey_vision_state_dict",
    "summarize_state_dict",
    "validate_state_dict",
]
