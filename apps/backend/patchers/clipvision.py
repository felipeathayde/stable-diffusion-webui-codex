"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Clip vision patcher adapter backed by the Codex runtime vision encoder.
Provides legacy-facing load/preprocess/encode helpers, normalizing state dict keys and delegating model details to `apps.backend.runtime.vision.clip`.

Symbols (top-level; keep in sync; no ghosts):
- `CLIP_VISION_G` (constant): Hugging Face kwargs for the CLIP-ViT-bigG-14 vision variant.
- `CLIP_VISION_H` (constant): Hugging Face kwargs for the CLIP-ViT-H-14 vision variant.
- `CLIP_VISION_VITL` (constant): Hugging Face kwargs for the CLIP-ViT-L vision variant.
- `Output` (class): Dict-like wrapper around `ClipVisionOutput` with attribute-style access.
- `ClipVisionModel` (class): Legacy-facing wrapper around `ClipVisionEncoder` with load/encode helpers.
- `clip_preprocess` (function): Preprocesses an image tensor using a `ClipVisionPreprocessSpec` (size override supported).
- `_normalize_state_dict` (function): Normalizes and filters vision state dict keys (optional OpenCLIP conversion).
- `load_clipvision_from_sd` (function): Builds a `ClipVisionModel` from an in-memory state dict.
- `load` (function): Loads a CLIP vision checkpoint from disk and returns a `ClipVisionModel`.
"""

from __future__ import annotations

import logging
from typing import Mapping, MutableMapping

import torch

from apps.backend.runtime.checkpoint_io import load_torch_file
from apps.backend.runtime.vision.clip.encoder import ClipVisionEncoder
from apps.backend.runtime.vision.clip.errors import ClipVisionError, ClipVisionLoadError
from apps.backend.runtime.vision.clip.preprocess import preprocess_image
from apps.backend.runtime.vision.clip.registry import detect_variant_from_state_dict
from apps.backend.runtime.vision.clip.specs import ClipVisionPreprocessSpec, ClipVisionVariant, get_variant_spec
from apps.backend.runtime.vision.clip.state_dict import (
    cleaned_state_dict,
    convert_openclip_checkpoint,
    rekey_vision_state_dict,
    summarize_state_dict,
)
from apps.backend.runtime.vision.clip.types import ClipVisionOutput

logger = logging.getLogger("backend.patchers.clipvision")

_DEFAULT_PREPROCESS = ClipVisionPreprocessSpec(
    image_size=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)

CLIP_VISION_G = dict(get_variant_spec(ClipVisionVariant.G).to_huggingface_kwargs())
CLIP_VISION_H = dict(get_variant_spec(ClipVisionVariant.H).to_huggingface_kwargs())
CLIP_VISION_VITL = dict(get_variant_spec(ClipVisionVariant.VIT_L).to_huggingface_kwargs())


class Output(ClipVisionOutput):
    def __setitem__(self, key, item):
        setattr(self, key, item)


class ClipVisionModel:
    def __init__(self):
        self.encoder: ClipVisionEncoder | None = None
        self.spec = None
        self.patcher = None
        self.model: torch.nn.Module | None = None
        self.load_device: torch.device | None = None
        self.offload_device: torch.device | None = None
        self.dtype: torch.dtype | None = None

    def load_sd(self, state_dict: Mapping[str, torch.Tensor]):
        encoder = ClipVisionEncoder.from_state_dict(state_dict)
        self.encoder = encoder
        self.spec = encoder.spec
        self.patcher = encoder.patcher
        self.model = encoder.model
        self.load_device = encoder.load_device
        self.offload_device = encoder.offload_device
        self.dtype = encoder.runtime_dtype
        logger.info(
            "Loaded clip vision encoder variant=%s tensors=%d",
            encoder.spec.variant.value,
            sum(p.numel() for p in encoder.model.parameters()),
        )
        return [], []

    def get_sd(self):
        if self.encoder is None:
            raise ClipVisionLoadError("Clip vision encoder is not loaded.")
        return dict(self.encoder.model.state_dict())

    def encode_image(self, image: torch.Tensor) -> Output:
        if self.encoder is None:
            raise ClipVisionLoadError("Clip vision encoder is not loaded.")
        result = self.encoder.encode(image, crop=True)
        return Output(
            last_hidden_state=result.last_hidden_state,
            penultimate_hidden_states=result.penultimate_hidden_states,
            image_embeds=result.image_embeds,
            all_hidden_states=result.all_hidden_states,
            mm_projected=result.mm_projected,
        )


def clip_preprocess(image: torch.Tensor, size: int = 224) -> torch.Tensor:
    if size != _DEFAULT_PREPROCESS.image_size:
        spec = ClipVisionPreprocessSpec(
            image_size=size,
            mean=_DEFAULT_PREPROCESS.mean,
            std=_DEFAULT_PREPROCESS.std,
        )
    else:
        spec = _DEFAULT_PREPROCESS
    return preprocess_image(image, spec, crop=True)


def _normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    convert_keys: bool = False,
) -> MutableMapping[str, torch.Tensor]:
    working: MutableMapping[str, torch.Tensor] = dict(state_dict)
    if convert_keys:
        logger.debug("Converting OpenCLIP clip vision state dict with prefix '%s'.", prefix or "visual.")
        convert_openclip_checkpoint(working, prefix=prefix or "visual.")
    elif prefix:
        logger.debug("Re-keying clip vision state dict with prefix '%s'.", prefix)
        rekey_vision_state_dict(working, prefix=prefix)
    variant = detect_variant_from_state_dict(working)
    spec = get_variant_spec(variant)
    logger.debug("Detected clip vision variant %s.", spec.variant.value)
    filtered = cleaned_state_dict(
        working,
        keep_prefixes=(
            "vision_model.",
            "visual_projection.",
            "multi_modal_projector.",
            "mm_projector.",
        ),
    )
    if not filtered:
        raise ClipVisionLoadError("Clip vision state dict contains no recognised parameters after filtering.")
    tensors, params = summarize_state_dict(filtered)
    logger.debug("Filtered clip vision state dict: tensors=%d params=%d", tensors, params)
    return filtered


def load_clipvision_from_sd(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str = "",
    convert_keys: bool = False,
) -> ClipVisionModel:
    normalized = _normalize_state_dict(state_dict, prefix=prefix, convert_keys=convert_keys)
    model = ClipVisionModel()
    model.load_sd(normalized)
    return model


def load(ckpt_path: str) -> ClipVisionModel:
    try:
        state_dict = load_torch_file(ckpt_path)
    except Exception as exc:  # pragma: no cover - IO guard
        raise ClipVisionError(f"Failed to load clip vision checkpoint '{ckpt_path}': {exc}") from exc
    convert_required = "visual.transformer.resblocks.0.attn.in_proj_weight" in state_dict
    return load_clipvision_from_sd(
        state_dict,
        prefix="visual." if convert_required else "",
        convert_keys=convert_required,
    )
