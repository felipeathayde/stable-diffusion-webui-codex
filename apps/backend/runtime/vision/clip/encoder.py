"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime wrapper for Codex-native CLIP vision encoders.
Constructs and loads HF `CLIPVisionModelWithProjection`, applies memory-management policies, and returns structured outputs.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for clip vision encoder lifecycle and timing logs.
- `ClipVisionEncoder` (class): Encapsulates model construction/loading and provides `encode(...)` returning `ClipVisionOutput`.
"""

from __future__ import annotations

import logging
import time
from typing import Mapping, Optional

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, modeling_utils

from apps.backend.patchers.base import ModelPatcher
from apps.backend.runtime import ops as runtime_ops
from apps.backend.runtime.memory import memory_management

from .errors import ClipVisionInputError, ClipVisionLoadError
from .preprocess import preprocess_image
from .registry import get_spec_for_state_dict, validate_state_dict
from .specs import ClipVisionVariantSpec
from .types import ClipVisionOutput

logger = logging.getLogger("backend.runtime.vision.clip.encoder")


class ClipVisionEncoder:
    """Codex-native runtime wrapper for CLIP vision encoders."""

    def __init__(self, spec: ClipVisionVariantSpec):
        self.spec = spec
        self.load_device = memory_management.text_encoder_device()
        self.offload_device = memory_management.text_encoder_offload_device()
        self.runtime_dtype = memory_management.text_encoder_dtype(self.load_device)
        logger.debug(
            "Initialising clip vision encoder variant=%s load_device=%s offload_device=%s dtype=%s",
            spec.variant.value,
            self.load_device,
            self.offload_device,
            self.runtime_dtype,
        )
        config = CLIPVisionConfig(**spec.to_huggingface_kwargs())
        with runtime_ops.using_codex_operations():
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.runtime_dtype)
        self.model.eval()
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "ClipVisionEncoder":
        spec = get_spec_for_state_dict(state_dict)
        encoder = cls(spec)
        encoder.load_state_dict(state_dict)
        return encoder

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        validate_state_dict(state_dict, self.spec)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ClipVisionLoadError(
                f"Clip vision state dict mismatch (missing={len(missing)}, unexpected={len(unexpected)})."
            )
        logger.info(
            "Loaded clip vision encoder variant=%s with %d parameters.",
            self.spec.variant.value,
            sum(p.numel() for p in self.model.parameters()),
        )

    def encode(
        self,
        image: torch.Tensor,
        *,
        crop: bool = True,
        return_all_hidden_states: bool = False,
    ) -> ClipVisionOutput:
        if not isinstance(image, torch.Tensor):
            raise ClipVisionInputError("ClipVisionEncoder.encode expects a torch.Tensor input.")
        start = time.perf_counter()
        memory_management.load_model_gpu(self.patcher)
        processed = preprocess_image(image.to(self.load_device), self.spec.preprocess, crop=crop)
        outputs = self.model(
            pixel_values=processed,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states or ()
        if len(hidden_states) < 1:
            raise ClipVisionLoadError("Vision model did not return hidden states.")
        penultimate_index = -2 if len(hidden_states) >= 2 else -1
        intermediate_device = memory_management.intermediate_device()
        last_hidden = outputs.last_hidden_state.to(intermediate_device)
        penultimate = hidden_states[penultimate_index].to(intermediate_device)
        embeds = outputs.image_embeds.to(intermediate_device)
        all_hidden: Optional[torch.Tensor] = None
        if return_all_hidden_states:
            try:
                all_hidden = torch.stack(
                    [state.to(intermediate_device) for state in hidden_states],
                    dim=1,
                )
            except RuntimeError as exc:  # pragma: no cover - defensive guard
                raise ClipVisionInputError("Failed to stack hidden states for return.") from exc
        runtime = time.perf_counter() - start
        logger.debug(
            "Encoded clip vision batch=%d seq_len=%d hidden_size=%d embeddings=%s runtime=%.3fs",
            processed.shape[0],
            last_hidden.shape[1],
            last_hidden.shape[2],
            tuple(embeds.shape),
            runtime,
        )
        return ClipVisionOutput(
            last_hidden_state=last_hidden,
            penultimate_hidden_states=penultimate,
            image_embeds=embeds,
            all_hidden_states=all_hidden,
            mm_projected=None,
        )
