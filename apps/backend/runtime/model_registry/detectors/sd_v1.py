"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Diffusion 1.x (SD1.5-style) model detector for the model registry.
Detects classic UNet checkpoints by key patterns, channel shapes, and CLIP-L embedding size, and returns a `ModelSignature` including quantization hints.

Symbols (top-level; keep in sync; no ghosts):
- `SD15_REQUIRED_KEYS` (constant): Keys required to identify SD1.x/SD1.5-style checkpoints.
- `StableDiffusionV1Detector` (class): Detector for SD1.5-style UNet checkpoints (channels 4/9; CLIP-L text encoder).
- `_infer_quantization` (function): Infers `QuantizationHint` from state-dict key patterns (unsupported NF4/FP4 marker keys).
"""

from __future__ import annotations

from typing import List


from apps.backend.runtime.model_registry.detectors.base import REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks, has_all_keys
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    QuantizationKind,
    TextEncoderSignature,
    VAESignature,
)


SD15_REQUIRED_KEYS: List[str] = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "first_stage_model.decoder.conv_out.weight",
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
]


class StableDiffusionV1Detector:
    priority = 100

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *SD15_REQUIRED_KEYS):
            return False
        shape = bundle.shape("model.diffusion_model.input_blocks.0.0.weight")
        if not shape or shape[1] not in (4, 9):
            return False
        # Text encoder hidden size 768
        txt_shape = bundle.shape("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")
        if not txt_shape or txt_shape[-1] != 768:
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        in_w = bundle.state_dict["model.diffusion_model.input_blocks.0.0.weight"]
        out_w = bundle.state_dict["model.diffusion_model.out.2.weight"]
        channels_in = int(getattr(in_w, "shape", (0,))[1])
        channels_out = int(getattr(out_w, "shape", (0,))[0])
        context_dim = 768
        temporal = False
        depth = count_blocks(bundle.keys, "model.diffusion_model.output_blocks.{}.")
        quant_hint = _infer_quantization(bundle)
        return ModelSignature(
            family=ModelFamily.SD15,
            repo_hint="runwayml/stable-diffusion-v1-5",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_V1,
            quantization=quant_hint,
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.UNET,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=temporal,
                depth=depth,
                key_prefixes=["model.diffusion_model."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_l",
                    key_prefix="cond_stage_model.transformer.",
                    expected_dim=768,
                    tokenizer_hint="openai/clip-vit-large-patch14",
                )
            ],
            vae=VAESignature(
                key_prefix="first_stage_model.",
                latent_channels=channels_in if channels_in in (4, 9) else 4,
            ),
        )


def _infer_quantization(bundle: SignalBundle) -> QuantizationHint:
    for key in bundle.keys:
        if "bitsandbytes__nf4" in key:
            return QuantizationHint(kind=QuantizationKind.NF4, detail="key_marker")
        if "bitsandbytes__fp4" in key:
            return QuantizationHint(kind=QuantizationKind.FP4, detail="key_marker")
    return QuantizationHint(kind=QuantizationKind.NONE)


REGISTRY.register(StableDiffusionV1Detector())
