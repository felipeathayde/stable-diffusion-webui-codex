"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 model detector for the Codex model registry.
Matches WAN22 checkpoints by key suffixes and tensor shapes, infers patch/latent dimensions, detects embedded VAE/text-encoder components,
and returns a `ModelSignature` describing the WAN core transformer and assets.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_HEAD_KEY` (constant): Suffix used to locate the WAN modulation head in a state dict.
- `Wan22Detector` (class): Detector that matches WAN22 bundles and builds a `ModelSignature` (core dims + TE/VAE signatures).
- `_collect_text_encoders` (function): Collects embedded text encoder signatures (UMT5-XXL, CLIP-L) when present.
- `_tensor_last_dim` (function): Returns the last dimension of a tensor/shape (used for TE expected dims).
- `_find_key` (function): Finds the shortest matching key by suffix (optional prefix filtering).
- `_detect_model_type` (function): Heuristically classifies WAN variant (t2v/i2v/ti2v/vace/s2v/animate).
"""

from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    TextEncoderSignature,
    VAESignature,
)
from apps.backend.runtime.wan22.inference import infer_wan22_latent_channels, infer_wan22_patch_embedding


WAN_HEAD_KEY = "head.modulation"


class Wan22Detector(ModelDetector):
    priority = 170

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        key = _find_key(bundle, WAN_HEAD_KEY)
        if not key:
            return False
        prefix = key[: -len(WAN_HEAD_KEY)]
        patch = bundle.shape(f"{prefix}patch_embedding.weight")
        return bool(patch and len(patch) == 5)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        key = _find_key(bundle, WAN_HEAD_KEY)
        assert key is not None
        prefix = key[: -len(WAN_HEAD_KEY)]

        patch_shape = bundle.shape(f"{prefix}patch_embedding.weight")
        assert patch_shape is not None and len(patch_shape) == 5
        in_channels, model_dim, patch_size = infer_wan22_patch_embedding(patch_shape)

        head_shape = bundle.shape(f"{prefix}head.head.weight")
        assert head_shape is not None
        latent_channels = infer_wan22_latent_channels(
            head_shape,
            patch_size=patch_size,
            default_latent_channels=in_channels,
        )

        num_layers = count_blocks(bundle.keys, f"{prefix}blocks.{{}}.")

        model_type = _detect_model_type(bundle, prefix, in_channels)

        vae_sig: Optional[VAESignature] = None
        vae_key = _find_key(bundle, "decoder.conv_out.weight", search_prefix="vae.")
        if vae_key:
            vae_shape = bundle.shape(vae_key)
            if vae_shape and len(vae_shape) >= 2:
                vae_sig = VAESignature(key_prefix="vae.", latent_channels=int(vae_shape[1]))

        extras = {
            "model_type": model_type,
            "patch_size": patch_size,
            "blocks": num_layers,
            "channels_in": in_channels,
        }

        text_encoders = _collect_text_encoders(bundle, prefix)

        return ModelSignature(
            family=ModelFamily.WAN22,
            repo_hint="Wan-AI/Wan2.2",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.WAN22,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=in_channels,
                channels_out=latent_channels,
                context_dim=model_dim,
                temporal=True,
                depth=num_layers,
                key_prefixes=[prefix],
            ),
            text_encoders=text_encoders,
            vae=vae_sig,
            extras=extras,
        )


def _collect_text_encoders(bundle: SignalBundle, prefix: str) -> list[TextEncoderSignature]:
    encoders: list[TextEncoderSignature] = []
    # UMT5 XXL
    if f"{prefix}text_encoders.umt5xxl.transformer.encoder.final_layer_norm.weight" in bundle.state_dict:
        encoders.append(
            TextEncoderSignature(
                name="umt5xxl",
                key_prefix=f"{prefix}text_encoders.umt5xxl.",
                expected_dim=_tensor_last_dim(bundle, f"{prefix}text_encoders.umt5xxl.transformer.encoder.final_layer_norm.weight"),
                tokenizer_hint="Wan-AI/umt5xxl",
            )
        )
    if f"{prefix}text_encoders.clip_l.transformer.final_layer_norm.weight" in bundle.state_dict:
        encoders.append(
            TextEncoderSignature(
                name="clip_l",
                key_prefix=f"{prefix}text_encoders.clip_l.",
                expected_dim=_tensor_last_dim(bundle, f"{prefix}text_encoders.clip_l.transformer.final_layer_norm.weight"),
                tokenizer_hint="Wan-AI/clip-fp16",
            )
        )
    return encoders


def _tensor_last_dim(bundle: SignalBundle, key: str) -> Optional[int]:
    shape = bundle.shape(key)
    if shape:
        return int(shape[-1])
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return int(tensor.shape[-1])
    return None


def _find_key(bundle: SignalBundle, suffix: str, *, search_prefix: str | None = None) -> Optional[str]:
    candidates = []
    for key in bundle.state_dict:
        if search_prefix and not key.startswith(search_prefix):
            continue
        if key.endswith(suffix):
            candidates.append(key)
    return min(candidates, key=len) if candidates else None


def _detect_model_type(bundle: SignalBundle, prefix: str, in_channels: int) -> str:
    checks = {
        "vace": f"{prefix}vace_patch_embedding.weight",
        "s2v": f"{prefix}casual_audio_encoder.encoder.final_linear.weight",
        "animate": f"{prefix}face_adapter.fuser_blocks.0.k_norm.weight",
    }
    for model_type, key in checks.items():
        if key in bundle.state_dict:
            return model_type
    if in_channels >= 48:
        return "ti2v"
    if f"{prefix}img_emb.proj.0.bias" in bundle.state_dict:
        return "i2v"
    return "t2v"


REGISTRY.register(Wan22Detector())
