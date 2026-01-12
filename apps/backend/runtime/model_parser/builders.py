"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Build `CodexEstimatedConfig` from parser context + model signature.
Provides family presets for legacy UNet-like core config defaults, resolves a canonical Hugging Face repo id for external assets, and
normalizes quantization metadata by comparing detector hints with parser-detected tensor types.

Symbols (top-level; keep in sync; no ghosts):
- `_CORE_CONFIG_PRESETS` (constant): Default core config presets per `ModelFamily` for compatibility loaders (UNet-like models).
- `_FALLBACK_REPOS` (constant): Fallback diffusers repo ids used when `ModelSignature.repo_hint` is missing.
- `register_text_encoder` (function): Records a logical text encoder alias → component name mapping in `ParserContext.metadata`.
- `build_estimated_config` (function): Builds the `CodexEstimatedConfig` returned by `parse_state_dict` (components + repo id + extras + quantization).
"""

from __future__ import annotations

from typing import Dict

from apps.backend.runtime.model_registry.specs import ModelSignature, QuantizationKind, ModelFamily

from .errors import ValidationError
from .specs import CodexComponent, CodexEstimatedConfig, ParserContext
from .quantization import detect_quantization


_CORE_CONFIG_PRESETS: Dict[ModelFamily, Dict[str, object]] = {
    ModelFamily.SD15: {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": False,
        "context_dim": 768,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    },
    ModelFamily.SD20: {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 1024,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    },
    ModelFamily.SDXL: {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    },
    ModelFamily.SDXL_REFINER: {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2560,
        "in_channels": 4,
        "model_channels": 384,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 4,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    },
}

# Default diffusers repositories per family when the signature does not carry an
# explicit repo_hint. This keeps detectors focused on state-dict structure and
# lets configuration of base repos live in one place.
_FALLBACK_REPOS: Dict[ModelFamily, str] = {
    # Flux core-only GGUF checkpoints need a canonical diffusers repo to source
    # VAE and text encoders. Default to the public dev repo; operators can
    # override via repo_override or future configuration plumbing.
    ModelFamily.FLUX: "black-forest-labs/FLUX.1-dev",
}

_WAN22_REPO_BY_MODEL_TYPE: Dict[str, str] = {
    "t2v": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "ti2v": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "animate": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
}


def _resolve_wan22_repo_id(signature: ModelSignature) -> str:
    raw = str((signature.extras or {}).get("model_type", "")).strip().lower()
    return _WAN22_REPO_BY_MODEL_TYPE.get(raw, "")


def register_text_encoder(context: ParserContext, alias: str, component: str) -> None:
    mapping = context.metadata.setdefault("text_encoder_map", {})
    mapping[alias] = component


def build_estimated_config(
    context: ParserContext,
    signature: ModelSignature,
    *,
    repo_override: str | None = None,
    extra_metadata: Dict[str, object] | None = None,
) -> CodexEstimatedConfig:
    repo_id = repo_override or signature.repo_hint or ""
    if signature.family == ModelFamily.WAN22 and not repo_override:
        # Legacy signatures used a placeholder repo id; normalize to a concrete
        # diffusers repo so downstream asset resolution can locate vendored HF files.
        if not repo_id or repo_id.strip() in {"Wan-AI/Wan2.2"}:
            repo_id = _resolve_wan22_repo_id(signature)
    if not repo_id:
        repo_id = _FALLBACK_REPOS.get(signature.family, "")
    if not repo_id:
        raise ValidationError("Model signature missing repository hint", component="config")

    components = {
        name: CodexComponent(name=name, state_dict=component.tensors)
        for name, component in context.components.items()
        if component.tensors
    }

    text_map = dict(context.metadata.get("text_encoder_map", {}))
    extras = dict(signature.extras or {})
    if extra_metadata:
        extras.update(extra_metadata)

    detected = detect_quantization(context)
    quantization = signature.quantization
    if detected.kind != QuantizationKind.NONE:
        if quantization.kind not in (QuantizationKind.NONE, detected.kind):
            raise ValidationError(
                f"Detected quantization {detected.kind.value} conflicts with signature {quantization.kind.value}",
                component="config",
            )
        quantization = detected
        extras.setdefault("parser_quantization_detected", detected.kind.value)

    core_config = dict(_CORE_CONFIG_PRESETS.get(signature.family, {}))
    if core_config and "dtype" not in core_config:
        core_config["dtype"] = None

    return CodexEstimatedConfig(
        signature=signature,
        repo_id=repo_id,
        family=signature.family,
        prediction=signature.prediction,
        latent_format=signature.latent_format,
        quantization=quantization,
        components=components,
        text_encoder_map=text_map,
        extras=extras,
        core_config=core_config,
    )


__all__ = ["register_text_encoder", "build_estimated_config"]
