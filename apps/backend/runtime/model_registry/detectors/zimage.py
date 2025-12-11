from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
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
from apps.backend.runtime.utils import ParameterGGUF


# Core keys for Z Image Turbo (NextDiT/Lumina2 format)
# Some checkpoints use these directly, others have model.diffusion_model. prefix
ZIMAGE_CORE_KEYS = (
    "x_embedder.weight",
    "cap_embedder.0.weight",
    "t_embedder.mlp.0.weight",
    "layers.0.adaLN_modulation.0.weight",
    "final_layer.linear.weight",
)

# Prefix used by some safetensors exports (FP8, BF16)
_DIFFUSION_MODEL_PREFIX = "model.diffusion_model."


def _has_zimage_keys(bundle: SignalBundle) -> bool:
    """Check if bundle has Z Image core keys, with or without prefix."""
    keys_set = set(bundle.keys)
    
    # Check unprefixed (GGUF format)
    if all(k in keys_set for k in ZIMAGE_CORE_KEYS):
        return True
    
    # Check with model.diffusion_model. prefix (safetensors format)
    prefixed_keys = tuple(_DIFFUSION_MODEL_PREFIX + k for k in ZIMAGE_CORE_KEYS)
    if all(k in keys_set for k in prefixed_keys):
        return True
    
    return False


class ZImageDetector(ModelDetector):
    priority = 160

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not _has_zimage_keys(bundle):
            return False
        # Skip Wan family (temporal) by checking absence of modulation head.
        if any(key.endswith("head.modulation") for key in bundle.keys):
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        # Detect if this is a GGUF quantized checkpoint
        is_gguf = any(isinstance(v, ParameterGGUF) for v in bundle.state_dict.values())
        
        # Detect which key prefix is used (prefixed for safetensors, unprefixed for GGUF)
        keys_set = set(bundle.keys)
        prefix = ""
        if (_DIFFUSION_MODEL_PREFIX + "x_embedder.weight") in keys_set:
            prefix = _DIFFUSION_MODEL_PREFIX
        
        # NextDiT/Lumina2 format keys (with detected prefix)
        x_embedder = _shape(bundle, prefix + "x_embedder.weight")
        cap_embedder = _shape(bundle, prefix + "cap_embedder.0.weight")
        final_layer = _shape(bundle, prefix + "final_layer.linear.weight")
        t_embedder = _shape(bundle, prefix + "t_embedder.mlp.0.weight")

        # Infer dimensions from checkpoint shapes (with safe access)
        hidden_dim = x_embedder[0] if x_embedder and len(x_embedder) >= 1 else 3840
        context_dim = cap_embedder[1] if cap_embedder and len(cap_embedder) >= 2 else 2560
        
        # Output: patch_size^2 * latent_channels
        if final_layer and len(final_layer) >= 1:
            out_dim = final_layer[0]
            latent_channels = out_dim // 4  # patch_size=2, so 2*2=4
        else:
            latent_channels = 16
        
        channels_in = latent_channels * 4  # patch_size^2 * latent_channels
        channels_out = latent_channels

        num_layers = count_blocks(bundle.keys, prefix + "layers.{}.")
        num_layers = num_layers if num_layers else 30
        
        num_refiner_layers = count_blocks(bundle.keys, prefix + "context_refiner.{}.")
        num_refiner_layers = num_refiner_layers if num_refiner_layers else 2

        num_heads = hidden_dim // 128 if hidden_dim and hidden_dim % 128 == 0 else 30

        extras = {
            "hidden_dim": hidden_dim,
            "context_dim": context_dim,
            "num_layers": num_layers,
            "num_refiner_layers": num_refiner_layers,
            "num_heads": num_heads,
            "latent_channels": latent_channels,
            "guidance_embeds": False,
            # Both GGUF and prefixed safetensors (FP8/BF16) are core-only
            # They don't have embedded VAE or text encoder
            "gguf_core_only": is_gguf or bool(prefix),
        }

        text_encoders = [
            TextEncoderSignature(
                name="qwen3_4b",
                key_prefix="text_encoder.",
                expected_dim=context_dim,
                tokenizer_hint="Qwen/Qwen3-4B",
            )
        ]

        # Core-only models (GGUF or prefixed safetensors) don't have embedded VAE
        is_core_only = is_gguf or bool(prefix)
        vae = VAESignature(key_prefix="vae.", latent_channels=latent_channels) if not is_core_only else None
        
        # Set quantization hint based on detection
        if is_gguf:
            quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf")
        else:
            quantization = QuantizationHint()

        return ModelSignature(
            family=ModelFamily.ZIMAGE,
            repo_hint="Alibaba-TongYi/Z-Image-Turbo",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.ZIMAGE,
            quantization=quantization,
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=num_layers,
                key_prefixes=[prefix + "layers."] if prefix else ["layers."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


def _infer_latent_channels(bundle: SignalBundle) -> int:
    proj_out = _shape(bundle, "proj_out.weight")
    if proj_out and len(proj_out) == 2:
        patch_area = 4
        return int(proj_out[0] // patch_area) if proj_out[0] % patch_area == 0 else int(proj_out[0])
    latent = bundle.shape("vae.decoder.conv_out.weight")
    if latent:
        return int(latent[1])
    return 16


def _shape(bundle: SignalBundle, key: str) -> Optional[tuple[int, ...]]:
    shape = bundle.shape(key)
    if shape is not None:
        return tuple(int(v) for v in shape)
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return tuple(int(v) for v in tensor.shape)
    return None


REGISTRY.register(ZImageDetector())
