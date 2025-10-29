from __future__ import annotations

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks, has_all_keys
from apps.backend.runtime.model_registry.specs import (
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    TextEncoderSignature,
    UNetSignature,
    VAESignature,
)


SD3_REQUIRED_KEYS = (
    "joint_blocks.0.context_block.attn.qkv.weight",
    "x_embedder.proj.weight",
    "final_layer.linear.weight",
)


class StableDiffusion3Detector:
    priority = 160

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        return has_all_keys(bundle, *SD3_REQUIRED_KEYS)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        channels_in = _shape(bundle, "x_embedder.proj.weight", 1) or 16
        channels_out = _shape(bundle, "final_layer.linear.weight", 0) or channels_in
        context_dim = _shape(bundle, "context_embedder.weight", 0) or 4096
        depth = count_blocks(bundle.keys, "joint_blocks.{}.")

        text_encoders = [
            TextEncoderSignature(
                name="clip_l",
                key_prefix="text_encoders.clip_l.",
                expected_dim=768,
            ),
            TextEncoderSignature(
                name="clip_g",
                key_prefix="text_encoders.clip_g.",
                expected_dim=1280,
            ),
            TextEncoderSignature(
                name="t5xxl",
                key_prefix="text_encoders.t5xxl.",
                expected_dim=4096,
            ),
        ]

        vae = None
        if any(key.startswith("vae.") for key in bundle.keys):
            vae = VAESignature(key_prefix="vae.", latent_channels=16)

        extras = {
            "joint_blocks": depth,
        }

        return ModelSignature(
            family=ModelFamily.SD3,
            repo_hint="stabilityai/stable-diffusion-3-medium-diffusers",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_3,
            quantization=QuantizationHint(),
            unet=UNetSignature(
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=depth,
                key_prefixes=["joint_blocks."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


def _shape(bundle: SignalBundle, key: str, dim: int) -> int | None:
    shape = bundle.shape(key)
    if not shape or len(shape) <= dim:
        return None
    return int(shape[dim])


REGISTRY.register(StableDiffusion3Detector())
