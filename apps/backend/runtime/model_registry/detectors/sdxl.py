from __future__ import annotations

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
    TextEncoderSignature,
    VAESignature,
)


SDXL_REQUIRED_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
)


SDXL_REFINER_REQUIRED_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj_weight",
)


def _shape(bundle: SignalBundle, key: str, dim: int, default: int | None = None) -> int | None:
    shape = bundle.shape(key)
    if not shape or len(shape) <= dim:
        return default
    return int(shape[dim])


class StableDiffusionXLDetector:
    priority = 120

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *SDXL_REQUIRED_KEYS):
            return False
        if not bundle.has_prefix("first_stage_model."):
            return False
        # Expect dual embedders (clip-l + clip-g)
        if not bundle.has_prefix("conditioner.embedders.1."):
            return False
        embed_l = _shape(bundle, "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", -1)
        embed_g = _shape(bundle, "conditioner.embedders.1.model.token_embedding.weight", -1)
        if embed_l != 768 or embed_g != 1280:
            return False
        channels = _shape(bundle, "model.diffusion_model.input_blocks.0.0.weight", 1)
        return channels in (4, 9)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        channels_in = _shape(bundle, "model.diffusion_model.input_blocks.0.0.weight", 1, default=4) or 4
        channels_out = _shape(bundle, "model.diffusion_model.out.2.weight", 0, default=4) or 4
        depth = count_blocks(bundle.keys, "model.diffusion_model.output_blocks.{}.")

        return ModelSignature(
            family=ModelFamily.SDXL,
            repo_hint="stabilityai/stable-diffusion-xl-base-1.0",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_XL,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.UNET,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=2048,
                temporal=False,
                depth=depth,
                key_prefixes=["model.diffusion_model."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_l",
                    key_prefix="conditioner.embedders.0.transformer.",
                    expected_dim=768,
                    tokenizer_hint="openai/clip-vit-large-patch14",
                ),
                TextEncoderSignature(
                    name="clip_g",
                    key_prefix="conditioner.embedders.1.model.",
                    expected_dim=1280,
                    tokenizer_hint="openclip/ViT-bigG-14",
                ),
            ],
            vae=VAESignature(key_prefix="first_stage_model.", latent_channels=channels_out),
            extras={"sdxl_variant": "base"},
        )


class StableDiffusionXLRefinerDetector:
    priority = 125

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *SDXL_REFINER_REQUIRED_KEYS):
            return False
        if bundle.has_prefix("conditioner.embedders.1."):
            return False
        embed_dim = _shape(bundle, "conditioner.embedders.0.model.token_embedding.weight", -1)
        return embed_dim == 1280

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        channels_in = _shape(bundle, "model.diffusion_model.input_blocks.0.0.weight", 1, default=4) or 4
        channels_out = _shape(bundle, "model.diffusion_model.out.2.weight", 0, default=4) or 4
        depth = count_blocks(bundle.keys, "model.diffusion_model.output_blocks.{}.")

        return ModelSignature(
            family=ModelFamily.SDXL_REFINER,
            repo_hint="stabilityai/stable-diffusion-xl-refiner-1.0",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_XL,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.UNET,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=1280,
                temporal=False,
                depth=depth,
                key_prefixes=["model.diffusion_model."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_g",
                    key_prefix="conditioner.embedders.0.model.",
                    expected_dim=1280,
                    tokenizer_hint="openclip/ViT-bigG-14",
                ),
            ],
            vae=VAESignature(key_prefix="first_stage_model.", latent_channels=channels_out),
            extras={"sdxl_variant": "refiner"},
        )


REGISTRY.register(StableDiffusionXLDetector())
REGISTRY.register(StableDiffusionXLRefinerDetector())
