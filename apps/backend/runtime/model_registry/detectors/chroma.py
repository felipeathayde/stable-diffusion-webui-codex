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
    TextEncoderSignature,
    VAESignature,
)


CHROMA_KEYS = (
    "img_in.weight",
    "proj_out.weight",
    "context_embedder.weight",
    "distilled_guidance_layer.in_proj.weight",
)


class ChromaDetector(ModelDetector):
    priority = 150

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *CHROMA_KEYS):
            return False
        # Radiance variants include NeRF blocks; we treat them as not yet ported.
        if any(key.startswith("nerf_blocks.") for key in bundle.keys):
            raise NotImplementedError("Chroma Radiance detection not yet ported")
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        img_in = _shape(bundle, "img_in.weight")
        proj_out = _shape(bundle, "proj_out.weight")
        context_embedder = _shape(bundle, "context_embedder.weight")
        guidance_in = _shape(bundle, "distilled_guidance_layer.in_proj.weight")

        channels_in = img_in[1] if img_in and len(img_in) == 2 else 64
        channels_out = proj_out[0] if proj_out and len(proj_out) == 2 else channels_in
        hidden_dim = img_in[0] if img_in else (proj_out[1] if proj_out else 3072)
        context_dim = context_embedder[1] if context_embedder and len(context_embedder) == 2 else 4096
        guidance_dim = guidance_in[1] if guidance_in and len(guidance_in) == 2 else 64

        double_layers = count_blocks(bundle.keys, "double_blocks.{}.")
        single_layers = count_blocks(bundle.keys, "single_transformer_blocks.{}.")

        extras = {
            "double_layers": double_layers,
            "single_layers": single_layers,
            "hidden_dim": hidden_dim,
            "guidance_dim": guidance_dim,
        }

        text_encoders = [
            TextEncoderSignature(
                name="t5xxl",
                key_prefix="text_encoder.",
                expected_dim=context_dim,
                tokenizer_hint="lodestones/Chroma/tokenizer",
            )
        ]

        vae = VAESignature(key_prefix="vae.", latent_channels=_infer_latents(bundle))

        return ModelSignature(
            family=ModelFamily.CHROMA,
            repo_hint="lodestones/Chroma",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.FLOW16,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=double_layers + single_layers,
                key_prefixes=["double_blocks.", "single_transformer_blocks."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


def _infer_latents(bundle: SignalBundle) -> int:
    vae_key = "vae.decoder.conv_out.weight"
    vae_shape = bundle.shape(vae_key)
    if vae_shape and len(vae_shape) >= 2:
        return int(vae_shape[1])
    tensor = bundle.state_dict.get(vae_key)
    if isinstance(tensor, torch.Tensor):
        return int(tensor.shape[1])
    return 16


def _shape(bundle: SignalBundle, key: str) -> Optional[tuple[int, ...]]:
    shape = bundle.shape(key)
    if shape is not None:
        return tuple(int(v) for v in shape)
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return tuple(int(v) for v in tensor.shape)
    return None


REGISTRY.register(ChromaDetector())
