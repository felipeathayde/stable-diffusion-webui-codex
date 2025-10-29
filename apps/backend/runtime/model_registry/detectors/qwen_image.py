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


QWEN_CORE_KEYS = (
    "img_in.weight",
    "proj_out.weight",
    "txt_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
)


class QwenImageDetector(ModelDetector):
    priority = 160

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *QWEN_CORE_KEYS):
            return False
        # Skip Wan family (temporal) by checking absence of modulation head.
        if any(key.endswith("head.modulation") for key in bundle.keys):
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        img_in = _shape(bundle, "img_in.weight")
        proj_out = _shape(bundle, "proj_out.weight")
        txt_norm = _shape(bundle, "txt_norm.weight")
        timestep_1 = _shape(bundle, "time_text_embed.timestep_embedder.linear_1.weight")

        channels_in = img_in[1] if img_in and len(img_in) == 2 else 64
        latent_channels = _infer_latent_channels(bundle)
        channels_out = latent_channels
        hidden_dim = img_in[0] if img_in else (proj_out[1] if proj_out else 3584)
        context_dim = txt_norm[0] if txt_norm else hidden_dim

        num_layers = count_blocks(bundle.keys, "transformer_blocks.{}.")
        num_layers = num_layers if num_layers else 60

        num_heads = hidden_dim // 128 if hidden_dim and hidden_dim % 128 == 0 else 24
        axes_rope = bundle.shape("time_ids")  # none in checkpoint; keep default extras
        time_embed_dim = timestep_1[0] if timestep_1 and len(timestep_1) >= 1 else None

        extras = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "time_embed_dim": time_embed_dim,
            "guidance_embeds": False,
            "latent_channels": latent_channels,
        }

        text_encoders = [
            TextEncoderSignature(
                name="qwen2_5_vl",
                key_prefix="text_encoder.",
                expected_dim=context_dim,
                tokenizer_hint="Qwen/Qwen-Image/tokenizer",
            )
        ]

        vae = VAESignature(key_prefix="vae.", latent_channels=_infer_latent_channels(bundle))

        return ModelSignature(
            family=ModelFamily.QWEN_IMAGE,
            repo_hint="Qwen/Qwen-Image",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.QWEN_IMAGE,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=num_layers,
                key_prefixes=["transformer_blocks."],
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


REGISTRY.register(QwenImageDetector())
