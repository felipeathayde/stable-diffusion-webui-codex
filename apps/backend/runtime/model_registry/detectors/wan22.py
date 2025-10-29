from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks
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
        in_channels, model_dim, patch_size = _interpret_patch_shape(patch_shape)

        head_shape = bundle.shape(f"{prefix}head.head.weight")
        assert head_shape is not None
        latent_channels = _infer_latent_channels(head_shape, patch_size)

        num_layers = count_blocks(bundle.keys, f"{prefix}blocks.{{}}.")

        model_type = _detect_model_type(bundle, prefix)

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
        }

        text_encoders = _collect_text_encoders(bundle, prefix)

        return ModelSignature(
            family=ModelFamily.WAN22,
            repo_hint="Wan-AI/Wan2.2",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.WAN22,
            quantization=QuantizationHint(),
            unet=UNetSignature(
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


def _interpret_patch_shape(shape: tuple[int, ...]) -> tuple[int, int, tuple[int, int, int]]:
    if shape[0] > shape[-1]:  # PyTorch (out, in, kt, kh, kw)
        out_channels = int(shape[0])
        in_channels = int(shape[1])
        patch_size = (int(shape[2]), int(shape[3]), int(shape[4]))
    else:  # GGUF (kt, kh, kw, in, out)
        out_channels = int(shape[-1])
        in_channels = int(shape[-2])
        patch_size = (int(shape[-3]), int(shape[-4]), int(shape[-5]))
    # Ensure canonical ordering (t, h, w)
    if patch_size[0] <= 2 and patch_size[1] <= 2 and patch_size[2] <= 2:
        patch = patch_size
    else:
        patch = tuple(sorted(patch_size, reverse=False))
    # convert to (t,h,w) with expectation (1,2,2)
    patch = (patch[0], patch[1], patch[2])
    return in_channels, out_channels, patch


def _infer_latent_channels(head_shape: tuple[int, ...], patch_size: tuple[int, int, int]) -> int:
    patch_prod = int(patch_size[0] * patch_size[1] * patch_size[2])
    if len(head_shape) != 2:
        return patch_prod
    a, b = head_shape
    if a % patch_prod == 0:
        return int(a // patch_prod)
    if b % patch_prod == 0:
        return int(b // patch_prod)
    return int(a)


def _detect_model_type(bundle: SignalBundle, prefix: str) -> str:
    checks = {
        "vace": f"{prefix}vace_patch_embedding.weight",
        "camera": f"{prefix}control_adapter.conv.weight",
        "camera_2.2": f"{prefix}control_adapter.conv.weight",
        "s2v": f"{prefix}casual_audio_encoder.encoder.final_linear.weight",
        "humo": f"{prefix}audio_proj.audio_proj_glob_1.layer.bias",
        "animate": f"{prefix}face_adapter.fuser_blocks.0.k_norm.weight",
    }
    for model_type, key in checks.items():
        if key in bundle.state_dict:
            if model_type == "camera" and f"{prefix}control_adapter.conv.weight" in bundle.state_dict:
                if f"{prefix}img_emb.proj.0.bias" in bundle.state_dict:
                    return "camera"
                return "camera_2.2"
            return model_type
    if f"{prefix}img_emb.proj.0.bias" in bundle.state_dict:
        return "i2v"
    return "t2v"


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


REGISTRY.register(Wan22Detector())
