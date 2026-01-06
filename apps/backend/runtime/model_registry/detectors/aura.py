"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: AuraFlow model detector for the model registry.
Detects AuraFlow checkpoints via key signatures and builds a `ModelSignature` describing the core architecture, text encoders, and optional VAE.

Symbols (top-level; keep in sync; no ghosts):
- `AURA_KEYS` (constant): Key set used to identify AuraFlow checkpoints.
- `AuraFlowDetector` (class): Detector that builds a `ModelSignature` for AuraFlow checkpoints.
"""

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


AURA_KEYS = (
    "double_layers.0.attn.w1q.weight",
    "cond_seq_linear.weight",
    "positional_encoding",
)


class AuraFlowDetector:
    priority = 180

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        return has_all_keys(bundle, *AURA_KEYS)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        cond_seq = bundle.shape("cond_seq_linear.weight")
        cond_seq_dim = cond_seq[1] if cond_seq else 2048
        positional = bundle.shape("positional_encoding")
        seq_len = positional[1] if positional and len(positional) > 1 else None
        double_layers = count_blocks(bundle.keys, "double_layers.{}.")
        single_layers = count_blocks(bundle.keys, "single_layers.{}.")
        channels_in = bundle.shape("img_in.linear.weight")
        if channels_in:
            channels_in_val = channels_in[1]
        else:
            channels_in_val = 4
        channels_out = bundle.shape("out.linear.weight")
        if channels_out:
            channels_out_val = channels_out[0]
        else:
            channels_out_val = channels_in_val

        extras = {
            "cond_seq_dim": cond_seq_dim,
            "sequence_length": seq_len,
            "double_layers": double_layers,
            "single_layers": single_layers,
        }

        text_encoders = [
            TextEncoderSignature(
                name="aura_t5",
                key_prefix="text_encoders.aura_t5.",
                expected_dim=cond_seq_dim,
                tokenizer_hint="fal/AuraFlow/tokenizer",
            )
        ]

        vae = None
        if any(key.startswith("vae.") for key in bundle.keys):
            vae = VAESignature(key_prefix="vae.", latent_channels=16)

        return ModelSignature(
            family=ModelFamily.AURA,
            repo_hint="fal/AuraFlow",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.SD_XL,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=channels_in_val,
                channels_out=channels_out_val,
                context_dim=cond_seq_dim,
                temporal=False,
                depth=double_layers + single_layers,
                key_prefixes=["double_layers.", "single_layers."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


REGISTRY.register(AuraFlowDetector())
