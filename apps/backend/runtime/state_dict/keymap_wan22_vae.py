"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 VAE key-style detection + strict canonical remaps for 2D/3D lanes.
Owns WAN22 model keymap behavior for VAE checkpoints and keeps remap logic out of
router/payload seams. Supports 2D native LDM VAE keys and 3D Codex/Diffusers
keyspaces with fail-loud mixed-style/collision guards.

Symbols (top-level; keep in sync; no ghosts):
- `remap_wan22_vae_2d_state_dict` (function): Normalizes/validates WAN22 2D VAE keys.
- `remap_wan22_vae_3d_state_dict` (function): Normalizes/remaps WAN22 3D VAE keys (`diffusers|codex` → canonical codex keyspace).
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

from apps.backend.runtime.state_dict.key_mapping import KeyMappingError, strip_repeated_prefixes
from apps.backend.runtime.state_dict.keymap_wan21_vae import remap_wan21_vae_state_dict

_WRAPPER_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)

_WAN22_2D_REQUIRED_KEYS = (
    "encoder.conv_in.weight",
    "decoder.conv_in.weight",
)
_WAN22_2D_OPTIONAL_QUANT_KEYS = (
    "quant_conv.weight",
    "post_quant_conv.weight",
)

_FIXED_DIFFUSERS_TO_CODEX_KEYS: dict[str, str] = {
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "encoder.mid_block.resnets.0.norm1.gamma": "encoder.middle.0.residual.0.gamma",
    "encoder.mid_block.resnets.0.conv1.weight": "encoder.middle.0.residual.2.weight",
    "encoder.mid_block.resnets.0.conv1.bias": "encoder.middle.0.residual.2.bias",
    "encoder.mid_block.resnets.0.norm2.gamma": "encoder.middle.0.residual.3.gamma",
    "encoder.mid_block.resnets.0.conv2.weight": "encoder.middle.0.residual.6.weight",
    "encoder.mid_block.resnets.0.conv2.bias": "encoder.middle.0.residual.6.bias",
    "encoder.mid_block.resnets.1.norm1.gamma": "encoder.middle.2.residual.0.gamma",
    "encoder.mid_block.resnets.1.conv1.weight": "encoder.middle.2.residual.2.weight",
    "encoder.mid_block.resnets.1.conv1.bias": "encoder.middle.2.residual.2.bias",
    "encoder.mid_block.resnets.1.norm2.gamma": "encoder.middle.2.residual.3.gamma",
    "encoder.mid_block.resnets.1.conv2.weight": "encoder.middle.2.residual.6.weight",
    "encoder.mid_block.resnets.1.conv2.bias": "encoder.middle.2.residual.6.bias",
    "decoder.mid_block.resnets.0.norm1.gamma": "decoder.middle.0.residual.0.gamma",
    "decoder.mid_block.resnets.0.conv1.weight": "decoder.middle.0.residual.2.weight",
    "decoder.mid_block.resnets.0.conv1.bias": "decoder.middle.0.residual.2.bias",
    "decoder.mid_block.resnets.0.norm2.gamma": "decoder.middle.0.residual.3.gamma",
    "decoder.mid_block.resnets.0.conv2.weight": "decoder.middle.0.residual.6.weight",
    "decoder.mid_block.resnets.0.conv2.bias": "decoder.middle.0.residual.6.bias",
    "decoder.mid_block.resnets.1.norm1.gamma": "decoder.middle.2.residual.0.gamma",
    "decoder.mid_block.resnets.1.conv1.weight": "decoder.middle.2.residual.2.weight",
    "decoder.mid_block.resnets.1.conv1.bias": "decoder.middle.2.residual.2.bias",
    "decoder.mid_block.resnets.1.norm2.gamma": "decoder.middle.2.residual.3.gamma",
    "decoder.mid_block.resnets.1.conv2.weight": "decoder.middle.2.residual.6.weight",
    "decoder.mid_block.resnets.1.conv2.bias": "decoder.middle.2.residual.6.bias",
    "encoder.mid_block.attentions.0.norm.gamma": "encoder.middle.1.norm.gamma",
    "encoder.mid_block.attentions.0.to_qkv.weight": "encoder.middle.1.to_qkv.weight",
    "encoder.mid_block.attentions.0.to_qkv.bias": "encoder.middle.1.to_qkv.bias",
    "encoder.mid_block.attentions.0.proj.weight": "encoder.middle.1.proj.weight",
    "encoder.mid_block.attentions.0.proj.bias": "encoder.middle.1.proj.bias",
    "decoder.mid_block.attentions.0.norm.gamma": "decoder.middle.1.norm.gamma",
    "decoder.mid_block.attentions.0.to_qkv.weight": "decoder.middle.1.to_qkv.weight",
    "decoder.mid_block.attentions.0.to_qkv.bias": "decoder.middle.1.to_qkv.bias",
    "decoder.mid_block.attentions.0.proj.weight": "decoder.middle.1.proj.weight",
    "decoder.mid_block.attentions.0.proj.bias": "decoder.middle.1.proj.bias",
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
}

_DECODER_RESNET_TO_UPSAMPLE_INDEX: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 8,
    (2, 1): 9,
    (2, 2): 10,
    (3, 0): 12,
    (3, 1): 13,
    (3, 2): 14,
}

_DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX: dict[int, int] = {
    0: 3,
    1: 7,
    2: 11,
}


def _normalize_state_dict(state_dict: MutableMapping[str, Any], *, detector_name: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = strip_repeated_prefixes(str(key), _WRAPPER_PREFIXES)
        if normalized_key in normalized:
            raise KeyMappingError(
                f"{detector_name}: normalized key collision for key={normalized_key!r}."
            )
        normalized[normalized_key] = value
    return normalized


def _map_encoder_down_block_key(key: str) -> str:
    if ".resnets." in key:
        raise KeyMappingError(
            "wan22_vae_3d_key_style: unsupported encoder down_block layout "
            f"for key={key!r} (expected canonical encoder.down_blocks.<idx>.<field>)."
        )
    mapped = key.replace("encoder.down_blocks.", "encoder.downsamples.", 1)
    mapped = mapped.replace(".norm1.gamma", ".residual.0.gamma")
    mapped = mapped.replace(".conv1.weight", ".residual.2.weight")
    mapped = mapped.replace(".conv1.bias", ".residual.2.bias")
    mapped = mapped.replace(".norm2.gamma", ".residual.3.gamma")
    mapped = mapped.replace(".conv2.weight", ".residual.6.weight")
    mapped = mapped.replace(".conv2.bias", ".residual.6.bias")
    mapped = mapped.replace(".conv_shortcut.weight", ".shortcut.weight")
    mapped = mapped.replace(".conv_shortcut.bias", ".shortcut.bias")
    return mapped


def _map_decoder_up_block_key(key: str) -> str:
    resnet_match = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if resnet_match is not None:
        block_index = int(resnet_match.group(1))
        resnet_index = int(resnet_match.group(2))
        tail = str(resnet_match.group(3))
        if (block_index, resnet_index) not in _DECODER_RESNET_TO_UPSAMPLE_INDEX:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block residual index "
                f"(block={block_index} resnet={resnet_index}) for key={key!r}."
            )
        upsample_index = _DECODER_RESNET_TO_UPSAMPLE_INDEX[(block_index, resnet_index)]
        if tail == "norm1.gamma":
            mapped_tail = "residual.0.gamma"
        elif tail == "conv1.weight":
            mapped_tail = "residual.2.weight"
        elif tail == "conv1.bias":
            mapped_tail = "residual.2.bias"
        elif tail == "norm2.gamma":
            mapped_tail = "residual.3.gamma"
        elif tail == "conv2.weight":
            mapped_tail = "residual.6.weight"
        elif tail == "conv2.bias":
            mapped_tail = "residual.6.bias"
        elif tail.startswith("conv_shortcut."):
            mapped_tail = "shortcut." + tail[len("conv_shortcut.") :]
        else:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block residual field "
                f"tail={tail!r} key={key!r}."
            )
        return f"decoder.upsamples.{upsample_index}.{mapped_tail}"

    upsample_match = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", key)
    if upsample_match is not None:
        block_index = int(upsample_match.group(1))
        tail = str(upsample_match.group(2))
        if block_index not in _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block upsampler index "
                f"(block={block_index}) for key={key!r}."
            )
        upsample_index = _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX[block_index]
        return f"decoder.upsamples.{upsample_index}.{tail}"

    raise KeyMappingError(
        "wan22_vae_3d_key_style: unsupported decoder up_block key layout "
        f"for key={key!r}."
    )


def remap_wan22_vae_2d_state_dict(state_dict: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
    normalized = _normalize_state_dict(state_dict, detector_name="wan22_vae_2d_key_style")
    keys = tuple(normalized.keys())
    keys_set = frozenset(keys)

    missing_required = [key for key in _WAN22_2D_REQUIRED_KEYS if key not in keys_set]
    if missing_required:
        raise KeyMappingError(
            "wan22_vae_2d_key_style: remap output is missing required canonical keys. "
            f"missing_sample={missing_required[:10]}"
        )
    if not any(key in keys_set for key in _WAN22_2D_OPTIONAL_QUANT_KEYS):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: remap output is missing quantization convolution keys "
            f"(requires one of {sorted(_WAN22_2D_OPTIONAL_QUANT_KEYS)})."
        )
    if any(key.startswith("encoder.downsamples.") or key.startswith("decoder.upsamples.") for key in keys):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: received 3d codex keyspace in 2d lane (encoder.downsamples/decoder.upsamples)."
        )
    if any(key.startswith("encoder.down_blocks.") or key.startswith("decoder.up_blocks.") for key in keys):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: received 3d diffusers keyspace in 2d lane (encoder.down_blocks/decoder.up_blocks)."
        )
    return "ldm_2d", normalized


def remap_wan22_vae_3d_state_dict(state_dict: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
    normalized = _normalize_state_dict(state_dict, detector_name="wan22_vae_3d_key_style")

    has_codex = any(
        key.startswith("encoder.downsamples.")
        or key.startswith("decoder.upsamples.")
        or key in {"conv1.weight", "conv2.weight"}
        for key in normalized.keys()
    )
    has_diffusers = any(
        key.startswith("encoder.down_blocks.")
        or key.startswith("decoder.up_blocks.")
        or key.startswith("quant_conv.")
        or key.startswith("post_quant_conv.")
        for key in normalized.keys()
    )

    if has_codex and has_diffusers:
        raise KeyMappingError(
            "wan22_vae_3d_key_style: mixed codex/diffusers VAE keyspace detected "
            "(cannot resolve a single canonical lane)."
        )

    if has_diffusers:
        mapped: dict[str, Any] = {}
        for key, value in normalized.items():
            if key in _FIXED_DIFFUSERS_TO_CODEX_KEYS:
                mapped_key = _FIXED_DIFFUSERS_TO_CODEX_KEYS[key]
            elif key.startswith("encoder.down_blocks."):
                mapped_key = _map_encoder_down_block_key(key)
            elif key.startswith("decoder.up_blocks."):
                mapped_key = _map_decoder_up_block_key(key)
            else:
                mapped_key = key
            if mapped_key in mapped:
                raise KeyMappingError(
                    "wan22_vae_3d_key_style: remap produced output collision "
                    f"for mapped key={mapped_key!r} (source key={key!r})."
                )
            mapped[mapped_key] = value
        _, validated = remap_wan21_vae_state_dict(mapped)
        return "diffusers", validated

    _, validated = remap_wan21_vae_state_dict(normalized)
    return "codex", validated


__all__ = ["remap_wan22_vae_2d_state_dict", "remap_wan22_vae_3d_state_dict"]
