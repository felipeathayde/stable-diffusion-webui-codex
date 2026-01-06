"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ControlNet conversion helpers (config inference + key mapping).
Derives a supported ControlNet/UNet config from a ControlNet state dict and builds diffusers key maps used for weight conversion between
UNet variants.

Symbols (top-level; keep in sync; no ghosts):
- `_count_blocks` (function): Counts indexed block groups in a state dict given a prefix template.
- `_convert_config` (function): Normalizes/unrolls legacy-style UNet config fields into the expected diffusers shapes.
- `derive_controlnet_config` (function): Infers a compatible ControlNet config template from state dict shapes and returns a normalized config.
- `build_diffusers_key_map` (function): Builds a parameter name mapping between diffusers UNet keys and legacy/SD-style keys.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def _count_blocks(state_dict: Mapping[str, torch.Tensor], prefix: str) -> int:
    count = 0
    while True:
        template = prefix.format(count)
        if any(key.startswith(template) for key in state_dict.keys()):
            count += 1
        else:
            break
    return count


def _convert_config(unet_config: Dict[str, object]) -> Dict[str, object]:
    new_config = dict(unet_config)
    num_res_blocks = new_config.get("num_res_blocks")
    channel_mult = new_config.get("channel_mult")

    if isinstance(num_res_blocks, int) and isinstance(channel_mult, (list, tuple)):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    if isinstance(num_res_blocks, (list, tuple)) and "attention_resolutions" in new_config:
        attention_resolutions = new_config.pop("attention_resolutions")
        transformer_depth = new_config.get("transformer_depth")
        transformer_depth_middle = new_config.get("transformer_depth_middle")

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None and isinstance(transformer_depth, (list, tuple)):
            transformer_depth_middle = transformer_depth[-1]

        t_in: list[int] = []
        t_out: list[int] = []
        stride = 1
        for block_idx in range(len(num_res_blocks)):
            depth = transformer_depth[block_idx] if transformer_depth else 0
            res = num_res_blocks[block_idx]
            scaled_depth = depth if stride in attention_resolutions else 0
            t_in.extend([scaled_depth] * res)
            t_out.extend([scaled_depth] * (res + 1))
            stride *= 2
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    new_config["num_res_blocks"] = num_res_blocks
    return new_config


def derive_controlnet_config(state_dict: Mapping[str, torch.Tensor], dtype: torch.dtype | str | None = None) -> Dict[str, object]:
    match: Dict[str, object] = {}
    transformer_depth: list[int] = []

    attn_res = 1
    down_blocks = _count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = _count_blocks(state_dict, f"down_blocks.{i}.attentions.{{}}")
        res_blocks = _count_blocks(state_dict, f"down_blocks.{i}.resnets.{{}}")
        for ab in range(attn_blocks):
            transformer_count = _count_blocks(state_dict, f"down_blocks.{i}.attentions.{ab}.transformer_blocks.{{}}")
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                key = f"down_blocks.{i}.attentions.{ab}.transformer_blocks.0.attn2.to_k.weight"
                match["context_dim"] = state_dict[key].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            transformer_depth.extend([0] * res_blocks)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    def _config(base: Dict[str, object]) -> Dict[str, object]:
        cfg = dict(base)
        cfg["dtype"] = dtype
        return cfg

    SDXL_mid_cnet = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                     'num_classes': 'sequential', 'adm_in_channels': 2816, 'in_channels': 4, 'model_channels': 320,
                     'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 1,
                     'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     'use_temporal_attention': False, 'use_temporal_resblock': False})

    SDXL_small_cnet = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                       'num_classes': 'sequential', 'adm_in_channels': 2816, 'in_channels': 4, 'model_channels': 320,
                       'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 0, 0], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 0,
                       'use_linear_in_transformer': True, 'num_head_channels': 64, 'context_dim': 1, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'use_temporal_attention': False, 'use_temporal_resblock': False})

    SDXL_diffusers_inpaint = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'in_channels': 9, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False})

    SDXL_diffusers_ip2p = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                           'num_classes': 'sequential', 'adm_in_channels': 2816, 'in_channels': 8, 'model_channels': 320,
                           'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                           'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                           'use_temporal_attention': False, 'use_temporal_resblock': False})

    SD21 = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2],
            'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False})

    SD15 = _config({'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False})

    supported = [SDXL_mid_cnet, SDXL_small_cnet, SDXL_diffusers_inpaint, SDXL_diffusers_ip2p, SD21, SD15]

    for candidate in supported:
        if all(match.get(k) == candidate.get(k) for k in match.keys()):
            return _convert_config(candidate)

    raise ValueError("Unsupported ControlNet diffusers configuration")


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
}


def build_diffusers_key_map(unet_config: Dict[str, object]) -> Dict[str, str]:
    num_res_blocks = list(unet_config.get("num_res_blocks", []))
    channel_mult = list(unet_config.get("channel_mult", []))
    transformer_depth = list(unet_config.get("transformer_depth", []))
    transformer_depth_output = list(unet_config.get("transformer_depth_output", transformer_depth))
    transformer_depth_middle = unet_config.get("transformer_depth_middle", 0)

    diffusers_unet_map: Dict[str, str] = {}

    num_blocks = len(channel_mult)
    for block_idx in range(num_blocks):
        base = 1 + (num_res_blocks[block_idx] + 1) * block_idx
        for res_idx in range(num_res_blocks[block_idx]):
            diffusers_unet_map[f"down_blocks.{block_idx}.resnets.{res_idx}.conv1.weight"] = f"input_blocks.{base + res_idx}.0.in_layers.2.weight"
            diffusers_unet_map[f"down_blocks.{block_idx}.resnets.{res_idx}.conv1.bias"] = f"input_blocks.{base + res_idx}.0.in_layers.2.bias"
            diffusers_unet_map[f"down_blocks.{block_idx}.resnets.{res_idx}.conv2.weight"] = f"input_blocks.{base + res_idx}.0.out_layers.3.weight"
            diffusers_unet_map[f"down_blocks.{block_idx}.resnets.{res_idx}.conv2.bias"] = f"input_blocks.{base + res_idx}.0.out_layers.3.bias"
        if block_idx < num_blocks - 1:
            diffusers_unet_map[f"down_blocks.{block_idx}.downsamplers.0.conv.weight"] = f"input_blocks.{base + num_res_blocks[block_idx]}.0.op.weight"
            diffusers_unet_map[f"down_blocks.{block_idx}.downsamplers.0.conv.bias"] = f"input_blocks.{base + num_res_blocks[block_idx]}.0.op.bias"

    diffusers_unet_map["mid_block.attentions.0.to_k.weight"] = "middle_block.1.attn2.to_k.weight"
    diffusers_unet_map["mid_block.attentions.0.to_k.bias"] = "middle_block.1.attn2.to_k.bias"

    for block_idx in range(num_blocks):
        base = 3 * block_idx
        for res_idx in range(num_res_blocks[block_idx] + 1):
            diffusers_unet_map[f"up_blocks.{block_idx}.resnets.{res_idx}.conv1.weight"] = f"output_blocks.{base + res_idx}.0.in_layers.2.weight"
            diffusers_unet_map[f"up_blocks.{block_idx}.resnets.{res_idx}.conv1.bias"] = f"output_blocks.{base + res_idx}.0.in_layers.2.bias"
            diffusers_unet_map[f"up_blocks.{block_idx}.resnets.{res_idx}.conv2.weight"] = f"output_blocks.{base + res_idx}.0.out_layers.3.weight"
            diffusers_unet_map[f"up_blocks.{block_idx}.resnets.{res_idx}.conv2.bias"] = f"output_blocks.{base + res_idx}.0.out_layers.3.bias"
        if block_idx > 0:
            diffusers_unet_map[f"up_blocks.{block_idx}.upsamplers.0.conv.weight"] = f"output_blocks.{base + 2}.2.conv.weight"
            diffusers_unet_map[f"up_blocks.{block_idx}.upsamplers.0.conv.bias"] = f"output_blocks.{base + 2}.2.conv.bias"

    diffusers_unet_map["conv_in.weight"] = "input_blocks.0.0.weight"
    diffusers_unet_map["conv_in.bias"] = "input_blocks.0.0.bias"
    diffusers_unet_map["conv_out.weight"] = "out.2.weight"
    diffusers_unet_map["conv_out.bias"] = "out.2.bias"
    diffusers_unet_map["time_embedding.linear_1.weight"] = "time_embed.0.weight"
    diffusers_unet_map["time_embedding.linear_1.bias"] = "time_embed.0.bias"
    diffusers_unet_map["time_embedding.linear_2.weight"] = "time_embed.2.weight"
    diffusers_unet_map["time_embedding.linear_2.bias"] = "time_embed.2.bias"

    return diffusers_unet_map


__all__ = ["derive_controlnet_config", "build_diffusers_key_map"]
