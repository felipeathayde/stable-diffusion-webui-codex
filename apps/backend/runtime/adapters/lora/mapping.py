"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA key mapping helpers for CLIP and UNet modules.
Builds stable key maps translating LoRA naming conventions into model parameter names, including architecture-aware UNet mapping using diffusers key conversion.

Symbols (top-level; keep in sync; no ghosts):
- `LORA_CLIP_MAP` (constant): CLIP attention/MLP suffix mapping used for legacy LoRA key compatibility.
- `_register_generic_weights` (function): Adds generic `{prefix: weight}` mappings for raw state dict keys.
- `model_lora_keys_clip` (function): Builds the LoRA-key → CLIP/text-encoder parameter map.
- `model_lora_keys_unet` (function): Builds the LoRA-key → UNet parameter map (includes diffusers mapping for UNet architectures).
"""

from __future__ import annotations

from typing import Dict

from apps.backend.runtime.misc.diffusers_state_dict import unet_to_diffusers
from apps.backend.runtime.model_registry.specs import CodexCoreArchitecture


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def _register_generic_weights(state_dict_keys, key_map):
    for key in state_dict_keys:
        if key.endswith(".weight"):
            key_map[f"text_encoders.{key[:-7]}"] = key
            key_map[key[:-7]] = key


def model_lora_keys_clip(model, key_map: Dict[str, str] | None = None) -> Dict[str, str]:
    state_keys = list(model.state_dict().keys())
    out = dict(key_map or {})
    _register_generic_weights(state_keys, out)

    config = getattr(model, "model_config", None)
    text_map = getattr(config, "text_encoder_map", {}) if config else {}

    alias_set = {key.split(".")[0] for key in state_keys if "." in key}
    preferred_order = ["clip_l", "clip_g", "clip_h", "t5xxl", "t5"]
    alias_order = [alias for alias in preferred_order if alias in alias_set]
    alias_order.extend(sorted(alias_set - set(alias_order)))

    alias_indices: Dict[str, int] = {}
    for alias in alias_order:
        alias_indices[alias] = len(alias_indices) + 1

    def _component_for_alias(alias: str) -> str | None:
        if alias in text_map:
            return text_map[alias]
        if alias == "clip_l" or alias == "clip_h":
            return "text_encoder"
        if alias == "clip_g":
            return "text_encoder_2"
        if alias.startswith("t5"):
            index = alias_indices.get(alias, 0)
            return f"text_encoder_{index}" if index > 1 else "text_encoder"
        return None

    for alias in alias_order:
        alias_index = alias_indices[alias]
        component_name = _component_for_alias(alias)

        # CLIP-style layers
        for layer in range(32):
            for suffix, mapped in LORA_CLIP_MAP.items():
                key = f"{alias}.transformer.text_model.encoder.layers.{layer}.{suffix}.weight"
                if key not in state_keys:
                    continue
                out[f"lora_te{alias_index}_text_model_encoder_layers_{layer}_{mapped}"] = key
                out[f"lora_te_text_model_encoder_layers_{layer}_{mapped}"] = key
                if component_name:
                    out[f"{component_name}.text_model.encoder.layers.{layer}.{suffix}"] = key

        # T5-style layers
        for key in state_keys:
            if not key.startswith(f"{alias}.transformer.") or not key.endswith(".weight"):
                continue
            logical = key[len(f"{alias}.transformer.") : -len(".weight")].replace(".", "_")
            out[f"lora_te{alias_index}_{logical}"] = key
            if component_name:
                out[f"{component_name}.{logical}"] = key

        proj_key = f"{alias}.transformer.text_projection.weight"
        if proj_key in state_keys:
            out[f"lora_te{alias_index}_text_projection"] = proj_key
            out["lora_prior_te_text_projection"] = proj_key

    return out


def model_lora_keys_unet(model, key_map: Dict[str, str] | None = None) -> Dict[str, str]:
    sd = model.state_dict()
    out = dict(key_map or {})

    for key in sd.keys():
        if key.startswith("diffusion_model."):
            core = key[len("diffusion_model.") :]
            if key.endswith(".weight"):
                clean = core[:-len(".weight")].replace(".", "_")
                out[f"lora_unet_{clean}"] = key
                out[key[:-len(".weight")]] = key
            else:
                out[key] = key

    model_config = getattr(model, "model_config", None)
    core_config = getattr(model_config, "core_config", None)
    core_signature = getattr(getattr(model_config, "signature", None), "core", None)
    if (
        isinstance(core_config, dict)
        and getattr(core_signature, "architecture", None) == CodexCoreArchitecture.UNET
    ):
        diffusers_keys = unet_to_diffusers(core_config)
        for diff_key, mapped in diffusers_keys.items():
            if not diff_key.endswith(".weight"):
                continue
            unet_param = f"diffusion_model.{mapped}"
            clean = diff_key[:-len(".weight")].replace(".", "_")
            out[f"lora_unet_{clean}"] = unet_param
            out[f"lycoris_{clean}"] = unet_param
            for prefix in ("", "unet."):
                compat = f"{prefix}{diff_key[:-len('.weight')]}".replace(".to_", ".processor.to_")
                if compat.endswith(".to_out.0"):
                    compat = compat[:-2]
                out[compat] = unet_param

    return out
