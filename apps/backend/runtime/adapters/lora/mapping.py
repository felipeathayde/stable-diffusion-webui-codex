"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA key mapping helpers for CLIP and UNet modules.
Builds stable key maps translating LoRA naming conventions into model patch targets (usually parameter names; sometimes `(parameter, offset)` tuples for slice patches),
including architecture-aware UNet mapping using diffusers key conversion.

Symbols (top-level; keep in sync; no ghosts):
- `LORA_CLIP_MAP` (constant): CLIP attention/MLP suffix mapping used for legacy LoRA key compatibility.
- `_register_generic_weights` (function): Adds generic `{prefix: weight}` mappings for raw state dict keys.
- `_build_unet_state_lookup` (function): Builds a canonical+runtime UNet key lookup using the SDXL keymap and runtime state keys.
- `_resolve_active_model_config` (function): Resolves the active runtime config object used by keymap generation.
- `model_lora_keys_clip` (function): Builds the LoRA-key → CLIP/text-encoder patch-target map (supports slice targets for fused-QKV encoders).
- `model_lora_keys_unet` (function): Builds the LoRA-key → UNet parameter map (includes diffusers mapping for UNet architectures).
"""

from __future__ import annotations

from typing import Dict

from apps.backend.runtime.misc.diffusers_state_dict import unet_to_diffusers
from apps.backend.runtime.model_registry.specs import CodexCoreArchitecture
from apps.backend.runtime.adapters.base import PatchTarget
from apps.backend.runtime.state_dict.keymap_sdxl_checkpoint import (
    remap_sdxl_checkpoint_state_dict,
)


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


def _build_unet_state_lookup(state_dict_keys) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for raw_key in state_dict_keys:
        key = str(raw_key)
        lookup.setdefault(key, key)
        if key.startswith("model."):
            lookup.setdefault(key[len("model.") :], key)

    canonical_source = {str(raw_key): str(raw_key) for raw_key in state_dict_keys}
    try:
        _style, remapped = remap_sdxl_checkpoint_state_dict(canonical_source)
        for canonical_key, original_key in remapped.items():
            canonical = str(canonical_key)
            original = str(original_key)
            lookup.setdefault(canonical, original)
            if canonical.startswith("model."):
                lookup.setdefault(canonical[len("model.") :], original)
    except Exception:
        # Non-SDXL keys are still covered by direct runtime lookup.
        pass

    return lookup


def _resolve_active_model_config(model):
    candidates = [
        getattr(model, "model_config", None),
        getattr(model, "config", None),
    ]
    diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is not None:
        candidates.extend(
            [
                getattr(diffusion_model, "model_config", None),
                getattr(diffusion_model, "codex_config", None),
            ]
        )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def model_lora_keys_clip(model, key_map: Dict[str, PatchTarget] | None = None) -> Dict[str, PatchTarget]:
    state = model.state_dict()
    state_keys = list(state.keys())
    state_key_set = set(state_keys)
    out: Dict[str, PatchTarget] = dict(key_map or {})
    _register_generic_weights(state_keys, out)

    config = _resolve_active_model_config(model)
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

        in_proj_probe = f"{alias}.transformer.text_model.encoder.layers.0.self_attn.in_proj.weight"
        fused_in_proj_chunk: int | None = None
        if in_proj_probe in state:
            tensor = state.get(in_proj_probe)
            shape = getattr(tensor, "shape", None)
            if shape and len(shape) >= 1:
                total = int(shape[0])
                if total % 3 != 0:
                    raise RuntimeError(
                        "LoRA mapping: fused CLIP in_proj first dim is not divisible by 3 "
                        f"(alias={alias!r} shape={shape!r})"
                    )
                fused_in_proj_chunk = total // 3

        def _fused_qkv_target(layer: int, proj: str) -> PatchTarget | None:
            if fused_in_proj_chunk is None:
                return None
            index = {"q": 0, "k": 1, "v": 2}.get(proj)
            if index is None:
                return None
            in_proj_key = f"{alias}.transformer.text_model.encoder.layers.{layer}.self_attn.in_proj.weight"
            if in_proj_key not in state:
                return None
            start = index * fused_in_proj_chunk
            return (in_proj_key, (0, start, fused_in_proj_chunk))

        # CLIP-style layers
        for layer in range(32):
            for suffix, mapped in LORA_CLIP_MAP.items():
                key = f"{alias}.transformer.text_model.encoder.layers.{layer}.{suffix}.weight"
                target: PatchTarget | None = key if key in state_key_set else None
                if target is None and suffix.startswith("self_attn.") and suffix.endswith("_proj"):
                    proj = suffix[len("self_attn.") : -len("_proj")]
                    target = _fused_qkv_target(layer, proj)
                if target is None:
                    continue
                out[f"lora_te{alias_index}_text_model_encoder_layers_{layer}_{mapped}"] = target
                out[f"lora_te_text_model_encoder_layers_{layer}_{mapped}"] = target
                if component_name:
                    out[f"{component_name}.text_model.encoder.layers.{layer}.{suffix}"] = target

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
    state_lookup = _build_unet_state_lookup(sd.keys())

    for logical_key, target_key in state_lookup.items():
        if logical_key.endswith(".weight"):
            clean = logical_key[:-len(".weight")].replace(".", "_")
            out[f"lora_unet_{clean}"] = target_key
            out[f"lycoris_{clean}"] = target_key
            out[logical_key[:-len(".weight")]] = target_key
            if logical_key.startswith("diffusion_model."):
                core_logical = logical_key[len("diffusion_model.") :]
                core_clean = core_logical[:-len(".weight")].replace(".", "_")
                out[f"lora_unet_{core_clean}"] = target_key
                out[f"lycoris_{core_clean}"] = target_key
            if logical_key.startswith("model.diffusion_model."):
                core_logical = logical_key[len("model.diffusion_model.") :]
                core_clean = core_logical[:-len(".weight")].replace(".", "_")
                out[f"lora_unet_{core_clean}"] = target_key
                out[f"lycoris_{core_clean}"] = target_key
        else:
            out[logical_key] = target_key

    model_config = _resolve_active_model_config(model)
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
            unet_param = state_lookup.get(f"diffusion_model.{mapped}")
            if not unet_param:
                continue
            clean = diff_key[:-len(".weight")].replace(".", "_")
            out[f"lora_unet_{clean}"] = unet_param
            out[f"lycoris_{clean}"] = unet_param
            for prefix in ("", "unet."):
                compat = f"{prefix}{diff_key[:-len('.weight')]}".replace(".to_", ".processor.to_")
                if compat.endswith(".to_out.0"):
                    compat = compat[:-2]
                out[compat] = unet_param

    return out
