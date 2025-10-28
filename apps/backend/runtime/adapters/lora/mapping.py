from __future__ import annotations

from typing import Dict

from apps.backend.runtime.misc.diffusers_state_dict import unet_to_diffusers


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

    clip_l_seen = False
    clip_g_seen = False

    for layer in range(32):
        for suffix, mapped in LORA_CLIP_MAP.items():
            k_l = f"clip_l.transformer.text_model.encoder.layers.{layer}.{suffix}.weight"
            if k_l in state_keys:
                clip_l_seen = True
                out[f"lora_te1_text_model_encoder_layers_{layer}_{mapped}"] = k_l
                out[f"text_encoder.text_model.encoder.layers.{layer}.{suffix}"] = k_l
                out[f"lora_te_text_model_encoder_layers_{layer}_{mapped}"] = k_l
            k_g = f"clip_g.transformer.text_model.encoder.layers.{layer}.{suffix}.weight"
            if k_g in state_keys:
                clip_g_seen = True
                if clip_l_seen:
                    out[f"lora_te2_text_model_encoder_layers_{layer}_{mapped}"] = k_g
                    out[f"text_encoder_2.text_model.encoder.layers.{layer}.{suffix}"] = k_g
                else:
                    out[f"lora_te_text_model_encoder_layers_{layer}_{mapped}"] = k_g
                out[f"lora_prior_te_text_model_encoder_layers_{layer}_{mapped}"] = k_g
            k_h = f"clip_h.transformer.text_model.encoder.layers.{layer}.{suffix}.weight"
            if k_h in state_keys:
                out[f"lora_te1_text_model_encoder_layers_{layer}_{mapped}"] = k_h
                out[f"text_encoder.text_model.encoder.layers.{layer}.{suffix}"] = k_h

    for key in state_keys:
        if key.endswith(".weight"):
            if key.startswith("t5xxl.transformer."):
                logical = key[len("t5xxl.transformer.") : -len(".weight")].replace(".", "_")
                index = 1 + int(clip_l_seen) + int(clip_g_seen)
                out[f"lora_te{index}_{logical}"] = key
                if clip_l_seen and index == 2:
                    out[f"lora_te{index+1}_{logical}"] = key
            if key.startswith("hydit_clip.transformer.bert."):
                logical = key[len("hydit_clip.transformer.bert.") : -len(".weight")].replace(".", "_")
                out[f"lora_te1_{logical}"] = key

    proj = "clip_g.transformer.text_projection.weight"
    if proj in state_keys:
        out["lora_te2_text_projection"] = proj
        out["lora_prior_te_text_projection"] = proj
    proj_l = "clip_l.transformer.text_projection.weight"
    if proj_l in state_keys:
        out["lora_te1_text_projection"] = proj_l

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

    unet_config = getattr(getattr(model, "model_config", None), "unet_config", None)
    if isinstance(unet_config, dict):
        diffusers_keys = unet_to_diffusers(unet_config)
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
