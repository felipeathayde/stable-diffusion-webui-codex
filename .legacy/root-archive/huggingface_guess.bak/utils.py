"""
Vendored from https://github.com/lllyasviel/huggingface_guess @ 70942022b6bcd17d941c1b4172804175758618e2

Note: This vendored module adds a small compatibility shim `unet_to_diffusers`
to interoperate with Forge loaders. The implementation delegates to the
project's internal mapper in `backend.misc.diffusers_state_dict` which expects
an UNet config dictionary shaped like those produced by huggingface_guess.
"""
import torch
import math
import struct
from typing import Dict, Any

try:
    # Prefer the project’s canonical mapper
    from backend.misc.diffusers_state_dict import unet_to_diffusers as _unet_to_diffusers  # type: ignore
except Exception:
    _unet_to_diffusers = None


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            params += w.nelement()
    return params


def weight_dtype(sd, prefix=""):
    dtypes = {}
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            dtypes[w.dtype] = dtypes.get(w.dtype, 0) + 1

    return max(dtypes, key=dtypes.get)


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    sd["{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)] = weights[x * shape_from:(x + 1) * shape_from, :]


def unet_to_diffusers(unet_config: Dict[str, Any]):
    """
    Return a mapping dict from diffusers UNet keys to Forge UNet keys.

    This shim delegates to the internal mapper used across the project to keep
    key translations consistent. The mapper expects a config dict containing at
    least:
      - num_res_blocks: List[int]
      - channel_mult: List[int] (length = number of blocks)
      - transformer_depth: List[int]
      - transformer_depth_output: List[int]
      - transformer_depth_middle: int
    """
    if _unet_to_diffusers is None:
        # Fall back to an empty mapping; callers typically add ControlNet own
        # mappings afterwards. This avoids hard crashes on environments that
        # cannot import the internal mapper, but keeps behavior explicit.
        return {}
    return _unet_to_diffusers(unet_config)
