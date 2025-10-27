"""
Vendored from https://github.com/lllyasviel/huggingface_guess @ 70942022b6bcd17d941c1b4172804175758618e2
"""
import math
import logging
import torch

from huggingface_guess import utils, model_list
from typing import Dict, List, Any


def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        time_stack = '{}1.time_stack.0.attn1.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn1.to_q.weight'.format(prefix) in state_dict
        time_stack_cross = '{}1.time_stack.0.attn2.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn2.to_q.weight'.format(prefix) in state_dict
        return last_transformer_depth, context_dim, use_linear_in_transformer, time_stack, time_stack_cross
    return None


def detect_unet_config(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())

    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys:  # mmdit model
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        patch_size = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (patch_size * patch_size)

        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        unet_config["input_size"] = None
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        context_key = '{}context_embedder.weight'.format(key_prefix)
        if context_key in state_dict_keys:
            in_features = state_dict[context_key].shape[1]
            out_features = state_dict[context_key].shape[0]
            unet_config["context_embedder_config"] = {"target": "torch.nn.Linear", "params": {"in_features": in_features, "out_features": out_features}}

        # print('[CFG] mmdit unet_config detected: \n{}'.format(json.dumps(unet_config, indent=4)))
        return unet_config

    if ('{}input_blocks.0.0.weight'.format(key_prefix) in state_dict_keys) and ('{}context_embedder.parameters.0.weight'.format(key_prefix) in state_dict_keys or '{}context_embedder.*.0.weight'.format(key_prefix) in state_dict_keys):
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]
        unet_config["out_channels"] = state_dict['{}out.2.weight'.format(key_prefix)].shape[0]
        return unet_config

    if '{}x_embedder.proj.weight'.format(key_prefix) in state_dict_keys:  # stable cascade model
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        sp_size = list(state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape)
        unet_config["patch_size"] = sp_size[2]
        unet_config["depth"] = sp_size[0] // 64
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (sp_size[2] * sp_size[2])

        if '{}depth_database'.format(key_prefix) in state_dict:
            unet_config["depth_database"] = True
        else:
            unet_config["depth_database"] = False

        unet_config["input_size"] = None
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        # print('[CFG] stable cascade unet_config detected: \n{}'.format(json.dumps(unet_config, indent=4)))
        return unet_config

    if '{}control'.format(key_prefix) in state_dict_keys:  # controlnet model
        unet_config = {}
        unet_config["controlnet"] = True
        # print('[CFG] controlnet unet_config detected: \n{}'.format(json.dumps(unet_config, indent=4)))
        return unet_config

    unet_config = {}
    unet_config["in_channels"] = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]
    unet_config["out_channels"] = state_dict['{}out.2.weight'.format(key_prefix)].shape[0]

    return unet_config


def detect_optional_config(state_dict, state_dict_keys, key_prefix):
    unet_extra_config = {}
    use_linear_in_transformer = False
    context_embedder_config = None
    adm_in_channels = None
    num_heads = -1
    num_head_channels = None
    learned_sigma = None
    sigma_min = None
    sigma_max = None
    sigma_data = None
    sigma_dist = 'v'  # 'log_normal' or 'v'
    sample_omp_num = None
    sampling_res_error = []
    v_slices = None
    v_projector = None
    residual_16x = None
    residual_8x = None
    residual_4x = None
    image_model_updown_feature = None
    image_model_concat_feature = None
    image_model_output_image = None
    image_model_crop_out = None
    rgb_embedder_config = None
    ae_scale = None
    ae_latent_weight = None
    cfg_var = None

    conv_in = '{}conv_in.weight'.format(key_prefix)
    if conv_in in state_dict:
        unet_extra_config['input_size'] = state_dict[conv_in].shape[2]
    else:
        conv_in = '{}x_embedder.proj.weight'.format(key_prefix)
        if conv_in in state_dict:
            ps = state_dict[conv_in].shape[2]
            unet_extra_config['input_size'] = 2048 // (2 ** (int(math.log2(ps))) + 1) * ps
        else:
            unet_extra_config['input_size'] = 1024

    # Validate with state_dict later
    # detect the number of transformer blocks
    # use linear in transformer
    # vadim: I think that when use linear, 'y_embedder does not have consecutive blocks
    try:
        context_embedder_config = (state_dict['{}context_embedder.net.0.weight'.format(key_prefix)].shape[1], state_dict['{}context_embedder.net.0.weight'.format(key_prefix)].shape[0])
    except KeyError:
        try:
            context_embedder_config = (state_dict['{}context_embedder.mlp.0.weight'.format(key_prefix)].shape[1], state_dict['{}context_embedder.mlp.0.weight'.format(key_prefix)].shape[0])
        except KeyError:
            try:
                context_embedder_config = (state_dict['{}context_embedder.weight'.format(key_prefix)].shape[1], state_dict['{}context_embedder.weight'.format(key_prefix)].shape[0])
            except KeyError:
                pass

    try:
        adm_in_channels = state_dict['{}y_embedder.mlp.0.weight'.format(key_prefix)].shape[1]
    except KeyError:
        adm_in_channels = None

    ov = calculate_transformer_depth('{}output_blocks.'.format(key_prefix), state_dict_keys, state_dict)
    mv = calculate_transformer_depth('{}middle_block.'.format(key_prefix), state_dict_keys, state_dict)
    iv = calculate_transformer_depth('{}input_blocks.'.format(key_prefix), state_dict_keys, state_dict)
    n_transf = (ov[0] if ov is not None else 0) + (mv[0] if mv is not None else 0) + (iv[0] if iv is not None else 0)
    unet_extra_config['n_transformers'] = n_transf

    if ov and mv and iv:
        context_dim = ov[1]
        use_linear_in_transformer = ov[2]
    else:
        context_dim = None
        use_linear_in_transformer = False

    # Fallback: scan globally for cross-attn projections to recover context_dim
    if context_dim is None:
        try:
            for k in state_dict_keys:
                # Robust pattern for attn2.to_k.weight present in SD1.x/2.x/XL
                if k.endswith('attn2.to_k.weight') and hasattr(state_dict[k], 'shape'):
                    context_dim = int(state_dict[k].shape[1])
                    break
        except Exception:
            pass

    # Fallback: detect linear proj usage by presence of any proj_in.weight with 2D shape
    try:
        if not use_linear_in_transformer:
            for k in state_dict_keys:
                if k.endswith('proj_in.weight') and hasattr(state_dict[k], 'shape'):
                    if len(state_dict[k].shape) == 2:
                        use_linear_in_transformer = True
                        break
    except Exception:
        pass

    unet_extra_config["num_heads"] = num_heads
    unet_extra_config["num_head_channels"] = num_head_channels
    unet_extra_config['transformer_context_dim'] = context_dim
    # Provide canonical key expected by matchers/heuristics
    if context_dim is not None:
        unet_extra_config['context_dim'] = context_dim
    unet_extra_config['use_linear_in_transformer'] = use_linear_in_transformer

    unet_extra_config['learned_sigma'] = False

    if 'alpha' in state_dict:
        unet_extra_config['denoise_strength'] = float(state_dict['alpha'])
    else:
        unet_extra_config['denoise_strength'] = -1

    # Infer model_channels from first conv_in if available
    try:
        convw = state_dict['{}conv_in.weight'.format(key_prefix)]
        if hasattr(convw, 'shape'):
            unet_extra_config['model_channels'] = int(convw.shape[0])
    except Exception:
        pass

    return unet_extra_config


# TODO: support openclip
def detect_model_config(state_dict, unet_config, unet_extra_config, key_prefix):
    state_dict_keys = list(state_dict.keys())
    conf = None
    conf_list = model_list.registered_models

    # Merge a few optional hints into the matching dict (non-destructive)
    merged_for_match = dict(unet_config or {})
    for k in ("use_linear_in_transformer", "context_dim", "adm_in_channels"):
        if k not in merged_for_match and k in (unet_extra_config or {}):
            merged_for_match[k] = unet_extra_config[k]

    # Prefer specific models with non-empty class-level configs; skip generic catch-alls
    for c in conf_list:
        try:
            class_cfg = getattr(c, 'unet_config', {}) or {}
            if not isinstance(class_cfg, dict) or len(class_cfg) == 0:
                continue
        except Exception:
            continue
        if c.matches(merged_for_match, state_dict):
            conf = c
            break

    # Heuristic fallback by context_dim and transformer usage
    if conf is None:
        ctx = merged_for_match.get('context_dim', None)
        adm = merged_for_match.get('adm_in_channels', None)
        use_lin = bool(merged_for_match.get('use_linear_in_transformer', False))
        try:
            if ctx == 2048 and use_lin:
                conf = model_list.SDXL
            elif ctx == 1280 and (adm == 2560 or adm is not None):
                conf = model_list.SDXLRefiner
            elif ctx == 1024 and use_lin:
                conf = model_list.SD20
            elif ctx == 768 and not use_lin:
                conf = model_list.SD15
            else:
                conf = model_list.BASE
        except Exception:
            conf = model_list.BASE

    # print('[CFG] model_config detected: {}'.format(conf.__name__))
    try:
        model_type = conf.model_type(conf, state_dict, prefix=key_prefix)
    except Exception:
        logging.warning('cannot detect model type for {}. Use EPS as default'.format(conf.__name__))
        model_type = model_list.ModelType.EPS

    return conf, model_type


def unet_prefix_from_state_dict(state_dict):
    potential_keys = ['model.diffusion_model.', 'model.model.', 'unet.']
    state_dict_keys = list(state_dict.keys())
    for key in potential_keys:
        for k in state_dict_keys:
            if k.startswith(key):
                return key
    return ''


def model_config_from_unet(state_dict, key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, key_prefix)
    unet_extra_config = detect_optional_config(state_dict, list(state_dict.keys()), key_prefix)
    model_config, model_type = detect_model_config(state_dict, unet_config, unet_extra_config, key_prefix)

    # Upstream classes expect the UNet config in the constructor.
    # Our initial vendored version mistakenly called the class with no args,
    # which raises: BASE.__init__() missing 1 required positional argument: 'unet_config'.
    inferred_config = model_config(unet_config)
    # Preserve the callable model_type method; store detected enum separately
    try:
        setattr(inferred_config, 'detected_model_type', model_type)
    except Exception:
        pass
    inferred_config.unet_config = unet_config
    inferred_config.unet_extra_config = unet_extra_config

    try:
        dbg_ctx = inferred_config.unet_extra_config.get('context_dim') if isinstance(inferred_config.unet_extra_config, dict) else None
        dbg_lin = inferred_config.unet_extra_config.get('use_linear_in_transformer') if isinstance(inferred_config.unet_extra_config, dict) else None
        dbg_channels = inferred_config.unet_extra_config.get('model_channels') if isinstance(inferred_config.unet_extra_config, dict) else None
        print(f"[hg-guess] class={getattr(model_config,'__name__','?')} ctx={dbg_ctx} linear={dbg_lin} channels={dbg_channels}")
    except Exception:
        pass
    return inferred_config


def model_config_from_diffusers_unet(unet):
    raise NotImplementedError('this function is not implemented yet')


def _int_keys_under(prefix: str, keys: List[str]) -> List[int]:
    out = set()
    p = prefix
    n = len(p)
    for k in keys:
        if k.startswith(p):
            rest = k[n:]
            try:
                idx = int(rest.split('.', 1)[0])
                out.add(idx)
            except Exception:
                pass
    return sorted(out)


def unet_config_from_diffusers_unet(controlnet_sd: Dict[str, Any], dtype=None) -> Dict[str, Any]:
    """
    Compatibility helper for Forge ControlNet loaders that expect a function
    named `unet_config_from_diffusers_unet` in huggingface_guess.

    Given a diffusers-style ControlNet state_dict (keys like
    `down_blocks.0.resnets.0.*`, `down_blocks.0.attentions.0.*`,
    `mid_block.*`, `controlnet_cond_embedding.*`), infer a minimal
    UNet config dictionary sufficient for:
      1) Generating a Diffusers→Forge key map via utils.unet_to_diffusers
      2) Instantiating backend.nn.cnets.cldm.ControlNet(**config)

    Notes:
    - We purposely keep this inference conservative and data-driven.
    - Fields not critical for ControlNet or mapping are omitted.
    - `channel_mult` values aren’t used by the mapper logic; only length
      matters. We therefore set it to `[1] * num_blocks`.
    - `transformer_depth_output` is only used for up blocks mapping; ControlNet
      weights typically don’t include up blocks. We provide a zero-list with the
      required length to satisfy the mapper.
    """
    keys = list(controlnet_sd.keys())

    # Detect number of down blocks
    down_block_ids = _int_keys_under('down_blocks.', keys)
    num_blocks = (max(down_block_ids) + 1) if down_block_ids else 4

    # Count resnets per down block
    num_res_blocks: List[int] = []
    for x in range(num_blocks):
        res_ids = _int_keys_under(f'down_blocks.{x}.resnets.', keys)
        if not res_ids:
            # Fallback to 2 if not directly discoverable
            num_res_blocks.append(2)
        else:
            num_res_blocks.append(max(res_ids) + 1)

    # Build transformer depth list (per-resnet in down path)
    transformer_depth: List[int] = []
    for x in range(num_blocks):
        res_ids = list(range(num_res_blocks[x]))
        for i in res_ids:
            t_ids = _int_keys_under(f'down_blocks.{x}.attentions.{i}.transformer_blocks.', keys)
            transformer_depth.append((max(t_ids) + 1) if t_ids else 0)

    # Mid block transformer depth
    mid_t_ids = _int_keys_under('mid_block.attentions.0.transformer_blocks.', keys)
    transformer_depth_middle = (max(mid_t_ids) + 1) if mid_t_ids else 0

    # transformer_depth_output length must match mapper consumption
    tdo_len = sum((n + 1) for n in num_res_blocks)
    transformer_depth_output = [0] * tdo_len

    # Infer model_channels from the very first down resnet conv
    model_channels = None
    try:
        w = controlnet_sd['down_blocks.0.resnets.0.in_layers.2.weight']
        model_channels = int(w.shape[0])
    except Exception:
        # Conservative fallback used by SD1.5/SDXL-style UNets
        model_channels = 320

    # Heuristic for head dim: prefer divisors of model_channels
    def _pick_head_dim(ch: int) -> int:
        candidates = [128, 96, 64, 80, 160, 192, 256, 32]
        for c in candidates:
            if ch % c == 0:
                return c
        return 64

    num_head_channels = _pick_head_dim(model_channels)

    # Context dim: read from a cross-attn projection if present
    context_dim = None
    try:
        k_w = controlnet_sd['down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight']
        context_dim = int(k_w.shape[1])
    except Exception:
        # Common defaults: 768 (SD1.x), 1024 (SDXL)
        context_dim = 1024 if model_channels >= 320 else 768

    use_spatial_transformer = any(d > 0 for d in transformer_depth) or transformer_depth_middle > 0

    # Build minimal config
    unet_config = {
        'in_channels': 4,  # latent channels
        'model_channels': model_channels,
        'hint_channels': 1,  # placeholder; caller will overwrite with actual hint channels
        'num_res_blocks': num_res_blocks,
        'channel_mult': [1] * num_blocks,
        'dims': 2,
        'use_spatial_transformer': use_spatial_transformer,
        'transformer_depth': transformer_depth,
        'transformer_depth_output': transformer_depth_output,
        'transformer_depth_middle': transformer_depth_middle,
        'context_dim': context_dim if use_spatial_transformer else None,
        'num_heads': -1,
        'num_head_channels': num_head_channels,
        'dtype': dtype,
    }

    return unet_config
