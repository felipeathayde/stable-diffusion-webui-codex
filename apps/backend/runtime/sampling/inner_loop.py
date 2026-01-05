"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Torch-bound sampling inner loop (kept separate so `apps.backend.runtime.sampling` stays import-light for API/UI imports).
Implements conditioning batching, CFG routing, and sampling lifecycle hooks (prepare/cleanup) for native samplers.

Symbols (top-level; keep in sync; no ghosts):
- `get_area_and_mult` (function): Computes per-conditioning spatial area crop + mask multiplier (supports `area`, `mask`, `strength`,
  and timestep gates) and returns the prepared slice for batching.
- `cond_equal_size` (function): Checks whether two compiled conditionings are size-compatible for batching.
- `can_concat_cond` (function): Checks whether two conditioning entries can be concatenated into the same UNet batch (area/control/patch compat).
- `cond_cat` (function): Concatenates a list of compiled conditioning dicts into a single dict with canonical keys (`c_crossattn`, `y`, `c_concat`).
- `compute_cond_mark` (function): Builds a cond/uncond mark tensor aligned to the sigma ladder (used for chunked batching/indexing).
- `compute_cond_indices` (function): Computes flat indices for conditional vs unconditional slices in a packed `(batch*sigmas)` tensor layout.
- `calc_cond_uncond_batch` (function): Runs batched UNet calls to compute conditional/unconditional predictions with area masks, memory-aware
  batching, and strict conditioning validation (no fallbacks).
- `sampling_function_inner` (function): Core CFG math and hook routing; handles distilled/turbo `uncond=None`, optional deep debug logs,
  and sampler pre/post cfg modifiers.
- `sampling_function` (function): Wrapper around `sampling_function_inner` for the denoiser interface; applies conditioning modifiers and
  control/image concat plumbing, returning denoised + (cond/uncond) predictions.
- `sampling_prepare` (function): Pre-sampling hook; activates ControlNet runtime, loads required models to GPU, and prepares smart-offload state.
- `sampling_cleanup` (function): Post-sampling hook; cleans up ControlNet, smart-offload state, unloads models, and triggers op cache cleanup.
"""

import os
import torch
import math
import collections
import logging

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
from apps.backend.runtime import utils
from apps.backend.infra.config.env_flags import env_flag, env_int


logger = logging.getLogger("backend.runtime.sampling")
from .condition import Condition, compile_conditions, compile_weighted_conditions
from apps.backend.infra.config.args import dynamic_args, args

_ZIMAGE_SAMPLING_DEBUG_COUNT = 0


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]

    if 'mask' in conds:
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert (mask.shape[1] == x_in.shape[2])
        assert (mask.shape[2] == x_in.shape[3])
        mask = mask[:, area[2]:area[0] + area[2], area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:, :, t:1 + t, :] *= ((1.0 / rr) * (t + 1))
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:, :, area[0] - 1 - t:area[0] - t, :] *= ((1.0 / rr) * (t + 1))
        if area[3] != 0:
            for t in range(rr):
                mult[:, :, :, t:1 + t] *= ((1.0 / rr) * (t + 1))
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:, :, :, area[1] - 1 - t:area[1] - t] *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get('control', None)

    patches = None
    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])

    cond_mark = []
    for cx in cond_or_uncond:
        cond_mark += [cx] * cond_or_uncond_size

    cond_mark = torch.Tensor(cond_mark).to(sigmas)
    return cond_mark


def compute_cond_indices(cond_or_uncond, sigmas):
    cl = int(sigmas.shape[0])

    cond_indices = []
    uncond_indices = []
    for i, cx in enumerate(cond_or_uncond):
        if cx == 0:
            cond_indices += list(range(i * cl, (i + 1) * cl))
        else:
            uncond_indices += list(range(i * cl, (i + 1) * cl))

    return cond_indices, uncond_indices


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        if memory_management.signal_empty_cache:
            memory_management.soft_empty_cache()

        free_memory = memory_management.get_free_memory(x_in.device)

        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)

        # Validate assembled conditioning before UNet call (no fallbacks)
        if 'c_crossattn' not in c or not isinstance(c['c_crossattn'], torch.Tensor) or c['c_crossattn'].ndim != 3:
            raise ValueError(
                f"Missing or invalid 'c_crossattn' for UNet: got type={type(c.get('c_crossattn'))} "
                f"shape={getattr(c.get('c_crossattn'), 'shape', None)} (expected 3D tensor BxSxC)."
            )
        needs_y = getattr(model, 'diffusion_model', None) is not None and getattr(model.diffusion_model, 'num_classes', None) is not None
        if needs_y:
            if 'y' not in c or not isinstance(c['y'], torch.Tensor) or c['y'].ndim != 2:
                raise ValueError(
                    "UNet requires ADM 'y' vector (2D tensor BxV) but it is missing or invalid. "
                    "Ensure SDXL pooled embedding is wired as 'vector' and compiled to 'y'."
                )

        # Align dtype/device for conditioning tensors
        target_dtype = getattr(model, 'computation_dtype', None) or input_x.dtype
        dev = input_x.device
        c['c_crossattn'] = c['c_crossattn'].to(dtype=target_dtype, device=dev)
        if 'y' in c and isinstance(c['y'], torch.Tensor):
            c['y'] = c['y'].to(device=dev)
        if 'c_concat' in c and isinstance(c['c_concat'], torch.Tensor):
            c['c_concat'] = c['c_concat'].to(device=dev)
        timestep_ = torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Control batch: size=%d cond=%d uncond=%d sigma_shape=%s",
                len(cond_or_uncond),
                sum(1 for flag in cond_or_uncond if flag == COND),
                sum(1 for flag in cond_or_uncond if flag != COND),
                tuple(timestep.shape),
            )

        c['transformer_options'] = transformer_options

        if control:
            control.set_transformer_options(transformer_options)
            control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
            try:
                c['control'] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
            except Exception as err:
                logger.error("ControlNet get_control failed: %s", err, exc_info=True)
                raise
            c['control_model'] = control

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False):
    edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    # Optional deep diagnostics for flow models (Z Image/Flux): log CFG routing and tensor norms.
    global _ZIMAGE_SAMPLING_DEBUG_COUNT
    debug_enabled = env_flag("CODEX_ZIMAGE_DEBUG") or env_flag("CODEX_ZIMAGE_DEBUG_SAMPLING_INNER")
    debug_limit = env_int("CODEX_ZIMAGE_DEBUG_SAMPLING_INNER_N", 3, min_value=0)
    if debug_enabled and _ZIMAGE_SAMPLING_DEBUG_COUNT < debug_limit:
        try:
            sigma0 = float(timestep.detach().view(-1)[0].item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        except Exception:
            sigma0 = float("nan")
        try:
            cond_norm = float(cond_pred.detach().float().norm().item()) if isinstance(cond_pred, torch.Tensor) else float("nan")
        except Exception:
            cond_norm = float("nan")
        try:
            uncond_norm = float(uncond_pred.detach().float().norm().item()) if isinstance(uncond_pred, torch.Tensor) else float("nan")
        except Exception:
            uncond_norm = float("nan")
        logger.info(
            "[zimage-debug] sampling_inner sigma=%.6g cond_scale=%.4g edit_strength=%.4g uncond_present=%s cond_norm=%.6g uncond_norm=%.6g",
            sigma0,
            float(cond_scale),
            float(edit_strength),
            uncond_ is not None,
            cond_norm,
            uncond_norm,
        )
        _ZIMAGE_SAMPLING_DEBUG_COUNT += 1

    # Distilled / turbo models may omit unconditional conditioning entirely.
    # In that case, skip CFG math and return the conditional prediction as-is.
    if uncond_ is None:
        cfg_result = cond_pred
    elif "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    if return_full:
        return cfg_result, cond_pred, uncond_pred

    return cfg_result


def sampling_function(self, denoiser_params, cond_scale, cond_composition):
    denoiser_patcher = self.inner_model.inner_model.codex_objects.denoiser
    model = denoiser_patcher.model
    control = getattr(denoiser_patcher, "controlnet_linked_list", None)
    extra_concat_condition = getattr(denoiser_patcher, "extra_concat_condition", None)
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = compile_conditions(denoiser_params.text_uncond)
    cond = compile_weighted_conditions(denoiser_params.text_cond, cond_composition)
    model_options = denoiser_patcher.model_options
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            if uncond is not None:
                for i in range(len(uncond)):
                    uncond[i]['model_conds']['c_concat'] = Condition(image_cond_in)
            for i in range(len(cond)):
                cond[i]['model_conds']['c_concat'] = Condition(image_cond_in)

    if control:
        for h in cond:
            h['control'] = control
    if control and uncond is not None:
        for h in uncond:
            h['control'] = control

    for modifier in model_options.get('conditioning_modifiers', []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised, cond_pred, uncond_pred = sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options, seed, return_full=True)
    return denoised, cond_pred, uncond_pred


def sampling_prepare(denoiser, x):
    B, C, H, W = x.shape

    memory_estimation_function = denoiser.model_options.get('memory_peak_estimation_modifier', denoiser.memory_required)

    denoiser_inference_memory = memory_estimation_function([B * 2, C, H, W])
    additional_inference_memory = int(getattr(denoiser, "extra_preserved_memory_during_sampling", 0) or 0)
    additional_model_patchers = list(getattr(denoiser, "extra_model_patchers_during_sampling", []) or [])

    control_runtime = denoiser.activate_control() if hasattr(denoiser, "activate_control") else None
    if control_runtime:
        additional_inference_memory += control_runtime.inference_memory_requirements(denoiser.model_dtype())
        additional_model_patchers += control_runtime.get_models()
        logger.debug(
            "Control runtime activated: extra_memory=%s models=%d",
            additional_inference_memory,
            len(additional_model_patchers),
        )

    if denoiser.has_online_lora():
        lora_memory = utils.nested_compute_size(
            denoiser.lora_patches,
            element_size=utils.dtype_to_element_size(denoiser.model.computation_dtype),
        )
        additional_inference_memory += lora_memory

    models_to_load = [denoiser] + additional_model_patchers
    memory_management.load_models_gpu(
        models=models_to_load,
        memory_required=denoiser_inference_memory,
        hard_memory_preservation=additional_inference_memory
    )

    if smart_offload_enabled():
        setattr(denoiser, "_codex_smart_offload_models", models_to_load)
    else:
        setattr(denoiser, "_codex_smart_offload_models", [])

    if denoiser.has_online_lora():
        utils.nested_move_to_device(
            denoiser.lora_patches,
            device=denoiser.current_device,
            dtype=denoiser.model.computation_dtype,
        )

    real_model = denoiser.model

    percent_to_timestep_function = lambda p: real_model.predictor.percent_to_sigma(p)

    if control_runtime:
        control_runtime.prepare(real_model, percent_to_timestep_function)
        logger.debug("Control runtime prepared with model %s", type(real_model).__name__)

    return


def sampling_cleanup(denoiser):
    if denoiser.has_online_lora():
        utils.nested_move_to_device(denoiser.lora_patches, device=denoiser.offload_device)
    control_runtime = getattr(denoiser, "controlnet_linked_list", None)
    if control_runtime:
        control_runtime.cleanup()
        logger.debug("Control runtime cleaned up after sampling")
    if hasattr(denoiser, "clear_control"):
        denoiser.clear_control()
    if smart_offload_enabled():
        models_to_unload = getattr(denoiser, "_codex_smart_offload_models", [])
        for model in models_to_unload:
            memory_management.unload_model(model)
        setattr(denoiser, "_codex_smart_offload_models", [])
    from apps.backend.runtime.ops import cleanup_cache
    cleanup_cache()
    return
