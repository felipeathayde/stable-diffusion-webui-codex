"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ControlNet LoRA module that materialises a ControlNet model from LoRA weights.
Constructs a CLDM-style ControlNet module at runtime, normalizes storage dtype hints (GGUF → fp16 for construction), and patches weights before delegating execution to the SD `ControlNet`.

Symbols (top-level; keep in sync; no ghosts):
- `ControlLora` (class): LoRA-backed ControlNet module that materialises an inner ControlNet model during `pre_run`.
"""

from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.families.sd.cnets import cldm
from apps.backend.runtime.ops.operations import using_codex_operations

from ...base import ControlModuleBase
from ...ops.lora import ControlLoraOps
from .control import ControlNet


class ControlLora(ControlModuleBase):
    """ControlNet LoRA module that materialises a ControlNet on demand."""

    def __init__(self, control_weights, *, global_average_pooling: bool = False, device=None) -> None:
        super().__init__(device=device, global_average_pooling=global_average_pooling)
        self.control_weights = control_weights
        self.control_model = None
        self.manual_cast_dtype: Optional[torch.dtype] = None
        self.control_proxy: Optional[ControlNet] = None

    def pre_run(self, model, percent_to_timestep_function) -> None:
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.diffusion_model.config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]

        dtype = model.storage_dtype
        if dtype == "gguf":
            dtype = torch.float16

        controlnet_config["dtype"] = dtype
        self.manual_cast_dtype = model.computation_dtype

        with using_codex_operations(
            operations=ControlLoraOps,
            dtype=dtype,
            manual_cast_enabled=self.manual_cast_dtype != dtype,
        ):
            self.control_model = cldm.ControlNet(**controlnet_config)

        core_device = memory_management.manager.get_device(DeviceRole.CORE)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
        self.control_model.to(device=offload_device, dtype=dtype)
        diffusion_model = model.diffusion_model
        state_dict = diffusion_model.state_dict()

        for key, weight in state_dict.items():
            try:
                utils.set_attr(self.control_model, key, weight)
            except AttributeError:
                pass

        for key, weight in self.control_weights.items():
            if key == "lora_controlnet":
                continue
            utils.set_attr(
                self.control_model,
                key,
                weight.to(device=offload_device, dtype=dtype),
            )

        self.control_proxy = ControlNet(
            self.control_model,
            global_average_pooling=self.global_average_pooling,
            device=self.device,
            load_device=core_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )
        self.control_proxy.previous_control = self.previous_control
        self.control_proxy.weight_schedule = self.weight_schedule
        self.control_proxy.mask_config = self.mask_config
        self.control_proxy.pre_run(model, percent_to_timestep_function)

    def get_control(self, x_noisy, t, cond, batched_number):
        if self.control_model is None or self.control_proxy is None:
            raise RuntimeError("ControlLora requires pre_run to materialise the control model")

        proxy = self.control_proxy
        proxy.transformer_options = self.transformer_options
        proxy.previous_control = self.previous_control
        proxy.weight_schedule = self.weight_schedule
        proxy.mask_config = self.mask_config
        proxy.cond_hint_original = self.cond_hint_original
        proxy.strength = self.strength
        proxy.timestep_percent_range = self.timestep_percent_range
        proxy.cond_hint = None
        return proxy.get_control(x_noisy, t, cond, batched_number)

    def cleanup(self) -> None:
        if self.control_proxy is not None:
            self.control_proxy.cleanup()
            self.control_proxy = None
        if self.control_model is not None:
            del self.control_model
            self.control_model = None
        super().cleanup()

    def get_models(self) -> list[object]:
        if self.control_proxy is not None:
            return self.control_proxy.get_models()
        return super().get_models()

    def copy(self):
        clone = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling, device=self.device)
        self._copy_runtime_state_to(clone)
        return clone

    def _clone_impl(self):
        return ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling, device=self.device)

    def inference_memory_requirements(self, dtype):
        return (
            utils.calculate_parameters(self.control_weights) * torch.empty((), dtype=dtype).element_size()
            + super().inference_memory_requirements(dtype)
        )
