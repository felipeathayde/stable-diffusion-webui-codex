from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from apps.backend.runtime.misc import adaptive_resize
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.patchers.base import ModelPatcher
from ...base import ControlModuleBase
from ...weighting import broadcast_image_to


class ControlNet(ControlModuleBase):
    """ControlNet module compatible with Codex UNet patching."""

    def __init__(
        self,
        control_model,
        *,
        global_average_pooling: bool = False,
        device: Optional[torch.device] = None,
        load_device: Optional[torch.device] = None,
        manual_cast_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(device=device, global_average_pooling=global_average_pooling)

        self.control_model = control_model
        self.load_device = load_device or memory_management.manager.get_device(DeviceRole.CORE)
        self.manual_cast_dtype = manual_cast_dtype

        self.control_model_wrapped = ModelPatcher(
            self.control_model,
            load_device=self.load_device,
            offload_device=memory_management.manager.get_offload_device(DeviceRole.CORE),
        )

        self.model_sampling_current = None

    # ------------------------------------------------------------------ #
    # Runtime
    # ------------------------------------------------------------------ #

    def get_control(self, x_noisy, t, cond, batched_number):
        transformer_options = self.transformer_options

        for modifier in transformer_options.get("controlnet_conditioning_modifiers", []):
            x_noisy, t, cond, batched_number = modifier(self, x_noisy, t, cond, batched_number)

        control_prev = None
        if self.previous_control is not None:
            control_prev = self.previous_control.get_control(x_noisy, t, cond, batched_number)

        if self._should_skip_timestep(t):
            return control_prev

        if self.model_sampling_current is None:
            raise RuntimeError("ControlNet requires pre_run to be executed before get_control")

        dtype = getattr(self.control_model, "dtype", None)
        if dtype is None:
            dtype = next(self.control_model.parameters()).dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or self._hint_mismatch(x_noisy):
            self.cond_hint = self._resize_hint(x_noisy, dtype)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond["c_crossattn"]
        y = cond.get("y", None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy_transformed = self.model_sampling_current.calculate_input(t, x_noisy)

        model_wrapper = transformer_options.get("controlnet_model_function_wrapper")
        if model_wrapper is not None:
            control = model_wrapper(
                model=self,
                inner_model=self.control_model,
                x=x_noisy_transformed.to(dtype),
                hint=self.cond_hint.to(dtype),
                timesteps=timestep.float(),
                context=context.to(dtype),
                y=y,
            )
        else:
            control = self.control_model(
                x=x_noisy_transformed.to(dtype),
                hint=self.cond_hint.to(self.device),
                timesteps=timestep.float(),
                context=context.to(dtype),
                y=y,
            )

        return self.merge_control_outputs(
            control_input=None,
            control_output=control,
            control_prev=control_prev,
            output_dtype=output_dtype,
        )

    def pre_run(self, model, percent_to_timestep_function) -> None:
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.predictor

    def cleanup(self) -> None:
        self.model_sampling_current = None
        super().cleanup()

    def get_models(self) -> list[object]:
        models = super().get_models()
        models.append(self.control_model_wrapped)
        return models

    def copy(self):
        clone = ControlNet(
            self.control_model,
            global_average_pooling=self.global_average_pooling,
            device=self.device,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )
        self._copy_runtime_state_to(clone)
        return clone

    def _clone_impl(self):
        return ControlNet(
            self.control_model,
            global_average_pooling=self.global_average_pooling,
            device=self.device,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _hint_mismatch(self, x_noisy: torch.Tensor) -> bool:
        if self.cond_hint is None:
            return True
        return (
            x_noisy.shape[2] * 8 != self.cond_hint.shape[2]
            or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]
        )

    def _resize_hint(self, x_noisy: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.cond_hint is not None:
            del self.cond_hint
        self.cond_hint = None
        resized = adaptive_resize(
            self.cond_hint_original,
            x_noisy.shape[3] * 8,
            x_noisy.shape[2] * 8,
            "nearest-exact",
            "center",
        ).to(dtype)
        return resized
