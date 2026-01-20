"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: T2I-Adapter ControlNet module and loader.
Provides the `T2IAdapter` patcher module plus helpers to detect/load adapter state dicts into a runnable SD-family model.

Symbols (top-level; keep in sync; no ghosts):
- `T2IAdapter` (class): Adapter-based control module projecting hints via dedicated adapter networks.
- `load_t2i_adapter` (function): Loads a T2I-Adapter state dict into a `T2IAdapter` instance (or returns `None`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from apps.backend.runtime.models.state_dict import state_dict_prefix_replace
from apps.backend.runtime.families.sd.cnets import t2i_adapter
from apps.backend.runtime.misc.image_resize import adaptive_resize

from ...base import ControlModuleBase
from ...weighting import broadcast_image_to


class T2IAdapter(ControlModuleBase):
    """Adapter-based control module that projects hints via dedicated adapter networks."""

    def __init__(self, t2i_model, channels_in: int, *, device=None) -> None:
        super().__init__(device=device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def scale_image_to(self, width: int, height: int) -> tuple[int, int]:
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        transformer_options = self.transformer_options

        for modifier in transformer_options.get("controlnet_conditioning_modifiers", []):
            x_noisy, t, cond, batched_number = modifier(self, x_noisy, t, cond, batched_number)

        control_prev = None
        if self.previous_control is not None:
            control_prev = self.previous_control.get_control(x_noisy, t, cond, batched_number)

        if self._should_skip_timestep(t):
            return control_prev

        if self.cond_hint is None or self._hint_mismatch(x_noisy):
            width, height = self.scale_image_to(x_noisy.shape[3] * 8, x_noisy.shape[2] * 8)
            self.cond_hint = self._resize_hint(width, height)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)

            wrapper = transformer_options.get("controlnet_model_function_wrapper")
            if wrapper is not None:
                self.control_input = wrapper(
                    model=self,
                    inner_model=self.t2i_model,
                    inner_t2i_model=self.t2i_model,
                    hint=self.cond_hint.to(x_noisy.dtype),
                )
            else:
                self.control_input = self.t2i_model(self.cond_hint.to(x_noisy))

            self.t2i_model.cpu()

        control_input = [tensor.clone() if tensor is not None else None for tensor in self.control_input]
        middle = None
        if getattr(self.t2i_model, "xl", False):
            middle = control_input[-1:]
            control_input = control_input[:-1]

        return self.merge_control_outputs(
            control_input=control_input,
            control_output=middle,
            control_prev=control_prev,
            output_dtype=x_noisy.dtype,
        )

    def cleanup(self) -> None:
        self.control_input = None
        super().cleanup()

    def copy(self):
        clone = T2IAdapter(self.t2i_model, self.channels_in, device=self.device)
        self._copy_runtime_state_to(clone)
        return clone

    def _clone_impl(self):
        return T2IAdapter(self.t2i_model, self.channels_in, device=self.device)

    def _hint_mismatch(self, x_noisy: torch.Tensor) -> bool:
        if self.cond_hint is None:
            return True
        return (
            x_noisy.shape[2] * 8 != self.cond_hint.shape[2]
            or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]
        )

    def _resize_hint(self, width: int, height: int) -> torch.Tensor:
        if self.cond_hint is not None:
            del self.cond_hint
        hint = adaptive_resize(self.cond_hint_original, width, height, "nearest-exact", "center").float()
        if self.channels_in == 1 and hint.shape[1] > 1:
            hint = torch.mean(hint, 1, keepdim=True)
        return hint


def load_t2i_adapter(t2i_data) -> Optional[T2IAdapter]:
    if "adapter" in t2i_data:
        t2i_data = t2i_data["adapter"]
    if "adapter.body.0.resnets.0.block1.weight" in t2i_data:  # diffusers format
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace[f"adapter.body.{i}.resnets.{j}."] = f"body.{i * 2 + j}."
            prefix_replace[f"adapter.body.{i}."] = f"body.{i * 2}."
        prefix_replace["adapter."] = ""
        t2i_data = state_dict_prefix_replace(t2i_data, prefix_replace)

    keys = t2i_data.keys()

    if "body.0.in_conv.weight" in keys:
        cin = t2i_data["body.0.in_conv.weight"].shape[1]
        model_adapter = t2i_adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif "conv_in.weight" in keys:
        cin = t2i_data["conv_in.weight"].shape[1]
        channel = t2i_data["conv_in.weight"].shape[0]
        ksize = t2i_data["body.0.block2.weight"].shape[2]
        use_conv = any(key.endswith("down_opt.op.weight") for key in keys)
        xl = cin in {256, 768}
        model_adapter = t2i_adapter.Adapter(
            cin=cin,
            channels=[channel, channel * 2, channel * 4, channel * 4][:4],
            nums_rb=2,
            ksize=ksize,
            sk=True,
            use_conv=use_conv,
            xl=xl,
        )
    else:
        return None

    missing, unexpected = model_adapter.load_state_dict(t2i_data)
    if missing:
        print("t2i missing", missing)
    if unexpected:
        print("t2i unexpected", unexpected)

    return T2IAdapter(model_adapter, model_adapter.input_channels)
