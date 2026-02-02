"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR UNet variant (LightGLVUNet).
Consumes control tensors produced by `GLVControl` and produces the noise prediction for sampling.

This module is intentionally structured to remain weight-compatible with common SUPIR checkpoints.

Symbols (top-level; keep in sync; no ghosts):
- `LightGLVUNet` (class): SUPIR UNet variant.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import torch
from torch import nn

from apps.backend.runtime.common.nn.unet.layers import SpatialTransformer, TimestepBlock, Upsample
from apps.backend.runtime.common.nn.unet.model import UNet2DConditionModel
from apps.backend.runtime.common.nn.unet.utils import timestep_embedding

from .zero import ZeroCrossAttn, ZeroSFT


class LightGLVUNet(UNet2DConditionModel):
    def __init__(
        self,
        *,
        mode: str,
        project_type: str = "ZeroSFT",
        project_channel_scale: float = 1.0,
        **unet_kwargs: Any,
    ):
        super().__init__(**unet_kwargs)

        if mode == "XL-base":
            cond_output_channels = [320] * 4 + [640] * 3 + [1280] * 3
            project_channels = [160] * 4 + [320] * 3 + [640] * 3
            concat_channels = [320] * 2 + [640] * 3 + [1280] * 4 + [0]
            cross_attn_insert_idx = [6, 3]
            self.progressive_mask_nums = [0, 3, 7, 11]
        elif mode == "XL-refine":
            # SUPIR supports SDXL base, not refiner; keep mode for completeness and fail-loud use-cases.
            cond_output_channels = [384] * 4 + [768] * 3 + [1536] * 6
            project_channels = [192] * 4 + [384] * 3 + [768] * 6
            concat_channels = [384] * 2 + [768] * 3 + [1536] * 7 + [0]
            cross_attn_insert_idx = [9, 6, 3]
            self.progressive_mask_nums = [0, 3, 6, 10, 14]
        else:
            raise ValueError(f"LightGLVUNet: unsupported mode={mode!r}")

        scale = float(project_channel_scale)
        project_channels = [int(c * scale) for c in project_channels]

        self.project_modules = nn.ModuleList()
        for i in range(len(cond_output_channels)):
            if project_type == "ZeroSFT":
                self.project_modules.append(
                    ZeroSFT(project_channels[i], cond_output_channels[i], concat_channels=concat_channels[i])
                )
            elif project_type == "ZeroCrossAttn":
                self.project_modules.append(ZeroCrossAttn(cond_output_channels[i], project_channels[i]))
            else:
                raise ValueError(f"LightGLVUNet: unsupported project_type={project_type!r}")

        # Insert extra cross-attention adapters at fixed indices (weight-compat behaviour).
        for idx in cross_attn_insert_idx:
            self.project_modules.insert(idx, ZeroCrossAttn(cond_output_channels[idx], concat_channels[idx]))

    def step_progressive_mask(self) -> None:
        if not getattr(self, "progressive_mask_nums", None):
            return
        mask_num = self.progressive_mask_nums.pop()
        for i in range(len(self.project_modules)):
            self.project_modules[i].mask = bool(i < mask_num)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        control: Sequence[torch.Tensor] | None = None,
        *,
        control_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("LightGLVUNet.forward requires context")
        if timesteps is None:
            raise ValueError("LightGLVUNet.forward requires timesteps")
        if control is None or not isinstance(control, Sequence) or not control:
            raise ValueError("LightGLVUNet.forward requires non-empty control sequence")
        if (y is not None) != (self.num_classes is not None):
            raise ValueError("LightGLVUNet.forward: y must be provided iff num_classes is not None")

        # Match upstream: cast x/context/y to the control dtype.
        dtype = control[0].dtype
        x = x.to(dtype)
        context = context.to(dtype)
        if y is not None:
            y = y.to(dtype)

        hs: List[torch.Tensor] = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y is not None
            emb = emb + self.label_emb(y)

        # Encoder pass (store skip activations).
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context, transformer_options=None)
            hs.append(h)

        adapter_idx = len(self.project_modules) - 1
        control_idx = len(control) - 1

        # Middle + adapter
        h = self.middle_block(h, emb, context, transformer_options=None)
        h = self.project_modules[adapter_idx](control[control_idx], h, control_scale=control_scale)
        adapter_idx -= 1
        control_idx -= 1

        # Decoder with control fusion.
        for module in self.output_blocks:
            if not hs:
                raise RuntimeError("LightGLVUNet: skip activations underflow (control/project mismatch)")
            h_skip = hs.pop()
            h = self.project_modules[adapter_idx](control[control_idx], h_skip, h, control_scale=control_scale)
            adapter_idx -= 1

            output_shape = hs[-1].shape if hs else None

            if len(module) == 3 and isinstance(module[2], Upsample):
                # Apply first layers manually so we can inject a cross-attn adapter before upsampling.
                for layer in module[:2]:
                    if isinstance(layer, TimestepBlock):
                        h = layer(h, emb, transformer_options=None)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context, transformer_options=None)
                    else:
                        h = layer(h)
                h = self.project_modules[adapter_idx](control[control_idx], h, control_scale=control_scale)
                adapter_idx -= 1
                h = module[2](h, output_shape=output_shape)
            else:
                h = module(h, emb, context, transformer_options=None, output_shape=output_shape)

            control_idx -= 1

        return self.out(h)


__all__ = ["LightGLVUNet"]

