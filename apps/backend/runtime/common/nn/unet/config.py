"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: `UNetConfig` dataclass and helpers for normalizing scalar/sequence configuration values.

Symbols (top-level; keep in sync; no ghosts):
- `_ensure_tuple` (function): Internal helper to normalize scalar/sequence values into tuples.
- `UNetConfig` (dataclass): Typed configuration for `UNet2DConditionModel` (includes helper expansion methods).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


def _ensure_tuple(value, *, default: int | None = None) -> tuple:
    if value is None:
        return () if default is None else (default,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: Sequence[int] | int
    dropout: float = 0.0
    channel_mult: Sequence[int] = field(default_factory=lambda: (1, 2, 4, 8))
    conv_resample: bool = True
    dims: int = 2
    num_classes: int | str | None = None
    use_checkpoint: bool = False
    num_heads: int = -1
    num_head_channels: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_spatial_transformer: bool = False
    transformer_depth: Sequence[int] | int = 1
    context_dim: Sequence[int] | int | None = None
    disable_self_attentions: Sequence[bool] | None = None
    num_attention_blocks: Sequence[int] | None = None
    disable_middle_self_attn: bool = False
    use_linear_in_transformer: bool = False
    adm_in_channels: int | None = None
    transformer_depth_middle: int | None = None
    transformer_depth_output: Sequence[int] | None = None

    def expanded_num_res_blocks(self) -> tuple[int, ...]:
        channel_mult = tuple(self.channel_mult)
        blocks = _ensure_tuple(self.num_res_blocks)
        if len(blocks) == 1:
            blocks = tuple(blocks[0] for _ in channel_mult)
        return blocks

    def transformer_depth_list(self, total_blocks: int) -> list[int]:
        values = _ensure_tuple(self.transformer_depth, default=0)
        if len(values) == 1 and total_blocks > 1:
            values = tuple(values[0] for _ in range(total_blocks))
        return list(values)

    def transformer_depth_output_list(self, total_blocks: int) -> list[int]:
        values = _ensure_tuple(self.transformer_depth_output, default=0)
        if not values:
            values = (0,)
        if len(values) == 1 and total_blocks > 1:
            values = tuple(values[0] for _ in range(total_blocks))
        return list(values)
