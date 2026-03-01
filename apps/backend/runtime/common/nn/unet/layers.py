"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Core UNet building blocks (timestep-aware residual blocks, spatial transformer blocks, attention, and up/down sampling layers).
Defines the nn.Module layers used by the Codex-native `UNet2DConditionModel` implementation, including cross-attention via `context`
and optional ADM conditioning via `y` (enforced by the UNet model invariants; no fallbacks).

Symbols (top-level; keep in sync; no ghosts):
- `TimestepBlock` (class): Marker base class for layers that accept a timestep embedding.
- `TimestepEmbedSequential` (class): Sequential container that routes `(x, emb, context, transformer_options)` through mixed layer types
  and applies `block_inner_modifiers` hooks (contains nested per-layer dispatch logic).
- `Timestep` (class): Timestep embedding wrapper (calls `timestep_embedding` from `.utils`).
- `GEGLU` (class): Gated GELU linear projection used in transformer feed-forward blocks.
- `FeedForward` (class): Transformer feed-forward network (optionally GEGLU-gated) with dropout.
- `CrossAttention` (class): Cross-attention module used by spatial transformers (supports transformer options + attention backend selection).
- `BasicTransformerBlock` (class): Transformer block (attn + FFN + norms) used inside `SpatialTransformer`.
- `SpatialTransformer` (class): Applies transformer blocks over spatial feature maps (rearrange to sequences + back; routes `context`).
- `Upsample` (class): UNet upsampling layer (supports explicit output_shape).
- `Downsample` (class): UNet downsampling layer.
- `ResBlock` (class): Timestep-conditioned residual block used in UNet encoder/decoder paths (supports scale/shift norms and optional up/down).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from apps.backend.runtime.attention import attention_function
from apps.backend.runtime.sampling.block_progress import (
    BLOCK_PROGRESS_INDEX_KEY,
    BLOCK_PROGRESS_TOTAL_KEY,
    resolve_block_progress_callback,
)
from .utils import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    default,
    exists,
)


class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, transformer_options=None, output_shape=None):
        transformer_options = transformer_options or {}
        block_inner_modifiers = transformer_options.get("block_inner_modifiers", [])
        for layer_index, layer in enumerate(self):
            for modifier in block_inner_modifiers:
                x = modifier(x, "before", layer, layer_index, self, transformer_options)
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, transformer_options)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
            for modifier in block_inner_modifiers:
                x = modifier(x, "after", layer, layer_index, self, transformer_options)
        return x


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        from .utils import timestep_embedding

        return timestep_embedding(timesteps, self.dim)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None, transformer_options=None):
        transformer_options = transformer_options or {}
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
        else:
            v = self.to_v(context)
        out = attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint_enabled=True,
        ff_in=False,
        inner_dim=None,
        disable_self_attn=False,
    ):
        super().__init__()
        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim
        self.is_res = inner_dim == dim
        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.norm1 = nn.LayerNorm(inner_dim)
        self.attn2 = CrossAttention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(inner_dim)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.checkpoint_enabled = checkpoint_enabled
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x, context=None, transformer_options=None):
        transformer_options = transformer_options or {}
        return checkpoint(self._forward, (x, context, transformer_options), None, self.checkpoint_enabled)

    def _forward(self, x, context=None, transformer_options=None):
        transformer_options = transformer_options or {}
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}
        for key, value in transformer_options.items():
            if key == "patches":
                transformer_patches = value
            elif key == "patches_replace":
                transformer_patches_replace = value
            else:
                extra_options[key] = value
        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x = x + x_skip
        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None
        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for modifier in patch:
                n, context_attn1, value_attn1 = modifier(n, context_attn1, value_attn1, extra_options)
        transformer_block = (block[0], block[1], block_index) if block is not None else None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block if transformer_block in attn1_replace_patch else block
        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=extra_options)
        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for modifier in patch:
                n = modifier(n, extra_options)
        x = x + n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for modifier in patch:
                x = modifier(x, extra_options)
        if self.attn2 is not None:
            n = self.norm2(x)
            context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for modifier in patch:
                    n, context_attn2, value_attn2 = modifier(n, context_attn2, value_attn2, extra_options)
            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block if transformer_block in attn2_replace_patch else block
            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=extra_options)
        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for modifier in patch:
                n = modifier(n, extra_options)
        x = x + n
        x_skip = x if self.is_res else 0
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x + x_skip
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint_enabled=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options=None):
        transformer_options = transformer_options or {}
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)

        block_progress_callback = resolve_block_progress_callback(transformer_options)
        total_blocks_raw = transformer_options.get(BLOCK_PROGRESS_TOTAL_KEY, 0)
        if isinstance(total_blocks_raw, bool) or not isinstance(total_blocks_raw, int):
            raise RuntimeError(
                "SpatialTransformer transformer_options block-progress total must be an integer "
                f"(got {type(total_blocks_raw).__name__})."
            )
        total_blocks = int(total_blocks_raw)
        if total_blocks < 0:
            raise RuntimeError(
                "SpatialTransformer transformer_options block-progress total must be >= 0 "
                f"(got {total_blocks})."
            )

        global_index_raw = transformer_options.get(BLOCK_PROGRESS_INDEX_KEY, 0)
        if isinstance(global_index_raw, bool) or not isinstance(global_index_raw, int):
            raise RuntimeError(
                "SpatialTransformer transformer_options block-progress index must be an integer "
                f"(got {type(global_index_raw).__name__})."
            )
        global_block_index = int(global_index_raw)
        if global_block_index < 0:
            raise RuntimeError(
                "SpatialTransformer transformer_options block-progress index must be >= 0 "
                f"(got {global_block_index})."
            )

        if block_progress_callback is not None and total_blocks <= 0:
            raise RuntimeError(
                "SpatialTransformer block progress callback requires a positive global total."
            )

        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for index, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = index
            global_block_index += 1
            if total_blocks > 0 and global_block_index > total_blocks:
                raise RuntimeError(
                    "SpatialTransformer block progress index exceeded configured total "
                    f"(index={global_block_index}, total={total_blocks})."
                )
            transformer_options[BLOCK_PROGRESS_INDEX_KEY] = global_block_index
            if block_progress_callback is not None:
                block_progress_callback(global_block_index, total_blocks)
            x = block(x, context=context[index], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + residual


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        x = torch.nn.functional.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
            )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, transformer_options=None):
        transformer_options = transformer_options or {}
        return checkpoint(self._forward, (x, emb, transformer_options), None, self.use_checkpoint)

    def _forward(self, x, emb, transformer_options=None):
        transformer_options = transformer_options or {}
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            if "group_norm_wrapper" in transformer_options:
                in_norm, in_rest = in_rest[0], in_rest[1:]
                h = transformer_options["group_norm_wrapper"](in_norm, x, transformer_options)
                h = in_rest(h)
            else:
                h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            if "group_norm_wrapper" in transformer_options:
                in_norm = self.in_layers[0]
                h = transformer_options["group_norm_wrapper"](in_norm, x, transformer_options)
                h = self.in_layers[1:](h)
            else:
                h = self.in_layers(x)
        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            if "group_norm_wrapper" in transformer_options:
                h = transformer_options["group_norm_wrapper"](out_norm, h, transformer_options)
            else:
                h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = h * (1 + scale)
                h = h + shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            if "group_norm_wrapper" in transformer_options:
                h = transformer_options["group_norm_wrapper"](self.out_layers[0], h, transformer_options)
                h = self.out_layers[1:](h)
            else:
                h = self.out_layers(h)
        return self.skip_connection(x) + h


__all__ = [
    "BasicTransformerBlock",
    "CrossAttention",
    "Downsample",
    "FeedForward",
    "ResBlock",
    "SpatialTransformer",
    "Timestep",
    "TimestepBlock",
    "TimestepEmbedSequential",
    "Upsample",
    "GEGLU",
]
