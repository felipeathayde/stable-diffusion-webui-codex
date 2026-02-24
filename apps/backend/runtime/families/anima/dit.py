"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Cosmos Predict2 (MiniTrainDiT) runtime for Anima.
Provides a clean, Codex-native implementation of the core DiT used by Anima, including:
- PatchEmbed/unpatchify path (5D latents),
- RoPE3D positional embeddings,
- AdaLN (optionally LoRA-augmented) modulation,
- Self-attention + cross-attention + MLP transformer blocks.

Symbols (top-level; keep in sync; no ghosts):
- `MiniTrainDiT` (class): Core Cosmos Predict2 DiT model (expects 5D latent input; images use T=1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend
from .nn import RMSNorm
from .position_embedding import LearnablePosEmbAxis, VideoRopePosition3DEmb


def _apply_rotary_pos_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (..., head_dim)
    # freqs: broadcastable to (..., head_dim/2, 2, 2)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE expects an even head_dim; got {int(x.shape[-1])}")
    compute_dtype = freqs.dtype if freqs.is_floating_point() else torch.float32
    x_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    if x_.dtype != compute_dtype:
        x_ = x_.to(dtype=compute_dtype)
    if freqs.dtype != compute_dtype:
        freqs = freqs.to(dtype=compute_dtype)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    out = out.movedim(-1, -2).reshape(*x.shape).to(dtype=x.dtype)
    return out


def _promote_fp16_residual_stream(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16:
        return x.float()
    return x


class GPT2FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, device: torch.device | None, dtype: torch.dtype | None) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(int(dim), int(hidden_dim), bias=False, device=device, dtype=dtype)
        self.layer2 = nn.Linear(int(hidden_dim), int(dim), bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        *,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        ctx_dim = int(query_dim if context_dim is None else context_dim)
        inner_dim = int(head_dim) * int(num_heads)

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.query_dim = int(query_dim)
        self.context_dim = ctx_dim

        self.q_proj = nn.Linear(self.query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, self.query_dim, bias=False, device=device, dtype=dtype)
        self.output_dropout = nn.Dropout(float(dropout)) if float(dropout) > 1e-4 else nn.Identity()

    def compute_qkv(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None,
        rope_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Attention expects x as (B,S,C); got shape={tuple(x.shape)}")
        ctx = x if context is None else context
        q = self.q_proj(x)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        q, k, v = map(
            lambda t: rearrange(t, "b s (h d) -> b s h d", h=self.num_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = _apply_rotary_pos_emb(q, rope_emb)
            k = _apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        del transformer_options
        q, k, v = self.compute_qkv(x, context=context, rope_emb=rope_emb)
        # SDPA expects (B,H,S,D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = attention_function_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=False,
            backend=AttentionBackend.PYTORCH,
        )
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.shape[0], out.shape[1], self.num_heads * self.head_dim)
        return self.output_dropout(self.output_proj(out))

    def init_weights(self) -> None:
        torch.nn.init.zeros_(self.output_proj.weight)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = int(num_channels)

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        if timesteps_B_T.ndim != 2:
            raise ValueError(f"Timesteps expects (B,T) tensor; got shape={tuple(timesteps_B_T.shape)}")
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / float(max(half_dim, 1))
        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        out = torch.cat([cos_emb, sin_emb], dim=-1)
        return rearrange(out, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_adaln_lora: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_features)
        self.out_dim = int(out_features)
        self.use_adaln_lora = bool(use_adaln_lora)
        # Upstream intent: when AdaLN-LoRA is enabled, bias is disabled for backward compatibility.
        self.linear_1 = nn.Linear(self.in_dim, self.out_dim, bias=not self.use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        if self.use_adaln_lora:
            self.linear_2 = nn.Linear(self.out_dim, 3 * self.out_dim, bias=False, device=device, dtype=dtype)
        else:
            self.linear_2 = nn.Linear(self.out_dim, self.out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_channels: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = int(spatial_patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t rt) (h rh) (w rw) -> b t h w (c rt rh rw)",
                rt=self.temporal_patch_size,
                rh=self.spatial_patch_size,
                rw=self.spatial_patch_size,
            ),
            nn.Linear(
                int(in_channels) * self.spatial_patch_size * self.spatial_patch_size * self.temporal_patch_size,
                int(out_channels),
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"PatchEmbed expects (B,C,T,H,W); got shape={tuple(x.shape)}")
        _, _, t, h, w = x.shape
        if (h % self.spatial_patch_size) != 0 or (w % self.spatial_patch_size) != 0:
            raise ValueError(f"H,W={h,w} must be divisible by spatial_patch_size={self.spatial_patch_size}")
        if (t % self.temporal_patch_size) != 0:
            raise ValueError(f"T={t} must be divisible by temporal_patch_size={self.temporal_patch_size}")
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_adaln_chunks = 2
        self.use_adaln_lora = bool(use_adaln_lora)
        self.adaln_lora_dim = int(adaln_lora_dim)

        self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = nn.Linear(
            self.hidden_size,
            int(spatial_patch_size) * int(spatial_patch_size) * int(temporal_patch_size) * int(out_channels),
            bias=False,
            device=device,
            dtype=dtype,
        )

        if self.use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(self.adaln_lora_dim, self.n_adaln_chunks * self.hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.n_adaln_chunks * self.hidden_size, bias=False, device=device, dtype=dtype),
            )

    def forward(self, x_B_T_H_W_D: torch.Tensor, emb_B_T_D: torch.Tensor, *, adaln_lora_B_T_3D: torch.Tensor | None) -> torch.Tensor:
        if self.use_adaln_lora:
            if adaln_lora_B_T_3D is None:
                raise ValueError("adaln_lora_B_T_3D is required when use_adaln_lora=True")
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1.0 + scale) + shift
        return self.linear(x)


class Block(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)

        self.layer_norm_self_attn = nn.LayerNorm(self.x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = Attention(
            self.x_dim,
            None,
            num_heads=int(num_heads),
            head_dim=self.x_dim // int(num_heads),
            device=device,
            dtype=dtype,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(self.x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            self.x_dim,
            int(context_dim),
            num_heads=int(num_heads),
            head_dim=self.x_dim // int(num_heads),
            device=device,
            dtype=dtype,
        )

        self.layer_norm_mlp = nn.LayerNorm(self.x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.mlp = GPT2FeedForward(self.x_dim, int(self.x_dim * float(mlp_ratio)), device=device, dtype=dtype)

        self.use_adaln_lora = bool(use_adaln_lora)
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.x_dim, int(adaln_lora_dim), bias=False, device=device, dtype=dtype),
                nn.Linear(int(adaln_lora_dim), 3 * self.x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.x_dim, int(adaln_lora_dim), bias=False, device=device, dtype=dtype),
                nn.Linear(int(adaln_lora_dim), 3 * self.x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.x_dim, int(adaln_lora_dim), bias=False, device=device, dtype=dtype),
                nn.Linear(int(adaln_lora_dim), 3 * self.x_dim, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(self.x_dim, 3 * self.x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(self.x_dim, 3 * self.x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.x_dim, 3 * self.x_dim, bias=False, device=device, dtype=dtype))

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            if adaln_lora_B_T_3D is None:
                raise ValueError("adaln_lora_B_T_3D is required when use_adaln_lora=True")
            shift_self, scale_self, gate_self = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        shift_self = rearrange(shift_self, "b t d -> b t 1 1 d")
        scale_self = rearrange(scale_self, "b t d -> b t 1 1 d")
        gate_self = rearrange(gate_self, "b t d -> b t 1 1 d")

        shift_cross = rearrange(shift_cross, "b t d -> b t 1 1 d")
        scale_cross = rearrange(scale_cross, "b t d -> b t 1 1 d")
        gate_cross = rearrange(gate_cross, "b t d -> b t 1 1 d")

        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d")
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d")
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d")

        b, t, h, w, d = x_B_T_H_W_D.shape

        def _fn(_x: torch.Tensor, norm: nn.Module, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return norm(_x) * (1.0 + scale) + shift

        normed = _fn(x_B_T_H_W_D, self.layer_norm_self_attn, scale_self, shift_self)
        sa = self.self_attn(
            rearrange(normed.to(dtype=compute_dtype), "b t h w d -> b (t h w) d"),
            None,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        sa = rearrange(sa, "b (t h w) d -> b t h w d", t=t, h=h, w=w)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self.to(dtype=residual_dtype) * sa.to(dtype=residual_dtype)

        normed = _fn(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross, shift_cross)
        ca = self.cross_attn(
            rearrange(normed.to(dtype=compute_dtype), "b t h w d -> b (t h w) d"),
            crossattn_emb,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        ca = rearrange(ca, "b (t h w) d -> b t h w d", t=t, h=h, w=w)
        x_B_T_H_W_D = (ca.to(dtype=residual_dtype) * gate_cross.to(dtype=residual_dtype)) + x_B_T_H_W_D

        normed = _fn(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        mlp_out = self.mlp(normed.to(dtype=compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp.to(dtype=residual_dtype) * mlp_out.to(dtype=residual_dtype)
        return x_B_T_H_W_D


@dataclass(frozen=True, slots=True)
class MiniTrainDiTConfig:
    max_img_h: int
    max_img_w: int
    max_frames: int
    in_channels: int
    out_channels: int
    patch_spatial: int
    patch_temporal: int
    concat_padding_mask: bool
    model_channels: int
    num_blocks: int
    num_heads: int
    mlp_ratio: float
    crossattn_emb_channels: int
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = False
    pos_emb_interpolation: str = "crop"
    min_fps: int = 1
    max_fps: int = 30
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    extra_per_block_abs_pos_emb: bool = False
    extra_h_extrapolation_ratio: float = 1.0
    extra_w_extrapolation_ratio: float = 1.0
    extra_t_extrapolation_ratio: float = 1.0
    rope_enable_fps_modulation: bool = True


def _pad_to_patch_size_5d(x: torch.Tensor, *, patch: tuple[int, int, int]) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected 5D (B,C,T,H,W) input; got shape={tuple(x.shape)}")
    pt, ph, pw = (int(v) for v in patch)
    if pt <= 0 or ph <= 0 or pw <= 0:
        raise ValueError(f"Invalid patch size: {patch}")
    _, _, t, h, w = x.shape
    pad_t = (-int(t)) % pt
    pad_h = (-int(h)) % ph
    pad_w = (-int(w)) % pw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))


class MiniTrainDiT(nn.Module):
    def __init__(
        self,
        *,
        config: MiniTrainDiTConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.max_img_h = int(config.max_img_h)
        self.max_img_w = int(config.max_img_w)
        self.max_frames = int(config.max_frames)
        self.in_channels = int(config.in_channels)
        self.out_channels = int(config.out_channels)
        self.patch_spatial = int(config.patch_spatial)
        self.patch_temporal = int(config.patch_temporal)
        self.num_heads = int(config.num_heads)
        self.num_blocks = int(config.num_blocks)
        self.model_channels = int(config.model_channels)
        self.concat_padding_mask = bool(config.concat_padding_mask)

        if self.patch_spatial <= 0 or self.patch_temporal <= 0:
            raise ValueError(f"Invalid patch sizes: spatial={self.patch_spatial} temporal={self.patch_temporal}")
        if self.max_img_h <= 0 or self.max_img_w <= 0 or self.max_frames <= 0:
            raise ValueError(
                f"Invalid max dims: max_img_h={self.max_img_h} max_img_w={self.max_img_w} max_frames={self.max_frames}"
            )
        if (self.max_img_h % self.patch_spatial) != 0 or (self.max_img_w % self.patch_spatial) != 0:
            raise ValueError(
                "max_img_h/max_img_w must be divisible by patch_spatial. "
                f"max_img_h={self.max_img_h} max_img_w={self.max_img_w} patch_spatial={self.patch_spatial}"
            )
        if (self.max_frames % self.patch_temporal) != 0:
            raise ValueError(
                "max_frames must be divisible by patch_temporal. "
                f"max_frames={self.max_frames} patch_temporal={self.patch_temporal}"
            )
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.model_channels <= 0:
            raise ValueError("model_channels must be > 0")
        if (self.model_channels % self.num_heads) != 0:
            raise ValueError(
                "model_channels must be divisible by num_heads. "
                f"model_channels={self.model_channels} num_heads={self.num_heads}"
            )
        head_dim = self.model_channels // self.num_heads
        if (head_dim % 2) != 0:
            raise ValueError(f"RoPE head_dim must be even; got head_dim={head_dim} (model_channels/num_heads).")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be >= 1")

        self.pos_emb_cls = str(config.pos_emb_cls)
        self.pos_emb_learnable = bool(config.pos_emb_learnable)
        self.pos_emb_interpolation = str(config.pos_emb_interpolation)
        self.min_fps = int(config.min_fps)
        self.max_fps = int(config.max_fps)
        self.rope_h_extrapolation_ratio = float(config.rope_h_extrapolation_ratio)
        self.rope_w_extrapolation_ratio = float(config.rope_w_extrapolation_ratio)
        self.rope_t_extrapolation_ratio = float(config.rope_t_extrapolation_ratio)
        self.extra_per_block_abs_pos_emb = bool(config.extra_per_block_abs_pos_emb)
        self.extra_h_extrapolation_ratio = float(config.extra_h_extrapolation_ratio)
        self.extra_w_extrapolation_ratio = float(config.extra_w_extrapolation_ratio)
        self.extra_t_extrapolation_ratio = float(config.extra_t_extrapolation_ratio)
        self.rope_enable_fps_modulation = bool(config.rope_enable_fps_modulation)

        self.use_adaln_lora = bool(config.use_adaln_lora)
        self.adaln_lora_dim = int(config.adaln_lora_dim)

        self._build_pos_embed(device=device, dtype=dtype)

        self.t_embedder = nn.Sequential(
            Timesteps(self.model_channels),
            TimestepEmbedding(
                self.model_channels,
                self.model_channels,
                use_adaln_lora=self.use_adaln_lora,
                device=device,
                dtype=dtype,
            ),
        )

        patch_in_channels = self.in_channels + (1 if self.concat_padding_mask else 0)
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=patch_in_channels,
            out_channels=self.model_channels,
            device=device,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=self.model_channels,
                    context_dim=int(config.crossattn_emb_channels),
                    num_heads=self.num_heads,
                    mlp_ratio=float(config.mlp_ratio),
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            device=device,
            dtype=dtype,
        )
        self.t_embedding_norm = RMSNorm(self.model_channels, eps=1e-6, device=device, dtype=dtype)

    def _build_pos_embed(self, *, device: torch.device | None, dtype: torch.dtype | None) -> None:
        if str(self.pos_emb_cls).strip().lower() != "rope3d":
            raise ValueError(f"Unsupported pos_emb_cls={self.pos_emb_cls!r} (expected 'rope3d').")

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=self.model_channels // self.num_heads,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                interpolation=self.pos_emb_interpolation,
                model_channels=self.model_channels,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
                device=device,
                dtype=dtype,
            )

    def _prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        *,
        fps: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0],
                    1,
                    x_B_C_T_H_W.shape[3],
                    x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype,
                    device=x_B_C_T_H_W.device,
                )
            else:
                if padding_mask.ndim == 3:
                    padding_mask = padding_mask.unsqueeze(1)
                if padding_mask.ndim != 4 or padding_mask.shape[1] != 1:
                    raise ValueError(f"padding_mask must be (B,H,W) or (B,1,H,W); got {tuple(padding_mask.shape)}")
                target_hw = (int(x_B_C_T_H_W.shape[-2]), int(x_B_C_T_H_W.shape[-1]))
                if tuple(padding_mask.shape[-2:]) != target_hw:
                    padding_mask = F.interpolate(padding_mask.float(), size=target_hw, mode="nearest").to(dtype=x_B_C_T_H_W.dtype)
            mask_5d = padding_mask.unsqueeze(2).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, mask_5d], dim=1)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(
                x_B_T_H_W_D,
                fps=fps,
                device=x_B_C_T_H_W.device,
                dtype=x_B_C_T_H_W.dtype,
            )

        rope = self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, rope, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_B_T_H_W_M,
            "b t h w (p1 p2 pt c) -> b c (t pt) (h p1) (w p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            pt=self.patch_temporal,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        transformer_options: dict | None = None,
        control: object | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del control, kwargs
        squeeze_time = False
        if x.ndim == 4:
            squeeze_time = True
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"MiniTrainDiT expects latent input as 4D/5D tensor; got shape={tuple(x.shape)}")

        orig_shape = tuple(x.shape)
        x = _pad_to_patch_size_5d(x, patch=(self.patch_temporal, self.patch_spatial, self.patch_spatial))

        x_B_C_T_H_W = x
        if timesteps.ndim == 1:
            timesteps_B_T = timesteps.unsqueeze(1)
        elif timesteps.ndim == 2:
            timesteps_B_T = timesteps
        else:
            raise ValueError(f"timesteps must be 1D or 2D; got shape={tuple(timesteps.shape)}")

        if context.ndim != 3:
            raise ValueError(f"context must be (B,S,C); got shape={tuple(context.shape)}")

        x_B_T_H_W_D, rope_emb, extra_pos_emb = self._prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.shape[0] != x_B_T_H_W_D.shape[0]:
            raise ValueError("Batch mismatch between latents and timesteps.")

        t_raw = self.t_embedder[0](timesteps_B_T).to(dtype=x_B_T_H_W_D.dtype)
        t_emb, adaln_lora = self.t_embedder[1](t_raw)
        t_emb = self.t_embedding_norm(t_emb)

        if extra_pos_emb is not None and extra_pos_emb.shape != x_B_T_H_W_D.shape:
            raise RuntimeError(f"extra_pos_emb shape mismatch: {tuple(extra_pos_emb.shape)} vs {tuple(x_B_T_H_W_D.shape)}")

        # Upstream Cosmos intent for fp16 stability:
        # keep residual stream in fp32 while block compute stays at embedding/weight dtype.
        x_B_T_H_W_D = _promote_fp16_residual_stream(x_B_T_H_W_D)

        rope_broadcast = rope_emb.unsqueeze(1).unsqueeze(0)
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_emb,
                context,
                rope_emb_L_1_1_D=rope_broadcast,
                adaln_lora_B_T_3D=adaln_lora,
                extra_per_block_pos_emb=extra_pos_emb,
                transformer_options=transformer_options,
            )

        final_input = x_B_T_H_W_D.to(dtype=context.dtype) if x_B_T_H_W_D.dtype != context.dtype else x_B_T_H_W_D
        x_B_T_H_W_O = self.final_layer(final_input, t_emb, adaln_lora_B_T_3D=adaln_lora)
        out_5d = self.unpatchify(x_B_T_H_W_O)
        out_5d = out_5d[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
        if squeeze_time:
            out_4d = out_5d.squeeze(2)
            if out_4d.ndim != 4:
                raise RuntimeError("Failed to squeeze time dimension for image inference.")
            return out_4d
        return out_5d
