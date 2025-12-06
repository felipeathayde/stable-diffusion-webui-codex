"""WAN 2.2 Transformer model as nn.Module (format-agnostic).

This module provides WanTransformer2DModel, a native PyTorch implementation
of the WAN diffusion transformer that works with any weight format (GGUF,
safetensors, etc.) via standard nn.Module weights.

Unlike WanDiTGGUF which operates diretamente sobre state dicts GGUF, este
modelo usa camadas nn.Module, deixando o formato de pesos transparente
para quem chama.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger("backend.runtime.wan22.model")


@dataclass(frozen=True)
class WanArchitectureConfig:
    """Configuration for WAN transformer architecture."""

    d_model: int = 5120
    n_heads: int = 80
    n_blocks: int = 40
    mlp_ratio: float = 4.0
    context_dim: int = 4096
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # T, H, W
    in_channels: int = 16
    latent_channels: int = 16
    qkv_bias: bool = True
    use_guidance: bool = True


class WanRMSNorm(nn.Module):
    """RMS LayerNorm (no learned parameters, just normalization)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class WanSelfAttention(nn.Module):
    """Self-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        if scale is not None or shift is not None:
            if shift is not None:
                x = x + shift[:, None, :]
            if scale is not None:
                x = x * (1 + scale[:, None, :])

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)

        return self.out_proj(attn_out)


class WanCrossAttention(nn.Module):
    """Cross-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(context_dim)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        _, ctx_len, _ = context.shape

        x_norm = self.norm_q(x)
        ctx_norm = self.norm_k(context)

        q = self.q_proj(x_norm)
        k = self.k_proj(ctx_norm)
        v = self.v_proj(ctx_norm)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)

        return self.out_proj(attn_out)


class WanFFN(nn.Module):
    """Feed-forward network with SiLU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scale is not None or shift is not None:
            if shift is not None:
                x = x + shift[:, None, :]
            if scale is not None:
                x = x * (1 + scale[:, None, :])

        x = self.fc1(x)
        x = x * torch.sigmoid(x)
        x = self.fc2(x)
        return x


class WanTransformerBlock(nn.Module):
    """Single transformer block for WAN model.

    Each block contains:
    - Self-attention with modulation
    - Cross-attention with context
    - Feed-forward with modulation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = WanRMSNorm(dim)
        self.self_attn = WanSelfAttention(dim, num_heads, qkv_bias)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = WanCrossAttention(dim, context_dim, num_heads, qkv_bias)

        self.norm3 = WanRMSNorm(dim)
        self.ffn = WanFFN(dim, mlp_ratio)

        self.modulation = nn.Parameter(torch.zeros(6, dim))

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        mod = time_emb + self.modulation.unsqueeze(0)

        sa_shift, sa_scale, sa_gate = mod[:, 0], mod[:, 1], mod[:, 2]
        ffn_shift, ffn_scale, ffn_gate = mod[:, 3], mod[:, 4], mod[:, 5]

        x_norm = self.norm1(x)
        sa_out = self.self_attn(x_norm, scale=sa_scale, shift=sa_shift)
        x = x + sa_out * sa_gate[:, None, :]

        ca_out = self.cross_attn(x, context)
        x = x + ca_out

        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm, scale=ffn_scale, shift=ffn_shift)
        x = x + ffn_out * ffn_gate[:, None, :]

        return x


class WanTransformer2DModel(nn.Module):
    """WAN Diffusion Transformer as nn.Module."""

    def __init__(self, config: WanArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_blocks = config.n_blocks

        k_t, k_h, k_w = config.patch_size
        patch_dim = config.in_channels * k_t * k_h * k_w
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=(1, k_h, k_w),
            padding=0,
        )

        time_dim = config.d_model
        self.time_embed = nn.Sequential(
            nn.Linear(256, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * config.d_model),
        )

        self.text_embed = nn.Sequential(
            nn.Linear(config.context_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=config.d_model,
                    num_heads=config.n_heads,
                    context_dim=config.d_model,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.n_blocks)
            ],
        )

        self.norm_out = nn.LayerNorm(config.d_model)
        self.head_modulation = nn.Parameter(torch.zeros(2, config.d_model))
        self.head = nn.Linear(config.d_model, patch_dim)

        logger.info(
            "WanTransformer2DModel created: d_model=%d, n_heads=%d, n_blocks=%d",
            config.d_model,
            config.n_heads,
            config.n_blocks,
        )

    def _timestep_embedding(
        self,
        t: torch.Tensor,
        dim: int = 256,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(
                half,
                device=t.device,
                dtype=torch.float32,
            )
            / half,
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        bsz, _, _, _, _ = x.shape

        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        if timestep.numel() == 1 and bsz > 1:
            timestep = timestep.expand(bsz)

        t_emb = self._timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb.to(dtype))

        t_proj = self.time_proj(t_emb)
        t_proj = t_proj.view(bsz, 6, self.d_model)

        ctx = self.text_embed(context.to(dtype))

        tokens = self.patch_embed(x)
        _, _, t2, h2, w2 = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens, ctx, t_proj)

        tokens = self.norm_out(tokens)
        mod = t_proj[:, :2] + self.head_modulation.unsqueeze(0)
        shift, scale = mod[:, 0], mod[:, 1]
        tokens = tokens * (1 + scale[:, None, :]) + shift[:, None, :]
        patches = self.head(tokens)

        k_t, k_h, k_w = self.config.patch_size
        out = patches.view(
            bsz,
            t2,
            h2,
            w2,
            k_t,
            k_h,
            k_w,
            self.config.latent_channels,
        )
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(
            bsz,
            self.config.latent_channels,
            t2 * k_t,
            h2 * k_h,
            w2 * k_w,
        )

        return out


def load_wan_transformer_from_state_dict(
    state_dict: dict,
    config: Optional[WanArchitectureConfig] = None,
) -> WanTransformer2DModel:
    """Load WanTransformer2DModel from a state dict."""
    if config is None:
        n_blocks = 0
        for key in state_dict.keys():
            if key.startswith("blocks."):
                try:
                    idx = int(key.split(".")[1])
                    n_blocks = max(n_blocks, idx + 1)
                except (IndexError, ValueError):
                    continue

        d_model = 5120
        for key in ("time_embed.0.weight", "blocks.0.self_attn.q_proj.weight"):
            if key in state_dict:
                shape = state_dict[key].shape
                if shape:
                    d_model = int(shape[0])
                break

        config = WanArchitectureConfig(
            d_model=d_model,
            n_blocks=n_blocks,
        )

    model = WanTransformer2DModel(config)
    model.load_state_dict(state_dict, strict=False)

    logger.info(
        "Loaded WanTransformer2DModel: %d blocks, d_model=%d",
        config.n_blocks,
        config.d_model,
    )

    return model

