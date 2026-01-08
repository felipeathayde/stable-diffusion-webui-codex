"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 Transformer model as nn.Module (format-agnostic).
Provides `WanTransformer2DModel`, a native PyTorch implementation of the WAN diffusion transformer that can load weights from
multiple formats (GGUF, safetensors, etc.) via the operations registry; GGUF handling is transparent through `CodexOperationsGGUF`.

Symbols (top-level; keep in sync; no ghosts):
- `WanArchitectureConfig` (dataclass): Architecture hyperparameters for WAN (dims/heads/blocks/patch size/etc) used for construction/inference.
- `WanRMSNorm` (class): RMSNorm with optional GGUF parameter dequantization (supports `CodexParameter` weights).
- `WanSelfAttention` (class): Self-attention block for WAN (QKV projection + SDPA implementation).
- `WanCrossAttention` (class): Cross-attention block for WAN (text context attention path).
- `WanFFN` (class): Feed-forward (MLP) block used in WAN transformer blocks.
- `WanTransformerBlock` (class): One transformer block combining attention + FFN + norms/residuals.
- `WanTransformer2DModel` (class): Full WAN transformer stack (embeddings/blocks/forward); used by `runtime/wan22/wan22.py`.
- `remap_wan22_gguf_state_dict` (function): Remaps WAN22 GGUF key names into this module’s expected parameter keys.
- `infer_wan_architecture_from_state_dict` (function): Infers `WanArchitectureConfig` from a loaded state dict (dims/layers/heads).
- `load_wan_transformer_from_state_dict` (function): Constructs `WanTransformer2DModel` and loads weights from a state dict (with remapping).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from apps.backend.runtime.ops.operations_gguf import CodexParameter, dequantize_tensor as gguf_dequantize_tensor

from .inference import infer_wan22_latent_channels, infer_wan22_patch_embedding, infer_wan22_patch_size_and_in_channels
from .sdpa import sdpa as wan_sdpa

logger = logging.getLogger("backend.runtime.wan22.model")


# Configuration
@dataclass(frozen=True)
class WanArchitectureConfig:
    """Configuration for WAN transformer architecture."""

    d_model: int = 5120
    # WAN2.2 commonly uses head_dim=128 => n_heads=d_model//128 (ex.: 5120 -> 40).
    n_heads: int = 40
    n_blocks: int = 40
    mlp_ratio: float = 4.0
    context_dim: int = 4096
    time_embed_dim: int = 256
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # T, H, W
    in_channels: int = 16
    latent_channels: int = 16
    qkv_bias: bool = True
    use_text_projection: bool = True
    use_guidance: bool = True


# Submodules
class WanRMSNorm(nn.Module):
    """RMS normalization with learned weight (WAN GGUF uses affine RMSNorm weights)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if isinstance(w, CodexParameter) and w.qtype is not None:
            w = gguf_dequantize_tensor(w)
        if not torch.is_tensor(w):
            w = torch.as_tensor(w)
        w = w.to(device=x.device, dtype=x.dtype)
        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)) * w


class WanSelfAttention(nn.Module):
    """Self-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C = x.shape

        # QKV projections
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_out = wan_sdpa(q, k, v, causal=False)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)

        return self.o(attn_out)


class WanCrossAttention(nn.Module):
    """Cross-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=True)

        # WAN GGUF uses an affine RMSNorm weight for the K path.
        self.norm_k = WanRMSNorm(context_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C = x.shape
        _, S, _ = context.shape

        # QKV projections
        q = self.q(x)
        k = self.k(self.norm_k(context))
        v = self.v(context)

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_out = wan_sdpa(q, k, v, causal=False)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)

        return self.o(attn_out)


class WanFFN(nn.Module):
    """Feed-forward network with SiLU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply modulation if provided
        if scale is not None or shift is not None:
            if shift is not None:
                x = x + shift[:, None, :]
            if scale is not None:
                x = x * (1 + scale[:, None, :])

        x = self.fc1(x)
        x = x * torch.sigmoid(x)  # SiLU
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
    ):
        super().__init__()
        # WAN GGUF semantics:
        # - norm1/norm2: LayerNorm without affine (pre-norm for SA/FFN)
        # - norm3: LayerNorm with affine (pre-norm for CA)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)

        self.self_attn = WanSelfAttention(dim, num_heads, qkv_bias)
        self.cross_attn = WanCrossAttention(dim, context_dim, num_heads, qkv_bias)

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=True),
        )

        # Per-block modulation: [6, dim] for [sa_shift, sa_scale, sa_gate, ffn_shift, ffn_scale, ffn_gate]
        self.modulation = nn.Parameter(torch.zeros(6, dim))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,  # [B, 6, dim]
    ) -> torch.Tensor:
        # Combine time embedding with per-block modulation
        mod = time_emb + self.modulation.unsqueeze(0)  # [B, 6, dim]

        sa_shift, sa_scale, sa_gate = mod[:, 0], mod[:, 1], mod[:, 2]
        ffn_shift, ffn_scale, ffn_gate = mod[:, 3], mod[:, 4], mod[:, 5]

        # Self-attention: pre-norm (no affine) + time modulation + gated residual
        x_sa = self.norm1(x)
        x_sa = x_sa * (1 + sa_scale[:, None, :]) + sa_shift[:, None, :]
        sa_out = self.self_attn(x_sa)
        x = x + sa_out * sa_gate[:, None, :]

        # Cross-attention: pre-norm3 (affine) + residual (no time modulation)
        x_ca = self.norm3(x)
        ca_out = self.cross_attn(x_ca, context)
        x = x + ca_out

        # FFN: pre-norm (no affine) + time modulation + gated residual
        x_ffn = self.norm2(x)
        x_ffn = x_ffn * (1 + ffn_scale[:, None, :]) + ffn_shift[:, None, :]
        ffn_out = self.ffn(x_ffn)
        x = x + ffn_out * ffn_gate[:, None, :]

        return x


class WanTransformer2DModel(nn.Module):
    """WAN Diffusion Transformer as nn.Module.

    This is a format-agnostic implementation that works with any weight
    format (GGUF, safetensors, etc.) via the operations registry.
    """

    def __init__(self, config: WanArchitectureConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_blocks = config.n_blocks

        # Patch embedding (3D conv for video)
        kT, kH, kW = config.patch_size
        patch_dim = config.latent_channels * kT * kH * kW
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=(1, kH, kW),
            padding=0,
        )

        # Time embedding
        time_dim = config.d_model
        self.time_embed = nn.Sequential(
            nn.Linear(config.time_embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Time projection to modulation
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * config.d_model),  # [6, d_model] per block
        )

        # Text embedding projection (optional; some checkpoints already output d_model)
        if config.use_text_projection:
            self.text_embed = nn.Sequential(
                nn.Linear(config.context_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.text_embed = nn.Identity()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                dim=config.d_model,
                num_heads=config.n_heads,
                context_dim=config.d_model,  # After text projection
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.n_blocks)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(config.d_model, elementwise_affine=False)
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
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embedding."""
        base_dim = int(dim if dim is not None else self.config.time_embed_dim)
        half = max(base_dim // 2, 1)
        freq = torch.arange(half, device=t.device, dtype=torch.float32)
        div_term = torch.exp(-math.log(10000.0) * freq / max(half - 1, 1))
        angles = t.to(dtype=torch.float32)[:, None] * div_term[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] != base_dim:
            emb = torch.nn.functional.pad(emb, (0, base_dim - emb.shape[1]))
        return emb

    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, H, W] latent video
        timestep: torch.Tensor,  # [B,] or scalar
        context: torch.Tensor,  # [B, L, context_dim] text embeddings
    ) -> torch.Tensor:
        """Forward pass of WAN transformer.

        Args:
            x: Input latent video [B, C, T, H, W]
            timestep: Diffusion timestep
            context: Text conditioning embeddings

        Returns:
            Output latent video [B, C, T, H, W]
        """
        device = x.device
        dtype = x.dtype
        B, C, T, H, W = x.shape

        # Timestep to scalar tensor
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        timestep = timestep.to(device=device, dtype=torch.float32).view(-1)
        if timestep.numel() == 1 and B > 1:
            timestep = timestep.expand(B)

        # Time embedding
        t_emb = self._timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb.to(dtype))  # [B, d_model]

        # Time projection to modulation [B, 6, d_model]
        t_proj = self.time_proj(t_emb)
        t_proj = t_proj.view(B, 6, self.d_model)

        # Text embedding projection
        ctx = self.text_embed(context.to(dtype))  # [B, L, d_model]

        # Patch embed: [B, C, T, H, W] -> [B, d_model, T', H', W'] -> [B, T'*H'*W', d_model]
        tokens = self.patch_embed(x)
        _, _, t_grid, h_grid, w_grid = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, L, d_model]

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens, ctx, t_proj)

        # Output head (WAN GGUF semantics: LN without affine + repeated modulation)
        tokens = self.norm_out(tokens)

        combined = self.head_modulation.unsqueeze(0) + t_proj.unsqueeze(2)
        shift6, scale6 = combined.unbind(dim=2)  # [B, 6, C]

        L = tokens.shape[1]
        groups = int(shift6.shape[1])
        repeats = (L + groups - 1) // groups
        pad = repeats * groups - L
        if pad:
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad))
        tokens_6 = tokens.view(B, repeats, groups, self.d_model).transpose(1, 2)  # [B, 6, repeats, C]
        fused = shift6[:, :, None, :] + tokens_6 * (1.0 + scale6[:, :, None, :])
        fused = fused.transpose(1, 2).reshape(B, repeats * groups, self.d_model)[:, :L]

        patches = self.head(fused)

        # Unpatchify: [B, L, patch_dim] -> [B, C, T, H, W]
        kT, kH, kW = self.config.patch_size
        out = patches.view(B, t_grid, h_grid, w_grid, kT, kH, kW, self.config.latent_channels)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, self.config.latent_channels, t_grid * kT, h_grid * kH, w_grid * kW)

        return out


# Weight loading helper
def remap_wan22_gguf_state_dict(state_dict: dict) -> dict:
    """Remap WAN GGUF checkpoint keys to WanTransformer2DModel keys.

    The WAN GGUF files use common checkpoint-export names (e.g. `patch_embedding.*`,
    `time_embedding.*`, `head.head.*`). The nn.Module implementation uses
    Codex-native names (e.g. `patch_embed.*`, `time_embed.*`, `head.*`).

    This helper makes the GGUF path loadable without keeping a WAN-specific
    state-dict runner.
    """
    remapped: dict[str, object] = {}
    for key, value in state_dict.items():
        k = str(key)
        if k.startswith("patch_embedding."):
            k = "patch_embed." + k[len("patch_embedding."):]
        elif k.startswith("time_embedding."):
            k = "time_embed." + k[len("time_embedding."):]
        elif k.startswith("time_projection."):
            k = "time_proj." + k[len("time_projection."):]
        elif k.startswith("text_embedding."):
            k = "text_embed." + k[len("text_embedding."):]
        elif k.startswith("head.head."):
            k = "head." + k[len("head.head."):]
        elif k == "head.modulation":
            k = "head_modulation"
        remapped[k] = value
    return remapped


def infer_wan_architecture_from_state_dict(state_dict: dict) -> WanArchitectureConfig:
    """Infer WAN architecture parameters from a (remapped) state_dict."""

    def _shape(key: str) -> tuple[int, ...] | None:
        value = state_dict.get(key)
        if value is None:
            return None
        try:
            shape = tuple(int(s) for s in getattr(value, "shape", ()) or ())
        except Exception:
            return None
        return shape or None

    d_model = 5120
    patch_embed_shape = _shape("patch_embed.weight")
    if patch_embed_shape and len(patch_embed_shape) == 5:
        _in, model_dim, _patch = infer_wan22_patch_embedding(patch_embed_shape, default_model_dim=d_model)
        d_model = int(model_dim)
    else:
        for key in ("time_embed.0.weight", "blocks.0.self_attn.q.weight"):
            shape = _shape(key)
            if shape and len(shape) >= 1:
                d_model = int(shape[0])
                break

    n_blocks = 0
    for key in state_dict.keys():
        ks = str(key)
        if not ks.startswith("blocks."):
            continue
        parts = ks.split(".", 2)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        n_blocks = max(n_blocks, idx + 1)

    # Default head_dim heuristic (matches the legacy GGUF runner).
    n_heads = 32
    for head_dim in (128, 64):
        if d_model % head_dim == 0:
            candidate = d_model // head_dim
            if 8 <= candidate <= 64:
                n_heads = int(candidate)
                break

    patch_shape = _shape("patch_embed.weight")
    head_shape = _shape("head.weight")
    patch_size, in_channels = infer_wan22_patch_size_and_in_channels(
        patch_shape,
        default_patch_size=(1, 2, 2),
        default_in_channels=16,
    )

    time_embed_dim = 256
    te0_shape = _shape("time_embed.0.weight")
    if te0_shape and len(te0_shape) == 2:
        time_embed_dim = int(te0_shape[1])

    use_text_projection = "text_embed.0.weight" in state_dict and "text_embed.2.weight" in state_dict
    context_dim = d_model
    if use_text_projection:
        t0_shape = _shape("text_embed.0.weight")
        if t0_shape and len(t0_shape) == 2:
            context_dim = int(t0_shape[1])

    mlp_ratio = 4.0
    ffn_shape = _shape("blocks.0.ffn.0.weight")
    if ffn_shape and len(ffn_shape) == 2 and ffn_shape[1] == d_model and d_model > 0:
        mlp_ratio = float(ffn_shape[0]) / float(d_model)

    qkv_bias = "blocks.0.self_attn.q.bias" in state_dict or "blocks.0.self_attn.k.bias" in state_dict

    latent_channels = infer_wan22_latent_channels(
        head_shape,
        patch_size=patch_size,
        default_latent_channels=in_channels,
    )

    return WanArchitectureConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks or 1,
        mlp_ratio=mlp_ratio,
        context_dim=context_dim,
        time_embed_dim=time_embed_dim,
        patch_size=patch_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
        qkv_bias=qkv_bias,
        use_text_projection=use_text_projection,
    )


def load_wan_transformer_from_state_dict(
    state_dict: dict,
    config: Optional[WanArchitectureConfig] = None,
) -> WanTransformer2DModel:
    """Load WanTransformer2DModel from a state dict.

    Can handle both native format and converted GGUF weights.

    Args:
        state_dict: Model weights (may contain CodexParameter)
        config: Model configuration (derived from state if not provided)

    Returns:
        Loaded WanTransformer2DModel
    """
    if config is None:
        config = infer_wan_architecture_from_state_dict(state_dict)

    model = WanTransformer2DModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("WAN state_dict missing %d keys (sample=%s)", len(missing), missing[:5])
    if unexpected:
        logger.debug("WAN state_dict has %d unexpected keys (sample=%s)", len(unexpected), unexpected[:5])

    logger.info(
        "Loaded WanTransformer2DModel: %d blocks, d_model=%d",
        config.n_blocks,
        config.d_model,
    )

    return model
