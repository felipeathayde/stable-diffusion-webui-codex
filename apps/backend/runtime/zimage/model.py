"""Z Image Turbo (Alibaba) Transformer Model.

NextDiT-based diffusion transformer for Z Image Turbo (6B params).
Based on Lumina2 architecture with z_image_modulation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("backend.runtime.zimage.model")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ZImageConfig:
    """Configuration for Z Image Transformer."""
    patch_size: int = 2
    in_channels: int = 16  # Flux VAE latent channels
    dim: int = 3840  # Z Image specific
    n_layers: int = 32
    n_refiner_layers: int = 2
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: float = 4.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560  # Qwen3 4B hidden size
    axes_dims: Tuple[int, ...] = (16, 56, 56)
    axes_lens: Tuple[int, ...] = (1, 512, 512)
    rope_theta: float = 10000.0
    z_image_modulation: bool = True


# =============================================================================
# Helper Functions
# =============================================================================

def modulate(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation: x * (1 + scale)."""
    return x * (1 + scale.unsqueeze(1))


def clamp_fp16(x: torch.Tensor) -> torch.Tensor:
    """Clamp values for fp16 stability."""
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


# =============================================================================
# RoPE Embedding
# =============================================================================

class EmbedND(nn.Module):
    """N-dimensional RoPE embedding."""
    
    def __init__(self, dim: int, theta: float, axes_dim: Tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
    
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, N, num_axes]
        n_axes = ids.shape[-1]
        emb = torch.cat([
            self._rope_1d(ids[..., i], self.axes_dim[i], self.theta)
            for i in range(n_axes)
        ], dim=-1)
        return emb
    
    def _rope_1d(self, pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
        args = pos.unsqueeze(-1).float() * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K."""
    # freqs: [B, head_dim, N]
    # q, k: [B, N, H, D]
    B, N, H, D = q.shape
    freqs = freqs.permute(0, 2, 1)  # [B, N, head_dim]
    
    cos = freqs[..., :D//2].cos().unsqueeze(2)  # [B, N, 1, D//2]
    sin = freqs[..., :D//2].sin().unsqueeze(2)  # [B, N, 1, D//2]
    
    q1, q2 = q[..., :D//2], q[..., D//2:]
    k1, k2 = k[..., :D//2], k[..., D//2:]
    
    q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    return q_out, k_out


# =============================================================================
# Core Layers
# =============================================================================

class TimestepEmbedder(nn.Module):
    """Timestep embedding using sinusoidal encoding + MLP."""
    
    def __init__(self, dim: int, output_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.output_size = output_size or dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, self.output_size),
        )
    
    def forward(self, t: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        # t: [B] normalized timestep 0-1
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb.to(dtype))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            norm = norm * self.weight
        return norm


class JointAttention(nn.Module):
    """Multi-head attention with QK normalization and RoPE."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.qkv = nn.Linear(dim, (n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split([
            self.n_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        ], dim=-1)
        
        q = q.view(B, N, self.n_heads, self.head_dim)
        k = k.view(B, N, self.n_kv_heads, self.head_dim)
        v = v.view(B, N, self.n_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q, k = apply_rope(q, k, freqs_cis)
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)
            v = v.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)
        
        # Attention
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        
        return self.out(attn)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, dim: int, multiple_of: int = 256, ffn_multiplier: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * ffn_multiplier)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(clamp_fp16(F.silu(self.w1(x)) * self.w3(x)))


class JointTransformerBlock(nn.Module):
    """Transformer block with modulation for diffusion."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_multiplier: float = 4.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        modulation: bool = True,
        z_image_modulation: bool = False,
    ):
        super().__init__()
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(dim, multiple_of, ffn_multiplier)
        
        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)
        
        self.modulation = modulation
        if modulation:
            mod_dim = min(dim, 256) if z_image_modulation else min(dim, 1024)
            if z_image_modulation:
                self.adaLN_modulation = nn.Linear(mod_dim, 4 * dim, bias=True)
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(mod_dim, 4 * dim, bias=True),
                )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                clamp_fp16(self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    mask,
                    freqs_cis,
                ))
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                clamp_fp16(self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                ))
            )
        else:
            x = x + self.attention_norm2(
                clamp_fp16(self.attention(self.attention_norm1(x), mask, freqs_cis))
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        
        return x


class FinalLayer(nn.Module):
    """Final projection layer with modulation."""
    
    def __init__(self, dim: int, patch_size: int, out_channels: int, z_image_modulation: bool = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        
        mod_dim = min(dim, 256) if z_image_modulation else min(dim, 1024)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, dim, bias=True),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        return self.linear(x)


# =============================================================================
# Main Model
# =============================================================================

class ZImageTransformer2DModel(nn.Module):
    """Z Image Turbo Diffusion Transformer.
    
    NextDiT architecture with z_image_modulation for efficient inference.
    """
    
    def __init__(self, config: Optional[ZImageConfig] = None):
        super().__init__()
        config = config or ZImageConfig()
        self.config = config
        
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.dim = config.dim
        
        # Patch embedding
        self.x_embedder = nn.Linear(
            config.patch_size * config.patch_size * config.in_channels,
            config.dim,
            bias=True,
        )
        
        # Noise refiners (with modulation)
        self.noise_refiner = nn.ModuleList([
            JointTransformerBlock(
                config.dim, config.n_heads, config.n_kv_heads,
                config.multiple_of, config.ffn_dim_multiplier,
                config.norm_eps, config.qk_norm,
                modulation=True, z_image_modulation=config.z_image_modulation,
            )
            for _ in range(config.n_refiner_layers)
        ])
        
        # Context refiners (no modulation)
        self.context_refiner = nn.ModuleList([
            JointTransformerBlock(
                config.dim, config.n_heads, config.n_kv_heads,
                config.multiple_of, config.ffn_dim_multiplier,
                config.norm_eps, config.qk_norm,
                modulation=False,
            )
            for _ in range(config.n_refiner_layers)
        ])
        
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(
            min(config.dim, 1024),
            output_size=256 if config.z_image_modulation else None,
        )
        
        # Caption embedder
        self.cap_embedder = nn.Sequential(
            RMSNorm(config.cap_feat_dim, config.norm_eps),
            nn.Linear(config.cap_feat_dim, config.dim, bias=True),
        )
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            JointTransformerBlock(
                config.dim, config.n_heads, config.n_kv_heads,
                config.multiple_of, config.ffn_dim_multiplier,
                config.norm_eps, config.qk_norm,
                modulation=True, z_image_modulation=config.z_image_modulation,
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm_final = RMSNorm(config.dim, config.norm_eps)
        self.final_layer = FinalLayer(
            config.dim, config.patch_size, self.out_channels,
            z_image_modulation=config.z_image_modulation,
        )
        
        # RoPE
        self.rope_embedder = EmbedND(
            config.dim // config.n_heads,
            config.rope_theta,
            config.axes_dims,
        )
    
    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert image to patch embeddings."""
        B, C, H, W = x.shape
        pH = pW = self.patch_size
        
        x = x.view(B, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)  # [B, N, patch**2 * C]
        
        return self.x_embedder(x), (H, W)
    
    def unpatchify(self, x: torch.Tensor, img_size: Tuple[int, int], cap_len: int) -> torch.Tensor:
        """Convert patch predictions back to image."""
        H, W = img_size
        pH = pW = self.patch_size
        
        x = x[:, cap_len:]  # Remove caption tokens
        x = x.view(-1, H // pH, W // pW, pH, pW, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(3, 4).flatten(1, 2)  # [B, C, H, W]
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W] latent
        timestep: torch.Tensor,  # [B] normalized 0-1
        context: torch.Tensor,  # [B, L, D] text embeddings
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Image latent [B, C, H, W].
            timestep: Normalized timestep [B].
            context: Text conditioning [B, L, cap_feat_dim].
            attention_mask: Optional attention mask.
        
        Returns:
            Velocity prediction [B, C, H, W].
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Time embedding
        t = 1.0 - timestep  # ComfyUI convention
        t_emb = self.t_embedder(t, dtype=dtype)
        
        # Patch embed image
        img_patches, img_size = self.patchify(x)
        H, W = img_size
        
        # Embed caption
        cap_feats = self.cap_embedder(context)
        cap_len = cap_feats.shape[1]
        
        # Position IDs for RoPE
        cap_pos_ids = torch.zeros(B, cap_len, 3, device=device, dtype=torch.float32)
        cap_pos_ids[:, :, 0] = torch.arange(cap_len, device=device).float() + 1.0
        
        H_tokens, W_tokens = H // self.patch_size, W // self.patch_size
        img_pos_ids = torch.zeros(B, img_patches.shape[1], 3, device=device, dtype=torch.float32)
        img_pos_ids[:, :, 0] = cap_len + 1
        img_pos_ids[:, :, 1] = torch.arange(H_tokens, device=device).view(-1, 1).repeat(1, W_tokens).flatten().float()
        img_pos_ids[:, :, 2] = torch.arange(W_tokens, device=device).view(1, -1).repeat(H_tokens, 1).flatten().float()
        
        freqs_cis = self.rope_embedder(torch.cat([cap_pos_ids, img_pos_ids], dim=1)).movedim(1, 2)
        
        # Refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, attention_mask, freqs_cis[:, :cap_len])
        
        # Refine noise
        for layer in self.noise_refiner:
            img_patches = layer(img_patches, None, freqs_cis[:, cap_len:], t_emb)
        
        # Concatenate for joint processing
        full_seq = torch.cat([cap_feats, img_patches], dim=1)
        
        # Main transformer
        for layer in self.layers:
            full_seq = layer(full_seq, None, freqs_cis, t_emb)
        
        # Final projection
        full_seq = self.final_layer(full_seq, t_emb)
        
        # Unpatchify
        out = self.unpatchify(full_seq, img_size, cap_len)
        
        return -out  # Velocity to noise


def load_zimage_from_state_dict(
    state_dict: dict,
    config: Optional[ZImageConfig] = None,
) -> ZImageTransformer2DModel:
    """Load Z Image model from state dict."""
    config = config or ZImageConfig()
    model = ZImageTransformer2DModel(config)
    
    # TODO: Add key mapping for different checkpoint formats
    model.load_state_dict(state_dict, strict=False)
    
    return model
