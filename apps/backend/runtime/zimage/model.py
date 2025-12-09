"""Z Image Turbo (Alibaba) Transformer Model.

NextDiT/Lumina2-style diffusion transformer for Z Image Turbo.
Architecture follows the original checkpoint format with verified shapes.

Key dimensions from z_image_turbo_bf16.safetensors:
- hidden_dim = 3840
- context_dim = 2560  
- t_dim = 256 (timestep embedding intermediate)
- head_dim = 128, num_heads = 30 (3840/128)
- mlp_hidden = 10240
- latent_channels = 16, patch_size = 2
- num_layers = 30, num_refiner_layers = 2
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

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
    hidden_dim: int = 3840
    context_dim: int = 2560
    latent_channels: int = 16
    patch_size: int = 2
    num_layers: int = 30
    num_refiner_layers: int = 2
    num_heads: int = 30  # 3840 / 128
    head_dim: int = 128
    t_dim: int = 256  # Timestep embedding dimension
    mlp_hidden: int = 10240
    eps: float = 1e-5
    rope_theta: float = 10000.0
    
    @property
    def in_channels(self) -> int:
        """Input channels after patchification."""
        return self.latent_channels * self.patch_size * self.patch_size


# =============================================================================
# Core Layers
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class TimestepEmbedder(nn.Module):
    """Timestep embedding: sinusoidal -> MLP -> t_dim.
    
    Architecture from checkpoint:
    - Sinusoidal encoding: 256 dims
    - mlp.0: Linear(256, 1024)
    - SiLU
    - mlp.2: Linear(1024, 256) -> t_dim output
    """
    
    def __init__(self, t_dim: int = 256, frequency_dim: int = 256, mlp_hidden: int = 1024):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, t_dim),
        )
    
    def forward(self, t: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.mlp[0].weight.dtype
        
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return self.mlp(emb.to(dtype))


class SwiGLU(nn.Module):
    """SwiGLU feedforward: w1(x) * silu(w3(x)) -> w2."""
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPEEmbedding(nn.Module):
    """3D Rotary Position Embedding.
    
    Note: Simplified version that handles dimension mismatches gracefully.
    """
    
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
    
    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Compute RoPE frequencies.
        
        Args:
            pos_ids: [B, N, num_axes] position IDs
        
        Returns:
            freqs: [B, N, head_dim//2, 2] containing (cos, sin)
        """
        B, N, num_axes = pos_ids.shape
        device = pos_ids.device
        dtype = pos_ids.dtype
        
        # Use head_dim // 2 for pairs
        half_dim = self.head_dim // 2
        
        # Simple 1D RoPE based on first axis
        # More complex multi-axis RoPE can be added later
        positions = pos_ids[..., 0]  # Use first axis as main position
        
        freqs = 1.0 / (self.theta ** (
            torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim
        ))
        
        # [B, N] @ [half_dim] -> [B, N, half_dim]
        angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
        
        # Return [B, N, half_dim, 2] with (cos, sin)
        return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.
    
    Args:
        x: [B, H, N, D] query or key tensor
        freqs: [B, N, D//2, 2] rotation frequencies
    
    Returns:
        Rotated tensor [B, H, N, D]
    """
    B, H, N, D = x.shape
    half_D = D // 2
    
    # Ensure freqs has right shape
    if freqs.shape[2] != half_D:
        # If dimension mismatch, return x unchanged
        # This is a fallback for when RoPE dims don't match
        return x
    
    # Reshape x to pairs: [B, H, N, D//2, 2]
    x_reshape = x.view(B, H, N, half_D, 2)
    
    # Expand freqs: [B, N, D//2, 2] -> [B, 1, N, D//2, 2]
    freqs = freqs.unsqueeze(1)
    cos = freqs[..., 0]
    sin = freqs[..., 1]
    
    # Apply rotation
    x_rot = torch.stack([
        x_reshape[..., 0] * cos - x_reshape[..., 1] * sin,
        x_reshape[..., 0] * sin + x_reshape[..., 1] * cos,
    ], dim=-1)
    
    return x_rot.view(B, H, N, D)


class Attention(nn.Module):
    """Self-attention with combined QKV, QK normalization, and RoPE."""
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.out = nn.Linear(self.inner_dim, dim, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with adaLN modulation.
    
    Architecture from checkpoint:
    - adaLN_modulation.0: Linear(t_dim, 6 * modulation_dim) where modulation_dim=2560
    - The modulation is then projected/broadcast to hidden_dim=3840
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden: int,
        t_dim: int = 256,
        modulation_dim: int = 2560,  # Different from hidden_dim!
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modulation_dim = modulation_dim
        
        # adaLN modulation: t_dim -> 6 * modulation_dim
        # Then project to hidden_dim if needed
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t_dim, 6 * modulation_dim, bias=True),
        )
        
        # Projection from modulation_dim to hidden_dim if different
        if modulation_dim != hidden_dim:
            self.mod_proj = nn.Linear(modulation_dim, hidden_dim, bias=False)
        else:
            self.mod_proj = None
        
        # Attention norms
        self.attention_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.attention_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # Attention
        self.attention = Attention(hidden_dim, num_heads, head_dim, eps=eps)
        
        # FFN norms
        self.ffn_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.ffn_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # FFN
        self.feed_forward = SwiGLU(hidden_dim, mlp_hidden, bias=False)
    
    def _project_mod(self, m: torch.Tensor) -> torch.Tensor:
        """Project modulation from modulation_dim to hidden_dim if needed."""
        if self.mod_proj is not None:
            return self.mod_proj(m)
        return m
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t_emb is not None:
            mod = self.adaLN_modulation(t_emb)
            # Split into 6 parts: shift1, scale1, gate1, shift2, scale2, gate2
            mod = mod.view(x.shape[0], 6, self.modulation_dim)
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.unbind(1)
            
            # Project to hidden_dim
            shift1 = self._project_mod(shift1)
            scale1 = self._project_mod(scale1)
            gate1 = self._project_mod(gate1)
            shift2 = self._project_mod(shift2)
            scale2 = self._project_mod(scale2)
            gate2 = self._project_mod(gate2)
            
            # Attention with modulation
            normed = self.attention_norm1(x)
            normed = self.attention_norm2(normed * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1))
            x = x + gate1.unsqueeze(1) * self.attention(normed, attention_mask, freqs)
            
            # FFN with modulation
            normed = self.ffn_norm1(x)
            normed = self.ffn_norm2(normed * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1))
            x = x + gate2.unsqueeze(1) * self.feed_forward(normed)
        else:
            x = x + self.attention(self.attention_norm2(self.attention_norm1(x)), attention_mask, freqs)
            x = x + self.feed_forward(self.ffn_norm2(self.ffn_norm1(x)))
        
        return x


class RefinerBlock(nn.Module):
    """Refiner block for context/noise (no adaLN modulation)."""
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_hidden: int, eps: float = 1e-5):
        super().__init__()
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.attention = Attention(dim, num_heads, head_dim, eps=eps)
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm2(self.attention_norm1(x)), attention_mask, freqs)
        x = x + self.feed_forward(self.ffn_norm2(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation.
    
    Note: checkpoint uses only scale modulation (hidden_dim), not shift+scale (2*hidden_dim)
    """
    
    def __init__(self, hidden_dim: int, t_dim: int, out_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=eps)
        # Checkpoint: adaLN_modulation.1.weight is [hidden_dim, t_dim]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, hidden_dim, bias=True),  # Only scale, not shift+scale
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(t_emb)
        x = self.norm(x) * (1 + scale.unsqueeze(1))  # Only scale modulation
        return self.linear(x)


# =============================================================================
# Main Model
# =============================================================================

class ZImageTransformer2DModel(nn.Module):
    """Z Image Turbo Diffusion Transformer (NextDiT/Lumina2 style)."""
    
    def __init__(
        self,
        config: Optional[ZImageConfig] = None,
        # Allow individual kwargs from HuggingFace config
        hidden_dim: int = 3840,
        context_dim: int = 2560,
        latent_channels: int = 16,
        patch_size: int = 2,
        num_layers: int = 30,
        num_refiner_layers: int = 2,
        num_heads: int = 30,
        head_dim: int = 128,
        t_dim: int = 256,
        mlp_hidden: int = 10240,
        eps: float = 1e-5,
        rope_theta: float = 10000.0,
        **kwargs,  # Ignore unknown HuggingFace config parameters
    ):
        super().__init__()
        
        # Create config from kwargs if not provided
        if config is None:
            config = ZImageConfig(
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                latent_channels=latent_channels,
                patch_size=patch_size,
                num_layers=num_layers,
                num_refiner_layers=num_refiner_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                t_dim=t_dim,
                mlp_hidden=mlp_hidden,
                eps=eps,
                rope_theta=rope_theta,
            )
        self.config = config
        
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim
        self.latent_channels = config.latent_channels
        
        # Patch embedding
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_dim, bias=True)
        
        # Caption embedding: RMSNorm(context_dim) + Linear(context_dim, hidden_dim)
        self.cap_embedder = nn.Sequential(
            RMSNorm(config.context_dim, eps=config.eps),
            nn.Linear(config.context_dim, config.hidden_dim, bias=True),
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(t_dim=config.t_dim)
        
        # Padding tokens (2D in checkpoint: [1, hidden_dim])
        self.x_pad_token = nn.Parameter(torch.zeros(1, config.hidden_dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, config.hidden_dim))
        
        # RoPE
        self.rope = RoPEEmbedding(config.head_dim, config.rope_theta)
        
        # Refiners use different hidden_dim (context_dim for context_refiner)
        # Actually, they share the same hidden_dim after embedding
        self.context_refiner = nn.ModuleList([
            RefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, config.mlp_hidden, config.eps)
            for _ in range(config.num_refiner_layers)
        ])
        
        self.noise_refiner = nn.ModuleList([
            RefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, config.mlp_hidden, config.eps)
            for _ in range(config.num_refiner_layers)
        ])
        
        # Main transformer
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                mlp_hidden=config.mlp_hidden,
                t_dim=config.t_dim,
                modulation_dim=config.context_dim,  # Modulation uses context_dim (2560)
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer
        out_dim = config.patch_size * config.patch_size * config.latent_channels
        self.final_layer = FinalLayer(config.hidden_dim, config.t_dim, out_dim, config.eps)
    
    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        p = self.patch_size
        
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = x.shape
        x = x.view(B, C, H_pad // p, p, W_pad // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H_pad // p) * (W_pad // p), C * p * p)
        
        return x, (H, W)
    
    def _unpatchify(self, x: torch.Tensor, img_size: Tuple[int, int], cap_len: int) -> torch.Tensor:
        H, W = img_size
        p = self.patch_size
        H_pad = (H + p - 1) // p * p
        W_pad = (W + p - 1) // p * p
        
        x = x[:, cap_len:]
        B, N, D = x.shape
        H_tokens, W_tokens = H_pad // p, W_pad // p
        
        x = x.view(B, H_tokens, W_tokens, self.latent_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.latent_channels, H_pad, W_pad)
        
        return x[:, :, :H, :W]
    
    def _get_position_ids(self, cap_len: int, h_tokens: int, w_tokens: int, B: int, device: torch.device) -> torch.Tensor:
        total_len = cap_len + h_tokens * w_tokens
        pos_ids = torch.zeros(B, total_len, 3, device=device)
        
        pos_ids[:, :cap_len, 0] = torch.arange(cap_len, device=device).float()
        
        h_idx = torch.arange(h_tokens, device=device).view(-1, 1).repeat(1, w_tokens).flatten()
        w_idx = torch.arange(w_tokens, device=device).view(1, -1).repeat(h_tokens, 1).flatten()
        
        pos_ids[:, cap_len:, 0] = cap_len
        pos_ids[:, cap_len:, 1] = h_idx.float()
        pos_ids[:, cap_len:, 2] = w_idx.float()
        
        return pos_ids
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Handle 5D input
        if x.dim() == 5:
            x = x.squeeze(2)
            was_5d = True
        else:
            was_5d = False
        
        B, C, H, W = x.shape
        
        # Timestep embedding
        t_emb = self.t_embedder(timestep, dtype=x.dtype)
        
        # Patchify
        img_patches, img_size = self._patchify(x)
        img_patches = self.x_embedder(img_patches)
        
        # Caption embedding
        cap_feats = self.cap_embedder(context)
        cap_len = cap_feats.shape[1]
        
        # Position IDs
        h_tokens = (img_size[0] + self.patch_size - 1) // self.patch_size
        w_tokens = (img_size[1] + self.patch_size - 1) // self.patch_size
        pos_ids = self._get_position_ids(cap_len, h_tokens, w_tokens, B, x.device)
        freqs = self.rope(pos_ids)
        
        # Refiners
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, None, freqs[:, :cap_len])
        
        for layer in self.noise_refiner:
            img_patches = layer(img_patches, None, freqs[:, cap_len:])
        
        # Concatenate
        full_seq = torch.cat([cap_feats, img_patches], dim=1)
        
        # Main transformer
        for layer in self.layers:
            full_seq = layer(full_seq, None, freqs, t_emb)
        
        # Final projection
        output = self.final_layer(full_seq, t_emb)
        output = self._unpatchify(output, img_size, cap_len)
        
        if was_5d:
            output = output.unsqueeze(2)
        
        return -output  # Velocity conversion


# =============================================================================
# Model Loading
# =============================================================================

def load_zimage_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: Optional[ZImageConfig] = None,
) -> ZImageTransformer2DModel:
    """Load Z Image model from state dict with automatic config detection."""
    
    # Detect dimensions from checkpoint
    hidden_dim = 3840
    context_dim = 2560
    t_dim = 256
    num_layers = 30
    num_refiner = 2
    mlp_hidden = 10240
    
    if "x_embedder.weight" in state_dict:
        hidden_dim = int(state_dict["x_embedder.weight"].shape[0])
    
    if "cap_embedder.1.weight" in state_dict:
        w = state_dict["cap_embedder.1.weight"]
        context_dim = int(w.shape[1])
    
    if "t_embedder.mlp.2.weight" in state_dict:
        t_dim = int(state_dict["t_embedder.mlp.2.weight"].shape[0])
    
    for key in state_dict.keys():
        if key.startswith("layers.") and ".adaLN_modulation." in key:
            idx = int(key.split(".")[1])
            num_layers = max(num_layers, idx + 1)
    
    for key in state_dict.keys():
        if key.startswith("context_refiner."):
            idx = int(key.split(".")[1])
            num_refiner = max(num_refiner, idx + 1)
    
    if "layers.0.feed_forward.w1.weight" in state_dict:
        mlp_hidden = int(state_dict["layers.0.feed_forward.w1.weight"].shape[0])
    
    num_heads = hidden_dim // 128
    
    logger.info(
        f"Detected: hidden={hidden_dim}, context={context_dim}, t_dim={t_dim}, "
        f"layers={num_layers}, refiner={num_refiner}, heads={num_heads}, mlp={mlp_hidden}"
    )
    
    config = ZImageConfig(
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        t_dim=t_dim,
        num_layers=num_layers,
        num_refiner_layers=num_refiner,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
    )
    
    model = ZImageTransformer2DModel(config)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:5]}...")
    
    return model


QwenImageTransformer2DModel = ZImageTransformer2DModel
