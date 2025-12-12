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

_DEFAULT_ZIMAGE_AXES_DIMS: tuple[int, int, int] = (32, 48, 48)

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
    qk_norm: bool = True
    qkv_bias: bool = False
    out_bias: bool = False
    t_dim: int = 256  # Timestep embedding dimension
    mlp_hidden: int = 10240
    eps: float = 1e-5
    # HF config: apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/transformer/config.json
    rope_theta: float = 256.0
    axes_dims: tuple[int, int, int] = (32, 48, 48)  # Must sum to head_dim
    t_scale: float = 1000.0
    
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
        args = t.float()[:, None] * freqs[None, :]
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
    """3D Rotary Position Embedding using ComfyUI's flux/math.py rotation-matrix format.
    
    This matches ComfyUI's EmbedND from comfy/ldm/flux/layers.py which is
    also used by NextDiT-style diffusion transformers.
    
    The output is a rotation matrix of shape [B, 1, N, head_dim//2, 2, 2]
    that can be applied to Q/K tensors via the apply_rope function.
    """
    
    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        axes_dims: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # axes_dims is specified in full head_dim units (must sum to head_dim),
        # matching Hugging Face config keys ("axes_dims") and ComfyUI's EmbedND.
        if axes_dims is None:
            # Z Image Turbo default: 1/4 time axis, rest split across spatial axes.
            # head_dim=128 -> (32, 48, 48)
            time_dim = max(2, head_dim // 4)
            time_dim -= time_dim % 2
            spatial = (head_dim - time_dim) // 2
            spatial -= spatial % 2
            last = head_dim - time_dim - spatial
            axes_dims = (time_dim, spatial, last)
        if sum(int(v) for v in axes_dims) != int(head_dim):
            raise ValueError(f"axes_dims must sum to head_dim={head_dim}; got {axes_dims}")
        if any(int(v) % 2 != 0 for v in axes_dims):
            raise ValueError(f"axes_dims entries must be even; got {axes_dims}")
        self.axes_dims = tuple(int(v) for v in axes_dims)
    
    def _rope_single_axis(self, pos: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute RoPE for a single axis.
        
        Args:
            pos: [B, N] positions for this axis
            dim: dimension to use for this axis
            
        Returns:
            Rotation matrices [B, N, dim//2, 2, 2]
        """
        assert dim % 2 == 0
        device = pos.device
        
        # Match ComfyUI's flux/math.py numeric path: build omega in float64 for stability,
        # then return float32 rotation matrices.
        scale = torch.linspace(
            0,
            (dim - 2) / dim,
            steps=dim // 2,
            dtype=torch.float64,
            device=device,
        )
        omega = 1.0 / (float(self.theta) ** scale)
        
        # [B, N] x [dim//2] -> [B, N, dim//2]
        out = pos.unsqueeze(-1).to(dtype=torch.float32) * omega.unsqueeze(0).unsqueeze(0)
        
        # Build rotation matrix [cos, -sin; sin, cos]
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        
        # Stack as [B, N, dim//2, 4] then reshape to [B, N, dim//2, 2, 2]
        stacked = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        rot_matrix = stacked.view(*pos.shape, dim // 2, 2, 2)
        
        return rot_matrix.to(dtype=torch.float32)
    
    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Compute RoPE rotation matrices for 3D positions.
        
        Args:
            pos_ids: [B, N, num_axes] position IDs (Time, Height, Width)
        
        Returns:
            freqs: [B, 1, N, head_dim//2, 2, 2] rotation matrices
        """
        B, N, num_axes = pos_ids.shape
        device = pos_ids.device
        
        # Compute rotation matrix for each axis and concatenate
        emb_list = []
        for i in range(min(num_axes, len(self.axes_dims))):
            axis_pos = pos_ids[..., i]  # [B, N]
            axis_dim = self.axes_dims[i]
            axis_emb = self._rope_single_axis(axis_pos, axis_dim)  # [B, N, dim_i//2, 2, 2]
            emb_list.append(axis_emb)
        
        # Concatenate along the dimension axis: [B, N, sum(axes_dim), 2, 2]
        emb = torch.cat(emb_list, dim=-3)
        
        # Add head dimension: [B, 1, N, head_dim//2, 2, 2]
        return emb.unsqueeze(1)


def apply_rope_single(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a single tensor.
    
    This matches ComfyUI's apply_rope1 from comfy/ldm/flux/math.py.
    
    Args:
        x: [B, N, H, D] query or key tensor where D = head_dim
        freqs: [B, N, 1, D//2, 2, 2] rotation matrices (after movedim)
    
    Returns:
        Rotated tensor [B, N, H, D]
    """
    # Reshape x to [B, N, H, D//2, 1, 2] 
    x_reshaped = x.to(dtype=freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    
    # freqs is [B, N, 1, D//2, 2, 2], broadcasts over H dimension (dim 2)
    # Compute: freqs[..., 0] * x[..., 0] + freqs[..., 1] * x[..., 1]
    x_out = freqs[..., 0] * x_reshaped[..., 0]
    x_out = x_out + freqs[..., 1] * x_reshaped[..., 1]
    
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope_pair(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple:
    """Apply rotary position embedding to query and key.
    
    This matches ComfyUI's apply_rope from comfy/ldm/flux/math.py.
    
    Args:
        q: [B, N, H, D] query tensor
        k: [B, N, H, D] key tensor
        freqs: [B, 1, N, D//2, 2, 2] rotation matrices from RoPEEmbedding
    
    Returns:
        Tuple of rotated (q, k) tensors [B, N, H, D]
    """
    # movedim(1, 2) to get [B, N, 1, D//2, 2, 2] for broadcasting
    freqs = freqs.movedim(1, 2)
    return apply_rope_single(q, freqs), apply_rope_single(k, freqs)


class Attention(nn.Module):
    """Self-attention with combined QKV, QK normalization, and RoPE.
    
    Matches ComfyUI's JointAttention dimension ordering for RoPE.
    Q/K/V are [B, N, H, D] during RoPE, then [B, H, N, D] for SDPA.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        *,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        out_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.out = nn.Linear(self.inner_dim, dim, bias=out_bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        
        # QKV projection and reshape to [B, N, 3, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Split into [B, N, H, D] each - matching ComfyUI JointAttention
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # QK normalization (operates on last dim = head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE in [B, N, H, D] format (matching ComfyUI)
        if freqs is not None:
            q, k = apply_rope_pair(q, k, freqs)
        
        # Transpose to [B, H, N, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with adaLN modulation.
    
    Architecture matching ComfyUI's NextDiT for z_image_modulation:
    - adaLN_modulation.0: Linear(min(dim, 256), 4 * dim) = Linear(256, 4*3840)
    - Outputs 4 modulation values: scale_msa, gate_msa, scale_mlp, gate_mlp
    - Uses tanh gating like NextDiT
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden: int,
        t_dim: int = 256,
        eps: float = 1e-5,
        **kwargs,  # Ignore extra args like modulation_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # z_image_modulation: Linear(min(dim, 256), 4 * dim)
        # For dim=3840, min(3840, 256) = 256, output = 4*3840 = 15360
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t_dim, 4 * hidden_dim, bias=True),
        )
        
        # Attention norms
        self.attention_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.attention_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # Attention
        self.attention = Attention(
            hidden_dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        
        # FFN norms
        self.ffn_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.ffn_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # FFN
        self.feed_forward = SwiGLU(hidden_dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t_emb is not None:
            # Get modulation values (4 * hidden_dim total)
            mod = self.adaLN_modulation(t_emb)  # [B, 4 * hidden_dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            
            # Attention with modulation (tanh gating like NextDiT)
            normed = self.attention_norm1(x)
            normed = normed * (1 + scale_msa.unsqueeze(1))
            attn_out = self.attention(normed, attention_mask, freqs)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(attn_out)
            
            # FFN with modulation
            normed = self.ffn_norm1(x)
            normed = normed * (1 + scale_mlp.unsqueeze(1))
            ffn_out = self.feed_forward(normed)
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(ffn_out)
        else:
            x = x + self.attention(self.attention_norm2(self.attention_norm1(x)), attention_mask, freqs)
            x = x + self.feed_forward(self.ffn_norm2(self.ffn_norm1(x)))
        
        return x


class RefinerBlock(nn.Module):
    """Refiner block for context (no adaLN modulation)."""
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_hidden: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.attention = Attention(
            dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,  # Ignored for context_refiner
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm2(self.attention_norm1(x)), attention_mask, freqs)
        x = x + self.feed_forward(self.ffn_norm2(self.ffn_norm1(x)))
        return x


class NoiseRefinerBlock(nn.Module):
    """Noise refiner block with adaLN modulation for Z Image.
    
    Unlike context_refiner, noise_refiner uses timestep-conditioned modulation
    similar to the main transformer blocks but with only 4 modulation values
    (scale_msa, gate_msa, scale_mlp, gate_mlp).
    """
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_hidden: int, 
                 t_dim: int = 256, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.dim = dim
        
        # adaLN modulation: t_dim -> 4 * dim (scale_msa, gate_msa, scale_mlp, gate_mlp)
        # For z_image_modulation, input is min(dim, 256) = 256
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t_dim, 4 * dim, bias=True),
        )
        
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.attention = Attention(
            dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t_emb is not None:
            # Get modulation values
            mod = self.adaLN_modulation(t_emb)  # [B, 4 * dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            
            # Attention with modulation (tanh gating like ComfyUI's NextDiT)
            normed = self.attention_norm1(x)
            normed = normed * (1 + scale_msa.unsqueeze(1))
            attn_out = self.attention(normed, attention_mask, freqs)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(attn_out)
            
            # FFN with modulation
            normed = self.ffn_norm1(x)
            normed = normed * (1 + scale_mlp.unsqueeze(1))
            ffn_out = self.feed_forward(normed)
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(ffn_out)
        else:
            x = x + self.attention(self.attention_norm2(self.attention_norm1(x)), attention_mask, freqs)
            x = x + self.feed_forward(self.ffn_norm2(self.ffn_norm1(x)))
        
        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation.
    
    Uses LayerNorm with elementwise_affine=False (no learnable params) to match
    the GGUF checkpoint structure from ComfyUI's NextDiT. The GGUF doesn't have
    weights for this norm layer because ComfyUI uses non-affine LayerNorm.
    """
    
    def __init__(self, hidden_dim: int, t_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        # ComfyUI uses LayerNorm(elementwise_affine=False), so no weight/bias in norm
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        # Checkpoint: adaLN_modulation.1.weight is [hidden_dim, t_dim]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, hidden_dim, bias=True),  # Only scale, not shift+scale
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(t_emb)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1))  # Only scale modulation
        return self.linear(x)



# =============================================================================
# Main Model
# =============================================================================

class ZImageTransformer2DModel(nn.Module):
    """Z Image Turbo Diffusion Transformer (NextDiT/Lumina2 style)."""
    
    def __init__(
        self,
        hidden_dim: int = 3840,
        context_dim: int = 2560,
        latent_channels: int = 16,
        patch_size: int = 2,
        num_layers: int = 30,
        num_refiner_layers: int = 2,
        num_heads: int = 30,
        head_dim: int = 128,
        qkv_bias: bool = False,  # Lumina uses bias=False for QKV
        out_bias: bool = False,
        t_dim: int = 256,
        mlp_hidden: int = 10240,
        eps: float = 1e-5,
        qk_norm: bool = True,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] | None = None,
        time_scale: float | None = None,
        config: Optional[ZImageConfig] = None,
        **kwargs,  # Ignore unknown HuggingFace config parameters
    ):
        super().__init__()

        # HuggingFace config compatibility: map common keys when present.
        # We keep canonical names (hidden_dim/context_dim/...) internally.
        if config is None:
            hidden_dim = int(kwargs.pop("dim", hidden_dim))
            context_dim = int(kwargs.pop("cap_feat_dim", context_dim))
            latent_channels = int(kwargs.pop("in_channels", latent_channels))
            if "all_patch_size" in kwargs and patch_size == 2:
                raw_patch = kwargs.pop("all_patch_size")
                if isinstance(raw_patch, (list, tuple)) and raw_patch:
                    patch_size = int(raw_patch[0])
            num_layers = int(kwargs.pop("n_layers", num_layers))
            num_refiner_layers = int(kwargs.pop("n_refiner_layers", num_refiner_layers))
            num_heads = int(kwargs.pop("n_heads", num_heads))
            eps = float(kwargs.pop("norm_eps", eps))
            qk_norm = bool(kwargs.pop("qk_norm", qk_norm))
            if axes_dims is None and "axes_dims" in kwargs:
                raw_axes = kwargs.pop("axes_dims")
                if isinstance(raw_axes, (list, tuple)) and len(raw_axes) == 3:
                    axes_dims = (int(raw_axes[0]), int(raw_axes[1]), int(raw_axes[2]))
            if time_scale is None and "t_scale" in kwargs:
                time_scale = float(kwargs.pop("t_scale"))

        # HF config key is "t_scale"; default to 1000.0 for Z Image Turbo.
        if time_scale is None:
            time_scale = 1000.0

        # Create config from resolved values if not provided
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
                axes_dims=axes_dims or _DEFAULT_ZIMAGE_AXES_DIMS,
                t_scale=float(time_scale),
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
            )
        self.config = config
        self.time_scale = float(getattr(config, "t_scale", float(time_scale)))
        
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
        
        # RoPE (axes_dims must sum to head_dim)
        self.rope = RoPEEmbedding(config.head_dim, config.rope_theta, axes_dims=getattr(config, "axes_dims", None))
        
        # Refiners use different hidden_dim (context_dim for context_refiner)
        # Actually, they share the same hidden_dim after embedding
        self.context_refiner = nn.ModuleList([
            RefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, config.mlp_hidden, config.eps,
                         qk_norm=config.qk_norm, qkv_bias=config.qkv_bias, out_bias=config.out_bias)
            for _ in range(config.num_refiner_layers)
        ])
        
        self.noise_refiner = nn.ModuleList([
            NoiseRefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, 
                              config.mlp_hidden, config.t_dim, config.eps,
                              qk_norm=config.qk_norm, qkv_bias=config.qkv_bias, out_bias=config.out_bias)
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
                eps=config.eps,
                qkv_bias=config.qkv_bias,
                out_bias=config.out_bias,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer
        out_dim = config.patch_size * config.patch_size * config.latent_channels
        self.final_layer = FinalLayer(config.hidden_dim, config.t_dim, out_dim, config.eps)
        
        self.cnt = 0
    
    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        # Flatten x: [B, C, H, W] -> [B, H*W, C] (wrong)
        # Patchify: split into patches of size p
        # [B, C, H, W] -> [B, (H/p)*(W/p), C*p*p]
        
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Image size ({H}, {W}) must be divisible by patch size {p}"
        
        h_tokens = H // p
        w_tokens = W // p
        
        # Unfold/Reshape approach
        # [B, C, H, W] -> [B, C, h, p, w, p]
        x = x.reshape(B, C, h_tokens, p, w_tokens, p)
        # Permute to [B, h, w, p, p, C]
        x = x.permute(0, 2, 4, 3, 5, 1)
        # Flatten to [B, h*w, p*p*C]
        x = x.reshape(B, h_tokens * w_tokens, p * p * C)
        
        return x, (H, W)

    def _unpatchify(self, x: torch.Tensor, original_size: Tuple[int, int], cap_len: int) -> torch.Tensor:
        # x starts with caption tokens, remove them
        # x: [B, N, D_out]
        # D_out = patch_size * patch_size * latent_channels
        
        x = x[:, cap_len:, :]
        
        H, W = original_size
        p = self.patch_size
        h_tokens = H // p
        w_tokens = W // p
        B = x.shape[0]
        C = self.latent_channels
        
        # [B, h*w, p*p*C] -> [B, h, w, p, p, C]
        x = x.reshape(B, h_tokens, w_tokens, p, p, C)
        # Permute to [B, C, h, p, w, p]
        x = x.permute(0, 5, 1, 3, 2, 4)
        # Reshape to [B, C, H, W]
        x = x.reshape(B, C, H, W)
        
        return x
    
    def _get_position_ids(self, cap_len: int, h_tokens: int, w_tokens: int, B: int, device: torch.device) -> torch.Tensor:
        total_len = cap_len + h_tokens * w_tokens
        pos_ids = torch.zeros(B, total_len, 3, device=device, dtype=torch.int32)
        
        # Caption tokens occupy the time axis [0..cap_len-1]
        pos_ids[:, :cap_len, 0] = torch.arange(cap_len, device=device, dtype=torch.int32)
        
        h_idx = torch.arange(h_tokens, device=device, dtype=torch.int32).view(-1, 1).repeat(1, w_tokens).flatten()
        w_idx = torch.arange(w_tokens, device=device, dtype=torch.int32).view(1, -1).repeat(h_tokens, 1).flatten()
        
        # Image tokens share a constant time axis = cap_len
        pos_ids[:, cap_len:, 0] = int(cap_len)
        pos_ids[:, cap_len:, 1] = h_idx
        pos_ids[:, cap_len:, 2] = w_idx
        
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
        
        # Patchify image input
        img_patches, img_size = self._patchify(x)
        img_patches = self.x_embedder(img_patches)
        
        # Timestep embedding
        # NextDiT/Lumina2 convention: invert timestep for flow matching
        t_inv = 1.0 - timestep
        t_emb = self.t_embedder(t_inv * self.time_scale, dtype=x.dtype)
        
        # Caption embedding
        cap_feats = self.cap_embedder(context)
        
        # DEBUG LOGS
        if self.cnt < 3: # Only log first few steps
            logger.info(f"[zimage-debug] timestep (sigma): {timestep[0]:.4f}")
            logger.info(f"[zimage-debug] t_inv (1-sigma): {t_inv[0]:.4f}")
            logger.info(f"[zimage-debug] t_emb={t_emb[0, :8]}... range=[{t_emb.min():.2f}, {t_emb.max():.2f}]")
            logger.info(f"[zimage-debug] img_patches range=[{img_patches.min():.2f}, {img_patches.max():.2f}] mean={img_patches.mean():.4f}")
            logger.info(f"[zimage-debug] cap_feats range=[{cap_feats.min():.2f}, {cap_feats.max():.2f}] mean={cap_feats.mean():.4f}")
            self.cnt += 1
            
        cap_len = cap_feats.shape[1]
        
        # Position IDs
        B = img_patches.shape[0]
        h_tokens = (img_size[0] + self.patch_size - 1) // self.patch_size
        w_tokens = (img_size[1] + self.patch_size - 1) // self.patch_size
        pos_ids = self._get_position_ids(cap_len, h_tokens, w_tokens, B, x.device)
        freqs = self.rope(pos_ids)
        
        # Refiners
        # freqs is now [B, 1, N, D//2, 2, 2], slice on dimension 2 (N)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, None, freqs[:, :, :cap_len])
        
        for layer in self.noise_refiner:
            img_patches = layer(img_patches, None, freqs[:, :, cap_len:], t_emb)
        
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
        
        # DEBUG: Log output statistics before negation
        if self.cnt < 3:
            logger.info(f"[zimage-debug] output BEFORE negation: range=[{output.min():.4f}, {output.max():.4f}] mean={output.mean():.4f} norm={output.norm():.2f}")
        
        result = -output  # Velocity conversion (negative velocity for flow matching)
        
        if self.cnt < 3:
            logger.info(f"[zimage-debug] output AFTER negation: range=[{result.min():.4f}, {result.max():.4f}] mean={result.mean():.4f} norm={result.norm():.2f}")
        
        return result


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
    
    model = ZImageTransformer2DModel(config=config)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:5]}...")
    
    return model


QwenImageTransformer2DModel = ZImageTransformer2DModel
