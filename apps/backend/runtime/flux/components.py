from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from apps.backend.runtime.attention import attention_function
from apps.backend.runtime import utils
from .geometry import apply_rotary_embeddings, build_rotary_frequencies

logger = logging.getLogger("backend.runtime.flux")


class RMSNorm(nn.Module):
    """Root mean square layer norm with dtype-aware casting."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale.dtype != x.dtype:
            self.scale = utils.tensor2parameter(self.scale.to(dtype=x.dtype))
        second_moment = torch.mean(x * x, dim=-1, keepdim=True)
        norm = torch.rsqrt(second_moment + self.eps)
        return x * norm * self.scale


class QKNorm(nn.Module):
    """Applies independent RMSNorm to query/key pairs with aligned dtype casts."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del v
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(k), k.to(q)


class SelfAttention(nn.Module):
    """Self-attention with shared rotary positional embedding support."""

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner_dim = dim * 3
        self.qkv = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rotary_freqs: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        b, seq_len, _ = qkv.shape
        qkv = qkv.view(b, seq_len, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        q, k = apply_rotary_embeddings(q, k, rotary_freqs)
        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        return self.proj(attn)


class ModulationMLP(nn.Module):
    """Modulation MLP producing shift/scale/gate tuples."""

    def __init__(self, dim: int, *, double: bool) -> None:
        super().__init__()
        multiplier = 6 if double else 3
        self.multiplier = multiplier
        self.lin = nn.Linear(dim, multiplier * dim, bias=True)
        self.double = double

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, ...]:
        activations = self.lin(nn.functional.silu(vec))
        chunks = activations[:, None, :].chunk(self.multiplier, dim=-1)
        return chunks


class DoubleStreamBlock(nn.Module):
    """Flux double-stream block with JOINT attention between text and image streams."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, *, qkv_bias: bool) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Modulation
        self.img_mod = ModulationMLP(hidden_size, double=True)
        self.txt_mod = ModulationMLP(hidden_size, double=True)
        
        # Attention components (shared QKV projection style)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # img_attn and txt_attn share structure but have separate weights
        self.img_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        
        # MLP
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, rotary_freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)

        # Prepare for attention with modulation
        img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        txt_modulated = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift

        # Compute QKV for both streams
        img_qkv = self.img_attn.qkv(img_modulated)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        
        b, img_len, _ = img_qkv.shape
        txt_len = txt_qkv.shape[1]
        head_dim = self.hidden_size // self.num_heads
        
        # Reshape to (B, seq, 3, heads, head_dim) then permute to (3, B, heads, seq, head_dim)
        img_qkv = img_qkv.view(b, img_len, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        txt_qkv = txt_qkv.view(b, txt_len, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]
        txt_q, txt_k, txt_v = txt_qkv[0], txt_qkv[1], txt_qkv[2]
        
        # Apply QK normalization
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        
        # JOINT ATTENTION: Concatenate txt and img for cross-attention
        # Order: txt first, then img (to match rotary_freqs which is built as cat(txt_ids, img_ids))
        q = torch.cat((txt_q, img_q), dim=2)  # (B, heads, txt_len+img_len, head_dim)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        # Apply rotary embeddings
        q, k = apply_rotary_embeddings(q, k, rotary_freqs)
        
        # Compute attention
        attn_out = attention_function(q, k, v, self.num_heads, skip_reshape=True)
        
        # Split the output back into txt and img
        txt_attn_out = attn_out[:, :txt_len, :]
        img_attn_out = attn_out[:, txt_len:, :]
        
        # Apply output projections
        img_attn_out = self.img_attn.proj(img_attn_out)
        txt_attn_out = self.txt_attn.proj(txt_attn_out)
        
        # Residual + gate
        img = img + img_mod1_gate * img_attn_out
        txt = txt + txt_mod1_gate * txt_attn_out

        # MLP with modulation
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)
        txt = utils.fp16_fix(txt)
        return img, txt


class SingleStreamBlock(nn.Module):
    """Flux single-stream block operating on concatenated tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden)
        self.linear2 = nn.Linear(hidden_size + mlp_hidden, hidden_size)
        self.modulation = ModulationMLP(hidden_size, double=False)
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, vec: torch.Tensor, rotary_freqs: torch.Tensor) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec)
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        q, k = apply_rotary_embeddings(q, k, rotary_freqs)
        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        x = x + mod_gate * output
        x = utils.fp16_fix(x)
        return x


class LastLayer(nn.Module):
    """Final output linear layer with adaptive layer norm modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm(x) + shift[:, None, :]
        return self.linear(x)
