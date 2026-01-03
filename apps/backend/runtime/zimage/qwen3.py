"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen3-4B text encoder implementation used by the ZImage runtime.
This is a standalone implementation that avoids `transformers`, enabling GGUF support through the quantization system.

Architecture notes (Qwen3-4B):
- hidden_size: 2560
- intermediate_size: 9728
- num_hidden_layers: 36
- num_attention_heads: 32
- num_key_value_heads: 8 (GQA)
- RoPE theta: 1_000_000
- Q/K normalization (Gemma3 style)
- SwiGLU activation

Symbols (top-level; keep in sync; no ghosts):
- `Qwen3Config` (dataclass): Configuration defaults for Qwen3-4B (dims/layers/heads/RoPE/norm eps).
- `RMSNorm` (class): RMSNorm implementation used throughout the encoder.
- `rotate_half` (function): Helper for RoPE rotation (splits and rotates half-dims).
- `apply_rotary_pos_emb` (function): Applies rotary positional embeddings to `(q, k)` tensors.
- `RotaryEmbedding` (class): Builds rotary embedding frequency tensors for a given head dim and max positions.
- `Attention` (class): Attention module (GQA support + optional Q/K norm + SDPA).
- `MLP` (class): SwiGLU feed-forward block.
- `TransformerBlock` (class): One transformer layer (attn + MLP + residual/norm plumbing).
- `Qwen3Model` (class): Core Qwen3 transformer stack (embeddings + blocks + forward).
- `Qwen3_4B` (class): Convenience wrapper for the 4B variant (loads config, provides encode-style forward usage).
- `remap_gguf_keys` (function): Remaps GGUF state dict keys to this implementation’s expected parameter names.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("backend.runtime.zimage.qwen3")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Qwen3Config:
    """Configuration for Qwen3-4B model."""
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    head_dim: int = 128  # hidden_size // num_attention_heads = 80, but Qwen uses 128
    qkv_bias: bool = False
    use_qk_norm: bool = True  # Qwen3 uses Q/K normalization


# =============================================================================
# Core Layers
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 40960, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache cos/sin
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int = 0, ops=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        ops = ops or nn

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = ops.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = ops.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = ops.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = ops.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Q/K normalization (Qwen3 style)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # DEBUG: Check Q/K/V projections
        if not hasattr(self, '_attn_debug_logged'):
            q_nan = torch.isnan(q).any().item()
            k_nan = torch.isnan(k).any().item()
            v_nan = torch.isnan(v).any().item()
            if q_nan or k_nan or v_nan:
                self._attn_debug_logged = True
                logger.error("[attn-debug] NaN after projections! q_nan=%s k_nan=%s v_nan=%s", q_nan, k_nan, v_nan)
                logger.error("[attn-debug] q stats: mean=%.4f std=%.4f", q.float().mean().item(), q.float().std().item())
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply Q/K normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        # DEBUG: Check after Q/K norm
        q_nan = torch.isnan(q).any().item()
        k_nan = torch.isnan(k).any().item()
        if q_nan or k_nan:
            logger.error("[attn-debug] NaN after Q/K norm! q_nan=%s k_nan=%s", q_nan, k_nan)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # DEBUG: Check after RoPE
        q_nan = torch.isnan(q).any().item()
        k_nan = torch.isnan(k).any().item()
        if q_nan or k_nan:
            logger.error("[attn-debug] NaN after RoPE! q_nan=%s k_nan=%s", q_nan, k_nan)
        
        # Repeat K, V for GQA
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # DEBUG: Log shapes and stats before SDPA
        if not hasattr(self, '_sdpa_debug_logged'):
            self._sdpa_debug_logged = True
            logger.info("[sdpa-debug] Q shape=%s K shape=%s V shape=%s", q.shape, k.shape, v.shape)
            logger.info("[sdpa-debug] Q dtype=%s min=%.4f max=%.4f", q.dtype, q.float().min().item(), q.float().max().item())
            logger.info("[sdpa-debug] K dtype=%s min=%.4f max=%.4f", k.dtype, k.float().min().item(), k.float().max().item())
            logger.info("[sdpa-debug] attention_mask shape=%s dtype=%s", 
                       attention_mask.shape if attention_mask is not None else None,
                       attention_mask.dtype if attention_mask is not None else None)
            if attention_mask is not None:
                # Check mask values
                mask_finite = torch.isfinite(attention_mask)
                # Get the diagonal of the mask (positions [i,i] where token can attend to itself)
                # For 4D mask [batch, 1, seq, seq], we need attention_mask[0, 0].diagonal()
                if len(attention_mask.shape) == 4:
                    mask_diag = attention_mask[0, 0].diagonal()
                else:
                    mask_diag = attention_mask.diagonal()
                logger.info("[sdpa-debug] mask finite=%d/%d diag[:5]=%s",
                           mask_finite.sum().item(), attention_mask.numel(),
                           mask_diag[:5].tolist())
        
        # Scaled dot-product attention
        # IMPORTANT: Always use is_causal=False! The causal mask is already in attention_mask.
        # This follows the ComfyUI pattern which constructs causal mask manually.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,  # Causal mask is in attention_mask, not is_causal flag
        )
        
        # DEBUG: Check after SDPA
        if torch.isnan(attn_output).any():
            logger.error("[attn-debug] NaN after SDPA! output_shape=%s", attn_output.shape)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, config: Qwen3Config, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int, ops=None):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, ops=ops)
        self.mlp = MLP(config, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # DEBUG: Check after input_layernorm (only first time NaN appears)
        if torch.isnan(hidden_states).any() and not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            logger.error("[block-debug] NaN after input_layernorm! residual_nan=%s", torch.isnan(residual).any())
            return residual + hidden_states  # Return early to pinpoint
        
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        
        # DEBUG: Check after attention
        if torch.isnan(hidden_states).any() and not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            logger.error("[block-debug] NaN after self_attn!")
            return residual + hidden_states
        
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # DEBUG: Check after post_attention_layernorm
        if torch.isnan(hidden_states).any() and not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            logger.error("[block-debug] NaN after post_attention_layernorm!")
            return residual + hidden_states
        
        hidden_states = self.mlp(hidden_states)
        
        # DEBUG: Check after MLP
        if torch.isnan(hidden_states).any() and not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            logger.error("[block-debug] NaN after mlp!")
        
        hidden_states = residual + hidden_states
        
        return hidden_states


# =============================================================================
# Main Model
# =============================================================================

class Qwen3Model(nn.Module):
    """Qwen3 transformer model (without LM head)."""
    
    def __init__(self, config: Qwen3Config, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        ops = ops or nn
        
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        
        # Create causal mask if attention_mask provided
        causal_mask = None
        if attention_mask is not None:
            # Convert attention_mask to causal format
            batch_size, seq_len = attention_mask.shape
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            # Combine with padding mask
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype)) * float("-inf")
            causal_mask = causal_mask.unsqueeze(0) + padding_mask
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, causal_mask)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return hidden_states, all_hidden_states


class Qwen3_4B(nn.Module):
    """Qwen3-4B for text encoding.
    
    This is a wrapper that provides the same interface as transformers models
    this is a wrapper that provides the same interface as transformers models
    but uses our native implementation.
    """
    
    def __init__(self, config: Optional[Qwen3Config] = None, dtype=None, device=None, ops=None):
        super().__init__()
        config = config or Qwen3Config()
        self.config = config
        self.model = Qwen3Model(config, ops=ops)
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.model.set_input_embeddings(value)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_output: Optional[int] = None,
        final_layer_norm_intermediate: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional intermediate output.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            intermediate_output: Layer index to extract intermediate output from
            final_layer_norm_intermediate: Whether to apply final norm to intermediate
        
        Returns:
            Tuple of (final_hidden_states, intermediate_hidden_states)
        """
        # Get embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        intermediate = None
        
        # DEBUG: Check embedding output
        if torch.isnan(hidden_states).any():
            logger.error("[qwen3-debug] NaN detected AFTER embedding! input_ids=%s hidden_states=%s", 
                        input_ids.shape if input_ids is not None else None, 
                        hidden_states.shape)
            # Check embedding weights
            emb_weight = self.model.embed_tokens.weight
            if hasattr(emb_weight, 'data'):
                emb_data = emb_weight.data
                logger.error("[qwen3-debug] Embedding weight type: %s, has_nan: %s", 
                            type(emb_weight), torch.isnan(emb_data).any() if hasattr(emb_data, 'isnan') else 'unknown')
        else:
            logger.info("[qwen3-debug] Embedding output OK: shape=%s mean=%.4f", 
                       hidden_states.shape, hidden_states.float().mean().item())
        
        # Create causal mask - ALWAYS create it (following ComfyUI pattern)
        # The mask is added to attention scores before softmax
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Create causal mask: 0 for positions that CAN attend, -inf for positions that CANNOT
        # Position (i, j) = 0 if i >= j (can attend to self and past)
        # Position (i, j) = -inf if i < j (cannot attend to future)
        causal_mask = torch.zeros(seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1),
            float("-inf")
        )
        
        # Reshape to 4D for SDPA: [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask is [batch, seq_len] with 1 for valid, 0 for padding
            # We want to fill padding positions (where mask is 0) with -inf
            # Expand to [batch, 1, 1, seq_len]
            expanded_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            # Use masked_fill: where mask is 0 (False in bool conversion? No, need explicit check)
            # CAREFUL: (attention_mask == 0) creates boolean mask of PADDING positions
            is_padding = (expanded_mask == 0)
            causal_mask = causal_mask.masked_fill(is_padding, float("-inf"))
        
        # Handle negative layer index
        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(self.model.layers) + intermediate_output
        
        # Forward through layers
        nan_detected_at = None
        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, causal_mask)
            
            # DEBUG: Check for NaN after each layer (only first few and when NaN appears)
            if nan_detected_at is None and torch.isnan(hidden_states).any():
                nan_detected_at = i
                logger.error("[qwen3-debug] NaN FIRST detected at layer %d", i)
            
            if intermediate_output is not None and i == intermediate_output:
                intermediate = hidden_states.clone()
        
        # Apply final norm
        hidden_states = self.model.norm(hidden_states)
        
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.model.norm(intermediate)
        
        # DEBUG: Final output check
        if torch.isnan(hidden_states).any():
            logger.error("[qwen3-debug] NaN in FINAL output! first_nan_layer=%s", nan_detected_at)
        else:
            logger.info("[qwen3-debug] Final output OK: shape=%s mean=%.4f", 
                       hidden_states.shape, hidden_states.float().mean().item())
        
        return hidden_states, intermediate
    
    def load_sd(self, state_dict: dict) -> Tuple[List[str], List[str]]:
        """Load state dict with key remapping for GGUF compatibility.
        
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        return self.load_state_dict(state_dict, strict=False)


# =============================================================================
# GGUF Key Mapping
# =============================================================================

GGUF_TO_NATIVE_KEY_MAP = {
    # Embeddings
    "token_embd.weight": "model.embed_tokens.weight",
    
    # Final norm
    "output_norm.weight": "model.norm.weight",
    
    # Per-layer mappings (use {i} as placeholder)
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
}


def remap_gguf_keys(gguf_state_dict: dict, num_layers: int = 36) -> dict:
    """Remap GGUF state dict keys to native model keys.
    
    Args:
        gguf_state_dict: State dict with GGUF-style keys
        num_layers: Number of transformer layers
    
    Returns:
        State dict with native model keys
    """
    remapped = {}
    
    for gguf_key, value in gguf_state_dict.items():
        native_key = None
        
        # Check direct mappings
        if gguf_key in GGUF_TO_NATIVE_KEY_MAP:
            native_key = GGUF_TO_NATIVE_KEY_MAP[gguf_key]
        else:
            # Check layer-indexed mappings
            for pattern, target in GGUF_TO_NATIVE_KEY_MAP.items():
                if "{i}" in pattern:
                    for i in range(num_layers):
                        gguf_pattern = pattern.replace("{i}", str(i))
                        if gguf_key == gguf_pattern:
                            native_key = target.replace("{i}", str(i))
                            break
                    if native_key:
                        break
        
        if native_key:
            remapped[native_key] = value
        else:
            # Keep original key if no mapping found
            remapped[gguf_key] = value
            logger.debug("No mapping for GGUF key: %s", gguf_key)
    
    return remapped


__all__ = [
    "Qwen3Config",
    "Qwen3_4B",
    "Qwen3Model",
    "remap_gguf_keys",
    "GGUF_TO_NATIVE_KEY_MAP",
]
