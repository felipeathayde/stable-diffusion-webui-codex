"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native CLIP text model implementation (lightweight, state-dict friendly).
Defines a minimal CLIP text encoder stack compatible with our converted state dicts and downstream expectations (HF-like `.text_model`),
including pooled output selection and optional hidden-state capture.

Symbols (top-level; keep in sync; no ghosts):
- `ClipActivation` (enum): Supported activation functions for MLP blocks.
- `_act` (function): Maps `ClipActivation` values to callable implementations.
- `CodexCLIPTextConfig` (dataclass): Minimal CLIP text config used to build the model.
- `_CLIPAttention` (class): Multi-head self-attention block.
- `_CLIPAttentionFusedQKV` (class): Multi-head self-attention with fused QKV projection (`in_proj`).
- `_CLIPMLP` (class): MLP block (`fc1`/`fc2` + activation).
- `_CLIPLayer` (class): Encoder layer (self-attn + MLP) with residuals and layer norms.
- `_CLIPEncoder` (class): Stack of `_CLIPLayer` blocks (optionally returning hidden states).
- `_CLIPEmbeddings` (class): Token+position embeddings.
- `_CLIPOutput` (class): HF-like output container (`last_hidden_state`, `hidden_states`, `pooler_output`).
- `CodexCLIPTextModel` (class): Top-level text model wrapper exposing `.text_model` for API parity.
- `CodexCLIPTextModelFusedQKV` (class): `CodexCLIPTextModel` variant using fused QKV attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend


class ClipActivation(str, Enum):
    QUICK_GELU = "quick_gelu"
    GELU = "gelu"
    GELU_PYTORCH_TANH = "gelu_pytorch_tanh"


def _act(fn: ClipActivation):
    if fn == ClipActivation.QUICK_GELU:
        return lambda a: a * torch.sigmoid(1.702 * a)
    if fn == ClipActivation.GELU:
        return F.gelu
    if fn == ClipActivation.GELU_PYTORCH_TANH:
        return lambda a: F.gelu(a, approximate="tanh")
    raise ValueError(f"Unsupported activation {fn}")


@dataclass(slots=True)
class CodexCLIPTextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    max_position_embeddings: int
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 49407  # CLIP default

    @classmethod
    def from_dict(cls, d: dict) -> "CodexCLIPTextConfig":
        return cls(
            hidden_size=int(d["hidden_size"]),
            num_hidden_layers=int(d["num_hidden_layers"]),
            num_attention_heads=int(d["num_attention_heads"]),
            intermediate_size=int(d["intermediate_size"]),
            hidden_act=str(d.get("hidden_act", "gelu")),
            max_position_embeddings=int(d.get("max_position_embeddings", 77)),
            layer_norm_eps=float(d.get("layer_norm_eps", 1e-5)),
            eos_token_id=int(d.get("eos_token_id", 49407)),
        )


class _CLIPAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H = self.heads
        q = self.q_proj(x).view(B, T, H, C // H).transpose(1, 2)  # B,H,T,hd
        k = self.k_proj(x).view(B, T, H, C // H).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, C // H).transpose(1, 2)
        # SDPA is the canonical attention path in this repo.
        # `mask` is expected to be additive (0 for keep, -inf for masked) and broadcastable
        # to (B, H, T, T). We combine causal + padding masks upstream.
        o = attention_function_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=False,
            backend=AttentionBackend.PYTORCH,
        )
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(o)


class _CLIPAttentionFusedQKV(nn.Module):
    """Self-attention with fused QKV projection (OpenCLIP-style `in_proj`)."""

    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H = self.heads
        qkv = self.in_proj(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = attention_function_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=False,
            backend=AttentionBackend.PYTORCH,
        )
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(o)


class _CLIPMLP(nn.Module):
    def __init__(self, embed_dim: int, intermediate: int, act: ClipActivation):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, embed_dim, bias=True)
        self._act = _act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self._act(self.fc1(x)))


class _CLIPLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        intermediate: int,
        act: ClipActivation,
        eps: float,
        *,
        attention_cls: type[nn.Module] = _CLIPAttention,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attn = attention_cls(embed_dim, heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.mlp = _CLIPMLP(embed_dim, intermediate, act)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _CLIPEncoder(nn.Module):
    def __init__(
        self,
        layers: int,
        embed_dim: int,
        heads: int,
        intermediate: int,
        act: ClipActivation,
        eps: float,
        *,
        attention_cls: type[nn.Module] = _CLIPAttention,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _CLIPLayer(embed_dim, heads, intermediate, act, eps, attention_cls=attention_cls) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        hidden: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, mask)
            if output_hidden_states:
                hidden.append(x)
        return x, hidden


class _CLIPEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, num_positions: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(num_positions, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        tok = self.token_embedding(input_ids)
        pos = self.position_embedding(positions)
        return tok + pos


class _CLIPOutput:
    __slots__ = ("last_hidden_state", "hidden_states", "pooler_output")

    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: Optional[List[torch.Tensor]], pooler_output: Optional[torch.Tensor]):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.pooler_output = pooler_output


class CodexCLIPTextModel(nn.Module):
    """Codex-native CLIP text model compatible with our converted state dict.

    Exposes `.text_model` like HF to match downstream expectations.
    """

    def __init__(self, config: CodexCLIPTextConfig):
        super().__init__()
        self.config = config
        act = ClipActivation(config.hidden_act)
        self.text_model = nn.Module()
        self.text_model.embeddings = _CLIPEmbeddings(config.hidden_size, vocab_size=49408, num_positions=config.max_position_embeddings)
        self.text_model.encoder = _CLIPEncoder(
            layers=config.num_hidden_layers,
            embed_dim=config.hidden_size,
            heads=config.num_attention_heads,
            intermediate=config.intermediate_size,
            act=act,
            eps=config.layer_norm_eps,
        )
        self.text_model.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Optional projection will be attached by the loader when needed

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  # accepted for API parity; unused
        output_hidden_states: bool = True,
    ) -> _CLIPOutput:
        x = self.text_model.embeddings(input_ids)
        compute_dtype = getattr(self, "compute_dtype", None)
        if isinstance(compute_dtype, torch.dtype) and compute_dtype != x.dtype:
            x = x.to(dtype=compute_dtype)
        attn_mask: torch.Tensor | None = None
        seq_len = int(x.shape[1])

        # --- Mask composition -------------------------------------------------
        # CLIP text attention is causal (tokens must not attend to future positions).
        # Combine:
        # - padding mask (B,1,T,T) derived from attention_mask (1 keep / 0 pad)
        # - causal mask (T,T) upper triangular with -inf above diagonal
        if attention_mask is not None:
            try:
                mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
                attn_mask = mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
                attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -torch.finfo(x.dtype).max)
            except Exception:
                attn_mask = None
        causal = torch.full(
            (seq_len, seq_len),
            -torch.finfo(x.dtype).max,
            dtype=x.dtype,
            device=x.device,
        ).triu_(1)
        if attn_mask is not None:
            attn_mask = attn_mask + causal
        else:
            attn_mask = causal
        x, hidden = self.text_model.encoder(x, attn_mask, output_hidden_states=output_hidden_states)
        x = self.text_model.final_layer_norm(x)

        # Pooled: take eos token when present, else last position
        pooled = None
        try:
            eos = self.config.eos_token_id
            # pick first eos in each row
            idx = (input_ids == eos).float()
            positions = torch.argmax(idx, dim=1)
            # if no eos (all zeros), default to last index
            positions = torch.where(idx.max(dim=1).values > 0, positions, torch.full_like(positions, input_ids.shape[1]-1))
            pooled = x[torch.arange(x.shape[0], device=x.device), positions]
        except Exception:
            pooled = x[:, -1]

        return _CLIPOutput(last_hidden_state=x, hidden_states=hidden if output_hidden_states else None, pooler_output=pooled)


class CodexCLIPTextModelFusedQKV(CodexCLIPTextModel):
    """CLIP text encoder variant with fused QKV attention parameters (`in_proj`)."""

    def __init__(self, config: CodexCLIPTextConfig):
        super().__init__(config)
        act = ClipActivation(config.hidden_act)
        self.text_model.encoder = _CLIPEncoder(
            layers=config.num_hidden_layers,
            embed_dim=config.hidden_size,
            heads=config.num_attention_heads,
            intermediate=config.intermediate_size,
            act=act,
            eps=config.layer_norm_eps,
            attention_cls=_CLIPAttentionFusedQKV,
        )


__all__ = [
    "CodexCLIPTextModel",
    "CodexCLIPTextModelFusedQKV",
    "CodexCLIPTextConfig",
    "ClipActivation",
]
