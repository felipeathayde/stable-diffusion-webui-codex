"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Rotary positional embedding (RoPE) utilities and timestep embeddings for Flux variants.

Symbols (top-level; keep in sync; no ghosts):
- `build_rotary_frequencies` (function): Construct RoPE rotation matrices for positional indices.
- `apply_rotary_embeddings` (function): Apply RoPE rotations to query/key tensors.
- `timestep_embedding` (function): Stable diffusion-style sinusoidal timestep embedding used by Flux variants.
"""

from __future__ import annotations

import math

import torch


def build_rotary_frequencies(
    positions: torch.Tensor,
    dim: int,
    theta: float,
) -> torch.Tensor:
    """Construct rotary positional embeddings (RoPE) for the provided positions.

    Args:
        positions: Tensor of shape ``(..., sequence_length)`` containing positional indices.
        dim: Rotary dimension (must be even).
        theta: Base frequency hyper-parameter.
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    if positions.device.type in {"mps", "xpu"}:
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim
    else:
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device) / dim
    omega = 1.0 / (theta ** scale)
    out = positions.unsqueeze(-1) * omega.unsqueeze(0)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked = torch.stack((cos_out, -sin_out, sin_out, cos_out), dim=-1)
    # stacked shape: (*positions.shape, dim//2, 4)
    # We want shape: (*positions.shape, dim//2, 2, 2) for rotation matrix format
    rotary = stacked.view(*positions.shape, dim // 2, 2, 2)
    return rotary.float()


def apply_rotary_embeddings(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to the provided query and key tensors."""
    if freqs.dim() == q.dim() + 1:
        freqs = freqs.unsqueeze(1)
    if freqs.dim() != q.dim() + 2:
        raise ValueError("Unexpected rotary frequency shape")
    freqs = freqs.to(q.dtype)
    freqs = freqs.expand(q.shape[0], q.shape[1], *freqs.shape[2:])
    q_view = q.float().reshape(q.shape[0], q.shape[1], q.shape[2], -1, 1, 2)
    k_view = k.float().reshape(k.shape[0], k.shape[1], k.shape[2], -1, 1, 2)
    q_out = freqs[..., 0] * q_view[..., 0] + freqs[..., 1] * q_view[..., 1]
    k_out = freqs[..., 0] * k_view[..., 0] + freqs[..., 1] * k_view[..., 1]
    return q_out.reshape_as(q).type_as(q), k_out.reshape_as(k).type_as(k)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10000.0,
    time_factor: float = 1000.0,
) -> torch.Tensor:
    """Stable diffusion-style sinusoidal embedding used by Flux variants."""
    scaled = time_factor * timesteps
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=scaled.device) / half
    )
    args = scaled[:, None].float() * freqs[None]
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    if torch.is_floating_point(timesteps):
        embedding = embedding.to(timesteps)
    return embedding
