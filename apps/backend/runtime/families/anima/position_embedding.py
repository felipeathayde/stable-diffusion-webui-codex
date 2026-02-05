"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Positional embedding helpers for Cosmos Predict2 (RoPE3D + optional learnable axis embeddings).
Re-implements the upstream Cosmos Predict2 positional embedding intent without importing archived `.refs/**` code.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_embeddings` (function): Normalize embeddings along the feature dimension (fp32 norm; dtype-preserving output).
- `VideoPositionEmbedding` (class): Base class for video/image position embeddings.
- `VideoRopePosition3DEmb` (class): RoPE3D frequency generator used by Cosmos Predict2 attention blocks.
- `LearnablePosEmbAxis` (class): Optional learnable per-axis positional embeddings (crop interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from einops import rearrange, repeat


def normalize_embeddings(x: torch.Tensor, *, dims: Iterable[int] | None = None, eps: float = 1e-6) -> torch.Tensor:
    if dims is None:
        dims = tuple(range(1, x.ndim))
    dims = tuple(int(d) for d in dims)
    if not x.is_floating_point():
        raise TypeError(f"normalize_embeddings expects a floating-point tensor; got dtype={x.dtype}.")
    with torch.no_grad():
        norm = torch.linalg.vector_norm(x.float(), dim=dims, keepdim=True)
        scale = (norm.numel() / max(int(x.numel()), 1)) ** 0.5
        norm = norm * float(scale)
    return x / (norm.to(dtype=x.dtype) + float(eps))


class VideoPositionEmbedding(nn.Module):
    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        *,
        fps: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        shape = x_B_T_H_W_C.shape
        return self.generate_embeddings(shape, fps=fps, device=device or x_B_T_H_W_C.device, dtype=dtype)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        *,
        fps: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _Rope3DAxisSplit:
    dim_h: int
    dim_w: int
    dim_t: int


def _split_rope_dims(head_dim: int) -> _Rope3DAxisSplit:
    dim = int(head_dim)
    if dim <= 0:
        raise ValueError("head_dim must be > 0")
    # Upstream intent: allocate 2/6 of head_dim to H, 2/6 to W, and the rest to T.
    dim_h = (dim // 6) * 2
    dim_w = dim_h
    dim_t = dim - (2 * dim_h)
    if dim_h <= 0 or dim_w <= 0 or dim_t <= 0:
        raise ValueError(f"Invalid RoPE3D axis split for head_dim={dim}: (h={dim_h}, w={dim_w}, t={dim_t})")
    if (dim_h + dim_w + dim_t) != dim:
        raise ValueError(f"RoPE3D axis split mismatch for head_dim={dim}: {dim_h}+{dim_w}+{dim_t} != {dim}")
    return _Rope3DAxisSplit(dim_h=dim_h, dim_w=dim_w, dim_t=dim_t)


class VideoRopePosition3DEmb(VideoPositionEmbedding):
    """Generate RoPE3D frequency tensors for a `(T,H,W)` grid.

    Output shape matches the upstream Cosmos Predict2 expectation:
    - input grid: (T, H, W)
    - output: (L, D/2, 2, 2) where L=T*H*W and D=head_dim
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        split = _split_rope_dims(int(head_dim))
        self.base_fps = int(base_fps)
        self.max_h = int(len_h)
        self.max_w = int(len_w)
        self.max_t = int(len_t)
        self.enable_fps_modulation = bool(enable_fps_modulation)

        def _range(dim_part: int) -> torch.Tensor:
            # Frequencies are built over the even indices and normalized by the part dim.
            idx = torch.arange(0, dim_part, 2, device=device, dtype=torch.float32)
            return (idx[: (dim_part // 2)] / float(dim_part)).contiguous()

        self.register_buffer("dim_spatial_range", _range(split.dim_h), persistent=False)
        self.register_buffer("dim_temporal_range", _range(split.dim_t), persistent=False)

        def _ntk_factor(ratio: float, dim_part: int) -> float:
            ratio = float(ratio)
            if ratio <= 0.0:
                raise ValueError("extrapolation_ratio must be > 0")
            if dim_part <= 2:
                return 1.0
            return ratio ** (float(dim_part) / float(dim_part - 2))

        self.h_ntk_factor = _ntk_factor(h_extrapolation_ratio, split.dim_h)
        self.w_ntk_factor = _ntk_factor(w_extrapolation_ratio, split.dim_w)
        self.t_ntk_factor = _ntk_factor(t_extrapolation_ratio, split.dim_t)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        *,
        fps: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del dtype  # RoPE freqs are always generated in fp32 for stability.
        if len(B_T_H_W_C) != 5:
            raise ValueError(f"Expected embedded sequence shape (B,T,H,W,C); got {tuple(B_T_H_W_C)}")
        B, T, H, W, _ = (int(x) for x in B_T_H_W_C)
        if B <= 0 or T <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"Invalid embedded sequence shape: {tuple(B_T_H_W_C)}")

        fps_value: float | None = None
        if fps is None:
            fps_value = None
        elif isinstance(fps, (int, float)):
            fps_value = float(fps)
        elif isinstance(fps, torch.Tensor):
            if int(fps.numel()) != 1:
                raise ValueError(
                    "fps must be a scalar (numel==1) when provided. "
                    "Per-sample fps tensors are not supported by this runtime."
                )
            fps_value = float(fps.detach().cpu().item())
        else:
            raise TypeError(f"fps must be a torch.Tensor, float, or None; got {type(fps).__name__}")

        if fps_value is not None and fps_value <= 0.0:
            raise ValueError("fps must be > 0 when provided.")

        seq = torch.arange(max(H, W, T), device=device, dtype=torch.float32)

        def _axis_freqs(theta: float, dim_range: torch.Tensor) -> torch.Tensor:
            return 1.0 / (float(theta) ** dim_range.to(device=device))

        h_theta = 10000.0 * float(self.h_ntk_factor)
        w_theta = 10000.0 * float(self.w_ntk_factor)
        t_theta = 10000.0 * float(self.t_ntk_factor)

        h_freqs = _axis_freqs(h_theta, self.dim_spatial_range)
        w_freqs = _axis_freqs(w_theta, self.dim_spatial_range)
        t_freqs = _axis_freqs(t_theta, self.dim_temporal_range)

        half_h = torch.outer(seq[:H], h_freqs)
        half_w = torch.outer(seq[:W], w_freqs)
        if fps_value is None or (not self.enable_fps_modulation):
            half_t = torch.outer(seq[:T], t_freqs)
        else:
            half_t = torch.outer(seq[:T] / float(fps_value) * float(self.base_fps), t_freqs)

        def _stack(x: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.cos(x), -torch.sin(x), torch.sin(x), torch.cos(x)], dim=-1)

        half_h = _stack(half_h)
        half_w = _stack(half_w)
        half_t = _stack(half_t)

        freqs = torch.cat(
            [
                repeat(half_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        # (T,H,W,D/2,4) -> (L, D/2, 2, 2)
        return rearrange(freqs, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class LearnablePosEmbAxis(VideoPositionEmbedding):
    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        interp = str(interpolation or "").strip().lower()
        if interp not in {"crop"}:
            raise ValueError(f"Unsupported interpolation={interpolation!r} (expected 'crop').")
        self.interpolation = interp
        self.pos_emb_h = nn.Parameter(torch.empty(int(len_h), int(model_channels), device=device, dtype=dtype))
        self.pos_emb_w = nn.Parameter(torch.empty(int(len_w), int(model_channels), device=device, dtype=dtype))
        self.pos_emb_t = nn.Parameter(torch.empty(int(len_t), int(model_channels), device=device, dtype=dtype))

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        *,
        fps: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del fps
        if len(B_T_H_W_C) != 5:
            raise ValueError(f"Expected embedded sequence shape (B,T,H,W,C); got {tuple(B_T_H_W_C)}")
        B, T, H, W, _ = (int(x) for x in B_T_H_W_C)

        if self.interpolation != "crop":
            raise RuntimeError("LearnablePosEmbAxis only supports crop interpolation in this runtime.")

        emb_h = self.pos_emb_h[:H].to(device=device, dtype=dtype)
        emb_w = self.pos_emb_w[:W].to(device=device, dtype=dtype)
        emb_t = self.pos_emb_t[:T].to(device=device, dtype=dtype)

        out = (
            repeat(emb_t, "t d -> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h, "h d -> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w, "w d -> b t h w d", b=B, t=T, h=H)
        )
        if list(out.shape)[:4] != [B, T, H, W]:
            raise RuntimeError(f"LearnablePosEmbAxis produced an invalid shape: {tuple(out.shape)}")
        return normalize_embeddings(out, dims=(-1,), eps=1e-6)
