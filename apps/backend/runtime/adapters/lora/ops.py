"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tensor math utilities for LoRA/LoHa/LoKr variants.
Implements low-level decompositions and Tucker/CP reconstruction helpers used when materializing adapter weights.

Symbols (top-level; keep in sync; no ghosts):
- `rebuild_cp_decomposition` (function): Reconstructs weights for CP-decomposed LoRA variants.
- `rebuild_conventional` (function): Reconstructs conventional LoRA weights (optionally with dynamic dim trimming).
- `factorization` (function): Finds a factor pair for a dimension (used for LoKr-style decompositions).
- `tucker_weight_from_conv` (function): Reconstructs Tucker weights from conv-style components.
- `tucker_weight` (function): Reconstructs Tucker weights from factor matrices.
"""

from __future__ import annotations

import torch


def rebuild_cp_decomposition(up: torch.Tensor, down: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    up_r = up.reshape(up.size(0), -1)
    down_r = down.reshape(down.size(0), -1)
    return torch.einsum("n m ... , i n, m j -> i j ...", mid, up_r, down_r)


def rebuild_conventional(up: torch.Tensor, down: torch.Tensor, shape, dyn_dim=None) -> torch.Tensor:
    up_r = up.reshape(up.size(0), -1)
    down_r = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up_r = up_r[:, :dyn_dim]
        down_r = down_r[:dyn_dim, :]
    return (up_r @ down_r).reshape(shape)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    if factor > 0 and dimension % factor == 0:
        m = factor
        n = dimension // factor
        if m > n:
            m, n = n, m
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    best = m + n
    while m < n:
        candidate = m + 1
        while dimension % candidate != 0:
            candidate += 1
        other = dimension // candidate
        if candidate + other > best or candidate > factor:
            break
        m, n = candidate, other
        best = m + n
    if m > n:
        m, n = n, m
    return m, n


def tucker_weight_from_conv(up: torch.Tensor, down: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    up_r = up.reshape(up.size(0), up.size(1))
    down_r = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ... , i m, n j -> i j ...", mid, up_r, down_r)


def tucker_weight(wa: torch.Tensor, wb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)
