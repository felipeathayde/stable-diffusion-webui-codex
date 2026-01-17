"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared ZImage checkpoint dimension inference (detector + runtime loader).
Centralizes hidden/context/latent/layer inference so model registry detectors and runtime loader logic don't drift.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageDims` (dataclass): Inferred core dimensions for ZImage transformer checkpoints.
- `_shape_of_value` (function): Best-effort helper that returns a tuple shape for a value exposing `.shape`.
- `_max_index` (function): Extracts the max numeric index for a key prefix (used to infer layer counts).
- `infer_zimage_dims` (function): Infers `ZImageDims` from key names + a shape getter (prefix-agnostic).
- `infer_zimage_dims_from_state_dict` (function): Convenience helper that infers dims from a state dict with common prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ZImageDims:
    hidden_dim: int
    context_dim: int
    latent_channels: int
    num_layers: int
    num_refiner_layers: int
    num_heads: int
    mlp_hidden: int
    t_dim: int


def _shape_of_value(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def _max_index(keys: Iterable[str], *, prefix: str, required_substring: str | None = None) -> int:
    max_idx = -1
    for k in keys:
        if not k.startswith(prefix):
            continue
        if required_substring is not None and required_substring not in k:
            continue
        parts = k.split(".")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[1])
        except Exception:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else 0


def infer_zimage_dims(
    keys: Iterable[str],
    shape_of: Callable[[str], tuple[int, ...] | None],
    *,
    patch_size: int = 2,
) -> ZImageDims:
    hidden_dim = 3840
    context_dim = 2560
    latent_channels = 16
    num_layers = 30
    num_refiner_layers = 2
    t_dim = 256
    mlp_hidden = 10240

    x_emb = shape_of("x_embedder.weight")
    if x_emb and len(x_emb) >= 1:
        hidden_dim = int(x_emb[0])

    # cap_embedder is Sequential(RMSNorm(context_dim), Linear(context_dim -> hidden_dim))
    cap_lin = shape_of("cap_embedder.1.weight")
    if cap_lin and len(cap_lin) == 2:
        context_dim = int(cap_lin[1])

    final = shape_of("final_layer.linear.weight")
    patch_area = int(max(1, patch_size) * max(1, patch_size))
    if final and len(final) >= 1 and patch_area > 0:
        out_dim = int(final[0])
        if out_dim % patch_area == 0:
            latent_channels = int(out_dim // patch_area)

    te2 = shape_of("t_embedder.mlp.2.weight")
    if te2 and len(te2) == 2:
        t_dim = int(te2[0])

    w1 = shape_of("layers.0.feed_forward.w1.weight")
    if w1 and len(w1) == 2:
        mlp_hidden = int(w1[0])

    detected_layers = _max_index(keys, prefix="layers.", required_substring=".adaLN_modulation.")
    if detected_layers:
        num_layers = int(detected_layers)

    detected_refiners = _max_index(keys, prefix="context_refiner.", required_substring=None)
    if detected_refiners:
        num_refiner_layers = int(detected_refiners)

    num_heads = int(hidden_dim // 128) if hidden_dim > 0 and hidden_dim % 128 == 0 else 30

    return ZImageDims(
        hidden_dim=int(hidden_dim),
        context_dim=int(context_dim),
        latent_channels=int(latent_channels),
        num_layers=int(num_layers),
        num_refiner_layers=int(num_refiner_layers),
        num_heads=int(num_heads),
        mlp_hidden=int(mlp_hidden),
        t_dim=int(t_dim),
    )


def infer_zimage_dims_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    patch_size: int = 2,
    prefixes: Sequence[str] = ("model.diffusion_model.", "diffusion_model.", "model.", ""),
) -> ZImageDims:
    keys = tuple(state_dict.keys())
    prefix = ""
    for p in prefixes:
        if f"{p}x_embedder.weight" in state_dict:
            prefix = p
            break

    if prefix:
        stripped = [k[len(prefix):] for k in keys if k.startswith(prefix)]

        def shape_of(name: str) -> tuple[int, ...] | None:
            return _shape_of_value(state_dict.get(prefix + name))
    else:
        stripped = list(keys)

        def shape_of(name: str) -> tuple[int, ...] | None:
            return _shape_of_value(state_dict.get(name))

    return infer_zimage_dims(stripped, shape_of, patch_size=patch_size)


__all__ = ["ZImageDims", "infer_zimage_dims", "infer_zimage_dims_from_state_dict"]
