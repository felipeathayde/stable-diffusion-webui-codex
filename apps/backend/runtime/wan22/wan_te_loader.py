"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN UMT5-XXL FP8 weights loader (safetensors) for CUDA TE kernels.
Reads a UMT5-XXL encoder weights file stored as FP8 (`uint8` + per-tensor scale) and builds a lightweight mapping usable by CUDA kernels.
Does not dequantize; validates shapes, collects scales, and fails fast on missing/invalid tensors.

Symbols (top-level; keep in sync; no ghosts):
- `LinearPack` (dataclass): Packed linear layer weights/bias with expected (Cout,Cin) shape.
- `WanTEFp8Weights` (dataclass): Structured FP8 TE weight bundle (embedding + per-layer linear packs + shape metadata).
- `_tensor_from_safe` (function): Loads one tensor from safetensors with strict missing-key errors.
- `_fp8_pack_from` (function): Builds one `LinearPack` by reading FP8 weights + optional scale/bias tensors.
- `load_umt5_xxl_fp8` (function): Loads a full UMT5-XXL FP8 weights file into a `WanTEFp8Weights` bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
import os

import torch
from safetensors import safe_open

from .wan_te_cuda import Fp8Weight

log = logging.getLogger("wan22.te.loader")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[wan22.te.loader] %(levelname)s: %(message)s'))
    log.addHandler(h)
log.setLevel(logging.INFO)
log.propagate = False


@dataclass
class LinearPack:
    w: Fp8Weight
    b: Optional[torch.Tensor]
    shape: Tuple[int, int]  # (Cout, Cin)


@dataclass
class WanTEFp8Weights:
    # Shared token embedding
    embed: Fp8Weight
    # Encoder blocks (list of dicts with keys: q,k,v,o, wi, wi_1(optional), wo, ln1_w, ln2_w)
    blocks: Dict[int, Dict[str, LinearPack]]
    num_layers: int
    d_model: int
    n_heads: int


def _tensor_from_safe(path: str, key: str, device: str = 'cpu') -> torch.Tensor:
    with safe_open(path, framework="pt", device=device) as f:
        if key not in f.keys():
            raise RuntimeError(f"TE weights missing tensor '{key}' in {os.path.basename(path)}")
        t = f.get_tensor(key)
    return t


def _fp8_pack_from(path: str, w_key: str, scale_key: Optional[str], b_key: Optional[str]) -> LinearPack:
    w_u8 = _tensor_from_safe(path, w_key)
    if w_u8.dtype != torch.uint8:
        raise RuntimeError(f"Expected uint8 for '{w_key}', got {w_u8.dtype}")
    if scale_key is None:
        # Single scale fallback (1.0) — caller should replace later if true FP8 scales are provided elsewhere
        w_scale = torch.ones((w_u8.shape[0],), dtype=torch.float32)
    else:
        w_scale = _tensor_from_safe(path, scale_key)
        if w_scale.dim() == 0:
            w_scale = w_scale.view(1).expand(w_u8.shape[0]).contiguous()
        if w_scale.numel() not in (1, w_u8.shape[0]):
            raise RuntimeError(f"Scale shape mismatch for '{scale_key}' vs '{w_key}'")
        if w_scale.numel() == 1:
            w_scale = w_scale.view(1).expand(w_u8.shape[0]).contiguous()
    b = _tensor_from_safe(path, b_key) if b_key is not None else None
    return LinearPack(Fp8Weight(w_u8=w_u8, scale=w_scale, fp8_format='e4m3fn'), b, (w_u8.shape[0], w_u8.shape[1]))


def load_umt5_xxl_fp8(path: str) -> WanTEFp8Weights:
    """Best-effort loader for UMT5-XXL FP8 safetensors.

    This function expects the following patterns (subject to change; logs sample keys):
      - shared.embedding.weight_u8 / shared.embedding.scale
      - encoder.block.{i}.layer.0.SelfAttention.(q|k|v|o).weight_u8 + .scale, optional bias .bias
      - encoder.block.{i}.layer.1.DenseReluDense.(wi|wi_1|wo).weight_u8 + .scale, optional bias

    Returns a WanTEFp8Weights with per-block packs. Errors are explicit when tensors are missing.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"TE safetensors not found: {path}")
    # Probe keys and log a sample for diagnostics
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    log.info("te file=%s tensors=%d sample=%s", os.path.basename(path), len(keys), keys[:6])

    # Shared embedding
    embed_u8_key = 'shared.embedding.weight_u8'
    embed_scale_key = 'shared.embedding.scale'
    if embed_u8_key not in keys:
        # Try alternative naming 'shared.weight_u8'
        embed_u8_key = 'shared.weight_u8' if 'shared.weight_u8' in keys else embed_u8_key
        embed_scale_key = 'shared.scale' if 'shared.scale' in keys else embed_scale_key
    embed_u8 = _tensor_from_safe(path, embed_u8_key)
    embed_scale = _tensor_from_safe(path, embed_scale_key) if embed_scale_key in keys else torch.tensor([1.0], dtype=torch.float32)
    embed = Fp8Weight(w_u8=embed_u8, scale=embed_scale if embed_scale.dim() else embed_scale.view(1), fp8_format='e4m3fn')

    # Discover number of layers by scanning keys
    max_layer = -1
    for k in keys:
        if k.startswith('encoder.block.'):
            try:
                idx = int(k.split('.')[2])
                max_layer = max(max_layer, idx)
            except Exception:
                continue
    if max_layer < 0:
        raise RuntimeError("No encoder.block.* keys found in TE safetensors")
    num_layers = max_layer + 1

    blocks: Dict[int, Dict[str, LinearPack]] = {}
    # Populate per-layer packs (best-effort: presence checked)
    for i in range(num_layers):
        prefix_attn = f'encoder.block.{i}.layer.0.SelfAttention.'
        prefix_ffn = f'encoder.block.{i}.layer.1.DenseReluDense.'
        layer: Dict[str, LinearPack] = {}
        for name in ('q', 'k', 'v', 'o'):
            w = f'{prefix_attn}{name}.weight_u8'
            s = f'{prefix_attn}{name}.scale'
            b = f'{prefix_attn}{name}.bias'
            if w in keys:
                layer[name] = _fp8_pack_from(path, w, s if s in keys else None, b if b in keys else None)
        # Optional LayerNorm weights (float)
        ln1 = f'encoder.block.{i}.layer.0.layer_norm.weight'
        ln2 = f'encoder.block.{i}.layer.1.layer_norm.weight'
        if ln1 in keys:
            ln1_w = _tensor_from_safe(path, ln1)
            layer['ln1_w'] = LinearPack(Fp8Weight(w_u8=torch.empty(0, dtype=torch.uint8), scale=torch.ones(1)), ln1_w, (ln1_w.numel(), 1))
        if ln2 in keys:
            ln2_w = _tensor_from_safe(path, ln2)
            layer['ln2_w'] = LinearPack(Fp8Weight(w_u8=torch.empty(0, dtype=torch.uint8), scale=torch.ones(1)), ln2_w, (ln2_w.numel(), 1))
        for name in ('wi', 'wi_1', 'wo'):
            w = f'{prefix_ffn}{name}.weight_u8'
            s = f'{prefix_ffn}{name}.scale'
            b = f'{prefix_ffn}{name}.bias'
            if w in keys:
                layer[name] = _fp8_pack_from(path, w, s if s in keys else None, b if b in keys else None)
        if not layer:
            raise RuntimeError(f"Encoder block {i} missing expected FP8 tensors")
        blocks[i] = layer

    # Infer d_model from a present projection (o or wi)
    d_model = None
    n_heads = None
    for i in range(num_layers):
        if 'o' in blocks[i]:
            d_model = blocks[i]['o'].shape[0]
            break
        if 'wi' in blocks[i]:
            d_model = blocks[i]['wi'].shape[1]
            break
    if d_model is None:
        raise RuntimeError("Failed to infer d_model from FP8 tensors")

    # Head count is not encoded — caller must provide, or infer from model config.
    n_heads = -1

    log.info("loaded TE FP8: layers=%d d_model=%s embed=%s", num_layers, d_model, tuple(embed.w_u8.shape))
    return WanTEFp8Weights(embed=embed, blocks=blocks, num_layers=num_layers, d_model=int(d_model), n_heads=int(n_heads))
