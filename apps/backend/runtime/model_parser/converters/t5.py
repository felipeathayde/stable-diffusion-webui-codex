from __future__ import annotations

from typing import Any, Dict

import torch


def _clone(sd: Dict[str, Any]) -> Dict[str, Any]:
    return dict(sd)


def _ensure_layer_norm_dtype(sd: Dict[str, Any], key: str) -> None:
    tensor = sd.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32:
        sd[key] = tensor.to(torch.float32)


def convert_t5_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    work = _clone(sd)
    # Ensure final layer norm remains in float32 for numerical stability.
    _ensure_layer_norm_dtype(work, "transformer.encoder.final_layer_norm.weight")
    _ensure_layer_norm_dtype(work, "transformer.final_layer_norm.weight")
    return work


def convert_t5xxl_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_t5_encoder(sd)


def convert_umt5_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_t5_encoder(sd)


__all__ = ["convert_t5_encoder", "convert_t5xxl_encoder", "convert_umt5_encoder"]
