from __future__ import annotations

from typing import Any, Dict

import torch

from apps.backend.runtime.models.state_dict import transformers_convert


def _ensure_position_ids_long(sd: Dict[str, Any], key: str) -> None:
    value = sd.get(key)
    if isinstance(value, torch.Tensor) and value.dtype != torch.long:
        sd[key] = value.round().to(torch.long)


def _with_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out


def _normalize_text_projection(sd: Dict[str, Any], alias: str, *, transpose: bool = False) -> None:
    key_plain = f"{alias}.text_projection"
    if key_plain in sd:
        tensor = sd.pop(key_plain)
        if isinstance(tensor, torch.Tensor) and transpose:
            tensor = tensor.transpose(0, 1).contiguous()
        sd[f"{alias}.text_projection.weight"] = tensor

    key_plain_weight = f"{alias}.text_projection.weight"
    if key_plain_weight in sd:
        sd[f"{alias}.text_projection.weight"] = sd.pop(key_plain_weight)

    key_transform = f"{alias}.transformer.text_projection"
    if key_transform in sd:
        sd[f"{alias}.text_projection.weight"] = sd.pop(key_transform)


def convert_clip(
    sd: Dict[str, Any],
    *,
    alias: str,
    layers: int,
    ensure_position_ids: bool = False,
    drop_logit_scale: bool = False,
    transpose_projection: bool = False,
) -> Dict[str, Any]:
    work = _with_prefix(dict(sd), f"{alias}.")
    # Accept OpenCLIP-style keys under "<alias>.transformer.resblocks.*" and normalize to
    # diffusers-style "<alias>.transformer.text_model.encoder.layers.*".
    transformers_convert(work, f"{alias}.", f"{alias}.transformer.text_model.", layers)
    if ensure_position_ids:
        _ensure_position_ids_long(work, f"{alias}.transformer.text_model.embeddings.position_ids")
    _normalize_text_projection(work, alias, transpose=transpose_projection)
    if drop_logit_scale:
        work.pop(f"{alias}.logit_scale", None)
    return _strip_prefix(work, f"{alias}.")


def convert_sd15_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=12,
        ensure_position_ids=True,
        drop_logit_scale=True,
        transpose_projection=False,
    )
    # Remove heads reconstructed at runtime.
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sd20_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_h",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
        transpose_projection=True,
    )


def convert_sdxl_clip_l(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sdxl_clip_g(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_g",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=False,
        transpose_projection=False,
    )
