"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL base/refiner guardrails for SUPIR.
SUPIR uses an SDXL base/finetune as the backbone and must reject SDXL Refiner checkpoints as invalid inputs.

This module provides a file-based detector that prefers header-only SafeTensors inspection when possible
(keeps the API path import-light and avoids loading large tensors unnecessarily).

Symbols (top-level; keep in sync; no ghosts):
- `SdxlVariant` (enum): SDXL base/refiner/unknown classification used by SUPIR validators.
- `detect_sdxl_variant_for_checkpoint` (function): Detect SDXL base vs refiner for a checkpoint file.
- `require_sdxl_base_checkpoint` (function): Validate a checkpoint is SDXL base (fail loud otherwise).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
from apps.backend.runtime.model_registry.specs import ModelFamily

from .errors import SupirBaseModelError


class SdxlVariant(str, Enum):
    BASE = "base"
    REFINER = "refiner"
    UNKNOWN = "unknown"


_SDXL_REQUIRED_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
)

_SDXL_REFINER_REQUIRED_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj_weight",
)


def _shape_from_header(meta: object) -> tuple[int, ...] | None:
    if not isinstance(meta, dict):
        return None
    raw = meta.get("shape")
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        return tuple(int(x) for x in raw)
    except Exception:
        return None


def _detect_from_safetensors_header(header: dict[str, object]) -> SdxlVariant:
    keys = set(header.keys())
    keys.discard("__metadata__")

    def has_prefix(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    # Base detector: must have dual embedders and VAE prefix.
    if all(k in keys for k in _SDXL_REQUIRED_KEYS) and has_prefix("first_stage_model.") and has_prefix("conditioner.embedders.1."):
        emb_l = _shape_from_header(header.get("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"))
        emb_g = _shape_from_header(header.get("conditioner.embedders.1.model.token_embedding.weight"))
        if emb_l and emb_g and emb_l[-1] == 768 and emb_g[-1] == 1280:
            return SdxlVariant.BASE

    # Refiner detector: must not have embedder 1, only CLIP-G.
    if all(k in keys for k in _SDXL_REFINER_REQUIRED_KEYS) and not has_prefix("conditioner.embedders.1."):
        emb_g = _shape_from_header(header.get("conditioner.embedders.0.model.token_embedding.weight"))
        if emb_g and emb_g[-1] == 1280:
            return SdxlVariant.REFINER

    return SdxlVariant.UNKNOWN


def _detect_from_state_dict(state_dict: Mapping[str, Any]) -> SdxlVariant:
    from apps.backend.runtime.model_registry.loader import detect_from_state_dict

    signature = detect_from_state_dict(state_dict)
    if signature.family is ModelFamily.SDXL:
        return SdxlVariant.BASE
    if signature.family is ModelFamily.SDXL_REFINER:
        return SdxlVariant.REFINER
    return SdxlVariant.UNKNOWN


def detect_sdxl_variant_for_checkpoint(path: Path) -> SdxlVariant:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".safetensors", ".safetensor"}:
        try:
            header = read_safetensors_header(p)
        except Exception:
            return SdxlVariant.UNKNOWN
        return _detect_from_safetensors_header(header)

    # Fallback path: load state_dict for key inspection (torch-heavy; keep local).
    from apps.backend.runtime.checkpoint.io import load_torch_file

    try:
        state_dict = load_torch_file(str(p), safe_load=True, device="cpu")
    except Exception:
        return SdxlVariant.UNKNOWN
    if not isinstance(state_dict, Mapping):
        return SdxlVariant.UNKNOWN
    return _detect_from_state_dict(state_dict)


def require_sdxl_base_checkpoint(path: Path) -> None:
    p = Path(path)
    if not p.is_file():
        raise SupirBaseModelError(f"SUPIR base model file not found: {p}")
    variant = detect_sdxl_variant_for_checkpoint(p)
    if variant is SdxlVariant.BASE:
        return
    if variant is SdxlVariant.REFINER:
        raise SupirBaseModelError("SUPIR base must be SDXL base/finetune; SDXL Refiner is not supported.")
    raise SupirBaseModelError(f"SUPIR base must be an SDXL base/finetune checkpoint; could not detect SDXL family for: {p.name}")


__all__ = [
    "SdxlVariant",
    "detect_sdxl_variant_for_checkpoint",
    "require_sdxl_base_checkpoint",
]

