"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR asset resolution helpers (Phase 2).
Centralizes the “resolve + validate” logic for SUPIR inputs:
- resolve SDXL base checkpoint by name/sha,
- reject SDXL Refiner (fail loud),
- resolve SUPIR variant ckpt from `supir_models` roots.

This module does **not** instantiate the SUPIR model yet; it only resolves validated paths so the API/router can enforce
HTTP-400 semantics for missing/invalid inputs before creating a task.

Symbols (top-level; keep in sync; no ghosts):
- `SupirResolvedAssets` (dataclass): Validated file paths required for a SUPIR enhance run.
- `resolve_supir_assets` (function): Resolve + validate base checkpoint + SUPIR variant ckpt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from apps.backend.runtime.models.types import CheckpointFormat

from .errors import SupirBaseModelError
from .sdxl_guard import require_sdxl_base_checkpoint
from .weights import SupirVariant, resolve_supir_weights


@dataclass(frozen=True)
class SupirResolvedAssets:
    base_checkpoint: Path
    variant_ckpt: Path


def resolve_supir_assets(*, base_model: str, variant: SupirVariant, supir_models_roots: Sequence[Path]) -> SupirResolvedAssets:
    """Resolve and validate required assets for SUPIR enhance."""

    from apps.backend.runtime.models import api as model_api

    base_ref = str(base_model or "").strip()
    if not base_ref:
        raise SupirBaseModelError("Missing 'supir_base_model'")

    record = model_api.find_checkpoint_by_sha(base_ref) or model_api.find_checkpoint(base_ref)
    if record is None:
        raise SupirBaseModelError(f"Unknown SUPIR base model: {base_ref!r}")

    base_path = Path(record.filename)
    if record.format is not CheckpointFormat.CHECKPOINT:
        raise SupirBaseModelError(
            "SUPIR base must be a full SDXL checkpoint file (.safetensors/.ckpt); "
            f"got format={record.format.value!r} for {record.title!r}"
        )
    if record.core_only or base_path.suffix.lower() == ".gguf":
        raise SupirBaseModelError("SUPIR base must be a full SDXL checkpoint (.safetensors/.ckpt), not a core-only GGUF.")

    require_sdxl_base_checkpoint(base_path)

    weights = resolve_supir_weights(roots=list(supir_models_roots), variant=variant)
    return SupirResolvedAssets(base_checkpoint=base_path, variant_ckpt=weights.ckpt_path)


__all__ = [
    "SupirResolvedAssets",
    "resolve_supir_assets",
]
