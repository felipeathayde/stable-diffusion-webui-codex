from __future__ import annotations

"""
Engine-specific text encoder roots registry (paths.json-backed).

This module exposes the directories configured under apps/paths.json for:
  - sd15_tenc
  - sdxl_tenc
  - flux_tenc
  - wan22_tenc

It does *not* load any models or import torch; it is safe to use from
inventory, diagnostics, and future selection UIs.
"""

from dataclasses import dataclass
from typing import Dict, List

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.model_registry.specs import ModelFamily

from .base import AssetEntry


_FAMILY_KEYS: Dict[ModelFamily, str] = {
    ModelFamily.SD15: "sd15_tenc",
    ModelFamily.SDXL: "sdxl_tenc",
    ModelFamily.FLUX: "flux_tenc",
    ModelFamily.WAN22: "wan22_tenc",
}


@dataclass
class TextEncoderRoot(AssetEntry):
    """Engine-specific text encoder root.

    name: human-friendly label (e.g., 'sd15/models/sd15-tenc')
    path: absolute filesystem path
    kind: fixed to 'text_encoder_root'
    tags: includes the ModelFamily value (sd15/sdxl/flux/wan22)
    """


def _build_name(family: ModelFamily, path: str) -> str:
    return f"{family.value}/{path}"


def list_text_encoder_roots() -> List[TextEncoderRoot]:
    """Return all configured text encoder roots across families."""
    roots: List[TextEncoderRoot] = []
    for family, key in _FAMILY_KEYS.items():
        for path in get_paths_for(key):
            roots.append(
                TextEncoderRoot(
                    name=_build_name(family, path),
                    path=path,
                    kind="text_encoder_root",
                    tags=[family.value],
                    meta={"family": family.value, "key": key},
                )
            )
    return roots


def list_text_encoder_roots_by_family() -> Dict[str, List[TextEncoderRoot]]:
    """Return text encoder roots grouped by model family value."""
    grouped: Dict[str, List[TextEncoderRoot]] = {}
    for root in list_text_encoder_roots():
        family = (root.meta.get("family") or "").strip() or "other"
        grouped.setdefault(family, []).append(root)
    return grouped


__all__ = ["TextEncoderRoot", "list_text_encoder_roots", "list_text_encoder_roots_by_family"]

