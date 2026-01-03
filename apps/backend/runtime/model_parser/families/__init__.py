"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model-family → parser-plan dispatch for the Codex model parser.
Maps `ModelFamily` values to the appropriate `build_plan(...)` implementation and provides a single `resolve_plan(...)` entrypoint used by
`parse_state_dict(...)`.

Symbols (top-level; keep in sync; no ghosts):
- `_BUILDERS` (constant): Mapping of supported `ModelFamily` values to plan-builder callables.
- `resolve_plan` (function): Resolves a `ParserPlanBundle` for a signature (raises `UnsupportedFamilyError` when missing).
"""

from __future__ import annotations

from typing import Callable, Dict

from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature

from ..errors import UnsupportedFamilyError
from ..specs import ParserPlanBundle
from . import chroma, flux, sd1, sd2, sd3, sdxl, wan22, zimage


_BUILDERS: Dict[ModelFamily, Callable[[ModelSignature], ParserPlanBundle]] = {
    ModelFamily.SD15: sd1.build_plan,
    ModelFamily.SD20: sd2.build_plan,
    ModelFamily.SDXL: sdxl.build_plan,
    ModelFamily.SDXL_REFINER: sdxl.build_plan,
    ModelFamily.SD3: sd3.build_plan,
    ModelFamily.SD35: sd3.build_plan,
    ModelFamily.FLUX: flux.build_plan,
    ModelFamily.FLUX_KONTEXT: flux.build_plan,
    ModelFamily.CHROMA: chroma.build_plan,
    ModelFamily.WAN22: wan22.build_plan,
    ModelFamily.ZIMAGE: zimage.build_plan,
}


def resolve_plan(signature: ModelSignature) -> ParserPlanBundle:
    builder = _BUILDERS.get(signature.family)
    if builder is None:
        raise UnsupportedFamilyError(signature.family.value)
    return builder(signature)
