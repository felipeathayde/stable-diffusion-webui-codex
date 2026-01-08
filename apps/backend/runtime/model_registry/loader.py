"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model signature detection by running the registered detectors against a state dict.
Builds a `SignalBundle` and returns the single matching `ModelSignature` (fails fast with `UnknownModelError`/`AmbiguousModelError`).

Symbols (top-level; keep in sync; no ghosts):
- `detect_from_state_dict` (function): Detects checkpoint family/architecture and returns a structured `ModelSignature`.
"""

from __future__ import annotations

from typing import Mapping, Any, List

from apps.backend.runtime.model_registry.detectors.base import REGISTRY, ModelDetector
from apps.backend.runtime.model_registry.errors import AmbiguousModelError, UnknownModelError
from apps.backend.runtime.model_registry.signals import build_bundle
from apps.backend.runtime.model_registry.specs import ModelSignature


def detect_from_state_dict(state_dict: Mapping[str, Any]) -> ModelSignature:
    bundle = build_bundle(state_dict)
    matches: List[ModelDetector] = []
    for detector in REGISTRY.detectors:
        try:
            if detector.matches(bundle):
                matches.append(detector)
        except Exception as exc:  # pragma: no cover - defensive
            raise UnknownModelError(f"detector {detector.__class__.__name__} failed", detail={"error": str(exc)}) from exc
    if not matches:
        raise UnknownModelError("No model detector matched state dict", detail={"keys": len(bundle.keys)})
    if len(matches) > 1:
        raise AmbiguousModelError(
            "Multiple detectors matched state dict",
            matches=[det.__class__.__name__ for det in matches],
        )
    return matches[0].build_signature(bundle)
