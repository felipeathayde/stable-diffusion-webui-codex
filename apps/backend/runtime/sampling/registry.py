"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler registry helpers mapping UI/API sampler names to validated sampler specifications.
Enforces scheduler compatibility before sampling context creation.

Symbols (top-level; keep in sync; no ghosts):
- `SamplerSpec` (dataclass): Canonical sampler specification (kind, default scheduler, allowed schedulers).
- `_build_specs` (function): Build the canonical sampler spec map from catalog entries (strict validation + canonicalization).
- `get_sampler_spec` (function): Resolve a sampler name to a validated `SamplerSpec`.
- `__all__` (constant): Public export list for registry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set

from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.sampling.catalog import (
    SAMPLER_DEFAULT_SCHEDULER,
    SAMPLER_OPTIONS,
    SUPPORTED_SCHEDULERS,
)


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    kind: SamplerKind
    default_scheduler: str
    allowed_schedulers: Set[str]

    def is_supported_scheduler(self, scheduler: str) -> bool:
        return scheduler in self.allowed_schedulers


_SPECS: Dict[str, SamplerSpec] = {}


def _build_specs() -> Dict[str, SamplerSpec]:
    specs: Dict[str, SamplerSpec] = {}

    for entry in SAMPLER_OPTIONS:
        name = str(entry["name"])
        raw_allowed: Iterable[str] = entry.get("schedulers", SUPPORTED_SCHEDULERS)
        allowed_canonical: Set[str] = set()
        for sched in raw_allowed:
            canonical = str(sched)
            if canonical in SUPPORTED_SCHEDULERS:
                allowed_canonical.add(canonical)
        default_scheduler = SAMPLER_DEFAULT_SCHEDULER.get(name)
        if not default_scheduler:
            raise RuntimeError(f"Sampler '{name}' is missing a default scheduler mapping")
        try:
            kind = SamplerKind.from_string(name)
        except Exception:
            # Keep unknown kinds for future porting but skip active registry
            continue
        specs[name] = SamplerSpec(
            name=name,
            kind=kind,
            default_scheduler=default_scheduler,
            allowed_schedulers=allowed_canonical or set(SUPPORTED_SCHEDULERS),
        )
    return specs


_SPECS = _build_specs()


def get_sampler_spec(name: str | None) -> SamplerSpec:
    if not isinstance(name, str):
        raise TypeError("sampler name must be a string")
    if not name:
        raise ValueError("sampler name must not be empty")
    canonical = name
    spec = _SPECS.get(canonical)
    if spec is None:
        raise ValueError(f"Unsupported sampler '{name}'. Valid: {[s for s in _SPECS]}")
    return spec


__all__ = ["SamplerSpec", "get_sampler_spec"]
