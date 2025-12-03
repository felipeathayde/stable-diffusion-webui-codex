# tags: sampling, registry, samplers
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.sampling.catalog import (
    AUTO_TOKENS,
    SAMPLER_ALIAS_TO_CANONICAL,
    SAMPLER_DEFAULT_SCHEDULER,
    SAMPLER_OPTIONS,
    SUPPORTED_SCHEDULERS,
)


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    kind: SamplerKind
    aliases: Set[str]
    default_scheduler: Optional[str]
    allowed_schedulers: Set[str]

    def is_supported_scheduler(self, scheduler: str | None) -> bool:
        if scheduler is None:
            return True
        key = scheduler.strip().lower()
        if key in AUTO_TOKENS:
            return True
        return key in self.allowed_schedulers


_SPECS: Dict[str, SamplerSpec] = {}


def _build_specs() -> Dict[str, SamplerSpec]:
    specs: Dict[str, SamplerSpec] = {}
    from apps.backend.runtime.sampling.catalog import SCHEDULER_ALIAS_TO_CANONICAL

    for entry in SAMPLER_OPTIONS:
        name = str(entry["name"])
        aliases = {name, *[str(a) for a in entry.get("aliases", [])]}
        raw_allowed: Iterable[str] = entry.get("schedulers", SUPPORTED_SCHEDULERS)
        allowed_canonical: Set[str] = set()
        for sched in raw_allowed:
            key = str(sched).strip().lower()
            canonical = SCHEDULER_ALIAS_TO_CANONICAL.get(key, key)
            if canonical in SUPPORTED_SCHEDULERS:
                allowed_canonical.add(canonical)
        default_scheduler = SAMPLER_DEFAULT_SCHEDULER.get(name)
        try:
            kind = SamplerKind.from_string(name)
        except Exception:
            # Keep unknown kinds for future porting but skip active registry
            continue
        specs[name] = SamplerSpec(
            name=name,
            kind=kind,
            aliases={a.strip().lower() for a in aliases},
            default_scheduler=default_scheduler,
            allowed_schedulers=allowed_canonical or set(SUPPORTED_SCHEDULERS),
        )
    return specs


_SPECS = _build_specs()


def get_sampler_spec(name: str | None) -> SamplerSpec:
    key = (name or "automatic").strip().lower()
    canonical = SAMPLER_ALIAS_TO_CANONICAL.get(key, key)
    spec = _SPECS.get(canonical)
    if spec is None:
        raise ValueError(f"Unsupported sampler '{name}'. Valid: {[s for s in _SPECS]}")
    return spec


__all__ = ["SamplerSpec", "get_sampler_spec"]
