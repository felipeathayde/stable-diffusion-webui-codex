from __future__ import annotations

from typing import Dict, List, Set

SCHEDULER_OPTIONS: List[Dict[str, object]] = [
    {
        "name": "automatic",
        "label": "Automatic",
        "aliases": ["auto", "", "use same scheduler", "use same", "same", "default"],
    },
    {
        "name": "karras",
        "label": "Karras",
        "aliases": ["karras"],
    },
    {
        "name": "exponential",
        "label": "Exponential",
        "aliases": ["exp", "exponential"],
    },
    {
        "name": "simple",
        "label": "Simple",
        "aliases": ["simple", "linear"],
    },
    {
        "name": "euler_discrete",
        "label": "Euler (discrete)",
        "aliases": [
            "euler",
            "euler a",
            "eulerdiscretescheduler",
            "eulerancestraldiscretescheduler",
        ],
    },
]

_CANONICAL_NAMES: Set[str] = {entry["name"] for entry in SCHEDULER_OPTIONS}

SCHEDULER_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for entry in SCHEDULER_OPTIONS:
    canonical = entry["name"]
    if isinstance(entry.get("aliases"), list):
        aliases = [alias for alias in entry["aliases"] if isinstance(alias, str)]
    else:
        aliases = []
    for alias in [canonical, *aliases]:
        SCHEDULER_ALIAS_TO_CANONICAL[alias.strip().lower()] = canonical  # type: ignore[arg-type]

# Default scheduler per sampler when user selects "Automatic" or "Use same".
SAMPLER_DEFAULT_SCHEDULER: Dict[str, str] = {
    "dpm++ 2m": "karras",
    "dpm++ sde": "karras",
    "dpm++ 2m sde": "exponential",
    "dpm++ 2m sde heun": "exponential",
    "dpm++ 2s a": "karras",
    "dpm++ 3m sde": "exponential",
    "dpm2": "karras",
    "dpm2 a": "karras",
    "restart": "karras",
}

AUTO_TOKENS: Set[str] = {"automatic", "auto", "use same", "use same scheduler", "same", "default", ""}

__all__ = [
    "SCHEDULER_OPTIONS",
    "SCHEDULER_ALIAS_TO_CANONICAL",
    "SAMPLER_DEFAULT_SCHEDULER",
    "AUTO_TOKENS",
]
