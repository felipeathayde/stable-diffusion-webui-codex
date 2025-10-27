from __future__ import annotations

from importlib import import_module

from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError
from apps.backend.core.registry import EngineRegistry

from apps.backend.engines.wan22.wan22_14b import Wan2214BEngine
from apps.backend.engines.wan22.wan22_5b import Wan225BEngine


def register_default_engines(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    """Register the canonical set of engines into the provided registry."""

    try:
        registration = import_module("apps.server.backend.engines.registration")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("apps.server.backend.engines.registration is unavailable; cannot register default engines") from exc

    registration.register_sd15(registry=registry, replace=replace)
    registration.register_sdxl(registry=registry, replace=replace)
    registration.register_flux(registry=registry, replace=replace)
    registration.register_sd20(registry=registry, replace=replace)
    registration.register_sd35(registry=registry, replace=replace)
    # Optional engines are not auto-registered in strict mode (no silent fallbacks)
    registration.register_wan22_videos(registry=registry, replace=replace)


__all__ = [
    'EngineLoadError',
    'EngineExecutionError',
    'register_default_engines',
    'Wan2214BEngine',
    'Wan225BEngine',
    # Compatibility aliases
    'WanI2V14BEngine',
    'WanT2V14BEngine',
]

# Backward compatibility aliases (v1 naming): old WAN video engines mapped to WAN 2.2 implementations
WanI2V14BEngine = Wan2214BEngine  # image-to-video 14B
WanT2V14BEngine = Wan2214BEngine  # text-to-video 14B (unified engine supports both)
