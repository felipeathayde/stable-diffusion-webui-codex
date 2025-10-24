from __future__ import annotations

from importlib import import_module

from apps.server.backend.core.exceptions import EngineExecutionError, EngineLoadError
from apps.server.backend.core.registry import EngineRegistry

from .diffusion.wan22_14b import Wan2214BEngine
from .diffusion.wan22_5b import Wan225BEngine


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
    registration.register_svd(registry=registry, replace=replace)
    registration.register_hunyuan_video(registry=registry, replace=replace)
    registration.register_wan22_videos(registry=registry, replace=replace)


__all__ = [
    'EngineLoadError',
    'EngineExecutionError',
    'register_default_engines',
    'Wan2214BEngine',
    'Wan225BEngine',
]
