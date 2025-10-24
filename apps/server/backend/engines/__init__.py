from __future__ import annotations

from importlib import import_module

from apps.server.backend.core.exceptions import EngineExecutionError, EngineLoadError
from apps.server.backend.core.registry import EngineRegistry

from .video.wan.i2v14b_engine import WanI2V14BEngine
from .video.wan.t2v14b_engine import WanT2V14BEngine


def register_default_engines(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    """Register the canonical set of engines into the provided registry."""

    try:
        registration = import_module("backend.engines.registration")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("backend.engines.registration is unavailable; cannot register default engines") from exc

    registration.register_sd15(registry=registry, replace=replace)
    registration.register_sdxl(registry=registry, replace=replace)
    registration.register_flux(registry=registry, replace=replace)
    registration.register_svd(registry=registry, replace=replace)
    registration.register_hunyuan_video(registry=registry, replace=replace)
    registration.register_wan_videos(registry=registry, replace=replace)


__all__ = [
    'EngineLoadError',
    'EngineExecutionError',
    'register_default_engines',
    'WanI2V14BEngine',
    'WanT2V14BEngine',
]
