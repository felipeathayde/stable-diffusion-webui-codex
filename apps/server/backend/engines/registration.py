from __future__ import annotations

from typing import Optional

from apps.server.backend.core.registry import EngineRegistry, register_engine


def _reg(key: str, cls, *, registry: Optional[EngineRegistry], replace: bool, aliases: tuple[str, ...] = ()) -> None:
    if registry is None:
        register_engine(key, cls, aliases=aliases, replace=replace)
    else:
        registry.register(key, cls, aliases=aliases, replace=replace)


def register_sd15(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.sd15 import StableDiffusion
    _reg("sd15", StableDiffusion, registry=registry, replace=replace, aliases=("sd-1.5",))


def register_sd20(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.sd20 import StableDiffusion2
    _reg("sd20", StableDiffusion2, registry=registry, replace=replace, aliases=("sd-2.0", "sd-2.1"))


def register_sdxl(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
    _reg("sdxl", StableDiffusionXL, registry=registry, replace=replace, aliases=("sd-xl",))
    _reg("sdxl_refiner", StableDiffusionXLRefiner, registry=registry, replace=replace, aliases=("sd-xl-refiner",))


def register_sd35(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.sd35 import StableDiffusion3
    _reg("sd35", StableDiffusion3, registry=registry, replace=replace, aliases=("sd-3.5", "sd3"))


def register_flux(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.flux import Flux
    _reg("flux", Flux, registry=registry, replace=replace, aliases=("flux.1", "flux-1"))


def register_chroma(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.chroma import Chroma
    _reg("chroma", Chroma, registry=registry, replace=replace)


def register_svd(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    try:
        from .video.svd.engine import SvdEngine  # type: ignore
        _reg("svd", SvdEngine, registry=registry, replace=replace)
    except Exception:
        return


def register_hunyuan_video(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    try:
        from .video.hunyuan.engine import HunyuanVideoEngine  # type: ignore
        _reg("hunyuan_video", HunyuanVideoEngine, registry=registry, replace=replace, aliases=("hunyuan",))
    except Exception:
        return


def register_wan22_videos(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from .diffusion.wan22_14b import Wan2214BEngine
    from .diffusion.wan22_5b import Wan225BEngine
    _reg("wan22_14b", Wan2214BEngine, registry=registry, replace=replace, aliases=("wan22-14b",))
    _reg("wan22_5b", Wan225BEngine, registry=registry, replace=replace, aliases=("wan22-5b",))

