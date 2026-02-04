"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical engine registration functions for the backend.
Defines one `register_<engine>(...)` helper per engine family and wires aliases into the shared `EngineRegistry`.

Symbols (top-level; keep in sync; no ghosts):
- `register_sd15` (function): Registers the SD 1.5 engine and aliases.
- `register_sd20` (function): Registers the SD 2.x engine and aliases.
- `register_sdxl` (function): Registers SDXL base/refiner engines and aliases.
- `register_sd35` (function): Registers the SD 3.5 engine and aliases.
- `register_flux` (function): Registers the Flux engine.
- `register_kontext` (function): Registers the Flux Kontext engine.
- `register_chroma` (function): Registers the Chroma engine.
- `register_wan22_videos` (function): Registers WAN22 video engines (5B/animate) and aliases.
- `register_wan22_14b_experimental` (function): Registers the experimental WAN22 14B engine (not registered by default).
- `register_zimage` (function): Registers the Z-Image engine and aliases.
"""

from __future__ import annotations

from typing import Optional

from apps.backend.core.registry import EngineRegistry, register_engine


def _reg(key: str, cls, *, registry: Optional[EngineRegistry], replace: bool, aliases: tuple[str, ...] = ()) -> None:
    if registry is None:
        register_engine(key, cls, aliases=aliases, replace=replace)
    else:
        registry.register(key, cls, aliases=aliases, replace=replace)


def register_sd15(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sd15 import StableDiffusion
    _reg("sd15", StableDiffusion, registry=registry, replace=replace, aliases=("sd-1.5",))


def register_sd20(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sd20 import StableDiffusion2
    _reg("sd20", StableDiffusion2, registry=registry, replace=replace, aliases=("sd-2.0", "sd-2.1"))


def register_sdxl(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
    _reg("sdxl", StableDiffusionXL, registry=registry, replace=replace, aliases=("sd-xl",))
    _reg("sdxl_refiner", StableDiffusionXLRefiner, registry=registry, replace=replace, aliases=("sd-xl-refiner",))


def register_sd35(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sd35 import StableDiffusion3
    _reg("sd35", StableDiffusion3, registry=registry, replace=replace, aliases=("sd-3.5", "sd3"))


def register_flux(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.flux import Flux
    _reg("flux1", Flux, registry=registry, replace=replace, aliases=())


def register_kontext(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.kontext import Kontext
    _reg(
        "flux1_kontext",
        Kontext,
        registry=registry,
        replace=replace,
        aliases=(),
    )


def register_chroma(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.chroma import Chroma
    _reg("flux1_chroma", Chroma, registry=registry, replace=replace, aliases=())


def register_svd(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    from .video.svd.engine import SvdEngine  # type: ignore
    _reg("svd", SvdEngine, registry=registry, replace=replace)


def register_hunyuan_video(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    from .video.hunyuan.engine import HunyuanVideoEngine  # type: ignore
    _reg("hunyuan_video", HunyuanVideoEngine, registry=registry, replace=replace, aliases=("hunyuan",))


def register_wan22_videos(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.wan22.wan22_5b import Wan225BEngine
    from apps.backend.engines.wan22.wan22_animate_14b import Wan22Animate14BEngine
    _reg("wan22_5b", Wan225BEngine, registry=registry, replace=replace, aliases=("wan22-5b",))
    _reg(
        "wan22_animate_14b",
        Wan22Animate14BEngine,
        registry=registry,
        replace=replace,
        aliases=("wan22-animate-14b", "wan-animate"),
    )


def register_wan22_14b_experimental(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.wan22.wan22_14b import Wan2214BEngine
    _reg("wan22_14b", Wan2214BEngine, registry=registry, replace=replace, aliases=("wan22-14b",))


def register_zimage(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.zimage.zimage import ZImageEngine
    _reg("zimage", ZImageEngine, registry=registry, replace=replace, aliases=("z-image", "z-image-turbo"))
