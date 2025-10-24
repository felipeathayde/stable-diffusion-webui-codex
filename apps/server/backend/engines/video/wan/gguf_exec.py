from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WanStageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    seed: Optional[int]
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None
    lightning: Optional[bool] = None


@dataclass(frozen=True)
class WanGGUFRunConfig:
    # common
    width: int
    height: int
    fps: int
    num_frames: int
    guidance_scale: Optional[float]
    dtype: str
    device: str
    # text
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    # image
    init_image: Optional[object] = None
    # overrides
    vae_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    # stages
    high: Optional[WanStageConfig] = None
    low: Optional[WanStageConfig] = None


class GGUFExecutorUnavailable(RuntimeError):
    pass


def _import_executor():
    """Prefer native core if present, then plugin wrapper, else in-core.

    - backend_ext.wan_gguf_core: direct native executor (uses Diffusers VAE + GGUF UNet)
    - backend_ext.wan_gguf or wan_gguf: plugin wrapper that delegates to native or in-core
    - backend.engines.video.wan.gguf_incore: in-repo fallback (incomplete)
    """
    import importlib, logging
    log = logging.getLogger("backend.engines.video.wan.gguf_exec")

    # 1) Native core directly
    try:
        core = importlib.import_module("backend_ext.wan_gguf_core")
        if getattr(core, "IS_OPERATIONAL", False) and hasattr(core, "run_img2vid") and hasattr(core, "run_txt2vid"):
            log.info("[wan-gguf] using native core (backend_ext.wan_gguf_core)")
            return core
        if hasattr(core, "IS_OPERATIONAL") and not core.IS_OPERATIONAL:
            log.info("[wan-gguf] native core present but marked non-operational; skipping")
    except Exception as ex:
        log.info("[wan-gguf] native core import failed: %s", ex)

    # 2) Plugin wrapper (may fallback internally)
    for name in ("backend_ext.wan_gguf", "wan_gguf"):
        try:
            mod = importlib.import_module(name)
            return mod
        except Exception as ex:
            log.info("[wan-gguf] plugin import failed from %s: %s", name, ex)

    # 3) Fallback to in-core stub/executor-in-progress
    try:
        return importlib.import_module("apps.server.backend.engines.video.wan.gguf_incore")
    except Exception as ex:  # pragma: no cover
        raise GGUFExecutorUnavailable(
            f"WAN GGUF execution backend not available: {ex}"
        )


def run_txt2vid(cfg: WanGGUFRunConfig, logger) -> List[object]:
    mod = _import_executor()
    if not hasattr(mod, "run_txt2vid"):
        raise GGUFExecutorUnavailable("wan_gguf executor missing 'run_txt2vid'")
    frames = mod.run_txt2vid(cfg, logger=logger)  # type: ignore[attr-defined]
    if not isinstance(frames, (list, tuple)):
        raise RuntimeError("WAN GGUF executor returned invalid frames (expected list)")
    return list(frames)


def run_img2vid(cfg: WanGGUFRunConfig, logger) -> List[object]:
    mod = _import_executor()
    if not hasattr(mod, "run_img2vid"):
        raise GGUFExecutorUnavailable("wan_gguf executor missing 'run_img2vid'")
    frames = mod.run_img2vid(cfg, logger=logger)  # type: ignore[attr-defined]
    if not isinstance(frames, (list, tuple)):
        raise RuntimeError("WAN GGUF executor returned invalid frames (expected list)")
    return list(frames)
