from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from typing import List


def run_i2v_high(cfg, logger=None) -> List[object]:
    """Run High stage (skeleton): sets up scheduler and UNet, then raises until forward is mapped."""
    from .scheduler import EulerSimpleConfig, EulerSimpleStepper
    from .unet_gguf import GGUFUNet
    from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable

    log = logger
    if hasattr(log, "info"):
        log.info("[wan-gguf-core] I2V High: preparing Euler(Simple) stepper")

    steps = int(getattr(getattr(cfg, "high", None) or cfg, "steps", 12) or 12)
    e_cfg = EulerSimpleConfig(num_inference_steps=steps, guidance_scale=getattr(cfg, "guidance_scale", None))
    stepper = EulerSimpleStepper(e_cfg)
    stepper.set_timesteps(steps)

    if hasattr(log, "info"):
        log.info("[wan-gguf-core] I2V High: loading GGUF UNet from %s", getattr(getattr(cfg, "high", None), "model_dir", None))
    unet = GGUFUNet(getattr(getattr(cfg, "high", None), "model_dir", None) or "", logger=logger)

    # Until forward is implemented, abort with explicit message
    raise GGUFExecutorUnavailable("GGUF UNet forward mapping pending (High stage)")
