"""WAN Video Sampling Module.

Flow-matching sampler para WAN 2.2 com latentes 5D [B, C, T, H, W].
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator, Optional

import torch
from torch import nn

logger = logging.getLogger("backend.runtime.wan22.sampler")


def get_flow_sigmas(
    num_steps: int,
    shift: float = 8.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate sigma schedule for flow-matching."""
    if device is None:
        device = torch.device("cpu")

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    sigmas = shift * timesteps / (1 + (shift - 1) * timesteps)
    return sigmas


class WanVideoSampler:
    """Flow-matching video sampler for WAN models."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device is None:
            device = torch.device("cuda")

        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self._logger = logging.getLogger(__name__)

    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        cond: torch.Tensor,
        uncond: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        cfg_scale: float = 7.5,
        flow_shift: float = 8.0,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Sample video latents using flow-matching Euler integration."""
        batch, _, _, _, _ = shape
        device = self.device
        dtype = self.dtype

        if seed is not None:
            torch.manual_seed(seed)

        latents = torch.randn(shape, device=device, dtype=dtype)
        sigmas = get_flow_sigmas(num_steps, shift=flow_shift, device=device, dtype=dtype)

        if uncond is None:
            uncond = torch.zeros_like(cond)

        self._logger.info(
            "WAN sampling: shape=%s steps=%d cfg=%.1f shift=%.1f",
            shape,
            num_steps,
            cfg_scale,
            flow_shift,
        )

        for step in range(num_steps):
            t_curr = sigmas[step]
            t_next = sigmas[step + 1]

            timestep = torch.full((batch,), float(t_curr), device=device, dtype=dtype)

            x_input = torch.cat([latents, latents], dim=0)
            cond_input = torch.cat([cond, uncond], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)

            v_pred = self.transformer(
                x_input,
                timestep_input,
                cond_input,
            )

            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            dt = float(t_next) - float(t_curr)
            latents = latents + dt * v

            if callback is not None:
                callback(step + 1, num_steps, latents)

            if (step + 1) % 5 == 0 or step == 0:
                self._logger.debug(
                    "Step %d/%d: t=%.4f->%.4f norm=%.2f",
                    step + 1,
                    num_steps,
                    float(t_curr),
                    float(t_next),
                    float(latents.norm()),
                )

        self._logger.info("WAN sampling complete")
        return latents


@torch.inference_mode()
def sample_txt2vid(
    transformer: nn.Module,
    vae: nn.Module,
    *,
    cond: torch.Tensor,
    uncond: Optional[torch.Tensor] = None,
    width: int = 768,
    height: int = 432,
    num_frames: int = 16,
    num_steps: int = 20,
    cfg_scale: float = 7.5,
    flow_shift: float = 8.0,
    seed: Optional[int] = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """High-level txt2vid sampling usando WanVideoSampler + VAE.

    `transformer` deve aceitar latentes [B, C, T, H, W], timesteps [B] e condicionamento [B, L, D].
    `vae` é tratado como decodificador de imagens; espera-se que exponha um método
    `decode(latents_4d)` que aceite [B, C, H, W] e produza [B, H, W, C] ou [B, C, H, W].
    """
    if device is None:
        device = torch.device("cuda")

    latent_h = height // 8
    latent_w = width // 8
    latent_c = 16

    batch = cond.shape[0]
    shape = (batch, latent_c, num_frames, latent_h, latent_w)

    sampler = WanVideoSampler(transformer, device=device, dtype=dtype)
    latents = sampler.sample(
        shape,
        cond=cond,
        uncond=uncond,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        seed=seed,
        callback=callback,
    )

    logger.info("Decoding latents through VAE")

    # Flatten time dimension and decode frame-wise through a VAE that expects 4D latents.
    batch_size, channels, frames, h_lat, w_lat = latents.shape
    latents_4d = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, h_lat, w_lat)

    decoded = vae.decode(latents_4d)
    if decoded.ndim != 4:
        raise RuntimeError(f"WAN22 VAE decode expected 4D tensor, got shape={tuple(decoded.shape)}")

    # Normalizar para formato [B, C, T, H, W]
    if decoded.shape[1] in (3, 4):
        # Canais-first: [B*T, C, H, W]
        video = decoded.view(batch_size, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3]).permute(
            0,
            2,
            1,
            3,
            4,
        )
    else:
        # Canais-last: [B*T, H, W, C]
        video = decoded.view(batch_size, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3]).permute(
            0,
            4,
            1,
            2,
            3,
        )

    return video.contiguous()
