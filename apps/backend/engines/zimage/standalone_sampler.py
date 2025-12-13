"""Standalone Z Image Sampler using Diffusers math.

Uses OUR loader/encoder/transformer but follows Diffusers scheduler exactly.
This bypasses all k-diffusion machinery and uses FlowMatchEulerDiscreteScheduler.

Key insight from Diffusers pipeline_z_image.py:
- noise_pred = -model_output (line 558)
- scheduler.step(noise_pred, t, latents) (line 561)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

logger = logging.getLogger("backend.zimage.standalone")


def sample_zimage_diffusers_math(
    transformer: torch.nn.Module,
    text_embeddings: torch.Tensor,
    *,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    latent_channels: int = 16,
    patch_size: int = 2,
) -> torch.Tensor:
    """Sample using Diffusers FlowMatchEulerDiscreteScheduler.
    
    This is a standalone sampler that:
    - Uses OUR transformer directly (GGUF-loaded ZImageTransformer2DModel)
    - Uses OUR text embeddings (from ZImageTextEncoder)
    - Uses Diffusers scheduler for timestep/sigma management
    - Applies NEGATION to model output as Diffusers does
    
    Args:
        transformer: Our ZImageTransformer2DModel
        text_embeddings: Pre-encoded text embeddings from our encoder [B, seq, hidden]
        height: Image height
        width: Image width
        num_inference_steps: Sampling steps (9 for Turbo)
        guidance_scale: CFG scale (0.0 for Turbo)
        generator: Optional RNG generator
        device: Target device
        dtype: Computation dtype
        latent_channels: Number of latent channels (16)
        patch_size: Patch size (2)
        
    Returns:
        latents: Final denoised latents [B, C, H//8, W//8]
    """
    batch_size = text_embeddings.shape[0]
    
    # Calculate latent dimensions (VAE downscale = 8)
    vae_scale = 8
    latent_height = height // vae_scale
    latent_width = width // vae_scale
    
    # Create scheduler (shift=1.0 for Turbo)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    logger.info(
        "[diffusers-sampler] steps=%d, timesteps=%s",
        num_inference_steps, 
        [round(float(t), 3) for t in timesteps[:4].tolist()]
    )
    
    # Initialize latents
    latents_shape = (batch_size, latent_channels, latent_height, latent_width)
    latents = torch.randn(
        latents_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,  # Scheduler expects float32
    )
    
    # Flow-matching starts with pure noise (sigma=1), no scaling needed
    
    logger.info("[diffusers-sampler] latents shape=%s", latents.shape)
    
    # Sampling loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Prepare model input
            latent_model_input = latents.to(dtype)
            
            # Create timestep tensor for transformer
            timestep_tensor = t.expand(batch_size).to(device)
            
            # Call our transformer
            # OUR transformer forward (model.py lines 788-795) expects:
            # - sigma in [1→0] where 1=start/noise, 0=end/clean
            # - It does: t_scaled = sigma * time_scale (1000)
            # 
            # Scheduler returns t: 1000=start → 0=end
            # So: sigma = t/1000 gives us sigma=1 at start, sigma=0 at end
            sigma = float(t) / 1000.0
            sigma_tensor = torch.full((batch_size,), sigma, device=device, dtype=dtype)
            
            model_output = transformer(
                latent_model_input,
                sigma_tensor,
                context=text_embeddings.to(dtype),
            )
            
            # CRITICAL: Negate model output as Diffusers does!
            # From pipeline_z_image.py line 558: noise_pred = -noise_pred
            noise_pred = -model_output.float()
            
            # Compute previous sample using scheduler
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Log progress
            if i < 3 or i == len(timesteps) - 1:
                norm_x = float(latents.norm())
                norm_pred = float(noise_pred.norm())
                logger.info(
                    "[diffusers-sampler] step=%d/%d t=%.3f norm(x)=%.2f norm(pred)=%.2f",
                    i + 1, len(timesteps), float(t), norm_x, norm_pred
                )
    
    return latents


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images using VAE.
    
    Args:
        vae: Our VAE wrapper with .decode() and .first_stage_model
        latents: Latents [B, C, H, W] in normalized space
        
    Returns:
        images: Tensor [B, 3, H*8, W*8] in range [0, 1]
    """
    # Denormalize latents for VAE
    # VAE expects: (latents / scaling_factor) + shift_factor
    scaling_factor = 0.3611  # From flux VAE
    shift_factor = 0.1159
    
    latents = (latents / scaling_factor) + shift_factor
    
    # Decode
    images = vae.decode(latents)
    
    # Convert to [0, 1] range
    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)
    
    return images


__all__ = ["sample_zimage_diffusers_math", "decode_latents"]
