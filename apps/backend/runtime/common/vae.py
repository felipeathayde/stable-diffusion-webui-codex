"""Shared VAE utilities for Flow-based models.

This module provides VAE loading and configuration for models that use
16-channel latent spaces (Flux, Z Image, etc).

The Flow16VAE is a standard AutoencoderKL with 16 latent channels instead
of the typical 4 channels used by SD1.x/SDXL.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("backend.runtime.common.vae")


# Configuration for 16-channel flow-based VAE (used by Flux, Z Image)
FLOW16_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "in_channels": 3,
    "latent_channels": 16,  # 16-channel latent space
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
}


def load_flow16_vae(
    vae_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> object:
    """Load a 16-channel flow VAE from a path.
    
    This function handles loading from:
    - Single .safetensors file
    - Diffusers directory format
    
    Args:
        vae_path: Path to VAE file or directory.
        dtype: Target dtype for the model.
        device: Target device (default: None = keep on CPU initially).
    
    Returns:
        Loaded AutoencoderKL model.
    
    Raises:
        ValueError: If loading fails.
    """
    from diffusers import AutoencoderKL
    
    logger.info("Loading Flow16 VAE from: %s", vae_path)
    
    try:
        if os.path.isdir(vae_path):
            # Diffusers directory format
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            # Single safetensors file - create with correct config
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
            
            vae = AutoencoderKL(**FLOW16_VAE_CONFIG)
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning("VAE missing keys: %s", missing[:5])
            if unexpected:
                logger.debug("VAE unexpected keys: %s", unexpected[:5])
            
            vae = vae.to(dtype=dtype)
        
        if device:
            vae = vae.to(device=device)
        
        param_count = sum(p.numel() for p in vae.parameters())
        logger.info("Loaded Flow16 VAE: %d params, dtype=%s", param_count, dtype)
        return vae
        
    except Exception as e:
        logger.error("Failed to load Flow16 VAE from %s: %s", vae_path, e)
        raise ValueError(f"Failed to load VAE from {vae_path}: {e}") from e


def find_flow16_vae(search_paths: list[str]) -> Optional[str]:
    """Find a Flow16 VAE in the given search paths.
    
    Args:
        search_paths: List of paths to search (files or directories).
    
    Returns:
        Path to VAE if found, None otherwise.
    """
    for path in search_paths:
        if not path:
            continue
            
        if os.path.isdir(path):
            # Check for diffusers-format VAE directory
            if os.path.exists(os.path.join(path, "config.json")):
                logger.info("Found VAE directory: %s", path)
                return path
            
            # Check for safetensors files
            for f in os.listdir(path):
                if f.endswith(".safetensors"):
                    vae_path = os.path.join(path, f)
                    logger.info("Found VAE file: %s", vae_path)
                    return vae_path
                    
        elif os.path.isfile(path) and path.endswith(".safetensors"):
            logger.info("Found VAE file: %s", path)
            return path
    
    return None


__all__ = [
    "FLOW16_VAE_CONFIG",
    "load_flow16_vae",
    "find_flow16_vae",
]
