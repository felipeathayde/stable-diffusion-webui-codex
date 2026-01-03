"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared Flow16 VAE utilities (16-channel latent AutoencoderKL) for Flux/Z Image families.
Defines the canonical Flow16 VAE config parity used by diffusers (no quant/post-quant conv), plus helpers to locate and load a Flow16 VAE
from either a diffusers directory or a single `.safetensors` file.

Symbols (top-level; keep in sync; no ghosts):
- `FLOW16_VAE_CONFIG` (constant): Canonical diffusers-like config dict for Flow16 VAEs (16 latent channels, scaling/shift factors).
- `load_flow16_vae` (function): Loads a Flow16 VAE from a directory or `.safetensors` file with strict latent-channel validation.
- `find_flow16_vae` (function): Searches candidate directories for a valid Flow16 VAE path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Mapping, Any

import torch

logger = logging.getLogger("backend.runtime.common.vae")


# Configuration for 16-channel flow-based VAE (used by Flux, Z Image)
# NOTE: Flow16 VAE config mirrors the canonical diffusers configs shipped for:
# - `apps/backend/huggingface/black-forest-labs/FLUX.1-dev/vae/config.json`
# - `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/vae/config.json`
#
# In particular: these VAEs disable quant/post-quant convs (`use_quant_conv=false`)
# so the weight files may legitimately omit `quant_conv.*` and `post_quant_conv.*`.
FLOW16_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,  # 16-channel latent space
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
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
    "use_post_quant_conv": False,
    "use_quant_conv": False,
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

    def _strip_known_prefixes(sd: Mapping[str, Any]) -> dict[str, Any]:
        """Strip common VAE prefixes (Comfy/SD checkpoints) to diffusers keys.

        Flow16 VAEs show up in a few layouts:
        - diffusers keys (encoder.*, decoder.*)
        - same keys prefixed with first_stage_model./vae./model./module.

        We normalise by repeatedly removing known prefixes.
        """
        prefixes = (
            "first_stage_model.",
            "vae.",
            "model.",
            "module.",
        )
        out: dict[str, Any] = {}
        for raw_key, value in sd.items():
            key = str(raw_key)
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                        changed = True
                        break
            out[new_key] = value
        return out
    
    try:
        if os.path.isdir(vae_path):
            # Diffusers directory format
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            # Single safetensors file - create with correct config
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
            if isinstance(state_dict, Mapping):
                state_dict = _strip_known_prefixes(state_dict)
                # Normalize LDM-style Flow16 VAEs (same conversion as SDXL/Flux).
                try:
                    from types import SimpleNamespace
                    from apps.backend.runtime.models.loader import _maybe_convert_sdxl_vae_state_dict
                    from apps.backend.runtime.model_registry.specs import ModelFamily

                    state_dict = _maybe_convert_sdxl_vae_state_dict(
                        state_dict,
                        SimpleNamespace(family=ModelFamily.ZIMAGE),
                    )
                except Exception as exc:
                    logger.debug("Flow16 VAE key conversion skipped/failed: %s", exc)
            
            vae = AutoencoderKL(**FLOW16_VAE_CONFIG)
            expected_total = len(vae.state_dict())
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning("VAE missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.debug("VAE unexpected keys (%d): %s", len(unexpected), unexpected[:5])

            # Fail loudly if this is not actually a Flow16 VAE.
            # A mismatched 4-channel VAE will otherwise decode pure noise.
            if missing:
                ratio = len(missing) / max(expected_total, 1)
                if ratio > 0.05:
                    raise ValueError(
                        f"Incompatible Flow16 VAE at {vae_path}: missing {len(missing)}/{expected_total} keys "
                        f"after prefix stripping. Please supply a 16-channel Flow VAE (Flux/Z Image)."
                    )
            
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
