"""Payload validation key definitions.

These keys are concatenated to build validation sets.
HIRES reuses keys from CORE (prompt, steps, sampler, scheduler).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ShaKeys:
    """SHA256 keys for asset selection."""
    MODEL: FrozenSet[str] = frozenset({"model_sha"})
    TENC: FrozenSet[str] = frozenset({"tenc_sha"})
    VAE: FrozenSet[str] = frozenset({"vae_sha"})
    LORA: FrozenSet[str] = frozenset({"lora_sha"})
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.MODEL | self.TENC | self.VAE | self.LORA


@dataclass(frozen=True)
class Txt2ImgKeys:
    """Keys for txt2img payload."""
    
    # Generation params (all models) - reused by HIRES
    CORE: FrozenSet[str] = frozenset({
        "prompt",
        "width",
        "height",
        "steps",
        "sampler",
        "scheduler",
        "seed",
        "clip_skip",
        "styles",
        "metadata",
    })
    
    # Diffusion (SD, SDXL)
    DIFFUSION: FrozenSet[str] = frozenset({
        "negative_prompt",
        "cfg",
    })
    
    # Flow (Flux, Z Image)
    FLOW: FrozenSet[str] = frozenset({
        "distilled_cfg",
    })
    
    # Hires-only (CORE is shared)
    HIRES: FrozenSet[str] = frozenset({
        "enable",
        "denoise",
        "scale",
        "resize_x",
        "resize_y",
        "upscaler",
        "checkpoint",
        "modules",
    })
    
    # Infra
    DEVICE: FrozenSet[str] = frozenset({"device"})
    MODEL: FrozenSet[str] = frozenset({"engine", "model"})
    SMART: FrozenSet[str] = frozenset({"smart_offload", "smart_fallback", "smart_cache"})
    
    # Extras container (passed through to engine)
    EXTRAS: FrozenSet[str] = frozenset({"extras"})
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.CORE | self.DIFFUSION | self.FLOW | self.DEVICE | self.MODEL | self.SMART | self.EXTRAS
    
    @property
    def HIRES_ALL(self) -> FrozenSet[str]:
        """Hires uses CORE + DIFFUSION + HIRES-specific."""
        return self.CORE | self.DIFFUSION | self.HIRES


@dataclass(frozen=True)
class ExtrasKeys:
    """Keys for payload.extras."""
    
    COMMON: FrozenSet[str] = frozenset({
        "highres",
        "hires",
        "refiner",
        "text_encoder_override",
        "batch_size",
        "batch_count",
        "randn_source",
        "eta_noise_seed_delta",
    })
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.COMMON | SHA_KEYS.ALL


# Singletons
SHA_KEYS = ShaKeys()
TXT2IMG_KEYS = Txt2ImgKeys()
EXTRAS_KEYS = ExtrasKeys()

__all__ = ["ShaKeys", "Txt2ImgKeys", "ExtrasKeys", "SHA_KEYS", "TXT2IMG_KEYS", "EXTRAS_KEYS"]
