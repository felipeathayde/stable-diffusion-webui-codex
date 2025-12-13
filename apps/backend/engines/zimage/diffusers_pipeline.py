"""Diffusers Pipeline Bypass for Z Image.

Uses Diffusers ZImagePipeline directly with our GGUF model loader.
This bypasses Codex's k-diffusion sampler and uses Diffusers scheduler instead.

Usage in txt2img:
    from apps.backend.engines.zimage.diffusers_pipeline import run_zimage_diffusers
    
    images = run_zimage_diffusers(
        transformer=loaded_gguf_transformer,
        text_encoder=loaded_qwen3,
        vae=loaded_vae,
        prompt="...",
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    )
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.z_image import ZImagePipeline
from transformers import AutoTokenizer

logger = logging.getLogger("backend.zimage.diffusers")


class TextEncoderOutput:
    """Simple output wrapper with hidden_states for Diffusers compatibility."""
    def __init__(self, hidden_states_list: List[torch.Tensor]):
        self.hidden_states = hidden_states_list


class DiffusersTextEncoderWrapper(nn.Module):
    """Wraps our text encoder to produce Diffusers-compatible output.
    
    Diffusers expects text_encoder(..., output_hidden_states=True).hidden_states[-2]
    Our encoder returns the tensor directly. This wrapper adapts the interface.
    """
    
    def __init__(self, text_encoder):
        super().__init__()
        self._encoder = text_encoder
    
    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self._encoder.parameters()).dtype
        except StopIteration:
            return torch.bfloat16
    
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        # Our encoder expects texts, but Diffusers passes input_ids
        # We need to get the embeddings directly from the model
        if hasattr(self._encoder, 'model'):
            model = self._encoder.model
            # Run forward on the underlying model
            if hasattr(model, 'forward'):
                result = model(input_ids=input_ids, attention_mask=attention_mask, 
                              output_hidden_states=True, **kwargs)
                if hasattr(result, 'hidden_states'):
                    return result
                # If result is tuple (hidden, intermediate), wrap it
                if isinstance(result, tuple):
                    # Assume second-to-last layer is in the result
                    hidden = result[1] if len(result) > 1 and result[1] is not None else result[0]
                    # Diffusers uses [-2], so we need at least 2 entries
                    return TextEncoderOutput([hidden, hidden])
        
        # Fallback: call encoder as-is and wrap result
        hidden = self._encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if isinstance(hidden, torch.Tensor):
            return TextEncoderOutput([hidden, hidden])
        return hidden


def create_zimage_pipeline(
    transformer: torch.nn.Module,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    tokenizer_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    enable_cpu_offload: bool = True,  # Enabled: model now has .dtype property
) -> ZImagePipeline:
    """Create a Diffusers ZImagePipeline with externally loaded components.
    
    Args:
        transformer: GGUF-loaded ZImageTransformer2DModel
        text_encoder: Qwen3 text encoder
        vae: Flux VAE
        tokenizer_path: Path to Qwen3 tokenizer
        device: Target device
        dtype: Computation dtype
        enable_cpu_offload: Enable automatic CPU offload for low VRAM
    
    Returns:
        ZImagePipeline ready for inference
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Create scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,  # Turbo uses linear schedule
    )
    
    # Don't move to device yet if using CPU offload
    if not enable_cpu_offload:
        transformer = transformer.to(device=device, dtype=dtype)
        text_encoder = text_encoder.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)
    
    # Wrap text encoder to adapt output format for Diffusers
    wrapped_text_encoder = DiffusersTextEncoderWrapper(text_encoder)
    
    # Create pipeline
    pipeline = ZImagePipeline(
        transformer=transformer,
        text_encoder=wrapped_text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    
    # Enable CPU offload (moves models to GPU only when needed)
    if enable_cpu_offload:
        logger.info("[diffusers-bypass] Enabling CPU offload...")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    
    return pipeline


def run_zimage_diffusers(
    transformer: torch.nn.Module,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    tokenizer_path: str,
    prompt: Union[str, List[str]],
    *,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,  # Turbo uses 0
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    output_type: str = "pil",
    **kwargs: Any,
) -> List[Any]:
    """Run Z Image generation using Diffusers pipeline directly.
    
    This bypasses all Codex sampling and uses Diffusers scheduler exactly.
    
    Args:
        transformer: GGUF-loaded ZImageTransformer2DModel
        text_encoder: Qwen3 text encoder  
        vae: Flux VAE
        tokenizer_path: Path to Qwen3 tokenizer
        prompt: Text prompt(s)
        negative_prompt: Negative prompt(s) for CFG
        height: Image height
        width: Image width
        num_inference_steps: Sampling steps (9 for Turbo)
        guidance_scale: CFG scale (0.0 for Turbo)
        generator: Optional RNG generator
        seed: Optional seed (creates generator if not provided)
        device: Target device
        dtype: Computation dtype
        output_type: "pil" or "latent"
    
    Returns:
        List of generated images
    """
    logger.info("[diffusers-bypass] Creating pipeline...")
    
    pipeline = create_zimage_pipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
    )
    
    # Create generator if seed provided
    if generator is None and seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    logger.info(
        "[diffusers-bypass] Running: %dx%d, steps=%d, cfg=%.1f",
        width, height, num_inference_steps, guidance_scale
    )
    
    # Run pipeline using Diffusers exactly
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type=output_type,
        **kwargs,
    )
    
    logger.info("[diffusers-bypass] Generation complete")
    return result.images


__all__ = ["create_zimage_pipeline", "run_zimage_diffusers"]
