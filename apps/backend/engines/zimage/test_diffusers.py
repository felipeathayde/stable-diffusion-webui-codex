"""Test endpoint for Diffusers bypass pipeline.

Usage from Python console or test script:
    from apps.backend.engines.zimage.test_diffusers import test_zimage_diffusers
    
    images = test_zimage_diffusers(
        "a beautiful sunset over mountains",
        seed=42,
    )
    images[0].save("test_output.png")
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("backend.zimage.test")


def test_zimage_diffusers(
    prompt: str,
    *,
    height: int = 1024, 
    width: int = 1024,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None,
    model_name: str = "z_image_turbo-Q8_0.gguf",
) -> list:
    """Test Z Image generation using Diffusers pipeline bypass.
    
    This loads the model and runs generation using Diffusers scheduler
    instead of Codex k-diffusion.
    
    Args:
        prompt: Text prompt
        height: Image height  
        width: Image width
        num_inference_steps: Sampling steps
        guidance_scale: CFG scale (0.0 for Turbo)
        seed: Random seed
        model_name: GGUF model filename
    
    Returns:
        List of PIL images
    """
    from apps.backend.core.model_registry import get_model_registry
    from apps.backend.engines.zimage import ZImageEngine
    
    logger.info("=== Testing Diffusers Bypass Pipeline ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Size: {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}")
    
    # Get model path
    registry = get_model_registry()
    model_info = registry.get_model_by_name(model_name)
    if not model_info:
        raise ValueError(f"Model not found: {model_name}")
    
    # Create engine and load
    engine = ZImageEngine()
    engine.load_model(model_info.path)
    
    try:
        # Run with Diffusers bypass
        images = engine.sample_with_diffusers(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        logger.info(f"Generation complete: {len(images)} images")
        return images
        
    finally:
        engine.unload_model()


if __name__ == "__main__":
    import sys
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a beautiful sunset over mountains"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    images = test_zimage_diffusers(prompt, seed=seed)
    
    output_path = f"test_zimage_diffusers_s{seed}.png"
    images[0].save(output_path)
    print(f"Saved: {output_path}")
