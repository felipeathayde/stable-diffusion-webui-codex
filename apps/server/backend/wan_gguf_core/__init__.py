from __future__ import annotations

"""WAN 2.2 — Native GGUF Core (skeleton, CUDA 12.8, bf16 preferred).

Status
- This is a non-operational skeleton meant to evolve into the high-performance
  GGUF executor. It declares IS_OPERATIONAL=False so the plugin wrapper will
  fall back to the in-core Python executor until functionality is ready.

Goals
- Avoid materializing attention intermediates by relying on PyTorch SDPA /
  FlashAttention backends when available.
- Stage buffers/workspace on GPU to minimize reallocations.
- Support bf16/fp16, Euler (Simple) scheduler, and staged High→Low I2V.

Design (incremental)
- Python front-ends call into the native PyTorch operators exclusively. We do
  not author custom CUDA kernels for functionality already provided by PyTorch.
"""

from typing import Any, List

# The plugin wrapper will check this flag; keep False until a minimal forward
# path exists. This avoids hijacking execution prematurely.
IS_OPERATIONAL = False


def _get_logger(logger: Any):
    import logging
    return logger or logging.getLogger("wan_gguf_native")


def device_capabilities():
    """Return runtime device capability summary (dtype/backends) for logs."""
    try:
        import torch
        cap = {
            "cuda": torch.cuda.is_available(),
            "bf16": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
            "sm": getattr(torch.cuda.get_device_capability(), "__call__", None)() if torch.cuda.is_available() else None,  # type: ignore
            "flash_sdp": getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)(),
        }
        return cap
    except Exception:
        return {"cuda": False}


def _ensure_flash_attention_enabled(log):
    """Enable SDPA Flash backend if available (no intermediates materialized)."""
    try:
        import torch
        # Prefer flash kernels (PyTorch SDPA will choose the best available)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        log.info("[wan-gguf-core] SDPA flash enabled (no mat. of attention probs)")
    except Exception as ex:
        log.warning("[wan-gguf-core] Could not enable flash SDPA: %s", ex)


def _fa_sdpa(q, k, v, *, is_causal=False, dropout_p=0.0):
    """Attention via PyTorch SDPA (leverages FlashAttention when available).

    This path avoids allocating the NxN attention matrix explicitly and uses
    the fused flash/mem-efficient kernels when enabled.
    """
    import torch
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
    )


def _workspace_like(x, size, *, dtype=None):
    """Allocate a persistent on-device workspace buffer (acts as fast scratch)."""
    import torch
    return torch.empty(size, device=x.device, dtype=dtype or x.dtype)


def run_txt2vid(cfg, logger=None) -> List[object]:
    log = _get_logger(logger)
    # Skeleton: refuse execution until forward is wired, but provide rich hints.
    cap = device_capabilities()
    _ensure_flash_attention_enabled(log)
    from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable  # late import
    raise GGUFExecutorUnavailable(
        "WAN native core requires precomputed text context for cross-attention; not wired yet. "
        "Forward mapping for cross-attn is implemented, but text encoder integration is pending."
    )


def run_img2vid(cfg, logger=None) -> List[object]:
    # Prepare text context using the high stage dir to ensure dims match
    log = _get_logger(logger)
    model_dir = (
        getattr(getattr(cfg, "high", None), "model_dir", None)
        or getattr(getattr(cfg, "low", None), "model_dir", None)
        or getattr(cfg, "model_dir", None)
    )
    if not model_dir:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable  # late import
        raise GGUFExecutorUnavailable("No model_dir found to prepare text context")

    # Get prompt/negative from cfg
    prompt = getattr(cfg, "prompt", "")
    negative = getattr(cfg, "negative_prompt", None)
    device = getattr(cfg, "device", "cuda")
    dtype = getattr(cfg, "dtype", "bf16")

    # Prepare text context
    try:
        from .text_context import get_text_context
        import inspect
        kwargs = dict(
            device=device, dtype=dtype,
            text_encoder_dir=getattr(cfg, 'text_encoder_dir', None),
            tokenizer_dir=getattr(cfg, 'tokenizer_dir', None),
            vae_dir=getattr(cfg, 'vae_dir', None),
        )
        sig = inspect.signature(get_text_context)
        if 'model_key' in sig.parameters:
            kwargs['model_key'] = 'wan_i2v_14b'
        pctx, nctx = get_text_context(model_dir, prompt, negative, **kwargs)
        log.info("[wan-gguf-core] text context prepared: %s", tuple(pctx.shape) if hasattr(pctx, 'shape') else type(pctx))
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable  # late import
        raise GGUFExecutorUnavailable(f"Failed to prepare text context: {ex}")

    # Build UNet and scheduler loop
    from .unet_gguf import GGUFUNet
    from .scheduler import EulerSimpleConfig, EulerSimpleStepper
    from .loop import DiffusionLoop, LoopConfig
    from .patch import patch_embed, patch_unembed
    from .latents import encode_init_image_to_latents, decode_latents_to_images
    import torch

    unet = GGUFUNet(getattr(getattr(cfg, "high", None), "model_dir", model_dir) or model_dir, logger=log)
    steps = int(getattr(getattr(cfg, "high", None), "steps", getattr(cfg, "steps", 12)) or 12)
    stepper = EulerSimpleStepper(EulerSimpleConfig(num_inference_steps=steps, guidance_scale=getattr(cfg, "guidance_scale", None)))
    loop = DiffusionLoop(stepper, logger=log)

    B = 1
    T = int(getattr(cfg, "num_frames", 16) or 16)
    H = int(getattr(cfg, "height", 432) or 432)
    W = int(getattr(cfg, "width", 768) or 768)

    # Prepare initial latents per-frame (img2vid: encode init_image then tile across T)
    try:
        if getattr(cfg, "init_image", None) is None:
            raise RuntimeError("init_image required for img2vid")
        lat0 = encode_init_image_to_latents(getattr(cfg, "init_image"), device=device, dtype=dtype, vae_dir=getattr(cfg, 'vae_dir', None))  # [B?, C?, H', W']
        if not hasattr(lat0, 'shape'):
            raise RuntimeError("unexpected latents shape")
        lat0 = lat0 if lat0.dim() == 4 else lat0.view(1, *lat0.shape)
        # Compose video latents [B,C,T,H',W'] by tiling across time
        vH, vW = lat0.shape[-2], lat0.shape[-1]
        video_lat = lat0[:, :, None, :, :].repeat(1, 1, T, 1, 1)
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
        raise GGUFExecutorUnavailable(f"Failed to prepare initial latents: {ex}")

    # Patch-embed to tokens
    try:
        pe_w = unet.state.get('patch_embedding.weight')
        pe_b = unet.state.get('patch_embedding.bias')
        if pe_w is None or pe_b is None:
            raise RuntimeError('patch_embedding.* not found in GGUF')
        tokens, grid = patch_embed(video_lat, pe_w, pe_b)  # [B, L, C]
        log.info("[wan-gguf-core] tokens: %s, grid=%s", tuple(tokens.shape), grid)
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
        raise GGUFExecutorUnavailable(f"Patch embedding failed: {ex}")

    # Run diffusion loop in token space
    try:
        sample_tokens = loop.run(unet, tokens, pctx, nctx, cfg=LoopConfig(steps=steps, guidance_scale=getattr(cfg, "guidance_scale", None), dtype=dtype, device=device))
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
        raise GGUFExecutorUnavailable(f"Diffusion loop failed: {ex}")

    # Unembed back to video latents
    try:
        video_out = patch_unembed(sample_tokens, pe_w, grid)  # [B,C,T,H',W']
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
        raise GGUFExecutorUnavailable(f"Patch unembedding failed: {ex}")

    # Decode frames via VAE (High)
    try:
        frames_hi = decode_latents_to_images(video_out, model_dir=model_dir, device=device, dtype=dtype, vae_dir=getattr(cfg, 'vae_dir', None))
    except Exception as ex:
        from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
        raise GGUFExecutorUnavailable(f"VAE decode failed (High): {ex}")

    # If low stage provided, refine using last frame as seed
    low_cfg = getattr(cfg, "low", None)
    if low_cfg and getattr(low_cfg, "model_dir", None):
        try:
            # Seed from last frame of High
            seed_img = frames_hi[-1]
            lat_seed = encode_init_image_to_latents(seed_img, device=device, dtype=dtype, vae_dir=getattr(cfg, 'vae_dir', None))
            lat_seed = lat_seed if lat_seed.dim() == 4 else lat_seed.view(1, *lat_seed.shape)
            T = int(getattr(cfg, "num_frames", 16) or 16)
            video_lat_lo = lat_seed[:, :, None, :, :].repeat(1, 1, T, 1, 1)

            unet_lo = GGUFUNet(getattr(low_cfg, "model_dir"), logger=log)
            steps_lo = int(getattr(low_cfg, "steps", getattr(cfg, "steps", 12)) or 12)
            loop_lo = DiffusionLoop(EulerSimpleStepper(EulerSimpleConfig(num_inference_steps=steps_lo, guidance_scale=getattr(low_cfg, "cfg_scale", getattr(cfg, "guidance_scale", None)))), logger=log)

            pe_w = unet_lo.state.get('patch_embedding.weight')
            pe_b = unet_lo.state.get('patch_embedding.bias')
            if pe_w is None or pe_b is None:
                raise RuntimeError('patch_embedding.* not found in GGUF (Low)')
            tokens_lo, grid_lo = patch_embed(video_lat_lo, pe_w, pe_b)
            sample_tokens_lo = loop_lo.run(unet_lo, tokens_lo, pctx, nctx, cfg=LoopConfig(steps=steps_lo, guidance_scale=getattr(low_cfg, "cfg_scale", getattr(cfg, "guidance_scale", None)), dtype=dtype, device=device))
            video_out_lo = patch_unembed(sample_tokens_lo, pe_w, grid_lo)
            frames_lo = decode_latents_to_images(video_out_lo, model_dir=model_dir, device=device, dtype=dtype, vae_dir=getattr(cfg, 'vae_dir', None))
            return frames_lo
        except Exception as ex:
            from apps.server.backend.engines.video.wan.gguf_exec import GGUFExecutorUnavailable
            raise GGUFExecutorUnavailable(f"Low stage failed: {ex}")

    return frames_hi


__all__ = [
    "IS_OPERATIONAL",
    "run_txt2vid",
    "run_img2vid",
]
