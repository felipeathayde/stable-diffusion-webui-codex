from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from typing import Any, Optional, Tuple


def _map_dtype(dtype: str):
    import torch
    return {
        "bf16": getattr(torch, "bfloat16", torch.float16),
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.float16)


def _load_vae(vae_path: Optional[str], *, torch_dtype):
    """Load Wan VAE from an explicit directory or safetensors file."""
    import os
    from diffusers import AutoencoderKLWan  # type: ignore

    if not vae_path:
        raise RuntimeError("wan_vae_dir is required when running WAN GGUF (VAE path missing)")

    path = os.path.expanduser(str(vae_path))
    if os.path.isdir(path):
        return AutoencoderKLWan.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if os.path.isfile(path):
        loader = getattr(AutoencoderKLWan, "from_single_file", None)
        if loader is None:
            raise RuntimeError(
                f"AutoencoderKLWan.from_single_file not available; provide a directory instead of file: {path}"
            )
        return loader(path, torch_dtype=torch_dtype)  # type: ignore[misc]

    raise RuntimeError(f"VAE path not found: {path}")


def _get_scale_shift(vae) -> Tuple[float, float]:
    """Return (scaling_factor, shift_factor) from VAE config with sane defaults."""
    try:
        cfg = getattr(vae, 'config', None) or {}
        sf = float(getattr(cfg, 'scaling_factor', getattr(cfg, 'scaling_factor', 0.18215)))
        sh = float(getattr(cfg, 'shift_factor', getattr(cfg, 'shift_factor', 0.0)))
        # Some configs store as dict
        if isinstance(cfg, dict):
            sf = float(cfg.get('scaling_factor', sf))
            sh = float(cfg.get('shift_factor', sh))
        return sf, sh
    except Exception:
        return 0.18215, 0.0


def encode_init_image_to_latents(init_image: Any, *, device: str, dtype: str, vae_dir: str | None = None, logger: Any | None = None):
    """Encode an initial image to latents using AutoencoderKLWan (diffusers).

    dtype: 'bf16' | 'fp16' | 'fp32'
    """
    import torch

    torch_dtype = _map_dtype(dtype)

    # We expect the VAE to be colocated with the model (current working dir or model dir)
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    sf, sh = _get_scale_shift(vae)
    if logger is not None:
        try:
            logger.info("[wan22.gguf] VAE encode scale=%.6f shift=%.6f", sf, sh)
        except Exception:
            pass
    target = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    vae = vae.to(device=target, dtype=torch_dtype)

    # Accept PIL.Image or numpy arrays and convert to torch tensor in [-1, 1]
    if not hasattr(init_image, "to"):
        try:
            from PIL import Image
            import numpy as np
            if isinstance(init_image, Image.Image):
                img = init_image.convert('RGB')
                arr = np.array(img).astype('float32') / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
                t = t.to(target).to(torch_dtype)
                init_image = t * 2.0 - 1.0
            else:
                # Try numpy array HxWxC or CxHxW
                arr = np.asarray(init_image).astype('float32')
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    arr = arr / 255.0 if arr.max() > 1.0 else arr
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                    t = torch.from_numpy(arr).unsqueeze(0)
                else:
                    raise RuntimeError("unsupported init_image array shape")
                t = t.to(target).to(torch_dtype)
                init_image = t * 2.0 - 1.0
        except Exception as ex:
            raise RuntimeError(f"init_image must be a tensor or PIL/numpy convertible: {ex}")

    with torch.no_grad():
        latents = vae.encode(init_image).latent_dist.sample()
        latents = (latents - sh) * sf
    return latents


def decode_latents_to_images(video_latents: Any, *, model_dir: str, device: str, dtype: str, vae_dir: str | None = None, logger: Any | None = None):
    """Decode video latents [B,C,T,H,W] into a list of PIL images (per frame).

    Uses AutoencoderKLWan.decode frame-by-frame to limit memory.
    """
    import torch
    from PIL import Image

    torch_dtype = _map_dtype(dtype)
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    sf, sh = _get_scale_shift(vae)
    if logger is not None:
        try:
            logger.info("[wan22.gguf] VAE decode scale=%.6f shift=%.6f", sf, sh)
        except Exception:
            pass
    vae = vae.to(device=dev, dtype=torch_dtype)
    B, C, T, H, W = video_latents.shape
    frames: list[Image.Image] = []
    with torch.no_grad():
        for t in range(T):
            lat = video_latents[:, :, t]
            lat = (lat / sf) + sh
            img = vae.decode(lat).sample  # [B,3,H*,W*]
            img0 = img[0].detach().clamp(0,1)
            arr = (img0.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
            frames.append(Image.fromarray(arr))
    return frames
