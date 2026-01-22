"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF VAE IO helpers (I2V condition encode + decode to frames).
Loads the WAN VAE via Diffusers, applies latent normalization, and converts between latents and RGB frames for the WAN22 GGUF runtime.

Symbols (top-level; keep in sync; no ghosts):
- `load_vae` (function): Loads the WAN VAE component (from directory or single-file weights).
- `vae_encode_video_condition` (function): Encodes the Diffusers-style I2V conditioning video into latents (deterministic mode).
- `vae_decode_video` (function): Decodes video latents to frames; can validate the expected output frame count.
- `decode_latents_to_frames` (function): Adapts latents to the expected 16-channel VAE decode input and returns frames (optional frame-count validation).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_fallback_enabled

from .config import RunConfig, as_torch_dtype, resolve_device_name, resolve_i2v_order
from .diagnostics import cuda_empty_cache, get_logger
from .wan_latent_norms import resolve_norm


def load_vae(vae_path: Optional[str], *, torch_dtype: torch.dtype, enable_tiling: bool = False):
    if not vae_path:
        raise RuntimeError("WAN22 GGUF: wan_vae_dir is required when running the GGUF runtime (VAE path missing).")

    from diffusers import AutoencoderKLWan  # type: ignore

    path = os.path.expanduser(str(vae_path))
    if os.path.isdir(path):
        vae = AutoencoderKLWan.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass
        return vae
    if os.path.isfile(path):
        loader = getattr(AutoencoderKLWan, "from_single_file", None)
        if loader is None:
            raise RuntimeError(
                f"AutoencoderKLWan.from_single_file not available; provide a directory instead of file: {path}"
            )
        vae = loader(path, torch_dtype=torch_dtype)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass
        return vae
    raise RuntimeError(f"WAN22 GGUF: VAE path not found: {path}")


def _retrieve_latents(encoder_output: Any, *, sample_mode: str) -> torch.Tensor:
    dist = getattr(encoder_output, "latent_dist", None)
    if dist is not None:
        mode = str(sample_mode or "").strip().lower()
        if mode in {"mode", "argmax"}:
            return dist.mode()
        if mode in {"sample", ""}:
            return dist.sample()
        raise ValueError(f"Unsupported VAE sample_mode: {sample_mode!r} (expected 'mode' or 'sample')")
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("VAE encode output has neither latent_dist nor latents")


def _maybe_resize_hw(x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if x.ndim != 4:
        return x
    _, _, h, w = x.shape
    if int(h) == int(height) and int(w) == int(width):
        return x
    import torch.nn.functional as F

    return F.interpolate(x, size=(int(height), int(width)), mode="bilinear", align_corners=False)


def _prepare_init_image_tensor(
    init_image: Any,
    *,
    device: str,
    dtype: str,
    height: int,
    width: int,
) -> torch.Tensor:
    torch_dtype = as_torch_dtype(dtype)
    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"

    if hasattr(init_image, "to"):
        t = init_image
        if hasattr(t, "ndim") and int(t.ndim) == 5:
            t = t[:, :, 0, ...]
        if hasattr(t, "ndim") and int(t.ndim) == 3:
            t = t.unsqueeze(0)
        if not hasattr(t, "ndim") or int(getattr(t, "ndim", 0)) != 4:
            raise RuntimeError(
                "WAN22 GGUF: init_image tensor must be 4D [B,C,H,W] (or 5D [B,C,T,H,W]); "
                f"got {getattr(t, 'shape', None)}"
            )
        try:
            t = t.to(target).to(torch_dtype)
        except Exception:
            t = t.to(target)
        t = _maybe_resize_hw(t, height=height, width=width)
        return t

    from PIL import Image
    import numpy as np

    if isinstance(init_image, Image.Image):
        img = init_image.convert("RGB")
        img = img.resize((int(width), int(height)), resample=Image.BICUBIC)
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(target).to(torch_dtype)
        return t * 2.0 - 1.0

    arr = np.asarray(init_image).astype("float32")
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = arr / 255.0 if arr.max() > 1.0 else arr
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        t = torch.from_numpy(arr).unsqueeze(0)
    else:
        raise RuntimeError("WAN22 GGUF: unsupported init_image array shape")

    t = t.to(target).to(torch_dtype)
    t = _maybe_resize_hw(t, height=height, width=width)
    return t * 2.0 - 1.0


def vae_encode_video_condition(
    init_image: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
) -> torch.Tensor:
    log = get_logger(logger)
    torch_dtype = as_torch_dtype(dtype)
    vae = load_vae(vae_dir, torch_dtype=torch_dtype, enable_tiling=bool(memory_management.manager.vae_always_tiled))

    norm = resolve_norm(None, channels=16)
    log.info("[wan22.gguf] VAE latent norm=%s channels=%d", norm.name, norm.channels)

    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"

    vae = vae.to(device=target, dtype=torch_dtype)

    if int(num_frames) <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid num_frames={num_frames} for I2V video_condition")

    image = _prepare_init_image_tensor(init_image, device=device, dtype=dtype, height=height, width=width)
    if image.ndim != 4:
        raise RuntimeError(
            "WAN22 GGUF: expected preprocessed init image to be 4D [B,C,H,W], "
            f"got {tuple(image.shape)}"
        )

    # Diffusers-style I2V conditioning video: first frame is the init image; remaining frames are 0 (i.e., 0.5 gray).
    image = image.unsqueeze(2)  # [B,C,1,H,W]
    video_condition = torch.cat(
        [image, image.new_zeros((image.shape[0], image.shape[1], int(num_frames) - 1, int(height), int(width)))],
        dim=2,
    )

    with torch.no_grad():
        try:
            encoded = _retrieve_latents(vae.encode(video_condition), sample_mode="mode")
        except torch.OutOfMemoryError as exc:
            if target != "cuda" or not smart_fallback_enabled():
                raise
            log.warning("[wan22.gguf] VAE encode (video_condition) OOM on CUDA; retrying on CPU.")
            cuda_empty_cache(logger, label="vae-encode-oom")
            vae = vae.to(device="cpu", dtype=torch.float32)
            video_condition = video_condition.to(device="cpu", dtype=torch.float32)
            encoded = _retrieve_latents(vae.encode(video_condition), sample_mode="mode")
        encoded = norm.process_in(encoded)

    if offload_after:
        try:
            vae.to("cpu")
        except Exception:
            pass
        del vae
        cuda_empty_cache(logger, label="after-vae-encode")

    return encoded


def vae_decode_video(
    video_latents: Any,
    *,
    model_dir: str,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
    expected_frames: int | None = None,
) -> list[object]:
    _ = model_dir  # kept for signature symmetry (callers pass stage dir; current VAE loads from explicit path)
    log = get_logger(logger)
    torch_dtype = as_torch_dtype(dtype)
    vae = load_vae(vae_dir, torch_dtype=torch_dtype, enable_tiling=bool(memory_management.manager.vae_always_tiled))

    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"
    vae = vae.to(device=target, dtype=torch_dtype)

    norm = resolve_norm(None, channels=16)

    if hasattr(video_latents, "ndim"):
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(2)
        elif video_latents.ndim != 5:
            raise RuntimeError(
                f"WAN22 VAE decode expects 4D or 5D latents; got shape={tuple(getattr(video_latents,'shape',()))}"
            )

    b, c, t_lat, h, w = video_latents.shape

    if int(c) != 16:
        raise RuntimeError(
            f"WAN22 VAE decode expects 16 channels but received C={c}. "
            "If using I2V checkpoints that embed mask+image+latents, slice the latent channels before decode."
        )

    from PIL import Image

    frames: list[Image.Image] = []
    with torch.no_grad():
        lat = norm.process_out(video_latents)
        try:
            img = vae.decode(lat).sample
        except torch.OutOfMemoryError:
            if target != "cuda" or not smart_fallback_enabled():
                raise
            log.warning("[wan22.gguf] VAE decode OOM on CUDA; retrying on CPU.")
            cuda_empty_cache(logger, label="vae-decode-oom")
            vae = vae.to(device="cpu", dtype=torch.float32)
            lat = lat.to(device="cpu", dtype=torch.float32)
            img = vae.decode(lat).sample
            target = "cpu"

        log.info("[wan22.gguf] VAE decode output shape=%s", tuple(getattr(img, "shape", ())))
        if not hasattr(img, "ndim") or img.ndim != 5:
            raise RuntimeError(
                f"WAN22 GGUF: VAE decode produced unexpected rank: shape={tuple(getattr(img,'shape',()))}; expected [B,C,T,H,W]"
            )
        if int(img.shape[0]) < 1 or int(img.shape[1]) != 3:
            raise RuntimeError(
                f"WAN22 GGUF: VAE decode produced unexpected shape: {tuple(img.shape)}; expected B>=1 and C=3."
            )
        t_out = int(img.shape[2])
        if expected_frames is not None and int(expected_frames) != t_out:
            raise RuntimeError(
                "WAN22 GGUF: VAE decode time dimension mismatch: "
                f"expected T={int(expected_frames)} got T={t_out} (latent_T={int(t_lat)})."
            )

        for ti in range(t_out):
            x = img[0, :, ti, :, :].detach()
            if x.ndim != 3:
                raise RuntimeError(
                    f"WAN22 GGUF: VAE decode produced unexpected frame tensor rank: shape={tuple(x.shape)}; expected [C,H,W]"
                )
            if not torch.isfinite(x).all():
                n_bad = int((~torch.isfinite(x)).sum().item())
                raise RuntimeError(f"WAN22 GGUF: VAE decode produced non-finite outputs (count={n_bad}).")
            # Diffusers VAEs output [-1, 1]; convert to [0, 1] for image conversion.
            x = (x + 1.0) * 0.5
            x = x.clamp(0, 1)
            arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            frames.append(Image.fromarray(arr))

    if offload_after:
        try:
            vae.to("cpu")
        except Exception:
            pass
        del vae
        cuda_empty_cache(logger, label="after-vae-decode")

    return frames


def decode_latents_to_frames(
    *,
    latents: torch.Tensor,
    model_dir: str,
    cfg: RunConfig,
    logger: Any = None,
    debug_preview: bool = False,
    expected_frames: int | None = None,
) -> list[object]:
    log = get_logger(logger)
    x = latents
    log.info("[wan22.gguf] decode latents: shape=%s", tuple(x.shape))
    _ = debug_preview  # debug-only clamp removed (no env-driven behavior)

    c = int(x.shape[1])
    if c != 16:
        if c >= 16:
            if resolve_i2v_order() == "lat_first":
                x = x[:, :16, ...]
            else:
                x = x[:, -16:, ...]
            log.info("[wan22.gguf] decode latents: sliced to 16 channels from C=%d", c)
        else:
            raise RuntimeError(f"WAN22 GGUF: expected ≥16 latent channels for decode, got {c}")

    return vae_decode_video(
        x,
        model_dir=model_dir,
        device=resolve_device_name(cfg.device),
        dtype=cfg.dtype,
        vae_dir=cfg.vae_dir,
        logger=logger,
        expected_frames=expected_frames,
    )
