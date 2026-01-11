"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF VAE IO helpers (init encode + decode to frames).
Loads the WAN VAE via Diffusers, applies latent normalization, and converts between latents and RGB frames for the WAN22 GGUF runtime.

Symbols (top-level; keep in sync; no ghosts):
- `load_vae` (function): Loads the WAN VAE component (from directory or single-file weights).
- `vae_encode_init` (function): Encodes an init image into latents for img2vid.
- `vae_decode_video` (function): Decodes video latents to frames and optionally offloads/cleans up after decode.
- `decode_latents_to_frames` (function): Adapts latents to the expected 16-channel VAE decode input and returns frames.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from apps.backend.infra.config.env_flags import env_flag

from .config import RunConfig, as_torch_dtype, resolve_device_name, resolve_i2v_order
from .diagnostics import cuda_empty_cache, get_logger
from .wan_latent_norms import resolve_norm


def load_vae(vae_path: Optional[str], *, torch_dtype: torch.dtype):
    if not vae_path:
        raise RuntimeError("WAN22 GGUF: wan_vae_dir is required when running the GGUF runtime (VAE path missing).")

    from diffusers import AutoencoderKLWan  # type: ignore

    path = os.path.expanduser(str(vae_path))
    if os.path.isdir(path):
        return AutoencoderKLWan.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if os.path.isfile(path):
        loader = getattr(AutoencoderKLWan, "from_single_file", None)
        if loader is None:
            raise RuntimeError(
                f"AutoencoderKLWan.from_single_file not available; provide a directory instead of file: {path}"
            )
        return loader(path, torch_dtype=torch_dtype)
    raise RuntimeError(f"WAN22 GGUF: VAE path not found: {path}")


def vae_encode_init(
    init_image: Any,
    *,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
) -> torch.Tensor:
    log = get_logger(logger)
    torch_dtype = as_torch_dtype(dtype)
    vae = load_vae(vae_dir, torch_dtype=torch_dtype)

    norm = resolve_norm(os.getenv("WAN_VAE_NORM", "wan21"), channels=16)
    log.info("[wan22.gguf] VAE latent norm=%s channels=%d", norm.name, norm.channels)

    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"

    # Force tiled VAE decode/encode for WAN (legacy global switch, scoped).
    from apps.backend.runtime.memory import memory_management as _mm

    old_tiled = _mm.manager.vae_always_tiled
    try:
        _mm.manager.vae_always_tiled = True
        vae = vae.to(device=target, dtype=torch_dtype)
    finally:
        _mm.manager.vae_always_tiled = old_tiled

    # Preprocess init image into [-1,1] tensor [B,C,T,H,W] with T=1
    if not hasattr(init_image, "to"):
        from PIL import Image
        import numpy as np

        if isinstance(init_image, Image.Image):
            img = init_image.convert("RGB")
            arr = np.array(img).astype("float32") / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0
        else:
            arr = np.asarray(init_image).astype("float32")
            if arr.ndim == 3 and arr.shape[2] in (1, 3):
                arr = arr / 255.0 if arr.max() > 1.0 else arr
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                raise RuntimeError("WAN22 GGUF: unsupported init_image array shape")
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0

    if hasattr(init_image, "ndim"):
        if init_image.ndim == 4:
            init_image = init_image.unsqueeze(2)
        elif init_image.ndim != 5:
            raise RuntimeError("WAN22 GGUF: init_image must be 4D (B,C,H,W) or 5D (B,C,T,H,W) after preprocessing")

    with torch.no_grad():
        encoded = vae.encode(init_image).latent_dist.sample()
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
) -> list[object]:
    _ = model_dir  # kept for signature symmetry (callers pass stage dir; current VAE loads from explicit path)
    log = get_logger(logger)
    torch_dtype = as_torch_dtype(dtype)
    vae = load_vae(vae_dir, torch_dtype=torch_dtype)

    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"
    vae = vae.to(device=target, dtype=torch_dtype)

    norm = resolve_norm(os.getenv("WAN_VAE_NORM", "wan21"), channels=16)

    if hasattr(video_latents, "ndim"):
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(2)
        elif video_latents.ndim != 5:
            raise RuntimeError(
                f"WAN22 VAE decode expects 4D or 5D latents; got shape={tuple(getattr(video_latents,'shape',()))}"
            )

    b, c, t, h, w = video_latents.shape

    if env_flag("WAN_I2V_LAT_STATS", default=False):
        vt = video_latents
        if torch.is_tensor(vt):
            vt_cpu = vt.detach().to(device="cpu", dtype=torch.float32)
            finite = torch.isfinite(vt_cpu)
            n_total = int(vt_cpu.numel())
            n_bad = int((~finite).sum().item())
            if n_bad < n_total:
                vals = vt_cpu[finite]
                mn = float(vals.min().item())
                mx = float(vals.max().item())
                mean = float(vals.mean().item())
                std = float(vals.std(unbiased=False).item())
                log.info(
                    "[wan22.gguf] latents stats: B=%d C=%d T=%d H=%d W=%d min=%.4f max=%.4f mean=%.4f std=%.4f bad=%d",
                    b,
                    c,
                    t,
                    h,
                    w,
                    mn,
                    mx,
                    mean,
                    std,
                    n_bad,
                )
            else:
                log.info(
                    "[wan22.gguf] latents stats: B=%d C=%d T=%d H=%d W=%d (all non-finite: %d)",
                    b,
                    c,
                    t,
                    h,
                    w,
                    n_bad,
                )

    if int(c) != 16:
        raise RuntimeError(
            f"WAN22 VAE decode expects 16 channels but received C={c}. "
            "If using I2V checkpoints that embed mask+image+latents, slice the latent channels before decode."
        )

    from PIL import Image

    frames: list[Image.Image] = []
    with torch.no_grad():
        for ti in range(t):
            lat = video_latents[:, :, ti : ti + 1]
            lat = norm.process_out(lat)
            img = vae.decode(lat).sample
            if ti == 0:
                log.info("[wan22.gguf] VAE decode output shape=%s", tuple(getattr(img, "shape", ())))
            x = img[0].detach()
            if x.ndim == 4:
                if x.shape[1] == 1:
                    x = x[:, 0, ...]
                if x.ndim == 4:
                    x = x.squeeze()
            if x.ndim != 3:
                raise RuntimeError(
                    f"WAN22 GGUF: VAE decode produced unexpected tensor rank: shape={tuple(x.shape)}; expected [C,H,W]"
                )
            if not torch.isfinite(x).all():
                n_bad = int((~torch.isfinite(x)).sum().item())
                raise RuntimeError(f"WAN22 GGUF: VAE decode produced non-finite outputs (count={n_bad}).")
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
) -> list[object]:
    log = get_logger(logger)
    x = latents
    log.info("[wan22.gguf] decode latents: shape=%s", tuple(x.shape))

    if debug_preview:
        v = os.getenv("WAN_I2V_DEBUG_CLAMP", "").strip()
        if v:
            lim = float(v)
            if lim > 0:
                x = torch.clamp(x, min=-lim, max=lim)

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
    )
