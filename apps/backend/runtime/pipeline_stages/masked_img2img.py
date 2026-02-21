"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Masked img2img (“inpaint”) helpers for SD-family pipelines.
Normalizes masks (RGBA alpha semantics), applies invert/blur/round options, optionally builds an inpaint-full-res crop plan
(Forge-style zoom-crop + paste-back overlay), and produces latent-space masks for sampler enforcement.
When init/mask dimensions differ from target `processing.width/height`, normalizes both to target before latent encode.

Symbols (top-level; keep in sync; no ghosts):
- `InpaintFullResPlan` (dataclass): Full-res inpaint plan (crop region + overlay composite inputs).
- `MaskedImg2ImgBundle` (dataclass): Prepared init tensor/latents + latent masks + optional full-res plan.
- `LatentMaskEnforcer` (class): Latent masking helper implementing post-blend and per-step clamp hooks.
- `prepare_masked_img2img_bundle` (function): Build a masked img2img bundle from a processing object and sampling plan.
- `apply_inpaint_full_res_composite` (function): Paste-back + overlay composite for full-res inpaint outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from apps.backend.runtime.pipeline_stages.image_io import pil_to_tensor

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
_RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

MaskEnforcementMode = str
MASK_ENFORCEMENT_POST_BLEND = "post_blend"
MASK_ENFORCEMENT_PER_STEP_CLAMP = "per_step_clamp"
ALLOWED_MASK_ENFORCEMENTS = frozenset({MASK_ENFORCEMENT_POST_BLEND, MASK_ENFORCEMENT_PER_STEP_CLAMP})


@dataclass(slots=True)
class InpaintFullResPlan:
    """Full-res “Only masked” plan (Forge-style zoom-crop + paste-back)."""

    crop_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in init-image coordinates
    paste_to: Tuple[int, int, int, int]  # (x, y, w, h) in init-image coordinates
    overlay: Image.Image  # RGBA overlay with original unmasked pixels


@dataclass(slots=True)
class MaskedImg2ImgBundle:
    """Prepared init tensor/latents + latent masks for a masked img2img run."""

    init_tensor: torch.Tensor
    init_latent: torch.Tensor
    image_conditioning: torch.Tensor
    latent_masked: torch.Tensor  # 1 inside mask, 0 outside (shape 1x1xHlatentxWlatent)
    latent_unmasked: torch.Tensor  # 1 outside mask, 0 inside (shape 1x1xHlatentxWlatent)
    full_res: InpaintFullResPlan | None = None


class LatentMaskEnforcer:
    """Applies latent masking constraints (post-blend and/or per-step clamp)."""

    def __init__(self, *, init_latent: torch.Tensor, latent_masked: torch.Tensor, latent_unmasked: torch.Tensor) -> None:
        self._init_latent = init_latent
        self._latent_masked = latent_masked
        self._latent_unmasked = latent_unmasked
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _materialize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (x.device, x.dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        init_latent = self._init_latent
        if init_latent.device != x.device or init_latent.dtype != x.dtype:
            init_latent = init_latent.to(device=x.device, dtype=x.dtype)

        masked = self._latent_masked
        if masked.device != x.device or masked.dtype != x.dtype:
            masked = masked.to(device=x.device, dtype=x.dtype)

        unmasked = self._latent_unmasked
        if unmasked.device != x.device or unmasked.dtype != x.dtype:
            unmasked = unmasked.to(device=x.device, dtype=x.dtype)

        init_unmasked = init_latent * unmasked
        self._cache[key] = (masked, init_unmasked)
        return masked, init_unmasked

    def post_step(self, x: torch.Tensor, step: int, steps: int) -> None:  # noqa: ARG002 - hook signature
        masked, init_unmasked = self._materialize(x)
        x.mul_(masked)
        x.add_(init_unmasked)

    def post_sample(self, x: torch.Tensor) -> torch.Tensor:
        masked, init_unmasked = self._materialize(x)
        x.mul_(masked)
        x.add_(init_unmasked)
        return x


def _create_binary_mask(mask: Image.Image, *, round_mask: bool) -> Image.Image:
    if mask.mode == "RGBA":
        extrema = mask.getextrema()
        alpha_extrema = extrema[-1] if isinstance(extrema, tuple) and len(extrema) == 4 else None
        if alpha_extrema is not None and alpha_extrema != (255, 255):
            alpha = mask.split()[-1].convert("L")
            return alpha.point(lambda x: 255 if x > 128 else 0) if round_mask else alpha

    gray = mask.convert("L")
    return gray.point(lambda x: 255 if x > 128 else 0) if round_mask else gray


def _gaussian_kernel_1d(*, sigma: float, device: torch.device) -> tuple[torch.Tensor, int]:
    sigma = float(sigma)
    if sigma <= 0.0:
        kernel = torch.tensor([1.0], device=device, dtype=torch.float32)
        return kernel, 0
    radius = int(2.5 * sigma + 0.5)
    if radius <= 0:
        kernel = torch.tensor([1.0], device=device, dtype=torch.float32)
        return kernel, 0
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel, radius


def _blur_mask(mask: Image.Image, *, sigma_x: float, sigma_y: float) -> Image.Image:
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    if sigma_x <= 0.0 and sigma_y <= 0.0:
        return mask

    array = np.array(mask.convert("L"), dtype=np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0) / 255.0

    if sigma_x > 0.0:
        kernel, radius = _gaussian_kernel_1d(sigma=sigma_x, device=tensor.device)
        if radius > 0:
            tensor = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
        weight = kernel.view(1, 1, 1, -1)
        tensor = F.conv2d(tensor, weight)

    if sigma_y > 0.0:
        kernel, radius = _gaussian_kernel_1d(sigma=sigma_y, device=tensor.device)
        if radius > 0:
            tensor = F.pad(tensor, (0, 0, radius, radius), mode="replicate")
        weight = kernel.view(1, 1, -1, 1)
        tensor = F.conv2d(tensor, weight)

    out = tensor.squeeze(0).squeeze(0).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(out, mode="L")


def _get_crop_region(mask: Image.Image, *, pad: int) -> tuple[int, int, int, int] | None:
    if pad < 0:
        raise ValueError("pad must be >= 0")
    box = mask.getbbox()
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if pad == 0:
        return int(x1), int(y1), int(x2), int(y2)
    return (
        max(int(x1) - pad, 0),
        max(int(y1) - pad, 0),
        min(int(x2) + pad, mask.size[0]),
        min(int(y2) + pad, mask.size[1]),
    )


def _expand_crop_region(
    crop_region: tuple[int, int, int, int],
    *,
    processing_width: int,
    processing_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_region
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    if processing_width <= 0 or processing_height <= 0:
        raise ValueError("processing_width/processing_height must be positive")
    ratio_crop = float(crop_w) / float(crop_h)
    ratio_processing = float(processing_width) / float(processing_height)

    if ratio_crop > ratio_processing:
        desired_h = crop_w / ratio_processing
        diff = int(desired_h - crop_h)
        y1 -= diff // 2
        y2 += diff - diff // 2
        if y2 >= image_height:
            overflow = y2 - image_height
            y2 -= overflow
            y1 -= overflow
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_w = crop_h * ratio_processing
        diff = int(desired_w - crop_w)
        x1 -= diff // 2
        x2 += diff - diff // 2
        if x2 >= image_width:
            overflow = x2 - image_width
            x2 -= overflow
            x1 -= overflow
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 >= image_width:
            x2 = image_width

    return int(x1), int(y1), int(x2), int(y2)


def _overlay_from_mask(*, image: Image.Image, mask_for_overlay: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    inv = ImageOps.invert(mask_for_overlay.convert("L"))
    overlay.paste(image.convert("RGBA"), (0, 0), inv)
    return overlay


def _fill_masked_regions(image: Image.Image, *, mask: Image.Image) -> Image.Image:
    """Best-effort masked-region fill (blur-smear) used for `inpainting_fill` parity."""
    base = Image.new("RGBA", image.size, (0, 0, 0, 0))
    inv = ImageOps.invert(mask.convert("L"))
    scaffold = Image.new("RGBA", image.size, (0, 0, 0, 0))
    scaffold.paste(image.convert("RGBA"), (0, 0), inv)

    for radius, repeats in ((128, 1), (32, 1), (16, 1), (8, 2), (4, 4), (2, 2), (0, 1)):
        blurred = scaffold.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            base.alpha_composite(blurred)
    return base.convert("RGB")


def _latent_mask_from_image(
    mask: Image.Image,
    *,
    latent_width: int,
    latent_height: int,
    round_mask: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if latent_width <= 0 or latent_height <= 0:
        raise ValueError("latent_width/latent_height must be positive")
    resized = mask.convert("RGB").resize((latent_width, latent_height), resample=_RESAMPLE_LANCZOS)
    array = np.array(resized, dtype=np.float32) / 255.0
    plane = array[..., 0]
    if round_mask:
        plane = np.around(plane)
    tensor = torch.from_numpy(plane).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return tensor


def _validate_mask_enforcement(mode: Any) -> MaskEnforcementMode:
    value = str(mode or "").strip()
    if value not in ALLOWED_MASK_ENFORCEMENTS:
        raise ValueError(
            f"Invalid mask_enforcement '{value}'. Allowed: {sorted(ALLOWED_MASK_ENFORCEMENTS)}"
        )
    return value


def prepare_masked_img2img_bundle(
    processing: Any,
    plan: Any,
    *,
    enforce_mode: Any,
) -> tuple[MaskedImg2ImgBundle, LatentMaskEnforcer]:
    """Prepare masked img2img inputs (mask processing + optional full-res plan + latent masks).

    Notes:
    - Requires `processing.init_image` and `processing.mask`.
    - If init image size differs from `processing.width/height`, init image + mask are resized to target before latent encode.
    """
    init_image = getattr(processing, "init_image", None)
    if init_image is None:
        raise ValueError("masked img2img requires processing.init_image")

    raw_mask = getattr(processing, "mask", None) or getattr(processing, "image_mask", None)
    if raw_mask is None:
        raise ValueError("masked img2img requires processing.mask (or processing.image_mask)")

    _validate_mask_enforcement(enforce_mode)

    width = int(getattr(processing, "width", 0) or 0)
    height = int(getattr(processing, "height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("processing.width/height must be set for img2img")

    if raw_mask.size != init_image.size:
        raise ValueError(
            f"Mask size must match init image size; got mask={raw_mask.size} init={init_image.size}"
        )

    init_w, init_h = init_image.size
    if (init_w, init_h) != (width, height):
        original_width, original_height = init_w, init_h
        init_image = init_image.convert("RGB").resize((width, height), resample=_RESAMPLE_LANCZOS)
        raw_mask = raw_mask.resize((width, height), resample=_RESAMPLE_NEAREST)

        setattr(processing, "init_image", init_image)
        if hasattr(processing, "mask") and getattr(processing, "mask", None) is not None:
            setattr(processing, "mask", raw_mask)
        if hasattr(processing, "image_mask") and getattr(processing, "image_mask", None) is not None:
            setattr(processing, "image_mask", raw_mask)
        if hasattr(processing, "update_extra_param") and callable(getattr(processing, "update_extra_param")):
            processing.update_extra_param("Init resize", f"{original_width}x{original_height} -> {width}x{height}")

    mask_round = bool(getattr(processing, "mask_round", True))
    invert = bool(int(getattr(processing, "inpainting_mask_invert", 0) or 0))
    blur_x = float(getattr(processing, "mask_blur_x", getattr(processing, "mask_blur", 0)) or 0)
    blur_y = float(getattr(processing, "mask_blur_y", getattr(processing, "mask_blur", 0)) or 0)

    mask = _create_binary_mask(raw_mask, round_mask=mask_round)
    if invert:
        mask = ImageOps.invert(mask)
        processing.update_extra_param("Mask mode", "Inpaint not masked")

    if blur_x > 0.0 or blur_y > 0.0:
        mask = _blur_mask(mask, sigma_x=blur_x, sigma_y=blur_y)
        processing.update_extra_param("Mask blur", int(getattr(processing, "mask_blur", 0) or 0))

    full_res_plan: InpaintFullResPlan | None = None
    mask_for_sampling = mask
    image_for_sampling = init_image.convert("RGB")

    if bool(getattr(processing, "inpaint_full_res", True)):
        pad = int(getattr(processing, "inpaint_full_res_padding", 0) or 0)
        crop_region = _get_crop_region(mask.convert("L"), pad=pad)
        if crop_region is None:
            raise ValueError('Unable to perform "Inpaint only masked" because mask is blank')
        crop_region = _expand_crop_region(
            crop_region,
            processing_width=width,
            processing_height=height,
            image_width=mask.size[0],
            image_height=mask.size[1],
        )
        x1, y1, x2, y2 = crop_region
        paste_to = (x1, y1, x2 - x1, y2 - y1)
        full_res_plan = InpaintFullResPlan(
            crop_region=crop_region,
            paste_to=paste_to,
            overlay=_overlay_from_mask(image=init_image.convert("RGB"), mask_for_overlay=mask),
        )
        mask_crop = mask.crop(crop_region)
        mask_for_sampling = mask_crop.resize((width, height), resample=_RESAMPLE_LANCZOS)
        image_crop = image_for_sampling.crop(crop_region)
        image_for_sampling = image_crop.resize((width, height), resample=_RESAMPLE_LANCZOS)
        processing.update_extra_param("Inpaint area", "Only masked")
        processing.update_extra_param("Masked area padding", int(pad))

    inpainting_fill = int(getattr(processing, "inpainting_fill", 0) or 0)
    if inpainting_fill != 1:
        image_for_sampling = _fill_masked_regions(image_for_sampling, mask=mask_for_sampling)
        if inpainting_fill == 0:
            processing.update_extra_param("Masked content", "fill")

    init_tensor = pil_to_tensor([image_for_sampling])
    init_latent = processing.sd_model.encode_first_stage(init_tensor)

    latent_h = int(init_latent.shape[2])
    latent_w = int(init_latent.shape[3])
    latent_masked = _latent_mask_from_image(
        mask_for_sampling,
        latent_width=latent_w,
        latent_height=latent_h,
        round_mask=mask_round,
        device=init_latent.device,
        dtype=init_latent.dtype,
    )
    latent_unmasked = (1.0 - latent_masked).to(device=init_latent.device, dtype=init_latent.dtype)

    if inpainting_fill == 2:
        seeds: Sequence[int] = list(getattr(plan, "seeds", []) or [0])
        gens = []
        for seed in seeds:
            gen = torch.Generator(device=init_latent.device)
            gen.manual_seed(int(seed))
            gens.append(torch.randn(tuple(init_latent.shape[1:]), generator=gen, device=init_latent.device, dtype=init_latent.dtype))
        noise = torch.stack(gens, dim=0)
        init_latent = init_latent * latent_unmasked + noise * latent_masked
        processing.update_extra_param("Masked content", "latent noise")
    elif inpainting_fill == 3:
        init_latent = init_latent * latent_unmasked
        processing.update_extra_param("Masked content", "latent nothing")

    from apps.backend.runtime.processing.conditioners import img2img_conditioning

    image_conditioning = img2img_conditioning(
        processing.sd_model,
        init_tensor,
        init_latent,
        image_mask=mask_for_sampling,
        round_mask=bool(getattr(processing, "round_image_mask", True)),
    )

    bundle = MaskedImg2ImgBundle(
        init_tensor=init_tensor,
        init_latent=init_latent,
        image_conditioning=image_conditioning,
        latent_masked=latent_masked,
        latent_unmasked=latent_unmasked,
        full_res=full_res_plan,
    )
    enforcer = LatentMaskEnforcer(init_latent=init_latent, latent_masked=latent_masked, latent_unmasked=latent_unmasked)
    return bundle, enforcer


def _uncrop(image: Image.Image, *, dest_size: tuple[int, int], paste_to: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = paste_to
    base = Image.new("RGBA", dest_size, (0, 0, 0, 0))
    resized = image.resize((w, h), resample=_RESAMPLE_LANCZOS).convert("RGBA")
    base.paste(resized, (x, y))
    return base


def apply_inpaint_full_res_composite(
    images: Sequence[Image.Image],
    *,
    plan: InpaintFullResPlan,
) -> list[Image.Image]:
    """Apply paste-back + overlay compositing for full-res inpaint outputs."""
    if plan.overlay.mode != "RGBA":
        raise ValueError("full-res overlay must be RGBA")
    out: list[Image.Image] = []
    for img in images:
        base = _uncrop(img, dest_size=(plan.overlay.width, plan.overlay.height), paste_to=plan.paste_to)
        base.alpha_composite(plan.overlay)
        out.append(base.convert("RGB"))
    return out


__all__ = [
    "ALLOWED_MASK_ENFORCEMENTS",
    "InpaintFullResPlan",
    "LatentMaskEnforcer",
    "MASK_ENFORCEMENT_PER_STEP_CLAMP",
    "MASK_ENFORCEMENT_POST_BLEND",
    "MaskedImg2ImgBundle",
    "apply_inpaint_full_res_composite",
    "prepare_masked_img2img_bundle",
]
