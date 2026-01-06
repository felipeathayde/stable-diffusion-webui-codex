"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Edge/line preprocessors for ControlNet (Canny/binary/sobel/lineart/HED/PiDiNet/MLSD/anime/manga).
Registers a suite of edge detectors into the `ControlPreprocessorRegistry` and provides the preprocessing implementations, including
optional torchvision Canny and model-backed methods (HED/PiDiNet/MLSD/lineart variants).

Symbols (top-level; keep in sync; no ghosts):
- `EdgePreprocessorConfig` (dataclass): Configuration defaults (thresholds + weights filenames) for all edge preprocessors.
- `register_edge_preprocessors` (function): Registers all edge preprocessors by name into a registry using a shared config instance.
- `preprocess_canny` (function): Canny edge detection (uses torchvision if available; otherwise falls back to manual implementation).
- `preprocess_binary` (function): Simple binary-threshold edge map.
- `preprocess_sobel` (function): Sobel edge detection (gradient magnitude).
- `preprocess_lineart` (function): Simple “lineart” edge map (thresholded gradient / smoothing pipeline).
- `preprocess_hed` (function): HED model-based edge detection (loads weights via `load_hed_model`).
- `preprocess_pidinet` (function): PiDiNet model-based edge detection (loads weights via `load_pidinet_model`).
- `preprocess_mlsd` (function): MLSD line segment detection (loads weights, decodes lines, renders line map).
- `preprocess_lineart_anime` (function): Anime lineart model-based preprocessor.
- `preprocess_manga_line` (function): Manga line model-based preprocessor.
- `_ensure_image_batch` (function): Ensures input tensor has a batch dimension.
- `_to_grayscale` (function): Converts RGB tensor to grayscale.
- `_normalize_tensor` (function): Normalizes tensor to `[0, 1]` range (best-effort).
- `_sobel_filters` (function): Builds Sobel filter kernels for the given device/dtype.
- `_gaussian_kernel` (function): Builds a Gaussian blur kernel tensor.
- `_canny_manual` (function): Manual Canny implementation (gaussian blur + gradients + NMS + hysteresis).
- `_non_maximum_suppression` (function): Non-maximum suppression for gradient magnitude along orientation.
- `_hysteresis` (function): Hysteresis thresholding step for Canny.
- `_build_metadata` (function): Builds a metadata mapping for a `PreprocessorResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

import math
import torch
import torch.nn.functional as F

try:
    from torchvision.ops import canny as tv_canny
except ImportError:  # pragma: no cover - torchvision optional
    tv_canny = None

from .registry import ControlPreprocessorRegistry, PreprocessorResult
from .models.hed import HEDConfig, load_hed_model
from .models.pidinet import PiDiNetConfig, load_pidinet_model
from .models.mlsd import MLSDConfig, load_mlsd_model, decode_lines, render_lines
from .models.lineart_anime import LineartAnimeConfig, load_lineart_anime_model
from .models.manga_line import MangaLineConfig, load_manga_line_model

logger = logging.getLogger("backend.runtime.controlnet.preprocessors.edges")


@dataclass(frozen=True)
class EdgePreprocessorConfig:
    low_threshold: float = 0.1
    high_threshold: float = 0.2
    gaussian_kernel: int = 5
    gaussian_sigma: float = 1.0
    binary_threshold: float = 0.5
    lineart_threshold: float = 0.15
    hed_weights: str = "hed/ControlNetHED.pth"
    pidinet_weights: str = "pidinet/table5_pidinet.pth"
    pidinet_variant: str = "carv4"
    pidinet_dilation: int = 24
    pidinet_attention: bool = True
    mlsd_weights: str = "mlsd/mlsd_large_512_fp32.pth"
    lineart_anime_weights: str = "lineart_anime/netG.pth"
    manga_line_weights: str = "manga_line/res_skip.pth"


def register_edge_preprocessors(registry: ControlPreprocessorRegistry, *, config: Optional[EdgePreprocessorConfig] = None) -> None:
    cfg = config or EdgePreprocessorConfig()
    registry.register("canny", lambda image, **kwargs: preprocess_canny(image, cfg, **kwargs))
    registry.register("binary", lambda image, **kwargs: preprocess_binary(image, cfg, **kwargs))
    registry.register("sobel", lambda image, **kwargs: preprocess_sobel(image, cfg, **kwargs))
    registry.register("lineart", lambda image, **kwargs: preprocess_lineart(image, cfg, **kwargs))
    registry.register("hed", lambda image, **kwargs: preprocess_hed(image, cfg, **kwargs))
    registry.register("pidinet", lambda image, **kwargs: preprocess_pidinet(image, cfg, **kwargs))
    registry.register("mlsd", lambda image, **kwargs: preprocess_mlsd(image, cfg, **kwargs))
    registry.register("lineart_anime", lambda image, **kwargs: preprocess_lineart_anime(image, cfg, **kwargs))
    registry.register("manga_line", lambda image, **kwargs: preprocess_manga_line(image, cfg, **kwargs))


def preprocess_canny(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
    gaussian_kernel: Optional[int] = None,
    gaussian_sigma: Optional[float] = None,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    low_thr = float(low if low is not None else config.low_threshold)
    high_thr = float(high if high is not None else config.high_threshold)
    kernel = int(gaussian_kernel if gaussian_kernel is not None else config.gaussian_kernel)
    sigma = float(gaussian_sigma if gaussian_sigma is not None else config.gaussian_sigma)

    if tv_canny is None:
        logger.debug("torchvision.ops.canny unavailable; falling back to Sobel magnitude hysteresis")
        edges = _canny_manual(image, low_thr, high_thr, kernel, sigma)
    else:
        edges, _ = tv_canny(image, low_thr, high_thr, kernel, sigma)
    return PreprocessorResult(image=edges, metadata=_build_metadata("canny", {"low": low_thr, "high": high_thr}))


def preprocess_binary(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    threshold: Optional[float] = None,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    thr = float(threshold if threshold is not None else config.binary_threshold)
    grayscale = _to_grayscale(image)
    binary_mask = (grayscale >= thr).to(grayscale.dtype)
    return PreprocessorResult(image=binary_mask, metadata=_build_metadata("binary", {"threshold": thr}))


def preprocess_sobel(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    normalize: bool = True,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    grayscale = _to_grayscale(image)
    gx, gy = _sobel_filters(grayscale.device, grayscale.dtype)
    grad_x = F.conv2d(grayscale, gx, padding=1)
    grad_y = F.conv2d(grayscale, gy, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    if normalize:
        magnitude = _normalize_tensor(magnitude)
    return PreprocessorResult(image=magnitude, metadata=_build_metadata("sobel", {"normalize": normalize}))


def preprocess_lineart(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    threshold: Optional[float] = None,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    thr = float(threshold if threshold is not None else config.lineart_threshold)
    sobel_result = preprocess_sobel(image, config, normalize=True).image
    lineart = (sobel_result >= thr).to(sobel_result.dtype)
    return PreprocessorResult(image=lineart, metadata=_build_metadata("lineart", {"threshold": thr}))


def preprocess_hed(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    weights_path: Optional[str] = None,
    safe: bool = False,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    hed_config = HEDConfig(weights_path=weights_path or config.hed_weights)
    model = load_hed_model(hed_config)
    device = image.device
    tensor = image.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        resized = [
            F.interpolate(out, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
            for out in outputs
        ]
        stacked = torch.stack(resized, dim=0)
        fused = torch.sigmoid(stacked.mean(dim=0))
        if safe:
            fused = fused.clamp_(0.0, 1.0)
        fused = fused.to(torch.float32)

    metadata = _build_metadata("hed", {"weights": hed_config.weights_path, "safe": safe})
    return PreprocessorResult(image=fused, metadata=metadata)


def preprocess_pidinet(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    safe: bool = False,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    pidinet_config = PiDiNetConfig(
        weights_path=weights_path or config.pidinet_weights,
        variant=variant or config.pidinet_variant,
        dilation=config.pidinet_dilation,
        use_spatial_attention=config.pidinet_attention,
    )
    model = load_pidinet_model(pidinet_config)
    device = image.device
    tensor = image.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        fused = outputs[-1]
        if safe:
            fused = fused.clamp_(0.0, 1.0)
        fused = fused.to(torch.float32)

    metadata = _build_metadata(
        "pidinet",
        {
            "weights": pidinet_config.weights_path,
            "variant": pidinet_config.variant,
            "safe": safe,
        },
    )
    return PreprocessorResult(image=fused, metadata=metadata)


def preprocess_mlsd(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    weights_path: Optional[str] = None,
    score_threshold: float = 0.1,
    distance_threshold: float = 20.0,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    mlsd_config = MLSDConfig(weights_path=weights_path or config.mlsd_weights)
    model = load_mlsd_model(mlsd_config)
    device = image.device
    tensor = image.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
    h, w = tensor.shape[-2:]
    resized = F.interpolate(tensor, size=mlsd_config.input_size, mode="bilinear", align_corners=False)
    ones = torch.ones(resized.shape[0], 1, *resized.shape[-2:], device=device)
    batch = torch.cat([resized, ones], dim=1)
    batch = batch * 2.0 - 1.0
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        tp_map = model(batch)
        lines, scores = decode_lines(tp_map, score_threshold=score_threshold, dist_threshold=distance_threshold)
        if lines.numel() > 0:
            scale_x = w / mlsd_config.input_size[1]
            scale_y = h / mlsd_config.input_size[0]
            lines = lines.clone()
            lines[:, [0, 2]] *= scale_x
            lines[:, [1, 3]] *= scale_y
        rendered = render_lines(lines, h, w)

    metadata = _build_metadata(
        "mlsd",
        {
            "weights": mlsd_config.weights_path,
            "score_threshold": score_threshold,
            "distance_threshold": distance_threshold,
            "lines_detected": int(lines.shape[0]),
        },
    )
    return PreprocessorResult(image=rendered.to(torch.float32), metadata=metadata)


def preprocess_lineart_anime(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    weights_path: Optional[str] = None,
    safe: bool = False,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    lineart_config = LineartAnimeConfig(weights_path=weights_path or config.lineart_anime_weights)
    model = load_lineart_anime_model(lineart_config)
    device = image.device
    model = model.to(device=device, dtype=torch.float32)
    outputs = []
    for sample in image:
        sample = sample.unsqueeze(0).to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        h, w = sample.shape[-2:]
        target_h = 256 * math.ceil(h / 256)
        target_w = 256 * math.ceil(w / 256)
        resized = F.interpolate(sample, size=(target_h, target_w), mode="bilinear", align_corners=False)
        feed = resized * 2.0 - 1.0
        with torch.no_grad():
            line = model(feed)[0, 0]
        line = (line + 1.0) * 0.5
        line = F.interpolate(line.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)
        if safe:
            line = line.clamp_(0.0, 1.0)
        outputs.append(line)
    stacked = torch.cat(outputs, dim=0)
    metadata = _build_metadata("lineart_anime", {"weights": lineart_config.weights_path, "safe": safe})
    return PreprocessorResult(image=stacked, metadata=metadata)


def preprocess_manga_line(
    image: torch.Tensor,
    config: EdgePreprocessorConfig,
    *,
    weights_path: Optional[str] = None,
    safe: bool = False,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    manga_config = MangaLineConfig(weights_path=weights_path or config.manga_line_weights)
    model = load_manga_line_model(manga_config)
    device = image.device
    model = model.to(device=device, dtype=torch.float32)
    outputs = []
    for sample in image:
        sample = _to_grayscale(sample.unsqueeze(0)).to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        with torch.no_grad():
            line = model(sample)
        line = (line + 1.0) * 0.5
        if safe:
            line = line.clamp_(0.0, 1.0)
        outputs.append(line)
    stacked = torch.cat(outputs, dim=0)
    metadata = _build_metadata("manga_line", {"weights": manga_config.weights_path, "safe": safe})
    return PreprocessorResult(image=stacked, metadata=metadata)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("image must have shape [B, C, H, W]")
    if image.size(1) not in (1, 3):
        raise ValueError("image channel dimension must be 1 or 3")
    if image.dtype not in (torch.float16, torch.float32, torch.float64):
        image = image.float()
    return image


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.size(1) == 1:
        return image
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.amin(dim=(-2, -1), keepdim=True)
    max_val = tensor.amax(dim=(-2, -1), keepdim=True)
    eps = torch.finfo(tensor.dtype).eps
    return (tensor - min_val) / (max_val - min_val + eps)


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device,
        dtype=dtype,
    )
    gy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=device,
        dtype=dtype,
    )
    gx = gx.view(1, 1, 3, 3)
    gy = gy.view(1, 1, 3, 3)
    return gx, gy


def _gaussian_kernel(kernel_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("gaussian kernel_size must be odd")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    x = coords.view(1, -1)
    y = coords.view(-1, 1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _canny_manual(
    image: torch.Tensor,
    low: float,
    high: float,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    grayscale = _to_grayscale(image)
    gauss = _gaussian_kernel(kernel_size, sigma, device=grayscale.device, dtype=grayscale.dtype)
    blurred = F.conv2d(grayscale, gauss, padding=kernel_size // 2)
    gx, gy = _sobel_filters(blurred.device, blurred.dtype)
    grad_x = F.conv2d(blurred, gx, padding=1)
    grad_y = F.conv2d(blurred, gy, padding=1)

    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    orientation = torch.atan2(grad_y, grad_x)

    suppressed = _non_maximum_suppression(magnitude, orientation)
    high_mask = suppressed >= high
    low_mask = suppressed >= low

    strong = high_mask.clone()
    weak = low_mask & ~high_mask

    output = _hysteresis(strong, weak)
    return output.to(image.dtype)


def _non_maximum_suppression(magnitude: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    angle = orientation * (180.0 / torch.pi)
    angle = (angle + 180.0) % 180.0

    directions = torch.zeros_like(angle, dtype=torch.int64)
    directions[(angle >= 0) & (angle < 22.5)] = 0
    directions[(angle >= 22.5) & (angle < 67.5)] = 1
    directions[(angle >= 67.5) & (angle < 112.5)] = 2
    directions[(angle >= 112.5) & (angle < 157.5)] = 3
    directions[(angle >= 157.5) & (angle < 180.0)] = 0

    padded = F.pad(magnitude, (1, 1, 1, 1), mode="replicate")
    h, w = magnitude.shape[-2:]
    idx = torch.arange(h * w, device=magnitude.device)
    idx = idx.view(1, 1, h, w)

    offsets = {
        0: (0, 1),
        1: (1, 1),
        2: (1, 0),
        3: (1, -1),
    }

    suppressed = torch.zeros_like(magnitude)
    for direction, (dy, dx) in offsets.items():
        mask = directions == direction
        shifted_pos = padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        shifted_neg = padded[..., 1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep = (magnitude >= shifted_pos) & (magnitude >= shifted_neg) & mask
        suppressed = torch.where(keep, magnitude, suppressed)
    return suppressed


def _hysteresis(strong: torch.Tensor, weak: torch.Tensor) -> torch.Tensor:
    output = strong.clone()
    kernel = torch.ones((1, 1, 3, 3), device=strong.device, dtype=strong.dtype)
    prev_sum = torch.zeros_like(output)

    for _ in range(5):
        neighbors = F.conv2d(output.float(), kernel, padding=1)
        to_promote = (neighbors > 0) & weak
        output = output | to_promote
        if torch.allclose(output.float(), prev_sum):
            break
        prev_sum = output.float()
    return output.float()


def _build_metadata(name: str, extra: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    metadata: dict[str, object] = {"preprocessor": name}
    if extra:
        metadata.update(extra)
    return metadata
