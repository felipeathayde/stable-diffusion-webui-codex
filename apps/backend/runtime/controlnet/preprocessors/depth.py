"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Depth preprocessors for ControlNet (DPT, LeReS, ZoeDepth).
Implements depth map extraction pipelines and registers them into `ControlPreprocessorRegistry`.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for depth preprocessing diagnostics.
- `DepthPreprocessorConfig` (dataclass): Configuration for the DPT hybrid depth preprocessor.
- `register_depth_preprocessors` (function): Registers depth preprocessors into a registry.
- `preprocess_dpt_hybrid` (function): Depth map extraction via Hugging Face transformers DPT model.
- `preprocess_leres` (function): Depth map extraction via the LeReS model.
- `preprocess_zoe` (function): Depth map extraction via the ZoeDepth model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
except ImportError:  # pragma: no cover - transformers optional
    DPTForDepthEstimation = None
    DPTImageProcessor = None

from apps.backend.huggingface.assets import ensure_repo_minimal_files
from .registry import ControlPreprocessorRegistry, PreprocessorResult
from .models.leres import LeReSConfig, load_leres_model

from .models.zoe import ZoeDepthConfig, load_zoe_model

logger = logging.getLogger("backend.runtime.controlnet.preprocessors.depth")


@dataclass(frozen=True)
class DepthPreprocessorConfig:
    repo_id: str = "Intel/dpt-hybrid-midas"
    cache_subdir: str = "depth/dpt-hybrid-midas"
    image_size: int = 384


def register_depth_preprocessors(registry: ControlPreprocessorRegistry, *, config: Optional[DepthPreprocessorConfig] = None) -> None:
    cfg = config or DepthPreprocessorConfig()
    registry.register("depth_dpt_hybrid", lambda image, **kwargs: preprocess_dpt_hybrid(image, cfg, **kwargs))
    registry.register("depth_leres", lambda image, **kwargs: preprocess_leres(image, **kwargs))
    registry.register("depth_zoe", lambda image, **kwargs: preprocess_zoe(image, **kwargs))


def preprocess_dpt_hybrid(
    image: torch.Tensor,
    config: DepthPreprocessorConfig,
    *,
    repo_id: Optional[str] = None,
    to_meters: bool = False,
) -> PreprocessorResult:
    if DPTForDepthEstimation is None or DPTImageProcessor is None:
        raise RuntimeError("transformers library is required for DPT depth preprocessing")

    image = _ensure_image_batch(image)
    repo = repo_id or config.repo_id
    cache_dir = _resolve_cache_dir(config.cache_subdir, repo)
    processor = _load_dpt_processor(repo, cache_dir)
    model = _load_dpt_model(repo, cache_dir)
    device = image.device
    model = model.to(device=device)
    outputs = []
    with torch.no_grad():
        for sample in image:
            sample = sample.permute(1, 2, 0).cpu().numpy()
            inputs = processor(images=sample, return_tensors="pt")
            inputs = {k: v.to(device=device) for k, v in inputs.items()}
            pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(image.shape[-2], image.shape[-1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            if not to_meters:
                pred = pred - pred.amin(dim=(-2, -1), keepdim=True)
                pred = pred / (pred.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            outputs.append(pred.unsqueeze(1))
    depth_map = torch.cat(outputs, dim=0)
    metadata = {
        "preprocessor": "depth_dpt_hybrid",
        "repo_id": repo,
        "image_size": config.image_size,
        "to_meters": to_meters,
    }
    return PreprocessorResult(image=depth_map, metadata=metadata)


def preprocess_leres(
    image: torch.Tensor,
    *,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    config = LeReSConfig(weights_path=weights_path or LeReSConfig.weights_path, device=device, dtype=dtype)
    model = load_leres_model(config)
    outputs = []
    with torch.no_grad():
        for sample in image:
            sample = sample.unsqueeze(0)
            depth = model(sample)
            depth = depth - depth.amin(dim=(-2, -1), keepdim=True)
            depth = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            outputs.append(depth)
    depth_map = torch.cat(outputs, dim=0)
    metadata = {
        "preprocessor": "depth_leres",
        "weights_path": config.weights_path,
    }
    return PreprocessorResult(image=depth_map, metadata=metadata)


def preprocess_zoe(
    image: torch.Tensor,
    *,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    config = ZoeDepthConfig(weights_path=weights_path or ZoeDepthConfig.weights_path, device=device, dtype=dtype)
    model = load_zoe_model(config)
    outputs = []
    with torch.no_grad():
        for sample in image:
            sample = sample.unsqueeze(0).to(device or model.device)
            sample = sample.clamp_(0.0, 1.0)
            depth = model.infer(sample)["metric_depth"] if hasattr(model, "infer") else model(sample)
            if isinstance(depth, dict):
                depth = depth["metric_depth"]
            depth = depth.squeeze(1)
            # Percentile normalization similar to original Zoe pipeline
            vmin = torch.quantile(depth.view(depth.size(0), -1), 0.02, dim=1, keepdim=True)
            vmax = torch.quantile(depth.view(depth.size(0), -1), 0.85, dim=1, keepdim=True)
            depth = (depth - vmin.unsqueeze(-1)).clamp_min(0)
            depth = depth / (vmax.unsqueeze(-1) - vmin.unsqueeze(-1) + 1e-6)
            depth = 1.0 - depth
            outputs.append(depth.unsqueeze(1))
    depth_map = torch.cat(outputs, dim=0)
    metadata = {
        "preprocessor": "depth_zoe",
        "weights_path": config.weights_path,
    }
    return PreprocessorResult(image=depth_map, metadata=metadata)


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("image must have shape [B, C, H, W]")
    if image.size(1) not in (1, 3):
        raise ValueError("image channel dimension must be 1 or 3")
    if image.dtype != torch.float32:
        image = image.float()
    if image.size(1) == 1:
        image = image.repeat(1, 3, 1, 1)
    return image


def _resolve_cache_dir(subdir: str, repo_id: str) -> str:
    cache_root = torch.hub.get_dir()
    target = os.path.join(cache_root, "codex", "controlnet", subdir)
    os.makedirs(target, exist_ok=True)
    ensure_repo_minimal_files(
        repo_id,
        target,
        include=("config", "tokenizer", "scheduler"),
        offline=False,
    )
    return target


@lru_cache(maxsize=1)
def _load_dpt_processor(repo_id: str, cache_dir: str):
    return DPTImageProcessor.from_pretrained(repo_id, cache_dir=cache_dir)


@lru_cache(maxsize=1)
def _load_dpt_model(repo_id: str, cache_dir: str):
    model = DPTForDepthEstimation.from_pretrained(repo_id, cache_dir=cache_dir)
    model.eval()
    return model
