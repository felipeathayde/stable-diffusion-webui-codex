"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Optical flow guidance facade for video workflows.
Re-exports flow estimation and warping helpers used by flow-guided video pipelines, with lazy imports in the underlying implementation.

Symbols (top-level; keep in sync; no ghosts):
- `FlowGuidanceError` (class): Error raised when flow guidance is enabled but required deps are missing (re-export).
- `RaftFlowEstimator` (dataclass): Torchvision RAFT optical flow estimator (lazy-loaded) (re-export).
- `warp_frame` (function): Warps a frame using a backward flow field (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .torchvision_raft import FlowGuidanceError, RaftFlowEstimator, warp_frame

__all__ = [
    "FlowGuidanceError",
    "RaftFlowEstimator",
    "warp_frame",
]
