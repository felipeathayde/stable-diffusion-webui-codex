"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 GGUF runtime facade (entrypoints + config types).
Keeps the stable import path used by WAN engines by re-exporting the run functions and config dataclasses implemented in focused modules.

Symbols (top-level; keep in sync; no ghosts):
- `StageConfig` (class): Stage-level configuration for a WAN run (stage GGUF path + sampler/scheduler/steps/cfg + optional LoRA).
- `RunConfig` (class): Full run configuration for txt2vid/img2vid (assets, devices/dtypes, stage configs).
- `run_txt2vid` (function): Batch txt2vid runner (GGUF stages → sampling → VAE decode).
- `stream_txt2vid` (function): Streaming txt2vid generator yielding progress events and final frames.
- `run_img2vid` (function): Batch img2vid runner (VAE encode init → stages → VAE decode).
- `stream_img2vid` (function): Streaming img2vid generator yielding progress events and final frames.
- `__all__` (constant): Export list for the WAN22 GGUF runtime facade.
"""

from __future__ import annotations

from .config import RunConfig, StageConfig
from .run import run_img2vid, run_txt2vid, stream_img2vid, stream_txt2vid

__all__ = [
    "RunConfig",
    "StageConfig",
    "run_txt2vid",
    "run_img2vid",
    "stream_txt2vid",
    "stream_img2vid",
]
