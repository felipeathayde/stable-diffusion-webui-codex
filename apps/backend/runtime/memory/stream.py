"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Best-effort CUDA/XPU stream helpers for async swap operations (when `swap_method=async`).

Symbols (top-level; keep in sync; no ghosts):
- `stream_context` (function): Return the torch stream context manager for the active backend, or None.
- `get_current_stream` (function): Return the current device stream when safe/available.
- `get_new_stream` (function): Create and validate a new device stream when safe/available.
- `should_use_stream` (function): True when streams are activated and both streams are available.
- `stream_activated` (constant): Whether async swapping is enabled by config (`args.swap_method=="async"`).
- `current_stream` (constant): Best-effort current stream object (or None).
- `mover_stream` (constant): Best-effort mover stream object (or None).
"""

import torch
from apps.backend.infra.config.args import args


def stream_context():
    if torch.cuda.is_available():
        return torch.cuda.stream

    if torch.xpu.is_available():
        return torch.xpu.stream

    return None


def get_current_stream():
    # Use swap-method policy: async -> attempt streams even on Windows.
    if args.swap_method != "async":
        return None
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.current_stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if torch.xpu.is_available():
            device = torch.device("xpu")
            stream = torch.xpu.current_stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except Exception:  # noqa: BLE001
        return None


def get_new_stream():
    if args.swap_method != "async":
        return None
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if torch.xpu.is_available():
            device = torch.device("xpu")
            stream = torch.xpu.Stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except Exception:  # noqa: BLE001
        return None


def should_use_stream():
    return stream_activated and current_stream is not None and mover_stream is not None


stream_activated = args.swap_method == "async"
current_stream = get_current_stream()
mover_stream = get_new_stream()
