"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared primitives for segment-based runtime streaming across families.

Symbols (top-level; keep in sync; no ghosts):
- `StreamingPolicy` (enum): Streaming policy (`naive`/`window`/`aggressive`) controlling segment residency.
- `TransferStats` (dataclass): Tracks CPU↔GPU transfer bytes/counts/time for streaming telemetry.
- `StreamingController` (dataclass): Generic streaming controller operating on segment objects (duck-typed).
"""

from .controller import StreamingController, StreamingPolicy, TransferStats

__all__ = [
    "StreamingController",
    "StreamingPolicy",
    "TransferStats",
]

