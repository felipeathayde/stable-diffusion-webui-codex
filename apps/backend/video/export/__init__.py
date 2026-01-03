"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video export facade for backend video workflows.
Re-exports the ffmpeg-based `export_video(...)` helper and its result type used by txt2vid/img2vid/vid2vid pipelines.

Symbols (top-level; keep in sync; no ghosts):
- `VideoExportResult` (dataclass): Video export metadata returned by `export_video` (re-export).
- `export_video` (function): Exports a frame sequence to a video container via ffmpeg (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .ffmpeg_exporter import VideoExportResult, export_video

__all__ = [
    "VideoExportResult",
    "export_video",
]
