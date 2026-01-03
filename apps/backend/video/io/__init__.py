"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video IO facade for backend video tasks.
Re-exports ffmpeg/ffprobe-backed helpers used to probe videos and extract frames for video generation workflows.

Symbols (top-level; keep in sync; no ghosts):
- `FFmpegUnavailableError` (class): Raised when ffmpeg/ffprobe are missing or fail (re-export).
- `VideoProbe` (dataclass): Video metadata returned by `probe_video` (re-export).
- `extract_frames` (function): Extracts frames from a video file using ffmpeg (re-export).
- `probe_video` (function): Probes a video file using ffprobe (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .ffmpeg import FFmpegUnavailableError, VideoProbe, extract_frames, probe_video

__all__ = [
    "FFmpegUnavailableError",
    "VideoProbe",
    "extract_frames",
    "probe_video",
]
