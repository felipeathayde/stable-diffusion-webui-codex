"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed video export/interpolation option payloads.
Holds ffmpeg export options and optional interpolation settings to be passed through request schemas and UI/service layers.

Symbols (top-level; keep in sync; no ghosts):
- `VideoExportOptions` (dataclass): Export parameters for video output (format/codec knobs + metadata/save flags).
- `VideoInterpolationOptions` (dataclass): Optional interpolation settings (enable + model + times).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class VideoExportOptions:
    filename_prefix: Optional[str] = None
    format: Optional[str] = None
    pix_fmt: Optional[str] = None
    crf: Optional[int] = None
    loop_count: Optional[int] = None
    pingpong: Optional[bool] = None
    save_metadata: Optional[bool] = None
    save_output: Optional[bool] = None
    trim_to_audio: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VideoInterpolationOptions:
    enabled: bool = False
    model: Optional[str] = None
    times: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


__all__ = ["VideoExportOptions", "VideoInterpolationOptions"]
