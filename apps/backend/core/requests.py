"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed request and event objects for backend engines/orchestration.
Defines progress/result events and request payload dataclasses for image and video tasks, used by the API, orchestrator, engines, and services.

Symbols (top-level; keep in sync; no ghosts):
- `ProgressEvent` (dataclass): Progress update event (stage/percent/step + optional metadata).
- `ResultEvent` (dataclass): Result event carrying an engine payload and optional metadata.
- `InferenceEvent` (type alias): Union of `ProgressEvent` and `ResultEvent` produced by engines.
- `BaseRequest` (dataclass): Shared request fields across tasks (prompt/sampler/seed/LoRA/etc).
- `Txt2ImgRequest` (dataclass): Text-to-image request.
- `Img2ImgRequest` (dataclass): Image-to-image/inpaint request (init image + optional mask).
- `Txt2VidRequest` (dataclass): Text-to-video request.
- `Img2VidRequest` (dataclass): Image-to-video request.
- `Vid2VidRequest` (dataclass): Video-to-video request (source video + optional reference/pose/background inputs).
- `ImageRequest` (type alias): Union of image request dataclasses accepted by image engines.
- `VideoRequest` (type alias): Union of video request dataclasses accepted by video engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

from .engine_interface import TaskType


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    percent: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultEvent:
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


InferenceEvent = Union[ProgressEvent, ResultEvent]


# ---------------------------------------------------------------------------
# Shared request utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRequest:
    task: TaskType
    prompt: str
    negative_prompt: str = ""
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    batch_size: int = 1
    loras: Sequence[str] = field(default_factory=tuple)
    extra_networks: Sequence[str] = field(default_factory=tuple)
    clip_skip: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    smart_offload: bool = False
    smart_fallback: bool = False
    smart_cache: bool = True


@dataclass(frozen=True)
class Txt2ImgRequest(BaseRequest):
    width: int = 512
    height: int = 512
    steps: int = 20
    tiling: bool = False
    highres_fix: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Img2ImgRequest(BaseRequest):
    init_image: Any = None
    mask: Any = None
    mask_enforcement: Optional[str] = None
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    mask_blur: int = 4
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_round: bool = True
    denoise_strength: float = 0.5
    width: int = 512
    height: int = 512
    steps: int = 20
    extras: Mapping[str, Any] = field(default_factory=dict)
    highres_fix: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Txt2VidRequest(BaseRequest):
    width: int = 768
    height: int = 432
    steps: int = 30
    num_frames: int = 16
    fps: int = 24
    motion_strength: Optional[float] = None
    extrapolation: Optional[str] = None
    video_options: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Img2VidRequest(BaseRequest):
    init_image: Any = None
    width: int = 768
    height: int = 432
    steps: int = 30
    num_frames: int = 16
    fps: int = 24
    motion_strength: Optional[float] = None
    video_options: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Vid2VidRequest(BaseRequest):
    """Video-to-video request (source video + prompt → output video)."""

    video_path: str = ""
    # Optional reference conditioning for WAN Animate / similar pipelines.
    reference_image: Any = None
    pose_video_path: str = ""
    face_video_path: str = ""
    background_video_path: str = ""
    mask_video_path: str = ""
    animate_mode: str = "animate"  # 'animate' or 'replace'
    segment_frame_length: int = 77
    prev_segment_conditioning_frames: int = 1
    motion_encode_batch_size: Optional[int] = None

    width: int = 768
    height: int = 432
    steps: int = 30
    num_frames: int = 16
    fps: int = 24
    strength: Optional[float] = None
    motion_strength: Optional[float] = None
    video_options: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# Convenience tuple for type checkers / consumers that need to accept either
ImageRequest = Union[Txt2ImgRequest, Img2ImgRequest]
VideoRequest = Union[Txt2VidRequest, Img2VidRequest, Vid2VidRequest]
