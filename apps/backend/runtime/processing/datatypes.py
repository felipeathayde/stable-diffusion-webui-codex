"""Shared dataclasses for Codex generation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from apps.backend.core.rng import NoiseSettings


@dataclass(slots=True)
class ExtraNetworkDescriptor:
    """Structured description of an extra network (e.g. LoRA) parsed from prompts."""

    path: str
    weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptContext:
    """Normalized prompt state after preprocessing."""

    prompts: List[str]
    negative_prompts: List[str]
    loras: Sequence[Any]
    controls: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConditioningPayload:
    """Conditioning tensors assembled for a generation pass."""

    conditioning: Any
    unconditional: Any
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingPlan:
    """Complete specification of a sampling run."""

    sampler_name: str | None
    scheduler_name: str | None
    steps: int
    guidance_scale: float
    seeds: List[int]
    subseeds: List[int]
    subseed_strength: float
    noise_settings: NoiseSettings


@dataclass(slots=True)
class HiResPlan:
    """High-resolution second pass configuration."""

    enabled: bool
    target_width: int
    target_height: int
    steps: int
    denoise: float
    cfg_scale: float | None
    checkpoint_name: str | None
    additional_modules: Sequence[str] | None = None


@dataclass(slots=True)
class InitImageBundle:
    """Inputs derived from an initial image (img2img/img2vid/hires)."""

    tensor: Any
    latents: Any | None
    mask: Any | None = None
    mode: str = "pixel"  # or "latent"


@dataclass(slots=True)
class AppliedExtra:
    """Record of applied extra network or post-processing effect."""

    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """Outputs and diagnostics from a generation pass."""

    samples: Any
    decoded: Any | None
    applied_extras: List[AppliedExtra] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoPlan:
    """Execution plan for video generation tasks."""

    sampler_name: str | None
    scheduler_name: str | None
    steps: int
    frames: int
    fps: int
    width: int
    height: int
    guidance_scale: float | None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoResult:
    """Result bundle for video workflows."""

    frames: Sequence[Any]
    metadata: Dict[str, Any]
    video_meta: Dict[str, Any] | None = None
