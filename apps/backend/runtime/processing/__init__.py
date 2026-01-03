"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native processing primitives describing a generation run (txt2img/img2img/video) without legacy `modules.*` wrappers.
Re-exports the main processing config classes plus shared dataclasses used across workflows and engines.

Symbols (top-level; keep in sync; no ghosts):
- `CodexProcessingBase` (class): Base processing configuration shared across tasks (prompt, seeds, smart flags, etc.).
- `CodexProcessingTxt2Img` (class): Processing configuration for txt2img runs.
- `CodexProcessingImg2Img` (class): Processing configuration for img2img runs (init image/mask + strength/denoise).
- `CodexHighResConfig` (class): High-res pass configuration wrapper for two-pass workflows.
- `RefinerConfig` (class): SDXL refiner configuration (checkpoint + steps/denoise/cfg).
- `PromptContext` (dataclass): Normalized prompt state after preprocessing (prompts/negatives/loras/controls/metadata).
- `ConditioningPayload` (dataclass): Conditioning tensors assembled for a generation pass (cond/uncond + extras).
- `SamplingPlan` (dataclass): Complete specification of a sampling run (sampler/scheduler/steps/seeds/noise settings).
- `HiResPlan` (dataclass): High-resolution second pass plan (target size + steps/denoise/cfg).
- `InitImageBundle` (dataclass): Bundle derived from an initial image (pixels/latents + optional mask).
- `GenerationResult` (dataclass): Outputs and diagnostics from a generation pass (samples/decoded/metadata/applied_extras).
- `ExtraNetworkDescriptor` (dataclass): Parsed extra network descriptor (e.g. LoRA) with weight + metadata.
- `AppliedExtra` (dataclass): Record of applied extra network or post-processing effect.
- `VideoPlan` (dataclass): Execution plan for video workflows (frames/fps/steps/scheduler + extras).
- `VideoResult` (dataclass): Result bundle for video workflows (frames + metadata).
- `__all__` (constant): Export list for the processing facade.
"""

from .models import (
    CodexHighResConfig,
    RefinerConfig,
    CodexProcessingBase,
    CodexProcessingImg2Img,
    CodexProcessingTxt2Img,
)
from .datatypes import (
    PromptContext,
    ConditioningPayload,
    SamplingPlan,
    HiResPlan,
    InitImageBundle,
    GenerationResult,
    ExtraNetworkDescriptor,
    AppliedExtra,
    VideoPlan,
    VideoResult,
)

__all__ = [
    "CodexProcessingBase",
    "CodexProcessingTxt2Img",
    "CodexProcessingImg2Img",
    "CodexHighResConfig",
    "RefinerConfig",
    "PromptContext",
    "ConditioningPayload",
    "SamplingPlan",
    "HiResPlan",
    "InitImageBundle",
    "GenerationResult",
    "ExtraNetworkDescriptor",
    "AppliedExtra",
    "VideoPlan",
    "VideoResult",
]
