"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling adapter wrappers used by samplers/patchers and engine specs.
This module re-exports the adapter model wrapper and prediction helpers used by the native sampling stack.

Symbols (top-level; keep in sync; no ghosts):
- `SamplerModel` (class): Wrapper implementing `apply_model` for sampler call sites.
- `AbstractPrediction` (class): Base prediction module describing how schedulers interpret sigmas/timesteps.
- `Prediction` (class): Concrete prediction adapter (diffusion) derived from scheduler configuration.
- `FlowMatchEulerPrediction` (class): Flow-match Euler prediction adapter used by some Flow/EDM schedulers.
- `prediction_from_diffusers_scheduler` (function): Build a prediction adapter from a diffusers scheduler instance.
- `rescale_zero_terminal_snr_sigmas` (function): Rescale sigmas for zero-terminal-SNR schedules.
- `extra` (module): Optional sampler extras (native-only).
"""

from .sampler_model import SamplerModel
from .prediction import (
    AbstractPrediction,
    Prediction,
    FlowMatchEulerPrediction,
    prediction_from_diffusers_scheduler,
    rescale_zero_terminal_snr_sigmas,
)
from . import extra

__all__ = [
    "AbstractPrediction",
    "FlowMatchEulerPrediction",
    "SamplerModel",
    "Prediction",
    "extra",
    "prediction_from_diffusers_scheduler",
    "rescale_zero_terminal_snr_sigmas",
]
