"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: K-diffusion-style wrapper package for sampling adapters.
This module re-exports k-diffusion oriented wrappers and prediction helpers used by the sampling stack and engine specs.

Symbols (top-level; keep in sync; no ghosts):
- `KModel` (class): K-diffusion-style wrapper implementing `apply_model` for sampler call sites.
- `AbstractPrediction` (class): Base prediction module describing how schedulers interpret sigmas/timesteps.
- `Prediction` (class): Concrete prediction adapter (diffusion) derived from scheduler configuration.
- `FlowMatchEulerPrediction` (class): Flow-match Euler prediction adapter used by some Flow/EDM schedulers.
- `k_prediction_from_diffusers_scheduler` (function): Build a prediction adapter from a diffusers scheduler instance.
- `rescale_zero_terminal_snr_sigmas` (function): Rescale sigmas for zero-terminal-SNR schedules.
- `k_diffusion_extra` (module): Additional k-diffusion compatibility utilities.
"""

from .k_model import KModel
from .k_prediction import (
    AbstractPrediction,
    Prediction,
    FlowMatchEulerPrediction,
    k_prediction_from_diffusers_scheduler,
    rescale_zero_terminal_snr_sigmas,
)
from . import k_diffusion_extra

__all__ = [
    "AbstractPrediction",
    "FlowMatchEulerPrediction",
    "KModel",
    "Prediction",
    "k_diffusion_extra",
    "k_prediction_from_diffusers_scheduler",
    "rescale_zero_terminal_snr_sigmas",
]
