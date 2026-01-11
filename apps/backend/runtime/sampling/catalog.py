"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler/scheduler option catalog (canonical names + defaults) for the WebUI/API surface.
Defines the canonical sampler/scheduler names exposed to users, plus default scheduler selection rules used by the sampling driver and UI selectors.

Symbols (top-level; keep in sync; no ghosts):
- `SAMPLER_OPTIONS` (constant): UI-facing sampler option table (canonical name + optional scheduler allowlists).
- `SUPPORTED_SAMPLERS` (constant): Set of supported sampler canonical names.
- `SCHEDULER_OPTIONS` (constant): UI-facing scheduler option table (canonical name only).
- `SUPPORTED_SCHEDULERS` (constant): Set of supported scheduler canonical names.
- `SAMPLER_DEFAULT_SCHEDULER` (constant): Default scheduler per sampler (used by UI and sampling plan defaults).
"""

from __future__ import annotations

from typing import Dict, List, Set


SAMPLER_OPTIONS: List[Dict[str, object]] = [
    {"name": "euler", "supported": True},
    {"name": "euler a", "supported": True},
    {"name": "euler cfg++", "supported": True},
    {"name": "euler a cfg++", "supported": True},
    {"name": "heun", "supported": True},
    {"name": "heunpp2", "supported": True},
    {"name": "lms", "supported": True},
    {"name": "ddim", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "ddim cfg++", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "plms", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "pndm", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "dpm++ 2m", "supported": True},
    {"name": "dpm++ 2m cfg++", "supported": True},
    {"name": "dpm++ 2m sde", "supported": True},
    {"name": "dpm++ 2m sde heun", "supported": True},
    {"name": "dpm++ 2m sde gpu", "supported": True},
    {"name": "dpm++ 2m sde heun gpu", "supported": True},
    {"name": "dpm++ sde", "supported": True},
    {"name": "dpm++ 2s ancestral", "supported": True},
    {"name": "dpm++ 2s ancestral cfg++", "supported": True},
    {"name": "dpm++ 3m sde", "supported": True},
    {"name": "dpm 2", "supported": True},
    {"name": "dpm 2 ancestral", "supported": True},
    {"name": "dpm fast", "supported": True},
    {"name": "dpm adaptive", "supported": True},
    {"name": "uni-pc", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "uni-pc bh2", "supported": True, "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "ddpm", "supported": True},
    {"name": "lcm", "supported": True},
    {"name": "ipndm", "supported": True},
    {"name": "ipndm v", "supported": True},
    {"name": "deis", "supported": True},
    {"name": "res multistep", "supported": True},
    {"name": "res multistep cfg++", "supported": True},
    {"name": "res multistep ancestral", "supported": True},
    {"name": "res multistep ancestral cfg++", "supported": True},
    {"name": "gradient estimation", "supported": True},
    {"name": "gradient estimation cfg++", "supported": True},
    {"name": "er sde", "supported": True},
    {"name": "seeds 2", "supported": True},
    {"name": "seeds 3", "supported": True},
    {"name": "sa-solver", "supported": True},
    {"name": "sa-solver pece", "supported": True},
    {"name": "restart", "supported": True},
]

SUPPORTED_SAMPLERS: Set[str] = {entry["name"] for entry in SAMPLER_OPTIONS if entry.get("supported", True)}

SCHEDULER_OPTIONS: List[Dict[str, object]] = [
    {"name": "uniform", "supported": True},
    {
        "name": "karras",
        "supported": True,
    },
    {
        "name": "exponential",
        "supported": True,
    },
    {
        "name": "polyexponential",
        "supported": True,
    },
    {
        "name": "simple",
        "supported": True,
    },
    {
        "name": "euler_discrete",
        "supported": True,
    },
    {"name": "ddim", "supported": True},
    {
        "name": "sgm_uniform",
        "supported": True,
    },
    {
        "name": "ddim_uniform",
        "supported": True,
    },
    {
        "name": "beta",
        "supported": True,
    },
    {
        "name": "normal",
        "supported": True,
    },
    {
        "name": "linear_quadratic",
        "supported": True,
    },
    {
        "name": "kl_optimal",
        "supported": True,
    },
    {"name": "turbo", "supported": True},
    {"name": "align_your_steps", "supported": True},
    {"name": "align_your_steps_gits", "supported": True},
    {"name": "align_your_steps_11", "supported": True},
    {"name": "align_your_steps_32", "supported": True},
]

SUPPORTED_SCHEDULERS: Set[str] = {entry["name"] for entry in SCHEDULER_OPTIONS if entry.get("supported", True)}

# Default scheduler per sampler.
SAMPLER_DEFAULT_SCHEDULER: Dict[str, str] = {
    # Euler-family samplers default to the predictor ladder ("simple" schedule).
    "euler": "simple",
    "euler a": "simple",
    "euler cfg++": "euler_discrete",
    "euler a cfg++": "euler_discrete",
    "heun": "karras",
    "heunpp2": "karras",
    "lms": "karras",
    "ddim": "ddim",
    "ddim cfg++": "ddim",
    "plms": "karras",
    "pndm": "ddim",
    "dpm++ 2m": "karras",
    "dpm++ 2m cfg++": "karras",
    "dpm++ sde": "karras",
    "dpm++ 2m sde": "exponential",
    "dpm++ 2m sde heun": "exponential",
    "dpm++ 2m sde gpu": "exponential",
    "dpm++ 2m sde heun gpu": "exponential",
    "dpm++ 2s ancestral": "karras",
    "dpm++ 2s ancestral cfg++": "karras",
    "dpm++ 3m sde": "exponential",
    "dpm++ 3m sde gpu": "exponential",
    "dpm 2": "karras",
    "dpm 2 ancestral": "karras",
    "dpm fast": "karras",
    "dpm adaptive": "karras",
    "uni-pc": "simple",
    "uni-pc bh2": "simple",
    "ddpm": "beta",
    "lcm": "karras",
    "ipndm": "karras",
    "ipndm v": "karras",
    "deis": "karras",
    "res multistep": "karras",
    "res multistep cfg++": "karras",
    "res multistep ancestral": "karras",
    "res multistep ancestral cfg++": "karras",
    "gradient estimation": "karras",
    "gradient estimation cfg++": "karras",
    "er sde": "exponential",
    "seeds 2": "karras",
    "seeds 3": "karras",
    "sa-solver": "karras",
    "sa-solver pece": "karras",
    "restart": "karras",
}

__all__ = [
    "SAMPLER_OPTIONS",
    "SUPPORTED_SAMPLERS",
    "SCHEDULER_OPTIONS",
    "SUPPORTED_SCHEDULERS",
    "SAMPLER_DEFAULT_SCHEDULER",
]
