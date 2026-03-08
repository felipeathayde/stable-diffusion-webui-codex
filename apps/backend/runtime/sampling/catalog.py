"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler/scheduler option catalog (canonical names + defaults) for the WebUI/API surface.
Defines the canonical sampler/scheduler names exposed to users, derives executable sampler support from the current runtime-backed implementation
surface, and provides default scheduler selection rules used by the sampling driver and UI selectors.

Symbols (top-level; keep in sync; no ghosts):
- `SAMPLER_OPTIONS` (constant): UI-facing sampler option table (canonical name + optional scheduler allowlists).
- `SUPPORTED_SAMPLERS` (constant): Set of supported sampler canonical names.
- `SCHEDULER_OPTIONS` (constant): UI-facing scheduler option table (canonical name only).
- `SUPPORTED_SCHEDULERS` (constant): Set of supported scheduler canonical names.
- `SAMPLER_DEFAULT_SCHEDULER` (constant): Default scheduler per sampler (used by UI and sampling plan defaults).
"""

from __future__ import annotations

from typing import Dict, List, Set


SUPPORTED_SAMPLERS: Set[str] = {
    "euler",
    "euler a",
    "ddim",
    "plms",
    "pndm",
    "dpm++ 2m",
    "dpm++ 2m sde",
    "dpm fast",
    "uni-pc",
    "uni-pc bh2",
    "er sde",
}

SAMPLER_OPTIONS: List[Dict[str, object]] = [
    {"name": "euler"},
    {"name": "euler a"},
    {"name": "euler cfg++"},
    {"name": "euler a cfg++"},
    {"name": "heun"},
    {"name": "heunpp2"},
    {"name": "lms"},
    {"name": "ddim", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "ddim cfg++", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "plms", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "pndm", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "dpm++ 2m"},
    {"name": "dpm++ 2m cfg++"},
    {"name": "dpm++ 2m sde"},
    {"name": "dpm++ 2m sde heun"},
    {"name": "dpm++ 2m sde gpu"},
    {"name": "dpm++ 2m sde heun gpu"},
    {"name": "dpm++ sde"},
    {"name": "dpm++ 2s ancestral"},
    {"name": "dpm++ 2s ancestral cfg++"},
    {"name": "dpm++ 3m sde"},
    {"name": "dpm 2"},
    {"name": "dpm 2 ancestral"},
    {"name": "dpm fast"},
    {"name": "dpm adaptive"},
    {"name": "uni-pc", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "uni-pc bh2", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "ddpm"},
    {"name": "lcm"},
    {"name": "ipndm"},
    {"name": "ipndm v"},
    {"name": "deis"},
    {"name": "res multistep"},
    {"name": "res multistep cfg++"},
    {"name": "res multistep ancestral"},
    {"name": "res multistep ancestral cfg++"},
    {"name": "gradient estimation"},
    {"name": "gradient estimation cfg++"},
    {"name": "er sde"},
    {"name": "seeds 2"},
    {"name": "seeds 3"},
    {"name": "sa-solver"},
    {"name": "sa-solver pece"},
    {"name": "restart"},
]

for entry in SAMPLER_OPTIONS:
    entry["supported"] = str(entry["name"]) in SUPPORTED_SAMPLERS

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
