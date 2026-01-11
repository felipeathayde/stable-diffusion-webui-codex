"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Registry for custom LoRA patch calculators.
Allows extension points to register named patch handlers used by the merge path.

Symbols (top-level; keep in sync; no ghosts):
- `extra_weight_calculators` (dict): Maps custom patch tags to calculator callables.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

extra_weight_calculators: Dict[str, Callable[[torch.Tensor, float, object], torch.Tensor]] = {}

__all__ = [
    "extra_weight_calculators",
]

