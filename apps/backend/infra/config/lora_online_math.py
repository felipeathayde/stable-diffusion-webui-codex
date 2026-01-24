"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Online LoRA math selection (weight-merge vs activation-side).
Centralizes the meaning of the `--lora-online-math` flag used when `--lora-apply-mode=online`.

Symbols (top-level; keep in sync; no ghosts):
- `LoraOnlineMath` (enum): Supported online LoRA math modes (`weight_merge`, `activation`).
- `DEFAULT_LORA_ONLINE_MATH` (constant): Default online math mode when unset.
- `parse_lora_online_math` (function): Parses a string into `LoraOnlineMath` (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from enum import Enum


class LoraOnlineMath(Enum):
    """How online LoRA patches are applied at runtime."""

    WEIGHT_MERGE = "weight_merge"
    ACTIVATION = "activation"


DEFAULT_LORA_ONLINE_MATH = LoraOnlineMath.WEIGHT_MERGE


def parse_lora_online_math(raw: str) -> LoraOnlineMath:
    value = str(raw).strip().lower()
    for mode in LoraOnlineMath:
        if mode.value == value:
            return mode
    allowed = ", ".join(m.value for m in LoraOnlineMath)
    raise ValueError(f"--lora-online-math must be one of: {allowed}; got: {raw!r}")


__all__ = [
    "DEFAULT_LORA_ONLINE_MATH",
    "LoraOnlineMath",
    "parse_lora_online_math",
]

