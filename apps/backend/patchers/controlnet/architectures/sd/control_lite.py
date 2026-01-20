"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Placeholder stubs for ControlNet-Lite variants (not yet ported).
Raises explicit errors to avoid silent fallbacks while the architecture is being ported into Codex.

Symbols (top-level; keep in sync; no ghosts):
- `ControlLiteConfig` (dataclass): Placeholder configuration for ControlNet-Lite variants.
- `ControlNetLite` (class): Placeholder class raising `NotImplementedError` on construction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlLiteConfig:
    """Placeholder configuration for ControlNet-Lite variants."""

    message: str = "ControlNet Lite not yet ported"


class ControlNetLite:
    """ControlNet Lite placeholder raising explicit error until ported."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("ControlNet Lite not yet ported into Codex architecture package")
