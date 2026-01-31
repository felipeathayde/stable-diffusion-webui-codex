"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared prompt wrapper helpers for engines.
Centralizes the common per-batch prompt metadata flags (negative marker + smart-cache override) so engines can extend the
wrapper with their own guidance/CfG fields without duplicating the shared attributes.

Symbols (top-level; keep in sync; no ghosts):
- `PromptListBase` (class): `list[str]` wrapper carrying `is_negative_prompt` and `smart_cache` flags.
"""

from __future__ import annotations

from typing import Iterable


class PromptListBase(list[str]):
    """List-like prompt wrapper used to carry common per-run metadata."""

    def __init__(
        self,
        items: Iterable[str],
        *,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        super().__init__(items)
        self.is_negative_prompt = bool(is_negative_prompt)
        self.smart_cache = smart_cache

