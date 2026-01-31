"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Package marker for API task orchestration helpers.
Routers should import from this package to keep endpoint modules small and consistent.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.interfaces.api.tasks` (module): Namespace for task orchestration helpers (e.g. generation task runners).
"""

from __future__ import annotations

