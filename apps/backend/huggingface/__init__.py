"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hugging Face asset helper facade for strict/offline execution.
Re-exports `ensure_repo_minimal_files(...)`, which validates local Hugging Face mirrors and ensures required config/tokenizer assets are present for supported engines.

Symbols (top-level; keep in sync; no ghosts):
- `ensure_repo_minimal_files` (function): Ensures the minimal HF file set exists under a local mirror (re-export).
- `__all__` (constant): Explicit export list for this facade.
"""

from .assets import ensure_repo_minimal_files

__all__ = ["ensure_repo_minimal_files"]
