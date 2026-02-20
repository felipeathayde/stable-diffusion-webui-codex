"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-side attention backend selection for diffusers pipelines.
Applies an attention backend (PyTorch SDPA / xFormers) based on explicit input or persisted WebUI options.

Symbols (top-level; keep in sync; no ghosts):
- `_get_selected_backend` (function): Reads attention backend selection from WebUI options (defaults to `"pytorch"`).
- `apply_to_diffusers_pipeline` (function): Applies the chosen attention backend to a diffusers pipeline or raises with cause.
"""

from __future__ import annotations

from typing import Any, Optional
from apps.backend.services import options_store


def _get_selected_backend() -> str:
    # Settings are configured via WebUI options (apps/settings_values.json).
    return str(options_store.get_value("codex_attention_backend", "pytorch") or "pytorch")


def apply_to_diffusers_pipeline(pipe: Any, *, backend: Optional[str] = None, logger=None) -> str:
    """Apply the chosen attention backend to a diffusers pipeline (if supported).

    Returns the effective backend string applied or attempted.
    """
    choice = (backend or _get_selected_backend()).lower().strip()
    if choice not in ("pytorch", "xformers", "split", "quad"):
        raise ValueError(f"Invalid attention backend '{backend}'. Allowed: pytorch, xformers, split, quad")

    # Torch SDPA (Flash/Math/Mem) — default in PyTorch 2.x
    if choice == "pytorch":
        # If xformers was previously enabled, disable it when possible (failure is an error now)
        if hasattr(pipe, "disable_xformers_memory_efficient_attention"):
            pipe.disable_xformers_memory_efficient_attention()
        import torch  # type: ignore
        # Enable all SDPA modes; PyTorch will choose the best available
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        if logger:
            logger.info("attention backend: pytorch")
        return "pytorch"

    # xFormers memory-efficient attention
    if choice == "xformers":
        if not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            raise RuntimeError("Pipeline does not expose xformers enable hook")
        pipe.enable_xformers_memory_efficient_attention()
        if logger:
            logger.info("attention backend: xformers")
        return "xformers"

    if choice in {"split", "quad"}:
        raise NotImplementedError(
            "Attention backend "
            f"'{choice}' is not supported for diffusers pipelines. Use 'pytorch' or 'xformers'.",
        )

    # Should not reach here
    return choice
