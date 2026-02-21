"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-side attention backend selection for diffusers pipelines.
Applies an attention backend (PyTorch SDPA / xFormers) based on explicit input or runtime memory config.

Symbols (top-level; keep in sync; no ghosts):
- `_get_selected_backend` (function): Reads effective attention backend from runtime memory config (fallback: persisted options).
- `_selected_sdpa_flags` (function): Reads effective SDPA enable flags (`flash`, `mem_efficient`) from runtime memory config.
- `apply_to_diffusers_pipeline` (function): Applies the chosen attention backend to a diffusers pipeline or raises with cause.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from apps.backend.runtime.memory import memory_management
from apps.backend.services import options_store


_LOGGER = logging.getLogger("backend.engines.util.attention")
_FLASH_DIFFUSERS_FALLBACK_LOGGED: set[str] = set()


def _get_selected_backend() -> str:
    try:
        return str(memory_management.manager.config.attention.backend.value)
    except Exception:
        # Compatibility fallback for legacy callsites that still rely on saved options.
        return str(options_store.get_value("codex_attention_backend", "pytorch") or "pytorch")


def _selected_sdpa_flags() -> tuple[bool, bool]:
    try:
        attention_cfg = memory_management.manager.config.attention
        return bool(attention_cfg.enable_flash), bool(attention_cfg.enable_mem_efficient)
    except Exception:
        return True, True


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
        enable_flash, enable_mem_efficient = _selected_sdpa_flags()
        enable_math = not (enable_flash or enable_mem_efficient)

        if enable_flash and not enable_mem_efficient:
            flash_unavailable_reason: str | None = None
            if not torch.cuda.is_available():
                flash_unavailable_reason = "cuda_unavailable"
            else:
                try:
                    major, _minor = torch.cuda.get_device_capability()
                    if major < 8:
                        flash_unavailable_reason = f"compute_capability_sm{major}x"
                except Exception:
                    flash_unavailable_reason = "capability_probe_failed"
            if flash_unavailable_reason is not None:
                enable_flash = False
                enable_mem_efficient = True
                enable_math = True
                if flash_unavailable_reason not in _FLASH_DIFFUSERS_FALLBACK_LOGGED:
                    _FLASH_DIFFUSERS_FALLBACK_LOGGED.add(flash_unavailable_reason)
                    (logger or _LOGGER).warning(
                        "[attention] requested SDPA flash for diffusers, but flash appears unavailable (%s); "
                        "falling back to mem_efficient/math.",
                        flash_unavailable_reason,
                    )

        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
        torch.backends.cuda.enable_math_sdp(enable_math)
        if logger:
            logger.info(
                "attention backend: pytorch (sdpa flash=%s mem_efficient=%s math=%s)",
                enable_flash,
                enable_mem_efficient,
                enable_math,
            )
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
