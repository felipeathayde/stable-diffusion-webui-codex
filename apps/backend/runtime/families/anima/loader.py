"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Strict loader utilities for the Anima core transformer (`MiniTrainDiT` + `LLMAdapter`).
This module is intentionally fail-loud: any missing/unexpected weights are fatal and reported with actionable samples.

Symbols (top-level; keep in sync; no ghosts):
- `load_anima_dit_from_state_dict` (function): Instantiate + strict-load `AnimaDiT` from a transformer state dict.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch

from apps.backend.runtime.models.state_dict import safe_load_state_dict

from .config import AnimaConfig, infer_anima_config_from_state_dict
from .model import AnimaDiT


def load_anima_dit_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AnimaDiT:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping; got {type(state_dict).__name__}")

    # Parser guarantees `net.` prefix stripping; fail loud if a raw checkpoint dict is passed by mistake.
    if any(str(k).startswith("net.") for k in state_dict.keys()):
        raise RuntimeError(
            "Anima loader received a state_dict with a 'net.' prefix. "
            "Expected the parser-stripped transformer component (prefix removed)."
        )

    try:
        config: AnimaConfig = infer_anima_config_from_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 - surfaced as a load-time error with context
        raise RuntimeError(f"Anima config inference failed: {exc}") from exc
    model = AnimaDiT(config=config, device=device, dtype=dtype).eval()

    missing, unexpected = safe_load_state_dict(model, state_dict, log_name="anima.transformer")
    if missing or unexpected:
        sample_missing = ", ".join(missing[:10])
        sample_unexpected = ", ".join(unexpected[:10])
        raise RuntimeError(
            "Anima core transformer strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample=[{sample_missing}] unexpected_sample=[{sample_unexpected}]"
        )

    return model
