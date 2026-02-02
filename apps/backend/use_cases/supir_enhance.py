"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR enhance use-case (Option A).
Implements the canonical SUPIR pipeline:
decode upload → (optional preprocess) → sampling → decode/postprocess → return PIL image(s).

Symbols (top-level; keep in sync; no ghosts):
- `supir_enhance_pil_image` (function): Enhance one RGB PIL image via SUPIR (not yet ported).
"""

from __future__ import annotations

from typing import Any


def supir_enhance_pil_image(
    image,
    *,
    payload: dict[str, Any],
    base_model_path: str,
    variant_ckpt_path: str,
):
    raise NotImplementedError("SUPIR enhance not yet ported")
