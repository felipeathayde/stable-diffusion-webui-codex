"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt parsing and prompt-derived overrides for workflow processing objects.
Builds a `PromptContext` (cleaned prompts, negative prompts, LoRA tags, controls) and applies it onto a processing object.

Symbols (top-level; keep in sync; no ghosts):
- `build_prompt_context` (function): Parse prompts/negative prompts and return a `PromptContext` (includes extra-nets tags).
- `apply_prompt_context` (function): Apply a `PromptContext` onto a processing object (prompts + clip-skip controls).
- `apply_dimension_overrides` (function): Apply prompt-derived width/height controls onto the processing object.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from apps.backend.runtime.processing.datatypes import PromptContext
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras

logger = logging.getLogger(__name__)


def build_prompt_context(processing: Any, prompts: Sequence[str]) -> PromptContext:
    """Parse prompts, negative prompts, and extra network descriptors."""
    cleaned_prompts, prompt_loras, prompt_controls = parse_prompts_with_extras(list(prompts))
    controls = dict(prompt_controls)
    if "clip_skip" not in controls:
        meta = getattr(processing, "metadata", None)
        if isinstance(meta, dict) and meta.get("clip_skip") is not None:
            raw_clip_skip = meta.get("clip_skip")
            try:
                clip_skip = int(raw_clip_skip)  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError("Invalid clip_skip in metadata: must be an integer") from exc
            if clip_skip < 0:
                raise ValueError("Invalid clip_skip in metadata: must be >= 0")
            controls["clip_skip"] = clip_skip

    negative_prompts = list(
        getattr(processing, "negative_prompts", [getattr(processing, "negative_prompt", "")])
    )
    return PromptContext(
        prompts=cleaned_prompts,
        negative_prompts=negative_prompts,
        loras=prompt_loras,
        controls=controls,
    )


def apply_prompt_context(processing: Any, context: PromptContext) -> None:
    """Mutate processing object with normalized prompt data."""
    processing.prompts = context.prompts
    processing.negative_prompts = context.negative_prompts
    processing.cfg_scale = getattr(processing, "guidance_scale", 7.0)

    if "clip_skip" in context.controls:
        try:
            clip_skip = int(context.controls["clip_skip"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid clip_skip: must be an integer") from exc
        if clip_skip < 0:
            raise ValueError("Invalid clip_skip: must be >= 0")
        model = getattr(processing, "sd_model", None)
        if model is not None and hasattr(model, "set_clip_skip"):
            model.set_clip_skip(clip_skip)


def apply_dimension_overrides(processing: Any, controls: Mapping[str, Any]) -> None:
    """Apply dimension overrides parsed from prompt tags."""
    if "width" in controls:
        width = int(controls["width"])
        if width % 8 != 0 or width < 8 or width > 8192:
            raise ValueError("Invalid <width>: must be multiple of 8 and in [8,8192]")
        processing.width = width
    if "height" in controls:
        height = int(controls["height"])
        if height % 8 != 0 or height < 8 or height > 8192:
            raise ValueError("Invalid <height>: must be multiple of 8 and in [8,8192]")
        processing.height = height
