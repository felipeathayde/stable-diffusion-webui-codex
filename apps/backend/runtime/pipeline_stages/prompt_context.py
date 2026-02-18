"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt parsing and prompt-derived overrides for pipeline processing objects.
Builds a `PromptContext` (cleaned prompts, negative prompts, LoRA tags, controls) and applies it onto a processing object.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_lora_overrides` (function): Normalize `override_settings.lora_path` into deterministic `LoraSelection` entries.
- `build_prompt_context` (function): Parse prompts/negative prompts and return a `PromptContext` (includes extra-nets tags).
- `apply_prompt_context` (function): Apply a `PromptContext` onto a processing object (prompts + clip-skip controls).
- `apply_dimension_overrides` (function): Apply prompt-derived width/height controls onto the processing object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from apps.backend.runtime.adapters.lora.selections import LoraSelection
from apps.backend.runtime.processing.datatypes import PromptContext
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras

logger = logging.getLogger(__name__)


def _resolve_lora_overrides(processing: Any) -> list[LoraSelection]:
    overrides = getattr(processing, "override_settings", None)
    if not isinstance(overrides, dict):
        return []

    raw_lora_path = overrides.get("lora_path")
    if raw_lora_path is None:
        return []

    resolved_paths: list[str] = []
    if isinstance(raw_lora_path, str):
        normalized = raw_lora_path.strip()
        if normalized:
            resolved_paths.append(Path(normalized).expanduser().resolve(strict=False).as_posix())
    elif isinstance(raw_lora_path, list):
        for index, value in enumerate(raw_lora_path):
            if not isinstance(value, str):
                raise ValueError(
                    "Invalid override_settings.lora_path: expected array of strings "
                    f"(entry {index} was {type(value).__name__})."
                )
            normalized = value.strip()
            if normalized:
                resolved_paths.append(Path(normalized).expanduser().resolve(strict=False).as_posix())
    else:
        raise ValueError(
            "Invalid override_settings.lora_path: expected string or array of strings, "
            f"got {type(raw_lora_path).__name__}."
        )

    return [LoraSelection(path=path, weight=1.0, online=False) for path in resolved_paths]


def build_prompt_context(processing: Any, prompts: Sequence[str]) -> PromptContext:
    """Parse prompts, negative prompts, and extra network descriptors."""
    cleaned_prompts, parsed_loras, prompt_controls = parse_prompts_with_extras(list(prompts))
    prompt_loras = list(parsed_loras)
    lora_overrides = _resolve_lora_overrides(processing)
    if lora_overrides:
        seen_paths = {
            str(getattr(selection, "path", "") or "")
            for selection in prompt_loras
            if str(getattr(selection, "path", "") or "")
        }
        for selection in lora_overrides:
            if selection.path in seen_paths:
                continue
            prompt_loras.append(selection)
            seen_paths.add(selection.path)

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
