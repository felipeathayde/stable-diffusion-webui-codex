"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native “extra networks” prompt parser (LoRA/TI/control tags).
Parses `<lora:...>` / `<ti:...>` tags and a small set of control tags (sampler/scheduler/width/height/cfg/steps/seed/denoise/tiling),
returning a cleaned prompt plus resolved LoRA selections and parsed controls.

Symbols (top-level; keep in sync; no ghosts):
- `_TAG_RE` (constant): Primary regex matching supported `<...:...>` tags.
- `_normalize_lora_alias` (function): Canonicalize LoRA token aliases for case/slash-insensitive lookup.
- `_build_lora_alias_index` (function): Build alias -> matching-paths index for LoRA resolution (stem/filename/path variants).
- `_resolve_lora_path` (function): Resolve one LoRA tag name with warning-on-miss/ambiguity semantics.
- `_dedupe_lora_selections` (function): De-duplicate LoRA selections by path with last-weight-wins semantics.
- `ParsedExtras` (dataclass): Parsed extras bundle (cleaned prompt, selected LoRAs, parsed controls dict).
- `parse_prompt_for_extras` (function): Parse a single prompt, resolving LoRAs via the registry and stripping known tags.
- `parse_prompts` (function): Parse a list of prompts, returning cleaned prompts and deduplicated LoRA selections.
- `parse_prompts_with_extras` (function): Parse prompts and return merged controls in addition to cleaned prompts + LoRAs.
- `__all__` (constant): Export list for extra-net parsing helpers.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.inventory.scanners.loras import iter_lora_files
from apps.backend.runtime.adapters.lora.selections import LoraSelection


_TAG_RE = re.compile(
    r"<\s*(?P<kind>lora|ti|clip_skip|sampler|scheduler|merge|tm|width|height|w|h|cfg|steps|seed|denoise|tiling)\s*:\s*(?P<name>[^:>]+)(?::(?P<weight>[^>]*))?\s*>",
    re.IGNORECASE,
)
_LOGGER = logging.getLogger(__name__)


def _normalize_lora_alias(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _build_lora_alias_index() -> dict[str, list[str]]:
    alias_index: dict[str, list[str]] = defaultdict(list)

    try:
        repo_root = get_repo_root().resolve(strict=False)
    except Exception:
        repo_root = None

    def _add(alias: str, path: str) -> None:
        key = _normalize_lora_alias(alias)
        if not key:
            return
        matches = alias_index[key]
        if path not in matches:
            matches.append(path)

    for full_path in iter_lora_files():
        resolved = Path(full_path).expanduser().resolve(strict=False)
        canonical_path = resolved.as_posix()
        filename = resolved.name
        stem = resolved.stem

        _add(stem, canonical_path)
        _add(filename, canonical_path)
        _add(canonical_path, canonical_path)

        if repo_root is not None:
            try:
                relative = resolved.relative_to(repo_root).as_posix()
            except Exception:
                relative = ""
            if relative:
                _add(relative, canonical_path)

    return dict(alias_index)


def _resolve_lora_path(
    *,
    alias_index: dict[str, list[str]],
    token_name: str,
    warned_tokens: set[tuple[str, str]],
) -> str | None:
    normalized = _normalize_lora_alias(token_name)
    if not normalized:
        return None

    matches = alias_index.get(normalized, [])
    if len(matches) == 1:
        return matches[0]

    if not matches:
        warn_key = ("missing", normalized)
        if warn_key not in warned_tokens:
            warned_tokens.add(warn_key)
            _LOGGER.warning("LoRA tag ignored: '%s' not found in discovered LoRA inventory.", token_name)
        return None

    warn_key = ("ambiguous", normalized)
    if warn_key not in warned_tokens:
        warned_tokens.add(warn_key)
        preview = ", ".join(os.path.basename(path) for path in matches[:3])
        if len(matches) > 3:
            preview = f"{preview}, ..."
        _LOGGER.warning(
            "LoRA tag ignored: '%s' matches multiple LoRAs (%d): %s",
            token_name,
            len(matches),
            preview,
        )
    return None


def _dedupe_lora_selections(selections: list[LoraSelection]) -> list[LoraSelection]:
    unique_by_path: dict[str, LoraSelection] = {}
    for selection in selections:
        unique_by_path[selection.path] = selection
    return list(unique_by_path.values())


@dataclass
class ParsedExtras:
    prompt: str
    loras: List[LoraSelection]
    controls: dict


def parse_prompt_for_extras(
    prompt: str,
    *,
    _alias_index: dict[str, list[str]] | None = None,
) -> ParsedExtras:
    alias_index = _alias_index if _alias_index is not None else _build_lora_alias_index()
    warned_tokens: set[tuple[str, str]] = set()
    loras: List[LoraSelection] = []
    controls: dict = {}

    def _repl(m: re.Match) -> str:
        kind = m.group('kind').lower()
        name = (m.group('name') or '').strip()
        raw_weight = m.group('weight')
        weight_s = (raw_weight or '').strip()
        if kind == 'lora' and name:
            path = _resolve_lora_path(
                alias_index=alias_index,
                token_name=name,
                warned_tokens=warned_tokens,
            )
            if path:
                w = 1.0
                if raw_weight is not None:
                    try:
                        if not weight_s:
                            raise ValueError("LoRA weight cannot be blank when explicitly provided")
                        if ":" in weight_s:
                            raise ValueError("unexpected extra LoRA weight segment")
                        parsed_weight = float(weight_s)
                        if not math.isfinite(parsed_weight):
                            raise ValueError("LoRA weight must be finite")
                        w = max(0.0, parsed_weight)
                    except Exception:
                        invalid_key = (
                            "invalid_weight",
                            _normalize_lora_alias(f"{name}:{weight_s}"),
                        )
                        if invalid_key not in warned_tokens:
                            warned_tokens.add(invalid_key)
                            _LOGGER.warning(
                                "LoRA tag ignored: '%s' has invalid weight '%s'.",
                                name,
                                weight_s,
                            )
                        return ''
                loras.append(LoraSelection(path=path, weight=w, online=False))
            # remove the tag
            return ''
        if kind == 'ti' and name:
            # textual inversion by name — expand into weighted token; embeddings are loaded by name in the engine
            if raw_weight is None:
                return f"({name}:1.0)"
            try:
                if not weight_s:
                    raise ValueError("TI weight cannot be blank when explicitly provided")
                parsed_ti_weight = float(weight_s)
                if not math.isfinite(parsed_ti_weight):
                    raise ValueError("TI weight must be finite")
                return f"({name}:{parsed_ti_weight})"
            except Exception:
                invalid_ti_key = ("invalid_ti_weight", _normalize_lora_alias(f"{name}:{weight_s}"))
                if invalid_ti_key not in warned_tokens:
                    warned_tokens.add(invalid_ti_key)
                    _LOGGER.warning(
                        "TI tag ignored: '%s' has invalid weight '%s'.",
                        name,
                        weight_s,
                    )
                return m.group(0)
        # control tags: <clip_skip:N>, <sampler:NAME>, <scheduler:NAME>,
        # <width:N>/<height:N>, <cfg:x>, <steps:n>, <seed:n>, <denoise:x>
        try:
            if kind == 'clip_skip':
                n = int(float(weight_s or name))
                controls['clip_skip'] = max(0, n)
            elif kind == 'sampler':
                controls['sampler'] = name.lower()
            elif kind == 'scheduler':
                controls['scheduler'] = name
            elif kind in ('merge', 'tm'):
                # Token merging is intentionally not supported in Codex (never default).
                # Strip the tag to avoid sending it into the text encoders.
                pass
            elif kind in ('width','w'):
                controls['width'] = max(8, int(float(weight_s or name)))
            elif kind in ('height','h'):
                controls['height'] = max(8, int(float(weight_s or name)))
            elif kind == 'cfg':
                controls['cfg'] = float(weight_s or name)
            elif kind == 'steps':
                controls['steps'] = max(1, int(float(weight_s or name)))
            elif kind == 'seed':
                controls['seed'] = int(float(weight_s or name))
            elif kind == 'denoise':
                controls['denoise'] = float(weight_s or name)
            elif kind == 'tiling':
                s = (name or '').strip().lower()
                controls['tiling'] = s in ('1','true','yes','on','enable','enabled')
        except Exception:
            pass
        return ''

    cleaned = _TAG_RE.sub(_repl, prompt)
    # Collapse duplicated spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return ParsedExtras(prompt=cleaned, loras=loras, controls=controls)


def parse_prompts(prompts: List[str]) -> Tuple[List[str], List[LoraSelection]]:
    cleaned, loras, _ = parse_prompts_with_extras(prompts)
    return cleaned, loras


def parse_prompts_with_extras(prompts: List[str]) -> Tuple[List[str], List[LoraSelection], dict]:
    alias_index = _build_lora_alias_index()
    cleaned: List[str] = []
    acc: List[LoraSelection] = []
    controls: dict = {}
    for p in prompts:
        r = parse_prompt_for_extras(p, _alias_index=alias_index)
        cleaned.append(r.prompt)
        acc.extend(r.loras)
        controls.update(r.controls)
    return cleaned, _dedupe_lora_selections(acc), controls


__all__ = ["parse_prompts", "parse_prompt_for_extras", "parse_prompts_with_extras", "ParsedExtras"]
