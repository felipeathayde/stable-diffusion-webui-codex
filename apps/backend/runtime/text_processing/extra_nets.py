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
- `ParsedExtras` (dataclass): Parsed extras bundle (cleaned prompt, selected LoRAs, parsed controls dict).
- `parse_prompt_for_extras` (function): Parse a single prompt, resolving LoRAs via the registry and stripping known tags.
- `parse_prompts` (function): Parse a list of prompts, returning cleaned prompts and deduplicated LoRA selections.
- `parse_prompts_with_extras` (function): Parse prompts and return merged controls in addition to cleaned prompts + LoRAs.
- `__all__` (constant): Export list for extra-net parsing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re

from apps.backend.infra.registry.lora import list_loras
from apps.backend.runtime.adapters.lora.selections import LoraSelection


_TAG_RE = re.compile(r"<\s*(?P<kind>lora|ti|clip_skip|sampler|scheduler|merge|tm|width|height|w|h|cfg|steps|seed|denoise|tiling)\s*:\s*(?P<name>[^:>]+)(?::(?P<weight>[-+]?\d*\.?\d+))?\s*>", re.IGNORECASE)


@dataclass
class ParsedExtras:
    prompt: str
    loras: List[LoraSelection]
    controls: dict


def parse_prompt_for_extras(prompt: str) -> ParsedExtras:
    # Build a quick index {name_lower: path}
    idx = {e["name"].lower(): e["path"] for e in list_loras()}
    loras: List[LoraSelection] = []
    controls: dict = {}

    def _repl(m: re.Match) -> str:
        kind = m.group('kind').lower()
        name = (m.group('name') or '').strip()
        weight_s = (m.group('weight') or '').strip()
        if kind == 'lora' and name:
            path = idx.get(name.lower())
            if path:
                w = 1.0
                try:
                    if weight_s:
                        w = max(0.0, float(weight_s))
                except Exception:
                    w = 1.0
                loras.append(LoraSelection(path=path, weight=w, online=False))
            # remove the tag
            return ''
        if kind == 'ti' and name:
            # textual inversion by name — expand into weighted token; embeddings are loaded by name in the engine
            w = weight_s or '1.0'
            return f"({name}:{w})"
        # control tags: <clip_skip:N>, <sampler:NAME>, <scheduler:NAME>,
        # <width:N>/<height:N>, <cfg:x>, <steps:n>, <seed:n>, <denoise:x>
        try:
            if kind == 'clip_skip':
                n = int(float(weight_s or name))
                controls['clip_skip'] = max(1, n)
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

    # Also detect control tags with custom patterns
    control_re = re.compile(r"<\s*(clip_skip|sampler|scheduler|merge|tm|width|height|w|h|cfg|steps|seed|denoise|tiling)\s*:\s*([^:>]+)(?::([^:>]+))?\s*>", re.IGNORECASE)
    cleaned = control_re.sub(_repl, _TAG_RE.sub(_repl, prompt))
    # Collapse duplicated spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return ParsedExtras(prompt=cleaned, loras=loras, controls=controls)


def parse_prompts(prompts: List[str]) -> Tuple[List[str], List[LoraSelection]]:
    cleaned: List[str] = []
    acc: List[LoraSelection] = []
    for p in prompts:
        r = parse_prompt_for_extras(p)
        cleaned.append(r.prompt)
        acc.extend(r.loras)
    # Deduplicate by path while keeping last weight
    uniq: dict[str, LoraSelection] = {}
    for sel in acc:
        uniq[sel.path] = sel
    return cleaned, list(uniq.values())


def parse_prompts_with_extras(prompts: List[str]) -> Tuple[List[str], List[LoraSelection], dict]:
    cleaned, loras = parse_prompts(prompts)
    controls: dict = {}
    for p in prompts:
        r = parse_prompt_for_extras(p)
        controls.update(r.controls)
    return cleaned, loras, controls


__all__ = ["parse_prompts", "parse_prompt_for_extras", "parse_prompts_with_extras", "ParsedExtras"]
