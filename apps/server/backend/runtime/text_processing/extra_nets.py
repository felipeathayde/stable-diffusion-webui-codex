from __future__ import annotations

"""Native extra-networks parser.

Currently supports LoRA tags embedded in prompts, e.g.:
  - <lora:name>
  - <lora:name:0.8>

Returns a cleaned prompt string and a list of LoRA selections with resolved
paths against the native LoRA registry. Unknown names are ignored explicitly.
"""

from dataclasses import dataclass
from typing import List, Tuple
import re

from apps.server.backend.registry.lora import list_loras
from apps.server.backend.codex.lora import LoraSelection


_TAG_RE = re.compile(r"<\s*(?P<kind>lora|ti|clip_skip|sampler|scheduler|merge|tm|width|height|w|h|cfg|steps|seed|denoise)\s*:\s*(?P<name>[^:>]+)(?::(?P<weight>[-+]?\d*\.?\d+))?\s*>", re.IGNORECASE)


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
        # <merge:ratio[:strategy]>, <width:N>/<height:N>, <cfg:x>, <steps:n>, <seed:n>, <denoise:x>
        try:
            if kind == 'clip_skip':
                n = int(float(weight_s or name))
                controls['clip_skip'] = max(1, n)
            elif kind == 'sampler':
                controls['sampler'] = name.lower()
            elif kind == 'scheduler':
                controls['scheduler'] = name
            elif kind in ('merge', 'tm'):
                # name: ratio or strategy; weight is ratio if present
                ratio = float(weight_s) if weight_s else float(name)
                controls['token_merge_ratio'] = max(0.0, min(0.95, ratio))
                # if both provided as <merge:strategy:ratio>
                parts = (name or '').split(':')
                if len(parts) >= 2:
                    controls['token_merge_strategy'] = parts[0].lower()
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
        except Exception:
            pass
        return ''

    # Also detect control tags with custom patterns
    control_re = re.compile(r"<\s*(clip_skip|sampler|scheduler|merge|tm)\s*:\s*([^:>]+)(?::([^:>]+))?\s*>", re.IGNORECASE)
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
