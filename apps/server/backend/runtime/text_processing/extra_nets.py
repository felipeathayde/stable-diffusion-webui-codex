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


_TAG_RE = re.compile(r"<\s*(?P<kind>lora|ti)\s*:\s*(?P<name>[^:>]+)(?::(?P<weight>[-+]?\d*\.?\d+))?\s*>", re.IGNORECASE)


@dataclass
class ParsedExtras:
    prompt: str
    loras: List[LoraSelection]


def parse_prompt_for_extras(prompt: str) -> ParsedExtras:
    # Build a quick index {name_lower: path}
    idx = {e["name"].lower(): e["path"] for e in list_loras()}
    loras: List[LoraSelection] = []

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
        return ''

    cleaned = _TAG_RE.sub(_repl, prompt)
    # Collapse duplicated spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return ParsedExtras(prompt=cleaned, loras=loras)


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


__all__ = ["parse_prompts", "parse_prompt_for_extras", "ParsedExtras"]
