"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hydrate settings schema with dynamic choices and current values.
Injects conservative, backend-native `choices` for selected fields and attaches current option values from the backend options snapshot.

Symbols (top-level; keep in sync; no ghosts):
- `_choices_upscalers` (function): Returns upscaler choices (currently minimal-safe; no native registry yet).
- `_choices_sd_unet` (function): Returns UNet variant choices (currently none).
- `_choices_cross_attention` (function): Returns cross-attention optimization choices (currently only `Automatic`).
- `_choices_hypernetworks` (function): Returns hypernetwork choices (currently `None` only).
- `_choices_localizations` (function): Returns localization choices (currently `None` only).
- `HYDRATORS` (constant): Mapping from field key → hydrator function producing a `choices` list.
- `hydrate_schema` (function): Returns a hydrated schema dict with dynamic `choices` and `current` values when available.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List
from apps.backend.codex.options import get_snapshot as _opts_snapshot  # type: ignore


# Hydration choices are native and conservative. If richer choices are needed,
# add native providers under apps/backend and import them here.


def _choices_upscalers() -> List[str]:
    # No native upscaler registry yet; return a minimal-safe set
    return ["None"]


def _choices_sd_unet() -> List[str]:
    # No native UNet variants exposed to UI; empty (UI should hide when empty)
    return []


def _choices_cross_attention() -> List[str]:
    # Keep a single default
    return ["Automatic"]


def _choices_hypernetworks() -> List[str]:
    return ["None"]


def _choices_localizations() -> List[str]:
    return ["None"]


HYDRATORS: Dict[str, Callable[[], List[str]]] = {
    'upscaler_for_img2img': _choices_upscalers,
    'sd_unet': _choices_sd_unet,
    'cross_attention_optimization': _choices_cross_attention,
    'sd_hypernetwork': _choices_hypernetworks,
    'localization': _choices_localizations,
    # Multi-select candidates intentionally omitted (UI not implemented yet):
    # 'ui_tab_order', 'hidden_tabs', 'tabs_without_quick_settings_bar', 'ui_reorder_list',
    # 'quick_setting_list', 'infotext_skip_pasting', 'postprocessing_*'
}


def hydrate_schema(base: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of base schema with dynamic choices hydrated for known fields."""
    if not isinstance(base, dict):
        return base
    out = {
        'categories': list(base.get('categories', [])),
        'sections': list(base.get('sections', [])),
        'fields': [],
        'version': base.get('version', 1),
        'source': base.get('source', 'hydrated'),
    }
    snap = _opts_snapshot().as_dict()
    for f in base.get('fields', []):
        fk = f.get('key') if isinstance(f, dict) else None
        if fk in HYDRATORS:
            try:
                choices = HYDRATORS[fk]() or []
            except Exception:
                choices = []
            nf = dict(f)
            nf['choices'] = choices
            # ensure default in choices when possible
            if choices and nf.get('default') not in choices:
                # pick first choice rather than invalid default
                nf['default'] = choices[0]
            # current value from options snapshot when available
            if fk in snap:
                nf['current'] = snap[fk]
            out['fields'].append(nf)
        else:
            nf = dict(f)
            if fk in snap:
                nf['current'] = snap[fk]
            out['fields'].append(nf)
    return out
