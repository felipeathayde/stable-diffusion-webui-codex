from __future__ import annotations

from typing import Any, Callable, Dict, List


def _choices_upscalers() -> List[str]:
    try:
        from modules import shared as _shared  # type: ignore
        return [x.name for x in getattr(_shared, 'sd_upscalers', [])]
    except Exception:
        return []


def _choices_sd_unet() -> List[str]:
    try:
        from modules import shared_items as _items  # type: ignore
        return list(_items.sd_unet_items())
    except Exception:
        return []


def _choices_cross_attention() -> List[str]:
    try:
        from modules import shared_items as _items  # type: ignore
        return list(_items.cross_attention_optimizations())
    except Exception:
        return ['Automatic']


def _choices_hypernetworks() -> List[str]:
    try:
        from modules import shared as _shared  # type: ignore
        base = ["None"]
        others = list(getattr(_shared, 'hypernetworks', []) or [])
        return base + others
    except Exception:
        return ["None"]


def _choices_localizations() -> List[str]:
    try:
        from modules import localization as _loc  # type: ignore
        return ["None"] + list(getattr(_loc, 'localizations', {}).keys())
    except Exception:
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
            out['fields'].append(nf)
        else:
            out['fields'].append(f)
    return out

