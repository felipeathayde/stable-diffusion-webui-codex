from __future__ import annotations

"""Native options facade (Codex).

Backs options with a simple JSON file under `apps/settings_values.json`. This module
is the single point of truth for reading/writing option values from the backend.
"""

from typing import Any, List, Dict, Optional
from dataclasses import dataclass
import os
import json

from apps.backend.infra.config.repo_root import get_repo_root

from . import main as codex_main


_VALUES_PATH = str(get_repo_root() / 'apps' / 'settings_values.json')


def _load() -> Dict[str, Any]:
    try:
        if os.path.exists(_VALUES_PATH):
            with open(_VALUES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save(values: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_VALUES_PATH), exist_ok=True)
        with open(_VALUES_PATH, 'w', encoding='utf-8') as f:
            json.dump(values, f, indent=2)
    except Exception:
        pass


def get_value(key: str, default: Any = None) -> Any:
    return _load().get(key, default)


def set_values(payload: Dict[str, Any]) -> List[str]:
    if not isinstance(payload, dict):
        return []
    data = _load()
    updated: List[str] = []
    for k, v in payload.items():
        data[k] = v
        updated.append(k)
    _save(data)
    return updated


def get_selected_vae(default: str = "Automatic") -> str:
    return str(get_value('selected_vae', default))


def get_additional_modules() -> List[str]:
    return list(getattr(codex_main, "_SELECTIONS").additional_modules)


# Convenience keys used by the API
def get_mode(default: str = 'Normal') -> str:
    return str(get_value('codex_mode', default))


def get_engine(default: str = 'sd15') -> str:
    return str(get_value('codex_engine', default))


def get_current_checkpoint(default: str | None = None) -> str | None:
    return get_value('sd_model_checkpoint', default)


__all__ = [
    'get_value', 'set_values', 'get_snapshot',
    'get_selected_vae', 'get_additional_modules',
    'get_mode', 'get_engine', 'get_current_checkpoint',
]


@dataclass
class OptionsSnapshot:
    codex_mode: str = 'Normal'
    codex_engine: str = 'sd15'
    sd_model_checkpoint: Optional[str] = None
    codex_export_video: bool = False
    selected_vae: str = 'Automatic'
    codex_diffusion_device: Optional[str] = None
    codex_diffusion_dtype: Optional[str] = None
    codex_te_device: Optional[str] = None
    codex_te_dtype: Optional[str] = None
    codex_vae_device: Optional[str] = None
    codex_vae_dtype: Optional[str] = None
    codex_smart_offload: bool = False
    codex_smart_fallback: bool = False
    codex_smart_cache: bool = False
    codex_core_streaming: bool = False
    codex_wan22_use_spec_runtime: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            'codex_mode': self.codex_mode,
            'codex_engine': self.codex_engine,
            'sd_model_checkpoint': self.sd_model_checkpoint,
            'codex_export_video': self.codex_export_video,
            'selected_vae': self.selected_vae,
            'codex_diffusion_device': self.codex_diffusion_device,
            'codex_diffusion_dtype': self.codex_diffusion_dtype,
            'codex_te_device': self.codex_te_device,
            'codex_te_dtype': self.codex_te_dtype,
            'codex_vae_device': self.codex_vae_device,
            'codex_vae_dtype': self.codex_vae_dtype,
            'codex_smart_offload': self.codex_smart_offload,
            'codex_smart_fallback': self.codex_smart_fallback,
            'codex_smart_cache': self.codex_smart_cache,
            'codex_core_streaming': self.codex_core_streaming,
            'codex_wan22_use_spec_runtime': self.codex_wan22_use_spec_runtime,
        }


def get_snapshot() -> OptionsSnapshot:
    v = _load()
    snap = OptionsSnapshot(
        codex_mode=str(v.get('codex_mode', 'Normal')),
        codex_engine=str(v.get('codex_engine', 'sd15')),
        sd_model_checkpoint=v.get('sd_model_checkpoint'),
        codex_export_video=bool(v.get('codex_export_video', False)),
        selected_vae=str(v.get('selected_vae', 'Automatic')),
        codex_diffusion_device=v.get('codex_diffusion_device'),
        codex_diffusion_dtype=v.get('codex_diffusion_dtype'),
        codex_te_device=v.get('codex_te_device'),
        codex_te_dtype=v.get('codex_te_dtype'),
        codex_vae_device=v.get('codex_vae_device'),
        codex_vae_dtype=v.get('codex_vae_dtype'),
        codex_smart_offload=bool(v.get('codex_smart_offload', False)),
        codex_smart_fallback=bool(v.get('codex_smart_fallback', False)),
        codex_smart_cache=bool(v.get('codex_smart_cache', False)),
        codex_core_streaming=bool(v.get('codex_core_streaming', False)),
        codex_wan22_use_spec_runtime=bool(v.get('codex_wan22_use_spec_runtime', False)),
    )
    return snap
