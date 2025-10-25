from __future__ import annotations

"""Options helpers (native Codex naming).

Pure native implementation; no lookup in legacy option stores.
"""

from typing import Any, List

from . import main as codex_main


def get(name: str, default: Any = None) -> Any:
    # Minimal native options getter; extend as needed.
    return default


def get_selected_vae(default: str = "Automatic") -> str:
    # No native VAE registry yet; return default
    return default


def get_additional_modules() -> List[str]:
    return list(getattr(codex_main, "_SELECTIONS").additional_modules)
