from __future__ import annotations

"""Options compatibility helpers (Codex naming over legacy opts).

Read values from modules.shared.opts using `codex_*` names first, then fall
back to legacy `forge_*` names. Write paths can be added later if needed.
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
