from __future__ import annotations

"""Options compatibility helpers (Codex naming over legacy opts).

Read values from modules.shared.opts using `codex_*` names first, then fall
back to legacy `forge_*` names. Write paths can be added later if needed.
"""

from typing import Any, List


def _opts():
    try:
        from modules import shared as _shared  # type: ignore

        return getattr(_shared, "opts", None)
    except Exception:
        return None


def get(name: str, default: Any = None) -> Any:
    opts = _opts()
    if opts is None:
        return default
    # Try exact name first
    if hasattr(opts, name):
        return getattr(opts, name)
    # Fallback forge_ prefix
    if name.startswith("codex_"):
        legacy = "forge_" + name[len("codex_") :]
        if hasattr(opts, legacy):
            return getattr(opts, legacy)
    return default


def get_selected_vae(default: str = "Automatic") -> str:
    return str(get("codex_selected_vae", default))


def get_additional_modules() -> List[str]:
    v = get("codex_additional_modules", None)
    if isinstance(v, list):
        return v
    # legacy name
    v2 = get("forge_additional_modules", None)  # type: ignore[arg-type]
    return list(v2 or [])

