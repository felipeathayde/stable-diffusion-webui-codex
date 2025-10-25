from __future__ import annotations

"""Codex model/runtime management (native).

These names are the forward‑looking API the backend should use. When needed,
we integrate with the legacy stack by modifying `modules.shared.opts` and
triggering reloads via `modules.sd_models`.
"""

from typing import Any


def modules_change(spec: Any, *, save: bool, refresh: bool) -> bool:
    """Apply additional modules selection nativamente.

    For now we set `opts.forge_additional_modules` when present (legacy compat)
    and mirror into `opts.codex_additional_modules` for forward use.
    Returns True if the selection changed.
    """
    try:
        from modules import shared as _shared  # type: ignore
    except Exception:
        return False
    opts = _shared.opts
    before = list(getattr(opts, "forge_additional_modules", []) or [])
    # Normalize spec to list[str]
    if spec is None:
        after = []
    elif isinstance(spec, (list, tuple)):
        after = [str(x) for x in spec]
    else:
        after = [str(spec)]
    changed = before != after
    setattr(opts, "forge_additional_modules", after)
    setattr(opts, "codex_additional_modules", after)
    return changed


def checkpoint_change(name: str, *, save: bool, refresh: bool) -> bool:
    """Switch active checkpoint using modules API.

    Returns True if changed.
    """
    try:
        from modules import shared as _shared  # type: ignore
    except Exception:
        return False
    before = getattr(_shared.opts, "sd_model_checkpoint", None)
    if before == name:
        return False
    # Apply atomically via opts API if available
    try:
        _shared.opts.set("sd_model_checkpoint", name, is_api=True)
    except Exception:
        setattr(_shared.opts, "sd_model_checkpoint", name)
    return True


def refresh_model_loading_parameters() -> None:
    """Native stub: parameters are pulled from opts on reload; nothing to do."""
    return None
