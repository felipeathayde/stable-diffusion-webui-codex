from __future__ import annotations

"""Codex model/runtime management (native).

These names are the forward‑looking API the backend should use. When needed,
we integrate with the legacy stack by modifying `modules.shared.opts` and
triggering reloads via `modules.sd_models`.
"""

from typing import Any, List

from dataclasses import dataclass, field


@dataclass
class _Selections:
    additional_modules: List[str] = field(default_factory=list)
    checkpoint_name: str | None = None


_SELECTIONS = _Selections()


def modules_change(spec: Any, *, save: bool, refresh: bool) -> bool:
    """Apply additional modules selection nativamente.

    For now we set `opts.forge_additional_modules` when present (legacy compat)
    and mirror into `opts.codex_additional_modules` for forward use.
    Returns True if the selection changed.
    """
    before = list(_SELECTIONS.additional_modules)
    # Normalize spec to list[str]
    if spec is None:
        after = []
    elif isinstance(spec, (list, tuple)):
        after = [str(x) for x in spec]
    else:
        after = [str(spec)]
    changed = before != after
    _SELECTIONS.additional_modules = after
    return changed


def checkpoint_change(name: str, *, save: bool, refresh: bool) -> bool:
    """Switch active checkpoint using modules API.

    Returns True if changed.
    """
    before = _SELECTIONS.checkpoint_name
    if before == name:
        return False
    _SELECTIONS.checkpoint_name = name
    return True


def refresh_model_loading_parameters() -> None:
    """Native stub: parameters are pulled from opts on reload; nothing to do."""
    return None
