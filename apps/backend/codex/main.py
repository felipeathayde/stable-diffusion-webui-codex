"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Transitional model/module selection state for Codex bootstrap paths.
Stores additional module selections and checkpoint state for compatibility code paths that have not migrated to the native services/options stack.

Symbols (top-level; keep in sync; no ghosts):
- `_Selections` (dataclass): Global selection state (additional modules + checkpoint name).
- `_SELECTIONS` (constant): Singleton selection state instance.
- `modules_change` (function): Updates additional modules selection and returns whether it changed.
- `checkpoint_change` (function): Updates checkpoint selection and returns whether it changed.
- `refresh_model_loading_parameters` (function): Native stub for legacy reload hooks (no-op).
"""

from __future__ import annotations

from typing import Any, List

from dataclasses import dataclass, field


@dataclass
class _Selections:
    additional_modules: List[str] = field(default_factory=list)
    checkpoint_name: str | None = None


_SELECTIONS = _Selections()


def modules_change(spec: Any, *, save: bool, refresh: bool) -> bool:
    """Apply additional modules selection nativamente.

    Reserved for future hooks wiring `opts.codex_additional_modules`.
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
