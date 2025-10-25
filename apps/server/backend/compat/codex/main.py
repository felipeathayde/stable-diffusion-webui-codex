from __future__ import annotations

"""Codex wrappers over modules_forge.main_entry.

These names are the forward‑looking API the backend should use.
"""

from typing import Any


def modules_change(spec: Any, *, save: bool, refresh: bool) -> bool:
    from modules_forge import main_entry as _main  # type: ignore

    return _main.modules_change(spec, save=save, refresh=refresh)


def checkpoint_change(name: str, *, save: bool, refresh: bool) -> bool:
    from modules_forge import main_entry as _main  # type: ignore

    return _main.checkpoint_change(name, save=save, refresh=refresh)


def refresh_model_loading_parameters() -> None:
    from modules_forge import main_entry as _main  # type: ignore

    _main.refresh_model_loading_parameters()

