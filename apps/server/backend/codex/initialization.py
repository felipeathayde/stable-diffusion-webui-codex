from __future__ import annotations

"""Codex initialization (wraps legacy modules_forge when present)."""


def initialize_codex() -> None:
    try:
        from modules_forge.initialization import initialize_forge as _init  # type: ignore
    except ModuleNotFoundError:
        # Optional in environments without Forge bits; no-op for strict codex‑only flows
        def _init() -> None:  # type: ignore
            return None
    _init()
