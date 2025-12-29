"""Memory utilities for backend runtimes.

This package is intentionally import-light: modules like
`apps.backend.infra.config.args` import `apps.backend.runtime.memory.config`, and
we must not pull torch-heavy modules (or create circular imports) just by
importing the package.

Torch-bound runtime management lives in `memory_management.py` and is loaded on
demand.
"""

from __future__ import annotations

import importlib


def __getattr__(name: str):  # pragma: no cover - import-time dispatch
    if name == "memory_management":
        _mm = importlib.import_module(f"{__name__}.memory_management")
        globals()[name] = _mm
        return _mm
    if name == "stream":
        _stream = importlib.import_module(f"{__name__}.stream")
        globals()[name] = _stream
        return _stream
    raise AttributeError(name)


__all__ = ["memory_management", "stream"]
