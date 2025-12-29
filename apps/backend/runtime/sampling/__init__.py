"""Native sampling primitives for Codex engines.

This package is intentionally import-light so helpers like
`apps.backend.runtime.sampling.catalog` can be used by the API/UI without
pulling torch-heavy modules at import time.

Torch-bound sampling internals live in `inner_loop.py` and are loaded only when
engines execute sampling (via `driver.py`).
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - import-time dispatch
    if name in {
        "sampling_function",
        "sampling_function_inner",
        "sampling_prepare",
        "sampling_cleanup",
    }:
        from . import inner_loop as _inner_loop

        value = getattr(_inner_loop, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "sampling_cleanup",
    "sampling_function",
    "sampling_function_inner",
    "sampling_prepare",
]
