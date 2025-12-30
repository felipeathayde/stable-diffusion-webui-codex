"""Runtime model helpers (lazy exports).

This package must stay dependency-light at import time: API factories and test
stubs may import it without a full torch install. Heavy submodules (loader,
safety, state_dict) should only be imported by consumers that need them.
"""

from __future__ import annotations

_EXPORTS = {
    "api": "apps.backend.runtime.models.api",
    "loader": "apps.backend.runtime.models.loader",
    "registry": "apps.backend.runtime.models.registry",
    "safety": "apps.backend.runtime.models.safety",
    "state_dict": "apps.backend.runtime.models.state_dict",
    "types": "apps.backend.runtime.models.types",
}


def __getattr__(name: str):  # pragma: no cover - import-time laziness
    modpath = _EXPORTS.get(name)
    if not modpath:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(modpath)
    return module


__all__ = list(_EXPORTS.keys())

