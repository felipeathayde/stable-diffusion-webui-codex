"""Operational helpers (bnb, gguf, swap) for backend runtime (lazy exports).

Do not import heavy submodules at package import time; expose attributes
on-demand to avoid circular imports (e.g., when `utils` imports `ops.operations_gguf`).
"""

_SUBMODULES = (
    "apps.backend.runtime.ops.operations",
    "apps.backend.runtime.ops.operations_bnb",
    "apps.backend.runtime.ops.operations_gguf",
)

_CACHE = {}


def __getattr__(name: str):  # pragma: no cover - import-time indirection
    if name in _CACHE:
        return _CACHE[name]
    import importlib
    for modpath in _SUBMODULES:
        try:
            mod = importlib.import_module(modpath)
            if hasattr(mod, name):
                obj = getattr(mod, name)
                _CACHE[name] = obj
                return obj
        except Exception:
            # Ignore import errors to allow other modules to satisfy the symbol
            continue
    raise AttributeError(name)


__all__ = []
