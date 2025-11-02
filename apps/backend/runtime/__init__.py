"""Runtime-level helpers for backend execution (lazy export).

Avoid importing heavy modules (e.g., torch via `utils`) at package import time.
Modules are exposed lazily via ``__getattr__`` to break cycles and reduce early
memory pressure during tools like the TUI BIOS.
"""

_EXPORTS = {
    # Core utilities / small helpers
    "utils": "apps.backend.runtime.utils",
    "trace": "apps.backend.runtime.trace",
    "shared": "apps.backend.runtime.shared",
    # Memory stack
    "memory_management": "apps.backend.runtime.memory.memory_management",
    "stream": "apps.backend.runtime.memory.stream",
    # Heavier modules that may rely on memory
    "attention": "apps.backend.runtime.attention",
    "errors": "apps.backend.runtime.errors",
    "logging": "apps.backend.runtime.logging",
    "models": "apps.backend.runtime.models",
    "nn": "apps.backend.runtime.nn",
    "ops": "apps.backend.runtime.ops",
    "processing": "apps.backend.runtime.processing",
    "text_processing": "apps.backend.runtime.text_processing",
}


def __getattr__(name: str):  # pragma: no cover - import-time laziness
    modpath = _EXPORTS.get(name)
    if not modpath:
        raise AttributeError(name)
    import importlib
    module = importlib.import_module(modpath)
    return module


__all__ = list(_EXPORTS.keys())
