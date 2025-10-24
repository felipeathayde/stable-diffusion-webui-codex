"""Runtime-level helpers for backend execution (memory, streams, shared state).

Import order matters to avoid circular imports:
- `memory.memory_management` depends on `runtime.utils`.
- Several runtime submodules (e.g., attention, ops, models) depend on
  `memory_management`.

To keep imports acyclic during package initialization we:
1) import `utils` first,
2) then import memory submodules,
3) finally import heavier modules that depend on memory.
"""

# 1) Core utilities first so `utils` is attached to the package
from . import utils, trace, shared  # lightweight, no facade recursion

# 2) Memory stack after utils
from .memory import memory_management, stream

# 3) Remainder modules that may rely on memory
from . import attention, logging, models, nn, ops, text_processing

__all__ = [
    "attention",
    "logging",
    "memory_management",
    "models",
    "nn",
    "ops",
    "shared",
    "stream",
    "text_processing",
    "trace",
    "utils",
]
