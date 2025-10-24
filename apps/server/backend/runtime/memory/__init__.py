"""Memory management utilities for backend runtimes."""

from .memory_management import *  # noqa: F401,F403
from . import memory_management as _memory_management_module
from . import stream

memory_management = _memory_management_module

__all__ = [name for name in globals() if not name.startswith("_")]
