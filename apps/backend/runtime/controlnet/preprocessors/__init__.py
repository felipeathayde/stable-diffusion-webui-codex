"""ControlNet preprocessors."""

from .registry import ControlPreprocessorRegistry, default_registry, get_preprocessor
from .edges import register_edge_preprocessors
from .depth import register_depth_preprocessors

# Register built-in preprocessors when the package is imported.
register_edge_preprocessors(default_registry)
register_depth_preprocessors(default_registry)

__all__ = [
    "ControlPreprocessorRegistry",
    "default_registry",
    "get_preprocessor",
    "register_edge_preprocessors",
]
