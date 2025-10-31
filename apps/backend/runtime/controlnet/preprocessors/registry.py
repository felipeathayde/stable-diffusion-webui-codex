from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class PreprocessorResult:
    image: torch.Tensor
    metadata: Mapping[str, object]


Preprocessor = Callable[..., PreprocessorResult]


class ControlPreprocessorRegistry:
    """Registry of ControlNet preprocessors keyed by slug."""

    def __init__(self) -> None:
        self._registry: Dict[str, Preprocessor] = {}

    def register(self, name: str, fn: Preprocessor) -> None:
        if not callable(fn):
            raise TypeError("preprocessor must be callable")
        if name in self._registry:
            raise ValueError(f"preprocessor '{name}' already registered")
        self._registry[name] = fn

    def get(self, name: str) -> Preprocessor:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"preprocessor '{name}' not found") from exc

    def available(self) -> Mapping[str, Preprocessor]:
        return dict(self._registry)


default_registry = ControlPreprocessorRegistry()


def get_preprocessor(name: str, registry: Optional[ControlPreprocessorRegistry] = None) -> Preprocessor:
    if registry is None:
        registry = default_registry
    return registry.get(name)
