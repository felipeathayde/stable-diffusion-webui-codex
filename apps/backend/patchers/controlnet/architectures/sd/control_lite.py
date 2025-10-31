from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlLiteConfig:
    """Placeholder configuration for ControlNet-Lite variants."""

    message: str = "ControlNet Lite not yet ported"


class ControlNetLite:
    """ControlNet Lite placeholder raising explicit error until ported."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("ControlNet Lite not yet ported into Codex architecture package")
