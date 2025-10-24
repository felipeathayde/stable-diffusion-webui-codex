"""Legacy API shim for core orchestration primitives.

The canonical implementation now lives under ``apps.server.backend.core``.
"""

from apps.server.backend.core.engine_interface import (  # noqa: F401
    BaseInferenceEngine,
    EngineCapabilities,
    TaskType,
)
from apps.server.backend.core.requests import (  # noqa: F401
    Img2ImgRequest,
    Img2VidRequest,
    InferenceEvent,
    ProgressEvent,
    ResultEvent,
    Txt2ImgRequest,
    Txt2VidRequest,
)
from apps.server.backend.core.registry import (  # noqa: F401
    EngineDescriptor,
    EngineRegistry,
    registry,
)
from apps.server.backend.core.orchestrator import InferenceOrchestrator  # noqa: F401

__all__ = [
    "BaseInferenceEngine",
    "EngineCapabilities",
    "EngineDescriptor",
    "EngineRegistry",
    "Img2ImgRequest",
    "Img2VidRequest",
    "InferenceEvent",
    "InferenceOrchestrator",
    "ProgressEvent",
    "ResultEvent",
    "TaskType",
    "Txt2ImgRequest",
    "Txt2VidRequest",
    "registry",
]
