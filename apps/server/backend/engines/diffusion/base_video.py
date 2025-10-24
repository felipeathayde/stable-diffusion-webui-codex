from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional

from apps.server.backend.core.engine_interface import BaseInferenceEngine, EngineCapabilities, TaskType
from apps.server.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent


class BaseVideoEngine(BaseInferenceEngine):
    """Shared helpers for video engines (txt2vid/img2vid).

    Keeps engines simple and explicit while avoiding deep conditionals.
    """

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    # Subclasses should implement: load(), unload(), capabilities(), txt2vid(), img2vid().

    # ------------------------------
    # Utilities
    # ------------------------------
    def _to_json(self, obj: Any) -> str:
        def _default(o: Any) -> Any:
            try:
                return dict(o)
            except Exception:
                return str(o)

        try:
            return json.dumps(obj, default=_default, ensure_ascii=False)
        except Exception:
            # Best effort; never break response on serialization issues
            return str(obj)

    def _maybe_export_video(self, frames: List[object], *, fps: int, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Export frames to disk when requested.

        Minimal implementation: we acknowledge options and return a stub. A
        dedicated export utility can replace this later.
        """
        # Stub: acknowledge request but don't write files here.
        if options and options.get("save_output"):
            return {"saved": False, "reason": "exporter-not-wired", "fps": int(fps)}
        return None

