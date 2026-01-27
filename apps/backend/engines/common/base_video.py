"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for video engines (txt2vid/img2vid).
Provides a small `BaseInferenceEngine` wrapper with common event/serialization helpers and ffmpeg export wiring for video-oriented engines.

Symbols (top-level; keep in sync; no ghosts):
- `BaseVideoEngine` (class): Base class for video engines with shared JSON serialization and optional export wiring helpers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from apps.backend.core.engine_interface import BaseInferenceEngine


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

    def _maybe_export_video(
        self,
        frames: List[object],
        *,
        fps: int,
        options: Optional[Dict[str, Any]] = None,
        task: str = "video",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export frames to disk when requested.

        Uses the ffmpeg exporter to write a video file under `CODEX_ROOT/output`
        when `save_output` is enabled.
        """
        from apps.backend.video.export.ffmpeg_exporter import VideoExportError, export_video

        opts: Dict[str, Any] = dict(options or {})
        if not bool(opts.get("save_output", False)):
            return None

        frames_list = list(frames or [])
        if not frames_list:
            return {"saved": False, "reason": "no-frames", "fps": int(fps), "frames": 0}

        try:
            result = export_video(
                frames_list,
                fps=int(fps),
                options=opts,
                task=str(task or "video"),
                extra_metadata=extra_metadata,
            )
        except VideoExportError as exc:
            self._logger.warning("[video-export] failed: %s", exc)
            return {"saved": False, "reason": str(exc), "fps": int(fps), "frames": len(frames_list)}
        except Exception as exc:
            self._logger.warning("[video-export] failed: %s", exc, exc_info=True)
            return {"saved": False, "reason": str(exc), "fps": int(fps), "frames": len(frames_list)}

        if result is None:
            return None

        return {
            "saved": bool(getattr(result, "saved", False)),
            "rel_path": getattr(result, "rel_path", None),
            "mime": getattr(result, "mime", None),
            "reason": getattr(result, "reason", None),
            "fps": getattr(result, "fps", int(fps)),
            "frames": getattr(result, "frame_count", len(frames_list)),
        }
