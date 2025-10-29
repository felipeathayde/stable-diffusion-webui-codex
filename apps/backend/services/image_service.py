from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import io
import base64
import json

from apps.backend.core.engine_interface import TaskType
from apps.backend.core.orchestrator import InferenceOrchestrator
from apps.backend.core.requests import ProgressEvent, ResultEvent, Txt2ImgRequest, Img2ImgRequest


def _encode_images(images: Iterable[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for img in images or []:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({"format": "png", "data": base64.b64encode(buf.getvalue()).decode("ascii")})
    return out


class ImageService:
    """Native image generation service over Codex engines (no legacy WebUI dependency)."""

    def __init__(self) -> None:
        self._orch = InferenceOrchestrator()

    def txt2img(self, req: Txt2ImgRequest, *, engine_key: str, model_ref: Optional[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for ev in self._orch.run(TaskType.TXT2IMG, engine_key, req, model_ref=model_ref):
            if isinstance(ev, ResultEvent):
                payload = ev.payload or {}
                info_raw = payload.get("info", "{}")
                try:
                    info_obj = json.loads(info_raw)
                except Exception:
                    info_obj = info_raw
                images = _encode_images(payload.get("images", []))
                result = {"images": images, "info": info_obj}
        return result

    def img2img(self, req: Img2ImgRequest, *, engine_key: str, model_ref: Optional[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for ev in self._orch.run(TaskType.IMG2IMG, engine_key, req, model_ref=model_ref):
            if isinstance(ev, ResultEvent):
                payload = ev.payload or {}
                info_raw = payload.get("info", "{}")
                try:
                    info_obj = json.loads(info_raw)
                except Exception:
                    info_obj = info_raw
                images = _encode_images(payload.get("images", []))
                result = {"images": images, "info": info_obj}
        return result
