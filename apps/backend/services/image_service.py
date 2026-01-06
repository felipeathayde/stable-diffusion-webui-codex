# // tags: image-service, generation, outputs
"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image generation service wrapper for the API layer.
Runs `InferenceOrchestrator` for txt2img/img2img and returns base64-encoded PNG images plus the engine `info` payload, persisting samples via `save_generated_images(...)`.

Symbols (top-level; keep in sync; no ghosts):
- `_encode_images` (function): Encodes PIL images to `{format,data}` base64 PNG payloads.
- `_save_images_to_disk` (function): Persists generated images to disk via `save_generated_images`.
- `ImageService` (class): Orchestrates txt2img/img2img requests via `InferenceOrchestrator`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import io
import base64
import json

from apps.backend.core.engine_interface import TaskType
from apps.backend.core.orchestrator import InferenceOrchestrator
from apps.backend.core.requests import ResultEvent, Txt2ImgRequest, Img2ImgRequest
from apps.backend.services.output_service import save_generated_images


def _encode_images(images: Iterable[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for img in images or []:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({"format": "png", "data": base64.b64encode(buf.getvalue()).decode("ascii")})
    return out


def _save_images_to_disk(
    images: Iterable[Any],
    *,
    task: TaskType,
    info: Dict[str, Any] | None = None,
) -> None:
    save_generated_images(images, task=task, info=(info or None))


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
                info_dict: Dict[str, Any] = info_obj if isinstance(info_obj, dict) else {}
                images_pil = payload.get("images", [])
                _save_images_to_disk(images_pil, task=TaskType.TXT2IMG, info=info_dict)
                images = _encode_images(images_pil)
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
                info_dict: Dict[str, Any] = info_obj if isinstance(info_obj, dict) else {}
                images_pil = payload.get("images", [])
                _save_images_to_disk(images_pil, task=TaskType.IMG2IMG, info=info_dict)
                images = _encode_images(images_pil)
                result = {"images": images, "info": info_obj}
        return result
