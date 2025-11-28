# // tags: image-service, generation, outputs
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import io
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from apps.backend.core.engine_interface import TaskType
from apps.backend.core.orchestrator import InferenceOrchestrator
from apps.backend.core.requests import ProgressEvent, ResultEvent, Txt2ImgRequest, Img2ImgRequest


_LOGGER = logging.getLogger("backend.services.image_service")


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
    """Persist generated images under outputs/<mode>/<YYYY-MM-DD>/... for debugging.

    - Base directory defaults to `./outputs` relative to the working directory.
      Operators can override via CODEX_OUTPUT_ROOT to match an external layout.
    - This helper must never break the request flow; errors are logged and ignored.
    """

    try:
        images = list(images or [])
        if not images:
            return

        base = os.getenv("CODEX_OUTPUT_ROOT")
        # Fallback: workspace-root /output (single) to keep paths predictable
        root = Path(base) if base else Path.cwd() / "output"

        if task is TaskType.IMG2IMG:
            mode_dir = "img2img-images"
        else:
            mode_dir = "txt2img-images"

        date_dir = datetime.now().strftime("%Y-%m-%d")
        outdir = root / mode_dir / date_dir
        outdir.mkdir(parents=True, exist_ok=True)

        meta = info or {}
        engine = str(meta.get("engine", "") or "").strip() or "engine"
        sampler = str(meta.get("sampler", "") or "").strip() or "sampler"
        scheduler = str(meta.get("scheduler", "") or "").strip() or "scheduler"
        width = int(meta.get("width", 0) or 0)
        height = int(meta.get("height", 0) or 0)

        for idx, img in enumerate(images):
            try:
                stem_parts = [
                    engine,
                    f"{width}x{height}" if width and height else None,
                    sampler,
                    scheduler,
                    f"{idx:02d}",
                ]
                stem = "_".join(part for part in stem_parts if part)
                filename = f"{stem}.png" if stem else f"image_{idx:02d}.png"
                path = outdir / filename
                img.save(path, format="PNG")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Failed to save image to %s: %s", outdir, exc, exc_info=False)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Image disk persistence skipped due to error: %s", exc, exc_info=False)


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
