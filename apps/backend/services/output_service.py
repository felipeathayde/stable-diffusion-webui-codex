from __future__ import annotations

# // tags: outputs, saving, images

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from apps.backend.core.engine_interface import TaskType
from apps.backend.infra.config.repo_root import get_repo_root

_LOGGER = logging.getLogger("backend.services.output_service")


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return int(raw, 10)
        except Exception:
            return None
    return None


def _seed_for_index(info: Mapping[str, Any] | None, idx: int) -> int | None:
    if not info:
        return None

    seeds = info.get("all_seeds")
    if isinstance(seeds, list) and idx < len(seeds):
        seed = _safe_int(seeds[idx])
        if seed is not None:
            return seed

    return _safe_int(info.get("seed"))


def _pnginfo_for_image(image: object, metadata: Mapping[str, str] | None) -> object | None:
    if metadata is None and not isinstance(getattr(image, "info", None), dict):
        return None

    try:
        from PIL import PngImagePlugin  # type: ignore

        use_metadata = False
        pnginfo = PngImagePlugin.PngInfo()

        def _add_text(key: object, value: object) -> None:
            nonlocal use_metadata
            if not isinstance(key, str) or not isinstance(value, str):
                return
            if not value:
                return
            pnginfo.add_text(key, value)
            use_metadata = True

        info_items = getattr(image, "info", None)
        if isinstance(info_items, dict):
            for key, value in info_items.items():
                _add_text(key, value)

        if metadata:
            for key, value in metadata.items():
                _add_text(key, value)

        return pnginfo if use_metadata else None
    except Exception:
        return None


def save_generated_images(
    images: Iterable[object],
    *,
    task: TaskType,
    info: Mapping[str, Any] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> list[Path]:
    """Save generated images to disk under `CODEX_ROOT/output/<task>/<YYYY-MM-DD>/`.

    Filenames default to `seed{seed}_{timestamp}.png` (best-effort seed extraction).
    This helper must never break inference: errors are logged and ignored.
    """

    saved: list[Path] = []
    try:
        images_list = [img for img in (images or []) if img is not None]
        if not images_list:
            return saved

        if task not in {TaskType.TXT2IMG, TaskType.IMG2IMG}:
            return saved

        root = get_repo_root() / "output"
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        outdir = root / task.value / date_dir
        outdir.mkdir(parents=True, exist_ok=True)

        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        for idx, img in enumerate(images_list):
            seed = _seed_for_index(info, idx)
            if seed is None:
                filename = f"{timestamp}_{idx:02d}.png"
            else:
                filename = f"seed{seed}_{timestamp}.png"
                path = outdir / filename
                if path.exists():
                    filename = f"seed{seed}_{timestamp}_{idx:02d}.png"

            path = outdir / filename
            try:
                pnginfo = _pnginfo_for_image(img, metadata)
                save_kwargs: dict[str, object] = {"format": "PNG"}
                if pnginfo is not None:
                    save_kwargs["pnginfo"] = pnginfo
                img.save(path, **save_kwargs)  # type: ignore[attr-defined]
                saved.append(path)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Failed to save image to %s: %s", outdir, exc, exc_info=False)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Image disk persistence skipped due to error: %s", exc, exc_info=False)
    return saved


__all__ = ["save_generated_images"]

