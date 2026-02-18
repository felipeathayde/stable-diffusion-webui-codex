"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Upscaler registry + orchestration.
Discovers upscaler model files from configured roots, exposes a stable list for the UI, and provides a single entry point to run
either latent upscaling (torch interpolate) or Spandrel SR models (tiled).
Discovery respects the safeweights policy: when `CODEX_SAFE_WEIGHTS=1`, only `.safetensors` upscaler weights are surfaced (defense in depth at load-time).

Symbols (top-level; keep in sync; no ghosts):
- `list_upscalers` (function): Returns all available upscalers (builtin latent + local Spandrel models).
- `invalidate_upscalers_cache` (function): Clears cached upscaler discovery results (next call re-scans roots).
- `resolve_spandrel_path` (function): Maps a spandrel upscaler id to a local file path.
- `upscale_image_tensor` (function): Upscales a pixel-space tensor using the selected upscaler id.
- `upscale_latent_tensor` (function): Upscales a latent tensor using a latent upscaler id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from apps.backend.infra.config.paths import get_paths_for

from .errors import UpscalerNotFoundError
from .safeweights import allowed_upscaler_weight_suffixes, safeweights_enabled
from .specs import LATENT_UPSCALE_MODES, TileConfig, UpscalerDefinition, UpscalerKind


_UPSCALERS_CACHE: list[UpscalerDefinition] | None = None
_UPSCALERS_CACHE_KEY: tuple[tuple[str, str, int, float | None], ...] | None = None


def _iter_model_files(root: str) -> Iterable[Path]:
    base = Path(str(root))
    if not base.exists():
        return []
    if not base.is_dir():
        return []
    allowed_exts = set(allowed_upscaler_weight_suffixes())
    out: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        out.append(path)
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def _make_spandrel_id(*, root_key: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"spandrel:{root_key}:{rel}"


def invalidate_upscalers_cache() -> None:
    global _UPSCALERS_CACHE, _UPSCALERS_CACHE_KEY
    _UPSCALERS_CACHE = None
    _UPSCALERS_CACHE_KEY = None


def _directory_tree_signature(path: Path) -> tuple[int, float | None]:
    if not path.is_dir():
        return 0, None
    dir_count = 0
    newest_mtime: float | None = None
    stack: list[Path] = [path]
    while stack:
        current = stack.pop()
        if not current.is_dir():
            continue
        dir_count += 1
        try:
            current_mtime = float(current.stat().st_mtime)
        except Exception:
            current_mtime = None
        if current_mtime is None:
            pass
        elif newest_mtime is None or current_mtime > newest_mtime:
            newest_mtime = current_mtime
        try:
            for child in current.iterdir():
                if child.is_dir():
                    stack.append(child)
        except Exception:
            continue
    return dir_count, newest_mtime


def _compute_upscalers_cache_key() -> tuple[tuple[str, str, int, float | None], ...]:
    parts: list[tuple[str, str, int, float | None]] = []
    parts.append(("policy", f"safeweights={int(safeweights_enabled())}", 0, None))
    for root_key in ("upscale_models", "latent_upscale_models"):
        for root in get_paths_for(root_key):
            base = Path(str(root))
            count, newest_mtime = _directory_tree_signature(base)
            parts.append((root_key, base.as_posix(), count, newest_mtime))
    return tuple(parts)


def _validate_tile_config(tile: TileConfig) -> None:
    tile_size = int(getattr(tile, "tile", 0) or 0)
    overlap = int(getattr(tile, "overlap", 0) or 0)
    min_tile = int(getattr(tile, "min_tile", 0) or 0)
    fallback = bool(getattr(tile, "fallback_on_oom", True))

    if tile_size < 0:
        raise ValueError("tile.tile must be >= 0")
    if overlap < 0:
        raise ValueError("tile.overlap must be >= 0")
    if min_tile <= 0:
        raise ValueError("tile.min_tile must be > 0")
    if tile_size > 0 and overlap >= tile_size:
        raise ValueError("tile.overlap must be < tile.tile")
    if fallback and tile_size > 0 and min_tile > tile_size:
        raise ValueError("tile.min_tile must be <= tile.tile when fallback_on_oom is enabled")


def list_upscalers() -> list[UpscalerDefinition]:
    global _UPSCALERS_CACHE, _UPSCALERS_CACHE_KEY

    cache_key = _compute_upscalers_cache_key()
    if _UPSCALERS_CACHE is not None and _UPSCALERS_CACHE_KEY == cache_key:
        return list(_UPSCALERS_CACHE)

    upscalers: list[UpscalerDefinition] = []
    seen_ids: set[str] = set()

    # Built-in latent modes (hires-fix).
    for uid, mode in LATENT_UPSCALE_MODES.items():
        if uid in seen_ids:
            continue
        seen_ids.add(uid)
        label = uid.split(":", 1)[1].replace("-", " ")
        upscalers.append(
            UpscalerDefinition(
                id=uid,
                label=f"Latent ({label})",
                kind=UpscalerKind.LATENT,
                meta={"mode": mode.mode, "antialias": mode.antialias},
            )
        )

    # Local Spandrel models.
    for root_key in ("upscale_models", "latent_upscale_models"):
        for root in get_paths_for(root_key):
            base = Path(str(root))
            for file_path in _iter_model_files(str(base)):
                try:
                    rel = file_path.relative_to(base).as_posix()
                except Exception:
                    rel = file_path.name
                uid = _make_spandrel_id(root_key=root_key, rel_path=rel)
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                upscalers.append(
                    UpscalerDefinition(
                        id=uid,
                        label=f"{file_path.stem}",
                        kind=UpscalerKind.SPANDREL,
                        meta={"root": root_key, "rel_path": rel},
                    )
                )

    # Stable ordering: latent first, then spandrel by label.
    upscalers.sort(key=lambda u: (0 if u.kind == UpscalerKind.LATENT else 1, u.label.lower(), u.id))
    _UPSCALERS_CACHE = list(upscalers)
    _UPSCALERS_CACHE_KEY = cache_key
    return list(upscalers)


def resolve_spandrel_path(upscaler_id: str) -> Path:
    raw = str(upscaler_id or "")
    if not raw.startswith("spandrel:"):
        raise UpscalerNotFoundError(f"Not a spandrel upscaler id: {upscaler_id!r}")
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise UpscalerNotFoundError(f"Invalid spandrel upscaler id: {upscaler_id!r}")
    _, root_key, rel = parts
    rel_path = rel.replace("\\", "/").lstrip("/")
    roots = get_paths_for(root_key)
    if not roots:
        raise UpscalerNotFoundError(
            f"Upscaler root '{root_key}' has no configured paths in apps/paths.json (id={upscaler_id!r})."
        )

    for root in roots:
        base = Path(str(root))
        candidate = (base / rel_path)
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(Path(str(r)) / rel_path) for r in roots)
    raise UpscalerNotFoundError(f"Upscaler file not found for id: {upscaler_id!r}. Tried: {tried}")


def upscale_image_tensor(
    image_bchw_01: "torch.Tensor",
    *,
    upscaler_id: str,
    target_width: int,
    target_height: int,
    tile: TileConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> "torch.Tensor":
    """Upscale a pixel-space tensor (range [0..1]) to target size."""

    import torch
    from apps.backend.runtime.memory import memory_management
    from apps.backend.runtime.memory.config import DeviceRole
    from apps.backend.runtime.misc.image_resize import lanczos as pixel_lanczos

    if image_bchw_01.ndim != 4 or image_bchw_01.shape[1] != 3:
        raise ValueError(f"image must be BCHW RGB; got shape={tuple(image_bchw_01.shape)}")

    uid = str(upscaler_id or "").strip()
    if not uid:
        raise UpscalerNotFoundError("Missing upscaler id")

    # Built-in latent ids are not valid for pixel upscaling.
    if uid in LATENT_UPSCALE_MODES:
        raise UpscalerNotFoundError(f"Latent upscaler '{uid}' cannot be used for pixel upscaling.")

    if uid.startswith("spandrel:"):
        from .spandrel_backend import load_spandrel_model, run_spandrel_upscale

        _validate_tile_config(tile)

        path = resolve_spandrel_path(uid)
        handle = load_spandrel_model(path)
        device = memory_management.manager.get_device(DeviceRole.CORE)
        core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
        dtype = core_dtype if getattr(handle.descriptor, "supports_half", True) else torch.float32
        with torch.inference_mode():
            y = run_spandrel_upscale(
                handle,
                image_bchw_01,
                tile=tile,
                device=device,
                dtype=dtype,
                progress_callback=progress_callback,
            )
            # Ensure exact target size (A1111/Forge behavior: model upscale then Lanczos to exact dims).
            if int(y.shape[2]) != int(target_height) or int(y.shape[3]) != int(target_width):
                y = pixel_lanczos(y.to(dtype=torch.float32), int(target_width), int(target_height))
            return y.clamp(0.0, 1.0)

    raise UpscalerNotFoundError(f"Unknown upscaler id: {uid!r}")


def upscale_latent_tensor(
    latents: "torch.Tensor",
    *,
    upscaler_id: str,
    target_width: int,
    target_height: int,
) -> "torch.Tensor":
    """Upscale a latent tensor (BCHW) to target spatial size using torch interpolate."""

    import torch

    uid = str(upscaler_id or "").strip()
    mode = LATENT_UPSCALE_MODES.get(uid)
    if mode is None:
        raise UpscalerNotFoundError(f"Unknown latent upscaler id: {uid!r}")

    with torch.inference_mode():
        return torch.nn.functional.interpolate(
            latents,
            size=(int(target_height), int(target_width)),
            mode=mode.mode,
            antialias=bool(mode.antialias),
        )


__all__ = [
    "list_upscalers",
    "invalidate_upscalers_cache",
    "resolve_spandrel_path",
    "upscale_image_tensor",
    "upscale_latent_tensor",
]
