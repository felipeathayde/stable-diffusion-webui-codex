"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: In-repo RIFE frame interpolation adapter with deterministic model/runtime contracts.
Loads a pinned RIFE runtime (ccvfi), resolves the model checkpoint from deterministic repo-local paths, and interpolates PIL frame sequences with
explicit fail-loud diagnostics when dependencies or assets are unavailable.

Symbols (top-level; keep in sync; no ghosts):
- `RIFEUnavailableError` (class): Raised when RIFE interpolation cannot run due to missing/invalid runtime dependencies or assets.
- `_should_auto_provision_default_model` (function): Returns whether runtime may auto-provision the default `rife47.pth` checkpoint for this request.
- `_release_runtime` (function): Best-effort runtime/tensor cleanup hook used after interpolation to avoid stale residency.
- `maybe_interpolate_rife` (function): Interpolates a frame sequence using the RIFE IFNet runtime and returns an expanded list of frames.
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from apps.backend.video.runtime_dependencies import (
    VIDEO_RUNTIME_RIFE_MODEL_RELATIVE,
    VideoDependencyResolutionError,
    ensure_rife_model_file,
    resolve_rife_model_path,
)


class RIFEUnavailableError(RuntimeError):
    pass


def _should_auto_provision_default_model(model: Optional[str]) -> bool:
    raw = str(model or "").strip()
    if raw:
        normalized = raw.replace("\\", "/").strip().lower()
        default_relative = str(VIDEO_RUNTIME_RIFE_MODEL_RELATIVE).replace("\\", "/").lower()
        return normalized in {"rife47.pth", default_relative}

    env_override = str(os.environ.get("CODEX_RIFE_MODEL_PATH") or "").strip()
    return not env_override


def _resolve_runtime_model(model: Optional[str]) -> Path:
    try:
        return resolve_rife_model_path(model)
    except VideoDependencyResolutionError as exc:
        if not _should_auto_provision_default_model(model):
            raise RIFEUnavailableError(str(exc)) from exc

        try:
            provisioned = ensure_rife_model_file()
            return resolve_rife_model_path(str(provisioned))
        except Exception as provision_exc:
            raise RIFEUnavailableError(
                f"{exc} Automatic provisioning attempt for default RIFE model failed: {provision_exc}"
            ) from provision_exc


def _as_rgb_pil(frame: object, *, index: int):
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RIFEUnavailableError(f"Pillow is required for RIFE interpolation: {exc}") from exc

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    raise RIFEUnavailableError(
        f"RIFE interpolation expects PIL.Image frames; frame {index} has type {type(frame)!r}."
    )


def _load_model(model_path: Path):
    try:
        import torch
        from ccvfi import AutoConfig, AutoModel, ConfigType
    except Exception as exc:
        raise RIFEUnavailableError(
            f"RIFE runtime dependencies are unavailable ({exc}). Re-run install-webui to provision them."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device.type == "cuda"

    try:
        base_cfg = AutoConfig.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)
        if hasattr(base_cfg, "model_copy"):
            cfg = base_cfg.model_copy(update={"path": model_path, "url": None})
        else:  # pragma: no cover - defensive for older pydantic objects
            cfg = base_cfg.copy(update={"path": model_path, "url": None})

        model = AutoModel.from_config(
            cfg,
            device=device,
            fp16=fp16,
            model_dir=str(model_path.parent),
        )
    except Exception as exc:
        raise RIFEUnavailableError(
            f"Failed to initialize RIFE model from '{model_path}': {exc}"
        ) from exc
    return model, device, fp16, torch


def _to_tensor(image, *, device, fp16: bool, torch_mod):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch_mod.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return tensor.half() if fp16 else tensor.float()


def _to_pil(tensor, *, torch_mod):
    from PIL import Image  # type: ignore

    pixels = (
        tensor.detach()
        .to(dtype=torch_mod.float32)
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    uint8 = np.rint(pixels * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def _release_runtime(runtime: Any | None, *, torch_mod: Any | None, logger) -> None:
    if runtime is None:
        return

    try:
        unload = getattr(runtime, "unload", None)
        if callable(unload):
            unload()
    except Exception as exc:  # pragma: no cover - defensive cleanup telemetry
        if logger is not None:
            logger.warning("RIFE runtime unload hook failed: %s", exc)

    try:
        del runtime
    except Exception:
        pass

    try:
        cuda_mod = getattr(torch_mod, "cuda", None)
        cuda_available = callable(getattr(cuda_mod, "is_available", None)) and bool(cuda_mod.is_available())
        if cuda_available and callable(getattr(cuda_mod, "empty_cache", None)):
            cuda_mod.empty_cache()
    except Exception as exc:  # pragma: no cover - defensive cleanup telemetry
        if logger is not None:
            logger.warning("RIFE CUDA cache cleanup failed: %s", exc)

    gc.collect()


def maybe_interpolate_rife(
    frames: Sequence[object], *, model: Optional[str], times: int, logger
) -> Optional[List[object]]:
    if times <= 1:
        return list(frames)
    if len(frames) < 2:
        return list(frames)

    model_path = _resolve_runtime_model(model)
    runtime: Any | None = None
    torch_mod: Any | None = None
    try:
        runtime, device, fp16, torch_mod = _load_model(model_path)

        pil_frames = [_as_rgb_pil(frame, index=index + 1) for index, frame in enumerate(frames)]
        first_size = pil_frames[0].size
        for idx, frame in enumerate(pil_frames[1:], start=2):
            if frame.size != first_size:
                raise RIFEUnavailableError(
                    f"RIFE interpolation requires same-size frames; frame 1 is {first_size}, frame {idx} is {frame.size}."
                )

        output: list[object] = [pil_frames[0]]
        with torch_mod.inference_mode():
            for left, right in zip(pil_frames[:-1], pil_frames[1:]):
                pair = torch_mod.stack(
                    [
                        _to_tensor(left, device=device, fp16=fp16, torch_mod=torch_mod),
                        _to_tensor(right, device=device, fp16=fp16, torch_mod=torch_mod),
                    ],
                    dim=1,
                )
                try:
                    for step in range(1, int(times)):
                        timestep = float(step) / float(times)
                        out_tensor = runtime.inference(pair, timestep=timestep, scale=1.0)
                        try:
                            output.append(_to_pil(out_tensor[0], torch_mod=torch_mod))
                        finally:
                            del out_tensor
                    output.append(right)
                finally:
                    del pair

        if logger is not None:
            logger.info(
                "RIFE interpolation applied (model=%s, times=%s, in_frames=%s, out_frames=%s)",
                str(model_path),
                int(times),
                len(frames),
                len(output),
            )
        return output
    finally:
        _release_runtime(runtime, torch_mod=torch_mod, logger=logger)
