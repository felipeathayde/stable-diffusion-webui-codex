from __future__ import annotations

from typing import Any, Optional
import os


def _get_selected_accelerator() -> str:
    # Read from env to avoid legacy opts; default to none
    return os.getenv("CODEX_ACCELERATOR", "none")


def apply_to_diffusers_pipeline(pipe: Any, *, accelerator: Optional[str] = None, logger=None) -> str:
    """Apply a selected accelerator to a diffusers pipeline or raise with cause.

    - 'none'/'off' performs no acceleration.
    - 'tensorrt' requires the built-in accelerator to be available; otherwise an error is raised.
    """
    choice = (accelerator or _get_selected_accelerator()).lower().strip()
    if choice in ("", "none", "off"):
        return "none"

    if choice == "tensorrt":
        from ...accelerators.trt import TensorRTAccelerator  # type: ignore
        acc = TensorRTAccelerator()
        if not acc.is_available():
            raise RuntimeError("TensorRT accelerator requested but not available in this environment")
        try:
            acc.apply_to_diffusers(pipe)
        except Exception as ex:
            raise RuntimeError(f"Failed to apply TensorRT accelerator: {ex}") from ex
        if logger:
            logger.info("accelerator applied: TensorRT (built-in)")
        return "tensorrt"

    raise ValueError(f"Unsupported accelerator '{choice}'. Allowed: none, tensorrt")
