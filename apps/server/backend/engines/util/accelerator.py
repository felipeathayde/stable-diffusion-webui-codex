from __future__ import annotations

from typing import Any, Optional
import os


def _get_selected_accelerator() -> str:
    # Read from env to avoid legacy opts; default to none
    return os.getenv("CODEX_ACCELERATOR", "none")


def apply_to_diffusers_pipeline(pipe: Any, *, accelerator: Optional[str] = None, logger=None) -> str:
    """Attempt to apply an accelerator to a diffusers pipeline (no-op if none).

    Supports 'tensorrt' via the pluggable accelerator; unrecognized values are ignored.
    Returns the effective accelerator string used (or 'none').
    """
    choice = (accelerator or _get_selected_accelerator()).lower().strip()
    if choice in ('', 'none', 'off'):
        return 'none'

    if choice == 'tensorrt':
        # Prefer external plugin if present
        try:
            import importlib
            mod = None
            for name in ("backend_ext.trt_accel", "trt_accel"):
                try:
                    mod = importlib.import_module(name)
                    break
                except Exception:
                    mod = None
            if mod and hasattr(mod, 'apply_to_diffusers'):
                mod.apply_to_diffusers(pipe)  # type: ignore[attr-defined]
                if logger:
                    logger.info("accelerator applied: TensorRT (external)")
                return 'tensorrt'
        except Exception as ex:
            if logger:
                logger.warning("TensorRT external accelerator failed: %s", ex)

        # Fallback to built-in stub (will raise if truly unavailable)
        try:
            from ...accelerators.trt import TensorRTAccelerator  # type: ignore
            acc = TensorRTAccelerator()
            if not acc.is_available():
                if logger:
                    logger.warning("TensorRT not available in environment; skipping")
                return 'none'
            acc.apply_to_diffusers(pipe)
            if logger:
                logger.info("accelerator applied: TensorRT (built-in)")
            return 'tensorrt'
        except Exception as ex:
            if logger:
                logger.warning("TensorRT accelerator apply failed: %s", ex)
        return 'none'

    # Unknown choice
    return 'none'
