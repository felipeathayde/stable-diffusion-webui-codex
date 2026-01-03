"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Optional RIFE frame interpolation integration hook.
Attempts to import an out-of-tree RIFE integration module and run interpolation when requested, raising `RIFEUnavailableError` on runtime failures.

Symbols (top-level; keep in sync; no ghosts):
- `RIFEUnavailableError` (class): Raised when a RIFE integration is present but fails at runtime.
- `maybe_interpolate_rife` (function): Attempts RIFE interpolation via dynamic import; returns `None` when unavailable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


class RIFEUnavailableError(RuntimeError):
    pass


def maybe_interpolate_rife(
    frames: Sequence[object], *, model: Optional[str], times: int, logger
) -> Optional[List[object]]:
    """Attempt to interpolate frames using a RIFE implementation if available.

    This is a pluggable hook. We do not vendor a RIFE implementation here.
    If a compatible module is installed in the environment and provides a
    simple `interpolate(frames, model_path, times, logger)` function, we will
    use it. Otherwise return None to indicate unavailability.
    """
    if times <= 1:
        return list(frames)
    # Try a well-known local integration point first
    try:
        # Expected optional integration module maintained out-of-tree by users
        # or extensions: `backend_ext.rife_vfi`
        import importlib

        mod = None
        for candidate in (
            "backend_ext.rife_vfi",
            "rife_vfi",
        ):
            try:
                mod = importlib.import_module(candidate)
                break
            except Exception:  # module not found
                mod = None
        if mod is None:
            return None
        if not hasattr(mod, "interpolate"):
            logger.warning("rife_vfi module found but has no 'interpolate' function")
            return None
        out = mod.interpolate(frames, model_path=model, times=int(times), logger=logger)
        # Expect a list of frames back
        if out is None:
            return None
        return list(out)
    except Exception as ex:  # noqa: BLE001
        # Distinguish missing module from runtime error
        msg = str(ex)
        if "No module named" in msg:
            return None
        raise RIFEUnavailableError(msg)
