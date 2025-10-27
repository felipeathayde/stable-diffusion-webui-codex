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

