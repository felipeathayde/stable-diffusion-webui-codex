from __future__ import annotations

from typing import List, Tuple, Sequence, Optional, Dict, Any

from .rife import RIFEUnavailableError, maybe_interpolate_rife


def maybe_interpolate(
    frames: Sequence[object],
    *,
    enabled: bool,
    model: Optional[str],
    times: Optional[int],
    logger,
) -> Tuple[List[object], Dict[str, Any]]:
    """Attempt to interpolate frames using the requested method (RIFE for now).

    Returns a tuple (frames_out, meta) where meta contains at least:
    {"applied": bool, "method": str|None, "reason": str|None}
    """
    meta: Dict[str, Any] = {"applied": False, "method": None, "reason": None}
    if not enabled:
        meta["reason"] = "disabled"
        return list(frames), meta
    if not frames or len(frames) < 2:
        meta["reason"] = "not_enough_frames"
        return list(frames), meta
    if times is None or int(times) <= 1:
        meta["reason"] = "times_le_1"
        return list(frames), meta

    # RIFE is the only supported method; if unavailable or failing, raise.
    out = maybe_interpolate_rife(frames, model=model, times=int(times), logger=logger)
    if out is None:
        raise RuntimeError("video interpolation requested but RIFE is unavailable in this environment")
    meta.update({"applied": True, "method": "rife", "reason": None, "times": int(times), "model": model})
    return out, meta
