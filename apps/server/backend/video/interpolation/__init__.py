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

    # Try RIFE first
    try:
        out = maybe_interpolate_rife(frames, model=model, times=int(times), logger=logger)
        if out is not None:
            meta.update({"applied": True, "method": "rife", "reason": None, "times": int(times), "model": model})
            return out, meta
        meta.update({"reason": "rife_not_available"})
    except RIFEUnavailableError as ex:  # explicit unavailability
        meta.update({"reason": f"rife_unavailable: {ex}"})
    except Exception as ex:  # noqa: BLE001
        logger.warning("video interpolation (rife) failed: %s", ex)
        meta.update({"reason": "rife_exception"})

    # Optional blend fallback (disabled by default; enable via env if needed)
    import os
    if os.environ.get("CODEX_INTERPOLATION_ALLOW_BLEND", "0") == "1":
        try:
            from PIL import Image
            out_frames: List[object] = []
            T = int(times) if times is not None else 2
            for i in range(len(frames) - 1):
                a = frames[i]
                b = frames[i + 1]
                out_frames.append(a)
                for k in range(1, T):
                    alpha = k / float(T)
                    try:
                        out_frames.append(Image.blend(a, b, alpha))  # type: ignore[arg-type]
                    except Exception:
                        # If not PIL images, stop fallback politely
                        return list(frames), meta
            out_frames.append(frames[-1])
            meta.update({"applied": True, "method": "blend", "reason": None, "times": int(times)})
            return out_frames, meta
        except Exception:
            pass

    return list(frames), meta

