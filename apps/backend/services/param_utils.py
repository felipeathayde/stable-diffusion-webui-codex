from __future__ import annotations

from typing import Any, List, Optional


def require(payload: dict, key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def as_list(payload: dict, key: str) -> List[Any]:
    v = payload.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def as_int(payload: dict, key: str, default: Optional[int] = None) -> int:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing int field: {key}")
        return int(default)
    return int(v)


def as_float(payload: dict, key: str, default: Optional[float] = None) -> float:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing float field: {key}")
        return float(default)
    return float(v)


def as_float_optional(payload: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = payload.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def as_bool(payload: dict, key: str, default: Optional[bool] = None) -> bool:
    v = payload.get(key)
    if v is None:
        if default is None:
            raise ValueError(f"missing bool field: {key}")
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")

