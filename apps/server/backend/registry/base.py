from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List


@dataclass
class AssetEntry:
    name: str
    path: str
    kind: str
    tags: List[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _is_dir_with_any(root: str, subdirs: Iterable[str]) -> bool:
    try:
        for sd in subdirs:
            if os.path.isdir(os.path.join(root, sd)):
                return True
    except Exception:
        return False
    return False


def _iter_dirs(root: str) -> Iterable[str]:
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                yield full
    except Exception:
        return []


__all__ = ["AssetEntry", "_is_dir_with_any", "_iter_dirs"]

