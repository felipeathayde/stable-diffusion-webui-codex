from __future__ import annotations

import os
from typing import Dict, List

from .base import _iter_dirs


def _find_text_encoder_roots(repo_dir: str) -> List[str]:
    hits: List[str] = []
    for sub in ("text_encoder", "text_encoder_2", "t5", "clip", "tokenizer", "tokenizer_2"):
        p = os.path.join(repo_dir, sub)
        if os.path.isdir(p):
            hits.append(sub)
    return hits


def list_text_encoders(vendored_hf_root: str = "apps/server/backend/huggingface") -> Dict[str, List[str]]:
    """Return a mapping of available text encoders.

    The keys are repo names; the values are available subfolders (text_encoder, text_encoder_2, etc.).
    This is a lightweight discovery over vendored trees only.
    """
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(vendored_hf_root):
        return out
    try:
        for org in _iter_dirs(vendored_hf_root):
            for repo in _iter_dirs(org):
                name = os.path.basename(repo)
                subs = _find_text_encoder_roots(repo)
                if subs:
                    out[name] = sorted(subs)
    except Exception:
        return out
    return out


__all__ = ["list_text_encoders"]

