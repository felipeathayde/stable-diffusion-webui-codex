from __future__ import annotations

from typing import Dict, Any

import logging


_log = logging.getLogger("backend.model_parser.unet")


def normalize_label_embeddings(state: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested label embedding keys produced by some SDXL checkpoints.

    Legacy/Forge variants split the sequential label embedding layers as
    ``label_emb.0.0`` / ``label_emb.0.2``. Our UNet expects
    ``label_emb.0`` / ``label_emb.2``. Collapse the extra depth while keeping
    the tensor values intact.
    """

    normalized = dict(state)
    replacements = {}
    for key in list(state.keys()):
        if not key.startswith("label_emb.0."):
            continue
        segments = key.split(".")
        if len(segments) < 4:
            continue
        inner = segments[2]
        if not inner.isdigit():
            continue
        new_key = ".".join([segments[0], inner, *segments[3:]])
        replacements[key] = new_key

    if not replacements:
        return normalized

    for old_key, new_key in replacements.items():
        value = normalized.pop(old_key)
        normalized[new_key] = value

    _log.info("[unet] normalized label embeddings: %s", replacements)
    return normalized


__all__ = ["normalize_label_embeddings"]

