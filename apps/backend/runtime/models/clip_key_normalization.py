"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP state-dict key normalization for Codex integrated text encoders.
Provides a strict, model-keyspace-focused normalizer used by the loader to accept common wrapper/prefix variants, drop HF-only buffers
(`position_ids`), and canonicalize `logit_scale`/`text_projection` into the exact key layout expected by `IntegratedCLIP` + `CodexCLIPTextModel`.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_codex_clip_state_dict` (function): Normalizes a CLIP state dict into the Codex integrated CLIP key layout (strict essentials).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from apps.backend.runtime.models.state_dict import transformers_convert


_CLIP_PREFIXES: tuple[str, ...] = (
    "conditioner.embedders.0.transformer.",
    "conditioner.embedders.0.model.",
    "conditioner.embedders.0.",
    "conditioner.embedders.1.transformer.",
    "conditioner.embedders.1.model.",
    "conditioner.embedders.1.",
    "cond_stage_model.model.",
    "cond_stage_model.",
    "text_encoders.clip_l.",
    "text_encoders.clip_g.",
    "clip_l.",
    "clip_g.",
    "clip_h.",
    "model.text_model.",
    "model.",
)


_ESSENTIAL_KEYS: tuple[str, ...] = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.embeddings.position_embedding.weight",
    "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "transformer.text_model.final_layer_norm.weight",
)


def _strip_known_prefixes(sd: Mapping[str, Any]) -> Dict[str, Any]:
    stripped: Dict[str, Any] = {}
    source_keys: Dict[str, str] = {}
    for raw_key, value in sd.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in _CLIP_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        source = str(raw_key)
        previous_source = source_keys.get(key)
        if previous_source is not None and previous_source != source:
            raise RuntimeError(
                "CLIP prefix stripping collision: destination key "
                f"{key!r} maps to multiple source keys ({previous_source!r}, {source!r})."
            )
        stripped[key] = value
        source_keys[key] = source
    return stripped


def _normalize_text_projection(work: Dict[str, Any], *, keep_projection: bool, transpose: bool) -> None:
    if not keep_projection:
        for key in (
            "transformer.text_projection.weight",
            "transformer.text_projection",
            "text_projection.weight",
            "text_projection",
        ):
            work.pop(key, None)
        return

    projection = None
    for key in (
        "transformer.text_projection.weight",
        "text_projection.weight",
        "transformer.text_projection",
        "text_projection",
    ):
        if key in work:
            projection = work.pop(key)
            break

    if projection is None:
        return

    if isinstance(projection, torch.Tensor) and transpose:
        projection = projection.transpose(0, 1).contiguous()
    work["transformer.text_projection.weight"] = projection
    work.pop("text_projection.weight", None)
    work.pop("transformer.text_projection", None)
    work.pop("text_projection", None)


def _has_essentials(work: Mapping[str, Any]) -> bool:
    return all(key in work for key in _ESSENTIAL_KEYS)


def normalize_codex_clip_state_dict(
    state_dict: Mapping[str, Any],
    *,
    num_layers: int,
    keep_projection: bool,
    transpose_projection: bool,
) -> Dict[str, Any]:
    """Normalize common CLIP wrapper/prefix variants into Codex integrated CLIP keys.

    Notes:
    - Drops `*.position_ids` buffers (Codex generates them on the fly).
    - Canonicalizes `logit_scale` into the top-level `IntegratedCLIP` keyspace.
    - Canonicalizes projection weights into `transformer.text_projection.weight` only.
    """

    work = dict(_strip_known_prefixes(state_dict))

    # Add transformer. prefix to keys that start with text_model.
    # This handles CLIP files that use text_model.* instead of transformer.text_model.*
    keys_to_rename = [
        (k, f"transformer.{k}")
        for k in list(work.keys())
        if k.startswith("text_model.") and not k.startswith("transformer.")
    ]
    for old_key, new_key in keys_to_rename:
        if new_key in work and new_key != old_key:
            raise RuntimeError(
                "CLIP key normalization collision: destination key "
                f"{new_key!r} already exists while remapping {old_key!r}."
            )
        work[new_key] = work.pop(old_key)

    transformers_convert(work, "transformer.", "transformer.text_model.", num_layers)
    transformers_convert(work, "", "transformer.text_model.", num_layers)

    _normalize_text_projection(work, keep_projection=keep_projection, transpose=transpose_projection)

    # Drop HF-style position_ids buffers.
    for key in list(work.keys()):
        if key.endswith(".position_ids"):
            work.pop(key, None)

    # Canonicalize logit_scale to the IntegratedCLIP keyspace.
    logit = None
    for key in ("logit_scale", "transformer.logit_scale", "transformer.text_model.logit_scale"):
        if key in work:
            logit = work.pop(key)
            break
    if logit is None:
        # Match IntegratedCLIP default (ln 100).
        logit = torch.tensor(4.605170185988092)
    if not isinstance(logit, torch.Tensor):
        logit = torch.tensor(float(logit))
    work["logit_scale"] = logit
    work.pop("transformer.logit_scale", None)
    work.pop("transformer.text_model.logit_scale", None)

    # Never keep the HF alias; the Codex model only has the transformer-prefixed key.
    work.pop("text_projection.weight", None)

    if not _has_essentials(work):
        sample_keys = list(sorted(work.keys()))[:20]
        raise RuntimeError(
            "CLIP state dict normalization failed: missing essential tensors. "
            f"required={list(_ESSENTIAL_KEYS)} sample_keys={sample_keys}"
        )

    return work
