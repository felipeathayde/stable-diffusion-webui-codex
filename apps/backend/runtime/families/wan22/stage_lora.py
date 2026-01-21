"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Apply per-stage LoRA patches to WAN22 GGUF stage models.
Supports both offline merge (default) and online patching, controlled globally by `CODEX_LORA_APPLY_MODE`.
Implements a robust LoRA-key → model-parameter mapping strategy:
- Prefer direct matches against the Codex-native WAN transformer keys.
- Otherwise remap Diffusers/WAN-export naming into Codex keys via `remap_wan22_gguf_state_dict`.

Symbols (top-level; keep in sync; no ghosts):
- `apply_wan22_stage_lora` (function): Applies a LoRA file to a loaded stage model (merge or online).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Set

import safetensors.torch as sf
import torch

from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.patchers.lora_loader import CodexLoraLoader
from apps.backend.runtime.adapters.lora.pipeline import build_patch_dicts

from .diagnostics import get_logger
from .model import remap_wan22_gguf_state_dict
from .paths import normalize_win_path

_WAN22_LORA_PREFIXES = (
    "transformer_2.",
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)

_LORA_LOGICAL_SUFFIXES: tuple[str, ...] = (
    # Standard LoRA (multiple conventions)
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_mid.weight",
    "_lora.up.weight",
    "_lora.down.weight",
    ".lora_B.weight",
    ".lora_A.weight",
    ".lora.up.weight",
    ".lora.down.weight",
    ".lora_linear_layer.up.weight",
    ".lora_linear_layer.down.weight",
    # Optional metadata
    ".alpha",
    ".dora_scale",
    # DIFF / SET
    ".diff",
    ".diff_b",
    ".set_weight",
    # LoHa
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
    # LoKr
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
    # GLoRA
    ".a1.weight",
    ".a2.weight",
    ".b1.weight",
    ".b2.weight",
)


def _strip_known_prefixes(name: str) -> str:
    k = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in _WAN22_LORA_PREFIXES:
            if k.startswith(prefix):
                k = k[len(prefix) :]
                changed = True
                break
    return k


def _extract_logical_keys(tensors: Mapping[str, torch.Tensor]) -> Set[str]:
    logical: set[str] = set()
    for key in tensors.keys():
        s = str(key)
        for suffix in _LORA_LOGICAL_SUFFIXES:
            if s.endswith(suffix):
                logical.add(s[: -len(suffix)])
                break
    return logical


def _build_to_load_map(model: torch.nn.Module, tensors: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    """Return LoRA logical-key → model-param mappings for a WAN stage model.

    This is designed for WAN22 stage LoRAs (including LightX2V) that may be authored in:
    - Codex-native module naming, or
    - Diffusers/WAN export naming (converted via remap).
    """

    model_keys = set(str(k) for k in model.state_dict().keys())
    logical_keys = sorted(_extract_logical_keys(tensors))
    if not logical_keys:
        return {}

    out: dict[str, str] = {}
    target_owner: dict[str, str] = {}

    for logical_key in logical_keys:
        stripped = _strip_known_prefixes(logical_key)
        direct_weight_key = f"{stripped}.weight"

        target: str | None = None
        if direct_weight_key in model_keys:
            target = direct_weight_key
        else:
            remapped = remap_wan22_gguf_state_dict({direct_weight_key: 0})
            if len(remapped) == 1:
                candidate = next(iter(remapped.keys()))
                if candidate in model_keys:
                    target = candidate

        if target is None:
            continue

        previous_owner = target_owner.get(target)
        if previous_owner is not None and previous_owner != logical_key:
            raise RuntimeError(
                "WAN22 GGUF stage LoRA maps multiple logical keys to the same target weight. "
                f"target={target!r} keys={previous_owner!r},{logical_key!r}"
            )
        target_owner[target] = logical_key
        out[logical_key] = target

    return out


def apply_wan22_stage_lora(
    model: torch.nn.Module,
    *,
    stage: str,
    lora_path: Optional[str],
    lora_weight: Optional[float],
    logger: Any,
) -> None:
    """Apply a LoRA file to a loaded stage model (WAN22 GGUF runtime)."""

    if not lora_path:
        return

    log = get_logger(logger)

    resolved_path = normalize_win_path(os.path.expanduser(str(lora_path)))
    if not resolved_path.lower().endswith(".safetensors"):
        raise RuntimeError(f"WAN22 GGUF stage '{stage}': lora_path must be a .safetensors file, got: {resolved_path}")
    if not os.path.isfile(resolved_path):
        raise RuntimeError(f"WAN22 GGUF stage '{stage}': lora_path not found: {resolved_path}")

    strength = float(lora_weight) if lora_weight is not None else 1.0
    if not math.isfinite(strength):
        raise RuntimeError(f"WAN22 GGUF stage '{stage}': lora_weight must be finite, got: {lora_weight!r}")

    try:
        tensors = sf.load_file(resolved_path)
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF stage '{stage}': failed to load LoRA file {resolved_path}: {exc}") from exc

    to_load = _build_to_load_map(model, tensors)
    if not to_load:
        raise RuntimeError(
            "WAN22 GGUF stage '{stage}': LoRA file matched 0 targets; "
            "this LoRA key layout is not supported by the WAN transformer mapping. "
            "file={path}".format(stage=stage, path=resolved_path)
        )

    patch_dict = build_patch_dicts(tensors, to_load)
    if not patch_dict:
        raise RuntimeError(
            "WAN22 GGUF stage '{stage}': LoRA produced 0 patches after parsing; "
            "this usually indicates incomplete tensors for the mapped keys. "
            "file={path}".format(stage=stage, path=resolved_path)
        )

    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode is LoraApplyMode.ONLINE

    loader = getattr(model, "lora_loader", None)
    if not isinstance(loader, CodexLoraLoader):
        loader = CodexLoraLoader(model)
        model.lora_loader = loader

    lora_patches: dict[tuple[str, float, float, bool], dict[str, list[tuple]]] = {
        (resolved_path, strength, 1.0, online_mode): {
            key: [(strength, payload, 1.0, None, None)] for key, payload in patch_dict.items()
        }
    }
    loader.refresh(lora_patches, offload_device=torch.device("cpu"), force_refresh=True)

    log.info(
        "[wan22.gguf] stage LoRA applied: stage=%s mode=%s file=%s params=%d",
        stage,
        apply_mode.value,
        os.path.basename(resolved_path),
        len(patch_dict),
    )


__all__ = ["apply_wan22_stage_lora"]
