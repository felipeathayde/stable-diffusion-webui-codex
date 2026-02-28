"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Apply per-stage LoRA patches to WAN22 GGUF stage models (merge or online).
Controlled by `CODEX_LORA_APPLY_MODE` and maps LoRA keys to Codex WAN transformer keys via
`remap_wan22_lora_logical_key` from `keymap_wan22_transformer.py` (canonical keymap authority),
with optional strict logical-key coverage gating via `CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO`.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_stage_lora_offload_device` (function): Resolves stage-LoRA offload device from memory-manager policy.
- `apply_wan22_stage_lora` (function): Applies an ordered LoRA sequence to a loaded stage model (merge or online).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Set

import safetensors.torch as sf
import torch

from apps.backend.infra.config.bootstrap_env import get_bootstrap_env
from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.patchers.lora_loader import CodexLoraLoader
from apps.backend.runtime.adapters.lora.pipeline import build_patch_dicts
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.state_dict.keymap_wan22_transformer import remap_wan22_lora_logical_key

from .diagnostics import get_logger
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

_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO = "CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO"


def _read_min_match_ratio() -> float:
    raw = get_bootstrap_env(_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO)
    if raw is None:
        raw = os.getenv(_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO)
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text:
        return 0.0
    try:
        ratio = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be a float in [0, 1], got: {raw!r}"
        ) from exc
    if not math.isfinite(ratio):
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be finite, got: {raw!r}"
        )
    if ratio < 0.0 or ratio > 1.0:
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be in [0, 1], got: {raw!r}"
        )
    return ratio


def _resolve_stage_lora_offload_device() -> torch.device:
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "offload_device"):
        raise RuntimeError("WAN22 GGUF stage LoRA requires an active memory manager with offload_device().")
    offload_device = manager.offload_device()
    if not isinstance(offload_device, torch.device):
        raise RuntimeError(
            "WAN22 GGUF stage LoRA requires memory manager offload_device() to return torch.device "
            f"(got {type(offload_device).__name__})."
        )
    return offload_device


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

    Mapping authority is `remap_wan22_lora_logical_key` from WAN22 state-dict keymap;
    unsupported logical keys are left unmatched and handled by coverage/zero-match fail-loud
    checks at apply time.
    """

    model_keys = set(str(k) for k in model.state_dict().keys())
    logical_keys = sorted(_extract_logical_keys(tensors))
    if not logical_keys:
        return {}

    out: dict[str, str] = {}
    target_owner: dict[str, str] = {}

    for logical_key in logical_keys:
        stripped = _strip_known_prefixes(logical_key)
        logical_candidates: tuple[str, ...] = (
            (logical_key, stripped) if stripped != logical_key else (logical_key,)
        )

        target: str | None = None
        for candidate_logical in logical_candidates:
            mapped_weight_key = remap_wan22_lora_logical_key(candidate_logical)
            if mapped_weight_key is None:
                continue
            if mapped_weight_key in model_keys:
                target = mapped_weight_key
                break

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
    loras: Optional[Sequence[tuple[str, float]]],
    logger: Any,
) -> None:
    """Apply an ordered LoRA sequence to a loaded stage model (WAN22 GGUF runtime)."""

    if not loras:
        return

    log = get_logger(logger)
    min_match_ratio = _read_min_match_ratio()
    parsed_loras: list[tuple[str, float, dict[str, list[tuple]], int]] = []
    for index, raw_spec in enumerate(loras):
        if not isinstance(raw_spec, (tuple, list)) or len(raw_spec) != 2:
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}] must be a [path, weight] pair."
            )
        raw_path = raw_spec[0]
        raw_weight = raw_spec[1]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}][0] must be a non-empty path string."
            )
        resolved_path = normalize_win_path(os.path.expanduser(raw_path.strip()))
        if not resolved_path.lower().endswith(".safetensors"):
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}] path must be a .safetensors file, got: {resolved_path}"
            )
        if not os.path.isfile(resolved_path):
            raise RuntimeError(f"WAN22 GGUF stage '{stage}': loras[{index}] path not found: {resolved_path}")
        if raw_weight is None:
            strength = 1.0
        else:
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise RuntimeError(
                    f"WAN22 GGUF stage '{stage}': loras[{index}] weight must be numeric, got: {raw_weight!r}"
                )
            strength = float(raw_weight)
            if not math.isfinite(strength):
                raise RuntimeError(
                    f"WAN22 GGUF stage '{stage}': loras[{index}] weight must be finite, got: {raw_weight!r}"
                )

        try:
            tensors = sf.load_file(resolved_path)
        except Exception as exc:
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': failed to load LoRA file at loras[{index}] ({resolved_path}): {exc}"
            ) from exc

        logical_key_count = len(_extract_logical_keys(tensors))
        to_load = _build_to_load_map(model, tensors)
        if not to_load:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA file matched 0 targets; "
                "this LoRA key layout is not supported by the WAN transformer mapping. "
                "file={path}".format(stage=stage, path=resolved_path)
            )
        matched_count = len(to_load)
        coverage = (matched_count / logical_key_count) if logical_key_count > 0 else 0.0
        if min_match_ratio > 0.0 and coverage < min_match_ratio:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA logical-key coverage below threshold "
                f"(matched={matched_count}/{logical_key_count} ratio={coverage:.4f} required>={min_match_ratio:.4f}). "
                "Adjust CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO or use a compatible adapter mapping. "
                "file={path}".format(stage=stage, path=resolved_path)
            )
        if logical_key_count > 0 and coverage < 1.0:
            log.warning(
                "[wan22.gguf] stage LoRA partial logical-key coverage: stage=%s index=%d matched=%d total=%d ratio=%.4f required_ratio=%.4f",
                stage,
                index,
                matched_count,
                logical_key_count,
                coverage,
                min_match_ratio,
            )

        patch_dict = build_patch_dicts(tensors, to_load)
        if not patch_dict:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA produced 0 patches after parsing; "
                "this usually indicates incomplete tensors for the mapped keys. "
                "file={path}".format(stage=stage, path=resolved_path)
            )
        lora_patch_map = {
            key: [(strength, payload, 1.0, None, None)] for key, payload in patch_dict.items()
        }
        parsed_loras.append((resolved_path, strength, lora_patch_map, len(patch_dict)))

    if not parsed_loras:
        return
    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode is LoraApplyMode.ONLINE

    loader = getattr(model, "lora_loader", None)
    if not isinstance(loader, CodexLoraLoader):
        loader = CodexLoraLoader(model)
        model.lora_loader = loader

    lora_patches: dict[tuple[str, float, float, bool], dict[str, list[tuple]]] = {}
    for index, (resolved_path, strength, lora_patch_map, _patch_count) in enumerate(parsed_loras):
        patch_source = f"{resolved_path}#stage_index={index}"
        lora_patches[(patch_source, strength, 1.0, online_mode)] = lora_patch_map

    offload_device = _resolve_stage_lora_offload_device()
    loader.refresh(lora_patches, offload_device=offload_device, force_refresh=False)

    total_loras = len(parsed_loras)
    for index, (resolved_path, strength, _lora_patch_map, patch_count) in enumerate(parsed_loras):
        log.info(
            "[wan22.gguf] stage LoRA applied: stage=%s index=%d/%d mode=%s file=%s params=%d weight=%s offload_device=%s",
            stage,
            index + 1,
            total_loras,
            apply_mode.value,
            os.path.basename(resolved_path),
            patch_count,
            strength,
            offload_device,
        )


__all__ = ["apply_wan22_stage_lora"]
