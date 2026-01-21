"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers and typed options for WAN2.2 engines.
Defines dataclasses for engine/runtime wiring and strict resolution helpers for model repos and user-supplied assets.

Symbols (top-level; keep in sync; no ghosts):
- `EngineOpts` (dataclass): Minimal WAN engine load options (device/dtype).
- `WanComponents` (dataclass): Holder for instantiated WAN components/pipelines and resolved paths.
- `WanStageOptions` (dataclass): Stage-specific overrides for WAN pipelines (sampler/scheduler/steps/cfg/LoRA/lightning).
- `resolve_wan_repo_candidates` (function): Strict resolution of WAN Diffusers repo candidates (env override or known map; raises otherwise).
- `resolve_user_supplied_assets` (function): Extracts user-supplied asset paths (high/low stage dirs, VAE/text encoder) from extras payload.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List



@dataclass
class EngineOpts:
    device: str = "auto"
    dtype: str = "fp16"


@dataclass
class WanComponents:
    text_encoder: Any | None = None
    transformer: Any | None = None
    vae: Any | None = None
    pipeline: Any | None = None
    pipeline_high: Any | None = None
    pipeline_low: Any | None = None
    model_dir: str | None = None
    high_dir: str | None = None
    low_dir: str | None = None
    device: str = "cpu"
    dtype: str = "fp16"
    hf_repo_dir: Optional[str] = None
    hf_text_encoder_dir: Optional[str] = None
    hf_tokenizer_dir: Optional[str] = None
    hf_vae_dir: Optional[str] = None


@dataclass
class WanStageOptions:
    model_dir: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    steps: int = 12
    cfg_scale: Optional[float] = None
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None
    lightning: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def from_mapping(obj: Any, *, default_steps: int = 12, default_cfg: Optional[float] = None) -> "WanStageOptions":
        if not isinstance(obj, dict):
            return WanStageOptions(steps=default_steps, cfg_scale=default_cfg)
        if obj.get("lora_path") not in (None, ""):
            raise ValueError("WAN stage 'lora_path' is unsupported; use 'lora_sha' instead.")

        lora_sha = str(obj.get("lora_sha") or "").strip().lower() or None
        lora_weight = float(obj.get("lora_weight")) if obj.get("lora_weight") is not None else None
        if lora_weight is not None and not lora_sha:
            raise ValueError("WAN stage 'lora_weight' requires 'lora_sha'.")
        lora_path = None
        if lora_sha:
            if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                raise ValueError("WAN stage 'lora_sha' must be sha256 (64 lowercase hex).")
            from apps.backend.inventory.cache import resolve_asset_by_sha

            resolved = resolve_asset_by_sha(lora_sha)
            if not resolved:
                raise ValueError(f"WAN stage LoRA not found for sha: {lora_sha}")
            lora_path = os.path.expanduser(str(resolved))
            if not lora_path.lower().endswith(".safetensors"):
                raise ValueError(f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}")
            if not os.path.isfile(lora_path):
                raise ValueError(f"WAN stage LoRA file not found: {lora_path}")

        return WanStageOptions(
            model_dir=str(obj.get("model_dir")) if obj.get("model_dir") else None,
            sampler=str(obj.get("sampler")) if obj.get("sampler") else None,
            scheduler=str(obj.get("scheduler")) if obj.get("scheduler") else None,
            steps=int(obj.get("steps") or default_steps),
            cfg_scale=(float(obj.get("cfg_scale")) if obj.get("cfg_scale") is not None else default_cfg),
            lora_path=lora_path,
            lora_weight=lora_weight,
            lightning=bool(obj.get("lightning", False)),
        )


WAN_DIFFUSERS_REPO_CANDIDATES = {
    # Known, published Diffusers repos only (avoid ambiguous non-diffusers names)
    "wan22_14b": (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ),
    "wan22_5b": (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
    "wan22_animate_14b": (
        "Wan-AI/Wan2.2-Animate-14B-Diffusers",
    ),
    "wan_t2v_14b": (
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ),
    "wan_t2v_5b": (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
}


def resolve_wan_repo_candidates(model_key: Optional[str] = None) -> List[str]:
    """Return an ordered list of WAN 2.2 Diffusers repo IDs to try.

    Strict mode: no implicit or generic fallbacks. Resolution order is:
    1) Exact key match in `WAN_DIFFUSERS_REPO_CANDIDATES`

    If neither yields any candidate, raise with an actionable message.
    """
    key = (model_key or "").lower().strip()

    # 1) exact/contained key match from known map
    candidates: List[str] = []
    for mk, repos in WAN_DIFFUSERS_REPO_CANDIDATES.items():
        if mk in key:
            candidates.extend(repos)
            break

    if not candidates:
        valid = ", ".join(sorted(WAN_DIFFUSERS_REPO_CANDIDATES.keys()))
        raise ValueError(
            (
                "resolve_wan_repo_candidates: unable to resolve repo for key '{key}'. "
                "Provide a valid engine/model key (one of: {valid})."
            ).format(key=model_key, valid=valid)
        )

    # Deduplicate preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for rid in candidates:
        if rid and rid not in seen:
            uniq.append(rid)
            seen.add(rid)
    return uniq


__all__ = ["EngineOpts", "WanComponents", "WanStageOptions", "resolve_wan_repo_candidates", "WAN_DIFFUSERS_REPO_CANDIDATES"]


def resolve_user_supplied_assets(extras: dict | None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Strict parsing of WAN22 asset paths (no implicit fallbacks).

    Accepted keys only:
    - VAE: 'wan_vae_path' (file or directory)
    - Text encoder: 'wan_text_encoder_path' (file only; sha-selected)
    - Metadata/tokenizer: 'wan_metadata_dir' or 'wan_tokenizer_dir'

    Returns (vae_path, text_encoder_path, metadata_dir). No guessing, no defaults.
    """
    ex = extras or {}

    vae = str(ex.get("wan_vae_path")).strip() if ex.get("wan_vae_path") else None

    te = None
    if ex.get("wan_text_encoder_path"):
        te = str(ex.get("wan_text_encoder_path")).strip()
    if ex.get("wan_text_encoder_dir"):
        raise ValueError("WAN22: 'wan_text_encoder_dir' is unsupported in sha-only mode; provide 'wan_text_encoder_path' instead.")

    meta = None
    if ex.get("wan_metadata_dir"):
        meta = str(ex.get("wan_metadata_dir")).strip()
    elif ex.get("wan_tokenizer_dir"):
        meta = str(ex.get("wan_tokenizer_dir")).strip()

    return (vae if vae else None), (te if te else None), (meta if meta else None)
