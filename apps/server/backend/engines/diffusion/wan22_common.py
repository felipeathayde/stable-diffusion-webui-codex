from __future__ import annotations

import os
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
    sampler: Optional[str] = "Automatic"
    scheduler: Optional[str] = "Automatic"
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
        return WanStageOptions(
            model_dir=str(obj.get("model_dir")) if obj.get("model_dir") else None,
            sampler=str(obj.get("sampler")) if obj.get("sampler") else "Automatic",
            scheduler=str(obj.get("scheduler")) if obj.get("scheduler") else "Automatic",
            steps=int(obj.get("steps") or default_steps),
            cfg_scale=(float(obj.get("cfg_scale")) if obj.get("cfg_scale") is not None else default_cfg),
            lora_path=str(obj.get("lora_path")) if obj.get("lora_path") else None,
            lora_weight=(float(obj.get("lora_weight")) if obj.get("lora_weight") is not None else None),
            lightning=bool(obj.get("lightning", False)),
        )


WAN_DIFFUSERS_REPO_CANDIDATES = {
    "wan22_14b": (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ),
    "wan22_5b": (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
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
    key = (model_key or "").lower()
    for mk, repos in WAN_DIFFUSERS_REPO_CANDIDATES.items():
        if mk in key:
            return list(repos)
    candidates: List[str] = []
    env_repo = os.environ.get("CODEX_WAN_DIFFUSERS_REPO")
    if env_repo:
        candidates.append(env_repo)
    owner = "Wan-AI"
    variant = "14B" if "14b" in key else ("5B" if "5b" in key else "")
    mode = "Image-to-Video" if "i2v" in key else ("Text-to-Video" if "t2v" in key else "")
    if variant and mode:
        candidates.append(f"{owner}/Wan2.2-{mode}-{variant}")
    if not candidates:
        if "14b" in key:
            candidates.extend([
                f"{owner}/Wan2.2-I2V-A14B-Diffusers",
                f"{owner}/Wan2.2-T2V-A14B-Diffusers",
            ])
        else:
            candidates.append(f"{owner}/Wan2.2-TI2V-5B-Diffusers")
    seen: set[str] = set()
    uniq: List[str] = []
    for rid in candidates:
        if rid and rid not in seen:
            uniq.append(rid)
            seen.add(rid)
    return uniq


__all__ = ["EngineOpts", "WanComponents", "WanStageOptions", "resolve_wan_repo_candidates", "WAN_DIFFUSERS_REPO_CANDIDATES"]
