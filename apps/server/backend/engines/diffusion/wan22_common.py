from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


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


__all__ = ["EngineOpts", "WanComponents", "WanStageOptions"]

